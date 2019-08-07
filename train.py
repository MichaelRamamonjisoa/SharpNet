import torch
import datetime
from torch.optim import SGD, Adam
import argparse

from tensorboardX import SummaryWriter
from torchnet.meter import MovingAverageValueMeter
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import *
from dataset_manager import *
from sharpnet_model import *
from loss import *
from resnet import Bottleneck as ResBlock
from utils import *

import os
import sys


def train_epoch(train_loader, val_loader, model, criterion, optimizer, epoch,
                train_writer, val_writer,
                train_loss_meter, val_loss_meter,
                depth_loss_meter, grad_loss_meter,
                normals_loss_meter,
                date_str, model_save_path,
                args,
                boundary_loss_meter=None, consensus_loss_meter=None):

    batch_size = int(args.batch_size)
    iter_size = args.iter_size
    num_workers = int(args.num_workers)

    loader_iter = iter(train_loader)

    for iter_i, _ in enumerate(train_loader):
        optimizer.zero_grad()
        iter_loss = 0
        iter_normals_loss = 0
        iter_grad_loss = 0
        iter_depth_loss = 0
        iter_boundary_loss = 0
        iter_consensus_loss = 0

        freeze_decoders = args.decoder_freeze.split(',')
        freeze_model_decoders(model, freeze_decoders)

        # accumulated gradients
        for i in range(iter_size):
            # get ground truth sample
            input, mask_gt, depth_gt, normals_gt, boundary_gt = get_gt_sample(train_loader, loader_iter, args)
            # compute output
            depth_pred, normals_pred, boundary_pred = get_tensor_preds(input, model, args)
            # compute loss
            depth_loss, grad_loss, normals_loss, b_loss, geo_loss = criterion(mask_gt,
                                                                              d_pred=depth_pred,
                                                                              d_gt=depth_gt,
                                                                              n_pred=normals_pred,
                                                                              n_gt=normals_gt,
                                                                              b_pred=boundary_pred,
                                                                              b_gt=boundary_gt,
                                                                              use_grad=True)

            loss_real = depth_loss + grad_loss + normals_loss + b_loss + geo_loss
            loss = 1 * depth_loss + 0.1 * grad_loss + 0.5 * normals_loss + 0.005 * b_loss + 0.5 * geo_loss
            loss_real /= float(iter_size)
            loss /= float(iter_size)

            iter_loss += float(loss_real)
            iter_normals_loss += float(normals_loss)

            if grad_loss != 0:
                iter_grad_loss += float(grad_loss)
            if depth_loss != 0:
                iter_depth_loss += float(depth_loss)
            if b_loss != 0:
                iter_boundary_loss += float(b_loss)
            if geo_loss != 0:
                iter_consensus_loss += float(geo_loss)

            loss.backward()

        parameters = get_params(model)
        clip_grad_norm_(parameters, 10.0, norm_type=2)
        optimizer.step()

        if iter_normals_loss != 0:
            iter_normals_loss /= float(iter_size)
            normals_loss_meter.add(float(normals_loss))
        if iter_depth_loss != 0:
            iter_depth_loss /= float(iter_size)
            depth_loss_meter.add(float(iter_depth_loss))
        if iter_grad_loss != 0:
            iter_grad_loss /= float(iter_size)
            grad_loss_meter.add(float(iter_grad_loss))
        if iter_boundary_loss != 0:
            iter_boundary_loss /= float(iter_size)
            boundary_loss_meter.add(float(iter_boundary_loss))
        if iter_consensus_loss != 0:
            iter_consensus_loss /= float(iter_size)
            consensus_loss_meter.add(float(iter_consensus_loss))

        train_size = len(train_loader.dataset)
        iter_per_epoch = int(train_size/args.batch_size)
        train_loss_meter.add(float(iter_loss))
        print("epoch: " + str(epoch) + " | iter: {}/{} ".format(iter_i, iter_per_epoch) + "| Train Loss: " + str(float(iter_loss)))
        train_writer.add_scalar("train_loss", train_loss_meter.value()[0],
                                int(epoch) * iter_per_epoch + iter_i)

        write_loss_components(train_writer, iter_i, epoch, train_size, args,
                              depth_loss_meter, iter_depth_loss,
                              normals_loss_meter, iter_normals_loss,
                              boundary_loss_meter, iter_boundary_loss,
                              grad_loss_meter, iter_grad_loss,
                              consensus_loss_meter, iter_consensus_loss)

        if (iter_i + 1) % 50 == 0:
            val_loss = 0
            val_depth_loss = 0
            val_grad_loss = 0
            val_normals_loss = 0
            val_boundary_loss = 0
            val_consensus_loss = 0

            val_size = len(val_loader.dataset)

            with torch.no_grad():
                # evaluate on validation set
                model.eval()
                loader_iter = iter(val_loader)

                n_val_batches = int(float(val_size) / batch_size)
                for i in range(n_val_batches)[:50]:
                    # get ground truth sample
                    input, mask_gt, depth_gt, normals_gt, boundary_gt = get_gt_sample(val_loader, loader_iter, args)
                    # compute output
                    depth_pred, normals_pred, boundary_pred = get_tensor_preds(input, model, args)
                    # compute loss
                    depth_loss, grad_loss, normals_loss, b_loss, geo_loss = criterion(mask_gt,
                                                                                      d_pred=depth_pred,
                                                                                      d_gt=depth_gt,
                                                                                      n_pred=normals_pred,
                                                                                      n_gt=normals_gt,
                                                                                      b_pred=boundary_pred,
                                                                                      b_gt=boundary_gt,
                                                                                      use_grad=True)

                    iter_loss = depth_loss + normals_loss + grad_loss + b_loss + geo_loss

                    iter_loss = float(iter_loss) / 50
                    val_loss += iter_loss
                    if grad_loss != 0:
                        val_grad_loss += float(grad_loss) / 50
                    if depth_loss != 0:
                        val_depth_loss += float(depth_loss) / 50
                    if b_loss != 0:
                        val_boundary_loss += float(b_loss) / 50
                    if geo_loss != 0:
                        val_consensus_loss += float(geo_loss) / 50
                    if normals_loss != 0:
                        val_normals_loss += float(normals_loss) / 50

            val_loss_meter.add(val_loss)
            print("epoch: " + str(epoch) + " | iter: {}/{} ".format(iter_i, iter_per_epoch) + "| Val Loss: " + str(
                float(val_loss)))
            val_writer.add_scalar("val_loss", val_loss_meter.value()[0],
                                  int(epoch) * iter_per_epoch + iter_i)

            write_loss_components(val_writer, iter_i, epoch, train_size, args,
                                  depth_loss_meter, val_depth_loss,
                                  normals_loss_meter, val_normals_loss,
                                  boundary_loss_meter, val_boundary_loss,
                                  grad_loss_meter, val_grad_loss,
                                  consensus_loss_meter, val_consensus_loss)

            model.train()

            freeze_decoders = args.decoder_freeze.split(',')
            freeze_model_decoders(model, freeze_decoders)

        if (iter_i + 1) % 1000 == 0:
            print('Saving checkpoint')
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            torch.save(
                model.state_dict(),
                os.path.join(model_save_path, "checkpoint_{}_iter_{}.pth".format(epoch, iter_i + 1)),
            )
            print('Done')


def get_trainval_splits(args):
    t = {'SCALE': 2,
         'CROP': 320,
         'HORIZONTALFLIP': 1,
         'ROTATE': 6,
         'GAMMA': 0.15,
         'NORMALIZE': {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
         }

    if args.dataset != 'NYU':
        try:
            with open(os.path.join(args.root_dir, 'jobs_train.txt'), 'r') as f:
                list_train_files = [line.split('\n')[0] for line in f.readlines() if line != '\n']
        except Exception as e:
            print('The file containing the list of images does not exist')
            print(os.path.join(args.root_dir, 'jobs_train.txt'))
            sys.exit(0)

        try:
            with open(os.path.join(args.root_dir, 'jobs_val.txt'), 'r') as f:
                list_val_files = [line.split('\n')[0] for line in f.readlines() if line != '\n']
        except Exception as e:
            print('The file containing the list of images does not exist')
            print(os.path.join(args.root_dir, 'jobs_val.txt'))
            sys.exit(0)

        if len(list_train_files) < 2:
            print('Train file contains less than 2 files, error')
            sys.exit(0)
        if len(list_val_files) < 2:
            print('Val file contains less than 2 files, error')
            sys.exit(0)

        train_files = list_train_files
        val_files = list_val_files

    if args.dataset == 'PBRS':
        train_dataset = PBRSDataset(img_list=train_files, root_dir=args.root_dir,
                                    transforms=t,
                                    use_depth=True if args.depth else False,
                                    use_boundary=True if args.boundary else False,
                                    use_normals=True if args.normals else False)
        val_dataset = PBRSDataset(img_list=val_files, root_dir=args.root_dir,
                                  transforms=t,
                                  use_depth=True if args.depth else False,
                                  use_boundary=True if args.boundary else False,
                                  use_normals=True if args.normals else False)
    elif args.dataset == 'NYU':
        train_dataset = NYUDataset('nyu_depth_v2_labeled.mat', split_type='train', root_dir=args.root_dir,
                                   transforms=t,
                                   use_depth=True,
                                   use_boundary=False,
                                   use_normals=False)
        val_dataset = NYUDataset('nyu_depth_v2_labeled.mat', split_type='test', root_dir=args.root_dir,
                                 transforms=t,
                                 use_depth=True,
                                 use_boundary=False,
                                 use_normals=False)

    train_dataloader = DataLoader(train_dataset, batch_size=int(args.batch_size),
                                  shuffle=True, num_workers=int(args.num_workers))

    val_dataloader = DataLoader(val_dataset, batch_size=int(args.batch_size),
                                shuffle=True, num_workers=int(args.num_workers))

    return train_dataloader, val_dataloader


def main():
    parser = argparse.ArgumentParser(description="Train the SharpNet network")
    parser.add_argument('--dataset', '-d', dest='dataset', help='Name of the dataset (MLT, NYUv2 or pix3d)')
    parser.add_argument('--exp_name', dest='experiment_name', help='Custom name of the experiment', type=str, default=None)
    parser.add_argument('--batch-size', '-b', dest='batch_size', type=int, default=3, help='Batch size')
    parser.add_argument('--iter-size', dest='iter_size', type=int, default=3,
                        help='Iteration size (for accumulated gradients)')
    parser.add_argument('--boundary', action='store_true',
                        help='Use boundary decoder')
    parser.add_argument('--normals', action='store_true',
                        help='Use normals decoder')
    parser.add_argument('--depth', action='store_true',
                        help='Use depth decoder')
    parser.add_argument('--consensus', dest='geo_consensus', action='store_true')
    parser.add_argument('--freeze', dest='decoder_freeze', default='', type=str,
                        help='Decoders to freeze (comma seperated)')
    parser.add_argument('--verbose', action='store_true', help='Activate to display loss components terms')
    parser.add_argument('--rootdir', '-r', dest='root_dir', default='', help='Root Directory of the dataset')
    parser.add_argument('--nocuda', action="store_true", help='Use flag to use on CPU only (currently not supported)')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=1e-5, help='Initial learning rate')
    parser.add_argument('--lr-mode', dest='lr_mode', default='poly', help='Learning rate decay mode')
    parser.add_argument('--max-epoch', dest='max_epoch', type=int, default=1000, help='MAXITER')
    parser.add_argument('--step', '-s', dest='gradient_step', default=5e-2, help='gradient step')
    parser.add_argument('--cuda', dest='cuda_device', default="0", help='CUDA device ID')
    parser.add_argument('--cpu', dest='num_workers', default=4)
    parser.add_argument('--pretrained-model', dest='pretrained_model', default=None, help="Choose a model to fine tune")
    parser.add_argument('--start_epoch', dest='start_epoch', default=0, type=int, help="Starting epoch")
    parser.add_argument('--bias', action="store_true", help="Flag to learn bias in decoder convnet")
    parser.add_argument('--optimizer', dest='optimizer', default='SGD', type=str, help="Optimizer type: SGD  /  Adam")
    parser.add_argument('--decay', dest='decay', default=5e-5, type=float, help="Weight decay rate")

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
    cuda = False if args.nocuda else True

    resnet50_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'

    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Running on " + torch.cuda.get_device_name(current_device))
    else:
        print("Running on CPU")

    now = datetime.datetime.now()
    date_str = now.strftime("%d-%m-%Y_%H-%M")

    t = []
    torch.manual_seed(329)

    bias = True if args.bias else False

    # build model
    model = SharpNet(ResBlock, [3, 4, 6, 3], [2, 2, 2, 2, 2],
                     use_normals=True if args.normals else False,
                     use_depth=True if args.depth else False,
                     use_boundary=True if args.boundary else False,
                     bias_decoder=bias)

    model_dict = model.state_dict()

    # Load pretrained weights

    resnet_path = 'models/resnet50-19c8e357.pth'

    if not os.path.exists(resnet_path):
        command = 'wget ' + resnet50_url + ' && mkdir models/ && mv resnet50-19c8e357.pth models/'
        os.system(command)

    resnet50_dict = torch.load(resnet_path)

    resnet_dict = {k.replace('.', '_img.', 1): v for k, v in resnet50_dict.items() if
                   k.replace('.', '_img.', 1) in model_dict}  # load weights up to pool

    if args.pretrained_model is not None:
        model_path = args.pretrained_model
        tmp_dict = torch.load(model_path)
        if args.depth:
            pretrained_dict = {k: v for k, v in tmp_dict.items() if k in model_dict}
        else:
            pretrained_dict = {k: v for k, v in tmp_dict.items() if
                               (k in model_dict and not k.startswith('depth_decoder'))}

    else:
        pretrained_dict = resnet_dict

    try:
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('Successfully loaded pretrained ResNet weights')
    except:
        print('Could not load the pretrained model weights')
        sys.exit(0)

    model.to(device)
    model.zero_grad()
    model.train()

    freeze_decoders = args.decoder_freeze.split(',')
    freeze_model_decoders(model, freeze_decoders)

    if args.dataset != 'NYU':
        sharpnet_loss = SharpNetLoss(lamb=0.5, mu=1.0,
                                     use_depth=True if args.depth else False,
                                     use_boundary=True if args.boundary else False,
                                     use_normals=True if args.normals else False,
                                     use_geo_consensus=True if args.geo_consensus else False)
    else:
        sharpnet_loss = SharpNetLoss(lamb=0.5, mu=1.0,
                                     use_depth=True if args.depth else False,
                                     use_boundary=False,
                                     use_normals=False,
                                     use_geo_consensus=True if args.geo_consensus else False)

    if args.optimizer == 'SGD':
        optimizer = SGD(params=get_params(model),
                        lr=args.learning_rate,
                        weight_decay=args.decay,
                        momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = Adam(params=get_params(model),
                         lr=args.learning_rate,
                         weight_decay=args.decay)
    else:
        print('Could not configure the optimizer, please select --optimizer Adam or SGD')
        sys.exit(0)

    # TensorBoard Logger
    train_loss_meter = MovingAverageValueMeter(20)
    val_loss_meter = MovingAverageValueMeter(3)
    depth_loss_meter = MovingAverageValueMeter(3) if args.depth else None
    normals_loss_meter = MovingAverageValueMeter(3) if args.normals and args.dataset != 'NYU' else None
    grad_loss_meter = MovingAverageValueMeter(3) if args.depth else None
    boundary_loss_meter = MovingAverageValueMeter(3) if args.boundary and args.dataset != 'NYU' else None
    consensus_loss_meter = MovingAverageValueMeter(3) if args.geo_consensus else None

    exp_name = args.experiment_name if args.experiment_name is not None else ''
    print('Experiment Name: {}'.format(exp_name))

    log_dir = os.path.join('logs', 'Joint', str(exp_name) + '_' + date_str)
    cp_dir = os.path.join('checkpoints', 'Joint', str(exp_name) + '_' + date_str)
    print('Checkpoint Directory: {}'.format(cp_dir))

    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(log_dir, 'val'))

    if not os.path.exists(cp_dir):
        os.makedirs(cp_dir)
    if not os.path.exists(log_dir):
        os.makedirs(os.path.join(log_dir, 'train'))
        os.makedirs(os.path.join(log_dir, 'val'))

    train_dataloader, val_dataloader = get_trainval_splits(args)

    for epoch in range(args.max_epoch):
        if args.optimizer == 'SGD':
            adjust_learning_rate(args.learning_rate, args.lr_mode, args.gradient_step, args.max_epoch,
                                 optimizer, epoch)

        train_epoch(train_dataloader, val_dataloader, model, sharpnet_loss, optimizer, args.start_epoch + epoch,
                    train_writer, val_writer,
                    train_loss_meter, val_loss_meter,
                    depth_loss_meter, grad_loss_meter,
                    normals_loss_meter,
                    date_str=date_str, model_save_path=cp_dir,
                    args=args, boundary_loss_meter=boundary_loss_meter, consensus_loss_meter=consensus_loss_meter)

        # Save a model
        if epoch % 2 == 0 and epoch > int(0.9 * args.max_epoch):
            torch.save(
                model.state_dict(),
                os.path.join(cp_dir, 'checkpoint_{}_final.pth'.format(args.start_epoch + epoch)),
            )
        elif epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(cp_dir, 'checkpoint_{}_final.pth'.format(args.start_epoch + epoch)),
            )
    torch.save(
        model.state_dict(),
        os.path.join(cp_dir, 'checkpoint_{}_final.pth'.format(args.start_epoch + args.max_epoch)),
    )

    return None


if __name__ == "__main__":
    main()
