from PIL import Image
import cv2
import numpy as np
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from data_transforms import *


def round_down(num, divisor):
    return num - (num % divisor)


def get_np_preds(image_pil, model, device, args):
    normals = None
    boundary = None
    depth = None

    image_np = np.array(image_pil)
    w, h = image_pil.size

    scale = args.rescale_factor

    h_new = round_down(int(h * scale), 16)
    w_new = round_down(int(w * scale), 16)

    if len(image_np.shape) == 2 or image_np.shape[-1] == 1:
        print("Input image has only 1 channel, please use an RGB or RGBA image")
        sys.exit(0)

    if len(image_np.shape) == 4 or image_np.shape[-1] == 4:
        # RGBA image to be converted to RGB
        image_pil = image_pil.convert('RGBA')
        image = Image.new("RGB", (image_np.shape[1], image_np.shape[0]), (255, 255, 255))
        image.paste(image_pil.copy(), mask=image_pil.split()[3])
    else:
        image = image_pil

    image = image.resize((w_new, h_new), Image.ANTIALIAS)

    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    t = []
    t.extend([ToTensor(), normalize])
    transf = Compose(t)

    data = [image, None]
    image = transf(*data)

    image = torch.autograd.Variable(image).unsqueeze(0)
    image = image.to(device)

    if args.boundary:
        if args.depth and args.normals:
            depth_pred, normals_pred, boundary_pred = model(image)
            tmp = normals_pred.data.cpu()
        elif args.depth and not args.normals:
            depth_pred, boundary_pred = model(image)
            tmp = depth_pred.data.cpu()
        elif args.normals and not args.depth:
            normals_pred, boundary_pred = model(image)
            tmp = normals_pred.data.cpu()
        else:
            boundary_pred = model(image)
            tmp = boundary_pred.data.cpu()
    else:
        if args.depth:
            depth_pred = model(image)
            tmp = depth_pred.data.cpu()
        if args.depth and args.normals:
            depth_pred, normals_pred = model(image)
            tmp = normals_pred.data.cpu()
        if args.normals and not args.depth:
            normals_pred = model(image)
            tmp = normals_pred.data.cpu()

    shp = tmp.shape[2:]

    if args.normals:
        normals_pred = normals_pred.data.cpu().numpy()[0, ...]
        normals_pred = normals_pred.swapaxes(0, 1).swapaxes(1, 2)
        normals_pred[..., 0] = 0.5 * (normals_pred[..., 0] + 1)
        normals_pred[..., 1] = 0.5 * (normals_pred[..., 1] + 1)
        normals_pred[..., 2] = -0.5 * np.clip(normals_pred[..., 2], -1, 0) + 0.5

        normals_pred[..., 0] = normals_pred[..., 0] * 255
        normals_pred[..., 1] = normals_pred[..., 1] * 255
        normals_pred[..., 2] = normals_pred[..., 2] * 255

        normals = normals_pred.astype('uint8')

    if args.depth:
        depth_pred = depth_pred.data.cpu().numpy()[0, 0, ...] * 65535 / 1000
        depth_pred = (1 / scale) * cv2.resize(depth_pred, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
        m = np.min(depth_pred)
        M = np.max(depth_pred)
        depth_pred = (depth_pred - m) / (M - m)
        depth = Image.fromarray(np.uint8(plt.cm.jet(depth_pred) * 255))
        depth = np.array(depth)[:, :, :3]

    if args.boundary:
        boundary_pred = boundary_pred.data.cpu().numpy()[0, 0, ...]
        boundary_pred = cv2.resize(boundary_pred, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
        boundary_pred = np.clip(boundary_pred, 0, 10)
        boundary = (boundary_pred * 255).astype('uint8')

    return tuple([depth, normals, boundary])


def get_params(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            for p in m.parameters():
                yield p


def freeze_model_decoders(model, freeze_decoders):
    if 'normals' in freeze_decoders:
        model.normals_decoder.freeze()
    if 'depth' in freeze_decoders:
        model.depth_decoder.freeze()
    if 'boundary' in freeze_decoders:
        model.boundary_decoder.freeze()


def get_gt_sample(dataloader, loader_iter, args):
    try:
        data = next(loader_iter)
    except:
        loader_iter = iter(dataloader)
        data = next(loader_iter)

    if args.depth:
        if args.boundary and args.normals:
            if len(data) == 5:
                # normals and boundary GT
                input, mask_gt, depth_gt, normals_gt, boundary_gt = data
            else:
                # NYU
                input, mask_gt, depth_gt = data
                normals_gt = None
                boundary_gt = None
        elif args.boundary and not args.normals:
            input, mask_gt, depth_gt, boundary_gt = data
        elif args.boundary:
            input, mask_gt, depth_gt, normals_gt = data
        else:
            input, mask_gt, depth_gt = data
    else:
        if args.boundary and args.normals:
            input, mask_gt, normals_gt, boundary_gt = data
        elif args.normals and not args.boundary:
            input, mask_gt, normals_gt = data

    input = input.cuda(async=False)
    mask_gt = mask_gt.cuda(async=False)
    if normals_gt is not None:
        normals_gt = normals_gt.cuda(async=False)
        normals_gt = torch.autograd.Variable(normals_gt)
    if depth_gt is not None:
        depth_gt = depth_gt.cuda(async=False)
        depth_gt = torch.autograd.Variable(depth_gt)
    if boundary_gt is not None:
        boundary_gt = boundary_gt.cuda(async=False)
        boundary_gt = torch.autograd.Variable(boundary_gt)

    input = torch.autograd.Variable(input)
    mask_gt = torch.autograd.Variable(mask_gt)
    return input, mask_gt, depth_gt, normals_gt, boundary_gt


def write_loss_components(tb_writer, iteration, epoch, dataset_size, args,
                          depth_loss_meter=None, depth_loss=None,
                          normals_loss_meter=None, normals_loss=None,
                          boundary_loss_meter=None, boundary_loss=None,
                          grad_loss_meter=None, grad_loss=None,
                          consensus_loss_meter=None, consensus_loss=None):

    if args.normals and normals_loss_meter is not None:
        if args.verbose:
            print('Normals loss: ' + str(float(normals_loss)))
        tb_writer.add_scalar("normals_loss", normals_loss_meter.value()[0],
                             int(epoch) * int(dataset_size / args.batch_size) + iteration)
    if args.depth and depth_loss_meter is not None:
        if args.verbose:
            print('Depth loss: ' + str(float(depth_loss)))
            print('Gradient loss: ' + str(float(grad_loss)))
        tb_writer.add_scalar("Depth_loss", depth_loss_meter.value()[0],
                             int(epoch) * int(dataset_size / args.batch_size) + iteration)
        tb_writer.add_scalar("grad_loss", grad_loss_meter.value()[0],
                             int(epoch) * int(dataset_size / args.batch_size) + iteration)
    if args.boundary and boundary_loss_meter is not None:
        if args.verbose:
            print('Boundary loss: ' + str(float(boundary_loss)))
        tb_writer.add_scalar("boundary loss", boundary_loss_meter.value()[0],
                             int(epoch) * int(dataset_size / args.batch_size) + iteration)
    if args.geo_consensus and consensus_loss_meter is not None:
        if args.verbose:
            print('Consensus loss: ' + str(float(consensus_loss)))
        tb_writer.add_scalar("consensus loss", consensus_loss_meter.value()[0],
                             int(epoch) * int(dataset_size / args.batch_size) + iteration)


def get_tensor_preds(input, model, args):
    depth_pred = None
    normals_pred = None
    boundary_pred = None
    if args.depth:
        if args.boundary and args.normals:
            depth_pred, normals_pred, boundary_pred = model(input)
        elif args.boundary and not args.normals:
            depth_pred, boundary_pred = model(input)
        elif args.normals:
            depth_pred, normals_pred = model(input)
        else:
            depth_pred = model(input)
    else:
        if args.boundary and args.normals:
            normals_pred, boundary_pred = model(input)
        elif args.boundary and not args.normals:
            boundary_pred = model(input)
        else:
            normals_pred = model(input)

    return depth_pred, normals_pred, boundary_pred


def adjust_learning_rate(lr, lr_mode, step, max_epoch, optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    if lr_mode == 'step':
        lr = lr * (0.1 ** (epoch // step))
    elif lr_mode == 'poly':
        lr = lr * (1 - epoch / max_epoch) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
