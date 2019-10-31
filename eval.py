import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import Bottleneck as ResBlock
from sharpnet_model import SharpNet
from loss import *
from PIL import Image
try:
    from imageio import imread, imsave
except:
    from scipy.misc import imread, imsave
from data_transforms import ToTensor, Compose, Normalize
from skimage import feature
from scipy import ndimage
import time

import os
import scipy.io as io
import h5py
import cv2

parser = argparse.ArgumentParser(description="Test a model on an image")
parser.add_argument('--rootdir', dest='rootdir',
                    help="Directory containing and nyu_depth_v2_labeled.mat and nyuv2_splits.mat files")
parser.add_argument('--model', '-m', dest='model',
                    help="checkpoint.pth which contains the model")
parser.add_argument('--savepath', dest='savepath', type=str)
parser.add_argument('--cuda', dest='cuda_device', default='', help="To activate inference on GPU, set to GPU_ID")
parser.add_argument('--nocuda', action='store_true')
parser.add_argument('--crop', dest='eigen_crop', action='store_true',
                    help='Flag to evaluate on center crops defined by Eigen')
parser.add_argument('--edges', action='store_true', help='Flag to evaluate on occlusion boundaries')
parser.add_argument('--low', dest='low_threshold', type=float, default=0.03,
                    help='Low threshold of Canny edge detector')
parser.add_argument('--high', dest='high_threshold', type=float, default=0.05,
                    help='High threshold of Canny edge detector')

args = parser.parse_args()


def predict_depth(model, image):

    start_time = time.time()
    depth_pred = model(image)
    inference_time = time.time() - start_time

    return depth_pred, inference_time


def compute_depth_metrics(input, target, mask=None):
    if mask is None:
        rmse = np.sqrt(np.mean((input - target) ** 2))
        rmse_log = np.sqrt(np.mean(np.log10(np.clip(input, a_min=1e-12, a_max=1e12)) - np.log10(
            np.clip(target, a_min=1e-12, a_max=1e12))) ** 2)

        avg_log10 = np.mean(
            np.abs(
                np.log10(np.clip(input, a_min=1e-12, a_max=1e12)) - np.log10(np.clip(target, a_min=1e-12, a_max=1e12))))

        rel = np.mean(np.abs(input - target) / target)
    else:
        N = np.sum(mask)

        diff = mask * (input - target)
        diff = diff ** 2
        diff_log = mask * (np.log(np.clip(input, a_min=1e-12, a_max=1e12)) - np.log(
            np.clip(target, a_min=1e-12, a_max=1e12))) ** 2
        mse = np.sum(diff)
        mse_log = np.sum(diff_log)
        rmse = np.sqrt(float(mse) / N)
        rmse_log = np.sqrt(float(mse_log) / N)

        avg_log10 = np.sum(
            mask * np.abs(np.log10(np.clip(input, a_min=1e-12, a_max=1e8))
                          - np.log10(np.clip(target, a_min=1e-8, a_max=1e8))))
        avg_log10 = float(avg_log10) / N

        rel = float(np.sum(np.abs(input - target) / target)) / N

    acc_map = np.max((target / (input + 1e-8), input / (target + 1e-8)), axis=0)
    acc_1_map = acc_map < 1.25
    acc_2_map = acc_map < 1.25 ** 2
    acc_3_map = acc_map < 1.25 ** 3
    if mask is not None:
        acc_1_map[mask == 0] = False
        acc_2_map[mask == 0] = False
        acc_3_map[mask == 0] = False

        N = np.sum(mask)
    else:
        N = np.prod(input.shape)

    acc_1 = len(acc_1_map[acc_1_map == True]) / N
    acc_2 = len(acc_2_map[acc_2_map == True]) / N
    acc_3 = len(acc_2_map[acc_3_map == True]) / N

    return acc_1, acc_2, acc_3, rel, avg_log10, rmse, rmse_log




def compute_depth_boundary_error(edges_gt, pred, mask=None, low_thresh=0.15, high_thresh=0.3):
    # skip dbe if there is no ground truth distinct edge
    if np.sum(edges_gt) == 0:
        dbe_acc = np.nan
        dbe_com = np.nan
        edges_est = np.empty(pred.shape).astype(int)
    else:

        # normalize est depth map from 0 to 1
        pred_normalized = pred.copy().astype('f')
        pred_normalized[pred_normalized == 0] = np.nan
        pred_normalized = pred_normalized - np.nanmin(pred_normalized)
        pred_normalized = pred_normalized / np.nanmax(pred_normalized)

        # apply canny filter
        edges_est = feature.canny(pred_normalized, sigma=np.sqrt(2), low_threshold=low_thresh,
                                  high_threshold=high_thresh)

        # compute distance transform for chamfer metric
        D_gt = ndimage.distance_transform_edt(1 - edges_gt)
        D_est = ndimage.distance_transform_edt(1 - edges_est)

        max_dist_thr = 10.  # Threshold for local neighborhood

        mask_D_gt = D_gt < max_dist_thr  # truncate distance transform map

        E_fin_est_filt = edges_est * mask_D_gt  # compute shortest distance for all predicted edges
        if mask is None:
            mask = np.ones(shape=E_fin_est_filt.shape)
        E_fin_est_filt = E_fin_est_filt * mask
        D_gt = D_gt * mask

        if np.sum(E_fin_est_filt) == 0:  # assign MAX value if no edges could be detected in prediction
            dbe_acc = max_dist_thr
            dbe_com = max_dist_thr
        else:
            # accuracy: directed chamfer distance of predicted edges towards gt edges
            dbe_acc = np.nansum(D_gt * E_fin_est_filt) / np.nansum(E_fin_est_filt)

            # completeness: sum of undirected chamfer distances of predicted and gt edges
            ch1 = D_gt * edges_est  # dist(predicted,gt)
            ch1[ch1 > max_dist_thr] = max_dist_thr  # truncate distances
            ch2 = D_est * edges_gt  # dist(gt, predicted)
            ch2[ch2 > max_dist_thr] = max_dist_thr  # truncate distances
            res = ch1 + ch2  # summed distances
            dbe_com = np.nansum(res) / (np.nansum(edges_est) + np.nansum(edges_gt))  # normalized

    return dbe_acc, dbe_com, edges_est, D_est



os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
device = torch.device("cuda" if args.cuda_device != '' else "cpu")
print("Running on " + torch.cuda.get_device_name(device))

model = SharpNet(ResBlock, [3, 4, 6, 3], [2, 2, 2, 2, 2],
                 use_normals=False,
                 use_depth=True if args.depth else False,
                 use_boundary=False,
                 bias_decoder=True)

torch.set_grad_enabled(False)

model_dict = model.state_dict()

# Load model
trained_model_path = args.model
trained_model_dict = torch.load(trained_model_path, map_location=lambda storage, loc: storage)

# load image resnet encoder and mask_encoder and normals_decoder (not depth_decoder or normal resnet)
model_weights = {k: v for k, v in trained_model_dict.items() if k in model_dict}

model.load_state_dict(model_weights)
model.eval()
model.to(device)

t = []
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
t.extend([ToTensor(), normalize])
transf = Compose(t)

torch.set_grad_enabled(False)

################### Load Dataset #####################
nyu_splits = io.loadmat(os.path.join(args.rootdir, 'nyuv2_splits.mat'))
nyu_dataset = h5py.File(os.path.join(args.rootdir, 'nyu_depth_v2_labeled.mat'), 'r', libver='latest', swmr=True)

laina_pred = h5py.File(os.path.join(args.rootdir, 'laina_predictions_NYUval.mat'), 'r', libver='latest', swmr=True)
eigen_vgg_pred = io.loadmat(os.path.join(args.rootdir, 'predictions_depth_vgg.mat'))
eigen_alex_pred = io.loadmat(os.path.join(args.rootdir, 'predictions_depth_alexnet.mat'))
jiao_dir = os.path.join(args.rootdir, 'jiao_pred_mat')

ob_dir = os.path.join(args.rootdir, 'occlusion_boundaries')

ob_list = []

if args.edges:
    for root, dir, files in os.walk(os.path.join(args.rootdir, 'occlusion_boundaries')):       
        ob_list = [int(file.split('_ob', 1)[0]) for file in files if file.endswith('_ob.png')]

DORN_dict = {}
for root, dirs, files in os.walk(os.path.join(args.rootdir, 'NYUV2_DORN', 'NYUV2_DORN')):
    for file in files:
        idx = int(file.rsplit('_1', 1)[-1].replace('.mat', ''))
        DORN_dict[idx] = os.path.join(root, file)

n_train = len(nyu_splits['trainNdxs'])
n_test = len(nyu_splits['testNdxs'])

index_dict = {}
for i, idx in enumerate(nyu_splits['testNdxs']):
    index_dict[idx[0]] = i


def round_down(num, divisor):
    return num - (num % divisor)


scale = args.rescale_factor

i = 0
mse = 0

laina_mse = 0
final_log10 = 0
laina_final_log10 = 0
final_rel = 0
final_laina_rel = 0
final_acc1 = 0
laina_final_acc1 = 0
final_acc2 = 0
laina_final_acc2 = 0
final_acc3 = 0
laina_final_acc3 = 0

sil_angle_list = []
lad_angle_list = []

final_depth_score_ours = np.zeros(7)
final_depth_score_laina = np.zeros(7)
final_depth_score_eigen_vgg = np.zeros(7)
final_depth_score_eigen_alex = np.zeros(7)
final_depth_score_dorn = np.zeros(7)
final_depth_score_jiao = np.zeros(7)

final_dbe_scores = np.zeros((6, 2), dtype='float')
# acc1, acc2, acc3, rel, log10 , rmse, rmse_log

final_n_scores_sil = np.zeros(4)
final_n_scores_lad = np.zeros(4)

avg_inference_time = 0

if not args.edges:
    indices = nyu_splits['testNdxs']
else:
    indices = ob_list
    indices = np.array([[int(index) + 1] for index in indices])

# print(indices)
r = np.arange(len(indices))
np.random.shuffle(r)
shuffle_indices = indices[r]

if args.index is not None:
    shuffle_indices = list([[args.index]])

# shuffle_indices = [[1347]]
N = len(shuffle_indices)
for index in shuffle_indices:
    i += 1
    idx = index[0] - 1  # MATLAB

    image = nyu_dataset['images'][idx].swapaxes(0, 2)
    image = Image.fromarray(image)
    sem_labels = nyu_dataset['labels'][idx].swapaxes(0, 1)
    if args.edges:
        edge_labels = imread(os.path.join(args.rootdir, 'occlusion_boundaries', str(idx) + '_ob.png')) / 255
    DORN_pred = io.loadmat(DORN_dict[idx])
    jiao_pred = io.loadmat(os.path.join(jiao_dir, str(index_dict[idx + 1] + 1) + '.mat'))


    w, h = image.size

    box = (6, 6, w - 6, h - 6)
    image = image.crop(box)
    w, h = image.size

    image = image.resize((640, 480), Image.ANTIALIAS)

    depth_gt = nyu_dataset['depths'][idx].swapaxes(0, 1)
    laina_depth = laina_pred['predictions'][index_dict[idx + 1]].swapaxes(0, 1)
    eigen_vgg_depth = eigen_vgg_pred['depths'][:, :, index_dict[idx + 1]]
    eigen_vgg_depth = cv2.resize(eigen_vgg_depth, dsize=(laina_depth.shape[1], laina_depth.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)
    eigen_alex_depth = eigen_alex_pred['depths'][:, :, index_dict[idx + 1]]
    eigen_alex_depth = cv2.resize(eigen_alex_depth, dsize=(laina_depth.shape[1], laina_depth.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)
    DORN_depth = DORN_pred['pred']
    jiao_depth = jiao_pred['pred']

    image_original = image.copy()

    data = [image, None]
    image = transf(*data)

    image = torch.autograd.Variable(image).unsqueeze(0)
    image = image.to(device)

    depth_pred, inference_time = predict_depth(model, image, args)
    avg_inference_time += float(inference_time) / float(N)

    tmp = depth_pred.data.cpu()
    shp = tmp.shape[:2]
    mask_pred = np.ones(shape=shp)
    mask_display = mask_pred
    
    depth_pred_tmp = depth_pred.data.cpu().numpy()[0, 0, ...]
    depth_pred_tmp = depth_pred_tmp * 65535.0 / 1000.0
    depth_pred_tmp = scale * cv2.resize(depth_pred_tmp, dsize=(depth_gt.shape[1] - 12, depth_gt.shape[0] - 12),
                                        interpolation=cv2.INTER_LINEAR)
    depth_pred = cv2.copyMakeBorder(depth_pred_tmp, 6, 6, 6, 6, cv2.BORDER_REPLICATE)
    depth_pred = np.clip(depth_pred, 0.7, 10)

    depth_gt_copy = depth_gt.copy()
    laina_copy = laina_depth.copy()
    eigen_vgg_copy = eigen_vgg_depth.copy()
    eigen_alex_copy = eigen_alex_depth.copy()
    dorn_copy = DORN_depth.copy()
    jiao_copy = jiao_depth.copy()

    if args.eigen_crop:
        mask_eigen = np.zeros(shape=sem_labels.shape)
        eigen_roi = eigen_vgg_pred['predicted_region']
        x0 = 2 * eigen_roi[0][0] - 1
        x1 = 2 * eigen_roi[2][0]
        y0 = 2 * eigen_roi[1][0] - 1
        y1 = 2 * eigen_roi[3][0]

        mask_eigen[y0:y1, x0:x1] = 1
    else:
        mask_eigen = np.ones(shape=sem_labels.shape)

    if args.edges:
        dilatation_size = 5
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                            (dilatation_size, dilatation_size))

        dbe_acc, dbe_com, edges_vgg, d_vgg = compute_depth_boundary_error(edge_labels, eigen_vgg_copy, mask=mask_eigen,
                                                                          low_thresh=args.low_threshold,
                                                                          high_thresh=args.high_threshold)
        final_dbe_scores[0, :] += np.array([dbe_acc, dbe_com]) / N
        dbe_acc, dbe_com, edges_alex, d_alex = compute_depth_boundary_error(edge_labels, eigen_alex_copy, mask=mask_eigen,
                                                                            low_thresh=args.low_threshold,
                                                                            high_thresh=args.high_threshold)
        final_dbe_scores[1, :] += np.array([dbe_acc, dbe_com]) / N
        dbe_acc, dbe_com, edges_laina, d_laina = compute_depth_boundary_error(edge_labels, laina_copy, mask=mask_eigen,
                                                                              low_thresh=args.low_threshold,
                                                                              high_thresh=args.high_threshold)
        final_dbe_scores[2, :] += np.array([dbe_acc, dbe_com]) / N
        dbe_acc, dbe_com, edges_dorn, d_dorn = compute_depth_boundary_error(edge_labels, dorn_copy, mask=mask_eigen,
                                                                            low_thresh=args.low_threshold,
                                                                            high_thresh=args.high_threshold)
        final_dbe_scores[3, :] += np.array([dbe_acc, dbe_com]) / N
        dbe_acc, dbe_com, edges_jiao, d_jiao = compute_depth_boundary_error(edge_labels, jiao_copy, mask=mask_eigen,
                                                                            low_thresh=args.low_threshold,
                                                                            high_thresh=args.high_threshold)
        final_dbe_scores[4, :] += np.array([dbe_acc, dbe_com]) / N
        dbe_acc, dbe_com, edges_ours, d_ours = compute_depth_boundary_error(edge_labels, depth_pred, mask=mask_eigen,
                                                                            low_thresh=args.low_threshold,
                                                                            high_thresh=args.high_threshold)
        final_dbe_scores[5, :] += np.array([dbe_acc, dbe_com]) / N

        dbe_acc, dbe_com, edges_gt, d_gt = compute_depth_boundary_error(edge_labels, depth_gt_copy, mask=mask_eigen,
                                                                        low_thresh=args.low_threshold,
                                                                        high_thresh=args.high_threshold)



    # ########################################################

    m = np.min(depth_gt_copy)
    M = np.max(depth_gt_copy)

    if args.edges:

        the_dict = {}
        the_dict['vgg'] = (edges_vgg, d_vgg)
        the_dict['alex'] = (edges_alex, d_alex)
        the_dict['laina'] = (edges_laina, d_laina, laina_copy)
        the_dict['dorn'] = (edges_dorn, d_dorn, dorn_copy)
        the_dict['jiao'] = (edges_jiao, d_jiao, jiao_copy)
        the_dict['ours'] = (edges_ours, d_ours, depth_pred)
        the_dict['gt'] = (edges_gt, d_gt, depth_gt_copy)

        D_gt = ndimage.distance_transform_edt(1 - edge_labels)
        save_path = args.savepath
        suffix = 'L' + str(args.low_threshold) + 'H' + str(args.high_threshold)

        if args.savepath:
            if not os.path.exists(os.path.join(save_path, suffix)):
                os.makedirs(os.path.join(save_path, suffix))

        for j, method in enumerate(the_dict.keys()):

            dilatation_size = 1
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                                (dilatation_size, dilatation_size))
            large_edges = cv2.dilate(edge_labels, element)
            edges_detected = the_dict[method][0]
            edges_detected = edges_detected.astype('uint8')
            large_detected = cv2.dilate(edges_detected, element)
            large_detected = 1 - large_detected
            large_detected = 255 * large_detected
            output = 255 * np.ones(shape=(edges_detected.shape[0], edges_detected.shape[1], 3))

            output[large_detected == 0, :] = 0
            output[large_edges == 1, 0] = 255
            output[large_edges == 1, 1] = 0
            output[large_edges == 1, 2] = 0

            if args.save:
                imsave(
                    os.path.join(save_path, suffix, str(index[0] - 1) + '_' + str(method) + '_edges_gt_overlaid.png'),
                    output)
            edt = np.clip(the_dict[method][1], 0, 10)
            m_e = np.min(edt)
            M_e = np.max(edt)
            edt = (edt - m_e) / (M_e - m_e)
            edt *= 65535
            edt = edt.astype('uint16')
            if args.save:
                imsave(os.path.join(save_path, suffix, str(index[0] - 1) + '_' + str(method) + '_edt.png'), edt)

            # depth
            dep = the_dict[method][2]
            dep = (dep - m) / (M - m)
            dep = Image.fromarray(np.uint8(plt.cm.jet(dep) * 255))
            jet_depth = np.array(dep)[:, :, :3]
            if args.save:
                imsave(os.path.join(save_path, 'nyuv2_ob_depths', str(index[0] - 1) + '_' + str(method) + '_depth.png'),
                       jet_depth)

        # plt.show()

        # ########################################################

    if args.display:
        plt.subplot(2, 6, 1)
        plt.imshow(image_original)
        plt.title('Input')

        plt.subplot(2, 6, 7)
        plt.imshow(depth_gt_copy)
        plt.set_cmap('jet')
        plt.title('Depth GT')

        plt.subplot(2, 6, 2)
        p = plt.imshow(eigen_vgg_copy)
        plt.set_cmap('jet')
        p.set_clim(m, M)
        plt.title('Eigen')

        plt.subplot(2, 6, 3)
        p = plt.imshow(laina_copy)
        plt.set_cmap('jet')
        p.set_clim(m, M)
        plt.title('Laina')

        plt.subplot(2, 6, 4)
        plt.imshow(jiao_depth)
        plt.set_cmap('jet')
        plt.title('Look deeper')

        plt.subplot(2, 6, 5)
        p = plt.imshow(dorn_copy)
        plt.set_cmap('jet')
        p.set_clim(m, M)
        plt.title('DORN')

        plt.subplot(2, 6, 6)
        p = plt.imshow(depth_pred)
        plt.set_cmap('jet')
        p.set_clim(m, M)
        plt.title('Ours')

        plt.subplot(2, 6, 8)
        plt.imshow(np.abs(depth_gt - eigen_vgg_copy))
        plt.set_cmap('jet')
        plt.subplot(2, 6, 9)
        plt.imshow(np.abs(depth_gt - laina_copy))
        plt.set_cmap('jet')
        plt.subplot(2, 6, 10)
        plt.imshow(np.abs(depth_gt - jiao_copy))
        plt.set_cmap('jet')
        plt.subplot(2, 6, 11)
        plt.imshow(np.abs(depth_gt - dorn_copy))
        plt.set_cmap('jet')
        plt.subplot(2, 6, 12)
        plt.imshow(np.abs(depth_gt - depth_pred))
        plt.set_cmap('jet')

        fig = plt.gcf()
        fig.set_size_inches(22, 16)

        acc_1, acc_2, acc_3, rel, avg_log10, rmse, rmselog = compute_depth_metrics(depth_pred, depth_gt_copy, mask_eigen)

        laina_acc_1, laina_acc_2, laina_acc_3, laina_rel, laina_avg_log10, laina_rmse, laina_rmselog = compute_depth_metrics(
            laina_copy,
            depth_gt,
            mask_eigen)
        eigen_acc_1, eigen_acc_2, eigen_acc_3, eigen_rel, eigen_avg_log10, eigen_rmse, eigen_rmselog = compute_depth_metrics(
            eigen_vgg_copy,
            depth_gt,
            mask_eigen)
        jiao_acc_1, jiao_acc_2, jiao_acc_3, jiao_rel, jiao_avg_log10, jiao_rmse, jiao_rmselog = compute_depth_metrics(
            jiao_depth,
            depth_gt,
            mask_eigen)
        DORN_acc_1, DORN_acc_2, DORN_acc_3, DORN_rel, DORN_avg_log10, DORN_rmse, DORN_rmselog = compute_depth_metrics(
            dorn_copy,
            depth_gt,
            mask_eigen)

        plt.suptitle('RMSE: ours {} / laina {} / eigen {} / Jiao {} / DORN {}'.format(rmse,
                                                                                      laina_rmse,
                                                                                      eigen_rmse,
                                                                                      jiao_rmse,
                                                                                      DORN_rmse))

        plt.show()

    metrics_ours = compute_depth_metrics(depth_pred, depth_gt_copy, mask_eigen)
    metrics_laina = compute_depth_metrics(laina_copy, depth_gt_copy, mask_eigen)
    metrics_eigen_vgg = compute_depth_metrics(eigen_vgg_copy, depth_gt_copy, mask_eigen)
    metrics_eigen_alex = compute_depth_metrics(eigen_alex_copy, depth_gt_copy, mask_eigen)
    metrics_dorn = compute_depth_metrics(dorn_copy, depth_gt_copy, mask_eigen)
    metrics_jiao = compute_depth_metrics(jiao_copy, depth_gt_copy, mask_eigen)

    final_depth_score_ours[:5] += np.array(metrics_ours[:5]) / N
    final_depth_score_laina[:5] += np.array(metrics_laina[:5]) / N
    final_depth_score_eigen_vgg[:5] += np.array(metrics_eigen_vgg[:5]) / N
    final_depth_score_eigen_alex[:5] += np.array(metrics_eigen_alex[:5]) / N
    final_depth_score_dorn[:5] += np.array(metrics_dorn[:5]) / N
    final_depth_score_jiao[:5] += np.array(metrics_jiao[:5]) / N

    final_depth_score_ours[5:] += np.power(metrics_ours[5:], 2) / N
    final_depth_score_laina[5:] += np.power(metrics_laina[5:], 2) / N
    final_depth_score_eigen_vgg[5:] += np.power(metrics_eigen_vgg[5:], 2) / N
    final_depth_score_eigen_alex[5:] += np.power(metrics_eigen_alex[5:], 2) / N
    final_depth_score_dorn[5:] += np.power(metrics_dorn[5:], 2) / N
    final_depth_score_jiao[5:] += np.power(metrics_jiao[5:], 2) / N

    print('Done {} / {}'.format(i, N))
    print('RMSE: ours {0:.3f} / laina {1:.3f} / eigenVGG {2:.3f} / '
          'eigenAlexNet {3:.3f} / Look Deeper {4:.3f} / DORN {5:.3f}'.format(np.sqrt(metrics_ours[5]),
                                                                             np.sqrt(metrics_laina[5]),
                                                                             np.sqrt(metrics_eigen_vgg[5]),
                                                                             np.sqrt(metrics_eigen_alex[5]),
                                                                             np.sqrt(metrics_jiao[5]),
                                                                             np.sqrt(metrics_dorn[5])))
    print('RMSE (log): ours {0:.3f} / laina {1:.3f} / eigen {2:.3f} / '
          'Look Deeper {3:.3f} / DORN {4:.3f}'.format(np.sqrt(metrics_ours[6]),
                                                      np.sqrt(metrics_laina[6]),
                                                      np.sqrt(metrics_eigen_vgg[6]),
                                                      np.sqrt(metrics_eigen_alex[6]),
                                                      np.sqrt(metrics_jiao[6]),
                                                      np.sqrt(metrics_dorn[6])))
    print('log10: ours {0:.3f} / laina {1:.3f} / eigen {2:.3f} /'
          ' Look Deeper {3:.3f} / DORN {4:.3f}'.format(metrics_ours[4],
                                                       metrics_laina[4],
                                                       metrics_eigen_vgg[4],
                                                       metrics_eigen_alex[4],
                                                       metrics_jiao[4],
                                                       metrics_dorn[4]))
    print('rel: ours {0:.3f} / laina {1:.3f} / eigen {2:.3f} /'
          ' Look Deeper {3:.3f} / DORN {4:.3f}'.format(metrics_ours[3],
                                                       metrics_laina[3],
                                                       metrics_eigen_vgg[3],
                                                       metrics_eigen_alex[3],
                                                       metrics_jiao[3],
                                                       metrics_dorn[3]))
    print('acc1: ours {0:.3f} / laina {1:.3f} / eigen {2:.3f} /'
          ' Look Deeper {3:.3f} / DORN {4:.3f}'.format(metrics_ours[0],
                                                       metrics_laina[0],
                                                       metrics_eigen_vgg[0],
                                                       metrics_eigen_alex[0],
                                                       metrics_jiao[0],
                                                       metrics_dorn[0]))

    print('acc2: ours {0:.3f} / laina {1:.3f} / eigen {2:.3f} /'
          ' Look Deeper {3:.3f} / DORN {4:.3f}'.format(metrics_ours[1],
                                                       metrics_laina[1],
                                                       metrics_eigen_vgg[1],
                                                       metrics_eigen_alex[1],
                                                       metrics_jiao[1],
                                                       metrics_dorn[1]))
    print('acc3: ours {0:.3f} / laina {1:.3f} / eigen {2:.3f} / '
          'Look Deeper {3:.3f} / DORN {4:.3f}'.format(metrics_ours[2],
                                                      metrics_laina[2],
                                                      metrics_eigen_vgg[2],
                                                      metrics_eigen_alex[2],
                                                      metrics_jiao[2],
                                                      metrics_dorn[2]))

    if args.save:
        img_name = str(idx) + '.png'
        if not os.path.exists('results_NYU'):
            os.mkdir('results_NYU')

        cv2.imwrite('results_NYU/' + img_name, (1000 * depth_pred).astype(np.uint16))


rmse = np.sqrt(mse)
laina_rmse = np.sqrt(laina_mse)

final_depth_score_ours[5:] = np.sqrt(final_depth_score_ours[5:])
final_depth_score_laina[5:] = np.sqrt(final_depth_score_laina[5:])
final_depth_score_eigen_vgg[5:] = np.sqrt(final_depth_score_eigen_vgg[5:])
final_depth_score_eigen_alex[5:] = np.sqrt(final_depth_score_eigen_alex[5:])
final_depth_score_dorn[5:] = np.sqrt(final_depth_score_dorn[5:])
final_depth_score_jiao[5:] = np.sqrt(final_depth_score_jiao[5:])

# acc1, acc2, acc3, rel, log10 , rmse, rmselog

print('======================= Final results ===========================')

print('acc1: ours {0:.3f} / laina {1:.3f} / eigenVGG {2:.3f} / '
      'eigenAlex {3:.3f} / Look Deeper {4:.3f} / DORN {5:.3f}'.format(final_depth_score_ours[0],
                                                                      final_depth_score_laina[0],
                                                                      final_depth_score_eigen_vgg[0],
                                                                      final_depth_score_eigen_alex[0],
                                                                      final_depth_score_jiao[0],
                                                                      final_depth_score_dorn[0]))
print('acc2: ours {0:.3f} / laina {1:.3f} / eigenVGG {2:.3f} / '
      'eigenAlex {3:.3f} / Look Deeper {4:.3f} / DORN {5:.3f}'.format(final_depth_score_ours[1],
                                                                      final_depth_score_laina[1],
                                                                      final_depth_score_eigen_vgg[1],
                                                                      final_depth_score_eigen_alex[1],
                                                                      final_depth_score_jiao[1],
                                                                      final_depth_score_dorn[1]))
print('acc3: ours {0:.3f} / laina {1:.3f} / eigenVGG {2:.3f} / '
      'eigenAlex {3:.3f} / Look Deeper {4:.3f} / DORN {5:.3f}'.format(final_depth_score_ours[2],
                                                                      final_depth_score_laina[2],
                                                                      final_depth_score_eigen_vgg[2],
                                                                      final_depth_score_eigen_alex[2],
                                                                      final_depth_score_jiao[2],
                                                                      final_depth_score_dorn[2]))
print('rel: ours {0:.3f} / laina {1:.3f} / eigenVGG {2:.3f} / '
      'eigenAlex {3:.3f} / Look Deeper {4:.3f} / DORN {5:.3f}'.format(final_depth_score_ours[3],
                                                                      final_depth_score_laina[3],
                                                                      final_depth_score_eigen_vgg[3],
                                                                      final_depth_score_eigen_alex[3],
                                                                      final_depth_score_jiao[3],
                                                                      final_depth_score_dorn[3]))
print('log10: ours {0:.3f} / laina {1:.3f} / eigenVGG {2:.3f} / '
      'eigenAlex {3:.3f} / Look Deeper {4:.3f} / DORN {5:.3f}'.format(final_depth_score_ours[4],
                                                                      final_depth_score_laina[4],
                                                                      final_depth_score_eigen_vgg[4],
                                                                      final_depth_score_eigen_alex[4],
                                                                      final_depth_score_jiao[4],
                                                                      final_depth_score_dorn[4]))
print('RMSE: ours {0:.3f} / laina {1:.3f} / eigenVGG {2:.3f} / '
      'eigenAlex {3:.3f} / Look Deeper {4:.3f} / DORN {5:.3f}'.format(final_depth_score_ours[5],
                                                                      final_depth_score_laina[5],
                                                                      final_depth_score_eigen_vgg[5],
                                                                      final_depth_score_eigen_alex[5],
                                                                      final_depth_score_jiao[5],
                                                                      final_depth_score_dorn[5]))
print('RMSE log: ours {0:.3f} / laina {1:.3f} / eigenVGG {2:.3f} / '
      'eigenAlex {3:.3f} / Look Deeper {4:.3f} / DORN {5:.3f}'.format(final_depth_score_ours[6],
                                                                      final_depth_score_laina[6],
                                                                      final_depth_score_eigen_vgg[6],
                                                                      final_depth_score_eigen_alex[6],
                                                                      final_depth_score_jiao[6],
                                                                      final_depth_score_dorn[6]))

if args.edges:
    print('---------------- DBE --------------')
    print('DBE: ours {0:.3f} / laina {1:.3f} / eigen VGG {2:.3f} / '
          'eigen Alex {3:.3f} / Look Deeper {4:.3f} / DORN {5:.3f}'.format(final_dbe_scores[5, 0],
                                                                           final_dbe_scores[2, 0],
                                                                           final_dbe_scores[0, 0],
                                                                           final_dbe_scores[1, 0],
                                                                           final_dbe_scores[4, 0],
                                                                           final_dbe_scores[3, 0],
                                                                           ))
    print('DBE comp: ours {0:.3f} / laina {1:.3f} / eigen VGG {2:.3f} / '
          'eigen Alex {3:.3f} / Look Deeper {4:.3f} / DORN {5:.3f}'.format(final_dbe_scores[5, 1],
                                                                           final_dbe_scores[2, 1],
                                                                           final_dbe_scores[0, 1],
                                                                           final_dbe_scores[1, 1],
                                                                           final_dbe_scores[4, 1],
                                                                           final_dbe_scores[3, 1]
                                                                           ))

print('Eigen et al (VGG) & {0:.3f} & {1:.3f} & {2:.3f} & {3:.3f} & {4:.3f} & {5:.3f} & {6:.3f} \\\\'.format(
    final_depth_score_eigen_vgg[0],
    final_depth_score_eigen_vgg[1],
    final_depth_score_eigen_vgg[2],
    final_depth_score_eigen_vgg[3],
    final_depth_score_eigen_vgg[4],
    final_depth_score_eigen_vgg[5],
    final_depth_score_eigen_vgg[6]
    ))
print('Eigen et al (AlexNet) & {0:.3f} & {1:.3f} & {2:.3f} & {3:.3f} & {4:.3f} & {5:.3f} & {6:.3f} \\\\'.format(
    final_depth_score_eigen_alex[0],
    final_depth_score_eigen_alex[1],
    final_depth_score_eigen_alex[2],
    final_depth_score_eigen_alex[3],
    final_depth_score_eigen_alex[4],
    final_depth_score_eigen_alex[5],
    final_depth_score_eigen_alex[6]
    ))
print('Laina et al & {0:.3f} & {1:.3f} & {2:.3f} & {3:.3f} & {4:.3f} & {5:.3f} & {6:.3f} \\\\'.format(
    final_depth_score_laina[0],
    final_depth_score_laina[1],
    final_depth_score_laina[2],
    final_depth_score_laina[3],
    final_depth_score_laina[4],
    final_depth_score_laina[5],
    final_depth_score_laina[6]
    ))
print('Fu et al & {0:.3f} & {1:.3f} & {2:.3f} & {3:.3f} & {4:.3f} & {5:.3f} & {6:.3f} \\\\'.format(
    final_depth_score_dorn[0],
    final_depth_score_dorn[1],
    final_depth_score_dorn[2],
    final_depth_score_dorn[3],
    final_depth_score_dorn[4],
    final_depth_score_dorn[5],
    final_depth_score_dorn[6]
    ))
print('Jiao et al & {0:.3f} & {1:.3f} & {2:.3f} & {3:.3f} & {4:.3f} & {5:.3f} & {6:.3f} \\\\'.format(
    final_depth_score_jiao[0],
    final_depth_score_jiao[1],
    final_depth_score_jiao[2],
    final_depth_score_jiao[3],
    final_depth_score_jiao[4],
    final_depth_score_jiao[5],
    final_depth_score_jiao[6]
    ))
print(
    'Ours & {0:.3f} & {1:.3f} & {2:.3f} & {3:.3f} & {4:.3f} & {5:.3f} & {6:.3f} \\\\'.format(final_depth_score_ours[0],
                                                                                             final_depth_score_ours[1],
                                                                                             final_depth_score_ours[2],
                                                                                             final_depth_score_ours[3],
                                                                                             final_depth_score_ours[4],
                                                                                             final_depth_score_ours[5],
                                                                                             final_depth_score_ours[6]
                                                                                             ))

if args.edges:
    print('*** DBE table *****')

    print('\hline')
    print('Method & $\epsilon_{DBE}^{acc}$ (px) & $\epsilon_{DBE}^{comp}$ (px) \\\\')
    print('\hline \hline')
    print('Eigen et al (VGG) & {0:.3f} & {1:.3f} \\\\'.format(final_dbe_scores[0, 0], final_dbe_scores[0, 1]))
    print('Eigen et al (AlexNet) & {0:.3f} & {1:.3f} \\\\'.format(final_dbe_scores[1, 0], final_dbe_scores[1, 1]))
    print('Laina et al & {0:.3f} & {1:.3f} \\\\'.format(final_dbe_scores[2, 0], final_dbe_scores[2, 1]))
    print('Fu et al & {0:.3f} & {1:.3f} \\\\'.format(final_dbe_scores[3, 0], final_dbe_scores[3, 1]))
    print('Jiao et al & {0:.3f} & {1:.3f} \\\\'.format(final_dbe_scores[4, 0], final_dbe_scores[4, 1]))
    print('Ours & {0:.3f} & {1:.3f} \\\\'.format(final_dbe_scores[5, 0], final_dbe_scores[5, 1]))

print('Average inference time: {}'.format(avg_inference_time))
print('FPS: {}'.format(1.0 / avg_inference_time))
