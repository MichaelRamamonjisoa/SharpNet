import torch
import torch.nn as nn
from torch import log as thLog
from torch.autograd import Variable
from torch import Tensor, mul, dot, ones
from torch.nn import functional as F


class DoobNetLoss(nn.Module):
    # Based on Wang et al.
    def __init__(self, beta, gamma, sigma):
        super(DoobNetLoss, self).__init__()
        self.alpha = Variable(Tensor([0]))
        self.beta = beta
        self.gamma = gamma

        self.sigma = sigma

    def forward(self, b_pred, b_gt):
        N = b_gt.shape[0]
        b_pred = b_pred.view(-1, 1)
        b_gt = b_gt.view(-1, 1)
        b_gt = b_gt.float()

        sm = b_gt.sum()
        sz = b_gt.size()[0]

        self.alpha = 1.0 - sm.float() / float(sz)
        alfa = self.alpha * b_gt + (1.0 - self.alpha) * (1.0 - b_gt)

        pt = mul(b_gt, b_pred) + mul(1.0 - b_gt, 1.0 - b_pred)

        clamp_val = 1e-7  # to avoid exploding gradients when taking torch.log

        pt = pt.clamp(min=clamp_val, max=1.0-clamp_val)
        logpt = thLog(pt)
        power_pt = (1.0 - pt) ** self.gamma
        power_pt = power_pt * self.beta * logpt
        loss = -1.0 * alfa * power_pt
        loss = loss.sum()

        return (1.0 / N) * loss


class SharpNetLoss(nn.Module):
    def __init__(self, lamb, mu, use_depth=False, use_normals=False,
                 use_boundary=False, use_geo_consensus=False):
        super(SharpNetLoss, self).__init__()

        self.lamb = lamb
        self.mu = mu
        self.use_normals = use_normals
        self.use_depth = use_depth
        self.use_boundary_loss = use_boundary
        self.use_geo_consensus = use_geo_consensus

        self.masked_spatial_gradients_loss = SpatialGradientsLoss(clamp_value=1e-7,
                                                                  size_average=True,
                                                                  gradient_loss_on=True,
                                                                  smooth_error=True)

        if self.use_depth:
            self.masked_depth_loss = LainaBerHuLoss(use_logs=True)

        if self.use_boundary_loss:
            self.boundary_loss = DoobNetLoss(beta=4, gamma=0.5, sigma=3)

        if self.use_geo_consensus:
            self.norm_depth_bound_consensus_loss = NormalDepthConsensusLoss()
            self.depth_bound_consensus_loss = DepthBoundaryConsensusLoss()

    def forward(self, mask_gt,
                d_pred=None, d_gt=None,
                n_pred=None, n_gt=None,
                b_pred=None, b_gt=None,
                val=False, use_grad=False):

        d_loss = 0
        n_loss = 0
        grad_loss = 0
        b_loss = 0
        geo_loss = 0

        if len(mask_gt.shape) != 4:
            mask_gt = mask_gt.unsqueeze(1)

        mask_gt_valid = mask_gt[:, 0, ...].unsqueeze(1)

        if d_pred is not None:
            d_gt = d_gt.unsqueeze(1)
            d_loss = self.masked_depth_loss(d_pred, d_gt, mask_gt_valid)

            if use_grad:
                grad_loss = self.masked_spatial_gradients_loss(d_pred, d_gt, mask_gt_valid)
            else:
                grad_loss = 0

        if n_pred is not None:
            n_loss = normals_loss(n_pred, n_gt, mask_gt_valid)

        if self.use_boundary_loss:
            b_loss = self.boundary_loss(b_pred, b_gt)
            b_loss = 0.01 * b_loss

        if self.use_geo_consensus:
            db_loss = 0
            ndb_loss = 0

            if d_pred is not None and b_pred is not None:
                db_loss = self.depth_bound_consensus_loss(d_pred, b_pred)
            if n_pred is not None and d_pred is not None and b_pred is not None:
                ndb_loss = self.norm_depth_bound_consensus_loss(n_pred, d_pred, b_pred)

            geo_loss = db_loss + ndb_loss

        return d_loss, grad_loss, n_loss, b_loss, geo_loss


class LainaBerHuLoss(nn.Module):
    # Based on Laina et al.

    def __init__(self, size_average=True, use_logs=True, clamp_val=1e-9):
        super(LainaBerHuLoss, self).__init__()
        self.size_average = size_average
        self.use_log = use_logs
        self.clamp_val = clamp_val

    def forward(self, input, target, mask):
        if self.use_log:
            n = thLog(input.clamp(min=self.clamp_val)) - thLog(target.clamp(min=self.clamp_val))
        else:
            n = input - target

        n = torch.abs(n)
        n = mul(n, mask)

        n = n.squeeze(1)
        c = 0.2 * n.max()
        cond = n < c
        loss = torch.where(cond, n, (n ** 2 + c ** 2) / (2 * c + 1e-9))

        loss = loss.sum()

        if self.size_average:
            return loss / mask.sum()

        return loss


class HuberLoss(nn.Module):
    def __init__(self, size_average=True, use_logs=True, sigma=1):
        super(HuberLoss, self).__init__()
        self.size_average = size_average
        self.sigma = sigma

    def forward(self, input, target, mask=None):
        n = torch.abs(input - target)
        if mask is not None:
            n = mul(n, mask)

        cond = n < 1 / (self.sigma ** 2)
        loss = torch.where(cond, 0.5 * (self.sigma * n) ** 2, n - 0.5 / (self.sigma ** 2))
        if self.size_average:
            if mask is not None:
                return loss.sum() / mask.sum()
            else:
                return loss.mean()
        return loss.sum()


def normals_loss(input, target, mask=None):
    if input is None or target is None:
        return 0
    else:
        prod = mul(input, target)

        if mask is not None:
            n = mask.sum().float()
            prod = mul(prod, mask)
        else:
            n = target.numel().float()

        prod = 1.0 - (1.0 / n) * prod.sum()
        prod = prod.clamp(min=0)

        return prod


class SpatialGradientsLoss(nn.Module):
    def __init__(self, kernel_size=3, use_logs=True, clamp_value=1e-7, size_average=False,
                 smooth_error=True,
                 gradient_loss_on=True):
        super(SpatialGradientsLoss, self).__init__()

        self.size_average = size_average
        self.kernel_size = kernel_size
        self.clamp_value = clamp_value
        self.use_logs = use_logs
        self.smooth_error = smooth_error
        self.gradient_loss_on = gradient_loss_on

        if gradient_loss_on:
            self.masked_huber_loss = HuberLoss(sigma=3)

    def forward(self, input, target, mask=None):

        repeat_channels = target.shape[1]

        sobel_x = torch.Tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])

        sobel_x = sobel_x.view((1, 1, 3, 3))
        sobel_x = torch.autograd.Variable(sobel_x.cuda())

        sobel_y = torch.Tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]])

        sobel_y = sobel_y.view((1, 1, 3, 3))
        sobel_y = torch.autograd.Variable(sobel_y.cuda())
        if repeat_channels != 1:
            sobel_x = sobel_x.repeat(1, repeat_channels, 1, 1)
            sobel_y = sobel_y.repeat(1, repeat_channels, 1, 1)

        smooth_loss = 0
        grad_loss = 0

        if self.smooth_error:
            diff = thLog(input.clamp(min=self.clamp_value)) - thLog(target.clamp(min=self.clamp_value))

            gx_diff = F.conv2d(diff, (1.0 / 8.0) * sobel_x, padding=1)
            gy_diff = F.conv2d(diff, (1.0 / 8.0) * sobel_y, padding=1)

            gradients_diff = torch.pow(gx_diff, 2) + torch.pow(gy_diff, 2)

            if mask is None:
                smooth_loss = gradients_diff.sum()
                if self.size_average:
                    smooth_loss = smooth_loss * (1.0 / gradients_diff.numel())
            else:
                gradients_diff = mul(gradients_diff, mask.repeat(1, 3, 1, 1))
                smooth_loss = gradients_diff.sum()
                if self.size_average:
                    smooth_loss = smooth_loss * (1.0 / mask.sum())

        if self.gradient_loss_on:

            input = thLog(input.clamp(min=self.clamp_value))
            target = thLog(target.clamp(min=self.clamp_value))

            gx_input = F.conv2d(input, (1.0 / 8.0) * sobel_x, padding=1)
            gy_input = F.conv2d(input, (1.0 / 8.0) * sobel_y, padding=1)

            gx_target = F.conv2d(target, (1.0 / 8.0) * sobel_x, padding=1)
            gy_target = F.conv2d(target, (1.0 / 8.0) * sobel_y, padding=1)

            gradients_input = torch.pow(gx_input, 2) + torch.pow(gy_input, 2)
            gradients_target = torch.pow(gx_target, 2) + torch.pow(gy_target, 2)

            grad_loss = self.masked_huber_loss(gradients_input, gradients_target, mask)

        return smooth_loss + grad_loss


class DepthBoundaryConsensusLoss(nn.Module):
    def __init__(self, kernel_size=3, use_logs=True, clamp_value=1e-7, size_average=False):
        super(DepthBoundaryConsensusLoss, self).__init__()

        self.size_average = size_average
        self.kernel_size = kernel_size
        self.clamp_value = clamp_value

    def forward(self, depth, boundary, mask=None):
        repeat_channels = depth.shape[1]

        sobel_x = torch.Tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])

        sobel_x = sobel_x.view((1, 1, 3, 3))
        sobel_x = torch.autograd.Variable(sobel_x.cuda())

        sobel_y = torch.Tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]])

        sobel_y = sobel_y.view((1, 1, 3, 3))
        sobel_y = torch.autograd.Variable(sobel_y.cuda())

        lap = torch.Tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        lap = lap.view((1, 1, 3, 3))
        lap = torch.autograd.Variable(lap.cuda())

        if repeat_channels != 1:
            sobel_x = sobel_x.repeat(1, repeat_channels, 1, 1)
            sobel_y = sobel_y.repeat(1, repeat_channels, 1, 1)
            lap = lap.repeat(1, repeat_channels, 1, 1)

        lap_depth = F.conv2d(depth, (1 / 8.0) * lap, padding=1)

        gx = F.conv2d(depth, (1.0 / 8.0) * sobel_x, padding=1)
        gy = F.conv2d(depth, (1.0 / 8.0) * sobel_y, padding=1)
        g_depth = torch.pow(gx, 2) + torch.pow(gy, 2)
        boundary = boundary.clamp(min=self.clamp_value, max=1 - self.clamp_value)
        loss = torch.abs(mul(mul(g_depth, thLog(boundary)), lap_depth))
        loss = loss + 0.0001 * torch.abs(mul(thLog(1 - boundary), torch.exp(-lap_depth)))
        loss = loss + 0.0001 * torch.abs(boundary)

        if mask is None:
            return loss.sum() / (float(depth.numel()))
        else:
            loss = mul(loss, mask)
            return loss.sum() / float(mask.sum())


class NormalDepthConsensusLoss(nn.Module):
    def __init__(self, kernel_size=3, clamp_value=1e-7, size_average=False):
        super(NormalDepthConsensusLoss, self).__init__()

        self.size_average = size_average
        self.kernel_size = kernel_size
        self.clamp_value = clamp_value

    def forward(self, normals, depth, boundary):
        repeat_channels = depth.shape[1]

        sobel_x = torch.Tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])

        sobel_x = sobel_x.view((1, 1, 3, 3))
        sobel_x = torch.autograd.Variable(sobel_x.cuda())

        sobel_y = torch.Tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]])

        sobel_y = sobel_y.view((1, 1, 3, 3))
        sobel_y = torch.autograd.Variable(sobel_y.cuda())
        if repeat_channels != 1:
            sobel_x = sobel_x.repeat(1, repeat_channels, 1, 1)
            sobel_y = sobel_y.repeat(1, repeat_channels, 1, 1)

        gx_depth = F.conv2d(depth, (1.0 / 8.0) * sobel_x, padding=1)
        gy_depth = F.conv2d(depth, (1.0 / 8.0) * sobel_y, padding=1)

        g_depth = torch.cat((gx_depth, gy_depth), 1)
        g_depth = F.normalize(g_depth, p=2, dim=1)

        normal2d = normals[:, :2, ...]
        normal2d = F.normalize(normal2d, p=2, dim=1)

        prod = mul(g_depth, normal2d)
        prod = prod.sum(1).unsqueeze(1)

        prod = (1.0 - prod).clamp(min=0)

        prod = torch.abs(mul(prod, (-1.0) * thLog(boundary.clamp(min=self.clamp_value))))

        return prod.mean()
