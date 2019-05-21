import numpy as np
import torch


class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean)
        self.std = torch.FloatTensor(std)

    def __call__(self, image, labels=None):

        if image.device.type != 'cpu':
            means = [self.mean] * image.size()[0]
            stds = [self.std] * image.size()[0]
            for t, m, s in zip(image, means, stds):
                t.sub_(m[:, None, None].cuda()).div_(s[:, None, None].cuda())
        else:
            for t, m, s in zip(image, self.mean, self.std):
                t.sub_(m).div_(s)

        if labels is None:
            return image
        else:
            # final return should be a tuple
            return tuple([image] + list(labels))


class ToTensor(object):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic, labels=None):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic)
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
            if pic.mode == 'YCbCr':
                nchannel = 3
            else:
                nchannel = len(pic.mode)
            img = img.view(pic.size[1], pic.size[0], nchannel)
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255)

        if labels is None:
            return [img]
        else:
            for i, label in enumerate(labels):
                # ground truth mask
                if label is not None:
                    if i == 0:
                        if len(label.shape) == 3:
                            # case with two masks
                            labels[i] = torch.LongTensor(np.array(label.swapaxes(1, 2).swapaxes(0, 1), dtype=np.int))
                        else:
                            labels[i] = torch.LongTensor(np.array(label, dtype=np.int))
                    else:
                        if len(label.shape) == 3:
                            labels[i] = torch.FloatTensor(
                                np.array(label.swapaxes(1, 2).swapaxes(0, 1), dtype=np.float32))
                        else:
                            # depth, boundaries_out, orientations
                            labels[i] = torch.FloatTensor(np.array(label, dtype=np.float32))

            labels = [label for label in labels if label is not None]

        return img, labels


class Compose(object):
    """Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            # if not isinstance(t, RandomHorizontalFlip):
            args = t(*args)
        return args
