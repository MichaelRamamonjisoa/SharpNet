import random
import numpy as np
from PIL import Image
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


def pad_reflection(image, top, bottom, left, right):
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return image
    h, w = image.shape[:2]
    next_top = next_bottom = next_left = next_right = 0
    if top > h - 1:
        next_top = top - h + 1
        top = h - 1
    if bottom > h - 1:
        next_bottom = bottom - h + 1
        bottom = h - 1
    if left > w - 1:
        next_left = left - w + 1
        left = w - 1
    if right > w - 1:
        next_right = right - w + 1
        right = w - 1
    new_shape = list(image.shape)
    new_shape[0] += top + bottom
    new_shape[1] += left + right
    new_image = np.empty(new_shape, dtype=image.dtype)
    new_image[top:top+h, left:left+w] = image
    new_image[:top, left:left+w] = image[top:0:-1, :]
    new_image[top+h:, left:left+w] = image[-1:-bottom-1:-1, :]
    new_image[:, :left] = new_image[:, left*2:left:-1]
    new_image[:, left+w:] = new_image[:, -right-1:-right*2-1:-1]
    return pad_reflection(new_image, next_top, next_bottom,
                          next_left, next_right)


def pad_constant(image, top, bottom, left, right, value):
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return image

    h, w = image.shape[:2]
    new_shape = list(image.shape)
    new_shape[0] += top + bottom
    new_shape[1] += left + right
    new_image = np.empty(new_shape, dtype=image.dtype)
    new_image.fill(value)
    new_image[top:top + h, left:left + w] = image

    return new_image


def pad_image(mode, image, top, bottom, left, right, value=0):
    if mode == 'reflection':
        if type(image) == np.ndarray:
            return pad_reflection(np.asarray(image), top, bottom, left, right)
        else:
            return Image.fromarray(
                pad_reflection(np.asarray(image), top, bottom, left, right))
    elif mode == 'constant':
        if type(image) == np.ndarray:
            return pad_constant(np.asarray(image), top, bottom, left, right, value)
        else:
            return Image.fromarray(
                pad_constant(np.asarray(image), top, bottom, left, right, value))
    else:
        raise ValueError('Unknown mode {}'.format(mode))


def get_random_crop(data, tw, th):
    top = bottom = left = right = 0
    w, h = data[0].data.size

    if w < tw:
        left = (tw - w) // 2
        right = tw - w - left
    if h < th:
        top = (th - h) // 2
        bottom = th - h - top

    if left > 0 or right > 0 or top > 0 or bottom > 0:
        data[0].data = pad_image('reflection', data[0].data, top, bottom, left, right)
        for i, mode in enumerate(data[1:]):
            if mode is not None:
                data[i + 1].data = pad_image('constant', data[i + 1].data, top, bottom, left, right, value=0)

    w, h = data[0].data.size
    if w == tw and h == th:
        # should happen after above when image is smaller than crop size
        return data

    # crop next to objects
    [y_mask, x_mask] = np.where(data[1].data == 1)

    right_bb = np.max(x_mask)
    left_bb = np.min(x_mask)
    top_bb = np.min(y_mask)
    bottom_bb = np.max(y_mask)

    x_c = int(0.5 * (right_bb + left_bb))
    y_c = int(0.5 * (bottom_bb + top_bb))

    delta_x = np.max(x_mask) - np.min(x_mask)
    delta_y = np.max(y_mask) - np.min(y_mask)

    x_min = max(0, x_c - int(0.5 * (delta_x + tw)))
    x_max = max(0, min(w - tw, x_c + int(0.5 * (delta_x - tw))))
    y_min = max(0, y_c - int(0.5 * (delta_y + th)))
    y_max = max(0, min(h - th, y_c + int(0.5 * (delta_y - th))))

    if x_min > x_max:
        x1 = random.randint(0, x_max)
    else:
        x1 = random.randint(x_min, x_max)
    if y_min > y_max:
        y1 = random.randint(0, y_max)
    else:
        y1 = random.randint(y_min, y_max)

    data[0].crop(x1, y1, tw, th)
    for i, mode in enumerate(data[1:]):
        if mode is not None:
            data[i + 1].data = data[i + 1].data[y1:y1+th, x1:x1+tw]

    return data


def pad_reflection(image, top, bottom, left, right):
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return image
    h, w = image.shape[:2]
    next_top = next_bottom = next_left = next_right = 0
    if top > h - 1:
        next_top = top - h + 1
        top = h - 1
    if bottom > h - 1:
        next_bottom = bottom - h + 1
        bottom = h - 1
    if left > w - 1:
        next_left = left - w + 1
        left = w - 1
    if right > w - 1:
        next_right = right - w + 1
        right = w - 1
    new_shape = list(image.shape)
    new_shape[0] += top + bottom
    new_shape[1] += left + right
    new_image = np.empty(new_shape, dtype=image.dtype)
    new_image[top:top+h, left:left+w] = image
    new_image[:top, left:left+w] = image[top:0:-1, :]
    new_image[top+h:, left:left+w] = image[-1:-bottom-1:-1, :]
    new_image[:, :left] = new_image[:, left*2:left:-1]
    new_image[:, left+w:] = new_image[:, -right-1:-right*2-1:-1]
    return pad_reflection(new_image, next_top, next_bottom,
                          next_left, next_right)


def pad_constant(image, top, bottom, left, right, value):
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return image

    h, w = image.shape[:2]
    new_shape = list(image.shape)
    new_shape[0] += top + bottom
    new_shape[1] += left + right
    new_image = np.empty(new_shape, dtype=image.dtype)
    new_image.fill(value)
    new_image[top:top + h, left:left + w] = image

    return new_image


def pad_image(mode, image, top, bottom, left, right, value=0):
    if mode == 'reflection':
        if type(image) == np.ndarray:
            return pad_reflection(np.asarray(image), top, bottom, left, right)
        else:
            return Image.fromarray(
                pad_reflection(np.asarray(image), top, bottom, left, right))
    elif mode == 'constant':
        if type(image) == np.ndarray:
            return pad_constant(np.asarray(image), top, bottom, left, right, value)
        else:
            return Image.fromarray(
                pad_constant(np.asarray(image), top, bottom, left, right, value))
    else:
        raise ValueError('Unknown mode {}'.format(mode))

def get_random_bbox(data, tw, th):
    top = bottom = left = right = 0
    w, h = data[0].data.size

    if w < tw:
        left = (tw - w) // 2
        right = tw - w - left
    if h < th:
        top = (th - h) // 2
        bottom = th - h - top

    if left > 0 or right > 0 or top > 0 or bottom > 0:
        data[0].data = pad_image('reflection', data[0].data, top, bottom, left, right)
        for i, mode in enumerate(data[1:]):
            data[i + 1].data = pad_image('constant', data[i + 1].data, top, bottom, left, right, value=0)

    w, h = data[0].data.size
    if w == tw and h == th:
        # should happen after above when image is smaller than crop size
        return (0, 0, w, h)

    # crop next to objects
    [y_mask, x_mask] = np.where(data[1].data == 1)

    right_bb = np.max(x_mask)
    left_bb = np.min(x_mask)
    top_bb = np.min(y_mask)
    bottom_bb = np.max(y_mask)

    x_c = int(0.5 * (right_bb + left_bb))
    y_c = int(0.5 * (bottom_bb + top_bb))

    delta_x = np.max(x_mask) - np.min(x_mask)
    delta_y = np.max(y_mask) - np.min(y_mask)

    x_min = max(0, x_c - int(0.5 * (delta_x + tw)))
    x_max = max(0, min(w - tw, x_c + int(0.5 * (delta_x - tw))))
    y_min = max(0, y_c - int(0.5 * (delta_y + th)))
    y_max = max(0, min(h - th, y_c + int(0.5 * (delta_y - th))))

    if x_min > x_max:
        x1 = random.randint(0, x_max)
    else:
        x1 = random.randint(x_min, x_max)
    if y_min > y_max:
        y1 = random.randint(0, y_max)
    else:
        y1 = random.randint(y_min, y_max)

    return (x1, y1, tw, th)


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
