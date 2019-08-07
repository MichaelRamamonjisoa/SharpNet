import numpy as np
from scipy.ndimage import rotate as scp_rotate
import cv2
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from skimage.morphology import skeletonize


class Representation(object):
    """
    Intermediate representation object
    """

    def __init__(self, data=None, name=None):
        self.data = data
        self.name = name

    def set_data(self, data):
        self.data = data

    def shape(self):
        return (self.data).shape

    def rotate(self, angle, cval=0):
        self.data = scp_rotate(self.data, angle, reshape=False, order=0, mode='wrap', prefilter=False)

    def scale(self, ratio, interpolation='NEAREST'):
        h, w = self.data.shape[:2]
        tw = int(ratio * w)
        th = int(ratio * h)

        if interpolation == 'NEAREST':
            interpolation = cv2.INTER_NEAREST
        else:
            if ratio < 1:
                interpolation = cv2.INTER_LINEAR
            else:
                interpolation = cv2.INTER_CUBIC

        self.data = cv2.resize(self.data, dsize=(tw, th), interpolation=interpolation)

    def crop(self, x1, y1, tw, th):
        self.data = self.data[y1:y1 + th, x1:x1 + tw]

    def fliplr(self):
        self.data = np.fliplr(self.data)

    def to_tensor(self):
        self.data = torch.LongTensor(np.array(self.data, dtype=np.int))

    def normalize(self):
        return 1


class InputImage(Representation):
    """
    Image class
    """

    def __init__(self, data):
        super(InputImage, self).__init__(data=data, name='Image')
        # self.norm_mean = mean
        # self.norm_std = std

    def to_tensor(self):
        if isinstance(self.data, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(self.data)
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(self.data.tobytes()))
            # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
            if self.data.mode == 'YCbCr':
                nchannel = 3
            else:
                nchannel = len(self.data.mode)
            img = img.view(self.data.size[1], self.data.size[0], nchannel)
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255)

        self.set_data(img)

    def shape(self):
        return (self.data).size

    def rotate(self, angle, cval=0):
        tmp = self.data.copy()
        tmp = np.array(tmp)
        tmp = scp_rotate(tmp, angle, reshape=False, order=0, mode='constant', cval=cval, prefilter=False)
        self.data = Image.fromarray(tmp)

    def scale(self, ratio):
        w, h = self.shape()
        tw = int(ratio * w)
        th = int(ratio * h)

        if ratio < 1:
            interpolation = Image.ANTIALIAS
        else:
            interpolation = Image.CUBIC

        self.data = (self.data).resize((tw, th), interpolation)

    def fliplr(self):
        self.data = (self.data).transpose(Image.FLIP_LEFT_RIGHT)

    def crop(self, x1, y1, tw, th):
        self.data = self.data.crop((x1, y1, x1 + tw, y1 + th))

    def gamma(self, gamma_ratio):
        self.data = TF.adjust_gamma(self.data, gamma_ratio, gain=1)

    def normalize(self, mean, std):
        mean = torch.FloatTensor(mean)
        std = torch.FloatTensor(std)

        image = self.data
        
        if image.device.type != 'cpu':
            means = [mean] * image.size()[0]
            stds = [std] * image.size()[0]
            for t, m, s in zip(image, means, stds):
                t.sub_(m[:, None, None].cuda()).div_(s[:, None, None].cuda())
        else:
            for t, m, s in zip(image, mean, std):
                t.sub_(m).div_(s)

        self.set_data(image)

        return 1


class Normals(Representation):
    """
    Normals: overwrite transforms to handle specificity of normals transforms
    """

    def __init__(self, data):
        super(Normals, self).__init__(data=data, name='normals')
        # normalize normals
        n = np.linalg.norm(self.data, 2, axis=2)
        self.data = self.data / (np.expand_dims(n, axis=2).clip(1e-4))

    def scale(self, ratio):
        # transform normals
        super(Normals, self).scale(ratio, interpolation='NEAREST')
        self.data[..., 2] *= ratio
        norm = np.linalg.norm(self.data, 2, axis=2)
        self.data = self.data / (np.expand_dims(norm, axis=2).clip(1e-4))

    def rotate(self, angle, cval=0):
        # rotating around Z axis does not affect Z normal
        rad_angle = np.deg2rad(angle)
        cos_angle = np.cos(rad_angle)
        sin_angle = np.sin(rad_angle)
        self.data[..., 0] = self.data[..., 0] * cos_angle - self.data[..., 1] * sin_angle
        self.data[..., 1] = self.data[..., 0] * sin_angle + self.data[..., 1] * cos_angle

        # normals
        # self.data = scp_rotate(self.data, angle, reshape=False, order=0, mode='constant', cval=cval, prefilter=False)
        self.data = scp_rotate(self.data, angle, reshape=False, order=0, mode='wrap', prefilter=False)

    def crop(self, x1, y1, tw, th):
        self.data = self.data[y1:y1 + th, x1:x1 + tw, :]

    def fliplr(self):
        self.data = np.fliplr(self.data)
        self.data[..., 0] = -1.0 * self.data[..., 0]

    def to_tensor(self):
        self.data = torch.FloatTensor(np.array((self.data).swapaxes(1, 2).swapaxes(0, 1), dtype=np.float32))


class Depth(Representation):
    """
    Depth: overwrite scale
    """

    def __init__(self, data):
        super(Depth, self).__init__(data=data, name='depth')

    def scale(self, ratio):
        super(Depth, self).scale(ratio, interpolation='NEAREST')
        self.data = self.data / ratio

    def to_tensor(self):
        self.data = torch.FloatTensor(np.array(self.data, dtype=np.float32))


class Contours(Representation):
    """
    Contours: overwrite scale to always have contours with 1 pixel width
    """

    def __init__(self, data):
        super(Contours, self).__init__(data=data, name='contours')

    def scale(self, ratio, interpolation='LINEAR'):
        h, w = self.data.shape[:2]
        tw = int(ratio * w)
        th = int(ratio * h)

        # solve the missed edges
        if ratio > 1:
            im = cv2.resize(self.data, dsize=(tw, th), interpolation=cv2.INTER_LINEAR_EXACT)
            im[im > 0.2] = 1
            im = skeletonize(im)
        else:
            im = cv2.resize(self.data, dsize=(tw, th), interpolation=cv2.INTER_LINEAR_EXACT)
            im[im > 0.4] = 1
            im = skeletonize(im)
        self.data = im.copy()


class Mask(Representation):
    """
    Mask:
    """

    def __init__(self, data):
        super(Mask, self).__init__(data=data, name='mask')

    def rotate(self, angle, cval=0):
        self.data = scp_rotate(self.data, angle, reshape=False, order=0, mode='constant', cval=cval, prefilter=False)

