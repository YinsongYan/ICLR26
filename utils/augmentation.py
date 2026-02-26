import collections
import math
import numbers
import random

import numpy as np
import torch
import torchtoolbox.transform
import torchvision
import torchvision.transforms.functional as F
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms.functional import pad, crop, hflip




# class ToTensor:
#     """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor. This transform does not support torchscript.
#
#     Converts a PIL Image or numpy.ndarray (H x W x C) in the range
#     [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
#     if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
#     or if the numpy.ndarray has dtype = np.uint8
#
#     In the other cases, tensors are returned without scaling.
#
#     .. note::
#         Because the input image is scaled to [0.0, 1.0], this transformation should not be used when
#         transforming target image masks. See the `references`_ for implementing the transforms for image masks.
#
#     .. _references: https://github.com/pytorch/vision/tree/main/references/segmentation
#     """
#
#     def __call__(self, pic):
#         return torch.from_numpy(pic).float()
#
#     def __repr__(self):
#         return self.__class__.__name__ + '()'




class Padding:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, imgmap):
        return [ImageOps.expand(img, border=self.pad, fill=0) for img in imgmap]


class Scale:
    def __init__(self, size, interpolation=Image.NEAREST):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgmap):
        # assert len(imgmap) > 1 # list of images
        img1 = imgmap[0]
        if isinstance(self.size, int):
            w, h = img1.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return imgmap
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return [i.resize((ow, oh), self.interpolation) for i in imgmap]
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return [i.resize((ow, oh), self.interpolation) for i in imgmap]
        else:
            return [i.resize(self.size, self.interpolation) for i in imgmap]


class CenterCrop:
    def __init__(self, size, consistent=True):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgmap):
        img1 = imgmap[0]
        w, h = img1.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return [i.crop((x1, y1, x1 + tw, y1 + th)) for i in imgmap]


class RandomCropWithProb:
    def __init__(self, size, p=0.8, consistent=True):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.consistent = consistent
        self.threshold = p

    def __call__(self, imgmap):
        img1 = imgmap[0]
        w, h = img1.size
        if self.size is not None:
            th, tw = self.size
            if w == tw and h == th:
                return imgmap
            if self.consistent:
                if random.random() < self.threshold:
                    x1 = random.randint(0, w - tw)
                    y1 = random.randint(0, h - th)
                else:
                    x1 = int(round((w - tw) / 2.))
                    y1 = int(round((h - th) / 2.))
                return [i.crop((x1, y1, x1 + tw, y1 + th)) for i in imgmap]
            else:
                result = []
                for i in imgmap:
                    if random.random() < self.threshold:
                        x1 = random.randint(0, w - tw)
                        y1 = random.randint(0, h - th)
                    else:
                        x1 = int(round((w - tw) / 2.))
                        y1 = int(round((h - th) / 2.))
                    result.append(i.crop((x1, y1, x1 + tw, y1 + th)))
                return result
        else:
            return imgmap


class RandomCrop:
    def __init__(self, size, consistent=True):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.consistent = consistent

    def __call__(self, imgmap, flowmap=None):
        img1 = imgmap[0]
        w, h = img1.size
        if self.size is not None:
            th, tw = self.size
            if w == tw and h == th:
                return imgmap
            if not flowmap:
                if self.consistent:
                    x1 = random.randint(0, w - tw)
                    y1 = random.randint(0, h - th)
                    return [i.crop((x1, y1, x1 + tw, y1 + th)) for i in imgmap]
                else:
                    result = []
                    for i in imgmap:
                        x1 = random.randint(0, w - tw)
                        y1 = random.randint(0, h - th)
                        result.append(i.crop((x1, y1, x1 + tw, y1 + th)))
                    return result
            elif flowmap is not None:
                assert (not self.consistent)
                result = []
                for idx, i in enumerate(imgmap):
                    proposal = []
                    for j in range(3):  # number of proposal: use the one with largest optical flow
                        x = random.randint(0, w - tw)
                        y = random.randint(0, h - th)
                        proposal.append([x, y, abs(np.mean(flowmap[idx, y:y + th, x:x + tw, :]))])
                    [x1, y1, _] = max(proposal, key=lambda x: x[-1])
                    result.append(i.crop((x1, y1, x1 + tw, y1 + th)))
                return result
            else:
                raise ValueError('wrong case')
        else:
            return imgmap


class RandomSizedCrop:
    def __init__(self, size, interpolation=Image.BILINEAR, consistent=True, p=1.0):
        self.size = size
        self.interpolation = interpolation
        self.consistent = consistent
        self.threshold = p

    def __call__(self, imgmap):
        img1 = imgmap[0]
        if random.random() < self.threshold:  # do RandomSizedCrop
            for attempt in range(10):
                area = img1.size[0] * img1.size[1]
                target_area = random.uniform(0.5, 1) * area
                aspect_ratio = random.uniform(3. / 4, 4. / 3)

                w = int(round(math.sqrt(target_area * aspect_ratio)))
                h = int(round(math.sqrt(target_area / aspect_ratio)))

                if self.consistent:
                    if random.random() < 0.5:
                        w, h = h, w
                    if w <= img1.size[0] and h <= img1.size[1]:
                        x1 = random.randint(0, img1.size[0] - w)
                        y1 = random.randint(0, img1.size[1] - h)

                        imgmap = [i.crop((x1, y1, x1 + w, y1 + h)) for i in imgmap]
                        for i in imgmap: assert (i.size == (w, h))

                        return [i.resize((self.size, self.size), self.interpolation) for i in imgmap]
                else:
                    result = []
                    for i in imgmap:
                        if random.random() < 0.5:
                            w, h = h, w
                        if w <= img1.size[0] and h <= img1.size[1]:
                            x1 = random.randint(0, img1.size[0] - w)
                            y1 = random.randint(0, img1.size[1] - h)
                            result.append(i.crop((x1, y1, x1 + w, y1 + h)))
                            assert (result[-1].size == (w, h))
                        else:
                            result.append(i)

                    assert len(result) == len(imgmap)
                    return [i.resize((self.size, self.size), self.interpolation) for i in result]

                    # Fallback
            scale = Scale(self.size, interpolation=self.interpolation)
            crop = CenterCrop(self.size)
            return crop(scale(imgmap))
        else:  # don't do RandomSizedCrop, do CenterCrop
            crop = CenterCrop(self.size)
            return crop(imgmap)


class RandomHorizontalFlip:
    def __init__(self, consistent=True, command=None):
        self.consistent = consistent
        if command == 'left':
            self.threshold = 0
        elif command == 'right':
            self.threshold = 1
        else:
            self.threshold = 0.5

    def __call__(self, imgmap):
        if self.consistent:
            if random.random() < self.threshold:
                return [i.transpose(Image.FLIP_LEFT_RIGHT) for i in imgmap]
            else:
                return imgmap
        else:
            result = []
            for i in imgmap:
                if random.random() < self.threshold:
                    result.append(i.transpose(Image.FLIP_LEFT_RIGHT))
                else:
                    result.append(i)
            assert len(result) == len(imgmap)
            return result



# class RandomFlipAndShift:
#     def __init__(self, max_shift=5):
#         """
#         Args:
#             max_shift (int): Maximum number of pixels to shift in both height and width.
#         """
#         self.max_shift = max_shift
# 
#     def __call__(self, data):
#         """
#         Args:
#             data (torch.Tensor or list): Input data of shape [T, C, H, W] or a list of frames.
#         Returns:
#             torch.Tensor: Transformed data with random horizontal flip and spatial shift.
#         """
#         # If data is a list, process each frame individually
#         if isinstance(data, list):
#             data = torch.stack(data, dim=0)  # Convert list of frames to a tensor: [T, C, H, W]
# 
#         # Random horizontal flip
#         if random.random() > 0.5:
#             data = torch.flip(data, dims=(3,))  # Flip along the width (dim 3)
# 
#         # Random spatial shift
#         off1 = random.randint(-self.max_shift, self.max_shift)  # Vertical shift
#         off2 = random.randint(-self.max_shift, self.max_shift)  # Horizontal shift
#         data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))  # Apply roll shift
# 
#         return data


class RandomFlipAndShift:
    def __init__(self, max_shift=5, flip_prob=0.5):
        """
        Randomly flips the image horizontally and shifts it within the frame.

        Args:
            max_shift (int): Maximum number of pixels to shift in any direction.
            flip_prob (float): Probability of performing a horizontal flip.
        """
        self.max_shift = max_shift
        self.flip_prob = flip_prob

    def __call__(self, imgmap):
        """
        Randomly flip and shift a sequence of image frames.

        Args:
            imgmap (Tensor or List[Tensor]): A sequence of image frames (C, H, W) or a list of PIL images.

        Returns:
            Augmented imgmap with random flip and shift applied.
        """
        # Random horizontal flipping
        if random.random() < self.flip_prob:
            if isinstance(imgmap, torch.Tensor):  # Handle tensors
                imgmap = torch.flip(imgmap, dims=[-1])  # Flip along the width
            else:  # Handle lists of PIL images
                imgmap = [i.transpose(Image.FLIP_LEFT_RIGHT) for i in imgmap]

        # Random shifting
        shift_x = random.randint(-self.max_shift, self.max_shift)
        shift_y = random.randint(-self.max_shift, self.max_shift)

        if isinstance(imgmap, torch.Tensor):
            # Assume imgmap is a tensor of shape (C, H, W) or (N, C, H, W)
            if imgmap.ndim == 3:  # Single image (C, H, W)
                imgmap = pad(imgmap, (self.max_shift, self.max_shift), fill=0)
                imgmap = crop(imgmap, self.max_shift + shift_y, self.max_shift + shift_x, imgmap.size(-2), imgmap.size(-1))
            elif imgmap.ndim == 4:  # Batch of images (N, C, H, W)
                imgmap = torch.stack([
                    crop(
                        pad(img, (self.max_shift, self.max_shift), fill=0),
                        self.max_shift + shift_y, self.max_shift + shift_x, img.size(-2), img.size(-1)
                    ) for img in imgmap
                ])
        else:
            # Handle lists of PIL images
            imgmap = [
                crop(
                    pad(i, (self.max_shift, self.max_shift), fill=0),
                    self.max_shift + shift_y, self.max_shift + shift_x, i.size[1], i.size[0]
                ) for i in imgmap
            ]

        return imgmap



class RandomGray:
    '''Actually it is a channel splitting, not strictly grayscale images'''

    def __init__(self, consistent=True, p=0.5):
        self.consistent = consistent
        self.p = p  # probability to apply grayscale

    def __call__(self, imgmap):
        if self.consistent:
            if random.random() < self.p:
                return [self.grayscale(i) for i in imgmap]
            else:
                return imgmap
        else:
            result = []
            for i in imgmap:
                if random.random() < self.p:
                    result.append(self.grayscale(i))
                else:
                    result.append(i)
            assert len(result) == len(imgmap)
            return result

    def grayscale(self, img):
        channel = np.random.choice(3)
        np_img = np.array(img)[:, :, channel]
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, 'RGB')
        return img


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image. --modified from pytorch source code
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, consistent=False, p=1.0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        self.consistent = consistent
        self.threshold = p

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(torchvision.transforms.Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = torchvision.transforms.Compose(transforms)

        return transform

    def __call__(self, imgmap):
        if random.random() < self.threshold:  # do ColorJitter
            if self.consistent:
                transform = self.get_params(self.brightness, self.contrast,
                                            self.saturation, self.hue)
                return [transform(i) for i in imgmap]
            else:
                result = []
                for img in imgmap:
                    transform = self.get_params(self.brightness, self.contrast,
                                                self.saturation, self.hue)
                    result.append(transform(img))
                return result
        else:  # don't do ColorJitter, do nothing
            return imgmap

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


class RandomRotation:
    def __init__(self, consistent=True, degree=15, p=1.0):
        self.consistent = consistent
        self.degree = degree
        self.threshold = p

    def __call__(self, imgmap):
        if random.random() < self.threshold:  # do RandomRotation
            if self.consistent:
                deg = np.random.randint(-self.degree, self.degree, 1)[0]
                return [i.rotate(deg, expand=True) for i in imgmap]
            else:
                return [i.rotate(np.random.randint(-self.degree, self.degree, 1)[0], expand=True) for i in imgmap]
        else:  # don't do RandomRotation, do nothing
            return imgmap


class ToTensor:
    def __call__(self, imgmap):
        totensor = transforms.ToTensor()
        return [totensor(i) for i in imgmap]


class ToPILImage:
    def __call__(self, imgmap):
        topilimage = transforms.ToPILImage()
        return [topilimage(i) for i in imgmap]


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, imgmap):
        resize = transforms.Resize(self.size)
        return [resize(i) for i in imgmap]


class Cutout:
    def __call__(self, imgmap):
        if random.random() < 0.5:
            cutout = torchtoolbox.transform.Cutout(p=1)
            return [cutout(i) for i in imgmap]
        else:
            return imgmap


class Normalize:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, imgmap):
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        return [normalize(i) for i in imgmap]


class Roll:
    def __init__(self):
        self.off1 = random.randint(-5, 5)
        self.off2 = random.randint(-5, 5)

    def __call__(self, imgmap):
        return [torch.roll(i, shifts=(self.off1, self.off2), dims=(1, 2)) for i in imgmap]


if __name__ == '__main__':
    from utils.cifar10_dvs import CIFAR10DVS

    c_in = 2
    num_classes = 10

    transform_train = transforms.Compose([
        # CIFAR10_DVS_Aug(), # it has been resize 48
        ToPILImage(),
        Resize(48),
        RandomSizedCrop(48),
        RandomHorizontalFlip(),
        ToTensor(),
        # Roll(),
    ])

    transform_test = transforms.Compose([
        ToPILImage(),
        Resize(48),
        ToTensor(),
    ])
    data_dir = "./data_dir"
    trainset = CIFAR10DVS(data_dir, train=True, use_frame=True, frames_num=10, split_by='number',
                          normalization=None, transform=transform_train)
    testset = CIFAR10DVS(data_dir, train=False, use_frame=True, frames_num=10, split_by='number',
                         normalization=None, transform=transform_test)

    # train_data_loader = data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
    # test_data_loader = data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)
    print(trainset[0])
