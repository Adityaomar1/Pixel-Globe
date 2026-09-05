import random
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter, ImageEnhance


class JointCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, t1, t2, mask):
        for t in self.transforms:
            t1, t2, mask = t(t1, t2, mask)
        return t1, t2, mask


class JointRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, t1, t2, mask):
        if random.random() < self.p:
            t1 = TF.hflip(t1)
            t2 = TF.hflip(t2)
            mask = TF.hflip(mask)
        return t1, t2, mask


class JointRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, t1, t2, mask):
        if random.random() < self.p:
            t1 = TF.vflip(t1)
            t2 = TF.vflip(t2)
            mask = TF.vflip(mask)
        return t1, t2, mask


class JointRandomRotation90:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, t1, t2, mask):
        if random.random() < self.p:
            angle = random.choice([90, 180, 270])
            t1 = TF.rotate(t1, angle)
            t2 = TF.rotate(t2, angle)
            mask = TF.rotate(mask, angle)
        return t1, t2, mask


class JointResize:
    def __init__(self, size=(256, 256)):
        self.size = size

    def __call__(self, t1, t2, mask):
        t1 = TF.resize(t1, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        t2 = TF.resize(t2, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)
        return t1, t2, mask


class PhotometricAugmentation:
    """Simulates varying illumination and atmospheric changes between T1 and T2."""
    def __init__(self, p=0.4):
        self.p = p

    def __call__(self, t1, t2, mask):
        if random.random() < self.p:
            # Random brightness / contrast jitter
            factor_b = random.uniform(0.85, 1.15)
            factor_c = random.uniform(0.85, 1.15)
            
            if random.random() < 0.5:
                t1 = TF.adjust_brightness(t1, factor_b)
                t1 = TF.adjust_contrast(t1, factor_c)
            if random.random() < 0.5:
                t2 = TF.adjust_brightness(t2, factor_b)
                t2 = TF.adjust_contrast(t2, factor_c)
                
        return t1, t2, mask


class JointToTensorNormalize:
    """Converts PIL images to PyTorch tensors and normalizes RGB using ImageNet statistics."""
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, t1, t2, mask):
        t1 = TF.to_tensor(t1)
        t2 = TF.to_tensor(t2)
        t1 = TF.normalize(t1, mean=self.mean, std=self.std)
        t2 = TF.normalize(t2, mean=self.mean, std=self.std)

        # Process mask: ensure shape is (1, H, W) and values are binary {0, 1}
        mask_tensor = TF.to_tensor(mask)
        # If mask has 3 channels, take the first channel
        if mask_tensor.shape[0] > 1:
            mask_tensor = mask_tensor[0:1, :, :]
        # Threshold: any pixel > 0.5 (or > 127 in uint8) is 1.0, otherwise 0.0
        mask_tensor = (mask_tensor > 0.5).float()

        return t1, t2, mask_tensor


def get_transforms(mode='train', img_size=(256, 256)):
    """Returns transform pipelines for training, validation, or testing."""
    if mode == 'train':
        return JointCompose([
            JointResize(img_size),
            JointRandomHorizontalFlip(p=0.5),
            JointRandomVerticalFlip(p=0.5),
            JointRandomRotation90(p=0.5),
            PhotometricAugmentation(p=0.4),
            JointToTensorNormalize()
        ])
    else:
        return JointCompose([
            JointResize(img_size),
            JointToTensorNormalize()
        ])
