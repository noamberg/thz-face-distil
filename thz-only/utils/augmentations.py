#!/usr/bin/env python3
"""
Custom Augmentations for THz Imaging

THz-specific augmentations designed for face verification, posture classification,
and image reconstruction tasks.
"""

import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import random


class StripeMasking:
    """
    Apply horizontal or vertical stripe masking to simulate THz artifacts.

    Args:
        stripe_width: Width of stripes in pixels (default: 2-4)
        prob: Probability of applying stripe masking
        max_stripes: Maximum number of stripes to apply
    """

    def __init__(self, stripe_width=(2, 4), prob=0.2, max_stripes=3):
        self.stripe_width = stripe_width
        self.prob = prob
        self.max_stripes = max_stripes

    def __call__(self, img):
        if random.random() > self.prob:
            return img

        # Convert to numpy if PIL Image
        is_pil = isinstance(img, Image.Image)
        if is_pil:
            img_array = np.array(img).astype(np.float32) / 255.0
        else:
            img_array = np.array(img)

        h, w = img_array.shape
        num_stripes = random.randint(1, self.max_stripes)

        for _ in range(num_stripes):
            # Random stripe width
            width = random.randint(*self.stripe_width)

            # Randomly choose horizontal or vertical
            if random.random() < 0.5:
                # Horizontal stripe
                y = random.randint(0, h - width)
                img_array[y:y+width, :] = 0
            else:
                # Vertical stripe
                x = random.randint(0, w - width)
                img_array[:, x:x+width] = 0

        if is_pil:
            img_array = (img_array * 255).astype(np.uint8)
            return Image.fromarray(img_array)
        return img_array


class SpeckleNoise:
    """
    Add speckle noise to simulate THz sensor noise.

    Args:
        std: Standard deviation of Gaussian noise (default: 0.05-0.1)
        prob: Probability of applying noise (default: 1.0 for always)
    """

    def __init__(self, std=(0.05, 0.1), prob=1.0):
        self.std = std
        self.prob = prob

    def __call__(self, img):
        if random.random() > self.prob:
            return img

        # Convert to numpy if PIL Image
        is_pil = isinstance(img, Image.Image)
        if is_pil:
            img_array = np.array(img).astype(np.float32) / 255.0
        else:
            img_array = np.array(img)

        # Sample noise std
        noise_std = random.uniform(*self.std)

        # Add Gaussian noise
        noise = np.random.normal(0, noise_std, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 1)

        if is_pil:
            img_array = (img_array * 255).astype(np.uint8)
            return Image.fromarray(img_array)
        return img_array


class RandomCutout:
    """
    Apply random cutout patches to the image.

    Args:
        num_patches: Number of cutout patches (tuple of min, max)
        patch_size: Size of each patch in pixels (tuple of min, max)
        prob: Probability of applying cutout
    """

    def __init__(self, num_patches=(2, 4), patch_size=(8, 16), prob=0.3):
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.prob = prob

    def __call__(self, img):
        if random.random() > self.prob:
            return img

        # Convert to numpy if PIL Image
        is_pil = isinstance(img, Image.Image)
        if is_pil:
            img_array = np.array(img).astype(np.float32) / 255.0
        else:
            img_array = np.array(img)

        h, w = img_array.shape
        num_patches = random.randint(*self.num_patches)

        for _ in range(num_patches):
            # Random patch size
            patch_h = random.randint(*self.patch_size)
            patch_w = random.randint(*self.patch_size)

            # Random position
            y = random.randint(0, max(0, h - patch_h))
            x = random.randint(0, max(0, w - patch_w))

            # Apply cutout (set to 0)
            img_array[y:y+patch_h, x:x+patch_w] = 0

        if is_pil:
            img_array = (img_array * 255).astype(np.uint8)
            return Image.fromarray(img_array)
        return img_array


class BandDrop:
    """
    Drop horizontal or vertical bands to simulate THz scanning artifacts.

    Args:
        band_width: Width of band to drop (tuple of min, max)
        prob: Probability of applying band drop
    """

    def __init__(self, band_width=(4, 12), prob=0.3):
        self.band_width = band_width
        self.prob = prob

    def __call__(self, img):
        if random.random() > self.prob:
            return img

        # Convert to numpy if PIL Image
        is_pil = isinstance(img, Image.Image)
        if is_pil:
            img_array = np.array(img).astype(np.float32) / 255.0
        else:
            img_array = np.array(img)

        h, w = img_array.shape
        width = random.randint(*self.band_width)

        # Randomly choose horizontal or vertical
        if random.random() < 0.5:
            # Horizontal band
            y = random.randint(0, max(0, h - width))
            img_array[y:y+width, :] = 0
        else:
            # Vertical band
            x = random.randint(0, max(0, w - width))
            img_array[:, x:x+width] = 0

        if is_pil:
            img_array = (img_array * 255).astype(np.uint8)
            return Image.fromarray(img_array)
        return img_array


class RandomResizedCropGrayscale:
    """
    Random crop and resize for grayscale images.

    Args:
        size: Output size (int or tuple)
        scale: Range of size of the origin size cropped
        ratio: Range of aspect ratio of the origin aspect ratio cropped
    """

    def __init__(self, size, scale=(0.9, 1.0), ratio=(0.95, 1.05)):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img):
        # Use torchvision's RandomResizedCrop logic
        width, height = img.size
        area = width * height

        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)

            w = int(round(np.sqrt(target_area * aspect_ratio)))
            h = int(round(np.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                img = TF.crop(img, i, j, h, w)
                return img.resize(self.size, Image.BILINEAR)

        # Fallback to center crop
        in_ratio = width / height
        if in_ratio < min(self.ratio):
            w = width
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = height
            w = int(round(h * max(self.ratio)))
        else:
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        img = TF.crop(img, i, j, h, w)
        return img.resize(self.size, Image.BILINEAR)


class RandomRotationWithScale:
    """
    Combined random rotation and scaling.

    Args:
        degrees: Range of degrees for rotation
        scale: Range of scaling factor
        prob: Probability of applying this transform
    """

    def __init__(self, degrees=10, scale=(0.9, 1.1), prob=1.0):
        self.degrees = degrees
        self.scale = scale
        self.prob = prob

    def __call__(self, img):
        if random.random() > self.prob:
            return img

        # Random rotation angle
        angle = random.uniform(-self.degrees, self.degrees)

        # Random scale factor
        scale = random.uniform(*self.scale)

        # Apply rotation
        img = TF.rotate(img, angle, interpolation=Image.BILINEAR, fill=0)

        # Apply scaling by resizing then cropping/padding
        w, h = img.size
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)

        # Center crop or pad to original size
        if scale > 1.0:
            # Crop to original size
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            img = TF.crop(img, top, left, h, w)
        else:
            # Pad to original size
            padding = ((w - new_w) // 2, (h - new_h) // 2,
                      (w - new_w + 1) // 2, (h - new_h + 1) // 2)
            img = TF.pad(img, padding, fill=0)

        return img


class RandomTranslation:
    """
    Random translation (shift) of the image.

    Args:
        translate: Max translation in pixels (int) or fraction (float < 1.0)
        prob: Probability of applying translation
    """

    def __init__(self, translate=5, prob=1.0):
        self.translate = translate
        self.prob = prob

    def __call__(self, img):
        if random.random() > self.prob:
            return img

        w, h = img.size

        # Calculate max translation
        if isinstance(self.translate, float) and self.translate < 1.0:
            max_dx = int(self.translate * w)
            max_dy = int(self.translate * h)
        else:
            max_dx = max_dy = int(self.translate)

        # Random translation
        dx = random.randint(-max_dx, max_dx)
        dy = random.randint(-max_dy, max_dy)

        # Apply translation using affine
        img = TF.affine(img, angle=0, translate=(dx, dy), scale=1.0, shear=0, fill=0)

        return img
