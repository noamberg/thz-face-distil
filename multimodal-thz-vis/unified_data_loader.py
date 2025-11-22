#!/usr/bin/env python3
"""
Multimodal Triplets Data Loader for Privileged Information Learning
Extended to support visible-light images as privileged information during training

Supports:
- Face verification (triplet loss)
- Head posture classification (5 classes)
- Image reconstruction (concealed b#b -> unconcealed n#n)
- Visible images as privileged information (teacher modality)
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import random
from collections import defaultdict

from pathlib import Path
from utils.augmentations import (
    StripeMasking, SpeckleNoise, RandomCutout, BandDrop,
    RandomResizedCropGrayscale, RandomRotationWithScale, RandomTranslation
)

MIN_GROUP_SIZE_FOR_SPLIT = 3

def parse_filename(path_str):
    """Extract numeric identity, posture, and number from a file path."""
    try:
        path = Path(path_str)
        identity = path.parts[-3]
        stem_parts = path.stem.split('_')
        posture = stem_parts[1]
        number = stem_parts[2]
        return identity, posture, number
    except IndexError:
        return None, None, None

def get_posture_number(posture):
    """Extract posture number (0-4) from b#b or n#n format"""
    if len(posture) == 3 and posture[1].isdigit():
        return int(posture[1]) - 1
    return 0


class MultimodalTripletsDataset(Dataset):
    """
    Extended dataset for triplets-based multitask learning with visible modality support.
    Loads THz triplets + corresponding visible unconcealed image for privileged learning.
    """

    def __init__(self, csv_path=None, df=None, transform=None, target_transform=None,
                 input_size=224, data_dir=None, visible_data_dir=None):
        """
        Args:
            csv_path: Path to triplets CSV file.
            df: Pre-loaded DataFrame (alternative to csv_path).
            transform: Transform for THz input images (anchor, negative).
            target_transform: Transform for the target image (positive). If None, uses `transform`.
            input_size: Input image size.
            data_dir: Base directory for THz image files.
            visible_data_dir: Base directory for visible image files. If None, uses data_dir.
        """
        if df is not None:
            self.df = df.reset_index(drop=True)
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path).reset_index(drop=True)
        else:
            raise ValueError("Either csv_path or df must be provided")

        self.transform = transform
        self.target_transform = target_transform if target_transform is not None else transform
        self.input_size = input_size
        self.data_dir = data_dir
        self.visible_data_dir = visible_data_dir if visible_data_dir is not None else data_dir

        print(f"MultimodalTripletsDataset: {len(self.df)} samples, size {input_size}x{input_size}")
        if data_dir:
            print(f"THz image base directory: {data_dir}")
        if self.visible_data_dir:
            print(f"Visible image base directory: {self.visible_data_dir}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        try:
            # Construct full paths for THz images
            if self.data_dir:
                anchor_path = os.path.join(self.data_dir, row['anchor'])
                positive_path = os.path.join(self.data_dir, row['positive'])
                negative_path = os.path.join(self.data_dir, row['negative'])
            else:
                anchor_path = row['anchor']
                positive_path = row['positive']
                negative_path = row['negative']

            # Load THz triplet images as grayscale
            anchor_img = Image.open(anchor_path).convert('L')
            positive_img = Image.open(positive_path).convert('L')
            negative_img = Image.open(negative_path).convert('L')

            # Extract posture classes from all three filenames
            anchor_identity, anchor_posture, _ = parse_filename(row['anchor'])
            pos_identity, pos_posture, _ = parse_filename(row['positive'])
            neg_identity, neg_posture, _ = parse_filename(row['negative'])

            anchor_posture_class = get_posture_number(anchor_posture)
            positive_posture_class = get_posture_number(pos_posture)
            negative_posture_class = get_posture_number(neg_posture)

            # Apply THz transforms
            anchor_tensor = self.transform(anchor_img)
            positive_tensor = self.target_transform(positive_img)
            negative_tensor = self.transform(negative_img)

            # --- Load Visible Images (Privileged Information) ---
            def load_visible_image(csv_path):
                """Helper to load a visible image from CSV path."""
                if pd.notna(csv_path):
                    try:
                        # CSV now contains full paths, not relative
                        vis_path = csv_path if os.path.isabs(csv_path) else os.path.join(self.visible_data_dir, csv_path)
                        # Load visible image and resize to match THz input size
                        vis_img = Image.open(vis_path).convert('L')  # Convert to grayscale for consistency
                        # Resize visible image to match the THz input size before applying transforms
                        vis_img = vis_img.resize((self.input_size, self.input_size), Image.BILINEAR)
                        return self.transform(vis_img)  # Apply same transform as THz
                    except FileNotFoundError:
                        return None  # Handle missing files gracefully
                    except Exception as e:
                        print(f"Warning: Error loading visible image at {vis_path}: {e}")
                        return None
                return None

            # Load all three visible images
            vis_concealed_tensor = load_visible_image(row.get('visible_concealed', None))
            vis_unconcealed_tensor = load_visible_image(row.get('visible_unconcealed', None))
            vis_negative_tensor = load_visible_image(row.get('visible_negative', None))

            # Create ToTensor transform for originals
            to_tensor_transform = transforms.ToTensor()

            return {
                # THz triplets
                'anchor': anchor_tensor,
                'positive': positive_tensor,
                'negative': negative_tensor,

                # Posture classes
                'posture_class': anchor_posture_class,  # Keep for backward compatibility
                'anchor_posture_class': torch.tensor(anchor_posture_class, dtype=torch.long),
                'positive_posture_class': torch.tensor(positive_posture_class, dtype=torch.long),
                'negative_posture_class': torch.tensor(negative_posture_class, dtype=torch.long),

                # Visible modality (privileged information) - all three triplet images
                'visible_concealed': vis_concealed_tensor,  # Anchor visible (b1b, b2b, etc.)
                'visible_unconcealed': vis_unconcealed_tensor,  # Positive visible (n1n, n2n, etc.)
                'visible_negative': vis_negative_tensor,  # Negative visible (n1n, n2n, etc.)

                # Metadata
                'identity': anchor_identity,
                'posture': anchor_posture,
                'idx': idx,

                # Original (un-augmented) versions
                'original_anchor': to_tensor_transform(anchor_img),
                'original_positive': to_tensor_transform(positive_img),
                'original_negative': to_tensor_transform(negative_img)
            }

        except FileNotFoundError as e:
            raise RuntimeError(f"Image file not found at index {idx}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load sample at index {idx}: {e}")


def multimodal_collate_fn(batch):
    """
    Custom collate function to handle None values in visible_unconcealed field.
    When visible images are missing, replaces None with zero tensors of appropriate shape.
    """
    # Get a sample to determine tensor shapes
    sample_anchor = batch[0]['anchor']
    batch_size = len(batch)

    # Initialize the collated batch dictionary
    collated = {
        'anchor': torch.stack([item['anchor'] for item in batch]),
        'positive': torch.stack([item['positive'] for item in batch]),
        'negative': torch.stack([item['negative'] for item in batch]),
        'anchor_posture_class': torch.stack([item['anchor_posture_class'] for item in batch]),
        'positive_posture_class': torch.stack([item['positive_posture_class'] for item in batch]),
        'negative_posture_class': torch.stack([item['negative_posture_class'] for item in batch]),
        'posture_class': torch.stack([item['posture_class'] if isinstance(item['posture_class'], torch.Tensor)
                                      else torch.tensor(item['posture_class'], dtype=torch.long) for item in batch]),
        'original_anchor': torch.stack([item['original_anchor'] for item in batch]),
        'original_positive': torch.stack([item['original_positive'] for item in batch]),
        'original_negative': torch.stack([item['original_negative'] for item in batch]),
    }

    # Handle visible images which may contain None values
    def collate_visible_field(field_name):
        """Helper to collate a visible image field that may have None values."""
        visible_list = [item[field_name] for item in batch]
        if all(v is None for v in visible_list):
            # All visible images are missing - return None for the entire batch
            return None
        else:
            # Some or all visible images are available
            # Replace None with zero tensors matching the shape of available images
            non_none_visible = [v for v in visible_list if v is not None]
            if non_none_visible:
                reference_shape = non_none_visible[0].shape
                visible_tensors = []
                for v in visible_list:
                    if v is None:
                        # Create zero tensor with same shape as reference
                        visible_tensors.append(torch.zeros(reference_shape))
                    else:
                        visible_tensors.append(v)
                return torch.stack(visible_tensors)
            else:
                return None

    collated['visible_concealed'] = collate_visible_field('visible_concealed')
    collated['visible_unconcealed'] = collate_visible_field('visible_unconcealed')
    collated['visible_negative'] = collate_visible_field('visible_negative')

    # Handle metadata (identity, posture, idx) - keep as lists
    collated['identity'] = [item['identity'] for item in batch]
    collated['posture'] = [item['posture'] for item in batch]
    collated['idx'] = [item['idx'] for item in batch]

    return collated


def get_multimodal_triplets_data_loaders(train_csv=None, val_csv=None, test_csv=None, train_val_csv=None,
                                         input_size=224, batch_size=16, num_workers=4, val_ratio=0.15,
                                         random_seed=42, output_folder=None, use_custom_sampler=False,
                                         augmentation_mode='gentle', data_dir=None, visible_data_dir=None):
    """
    Create multimodal data loaders for triplets dataset with visible image support

    Args:
        train_csv: Path to train CSV file (new format)
        val_csv: Path to validation CSV file (new format)
        test_csv: Path to test CSV file (optional)
        train_val_csv: Path to train_val CSV file (legacy format)
        input_size: Input image size (64 for this project)
        batch_size: Batch size
        num_workers: Number of worker processes
        val_ratio: Validation split ratio (default 0.15)
        random_seed: Random seed for reproducible splits
        output_folder: Optional output folder to save split CSVs and statistics
        use_custom_sampler: Whether to use the custom balanced batch sampler
        augmentation_mode: 'gentle', 'medium', or 'strong'
        data_dir: Base directory for THz images
        visible_data_dir: Base directory for visible images

    Returns:
        train_loader, val_loader, test_loader (if test_csv provided)
    """

    # Handle both new format (separate train/val CSVs) and legacy format (single train_val CSV)
    if train_csv is not None and val_csv is not None:
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
    elif train_val_csv is not None:
        train_df, val_df = create_uniform_validation_split(
            train_val_csv, val_ratio, random_seed, output_folder
        )
    else:
        raise ValueError("Either (train_csv and val_csv) or train_val_csv must be provided")

    # --- Define Augmentation Pipelines ---
    # Target transform (minimal augmentation for reconstruction targets)
    target_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # GENTLE PROFILE
    # - Rotation [−10°, +10°] or scale [0.9,1.1]
    # - Random crop area [0.9,1.0] & resize
    # - Stripe masking (2–4 px width) with prob ~0.2
    # - Speckle noise ε~N(0, 0.05–0.1)
    gentle_train_transform = transforms.Compose([
        RandomRotationWithScale(degrees=20, scale=(0.8, 1.2), prob=1.0),
        RandomResizedCropGrayscale(size=input_size, scale=(0.8, 0.9)),
        StripeMasking(stripe_width=(4, 6), prob=0.5, max_stripes=3),
        SpeckleNoise(std=(0.05, 0.1), prob=1.0),
        transforms.ToTensor(),
    ])

    # MEDIUM PROFILE
    # - All of Gentle profile
    # - Rotation up to ±20°, random translation ±5 px
    # - Crop area [0.7,1.0] & resize
    # - Cutout: 2–4 patches of 8–16 px
    # - Strong speckle (ε~N(0,0.15–0.25)) or band-drop with prob ~0.3
    medium_train_transform = transforms.Compose([
        RandomRotationWithScale(degrees=20, scale=(0.9, 1.1), prob=1.0),
        RandomTranslation(translate=5, prob=1.0),
        RandomResizedCropGrayscale(size=input_size, scale=(0.7, 1.0)),
        StripeMasking(stripe_width=(2, 4), prob=0.2, max_stripes=3),
        RandomCutout(num_patches=(2, 4), patch_size=(8, 16), prob=0.3),
        BandDrop(band_width=(4, 12), prob=0.3),
        SpeckleNoise(std=(0.15, 0.25), prob=1.0),
        transforms.ToTensor(),
    ])

    # STRONG PROFILE (for verification/posture tasks)
    # - All of Medium profile
    # - More aggressive augmentations
    strong_train_transform = transforms.Compose([
        RandomRotationWithScale(degrees=20, scale=(0.85, 1.15), prob=1.0),
        RandomTranslation(translate=5, prob=1.0),
        RandomResizedCropGrayscale(size=input_size, scale=(0.7, 1.0)),
        StripeMasking(stripe_width=(2, 4), prob=0.3, max_stripes=4),
        RandomCutout(num_patches=(2, 4), patch_size=(8, 16), prob=0.4),
        BandDrop(band_width=(4, 12), prob=0.3),
        SpeckleNoise(std=(0.15, 0.25), prob=1.0),
        transforms.ToTensor(),
    ])

    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Select augmentation strategy
    if augmentation_mode == 'strong':
        print("Using STRONG augmentation pipeline for inputs.")
        train_transform = strong_train_transform
    elif augmentation_mode == 'medium':
        print("Using MEDIUM augmentation pipeline for inputs.")
        train_transform = medium_train_transform
    else:  # 'gentle' or default
        print("Using GENTLE augmentation pipeline for inputs.")
        train_transform = gentle_train_transform

    train_target_transform = target_transform

    # --- Create Datasets ---
    train_dataset = MultimodalTripletsDataset(
        df=train_df,
        transform=train_transform,
        target_transform=train_target_transform,
        input_size=input_size,
        data_dir=data_dir,
        visible_data_dir=visible_data_dir
    )

    val_dataset = MultimodalTripletsDataset(
        df=val_df,
        transform=val_transform,
        target_transform=val_transform,
        input_size=input_size,
        data_dir=data_dir,
        visible_data_dir=visible_data_dir
    )

    # --- Create Data Loaders ---
    if use_custom_sampler:
        print("Custom sampler not implemented for multimodal dataset. Using standard DataLoader.")
        use_custom_sampler = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=multimodal_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=multimodal_collate_fn
    )

    test_loader = None
    if test_csv is not None:
        test_dataset = MultimodalTripletsDataset(
            csv_path=test_csv,
            transform=val_transform,
            target_transform=val_transform,
            input_size=input_size,
            data_dir=data_dir,
            visible_data_dir=visible_data_dir
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=multimodal_collate_fn
        )

    print(f"Created multimodal data loaders:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Validation: {len(val_loader)} batches ({len(val_dataset)} samples)")
    if test_loader:
        print(f"  Test: {len(test_loader)} batches ({len(test_dataset)} samples)")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the multimodal data loader
    print("Testing MultimodalTripletsDataset...")

    # Example usage (you'll need to adjust paths for your setup)
    train_val_csv = "datafiles_fixed_29_08_sterile/faceData_cropped_64x64/dBm/train_val_dataset.csv"

    train_loader, val_loader, test_loader = get_multimodal_triplets_data_loaders(
        train_val_csv=train_val_csv,
        input_size=64,
        batch_size=8,
        val_ratio=0.15,
        data_dir='/home/noam/data_fixed/faceData_cropped_64x64',
        visible_data_dir='/home/noam/data_fixed/visible_images'  # Adjust to your visible data path
    )

    print("\nTesting data loading...")
    for i, batch in enumerate(train_loader):
        print(f"Batch {i}:")
        print(f"  Anchor shape: {batch['anchor'].shape}")
        print(f"  Positive shape: {batch['positive'].shape}")
        print(f"  Negative shape: {batch['negative'].shape}")
        print(f"  Visible unconcealed: {batch['visible_unconcealed'].shape if batch['visible_unconcealed'] is not None else 'None'}")
        print(f"  Posture classes: {batch['posture_class']}")
        break
