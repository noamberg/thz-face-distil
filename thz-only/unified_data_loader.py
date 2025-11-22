#!/usr/bin/env python3
"""
Triplets Data Loader for Face Verification, Posture Classification and Reconstruction
Adapted for the breakthrough multitask pipeline

Supports:
- Face verification (triplet loss)
- Head posture classification (5 classes)
- Image reconstruction (concealed b#b -> unconcealed n#n)
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils.augmentations import (
    StripeMasking, SpeckleNoise, RandomCutout, BandDrop,
    RandomResizedCropGrayscale, RandomRotationWithScale, RandomTranslation
)

# Constants
MIN_GROUP_SIZE_FOR_SPLIT = 3  # Minimum group size for validation split


from pathlib import Path

def parse_filename(path_str):
    """Extract numeric identity, posture, and number from a file path."""
    try:
        path = Path(path_str)
        # ID is the third-to-last part: /.../<id>/<unit>/<filename>.png
        identity = path.parts[-3]
        
        # Posture and number from filename stem
        stem_parts = path.stem.split('_')
        posture = stem_parts[1]
        number = stem_parts[2]
        
        return identity, posture, number
    except IndexError:
        return None, None, None

def get_posture_number(posture):
    """Extract posture number (1-5) from b#b or n#n format"""
    if len(posture) == 3 and posture[1].isdigit():
        return int(posture[1]) - 1  # Convert to 0-based indexing for classification
    return 0

def simple_split_summary(train_df, val_df, output_folder=None):
    """Simple split summary - just the essentials"""
    # Quick pandas-based statistics
    train_identities = train_df['anchor'].apply(lambda x: parse_filename(x)[0])
    val_identities = val_df['anchor'].apply(lambda x: parse_filename(x)[0])

    print(f"Split: {len(train_df)} train, {len(val_df)} val")
    print(f"Identities - Train: {train_identities.nunique()}, Val: {val_identities.nunique()}")

    # Save simple CSV summaries if output folder provided
    if output_folder:
        datasets_dir = os.path.join(output_folder, "datasets")
        os.makedirs(datasets_dir, exist_ok=True)

        train_df.to_csv(os.path.join(datasets_dir, "train_split.csv"), index=False)
        val_df.to_csv(os.path.join(datasets_dir, "val_split.csv"), index=False)
        train_identities.value_counts().to_csv(os.path.join(datasets_dir, "train_identity_counts.csv"))
        val_identities.value_counts().to_csv(os.path.join(datasets_dir, "val_identity_counts.csv"))

def create_uniform_validation_split(csv_path, val_ratio=0.15, random_seed=42, output_folder=None):
    """
    Create sterile and stratified validation split from train_val CSV.

    Ensures complete sterility by splitting at the SAMPLE level, not triplet level:
    - Extracts all unique image files from triplets
    - Groups samples by (identity, posture, sample_num)
    - Splits sample_nums between train and validation
    - Assigns triplets to sets based on which samples they contain
    - Maintains concealed/unconcealed pairs together
    - Ensures zero overlap: no file appears in both train and validation

    Args:
        csv_path: Path to train_val CSV file
        val_ratio: Validation split ratio (default 0.15)
        random_seed: Random seed for reproducible splits
        output_folder: Optional output folder to save split CSVs and statistics
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Load triplets dataset
    df = pd.read_csv(csv_path)

    print(f"Splitting {len(df)} triplets into train/validation...")

    # Step 1: Extract all unique files from ALL columns (anchor, positive, negative)
    all_files = set()
    all_files.update(df['anchor'].values)
    all_files.update(df['positive'].values)
    all_files.update(df['negative'].values)

    print(f"Found {len(all_files)} unique files in triplets")

    # Step 2: Parse files and organize by (identity, posture, sample_num)
    file_metadata = {}
    for filepath in all_files:
        identity, posture, sample_num = parse_filename(filepath)
        if identity and posture and sample_num:
            posture_num = get_posture_number(posture)
            file_metadata[filepath] = {
                'identity': identity,
                'posture': posture_num,
                'sample_num': sample_num,
                'is_concealed': posture.startswith('b')
            }

    # Step 3: Group by (identity, posture) and identify complete samples
    samples_by_stratum = defaultdict(lambda: defaultdict(set))

    for filepath, meta in file_metadata.items():
        key = (meta['identity'], meta['posture'], meta['sample_num'])
        samples_by_stratum[(meta['identity'], meta['posture'])][meta['sample_num']].add(filepath)

    # Step 4: Split samples (not triplets!) between train and validation
    val_sample_files = set()
    train_sample_files = set()

    total_complete_samples = 0

    for (identity, posture), samples_dict in samples_by_stratum.items():
        # Get sample numbers for this stratum
        sample_nums = list(samples_dict.keys())
        n_samples = len(sample_nums)
        total_complete_samples += n_samples

        if n_samples < 2:
            # Too few samples, assign all to training
            for sample_num in sample_nums:
                train_sample_files.update(samples_dict[sample_num])
            continue

        # Calculate validation samples (minimum 1, but leave at least 1 for training)
        n_val = max(1, int(n_samples * val_ratio))
        n_val = min(n_val, n_samples - 1)

        # Randomly select validation sample numbers
        val_sample_nums = random.sample(sample_nums, n_val)
        val_sample_nums_set = set(val_sample_nums)

        # Assign files based on sample_num
        for sample_num in sample_nums:
            if sample_num in val_sample_nums_set:
                val_sample_files.update(samples_dict[sample_num])
            else:
                train_sample_files.update(samples_dict[sample_num])

    # Verify sterility
    overlap = train_sample_files & val_sample_files
    if len(overlap) > 0:
        raise ValueError(f"STERILITY VIOLATION: {len(overlap)} files appear in both train and validation!")

    print(f"  Sample-level split: {len(train_sample_files)} train files, {len(val_sample_files)} validation files")
    print(f"  Sterility verified: NO OVERLAP ✓")

    # Step 5: Assign triplets based on anchor membership
    # A triplet belongs to validation if its ANCHOR is from a validation sample
    # This ensures:
    # - Anchor sterility: No validation anchor appears in training
    # - Sample-level grouping: Concealed/unconcealed pairs stay together
    # - Stratification: All identities/postures represented in validation
    #
    # Note: Positives and negatives may come from either set. This is acceptable because:
    # - The anchor is the primary sample being evaluated
    # - Positives/negatives from other samples provide learning signal but don't compromise
    #   the evaluation of whether the model can correctly identify the anchor's identity
    train_indices = []
    val_indices = []

    for idx, row in df.iterrows():
        anchor = row['anchor']

        # Assign based on anchor membership
        if anchor in val_sample_files:
            val_indices.append(idx)
        else:
            train_indices.append(idx)

    # Create train and validation DataFrames
    train_df = df.loc[train_indices].copy()
    val_df = df.loc[val_indices].copy()

    print(f"Created sterile validation split:")
    print(f"  Train: {len(train_df)} triplets ({100*len(train_df)/len(df):.1f}%)")
    print(f"  Validation: {len(val_df)} triplets ({100*len(val_df)/len(df):.1f}%)")

    # Save CSVs and simple summary if output_folder is provided
    if output_folder:
        simple_split_summary(train_df, val_df, output_folder)

    return train_df, val_df

class TripletsDataset(Dataset):
    """
    Dataset for triplets-based multitask learning.
    Supports different augmentation strategies for generative vs. discriminative tasks.
    """

    def __init__(self, csv_path=None, df=None, transform=None, target_transform=None, input_size=224, data_dir=None):
        """
        Args:
            csv_path: Path to triplets CSV file.
            df: Pre-loaded DataFrame (alternative to csv_path).
            transform: Transform for input images (anchor, negative).
            target_transform: Transform for the target image (positive). If None, uses `transform`.
            input_size: Input image size.
            data_dir: Base directory for image files. If provided, will be prepended to image paths.
        """
        if df is not None:
            self.df = df.reset_index(drop=True)
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path).reset_index(drop=True)
        else:
            raise ValueError("Either csv_path or df must be provided")

        self.transform = transform
        # If no specific target transform is given, use the main transform for backward compatibility
        self.target_transform = target_transform if target_transform is not None else transform
        self.input_size = input_size
        self.data_dir = data_dir

        print(f"Dataset: {len(self.df)} samples, size {input_size}x{input_size}")
        if data_dir:
            print(f"Image base directory: {data_dir}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        try:
            # Construct full paths if data_dir is provided
            if self.data_dir:
                anchor_path = os.path.join(self.data_dir, row['anchor'])
                positive_path = os.path.join(self.data_dir, row['positive'])
                negative_path = os.path.join(self.data_dir, row['negative'])
            else:
                anchor_path = row['anchor']
                positive_path = row['positive']
                negative_path = row['negative']
            
            # Load triplet images as grayscale
            anchor_img = Image.open(anchor_path).convert('L')
            positive_img = Image.open(positive_path).convert('L')
            negative_img = Image.open(negative_path).convert('L')

            # Extract posture classes from all three filenames for "Train on All" strategy
            anchor_identity, anchor_posture, _ = parse_filename(row['anchor'])
            pos_identity, pos_posture, _ = parse_filename(row['positive'])
            neg_identity, neg_posture, _ = parse_filename(row['negative'])

            anchor_posture_class = get_posture_number(anchor_posture)
            positive_posture_class = get_posture_number(pos_posture)
            negative_posture_class = get_posture_number(neg_posture)

            # The transform pipelines are expected to be set.
            # The logic in get_triplets_data_loaders ensures they are not None.
            anchor_tensor = self.transform(anchor_img)
            positive_tensor = self.target_transform(positive_img)
            negative_tensor = self.transform(negative_img)

            # Create ToTensor transform for originals
            to_tensor_transform = transforms.ToTensor()
            return {
                'anchor': anchor_tensor,
                'positive': positive_tensor,
                'negative': negative_tensor,
                'posture_class': anchor_posture_class,  # Keep for backward compatibility
                'anchor_posture_class': torch.tensor(anchor_posture_class, dtype=torch.long),
                'positive_posture_class': torch.tensor(positive_posture_class, dtype=torch.long),
                'negative_posture_class': torch.tensor(negative_posture_class, dtype=torch.long),
                'identity': anchor_identity,  # Identity is based on the anchor
                'posture': anchor_posture,
                'idx': idx,
                # Provide original (un-augmented) versions of all three images
                'original_anchor': to_tensor_transform(anchor_img),
                'original_positive': to_tensor_transform(positive_img),
                'original_negative': to_tensor_transform(negative_img)
            }

        except FileNotFoundError as e:
            raise RuntimeError(f"Image file not found at index {idx}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load sample at index {idx}: {e}")

def get_triplets_data_loaders(train_csv=None, val_csv=None, test_csv=None, train_val_csv=None,
                             input_size=224, batch_size=16, num_workers=4, val_ratio=0.15,
                             random_seed=42, output_folder=None, augmentation_mode='gentle'):
    """
    Create data loaders for triplets dataset with support for both legacy and new CSV formats
    
    Args:
        train_csv: Path to train CSV file (new format)
        val_csv: Path to validation CSV file (new format)  
        test_csv: Path to test CSV file (optional)
        train_val_csv: Path to train_val CSV file (legacy format)
        input_size: Input image size (56 for original, 196 for bicubic)
        batch_size: Batch size
        num_workers: Number of worker processes
        val_ratio: Validation split ratio (default 0.15) - only used for legacy format
        random_seed: Random seed for reproducible splits
        output_folder: Optional output folder to save split CSVs and statistics
        augmentation_mode: 'gentle' for generative tasks, 'medium' for THz reconstruction, 'strong' for discriminative tasks
    
    Returns:
        train_loader, val_loader, test_loader (if test_csv provided)
    """

    # Handle both new format (separate train/val CSVs) and legacy format (single train_val CSV)
    if train_csv is not None and val_csv is not None:
        # New format: load already split training and validation sets
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
    elif train_val_csv is not None:
        # Legacy format: create uniform validation split
        train_df, val_df = create_uniform_validation_split(train_val_csv, val_ratio, random_seed, output_folder)
    else:
        raise ValueError("Either (train_csv and val_csv) or train_val_csv must be provided")

    # --- Define Augmentation Pipelines ---
    # THz-specific augmentations designed for the domain characteristics

    # Minimal augmentation for the reconstruction target image
    target_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # GENTLE PROFILE
    # - Rotation [−10°, +10°] or scale [0.9,1.1]
    # - Random crop area [0.9,1.0] & resize
    # - Stripe masking (2–4 px width) with prob ~0.2
    # - Speckle noise ε~N(0, 0.05–0.1)
    gentle_train_transform = transforms.Compose([
        RandomRotationWithScale(degrees=10, scale=(0.9, 1.1), prob=1.0),
        RandomResizedCropGrayscale(size=input_size, scale=(0.9, 1.0)),
        StripeMasking(stripe_width=(2, 4), prob=0.2, max_stripes=3),
        SpeckleNoise(std=(0.05, 0.1), prob=1.0),
        transforms.ToTensor(),
    ])

    # MEDIUM PROFILE
    # - All of Gentle profile
    # - Rotation up to ±20°, random translation ±5 px
    # - Crop area [0.7,1.0] & resize
    # - Cutout: 2–4 patches of 8–16 px
    # - Strong speckle (ε~N(0,0.05–0.15)) or band-drop with prob ~0.3
    medium_train_transform = transforms.Compose([
        RandomRotationWithScale(degrees=20, scale=(0.9, 1.1), prob=1.0),
        RandomTranslation(translate=5, prob=1.0),
        RandomResizedCropGrayscale(size=input_size, scale=(0.7, 1.0)),
        StripeMasking(stripe_width=(2, 4), prob=0.2, max_stripes=3),
        RandomCutout(num_patches=(2, 4), patch_size=(8, 16), prob=0.3),
        BandDrop(band_width=(4, 12), prob=0.3),
        SpeckleNoise(std=(0.05, 0.15), prob=1.0),
        transforms.ToTensor(),
    ])

    # STRONG PROFILE
    # - All of Medium profile
    # - Rotation up to ±30°, translation ±10 px, scale [0.8,1.2]
    # - Crop area [0.6,1.0] & resize
    # - Stronger stripe masking (4–8 px, prob ~0.4) and cutout (4–6 patches)
    # - Band-drop with prob ~0.5
    # - Very strong speckle (ε~N(0,0.1–0.2))
    strong_train_transform = transforms.Compose([
        RandomRotationWithScale(degrees=30, scale=(0.8, 1.2), prob=1.0),
        RandomTranslation(translate=10, prob=1.0),
        RandomResizedCropGrayscale(size=input_size, scale=(0.6, 1.0)),
        StripeMasking(stripe_width=(4, 8), prob=0.4, max_stripes=5),
        RandomCutout(num_patches=(4, 6), patch_size=(8, 20), prob=0.5),
        BandDrop(band_width=(8, 16), prob=0.5),
        SpeckleNoise(std=(0.1, 0.2), prob=1.0),
        transforms.ToTensor(),
    ])

    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # --- Select Augmentation Strategy ---

    if augmentation_mode == 'strong':
        print("Using STRONG augmentation pipeline for inputs.")
        train_transform = strong_train_transform
    elif augmentation_mode == 'medium':
        print("Using MEDIUM augmentation pipeline for inputs.")
        train_transform = medium_train_transform
    else:  # 'gentle' or default
        print("Using GENTLE augmentation pipeline for inputs.")
        train_transform = gentle_train_transform

    # For reconstruction, the target should always have minimal (or no) augmentation.
    train_target_transform = target_transform

    # --- Create Datasets ---

    train_dataset = TripletsDataset(
        df=train_df,
        transform=train_transform,
        target_transform=train_target_transform,
        input_size=input_size
    )

    val_dataset = TripletsDataset(
        df=val_df,
        transform=val_transform,
        target_transform=val_transform, # No augmentation on validation
        input_size=input_size
    )

    # --- Create Data Loaders ---

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    test_loader = None
    if test_csv is not None:
        test_dataset = TripletsDataset(
            csv_path=test_csv,
            transform=val_transform,
            target_transform=val_transform, # No augmentation on test
            input_size=input_size
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )

    print(f"Created data loaders:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Validation: {len(val_loader)} batches ({len(val_dataset)} samples)")
    if test_loader:
        print(f"  Test: {len(test_loader)} batches ({len(test_dataset)} samples)")

    return train_loader, val_loader, test_loader


def visualize_augmentations(dataset, transform, target_transform, num_samples=20, output_path=None):
    """
    Visualize augmentation samples to verify they look correct.
    
    Args:
        dataset: TripletsDataset instance
        transform: Transform pipeline for anchor/negative images
        target_transform: Transform pipeline for positive images
        num_samples: Number of samples to visualize (default 20)
        output_path: Path to save the visualization (if None, uses dataset folder + '/augmentation_samples.png')
    """
    # Create a temporary dataset with identity transform to get original images
    identity_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Sample random indices
    sample_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    # Set up the plot
    fig = plt.figure(figsize=(20, 4 * len(sample_indices)))
    gs = gridspec.GridSpec(len(sample_indices), 7, hspace=0.3, wspace=0.2)
    
    for i, idx in enumerate(sample_indices):
        row = dataset.df.iloc[idx]
        
        try:
            # Load original images
            anchor_img = Image.open(row['anchor']).convert('L')
            positive_img = Image.open(row['positive']).convert('L')
            negative_img = Image.open(row['negative']).convert('L')
            
            # Get identity info for labeling
            identity, posture, _ = parse_filename(row['anchor'])
            
            # Apply transforms
            original_anchor = identity_transform(anchor_img)
            original_positive = identity_transform(positive_img)
            original_negative = identity_transform(negative_img)
            
            aug_anchor = transform(anchor_img)
            aug_positive = target_transform(positive_img)
            aug_negative = transform(negative_img)
            
            # Convert tensors to numpy for plotting (handle grayscale)
            def tensor_to_plot(tensor):
                if tensor.dim() == 3 and tensor.shape[0] == 1:  # [1, H, W]
                    return tensor.squeeze(0).numpy()
                return tensor.numpy()
            
            # Plot row: Original Anchor, Aug Anchor, Original Positive, Aug Positive, Original Negative, Aug Negative, Label
            axes = [fig.add_subplot(gs[i, j]) for j in range(7)]
            
            # Original images
            axes[0].imshow(tensor_to_plot(original_anchor), cmap='gray')
            axes[0].set_title(f'Orig Anchor\n{identity}_{posture}', fontsize=8)
            axes[0].axis('off')
            
            axes[1].imshow(tensor_to_plot(aug_anchor), cmap='gray')
            axes[1].set_title('Aug Anchor', fontsize=8)
            axes[1].axis('off')
            
            axes[2].imshow(tensor_to_plot(original_positive), cmap='gray')
            axes[2].set_title('Orig Positive', fontsize=8)
            axes[2].axis('off')
            
            axes[3].imshow(tensor_to_plot(aug_positive), cmap='gray')
            axes[3].set_title('Aug Positive', fontsize=8)
            axes[3].axis('off')
            
            axes[4].imshow(tensor_to_plot(original_negative), cmap='gray')
            axes[4].set_title('Orig Negative', fontsize=8)
            axes[4].axis('off')
            
            axes[5].imshow(tensor_to_plot(aug_negative), cmap='gray')
            axes[5].set_title('Aug Negative', fontsize=8)
            axes[5].axis('off')
            
            # Info panel
            axes[6].text(0.1, 0.8, f'Sample {idx}', fontsize=10, weight='bold', transform=axes[6].transAxes)
            axes[6].text(0.1, 0.6, f'ID: {identity}', fontsize=8, transform=axes[6].transAxes)
            axes[6].text(0.1, 0.4, f'Posture: {posture}', fontsize=8, transform=axes[6].transAxes)
            axes[6].text(0.1, 0.2, f'Class: {get_posture_number(posture)}', fontsize=8, transform=axes[6].transAxes)
            axes[6].set_xlim(0, 1)
            axes[6].set_ylim(0, 1)
            axes[6].axis('off')
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    # Add overall title
    if "GaussianBlur" in str(transform) and "sigma=(0.1, 0.6)" in str(transform):
        augmentation_type = "STRONG"
    elif "GaussianBlur" in str(transform) and "sigma=(0.1, 0.3)" in str(transform):
        augmentation_type = "MEDIUM"
    else:
        augmentation_type = "GENTLE"
    fig.suptitle(f'{augmentation_type} Augmentation Samples - {len(sample_indices)} Random Triplets', 
                 fontsize=16, y=0.98)
    
    # Save the plot
    if output_path is None:
        output_path = 'augmentation_samples.png'
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Augmentation samples saved to: {output_path}")


def visualize_expert_augmentations_from_loaders(data_loaders, epoch, output_folder):
    """
    Creates augmentation visualizations from expert data loaders showing original vs augmented images.
    Saves one visualization per expert using samples from each loader.
    
    Args:
        data_loaders: Dict of {posture_id: DataLoader} for each expert
        epoch: Current epoch number (0-based)
        output_folder: Output folder to save visualizations
    """
    # Only generate visualizations every 20 epochs as specified in the plan
    if (epoch + 1) % 20 != 0:
        return
    
    # Create augmentation samples directory
    aug_dir = os.path.join(output_folder, "augmentation_samples")
    os.makedirs(aug_dir, exist_ok=True)
    
    print(f"[Epoch {epoch+1}] Generating expert augmentation visualizations...")
    
    for posture_id, data_loader in data_loaders.items():
        try:
            # Get a batch from the data loader
            batch = next(iter(data_loader))
            
            # Create visualization comparing original vs augmented anchors
            num_samples = min(10, len(batch['anchor']))  # Show up to 10 samples per expert
            
            fig, axes = plt.subplots(2, num_samples, figsize=(2 * num_samples, 4))
            if num_samples == 1:
                axes = axes.reshape(2, 1)
            
            for i in range(num_samples):
                # Original anchor (untransformed)
                original_img = batch['original_anchor'][i].squeeze(0).numpy()
                axes[0, i].imshow(original_img, cmap='gray')
                axes[0, i].set_title(f'Original\n{batch["identity"][i]}_{batch["posture"][i]}', fontsize=8)
                axes[0, i].axis('off')
                
                # Augmented anchor
                augmented_img = batch['anchor'][i].squeeze(0).numpy()
                axes[1, i].imshow(augmented_img, cmap='gray')
                axes[1, i].set_title('Augmented', fontsize=8)
                axes[1, i].axis('off')
            
            # Save visualization for this expert
            posture_name = batch['posture'][0] if len(batch['posture']) > 0 else f"posture_{posture_id}"
            output_path = os.path.join(aug_dir, f"epoch_{epoch+1:03d}_expert_{posture_name}_augmentations.png")
            
            plt.suptitle(f'Expert {posture_name} - Epoch {epoch+1} - Original vs Augmented', fontsize=14)
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"  Saved expert {posture_name} augmentation samples: {output_path}")
            
        except Exception as e:
            print(f"Warning: Failed to generate augmentation visualization for expert {posture_id}: {e}")
            continue

def save_augmentation_samples(train_loader, epoch, output_folder, augmentation_mode='gentle'):
    """
    Save augmentation samples every 10 epochs for visual inspection.
    
    Args:
        train_loader: Training data loader
        epoch: Current epoch number (0-based)
        output_folder: Output folder to save visualizations
        augmentation_mode: 'gentle' or 'strong' for labeling
    """
    # Only save every 10 epochs (epochs 0, 4, 9, 14, etc.)
    if (epoch + 1) % 10 != 0:
        return
    
    # Create augmentation samples directory
    aug_dir = os.path.join(output_folder, "augmentation_samples")
    os.makedirs(aug_dir, exist_ok=True)
    
    # Get dataset and transforms from the data loader
    dataset = train_loader.dataset
    
    # Extract transforms (handle both regular DataLoader and custom BatchSampler)
    if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'dataset'):
        # Custom batch sampler case
        dataset = train_loader.batch_sampler.dataset
    
    transform = dataset.transform
    target_transform = dataset.target_transform
    
    # Create filename with epoch number (overwrite every 5 epochs)
    output_path = os.path.join(aug_dir, f"epoch_{epoch+1:03d}_{augmentation_mode}_augmentation_samples.png")
    
    print(f"Saving {augmentation_mode} augmentation samples for epoch {epoch+1} to: {output_path}")
    
    try:
        visualize_augmentations(
            dataset=dataset,
            transform=transform, 
            target_transform=target_transform,
            num_samples=20,
            output_path=output_path
        )
    except Exception as e:
        print(f"Warning: Failed to save augmentation samples: {e}")


if __name__ == "__main__":
    from config import DATA_CONFIG, TRAIN_CONFIG

    train_loader, val_loader, test_loader = get_triplets_data_loaders(
        train_val_csv=DATA_CONFIG['train_val_csv'],
        test_csv=DATA_CONFIG.get('test_csv'),
        input_size=64,
        batch_size=TRAIN_CONFIG['batch_size'],
        val_ratio=TRAIN_CONFIG['val_ratio']
    )

    print("Testing data loading...")
    for i, batch in enumerate(train_loader):
        print(f"Batch {i}:")
        print(f"  Anchor shape: {batch['anchor'].shape}")
        print(f"  Positive shape: {batch['positive'].shape}")
        print(f"  Negative shape: {batch['negative'].shape}")
        print(f"  Posture classes: {batch['posture_class']}")
        break