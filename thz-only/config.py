#!/usr/bin/env python3
"""
Configuration for THz-Only Face Recognition with Multi-Task Learning

All hyperparameters, paths, and settings for training, validation, and inference.
Edit this file to customize the system for your dataset and setup.
"""

import os
from pathlib import Path

# ============================================================================
# PATHS - Update these for your environment
# ============================================================================

# Project root (auto-detect from this file's location or set manually)
PROJECT_ROOT = str(Path(__file__).parent.parent.absolute())

# Data directories
DATA_CONFIG = {
    'data_dir': '/path/to/thz/data',  # THz image data directory
    'train_val_csv': 'data/train_val_dataset.csv',  # Train/val CSV (relative to PROJECT_ROOT)
    'train_csv': 'data/train_dataset.csv',  # Train CSV (alternative format)
    'val_csv': 'data/val_dataset.csv',  # Validation CSV (alternative format)
    'test_csv': 'data/test_dataset.csv',  # Test CSV
}

# Output directories
OUTPUT_CONFIG = {
    'runs_dir': 'runs',  # Base directory for training runs
    'test_runs_dir': 'test_runs',  # Base directory for inference runs
    'checkpoint_dir': 'checkpoints',  # Checkpoint subdirectory within runs
    'logs_dir': 'logs',  # Logs subdirectory within runs
    'plots_dir': 'plots',  # Plots subdirectory within runs
}

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

MODEL_CONFIG = {
    'backbone_type': 'custom',  # 'custom' or 'resnet18'
    'num_classes': 5,  # Number of posture classes
    'embed_dim': 128,  # Embedding dimension for verification
    'input_size': 64,  # Input image size (64x64)
    'dropout_rate': 0.5,  # Dropout rate
    'pretrained': True,  # Use pretrained weights for backbone (only for resnet18)
    'posture_ids': [1, 2, 3, 4, 5],  # Posture class IDs (MoE reconstruction)
}

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

TRAIN_CONFIG = {
    'batch_size': 64,
    'learning_rate': 1e-4,
    'epochs': 200,
    'weight_decay': 1e-4,
    'grad_clip': 1.0,  # Gradient clipping threshold
    'num_workers': 4,  # DataLoader workers
    'random_seed': 42,
    'use_scheduler': True,  # Use learning rate scheduler
    'scheduler_type': 'cosine_warmup',  # 'plateau', 'cosine_warmup', or 'step'
    'warmup_epochs': 10,  # Warmup epochs for cosine warmup scheduler
    'val_check_interval': 0.25,  # Validate every N epochs
    'val_ratio': 0.15,  # Validation split ratio
    'use_balanced_sampler': True,  # Use balanced batch sampler
}

# Scheduler-specific parameters
SCHEDULER_CONFIG = {
    # ReduceLROnPlateau
    'plateau': {
        'mode': 'min',
        'factor': 0.5,
        'patience': 10,
        'min_lr': 1e-7,
    },
    # CosineAnnealingWarmupRestarts
    'cosine_warmup': {
        'first_cycle_steps': 50,  # Steps in first cycle
        'cycle_mult': 1.0,  # Cycle length multiplier
        'max_lr': 1e-4,  # Maximum learning rate
        'min_lr': 1e-7,  # Minimum learning rate
        'warmup_steps': 10,  # Warmup steps
        'gamma': 0.9,  # Decay factor for max_lr
    },
    # StepLR
    'step': {
        'step_size': 30,
        'gamma': 0.1,
    }
}

# ============================================================================
# LOSS WEIGHTS
# ============================================================================

LOSS_WEIGHTS = {
    'verification': 1.0,  # Verification loss weight
    'classification': 1.0,  # Classification loss weight
    'reconstruction': 1.0,  # Reconstruction loss weight
    'reconstruction_weights': {
        'charbonnier': 1.0,
        'ssim': 0.8,
        'gradient': 0.2,
        'lpips': 0.5,
    }
}

# ============================================================================
# LOSS FUNCTION HYPERPARAMETERS
# ============================================================================

LOSS_CONFIG = {
    'verification_loss_type': 'supcon',  # 'triplet' or 'supcon'
    'triplet_margin': 0.3,  # Margin for triplet loss
    'supcon_temperature': 0.07,  # Temperature for supervised contrastive loss
    'classification_smoothing': 0.1,  # Label smoothing for classification
}

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

AUGMENTATION_CONFIG = {
    'augmentation_mode': 'gentle',  # 'gentle', 'medium', or 'strong'

    # Gentle augmentation parameters
    'gentle': {
        'rotation_degrees': 10,
        'scale_range': (0.9, 1.1),
        'crop_scale': (0.9, 1.0),
        'stripe_width': (2, 4),
        'stripe_prob': 0.2,
        'max_stripes': 3,
        'speckle_std': (0.05, 0.1),
    },

    # Medium augmentation parameters
    'medium': {
        'rotation_degrees': 20,
        'scale_range': (0.9, 1.1),
        'translation_px': 5,
        'crop_scale': (0.7, 1.0),
        'stripe_width': (2, 4),
        'stripe_prob': 0.2,
        'max_stripes': 3,
        'cutout_patches': (2, 4),
        'cutout_size': (8, 16),
        'cutout_prob': 0.3,
        'band_width': (4, 12),
        'band_prob': 0.3,
        'speckle_std': (0.15, 0.25),
    },

    # Strong augmentation parameters
    'strong': {
        'rotation_degrees': 20,
        'scale_range': (0.85, 1.15),
        'translation_px': 5,
        'crop_scale': (0.7, 1.0),
        'stripe_width': (2, 4),
        'stripe_prob': 0.3,
        'max_stripes': 4,
        'cutout_patches': (2, 4),
        'cutout_size': (8, 16),
        'cutout_prob': 0.4,
        'band_width': (4, 12),
        'band_prob': 0.3,
        'speckle_std': (0.15, 0.25),
    }
}

# ============================================================================
# BALANCED BATCH SAMPLER
# ============================================================================

SAMPLER_CONFIG = {
    'identities_per_batch': 8,  # Number of unique identities per batch
    'samples_per_identity': 8,  # Samples per identity (should divide batch_size evenly)
}

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================

INFERENCE_CONFIG = {
    'checkpoint_path': 'runs/best_model/checkpoints/best.pth',  # Path to checkpoint
    'max_samples': 500,  # Maximum number of samples to process during inference
    'batch_size_inference': 32,  # Batch size for inference
    'verification_margin': 0.2,  # Margin for verification accuracy calculation
}

# ============================================================================
# LOGGING AND VISUALIZATION
# ============================================================================

LOGGING_CONFIG = {
    'log_level': 'INFO',  # Logging level
    'log_interval_batches': 10,  # Log every N batches
    'plot_interval_epochs': 1,  # Save plots every N epochs
    'save_code_snapshot': True,  # Save code files for reproducibility
    'visualization_samples': 10,  # Number of samples to visualize
}

# ============================================================================
# HARDWARE
# ============================================================================

HARDWARE_CONFIG = {
    'device': 'cuda',  # 'cuda' or 'cpu'
    'pin_memory': True,  # Pin memory for DataLoader
    'non_blocking': True,  # Non-blocking transfers
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_full_config():
    """Return combined configuration dictionary for training"""
    config = {
        'project_root': PROJECT_ROOT,
        'input_size': MODEL_CONFIG['input_size'],
        'batch_size': TRAIN_CONFIG['batch_size'],
        'learning_rate': TRAIN_CONFIG['learning_rate'],
        'epochs': TRAIN_CONFIG['epochs'],
        'num_classes': MODEL_CONFIG['num_classes'],
        'embed_dim': MODEL_CONFIG['embed_dim'],
        'backbone_type': MODEL_CONFIG['backbone_type'],
        'dropout_rate': MODEL_CONFIG['dropout_rate'],
        'pretrained': MODEL_CONFIG['pretrained'],
        'num_workers': TRAIN_CONFIG['num_workers'],
        'random_seed': TRAIN_CONFIG['random_seed'],
        'weight_decay': TRAIN_CONFIG['weight_decay'],
        'grad_clip': TRAIN_CONFIG['grad_clip'],
        'use_scheduler': TRAIN_CONFIG['use_scheduler'],
        'scheduler_type': TRAIN_CONFIG['scheduler_type'],
        'warmup_epochs': TRAIN_CONFIG['warmup_epochs'],
        'val_check_interval': TRAIN_CONFIG['val_check_interval'],
        'val_ratio': TRAIN_CONFIG['val_ratio'],
        'posture_ids': MODEL_CONFIG['posture_ids'],
        'train_val_csv': DATA_CONFIG['train_val_csv'],
        'data_dir': DATA_CONFIG['data_dir'],
        'augmentation_mode': AUGMENTATION_CONFIG['augmentation_mode'],
        'validation_interval_epochs': TRAIN_CONFIG['val_check_interval'],
        'supcon_temperature': LOSS_CONFIG['supcon_temperature'],
        'classification_smoothing': LOSS_CONFIG['classification_smoothing'],
        'loss_weights': LOSS_WEIGHTS,
    }
    return config

def get_absolute_path(relative_path: str) -> str:
    """Convert relative path to absolute path relative to PROJECT_ROOT"""
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(PROJECT_ROOT, relative_path)

def validate_paths():
    """Validate that required paths exist"""
    required_dirs = []
    optional_dirs = [DATA_CONFIG['data_dir']]

    missing_required = [d for d in required_dirs if not os.path.exists(d)]
    if missing_required:
        raise FileNotFoundError(f"Required directories not found: {missing_required}")

    missing_optional = [d for d in optional_dirs if not os.path.exists(d)]
    if missing_optional:
        print(f"Warning: Optional directories not found: {missing_optional}")

def print_config_summary():
    """Print a summary of the configuration"""
    config = get_full_config()
    print("Configuration Summary:")
    print("=" * 60)
    for section in ['MODEL', 'TRAIN', 'LOSS', 'AUGMENTATION']:
        print(f"\n{section}:")
        section_config = globals()[f'{section}_CONFIG']
        for key, value in section_config.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

if __name__ == "__main__":
    print_config_summary()
