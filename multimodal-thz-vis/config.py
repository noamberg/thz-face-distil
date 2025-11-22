#!/usr/bin/env python3
"""
Configuration for Multimodal THz Face Recognition with Privileged Information Learning

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
    'thz_data_dir': '/path/to/thz/data',  # THz image data directory
    'visible_data_dir': '/path/to/visible/data',  # Visible image data directory
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
    'embed_dim': 64,  # Embedding dimension for verification
    'input_size': 64,  # Input image size (64x64)
    'dropout_rate': 0.5,  # Dropout rate
    'pretrained': False,  # Use pretrained weights for backbone
    'cross_attention_heads': 8,  # Number of cross-attention heads
    'distill_dim': 256,  # Distillation projection dimension
    'posture_ids': [0, 1, 2, 3, 4],  # Posture class IDs
}

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

TRAIN_CONFIG = {
    'batch_size': 256,
    'learning_rate': 1e-4,
    'epochs': 500,
    'weight_decay': 1e-3,
    'grad_clip': 1.0,  # Gradient clipping threshold
    'num_workers': 4,  # DataLoader workers
    'random_seed': 42,
    'use_scheduler': True,  # Use learning rate scheduler
    'validation_interval_epochs': 0.5,  # Run validation every N epochs
    'val_ratio': 0.15,  # Validation split ratio
}

# ============================================================================
# DISTILLATION AND TEACHER CONFIGURATION
# ============================================================================

DISTILL_CONFIG = {
    'freeze_teacher': False,  # Freeze teacher model parameters
    'distill_full_triplet': True,  # Distill anchor+positive+negative logits
    'distillation_temperature': 3.0,  # Temperature for logit distillation
}

# ============================================================================
# LOSS WEIGHTS
# ============================================================================

LOSS_WEIGHTS = {
    'verification_student': 1.0,  # Student verification loss weight
    'verification_teacher': 0.5,  # Teacher verification loss weight
    'classification_student': 1.0,  # Student classification loss weight
    'classification_teacher': 0.5,  # Teacher classification loss weight
    'feature_distillation': 1.0,  # Feature distillation loss weight
    'logit_distillation': 0.5,  # Logit distillation loss weight
    'reconstruction': 1.0,  # Reconstruction loss weight
    'reconstruction_weights': {
        'charbonnier': 1.0,
        'ssim': 0.6,
        'lpips': 0.4,
    }
}

# ============================================================================
# LOSS FUNCTION HYPERPARAMETERS
# ============================================================================

LOSS_CONFIG = {
    'supcon_temperature': 0.07,  # Supervised contrastive loss temperature
    'classification_smoothing': 0.1,  # Label smoothing for classification
}

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

AUGMENTATION_CONFIG = {
    'augmentation_mode': 'gentle',  # 'gentle', 'medium', or 'strong'

    # Gentle augmentation parameters
    'gentle': {
        'rotation_degrees': 20,
        'scale_range': (0.8, 1.2),
        'crop_scale': (0.8, 0.9),
        'stripe_width': (4, 6),
        'stripe_prob': 0.5,
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
# INFERENCE CONFIGURATION
# ============================================================================

INFERENCE_CONFIG = {
    'checkpoint_path': 'runs/best_model/checkpoints/best.pth',  # Path to checkpoint
    'max_samples': 2100,  # Maximum number of samples to process during inference
    'batch_size_inference': 128,  # Batch size for inference
}

# ============================================================================
# LOGGING AND VISUALIZATION
# ============================================================================

LOGGING_CONFIG = {
    'log_level': 'INFO',  # Logging level
    'plot_interval_steps': 50,  # Save plots every N steps
    'visualization_interval_steps': 100,  # Save visualizations every N steps
    'save_code_snapshot': True,  # Save code files for reproducibility
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
        'cross_attention_heads': MODEL_CONFIG['cross_attention_heads'],
        'distill_dim': MODEL_CONFIG['distill_dim'],
        'posture_ids': MODEL_CONFIG['posture_ids'],
        'num_workers': TRAIN_CONFIG['num_workers'],
        'random_seed': TRAIN_CONFIG['random_seed'],
        'weight_decay': TRAIN_CONFIG['weight_decay'],
        'grad_clip': TRAIN_CONFIG['grad_clip'],
        'use_scheduler': TRAIN_CONFIG['use_scheduler'],
        'validation_interval_epochs': TRAIN_CONFIG['validation_interval_epochs'],
        'val_ratio': TRAIN_CONFIG['val_ratio'],
        'freeze_teacher': DISTILL_CONFIG['freeze_teacher'],
        'distill_full_triplet': DISTILL_CONFIG['distill_full_triplet'],
        'distillation_temperature': DISTILL_CONFIG['distillation_temperature'],
        'thz_data_dir': DATA_CONFIG['thz_data_dir'],
        'visible_data_dir': DATA_CONFIG['visible_data_dir'],
        'train_val_csv': DATA_CONFIG['train_val_csv'],
        'augmentation_mode': AUGMENTATION_CONFIG['augmentation_mode'],
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
    optional_dirs = [DATA_CONFIG['thz_data_dir'], DATA_CONFIG['visible_data_dir']]

    missing_required = [d for d in required_dirs if not os.path.exists(d)]
    if missing_required:
        raise FileNotFoundError(f"Required directories not found: {missing_required}")

    missing_optional = [d for d in optional_dirs if not os.path.exists(d)]
    if missing_optional:
        print(f"Warning: Optional directories not found: {missing_optional}")

if __name__ == "__main__":
    config = get_full_config()
    print("Configuration Summary:")
    print("=" * 60)
    for section in ['MODEL', 'TRAIN', 'DISTILL', 'LOSS']:
        print(f"\n{section}:")
        section_config = globals()[f'{section}_CONFIG']
        for key, value in section_config.items():
            print(f"  {key}: {value}")
