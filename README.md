# THz Face Reconstruction with Multi-Task Learning, supports Cross-Modal Visible images + Knowledge Distillation
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
🏆 Best Student Paper Award in Artificial Intelligence for Security and Defence, SPIE Security + Defence 2025 (Madrid) 

Deep learning framework for THz face analysis with three tasks: verification, posture classification, and image reconstruction.

<img width="913" height="607" alt="pipe-obfuscated" src="https://github.com/user-attachments/assets/bd1e9936-4c90-478f-81b3-4f2d689a5b99" />


## Quick Start

```bash
# 1. Choose your implementation
cd thz-only                    # THz-only training
# OR
cd multimodal-thz-vis          # THz + visible (privileged learning)

# 2. Configure paths
vim config.py                  # Update data paths

# 3. Train
python train_unified_mtl.py

# 4. Inference
python unified_inference.py --checkpoint runs/best_model/checkpoints/best.pth
```

## Two Implementations

### thz-only/
Standard multi-task learning using only THz images.
- **Use when:** You only have THz data
- **Training:** THz images only
- **Inference:** THz images only

### multimodal-thz-vis/
Privileged information learning with teacher-student distillation.
- **Use when:** You have both THz and visible images
- **Training:** THz + visible images (teacher supervises student)
- **Inference:** THz images only (no visible images needed)

<img width="540" height="1134" alt="teacher-obfuscated" src="https://github.com/user-attachments/assets/ff6e0cbd-b5c5-4b8a-9984-d81b8d28804e" />

## Weights Files (pth)
Pretrained weights for Thz-only model and Cross-Modal Visible-THz student model are available for free at:
https://drive.google.com/drive/folders/1nU3UbM4dwTui0FnwviI1XxwHPXLV_av_?usp=sharing

- Can be used as a pretrained starting point for other modalities as well

## Three Tasks

1. **Face Verification:** Identify if two images are the same person
2. **Posture Classification:** Classify head pose (Front/Left/Right/Up/Down)
3. **Image Reconstruction:** Reconstruct unconcealed from concealed images

## Directory Structure

```
thz-face-distil/
├── README.md              # This file
├── thz-only/              # THz-only implementation
│   ├── config.py          # Configuration (edit this)
│   ├── train_unified_mtl.py
│   ├── unified_inference.py
│   ├── unified_architecture.py
│   ├── unified_data_loader.py
│   ├── unified_losses.py
│   └── utils/             # Augmentations, metrics, schedulers
└── multimodal-thz-vis/    # Multimodal implementation
    ├── config.py          # Configuration (edit this)
    ├── train_unified_mtl.py
    ├── unified_inference.py
    ├── unified_architecture.py
    ├── unified_data_loader.py
    ├── unified_losses.py
    └── utils/             # Augmentations
```

## Data Format

### CSV Format
Triplet format with columns: `anchor`, `positive`, `negative`
```csv
anchor,positive,negative
id1_b1b_1.png,id1_n1n_1.png,id2_b1b_1.png
```

For multimodal, add: `visible_concealed`, `visible_unconcealed`, `visible_negative` (optional)

### Image Naming
`<identity>_<posture>_<sample>.png`
- **Posture codes:** b1b-b5b (concealed), n1n-n5n (unconcealed)
  - 1=Front, 2=Left, 3=Right, 4=Up, 5=Down

## Configuration

Edit `config.py` in your chosen directory:

```python
# Data paths
DATA_CONFIG = {
    'data_dir': '/path/to/thz/data',
    'train_val_csv': 'data/train_val.csv',
    'test_csv': 'data/test.csv',
}

# Training
TRAIN_CONFIG = {
    'batch_size': 64,
    'learning_rate': 1e-4,
    'epochs': 200,
}

# Model
MODEL_CONFIG = {
    'backbone_type': 'custom',  # or 'resnet18'
    'embed_dim': 128,
    'num_classes': 5,
}
```

## Features

### Architecture
- **Encoder:** Custom CNN Encoder
- **Verification:** Supervised Contrastive Loss + MLP head
- **Classification:** 5-class linear classifier
- **Reconstruction:** Mixture-of-Experts U-Net (one expert per posture)

### Augmentation (THz-Specific)
Three profiles: `gentle` (default), `medium`, `strong`
- Stripe masking, speckle noise, band-drop
- Rotation, scaling, cropping, cutout
- Set in `config.py`: `AUGMENTATION_CONFIG['augmentation_mode']`

### Loss Functions
- **Verification:** Supervised Contrastive Loss (SupCon)
- **Classification:** Label Smoothing Cross-Entropy
- **Reconstruction:** Charbonnier + SSIM + LPIPS

### Multimodal-Specific
- Dual cross-attention fusion (THz ↔ Visible)
- Feature + logit distillation
- Teacher-student framework
- Inference uses student only (no visible images)

## Training

```bash
# Basic training
python train_unified_mtl.py

# With custom config
python train_unified_mtl.py --config my_config.json
```

### Outputs
```
runs/<experiment>_<timestamp>/
├── checkpoints/
│   ├── best.pth           # Best validation SSIM
│   └── latest.pth
├── logs/
│   ├── train_logs.csv
│   └── validation_logs.csv
└── plots/
    ├── latest.png         # Training curves
    └── latest_val.png     # Validation metrics
```

## Inference

```bash
python unified_inference.py \
    --checkpoint runs/best_model/checkpoints/best.pth \
    --test_csv data/test.csv \
    --output_dir results/
```

Generates:
- Per-sample metrics (all three tasks)
- Per-identity and per-posture statistics
- Visualizations

## Requirements

```bash
pip install torch torchvision pandas numpy pillow matplotlib tqdm
pip install pytorch-msssim lpips  # For losses
```

## Key Configuration Options

| Setting | Description | Default |
|---------|-------------|---------|
| `batch_size` | Training batch size | 64 |
| `learning_rate` | Initial learning rate | 1e-4 |
| `epochs` | Training epochs | 200 |
| `embed_dim` | Embedding dimension | 128 |
| `backbone_type` | 'custom' or 'resnet18' | custom |
| `augmentation_mode` | 'gentle', 'medium', or 'strong' | gentle |
| `val_check_interval` | Validation frequency (epochs) | 0.25 |

**Loss weights** (adjust relative task importance):
```python
LOSS_WEIGHTS = {
    'verification': 1.0,
    'classification': 1.0,
    'reconstruction': 1.0,
}
```

**Multimodal-only settings:**
```python
DISTILL_CONFIG = {
    'freeze_teacher': False,          # Freeze teacher parameters
    'distill_full_triplet': True,     # Distill all triplet images
    'distillation_temperature': 3.0,  # Temperature scaling
}
```

## Tips

1. **Start with thz-only** if you don't have visible images
2. **Use gentle augmentation** for reconstruction tasks
3. **Use strong augmentation** for verification/classification
4. **Adjust loss weights** if one task dominates
5. **Monitor SSIM** for reconstruction quality
6. **Check validation curves** to detect overfitting

## Citation

If you use this code, please cite:
```
Bergman N, Yildirim IO, Sahin AB, Altan H, Yitzhaky Y. A deep-learning framework for concealed and unconcealed face analysis in sub millimeter wave imaging. In Artificial Intelligence for Security and Defence Applications III 2025 Oct 27 (Vol. 13679, pp. 221-227). SPIE.
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

