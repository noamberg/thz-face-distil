#!/usr/bin/env python3
"""
Unified MTL Inference Pipeline
Performs inference using the unified multi-task learning model

Features:
1. Face verification with triplet metrics
2. Head posture classification accuracy  
3. Image reconstruction quality
4. Multi-resolution support (56x56 and 196x196)
5. Comprehensive visualization and metrics
"""

import os
import time
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
import argparse
import sys

# Set working directory to the project root
PROJECT_ROOT = "/home/noam/PycharmProjects/thz"
os.chdir(PROJECT_ROOT)

# Add the project root to the path so we can import modules
sys.path.append(PROJECT_ROOT)

# Import our custom modules
from mtl import UnifiedMultiTaskModel
from triplets_data_loader import parse_filename, get_posture_number
from mtl.classification_metrics import ClassificationMetricsCalculator
from mtl.verification_reconstruction_metrics import VerificationMetricsCalculator, ReconstructionMetricsCalculator

class TripletsTestDataset(Dataset):
    """Test dataset for triplets inference"""
    
    def __init__(self, csv_path, transform=None, target_transform=None, input_size=224, data_root="/home/noam/data_fixed/faceData_cropped_64x64"):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.target_transform = target_transform
        self.input_size = input_size
        self.data_root = data_root
        
        print(f"TripletsTestDataset initialized with {len(self.df)} samples")
        print(f"Input size: {input_size}x{input_size}")
        print(f"Data root: {data_root}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        try:
            # Load triplet images as grayscale with correct paths
            anchor_path = os.path.join(self.data_root, row['anchor'])
            positive_path = os.path.join(self.data_root, row['positive'])
            negative_path = os.path.join(self.data_root, row['negative'])
            
            anchor_img = Image.open(anchor_path).convert('L')
            positive_img = Image.open(positive_path).convert('L')
            negative_img = Image.open(negative_path).convert('L')
            
            # Extract posture class from anchor filename
            identity, posture, number = parse_filename(row['anchor'])
            posture_class = get_posture_number(posture) if posture else 0
            
            # Apply transforms
            if self.transform:
                anchor_tensor = self.transform(anchor_img)
                positive_tensor = self.transform(positive_img)
                negative_tensor = self.transform(negative_img)
            else:
                # Default transform for grayscale
                default_transform = transforms.Compose([
                    transforms.Resize((self.input_size, self.input_size)),
                    transforms.ToTensor(),  # Keep [0,1] range for THz data
                ])
                anchor_tensor = default_transform(anchor_img)
                positive_tensor = default_transform(positive_img)
                negative_tensor = default_transform(negative_img)
            
            # For reconstruction: anchor (concealed) -> positive (unconcealed)
            reconstruction_input = anchor_tensor
            reconstruction_target = positive_tensor if self.target_transform is None else self.target_transform(positive_img)
            
            return {
                # Face verification triplets
                'anchor': anchor_tensor,
                'positive': positive_tensor,
                'negative': negative_tensor,
                
                # Head posture classification
                'posture_class': torch.tensor(posture_class, dtype=torch.long),
                
                # Image reconstruction (concealed -> unconcealed)
                'reconstruction_input': reconstruction_input,
                'reconstruction_target': reconstruction_target,
                
                # Metadata for visualization
                'identity': identity,
                'posture': posture,
                'idx': idx,
                'anchor_path': row['anchor'],
                'positive_path': row['positive'],
                'negative_path': row['negative']
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to load sample at index {idx}: {e}")

def create_hstack_visualization(images, titles, save_path, task_name, sample_idx, metrics_text):
    """Create horizontal stack visualization for a sample"""
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    if len(images) == 1:
        axes = [axes]
    
    for i, (img, title) in enumerate(zip(images, titles)):
        # Convert tensor to numpy if needed
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()
        
        # Handle different image shapes
        if img.ndim == 3:
            if img.shape[0] == 1:  # Single channel
                img = img.squeeze(0)
            elif img.shape[0] == 3:  # RGB
                img = img.transpose(1, 2, 0)
        
        # Data is already in [0,1] range from THz preprocessing
        img = np.clip(img, 0, 1)
        
        # Handle size mismatches by resizing smaller images
        if i > 0 and hasattr(img, 'shape') and len(img.shape) == 2:
            # Get reference size from first image
            ref_img = images[0]
            if torch.is_tensor(ref_img):
                ref_img = ref_img.detach().cpu().numpy()
            if ref_img.ndim == 3 and ref_img.shape[0] == 1:
                ref_img = ref_img.squeeze(0)
            # Data is already in [0,1] range from THz preprocessing
            ref_img = np.clip(ref_img, 0, 1)
            
            # Resize if needed
            if img.shape != ref_img.shape:
                from PIL import Image as PILImage
                img_pil = PILImage.fromarray((img * 255).astype(np.uint8), mode='L')
                img_pil = img_pil.resize(ref_img.shape[::-1], PILImage.BILINEAR)
                img = np.array(img_pil) / 255.0
        
        try:
            axes[i].imshow(img, cmap='gray' if img.ndim == 2 else None)
            axes[i].set_title(title, fontsize=10)
            axes[i].axis('off')
        except Exception as e:
            print(f"Warning: Failed to display image {i} with shape {img.shape}: {e}")
            axes[i].text(0.5, 0.5, f'Display Error\nShape: {img.shape}', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(title, fontsize=10)
            axes[i].axis('off')
    
    # Add metrics text as suptitle
    # plt.suptitle(f'{task_name} - Sample {sample_idx}\n{metrics_text}', fontsize=12, y=0.95)
    plt.tight_layout()

    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def calculate_verification_metrics(anchor_emb, positive_emb, negative_emb, margin=0.2, distance='cosine'):
    """Calculate verification metrics using cosine or euclidean distance"""
    if distance == 'cosine':
        # Cosine distance (1 - cosine_similarity)
        pos_dist = 1 - F.cosine_similarity(anchor_emb, positive_emb, dim=1)
        neg_dist = 1 - F.cosine_similarity(anchor_emb, negative_emb, dim=1)
    else:  # euclidean
        pos_dist = F.pairwise_distance(anchor_emb, positive_emb, p=2)
        neg_dist = F.pairwise_distance(anchor_emb, negative_emb, p=2)
    
    # Verification accuracy: positive should be closer than negative
    ver_correct = (pos_dist < neg_dist).float()
    
    # Triplet loss margin satisfaction
    margin_satisfied = (pos_dist + margin < neg_dist).float()
    
    return {
        'pos_dist': pos_dist.mean().item(),
        'neg_dist': neg_dist.mean().item(),
        'accuracy': ver_correct.mean().item(),
        'margin_satisfied': margin_satisfied.mean().item(),
        'margin': (neg_dist - pos_dist).mean().item()
    }

def calculate_pixel_metrics(pred, target):
    """Calculate pixel-level metrics for reconstruction and SR"""
    # Ensure pred and target have the same dimensions
    if pred.shape != target.shape:
        # Resize target to match prediction size
        target_resized = F.interpolate(target, size=pred.shape[-2:], mode='bilinear', align_corners=False)
    else:
        target_resized = target
    
    # Data is already in valid [0, 1] range for SSIM from THz preprocessing
    pred_norm = torch.clamp(pred, 0, 1)
    target_norm = torch.clamp(target_resized, 0, 1)
    
    # SSIM
    ssim_score = ssim(pred_norm, target_norm, data_range=1.0).item()
    
    # PSNR (using normalized values)
    mse = F.mse_loss(pred_norm, target_norm)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8)).item()  # Add small epsilon to avoid log(0)
    
    # L1 loss (using original values)
    l1_loss = F.l1_loss(pred, target_resized).item()
    
    return {
        'ssim': ssim_score,
        'psnr': psnr,
        'l1_loss': l1_loss
    }

def run_triplets_inference(checkpoint_path, test_csv_path, output_dir, input_size=224, max_samples=50):
    """Run inference on triplets test data and generate comprehensive results"""
    
    # Create output directories with absolute paths
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    base_output_dir = os.path.abspath(os.path.join(output_dir, f'triplets_{input_size}_{timestamp}'))
    os.makedirs(base_output_dir, exist_ok=True)

    # Task-specific output directories
    task_dirs = {
        'verification': os.path.join(base_output_dir, 'verification'),
        'classification': os.path.join(base_output_dir, 'classification'),
        'reconstruction': os.path.join(base_output_dir, 'reconstruction')
    }

    # Ensure all task directories exist
    for task_dir in task_dirs.values():
        os.makedirs(task_dir, exist_ok=True)

    print(f"Output directory: {base_output_dir}")
    
    # Setup data transforms
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),  # Keep [0,1] range for THz data
    ])
    
    target_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),  # Keep [0,1] range for THz data
    ])
    
    # Create dataset and dataloader
    test_dataset = TripletsTestDataset(
        csv_path=test_csv_path,
        transform=transform,
        target_transform=target_transform,
        input_size=input_size
    )
    
    # Optimized batch processing for faster inference
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build the unified MTL network
    model = UnifiedMultiTaskModel(
        num_classes=5,
        embed_dim=128,
        backbone_type='custom',
        posture_ids=[0, 1, 2, 3, 4],
        input_size=input_size,
        dropout_rate=0.5,
        pretrained=True
    )
    model.to(device)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model state dict directly")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None
    
    model.eval()

    # Initialize enhanced metrics calculators
    cls_metrics_calculator = ClassificationMetricsCalculator(
        num_classes=5,
        class_names=['Front', 'Left', 'Right', 'Up', 'Down']
    )
    ver_metrics_calculator = VerificationMetricsCalculator()
    recon_metrics_calculator = ReconstructionMetricsCalculator(
        num_classes=5,
        class_names=['Front', 'Left', 'Right', 'Up', 'Down']
    )

    # Initialize metrics storage
    all_metrics = []
    task_metrics = {
        'verification': {'total_accuracy': 0, 'total_margin_satisfied': 0, 'count': 0,
                        'pos_dists': [], 'neg_dists': [], 'margins': []},
        'classification': {'correct': 0, 'total': 0},
        'reconstruction': {'ssim_scores': [], 'psnr_scores': [], 'l1_losses': []}
    }

    # Per-identity and per-posture metrics tracking
    per_identity_metrics = {}
    per_posture_metrics = {i: {
        'verification': {'accuracy': [], 'margin_satisfied': [], 'pos_dists': [], 'neg_dists': [], 'margins': []},
        'classification': {'correct': 0, 'total': 0},
        'reconstruction': {'ssim': [], 'psnr': [], 'l1': []}
    } for i in range(5)}
    
    print(f"Starting inference on {min(len(test_dataset), max_samples)} samples...")

    # Process samples in batches
    total_samples_processed = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Get actual batch size (may be smaller for last batch)
            batch_size = batch['anchor'].size(0)

            # Check if we've processed enough samples
            if total_samples_processed >= max_samples:
                break

            # Move batch to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device, non_blocking=True)

            # --- Unified MTL Forward Pass ---
            # Single forward pass through unified model for all tasks
            outputs = model(
                batch['anchor'],
                batch['positive'],
                batch['negative'],
                batch['posture_class']  # anchor posture ids
            )

            # Restructure outputs for compatibility with existing inference code
            unified_outputs = {
                'anchor_embedding': outputs['anchor_embedding'],
                'positive_embedding': outputs['positive_embedding'],
                'negative_embedding': outputs['negative_embedding'],
                'anchor_posture_logits': outputs['anchor_posture_logits'],
                'reconstruction_outputs': {'full': outputs['reconstructed_image']} # Match expected structure
            }
            outputs = unified_outputs
            # --- End of Unified MTL Logic ---

            # Update enhanced metrics calculators (batch-level)
            cls_metrics_calculator.update_batch(
                outputs['anchor_posture_logits'],
                batch['posture_class'],
                outputs['anchor_embedding']
            )

            ver_metrics_calculator.update_batch(
                outputs['anchor_embedding'],
                outputs['positive_embedding'],
                outputs['negative_embedding'],
                batch.get('identity', None)
            )

            # Calculate SSIM and PSNR for reconstruction batch (per-sample)
            recon_output = outputs['reconstruction_outputs']['full']
            recon_target = batch['reconstruction_target']
            if recon_output.shape != recon_target.shape:
                recon_target_resized = F.interpolate(recon_target, size=recon_output.shape[-2:], mode='bilinear', align_corners=False)
            else:
                recon_target_resized = recon_target

            # Compute per-sample SSIM
            batch_ssim_list = []
            for i in range(batch_size):
                sample_ssim = ssim(
                    torch.clamp(recon_output[i:i+1], 0, 1),
                    torch.clamp(recon_target_resized[i:i+1], 0, 1),
                    data_range=1.0
                )
                batch_ssim_list.append(sample_ssim.item())
            batch_ssim = torch.tensor(batch_ssim_list, device=recon_output.device)

            # Compute per-sample PSNR
            batch_mse = F.mse_loss(recon_output, recon_target_resized, reduction='none').mean(dim=[1,2,3])
            batch_psnr = 20 * torch.log10(1.0 / torch.sqrt(batch_mse + 1e-8))

            recon_metrics_calculator.update_batch(
                batch_ssim,
                batch_psnr,
                batch['posture_class']
            )

            # Process each sample in the batch
            for i in range(batch_size):
                sample_idx = total_samples_processed + i

                if sample_idx >= max_samples:
                    break

                # Calculate metrics for this sample
                sample_metrics = {'sample_idx': sample_idx}

                # 1. Face Verification Task
                if all(k in outputs for k in ['anchor_embedding', 'positive_embedding', 'negative_embedding']):
                    ver_metrics = calculate_verification_metrics(
                        outputs['anchor_embedding'][i:i+1],
                        outputs['positive_embedding'][i:i+1],
                        outputs['negative_embedding'][i:i+1]
                    )

                    sample_metrics.update({f'verification_{k}': v for k, v in ver_metrics.items()})

                    # Update task metrics
                    task_metrics['verification']['total_accuracy'] += ver_metrics['accuracy']
                    task_metrics['verification']['total_margin_satisfied'] += ver_metrics['margin_satisfied']
                    task_metrics['verification']['count'] += 1
                    task_metrics['verification']['pos_dists'].append(ver_metrics['pos_dist'])
                    task_metrics['verification']['neg_dists'].append(ver_metrics['neg_dist'])
                    task_metrics['verification']['margins'].append(ver_metrics['margin'])

                    # Verification visualization
                    if sample_idx < 500:
                        ver_images = [
                            batch['anchor'][i].cpu(),
                            batch['positive'][i].cpu(),
                            batch['negative'][i].cpu()
                        ]
                        ver_titles = [
                            f"Anchor ({batch['identity'][i]})",
                            f"Positive ({batch['identity'][i]})",
                            f"Negative"
                        ]
                        ver_metrics_text = (f"Accuracy: {ver_metrics['accuracy']:.3f} | "
                                          f"Pos Dist: {ver_metrics['pos_dist']:.3f} | "
                                          f"Neg Dist: {ver_metrics['neg_dist']:.3f} | "
                                          f"Margin: {ver_metrics['margin']:.3f}")

                        create_hstack_visualization(
                            ver_images, ver_titles,
                            os.path.join(task_dirs['verification'], f'sample_{sample_idx:03d}.jpg'),
                            'Face Verification', sample_idx, None
                        )

                # 2. Head Posture Classification Task
                if 'anchor_posture_logits' in outputs and 'posture_class' in batch:
                    pred_class = outputs['anchor_posture_logits'][i].argmax(dim=0)
                    true_class = batch['posture_class'][i]
                    is_correct = (pred_class == true_class).item()
                
                    sample_metrics['classification_accuracy'] = float(is_correct)
                    task_metrics['classification']['correct'] += is_correct
                    task_metrics['classification']['total'] += 1

                    # Classification visualization
                    if sample_idx < 500:
                        cls_images = [
                            batch['anchor'][i].cpu(),
                            batch['positive'][i].cpu(),
                            batch['negative'][i].cpu()
                        ]
                        cls_titles = [
                            f"Anchor - Pred: {pred_class.item()}, GT: {true_class.item()}",
                            f"Positive (same posture: {true_class.item()})",
                            f"Negative"
                        ]
                        cls_metrics_text = (f"Prediction: {'✓' if is_correct else '✗'} | "
                                          f"Predicted: {pred_class.item()} | "
                                          f"Ground Truth: {true_class.item()}")

                        create_hstack_visualization(
                            cls_images, cls_titles,
                            os.path.join(task_dirs['classification'], f'sample_{sample_idx:03d}.jpg'),
                            'Posture Classification', sample_idx, None
                        )

                # 3. Image Reconstruction Task
                if 'reconstruction_outputs' in outputs and 'reconstruction_target' in batch:
                    recon_output = outputs['reconstruction_outputs']['full'][i:i+1]
                    recon_target = batch['reconstruction_target'][i:i+1]

                    recon_metrics = calculate_pixel_metrics(recon_output, recon_target)
                    sample_metrics.update({f'reconstruction_{k}': v for k, v in recon_metrics.items()})

                    # Update task metrics
                    task_metrics['reconstruction']['ssim_scores'].append(recon_metrics['ssim'])
                    task_metrics['reconstruction']['psnr_scores'].append(recon_metrics['psnr'])
                    task_metrics['reconstruction']['l1_losses'].append(recon_metrics['l1_loss'])

                    # Reconstruction visualization
                    if sample_idx < 500:
                        recon_images = [
                            batch['reconstruction_input'][i].cpu(),
                            recon_output[0].cpu(),
                            recon_target[0].cpu()
                        ]
                        recon_titles = [
                            "Input (Concealed)",
                            "Reconstruction",
                            "Target (Unconcealed)"
                        ]
                        recon_metrics_text = (f"SSIM: {recon_metrics['ssim']:.3f} | "
                                            f"PSNR: {recon_metrics['psnr']:.1f}dB | "
                                            f"L1: {recon_metrics['l1_loss']:.3f}")

                        create_hstack_visualization(
                            recon_images, recon_titles,
                            os.path.join(task_dirs['reconstruction'], f'sample_{sample_idx:03d}.jpg'),
                            'Image Reconstruction', sample_idx, None
                        )

                # Extract identity and posture metadata
                identity = batch['identity'][i] if isinstance(batch['identity'], (list, tuple)) else batch['identity'][i].item() if torch.is_tensor(batch['identity']) else batch['identity']
                posture_class_val = batch['posture_class'][i].item() if torch.is_tensor(batch['posture_class']) else batch['posture_class'][i]

                # Add metadata to sample metrics
                sample_metrics['identity'] = identity
                sample_metrics['posture'] = posture_class_val

                # Track per-identity metrics
                if identity not in per_identity_metrics:
                    per_identity_metrics[identity] = {
                        'verification': {'accuracy': [], 'margin_satisfied': [], 'pos_dists': [], 'neg_dists': [], 'margins': []},
                        'classification': {'correct': 0, 'total': 0},
                        'reconstruction': {'ssim': [], 'psnr': [], 'l1': []}
                    }

                # Add verification metrics to per-identity tracking
                if 'verification_accuracy' in sample_metrics:
                    per_identity_metrics[identity]['verification']['accuracy'].append(sample_metrics['verification_accuracy'])
                    per_identity_metrics[identity]['verification']['margin_satisfied'].append(sample_metrics['verification_margin_satisfied'])
                    per_identity_metrics[identity]['verification']['pos_dists'].append(sample_metrics['verification_pos_dist'])
                    per_identity_metrics[identity]['verification']['neg_dists'].append(sample_metrics['verification_neg_dist'])
                    per_identity_metrics[identity]['verification']['margins'].append(sample_metrics['verification_margin'])

                # Add classification metrics to per-identity tracking
                if 'classification_accuracy' in sample_metrics:
                    per_identity_metrics[identity]['classification']['total'] += 1
                    if sample_metrics['classification_accuracy'] == 1.0:
                        per_identity_metrics[identity]['classification']['correct'] += 1

                # Add reconstruction metrics to per-identity tracking
                if 'reconstruction_ssim' in sample_metrics:
                    per_identity_metrics[identity]['reconstruction']['ssim'].append(sample_metrics['reconstruction_ssim'])
                    per_identity_metrics[identity]['reconstruction']['psnr'].append(sample_metrics['reconstruction_psnr'])
                    per_identity_metrics[identity]['reconstruction']['l1'].append(sample_metrics['reconstruction_l1_loss'])

                # Track per-posture metrics (only for valid posture classes)
                if 0 <= posture_class_val < 5:
                    # Add verification metrics to per-posture tracking
                    if 'verification_accuracy' in sample_metrics:
                        per_posture_metrics[posture_class_val]['verification']['accuracy'].append(sample_metrics['verification_accuracy'])
                        per_posture_metrics[posture_class_val]['verification']['margin_satisfied'].append(sample_metrics['verification_margin_satisfied'])
                        per_posture_metrics[posture_class_val]['verification']['pos_dists'].append(sample_metrics['verification_pos_dist'])
                        per_posture_metrics[posture_class_val]['verification']['neg_dists'].append(sample_metrics['verification_neg_dist'])
                        per_posture_metrics[posture_class_val]['verification']['margins'].append(sample_metrics['verification_margin'])

                    # Add classification metrics to per-posture tracking
                    if 'classification_accuracy' in sample_metrics:
                        per_posture_metrics[posture_class_val]['classification']['total'] += 1
                        if sample_metrics['classification_accuracy'] == 1.0:
                            per_posture_metrics[posture_class_val]['classification']['correct'] += 1

                    # Add reconstruction metrics to per-posture tracking
                    if 'reconstruction_ssim' in sample_metrics:
                        per_posture_metrics[posture_class_val]['reconstruction']['ssim'].append(sample_metrics['reconstruction_ssim'])
                        per_posture_metrics[posture_class_val]['reconstruction']['psnr'].append(sample_metrics['reconstruction_psnr'])
                        per_posture_metrics[posture_class_val]['reconstruction']['l1'].append(sample_metrics['reconstruction_l1_loss'])

                all_metrics.append(sample_metrics)

                if (sample_idx + 1) % 100 == 0:
                    print(f"Processed {sample_idx + 1} samples...")

            # Update total samples processed counter
            total_samples_processed += batch_size

    print(f"\nProcessed {len(all_metrics)} samples")

    # Compute enhanced metrics
    enhanced_cls_metrics = cls_metrics_calculator.compute_epoch_metrics()
    enhanced_ver_metrics = ver_metrics_calculator.compute_epoch_metrics()
    enhanced_recon_metrics = recon_metrics_calculator.compute_epoch_metrics()

    # Calculate summary metrics
    summary_metrics = {}

    # Verification metrics
    if task_metrics['verification']['count'] > 0:
        summary_metrics['verification_accuracy'] = task_metrics['verification']['total_accuracy'] / task_metrics['verification']['count']
        summary_metrics['verification_margin_satisfied'] = task_metrics['verification']['total_margin_satisfied'] / task_metrics['verification']['count']
        summary_metrics['verification_avg_pos_dist'] = np.mean(task_metrics['verification']['pos_dists'])
        summary_metrics['verification_avg_neg_dist'] = np.mean(task_metrics['verification']['neg_dists'])
        summary_metrics['verification_avg_margin'] = np.mean(task_metrics['verification']['margins'])

    # Classification metrics
    if task_metrics['classification']['total'] > 0:
        summary_metrics['classification_accuracy'] = task_metrics['classification']['correct'] / task_metrics['classification']['total']

    # Reconstruction metrics
    if task_metrics['reconstruction']['ssim_scores']:
        summary_metrics['reconstruction_ssim'] = np.mean(task_metrics['reconstruction']['ssim_scores'])
        summary_metrics['reconstruction_psnr'] = np.mean(task_metrics['reconstruction']['psnr_scores'])
        summary_metrics['reconstruction_l1'] = np.mean(task_metrics['reconstruction']['l1_losses'])

    # Add enhanced metrics to summary (top 5 only)
    summary_metrics['cls_conf_correct'] = enhanced_cls_metrics.get('cls_conf_correct', 0.0)
    summary_metrics['cls_f1_macro'] = enhanced_cls_metrics.get('cls_f1_macro', 0.0)
    summary_metrics['ver_distance_margin'] = enhanced_ver_metrics.get('ver_distance_margin', 0.0)
    summary_metrics['recon_ssim_mean'] = enhanced_recon_metrics.get('recon_ssim_mean', 0.0)
    summary_metrics['recon_hardest_posture_id'] = enhanced_recon_metrics.get('recon_hardest_posture_id', 0)

    # Calculate per-identity summary statistics
    per_identity_summary = []
    for identity, metrics in sorted(per_identity_metrics.items()):
        identity_row = {'identity': identity, 'sample_count': 0}

        # Verification metrics
        if metrics['verification']['accuracy']:
            identity_row['ver_accuracy'] = np.mean(metrics['verification']['accuracy'])
            identity_row['ver_margin_satisfied'] = np.mean(metrics['verification']['margin_satisfied'])
            identity_row['ver_avg_pos_dist'] = np.mean(metrics['verification']['pos_dists'])
            identity_row['ver_avg_neg_dist'] = np.mean(metrics['verification']['neg_dists'])
            identity_row['ver_avg_margin'] = np.mean(metrics['verification']['margins'])
            identity_row['sample_count'] = len(metrics['verification']['accuracy'])

        # Classification metrics
        if metrics['classification']['total'] > 0:
            identity_row['cls_accuracy'] = metrics['classification']['correct'] / metrics['classification']['total']
            identity_row['cls_samples'] = metrics['classification']['total']

        # Reconstruction metrics
        if metrics['reconstruction']['ssim']:
            identity_row['rec_ssim'] = np.mean(metrics['reconstruction']['ssim'])
            identity_row['rec_psnr'] = np.mean(metrics['reconstruction']['psnr'])
            identity_row['rec_l1'] = np.mean(metrics['reconstruction']['l1'])

        per_identity_summary.append(identity_row)

    # Calculate per-posture summary statistics
    per_posture_summary = []
    posture_names = {0: 'Front', 1: 'Left', 2: 'Right', 3: 'Up', 4: 'Down'}
    for posture_id in range(5):
        metrics = per_posture_metrics[posture_id]
        posture_row = {'posture_id': posture_id, 'posture_name': posture_names[posture_id], 'sample_count': 0}

        # Verification metrics
        if metrics['verification']['accuracy']:
            posture_row['ver_accuracy'] = np.mean(metrics['verification']['accuracy'])
            posture_row['ver_margin_satisfied'] = np.mean(metrics['verification']['margin_satisfied'])
            posture_row['ver_avg_pos_dist'] = np.mean(metrics['verification']['pos_dists'])
            posture_row['ver_avg_neg_dist'] = np.mean(metrics['verification']['neg_dists'])
            posture_row['ver_avg_margin'] = np.mean(metrics['verification']['margins'])
            posture_row['sample_count'] = len(metrics['verification']['accuracy'])

        # Classification metrics
        if metrics['classification']['total'] > 0:
            posture_row['cls_accuracy'] = metrics['classification']['correct'] / metrics['classification']['total']
            posture_row['cls_samples'] = metrics['classification']['total']

        # Reconstruction metrics
        if metrics['reconstruction']['ssim']:
            posture_row['rec_ssim'] = np.mean(metrics['reconstruction']['ssim'])
            posture_row['rec_psnr'] = np.mean(metrics['reconstruction']['psnr'])
            posture_row['rec_l1'] = np.mean(metrics['reconstruction']['l1'])

        per_posture_summary.append(posture_row)


    # Save results.txt
    results_txt_path = os.path.join(base_output_dir, 'results.txt')
    with open(results_txt_path, 'w') as f:
        f.write("Triplets Multi-Task Model Inference Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Test Data: {test_csv_path}\n")
        f.write(f"Input Size: {input_size}x{input_size}\n")
        f.write(f"Number of samples: {len(all_metrics)}\n\n")
        
        f.write("Task Performance Summary:\n")
        f.write("-" * 30 + "\n")
        
        if 'verification_accuracy' in summary_metrics:
            f.write(f"Face Verification:\n")
            f.write(f"  Accuracy: {summary_metrics['verification_accuracy']:.4f}\n")
            f.write(f"  Margin Satisfied: {summary_metrics['verification_margin_satisfied']:.4f}\n")
            f.write(f"  Avg Positive Distance: {summary_metrics['verification_avg_pos_dist']:.4f}\n")
            f.write(f"  Avg Negative Distance: {summary_metrics['verification_avg_neg_dist']:.4f}\n")
            f.write(f"  Avg Margin: {summary_metrics['verification_avg_margin']:.4f}\n\n")
        
        if 'classification_accuracy' in summary_metrics:
            f.write(f"Posture Classification:\n")
            f.write(f"  Accuracy: {summary_metrics['classification_accuracy']:.4f}\n\n")
        
        if 'reconstruction_ssim' in summary_metrics:
            f.write(f"Image Reconstruction:\n")
            f.write(f"  SSIM: {summary_metrics['reconstruction_ssim']:.4f}\n")
            f.write(f"  PSNR: {summary_metrics['reconstruction_psnr']:.2f} dB\n")
            f.write(f"  L1 Loss: {summary_metrics['reconstruction_l1']:.4f}\n\n")

        # Enhanced Classification Metrics
        f.write("ENHANCED CLASSIFICATION METRICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"cls_conf_correct: {enhanced_cls_metrics.get('cls_conf_correct', 0.0):.4f}\n")
        f.write(f"cls_conf_correct_std: {enhanced_cls_metrics.get('cls_conf_correct_std', 0.0):.4f}\n")
        f.write(f"cls_conf_error: {enhanced_cls_metrics.get('cls_conf_error', 0.0):.4f}\n")
        f.write(f"cls_conf_error_std: {enhanced_cls_metrics.get('cls_conf_error_std', 0.0):.4f}\n")
        f.write(f"cls_entropy: {enhanced_cls_metrics.get('cls_entropy', 0.0):.4f}\n")
        f.write(f"cls_entropy_std: {enhanced_cls_metrics.get('cls_entropy_std', 0.0):.4f}\n")
        f.write(f"cls_silhouette: {enhanced_cls_metrics.get('cls_silhouette', 0.0):.4f}\n")
        f.write(f"cls_f1_Front: {enhanced_cls_metrics.get('cls_f1_Front', 0.0):.4f}\n")
        f.write(f"cls_f1_Up: {enhanced_cls_metrics.get('cls_f1_Up', 0.0):.4f}\n")
        f.write(f"cls_f1_Down: {enhanced_cls_metrics.get('cls_f1_Down', 0.0):.4f}\n")
        f.write(f"cls_f1_Left: {enhanced_cls_metrics.get('cls_f1_Left', 0.0):.4f}\n")
        f.write(f"cls_f1_Right: {enhanced_cls_metrics.get('cls_f1_Right', 0.0):.4f}\n")
        f.write(f"cls_f1_macro: {enhanced_cls_metrics.get('cls_f1_macro', 0.0):.4f}\n")
        f.write(f"cls_f1_weighted: {enhanced_cls_metrics.get('cls_f1_weighted', 0.0):.4f}\n\n")

        # Enhanced Verification Metrics
        f.write("VERIFICATION METRICS\n")
        f.write("=" * 60 + "\n")
        f.write("Verification Distance Statistics\n")
        f.write("=" * 60 + "\n")
        f.write(f"ver_pos_dist_mean: {enhanced_ver_metrics.get('ver_pos_dist_mean', 0.0):.4f}\n")
        f.write(f"ver_pos_dist_std: {enhanced_ver_metrics.get('ver_pos_dist_std', 0.0):.4f}\n")
        f.write(f"ver_neg_dist_mean: {enhanced_ver_metrics.get('ver_neg_dist_mean', 0.0):.4f}\n")
        f.write(f"ver_neg_dist_std: {enhanced_ver_metrics.get('ver_neg_dist_std', 0.0):.4f}\n")
        f.write(f"ver_distance_margin: {enhanced_ver_metrics.get('ver_distance_margin', 0.0):.4f}\n")
        f.write(f"ver_normalized_margin: {enhanced_ver_metrics.get('ver_normalized_margin', 0.0):.4f}\n")
        f.write(f"ver_hard_pos_dist: {enhanced_ver_metrics.get('ver_hard_pos_dist', 0.0):.4f}\n")
        f.write(f"ver_hard_pos_ratio: {enhanced_ver_metrics.get('ver_hard_pos_ratio', 0.0):.4f}\n")
        f.write(f"ver_hard_neg_dist: {enhanced_ver_metrics.get('ver_hard_neg_dist', 0.0):.4f}\n")
        f.write(f"ver_hard_neg_ratio: {enhanced_ver_metrics.get('ver_hard_neg_ratio', 0.0):.4f}\n")
        f.write(f"ver_failure_rate: {enhanced_ver_metrics.get('ver_failure_rate', 0.0):.4f}\n")
        f.write(f"ver_dist_overlap: {enhanced_ver_metrics.get('ver_dist_overlap', 0.0):.4f}\n")
        f.write(f"ver_emb_norm_mean: {enhanced_ver_metrics.get('ver_emb_norm_mean', 0.0):.4f}\n")
        f.write(f"ver_emb_norm_std: {enhanced_ver_metrics.get('ver_emb_norm_std', 0.0):.4f}\n")
        f.write(f"ver_effective_dim: {enhanced_ver_metrics.get('ver_effective_dim', 0.0):.4f}\n")
        f.write(f"ver_variance_explained_95: {enhanced_ver_metrics.get('ver_variance_explained_95', 0.0):.4f}\n\n")

        # Enhanced Reconstruction Metrics
        f.write("RECONSTRUCTION METRICS\n")
        f.write("=" * 60 + "\n")
        f.write("Reconstruction Quality Statistics\n")
        f.write("=" * 60 + "\n")
        f.write(f"recon_ssim_mean: {enhanced_recon_metrics.get('recon_ssim_mean', 0.0):.4f}\n")
        f.write(f"recon_ssim_std: {enhanced_recon_metrics.get('recon_ssim_std', 0.0):.4f}\n")
        f.write(f"recon_ssim_min: {enhanced_recon_metrics.get('recon_ssim_min', 0.0):.4f}\n")
        f.write(f"recon_ssim_max: {enhanced_recon_metrics.get('recon_ssim_max', 0.0):.4f}\n")
        f.write(f"recon_psnr_mean: {enhanced_recon_metrics.get('recon_psnr_mean', 0.0):.4f}\n")
        f.write(f"recon_psnr_std: {enhanced_recon_metrics.get('recon_psnr_std', 0.0):.4f}\n")
        f.write(f"recon_psnr_min: {enhanced_recon_metrics.get('recon_psnr_min', 0.0):.4f}\n")
        f.write(f"recon_psnr_max: {enhanced_recon_metrics.get('recon_psnr_max', 0.0):.4f}\n")
        f.write(f"recon_ssim_cv: {enhanced_recon_metrics.get('recon_ssim_cv', 0.0):.4f}\n")
        f.write(f"recon_psnr_cv: {enhanced_recon_metrics.get('recon_psnr_cv', 0.0):.4f}\n")
        f.write(f"recon_ssim_Front: {enhanced_recon_metrics.get('recon_ssim_Front', 0.0):.4f}\n")
        f.write(f"recon_ssim_Up: {enhanced_recon_metrics.get('recon_ssim_Up', 0.0):.4f}\n")
        f.write(f"recon_ssim_Down: {enhanced_recon_metrics.get('recon_ssim_Down', 0.0):.4f}\n")
        f.write(f"recon_ssim_Left: {enhanced_recon_metrics.get('recon_ssim_Left', 0.0):.4f}\n")
        f.write(f"recon_ssim_Right: {enhanced_recon_metrics.get('recon_ssim_Right', 0.0):.4f}\n")
        f.write(f"recon_psnr_Front: {enhanced_recon_metrics.get('recon_psnr_Front', 0.0):.4f}\n")
        f.write(f"recon_psnr_Up: {enhanced_recon_metrics.get('recon_psnr_Up', 0.0):.4f}\n")
        f.write(f"recon_psnr_Down: {enhanced_recon_metrics.get('recon_psnr_Down', 0.0):.4f}\n")
        f.write(f"recon_psnr_Left: {enhanced_recon_metrics.get('recon_psnr_Left', 0.0):.4f}\n")
        f.write(f"recon_psnr_Right: {enhanced_recon_metrics.get('recon_psnr_Right', 0.0):.4f}\n")
        f.write(f"recon_hardest_posture_id: {enhanced_recon_metrics.get('recon_hardest_posture_id', 0.0):.4f}\n")
        posture_names = {0: 'Front', 1: 'Left', 2: 'Right', 3: 'Up', 4: 'Down'}
        hardest_id = int(enhanced_recon_metrics.get('recon_hardest_posture_id', 0))
        easiest_id = int(enhanced_recon_metrics.get('recon_easiest_posture_id', 0))
        f.write(f"recon_hardest_posture_name: {posture_names.get(hardest_id, 'Unknown')}\n")
        f.write(f"recon_easiest_posture_id: {enhanced_recon_metrics.get('recon_easiest_posture_id', 0.0):.4f}\n")
        f.write(f"recon_easiest_posture_name: {posture_names.get(easiest_id, 'Unknown')}\n")
        f.write(f"recon_ssim_class_range: {enhanced_recon_metrics.get('recon_ssim_class_range', 0.0):.4f}\n")
        f.write(f"recon_failure_rate_ssim: {enhanced_recon_metrics.get('recon_failure_rate_ssim', 0.0):.4f}\n")
        f.write(f"recon_excellent_rate_ssim: {enhanced_recon_metrics.get('recon_excellent_rate_ssim', 0.0):.4f}\n\n")
        
    
    # Save detailed results.csv
    results_csv_path = os.path.join(base_output_dir, 'results.csv')
    if all_metrics:
        df_results = pd.DataFrame(all_metrics)
        df_results.to_csv(results_csv_path, index=False)
    
    # Save summary metrics
    summary_csv_path = os.path.join(base_output_dir, 'summary.csv')
    with open(summary_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for key, value in summary_metrics.items():
            writer.writerow([key, f"{value:.4f}"])

    # Save per-identity results
    per_identity_csv_path = os.path.join(base_output_dir, 'per_identity_results.csv')
    if per_identity_summary:
        df_per_identity = pd.DataFrame(per_identity_summary)
        df_per_identity.to_csv(per_identity_csv_path, index=False)
        print(f"\nSaved per-identity results to: {per_identity_csv_path}")
        print(f"  Number of identities: {len(per_identity_summary)}")

    # Save per-posture results
    per_posture_csv_path = os.path.join(base_output_dir, 'per_posture_results.csv')
    if per_posture_summary:
        df_per_posture = pd.DataFrame(per_posture_summary)
        df_per_posture.to_csv(per_posture_csv_path, index=False)
        print(f"Saved per-posture results to: {per_posture_csv_path}")
        print(f"  Number of postures: {len(per_posture_summary)}")

    print(f"\nInference completed!")
    print(f"Results saved to: {base_output_dir}")
    print("\nSummary:")
    for key, value in summary_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    return base_output_dir, summary_metrics

def main():
    parser = argparse.ArgumentParser(description='Run unified MTL inference')
    parser.add_argument('--checkpoint', default='/home/noam/PycharmProjects/thz/runs/unified_mtl_bicubic_64_20251122_013235/checkpoints/best.pth',
                        help='Path to the unified MTL model checkpoint')
    parser.add_argument('--test_csv', default='datafiles_fixed_29_08_sterile/faceData_cropped_64x64/dBm/test_dataset.csv',
                       help='Path to test CSV file')
    parser.add_argument('--output_dir', default='test_runs', 
                       help='Output directory for results')
    parser.add_argument('--input_size', type=int, default=64, help='Input image size (64 for unified MTL)')
    parser.add_argument('--max_samples', type=int, default=2100, help='Maximum number of samples to process')
    
    args = parser.parse_args()
    
    # Verify working directory
    print(f"Working directory: {os.getcwd()}")
    print(f"Script location: {os.path.abspath(__file__)}")
    
    run_triplets_inference(
        checkpoint_path=args.checkpoint,
        test_csv_path=args.test_csv,
        output_dir=args.output_dir,
        input_size=args.input_size,
        max_samples=args.max_samples
    )

if __name__ == '__main__':
    main()