#!/usr/bin/env python3
"""
Privileged Information Multi-Task Learning Training Script (V2)

Trains a multi-task model using visible images as privileged information via teacher-student distillation.
The trained model can perform inference using only THz images.

Tasks:
1. Face verification (triplet loss)
2. Head posture classification (cross-entropy)
3. Image reconstruction (MoE with multiple losses)
Plus distillation from visible-guided teacher to THz-only student.
"""

import torch
import torch.optim as optim
import logging
import os
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from tqdm import tqdm
import csv
import argparse
import shutil
import sys

# Set working directory to the project root
PROJECT_ROOT = "/home/noam/PycharmProjects/thz"
os.chdir(PROJECT_ROOT)

# Add the project root to the path so we can import modules
sys.path.append(PROJECT_ROOT)

# Import V2 modules
from mtl.multimodal.V2.unified_architecture import PrivilegedDistillationModel
from mtl.multimodal.V2.unified_losses import create_privileged_distillation_loss
from mtl.multimodal.V2.unified_data_loader import get_multimodal_triplets_data_loaders
from mtl.classification_metrics import ClassificationMetricsCalculator
from mtl.verification_reconstruction_metrics import VerificationMetricsCalculator, ReconstructionMetricsCalculator


class PrivilegedTrainer:
    """Trainer for Privileged Information Multi-Task Learning"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        resolution = config['input_size']
        dataset_type = 'original' if resolution <= 56 else 'bicubic'
        self.run_dir = f"runs/privileged_mtl_{dataset_type}_{resolution}_{timestamp}"
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(f"{self.run_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{self.run_dir}/logs", exist_ok=True)
        os.makedirs(f"{self.run_dir}/plots", exist_ok=True)

        # Save config
        with open(f"{self.run_dir}/config.json", 'w') as f:
            json.dump(config, f, indent=2)

        # Setup logging
        log_file = f"{self.run_dir}/logs/privileged_mtl_training.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('privileged_mtl_training')

        # Initialize training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_ssim = 0.0
        self.train_losses = defaultdict(list)
        self.val_losses = defaultdict(list)
        self.val_metrics = defaultdict(list)
        self.batch_losses = defaultdict(list)  # Track batch-level losses for plotting
        self.batch_metrics = defaultdict(list)  # Track batch-level metrics (accuracy, SSIM, PSNR)
        self.global_step = 0  # Track global iteration count
        self.augmentation_samples = None  # Cache samples for augmentation visualization

        # Initialize CSV logging
        self.train_csv_path = f"{self.run_dir}/logs/train_logs.csv"
        self.val_csv_path = f"{self.run_dir}/logs/validation_logs.csv"
        self._initialize_csv_logging()
        self._save_code_files()

        self.logger.info(f"PrivilegedTrainer initialized")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Input size: {config['input_size']}")
        self.logger.info(f"Dataset type: {dataset_type}")
        self.logger.info(f"Run directory: {self.run_dir}")

    def _save_code_files(self):
        """Save code files for reproducibility"""
        code_dir = os.path.join(self.run_dir, 'code')
        os.makedirs(code_dir, exist_ok=True)

        # List of V2 code files to save
        v2_files = [
            'mtl/multimodal/V2/unified_architecture.py',
            'mtl/multimodal/V2/unified_losses.py',
            'mtl/multimodal/V2/unified_data_loader.py',
            'mtl/multimodal/V2/train_unified_mtl.py'
        ]

        # Save V2 code files
        for filename in v2_files:
            src_path = filename
            if os.path.exists(src_path):
                dst_path = os.path.join(code_dir, os.path.basename(filename))
                shutil.copy2(src_path, dst_path)
                self.logger.info(f"Saved code file: {filename}")

    def _initialize_csv_logging(self):
        """Initialize CSV files for logging"""
        # Training log headers (extended with distillation losses)
        train_headers = [
            'epoch', 'batch', 'total_loss',
            'verification_loss_student', 'verification_loss_teacher',
            'classification_loss_student', 'classification_loss_teacher',
            'feature_distill_loss', 'logit_distill_loss',
            'recon_total', 'recon_charbonnier', 'recon_ssim', 'recon_lpips',
            'learning_rate'
        ]
        with open(self.train_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(train_headers)

        # Validation log headers (extended with enhanced metrics)
        val_headers = [
            'epoch', 'total_loss',
            'verification_loss_student', 'verification_loss_teacher',
            'classification_loss_student', 'classification_loss_teacher',
            'feature_distill_loss', 'logit_distill_loss',
            'recon_total', 'recon_charbonnier', 'recon_ssim', 'recon_lpips',
            'verification_accuracy', 'classification_accuracy', 'ssim', 'psnr',
            # Top 5 enhanced metrics across all tasks
            'cls_conf_correct', 'cls_f1_macro', 'ver_distance_margin',
            'recon_ssim_mean', 'recon_hardest_posture_id'
        ]
        with open(self.val_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(val_headers)

    def setup_model_and_optimizer(self):
        """Setup model, loss function, and optimizer"""
        # Model
        self.model = PrivilegedDistillationModel(
            num_classes=self.config['num_classes'],
            embed_dim=self.config['embed_dim'],
            backbone_type=self.config['backbone_type'],
            posture_ids=self.config.get('posture_ids', [0, 1, 2, 3, 4]),
            input_size=self.config['input_size'],
            dropout_rate=self.config.get('dropout_rate', 0.5),
            pretrained=self.config.get('pretrained', True),
            cross_attention_heads=self.config.get('cross_attention_heads', 8),
            distill_dim=self.config.get('distill_dim', 256)
        ).to(self.device)

        # Apply teacher freezing if configured
        freeze_teacher = self.config.get('freeze_teacher', False)
        if freeze_teacher:
            self.model.freeze_teacher()
            self.logger.info("Teacher model frozen (requires_grad=False)")
        else:
            self.model.unfreeze_teacher()  # Ensure teacher is unfrozen (default)
            self.logger.info("Teacher model trainable (requires_grad=True)")

        # Get distillation configuration
        distill_full_triplet = self.config.get('distill_full_triplet', False)

        # Loss function
        self.criterion = create_privileged_distillation_loss(
            weights=self.config['loss_weights'],
            device=self.device,
            supcon_temperature=self.config.get('supcon_temperature', 0.07),
            cls_smoothing=self.config.get('classification_smoothing', 0.1),
            distill_temperature=self.config.get('distillation_temperature', 3.0),
            reconstruction_weights=self.config['loss_weights'].get('reconstruction_weights', {}),
            distill_full_triplet=distill_full_triplet
        )

        # Log distillation configuration
        if distill_full_triplet:
            self.logger.info("Distillation mode: Full triplet (anchor + positive + negative)")
        else:
            self.logger.info("Distillation mode: Anchor only")

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-4),
            betas=(0.9, 0.999)
        )

        # Scheduler
        if self.config.get('use_scheduler', True):
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=10
            )
        else:
            self.scheduler = None

        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        self.logger.info(f"Architecture: Dual Cross-Attention Teacher-Student Framework")
        self.logger.info(f"  - Visible-guided cross-attention: Q=Visible, K/V=THz ({self.model.cross_attention_heads} heads)")
        self.logger.info(f"  - THz-guided cross-attention: Q=THz, K/V=Visible ({self.model.cross_attention_heads} heads)")
        self.logger.info(f"  - Attention fusion: 2x{self.model.feature_dim}→{self.model.feature_dim}")

    def setup_data_loaders(self):
        """Setup multimodal data loaders"""
        train_val_csv = self.config.get('train_val_csv')
        data_dir = self.config.get('data_dir', '/home/noam/data_fixed/faceData_cropped_64x64')
        visible_data_dir = self.config.get('visible_data_dir', data_dir)

        # Create data loaders with visible image support
        self.train_loader, self.val_loader, _ = get_multimodal_triplets_data_loaders(
            train_val_csv=train_val_csv,
            input_size=self.config['input_size'],
            batch_size=self.config['batch_size'],
            num_workers=self.config.get('num_workers', 4),
            val_ratio=self.config.get('val_ratio', 0.15),
            random_seed=self.config['random_seed'],
            output_folder=self.run_dir,
            augmentation_mode=self.config.get('augmentation_mode', 'gentle'),
            data_dir=data_dir,
            visible_data_dir=visible_data_dir
        )

        self.logger.info(f"Data loaders created:")
        self.logger.info(f"  Train: {len(self.train_loader)} batches")
        self.logger.info(f"  Validation: {len(self.val_loader)} batches")

    def train_epoch(self):
        """Train for one epoch with privileged distillation"""
        self.model.train()
        epoch_losses = defaultdict(list)

        # Calculate validation frequency (configurable)
        val_epoch_fraction = self.config.get('validation_interval_epochs', 0.4)
        validation_interval = max(1, int(len(self.train_loader) * val_epoch_fraction))

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            anchor = batch['anchor'].to(self.device, non_blocking=True)
            positive = batch['positive'].to(self.device, non_blocking=True)
            negative = batch['negative'].to(self.device, non_blocking=True)
            anchor_posture_ids = batch['anchor_posture_class'].to(self.device, non_blocking=True)

            # Get all three visible images (may be None for some samples)
            visible_concealed = batch.get('visible_concealed', None)
            visible_unconcealed = batch.get('visible_unconcealed', None)
            visible_negative = batch.get('visible_negative', None)

            # Move visible images to device if available
            if visible_concealed is not None:
                visible_concealed = visible_concealed.to(self.device, non_blocking=True)
            if visible_unconcealed is not None:
                visible_unconcealed = visible_unconcealed.to(self.device, non_blocking=True)
            if visible_negative is not None:
                visible_negative = visible_negative.to(self.device, non_blocking=True)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with all three visible images
            outputs = self.model(
                anchor, positive, negative,
                anchor_posture_ids,
                visible_concealed=visible_concealed,
                visible_unconcealed=visible_unconcealed,
                visible_negative=visible_negative
            )

            # Compute loss
            total_loss, loss_log = self.criterion(outputs, batch)

            # Backward pass and optimization
            if not torch.isnan(total_loss) and not torch.isinf(total_loss):
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('grad_clip', 1.0))
                self.optimizer.step()

                # Log losses
                for key, value in loss_log.items():
                    epoch_losses[key].append(value)
                    self.batch_losses[key].append(value)

                # Compute and log train metrics
                with torch.no_grad():
                    # Verification accuracy
                    anchor_emb = outputs['anchor_embedding']
                    positive_emb = outputs['positive_embedding']
                    negative_emb = outputs['negative_embedding']
                    pos_dist = torch.norm(anchor_emb - positive_emb, p=2, dim=1)
                    neg_dist = torch.norm(anchor_emb - negative_emb, p=2, dim=1)
                    ver_acc = (pos_dist < neg_dist).float().mean().item()
                    self.batch_metrics['verification_accuracy'].append(ver_acc)

                    # Classification accuracy
                    pred = torch.argmax(outputs['anchor_posture_logits'], dim=1)
                    labels = anchor_posture_ids
                    cls_acc = (pred == labels).float().mean().item()
                    self.batch_metrics['classification_accuracy'].append(cls_acc)

                    # Reconstruction SSIM and PSNR
                    reconstructed = outputs['reconstructed_image']
                    target = positive
                    ssim = self._calculate_ssim(reconstructed, target).mean().item()
                    psnr = self._calculate_psnr(reconstructed, target).mean().item()
                    self.batch_metrics['ssim'].append(ssim)
                    self.batch_metrics['psnr'].append(psnr)

                # Increment global step counter
                self.global_step += 1

                # Log to CSV
                with open(self.train_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.current_epoch + 1, batch_idx + 1,
                        loss_log.get('total_loss', 0),
                        loss_log.get('verification_loss_student', 0),
                        loss_log.get('verification_loss_teacher', 0),
                        loss_log.get('classification_loss_student', 0),
                        loss_log.get('classification_loss_teacher', 0),
                        loss_log.get('feature_distill_loss', 0),
                        loss_log.get('logit_distill_loss', 0),
                        loss_log.get('recon_total', 0),
                        loss_log.get('recon_charbonnier', 0),
                        loss_log.get('recon_ssim', 0),
                        loss_log.get('recon_lpips', 0),
                        self.optimizer.param_groups[0]['lr']
                    ])

                # Save plots every 50 iterations
                if self.global_step % 50 == 0:
                    self._plot_and_save_metrics()  # Training losses
                    self._plot_train_metrics()  # Training evaluation metrics
                    self._visualize_augmentations(batch)  # Augmentation visualization

                # Save reconstruction visualization every 100 iterations
                if self.global_step % 100 == 0:
                    self._visualize_reconstructions(batch, outputs)
            else:
                self.logger.warning(f"Invalid loss detected, skipping batch")

            # Clear cached memory
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            # Update progress bar
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({
                    'total': f"{loss_log.get('total_loss', 0):.4f}",
                    'ver_s': f"{loss_log.get('verification_loss_student', 0):.4f}",
                    'cls_s': f"{loss_log.get('classification_loss_student', 0):.4f}",
                    'feat_d': f"{loss_log.get('feature_distill_loss', 0):.4f}",
                    'rec': f"{loss_log.get('recon_total', 0):.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })

            # Run validation every 0.2 epochs
            if (batch_idx + 1) % validation_interval == 0:
                self.logger.info(f"\nRunning mid-epoch validation at batch {batch_idx + 1}/{len(self.train_loader)}")

                # Run validation
                val_losses, val_metrics = self.validate_epoch()

                # Store losses and metrics
                for key, value in val_losses.items():
                    self.val_losses[key].append(value)
                for key, value in val_metrics.items():
                    self.val_metrics[key].append(value)

                # Plot validation metrics
                self._plot_validation_metrics()

                # Log validation results
                self.logger.info(
                    f"Mid-epoch validation - "
                    f"Val Loss: {val_losses['total_loss']:.4f}, "
                    f"Ver Acc: {val_metrics['verification_accuracy']:.4f}, "
                    f"Cls Acc: {val_metrics['classification_accuracy']:.4f}, "
                    f"SSIM: {val_metrics['ssim']:.4f}, "
                    f"PSNR: {val_metrics['psnr']:.2f}"
                )

                # Check if this is the best model so far
                if val_metrics['ssim'] > self.best_val_ssim:
                    self.best_val_ssim = val_metrics['ssim']
                    self.best_val_loss = val_losses['total_loss']
                    self.save_checkpoint(is_best=True)
                    self.logger.info(f"New best model! SSIM: {self.best_val_ssim:.4f}")

                # Update learning rate scheduler if enabled
                if self.scheduler:
                    self.scheduler.step(val_losses['total_loss'])

                # Set model back to training mode
                self.model.train()

        # Average epoch losses
        avg_losses = {key: np.mean(values) if values else 0 for key, values in epoch_losses.items()}
        return avg_losses

    def validate_epoch(self):
        """Validate for one epoch with enhanced classification metrics"""
        self.model.eval()
        epoch_losses = defaultdict(list)

        # Metrics tracking
        verification_correct = 0
        verification_total = 0
        classification_correct = 0
        classification_total = 0
        ssim_scores = []
        psnr_scores = []

        # Enhanced metrics calculators
        cls_metrics_calculator = ClassificationMetricsCalculator(
            num_classes=self.config['num_classes'],
            class_names=['Front', 'Up', 'Down', 'Left', 'Right']
        )
        ver_metrics_calculator = VerificationMetricsCalculator()
        recon_metrics_calculator = ReconstructionMetricsCalculator(
            num_classes=self.config['num_classes'],
            class_names=['Front', 'Up', 'Down', 'Left', 'Right']
        )

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                anchor = batch['anchor'].to(self.device, non_blocking=True)
                positive = batch['positive'].to(self.device, non_blocking=True)
                negative = batch['negative'].to(self.device, non_blocking=True)
                anchor_posture_ids = batch['anchor_posture_class'].to(self.device, non_blocking=True)

                # Get all three visible images
                visible_concealed = batch.get('visible_concealed', None)
                visible_unconcealed = batch.get('visible_unconcealed', None)
                visible_negative = batch.get('visible_negative', None)

                # Move visible images to device if available
                if visible_concealed is not None:
                    visible_concealed = visible_concealed.to(self.device, non_blocking=True)
                if visible_unconcealed is not None:
                    visible_unconcealed = visible_unconcealed.to(self.device, non_blocking=True)
                if visible_negative is not None:
                    visible_negative = visible_negative.to(self.device, non_blocking=True)

                # Forward pass
                outputs = self.model(
                    anchor, positive, negative,
                    anchor_posture_ids,
                    visible_concealed=visible_concealed,
                    visible_unconcealed=visible_unconcealed,
                    visible_negative=visible_negative
                )

                # Compute losses
                total_loss, loss_log = self.criterion(outputs, batch)

                for key, value in loss_log.items():
                    epoch_losses[key].append(value)

                # Calculate verification accuracy
                anchor_emb = outputs['anchor_embedding']
                positive_emb = outputs['positive_embedding']
                negative_emb = outputs['negative_embedding']

                pos_dist = torch.norm(anchor_emb - positive_emb, p=2, dim=1)
                neg_dist = torch.norm(anchor_emb - negative_emb, p=2, dim=1)
                verification_correct += (pos_dist < neg_dist).sum().item()
                verification_total += anchor.size(0)

                # Calculate classification accuracy (student path)
                for prefix in ['anchor', 'positive', 'negative']:
                    logits = outputs[f'{prefix}_posture_logits']
                    labels = batch[f'{prefix}_posture_class'].to(self.device)
                    pred = torch.argmax(logits, dim=1)
                    classification_correct += (pred == labels).sum().item()
                    classification_total += labels.size(0)

                # Enhanced classification metrics (use student path - anchor, positive, negative)
                # Collect all logits and labels from the triplet
                all_student_logits = torch.cat([
                    outputs['anchor_posture_logits'],
                    outputs['positive_posture_logits'],
                    outputs['negative_posture_logits']
                ], dim=0)
                all_labels = torch.cat([
                    batch['anchor_posture_class'],
                    batch['positive_posture_class'],
                    batch['negative_posture_class']
                ], dim=0).to(self.device)

                # Collect embeddings for silhouette score (use student embeddings)
                all_embeddings = torch.cat([
                    anchor_emb,
                    positive_emb,
                    negative_emb
                ], dim=0)

                # Update enhanced classification metrics calculator
                cls_metrics_calculator.update_batch(
                    all_student_logits,
                    all_labels,
                    all_embeddings
                )

                # Update verification metrics calculator
                ver_metrics_calculator.update_batch(
                    anchor_emb,
                    positive_emb,
                    negative_emb,
                    batch.get('identity', None)
                )

                # Calculate SSIM and PSNR
                reconstructed = outputs['reconstructed_image']
                target = positive

                ssim = self._calculate_ssim(reconstructed, target)
                ssim_scores.extend(ssim.cpu().numpy())

                psnr = self._calculate_psnr(reconstructed, target)
                psnr_scores.extend(psnr.cpu().numpy())

                # Update reconstruction metrics calculator
                recon_metrics_calculator.update_batch(
                    ssim,
                    psnr,
                    batch['anchor_posture_class']
                )

        # Average losses and metrics
        avg_losses = {key: np.mean(values) if values else 0 for key, values in epoch_losses.items()}

        verification_accuracy = verification_correct / verification_total if verification_total > 0 else 0
        classification_accuracy = classification_correct / classification_total if classification_total > 0 else 0
        avg_ssim = np.mean(ssim_scores) if ssim_scores else 0
        avg_psnr = np.mean(psnr_scores) if psnr_scores else 0

        # Compute all enhanced metrics
        enhanced_cls_metrics = cls_metrics_calculator.compute_epoch_metrics()
        enhanced_ver_metrics = ver_metrics_calculator.compute_epoch_metrics()
        enhanced_recon_metrics = recon_metrics_calculator.compute_epoch_metrics()

        # Log comprehensive report to file for detailed analysis
        cls_report = cls_metrics_calculator.get_classification_report()
        ver_report = ver_metrics_calculator.get_distance_statistics()
        recon_report = recon_metrics_calculator.get_reconstruction_report()

        with open(f"{self.run_dir}/logs/comprehensive_metrics_epoch_{self.current_epoch + 1}.txt", 'w') as f:
            f.write(f"Epoch {self.current_epoch + 1} - Comprehensive Metrics Report (Multi-Modal Model)\n")
            f.write("="*60 + "\n\n")

            f.write("CLASSIFICATION METRICS\n")
            f.write("="*60 + "\n")
            f.write(cls_report)
            f.write("\n\nEnhanced Classification Metrics:\n")
            f.write("-"*60 + "\n")
            for key, value in enhanced_cls_metrics.items():
                f.write(f"{key}: {value:.4f}\n")

            f.write("\n\nVERIFICATION METRICS\n")
            f.write("="*60 + "\n")
            f.write(ver_report)
            f.write("\nEnhanced Verification Metrics:\n")
            f.write("-"*60 + "\n")
            for key, value in enhanced_ver_metrics.items():
                f.write(f"{key}: {value:.4f}\n")

            f.write("\n\nRECONSTRUCTION METRICS\n")
            f.write("="*60 + "\n")
            f.write(recon_report)
            f.write("\nEnhanced Reconstruction Metrics:\n")
            f.write("-"*60 + "\n")
            for key, value in enhanced_recon_metrics.items():
                if isinstance(value, (int, float)):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")

        # Log to CSV (extended with enhanced metrics)
        with open(self.val_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.current_epoch + 1,
                avg_losses.get('total_loss', 0),
                avg_losses.get('verification_loss_student', 0),
                avg_losses.get('verification_loss_teacher', 0),
                avg_losses.get('classification_loss_student', 0),
                avg_losses.get('classification_loss_teacher', 0),
                avg_losses.get('feature_distill_loss', 0),
                avg_losses.get('logit_distill_loss', 0),
                avg_losses.get('recon_total', 0),
                avg_losses.get('recon_charbonnier', 0),
                avg_losses.get('recon_ssim', 0),
                avg_losses.get('recon_lpips', 0),
                verification_accuracy,
                classification_accuracy,
                avg_ssim,
                avg_psnr,
                # Top 5 enhanced metrics across all tasks
                enhanced_cls_metrics.get('cls_conf_correct', 0.0),
                enhanced_cls_metrics.get('cls_f1_macro', 0.0),
                enhanced_ver_metrics.get('ver_distance_margin', 0.0),
                enhanced_recon_metrics.get('recon_ssim_mean', 0.0),
                enhanced_recon_metrics.get('recon_hardest_posture_id', 0)
            ])

        # Return metrics including all enhanced metrics
        return avg_losses, {
            'verification_accuracy': verification_accuracy,
            'classification_accuracy': classification_accuracy,
            'ssim': avg_ssim,
            'psnr': avg_psnr,
            **enhanced_cls_metrics,  # Include all enhanced classification metrics
            **enhanced_ver_metrics,  # Include all enhanced verification metrics
            **enhanced_recon_metrics  # Include all enhanced reconstruction metrics
        }

    def _calculate_ssim(self, img1, img2):
        """Calculate SSIM between two images"""
        mu1 = img1.mean(dim=[2, 3], keepdim=True)
        mu2 = img2.mean(dim=[2, 3], keepdim=True)

        sigma1_sq = ((img1 - mu1) ** 2).mean(dim=[2, 3], keepdim=True)
        sigma2_sq = ((img2 - mu2) ** 2).mean(dim=[2, 3], keepdim=True)
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean(dim=[2, 3], keepdim=True)

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim.squeeze()

    def _calculate_psnr(self, img1, img2):
        """Calculate PSNR between two images"""
        mse = ((img1 - img2) ** 2).mean(dim=[1, 2, 3])
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_ssim': self.best_val_ssim,
            'config': self.config,
            'train_losses': dict(self.train_losses),
            'val_losses': dict(self.val_losses),
            'val_metrics': dict(self.val_metrics)
        }

        # Save latest checkpoint
        checkpoint_path = f"{self.run_dir}/checkpoints/latest.pth"
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = f"{self.run_dir}/checkpoints/best.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint at epoch {self.current_epoch + 1}")

    def _plot_and_save_metrics(self):
        """Generate and save plots for batch-level losses as latest.png"""
        plot_dir = f"{self.run_dir}/plots"

        # Skip if we don't have enough data yet
        if len(self.batch_losses.get('total_loss', [])) < 10:
            return

        # Create a combined figure with all metrics
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Training Progress (Step {self.global_step})', fontsize=16)

        # Plot 1: Total Loss (batch-level)
        ax1 = axes[0, 0]
        if 'total_loss' in self.batch_losses and self.batch_losses['total_loss']:
            ax1.plot(self.batch_losses['total_loss'], label='Total Loss', color='blue', alpha=0.3)
            # Add smoothed version (moving average)
            window = min(100, len(self.batch_losses['total_loss']) // 10)
            if window > 1:
                smoothed = np.convolve(self.batch_losses['total_loss'], np.ones(window)/window, mode='valid')
                ax1.plot(range(window-1, len(self.batch_losses['total_loss'])), smoothed,
                        label='Smoothed Total Loss', color='blue', linewidth=2)
        ax1.set_title('Total Loss (Batch-level)')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Task-specific Losses
        ax2 = axes[0, 1]
        if 'verification_loss_student' in self.batch_losses:
            ax2.plot(self.batch_losses['verification_loss_student'], label='Verification',
                    color='green', alpha=0.5)
        if 'classification_loss_student' in self.batch_losses:
            ax2.plot(self.batch_losses['classification_loss_student'], label='Classification',
                    color='red', alpha=0.5)
        if 'recon_total' in self.batch_losses:
            ax2.plot(self.batch_losses['recon_total'], label='Reconstruction',
                    color='purple', alpha=0.5)
        ax2.set_title('Task-specific Losses')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Distillation Losses
        ax3 = axes[1, 0]
        if 'feature_distill_loss' in self.batch_losses:
            ax3.plot(self.batch_losses['feature_distill_loss'], label='Feature Distillation',
                    color='orange', alpha=0.5)
        if 'logit_distill_loss' in self.batch_losses:
            ax3.plot(self.batch_losses['logit_distill_loss'], label='Logit Distillation',
                    color='brown', alpha=0.5)
        ax3.set_title('Distillation Losses')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Reconstruction Sub-losses
        ax4 = axes[1, 1]
        if 'recon_charbonnier' in self.batch_losses:
            ax4.plot(self.batch_losses['recon_charbonnier'], label='Charbonnier',
                    color='blue', alpha=0.5)
        if 'recon_ssim' in self.batch_losses:
            ax4.plot(self.batch_losses['recon_ssim'], label='SSIM',
                    color='green', alpha=0.5)
        if 'recon_lpips' in self.batch_losses:
            ax4.plot(self.batch_losses['recon_lpips'], label='LPIPS',
                    color='red', alpha=0.5)
        ax4.set_title('Reconstruction Sub-losses')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Loss')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{plot_dir}/latest.png", dpi=100)
        plt.close()

    def _plot_validation_metrics(self):
        """Generate and save validation plots comparing train vs validation - saves as latest_val.png"""
        plot_dir = f"{self.run_dir}/plots"

        # Skip if we don't have validation data yet
        if len(self.val_losses.get('total_loss', [])) == 0:
            return

        # Create a combined figure with all metrics
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Calculate actual x-axis values (validation run numbers)
        num_validations = len(self.val_losses['total_loss'])
        val_x_axis = range(1, num_validations + 1)

        fig.suptitle(f'Train vs Validation Metrics ({num_validations} validations)', fontsize=16)

        # Plot 1: Total Loss (Train vs Val)
        ax1 = axes[0, 0]
        # For train losses, we only have epoch-level data, so plot those
        if 'total_loss' in self.train_losses and self.train_losses['total_loss']:
            train_epochs = range(1, len(self.train_losses['total_loss']) + 1)
            ax1.plot(train_epochs, self.train_losses['total_loss'], label='Train Total Loss (epoch-level)',
                    color='blue', linewidth=2, marker='o', markersize=4, alpha=0.7)
        # Validation losses are at mid-epoch intervals
        if 'total_loss' in self.val_losses and self.val_losses['total_loss']:
            ax1.plot(val_x_axis, self.val_losses['total_loss'], label='Val Total Loss',
                    color='orange', linewidth=2, marker='s', markersize=3)
        ax1.set_title('Total Loss')
        ax1.set_xlabel('Validation Run')
        ax1.set_ylabel('Loss')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Task-specific Losses (Train vs Val)
        ax2 = axes[0, 1]
        # Note: Only plotting validation losses here since train is epoch-level
        if 'verification_loss_student' in self.val_losses:
            ax2.plot(val_x_axis, self.val_losses['verification_loss_student'],
                    label='Val Verification', color='green', linewidth=2, marker='o', markersize=3)
        if 'classification_loss_student' in self.val_losses:
            ax2.plot(val_x_axis, self.val_losses['classification_loss_student'],
                    label='Val Classification', color='red', linewidth=2, marker='s', markersize=3)
        if 'recon_total' in self.val_losses:
            ax2.plot(val_x_axis, self.val_losses['recon_total'],
                    label='Val Reconstruction', color='purple', linewidth=2, marker='^', markersize=3)
        ax2.set_title('Task-specific Losses (Validation)')
        ax2.set_xlabel('Validation Run')
        ax2.set_ylabel('Loss')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Validation Accuracy Metrics
        ax3 = axes[1, 0]
        if 'verification_accuracy' in self.val_metrics:
            ax3.plot(val_x_axis, self.val_metrics['verification_accuracy'],
                    label='Verification Accuracy', color='green', linewidth=2, marker='o', markersize=3)
        if 'classification_accuracy' in self.val_metrics:
            ax3.plot(val_x_axis, self.val_metrics['classification_accuracy'],
                    label='Classification Accuracy', color='red', linewidth=2, marker='s', markersize=3)
        ax3.set_title('Validation Accuracy Metrics')
        ax3.set_xlabel('Validation Run')
        ax3.set_ylabel('Accuracy')
        ax3.set_ylim([0, 1])
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Validation Image Quality Metrics
        ax4 = axes[1, 1]
        if 'ssim' in self.val_metrics:
            ax4_twin = ax4.twinx()
            line1 = ax4.plot(val_x_axis, self.val_metrics['ssim'],
                           label='SSIM', color='blue', linewidth=2, marker='o', markersize=3)
            if 'psnr' in self.val_metrics:
                line2 = ax4_twin.plot(val_x_axis, self.val_metrics['psnr'],
                                     label='PSNR (dB)', color='orange', linewidth=2, marker='s', markersize=3)
            ax4.set_xlabel('Validation Run')
            ax4.set_ylabel('SSIM', color='blue')
            ax4.tick_params(axis='y', labelcolor='blue')
            ax4_twin.set_ylabel('PSNR (dB)', color='orange')
            ax4_twin.tick_params(axis='y', labelcolor='orange')
            ax4.set_title('Reconstruction Quality')

            # Combine legends
            lines = line1
            if 'psnr' in self.val_metrics:
                lines += line2
            labels = [l.get_label() for l in lines]
            ax4.legend(lines, labels, loc='lower right')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No validation metrics yet',
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Reconstruction Quality')

        plt.tight_layout()
        plt.savefig(f"{plot_dir}/latest_val.png", dpi=100)
        plt.close()

    def _plot_train_metrics(self):
        """Generate and save train evaluation metrics plot - saves as latest_train.png"""
        plot_dir = f"{self.run_dir}/plots"

        # Skip if we don't have enough data yet
        if len(self.batch_metrics.get('verification_accuracy', [])) < 10:
            return

        # Create a combined figure with all train metrics
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Training Evaluation Metrics (Step {self.global_step})', fontsize=16)

        # Plot 1: Verification Accuracy
        ax1 = axes[0, 0]
        if 'verification_accuracy' in self.batch_metrics and self.batch_metrics['verification_accuracy']:
            ax1.plot(self.batch_metrics['verification_accuracy'],
                    label='Verification Accuracy', color='green', alpha=0.3)
            # Add smoothed version
            window = min(100, len(self.batch_metrics['verification_accuracy']) // 10)
            if window > 1:
                smoothed = np.convolve(self.batch_metrics['verification_accuracy'],
                                      np.ones(window)/window, mode='valid')
                ax1.plot(range(window-1, len(self.batch_metrics['verification_accuracy'])), smoothed,
                        label='Smoothed', color='green', linewidth=2)
        ax1.set_title('Verification Accuracy')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim([0, 1])
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Classification Accuracy
        ax2 = axes[0, 1]
        if 'classification_accuracy' in self.batch_metrics and self.batch_metrics['classification_accuracy']:
            ax2.plot(self.batch_metrics['classification_accuracy'],
                    label='Classification Accuracy', color='red', alpha=0.3)
            # Add smoothed version
            window = min(100, len(self.batch_metrics['classification_accuracy']) // 10)
            if window > 1:
                smoothed = np.convolve(self.batch_metrics['classification_accuracy'],
                                      np.ones(window)/window, mode='valid')
                ax2.plot(range(window-1, len(self.batch_metrics['classification_accuracy'])), smoothed,
                        label='Smoothed', color='red', linewidth=2)
        ax2.set_title('Classification Accuracy')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim([0, 1])
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: SSIM
        ax3 = axes[1, 0]
        if 'ssim' in self.batch_metrics and self.batch_metrics['ssim']:
            ax3.plot(self.batch_metrics['ssim'],
                    label='SSIM', color='blue', alpha=0.3)
            # Add smoothed version
            window = min(100, len(self.batch_metrics['ssim']) // 10)
            if window > 1:
                smoothed = np.convolve(self.batch_metrics['ssim'],
                                      np.ones(window)/window, mode='valid')
                ax3.plot(range(window-1, len(self.batch_metrics['ssim'])), smoothed,
                        label='Smoothed', color='blue', linewidth=2)
        ax3.set_title('Reconstruction SSIM')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('SSIM')
        ax3.set_ylim([0, 1])
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: PSNR
        ax4 = axes[1, 1]
        if 'psnr' in self.batch_metrics and self.batch_metrics['psnr']:
            ax4.plot(self.batch_metrics['psnr'],
                    label='PSNR', color='orange', alpha=0.3)
            # Add smoothed version
            window = min(100, len(self.batch_metrics['psnr']) // 10)
            if window > 1:
                smoothed = np.convolve(self.batch_metrics['psnr'],
                                      np.ones(window)/window, mode='valid')
                ax4.plot(range(window-1, len(self.batch_metrics['psnr'])), smoothed,
                        label='Smoothed', color='orange', linewidth=2)
        ax4.set_title('Reconstruction PSNR')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('PSNR (dB)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{plot_dir}/latest_train.png", dpi=100)
        plt.close()

    def _visualize_reconstructions(self, batch, outputs):
        """Visualize reconstruction results - saves as latest_reconstruction.png"""
        plot_dir = f"{self.run_dir}/plots"

        # Get images (detach and move to CPU)
        anchor = batch['anchor'].detach().cpu()
        positive = batch['positive'].detach().cpu()
        reconstructed = outputs['reconstructed_image'].detach().cpu()

        # Select up to 8 samples to display
        num_samples = min(8, anchor.size(0))

        fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f'Reconstruction Results (Step {self.global_step})', fontsize=14)

        for i in range(num_samples):
            # Anchor (concealed)
            axes[i, 0].imshow(anchor[i, 0], cmap='gray', vmin=0, vmax=1)
            axes[i, 0].set_title('Anchor (Concealed)')
            axes[i, 0].axis('off')

            # Positive (unconcealed - ground truth)
            axes[i, 1].imshow(positive[i, 0], cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title('Target (Unconcealed)')
            axes[i, 1].axis('off')

            # Reconstructed
            axes[i, 2].imshow(reconstructed[i, 0], cmap='gray', vmin=0, vmax=1)
            axes[i, 2].set_title('Reconstructed')
            axes[i, 2].axis('off')

        plt.tight_layout()
        plt.savefig(f"{plot_dir}/latest_reconstruction.png", dpi=100)
        plt.close()

    def _visualize_augmentations(self, batch):
        """Visualize augmentations on 30 randomly sampled triplets - saves as latest_augmentations.png"""
        plot_dir = f"{self.run_dir}/plots"

        # Get original and augmented images
        anchor_aug = batch['anchor'].detach().cpu()
        positive_aug = batch['positive'].detach().cpu()
        negative_aug = batch['negative'].detach().cpu()

        anchor_orig = batch['original_anchor'].detach().cpu()
        positive_orig = batch['original_positive'].detach().cpu()
        negative_orig = batch['original_negative'].detach().cpu()

        # Sample up to 10 triplets to keep visualization manageable
        num_samples = min(10, anchor_aug.size(0))

        fig, axes = plt.subplots(num_samples, 6, figsize=(18, 3*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f'Augmentation Visualization (Step {self.global_step})', fontsize=14)

        # Set column titles
        col_titles = ['Anchor\nOriginal', 'Anchor\nAugmented',
                     'Positive\nOriginal', 'Positive\nAugmented',
                     'Negative\nOriginal', 'Negative\nAugmented']

        for i in range(num_samples):
            # Anchor original
            axes[i, 0].imshow(anchor_orig[i, 0], cmap='gray', vmin=0, vmax=1)
            if i == 0:
                axes[i, 0].set_title(col_titles[0])
            axes[i, 0].axis('off')

            # Anchor augmented
            axes[i, 1].imshow(anchor_aug[i, 0], cmap='gray', vmin=0, vmax=1)
            if i == 0:
                axes[i, 1].set_title(col_titles[1])
            axes[i, 1].axis('off')

            # Positive original
            axes[i, 2].imshow(positive_orig[i, 0], cmap='gray', vmin=0, vmax=1)
            if i == 0:
                axes[i, 2].set_title(col_titles[2])
            axes[i, 2].axis('off')

            # Positive augmented
            axes[i, 3].imshow(positive_aug[i, 0], cmap='gray', vmin=0, vmax=1)
            if i == 0:
                axes[i, 3].set_title(col_titles[3])
            axes[i, 3].axis('off')

            # Negative original
            axes[i, 4].imshow(negative_orig[i, 0], cmap='gray', vmin=0, vmax=1)
            if i == 0:
                axes[i, 4].set_title(col_titles[4])
            axes[i, 4].axis('off')

            # Negative augmented
            axes[i, 5].imshow(negative_aug[i, 0], cmap='gray', vmin=0, vmax=1)
            if i == 0:
                axes[i, 5].set_title(col_titles[5])
            axes[i, 5].axis('off')

        plt.tight_layout()
        plt.savefig(f"{plot_dir}/latest_augmentations.png", dpi=100)
        plt.close()

    def train(self):
        """Main training loop"""
        self.logger.info("Starting privileged information MTL training...")

        # Log validation interval info
        val_interval = self.config.get('validation_interval_epochs', 0.2)
        batches_per_epoch = len(self.train_loader)
        val_interval_batches = max(1, int(batches_per_epoch * val_interval))
        validations_per_epoch = batches_per_epoch // val_interval_batches
        self.logger.info(f"Validation interval: {val_interval} epochs ({val_interval_batches} batches)")
        self.logger.info(f"Running validation {validations_per_epoch} times per epoch")

        for epoch in range(self.config['epochs']):
            self.current_epoch = epoch

            # Train epoch (includes mid-epoch validations every 0.2 epochs)
            train_losses = self.train_epoch()

            # Store epoch-level train losses
            for key, value in train_losses.items():
                self.train_losses[key].append(value)

            # Log epoch summary (validation metrics already logged during mid-epoch validations)
            self.logger.info(
                f"\n{'='*60}\n"
                f"Epoch {epoch+1}/{self.config['epochs']} completed\n"
                f"Train Loss: {train_losses['total_loss']:.4f}\n"
                f"Best Val SSIM so far: {self.best_val_ssim:.4f}\n"
                f"{'='*60}"
            )

            # Save checkpoint at epoch boundaries (in addition to best model saves)
            self.save_checkpoint(is_best=False)

        # Final plot generation
        self._plot_and_save_metrics()
        self._plot_train_metrics()
        self._plot_validation_metrics()

        self.logger.info("Training completed!")
        self.logger.info(f"Best validation SSIM: {self.best_val_ssim:.4f}")


def create_config():
    """
    Create default configuration for privileged learning

    Augmentation Modes:
        'gentle': Rotation ±10°, scale [0.9,1.1], crop [0.9,1.0], stripe masking, light speckle noise
        'medium': Rotation ±20°, translation ±5px, crop [0.7,1.0], cutout, band-drop, moderate speckle
        'strong': All medium + more aggressive parameters (for verification/posture tasks)

    Validation:
        'validation_interval_epochs': Run validation every N epochs (0.2 = every 20% of epoch)

    Distillation Control:
        'freeze_teacher': If True, teacher parameters are frozen (requires_grad=False)
        'distill_full_triplet': If True, distill posture logits for anchor+positive+negative (3x signal)
    """
    return {
        'input_size': 64,
        'batch_size': 256,
        'learning_rate': 1e-4,
        'epochs': 500,
        'num_classes': 5,
        'embed_dim': 64,  # Reduced from 128 to help prevent overfitting with SupCon
        'backbone_type': 'custom',
        'dropout_rate': 0.5,
        'pretrained': False,
        'num_workers': 4,
        'random_seed': 42,
        'weight_decay': 1e-3,
        'grad_clip': 1.0,
        'use_scheduler': True,
        'supcon_temperature': 0.07,  # Temperature for Supervised Contrastive Loss
        'classification_smoothing': 0.1,
        'distillation_temperature': 3.0,
        'cross_attention_heads': 8,
        'distill_dim': 256,
        'posture_ids': [0, 1, 2, 3, 4],
        'validation_interval_epochs': 0.5,  # Run validation every 0.2 epochs (5 times per epoch)
        'freeze_teacher': False,  # If True, freeze teacher model parameters
        'distill_full_triplet': True,  # If True, distill anchor+positive+negative logits (recommended)
        'train_val_csv': 'datafiles_fixed_29_08_sterile/faceData_cropped_64x64/dBm/train_val_dataset_visible.csv',
        'data_dir': '/home/noam/data_fixed/faceData_cropped_64x64',
        'visible_data_dir': '/home/noam/data_fixed/visible',  # Adjust to your visible data path
        'augmentation_mode': 'gentle',  # Options: 'gentle', 'medium', 'strong'
        'val_ratio': 0.15,
        'loss_weights': {
            'verification_student': 1.0,
            'verification_teacher': 0.5,
            'classification_student': 1.0,
            'classification_teacher': 0.5,
            'feature_distillation': 1.0,
            'logit_distillation': 0.5,
            'reconstruction': 1.0,
            'reconstruction_weights': {
                'charbonnier': 1.0,
                'ssim': 0.6,
                'lpips': 0.4
            }
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Privileged Information MTL Training')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    args = parser.parse_args()

    # Verify working directory
    print(f"Working directory: {os.getcwd()}")
    print(f"Script location: {os.path.abspath(__file__)}")

    # Load config
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_config()

    # Set random seeds
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    random.seed(config['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['random_seed'])

    # Create trainer and run training
    trainer = PrivilegedTrainer(config)
    trainer.setup_model_and_optimizer()
    trainer.setup_data_loaders()
    trainer.train()


if __name__ == "__main__":
    main()
