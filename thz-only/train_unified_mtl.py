#!/usr/bin/env python3
"""
Unified Multi-Task Learning Training Script
Trains a single model on all three tasks simultaneously:
1. Face verification (triplet loss)
2. Head posture classification (cross-entropy)
3. Image reconstruction (MoE with multiple losses)
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

from config import *
from unified_architecture import UnifiedMultiTaskModel
from unified_losses import UnifiedMTLLoss
from unified_data_loader import get_triplets_data_loaders, save_augmentation_samples
from utils import (
    ClassificationMetricsCalculator,
    VerificationMetricsCalculator,
    ReconstructionMetricsCalculator,
    calculate_ssim, calculate_psnr, tensor_to_numpy,
    plot_training_losses, plot_training_metrics, plot_validation_metrics,
    visualize_reconstructions, visualize_augmentations
)

class UnifiedTrainer:
    """Trainer for Unified Multi-Task Learning"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        resolution = config['input_size']
        dataset_type = 'original' if resolution <= 56 else 'bicubic'
        self.run_dir = f"runs/unified_mtl_{dataset_type}_{resolution}_{timestamp}"
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(f"{self.run_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{self.run_dir}/logs", exist_ok=True)
        os.makedirs(f"{self.run_dir}/plots", exist_ok=True)
        
        # Save config
        with open(f"{self.run_dir}/config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Setup logging
        log_file = f"{self.run_dir}/logs/unified_mtl_training.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('unified_mtl_training')
        
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

        # Initialize visualization tracking
        self.visualizations_saved_this_epoch = {}
        self.augmentation_samples = None  # Cache samples for augmentation visualization
        
        # Initialize CSV logging
        self.train_csv_path = f"{self.run_dir}/logs/train_logs.csv"
        self.val_csv_path = f"{self.run_dir}/logs/validation_logs.csv"
        self._initialize_csv_logging()
        self._save_code_files()
        
        self.logger.info(f"UnifiedTrainer initialized")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Input size: {config['input_size']}")
        self.logger.info(f"Dataset type: {dataset_type}")
        self.logger.info(f"Run directory: {self.run_dir}")
    
    def _save_code_files(self):
        """Save code files for reproducibility"""
        code_dir = os.path.join(self.run_dir, 'code')
        os.makedirs(code_dir, exist_ok=True)
        
        # List of MTL code files to save
        mtl_files = [
            'mtl/unified_architecture.py',
            'mtl/unified_losses.py', 
            'mtl/unified_data_loader.py',
            'mtl/train_unified_mtl.py'
        ]
        
        # Save MTL code files
        for filename in mtl_files:
            src_path = filename
            if os.path.exists(src_path):
                dst_path = os.path.join(code_dir, os.path.basename(filename))
                shutil.copy2(src_path, dst_path)
                self.logger.info(f"Saved code file: {filename}")
        
        # Save config files
        config_files = ['custom_batch_sampler.py']
        for filename in config_files:
            if os.path.exists(filename):
                dst_path = os.path.join(code_dir, filename)
                shutil.copy2(filename, dst_path)
    
    def _initialize_csv_logging(self):
        """Initialize CSV files for logging"""
        # Training log headers
        train_headers = ['epoch', 'batch', 'total_loss', 'verification_loss', 'classification_loss', 
                        'recon_total', 'recon_charbonnier', 'recon_ssim', 'recon_gradient', 'recon_lpips', 
                        'learning_rate']
        with open(self.train_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(train_headers)
        
        # Validation log headers (base + top 5 enhanced metrics)
        val_headers = ['epoch', 'total_loss', 'verification_loss', 'classification_loss',
                      'recon_total', 'recon_charbonnier', 'recon_ssim', 'recon_gradient', 'recon_lpips',
                      'verification_accuracy', 'classification_accuracy', 'ssim', 'psnr',
                      # Top 5 enhanced metrics across all tasks
                      'cls_conf_correct', 'cls_f1_macro', 'ver_distance_margin',
                      'recon_ssim_mean', 'recon_hardest_posture_id']
        with open(self.val_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(val_headers)

    def setup_model_and_optimizer(self):
        """Setup model, loss function, and optimizer"""
        # Model
        self.model = UnifiedMultiTaskModel(
            num_classes=self.config['num_classes'],
            embed_dim=self.config['embed_dim'],
            backbone_type=self.config['backbone_type'],
            posture_ids=self.config.get('posture_ids', [0, 1, 2, 3, 4]),
            dropout_rate=self.config.get('dropout_rate', 0.5),
            pretrained=self.config.get('pretrained', True)
        ).to(self.device)

        # Loss function (now using Supervised Contrastive Loss)
        self.criterion = UnifiedMTLLoss(
            weights=self.config['loss_weights'],
            device=self.device,
            supcon_temperature=self.config.get('supcon_temperature', 0.07),
            cls_smoothing=self.config.get('classification_smoothing', 0.1)
        )

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

    def setup_data_loaders(self):
        """Setup data loaders with THz-specific augmentations"""
        self.train_loader, self.val_loader, _ = get_triplets_data_loaders(
            train_val_csv=self.config.get('train_val_csv'),
            input_size=self.config['input_size'],
            batch_size=self.config['batch_size'],
            num_workers=self.config.get('num_workers', 4),
            val_ratio=self.config.get('val_ratio', 0.15),
            random_seed=self.config['random_seed'],
            output_folder=self.run_dir,
            augmentation_mode=self.config.get('augmentation_mode', 'gentle')
        )

        self.logger.info(f"Data loaders created with augmentation mode: {self.config.get('augmentation_mode', 'gentle')}")
        self.logger.info(f"  Train: {len(self.train_loader)} batches")
        self.logger.info(f"  Validation: {len(self.val_loader)} batches")

    def train_epoch(self):
        """Train for one epoch with unified MTL approach and mid-epoch validation"""
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

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass - single unified pass for all tasks
            outputs = self.model(anchor, positive, negative, anchor_posture_ids)

            # Compute unified loss - all tasks in one loss
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
                        loss_log.get('verification_loss', 0),
                        loss_log.get('classification_loss', 0),
                        loss_log.get('recon_total', 0),
                        loss_log.get('recon_charbonnier', 0),
                        loss_log.get('recon_ssim', 0),
                        loss_log.get('recon_gradient', 0),
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
                    'ver': f"{loss_log.get('verification_loss', 0):.4f}",
                    'cls': f"{loss_log.get('classification_loss', 0):.4f}",
                    'rec': f"{loss_log.get('recon_total', 0):.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })

            # Run validation every N batches (mid-epoch validation)
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

        # Check if this is a visualization epoch and set up trackers
        is_vis_epoch = (self.current_epoch + 1) % 25 == 0
        if is_vis_epoch:
            self.logger.info(f"Epoch {self.current_epoch+1} is a visualization epoch. Preparing to save samples.")
            self.visualizations_saved_this_epoch = {
                'recon': defaultdict(list),
                'ver': [], 'ver_ids': set(),
                'cls': [], 'cls_ids': set(),
                'aug_ids': set()
            }
            self.augmentation_samples_to_plot = []
            # Create directories
            os.makedirs(f"{self.run_dir}/visualizations/epoch_{self.current_epoch+1:03d}/reconstruction", exist_ok=True)
            os.makedirs(f"{self.run_dir}/visualizations/epoch_{self.current_epoch+1:03d}/verification", exist_ok=True)
            os.makedirs(f"{self.run_dir}/visualizations/epoch_{self.current_epoch+1:03d}/classification", exist_ok=True)

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                anchor = batch['anchor'].to(self.device, non_blocking=True)
                positive = batch['positive'].to(self.device, non_blocking=True)
                negative = batch['negative'].to(self.device, non_blocking=True)
                anchor_posture_ids = batch['anchor_posture_class'].to(self.device, non_blocking=True)

                # Forward pass
                outputs = self.model(anchor, positive, negative, anchor_posture_ids)

                # Generate visualizations if it's the right epoch
                if is_vis_epoch:
                    self._generate_and_save_visualizations(batch, outputs)

                    # Collect samples for augmentation plot
                    if len(self.augmentation_samples_to_plot) < 25:
                        for i in range(len(batch['anchor'])):
                            identity_id = batch['identity'][i]
                            if len(self.augmentation_samples_to_plot) < 25 and identity_id not in self.visualizations_saved_this_epoch['aug_ids']:
                                self.augmentation_samples_to_plot.append({
                                    'identity': identity_id,
                                    'original_anchor': batch['original_anchor'][i].cpu(),
                                    'anchor': batch['anchor'][i].cpu(),
                                    'original_positive': batch['original_positive'][i].cpu(),
                                    'positive': batch['positive'][i].cpu(),
                                    'original_negative': batch['original_negative'][i].cpu(),
                                    'negative': batch['negative'][i].cpu()
                                })
                                self.visualizations_saved_this_epoch['aug_ids'].add(identity_id)

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

                # Calculate classification accuracy
                for prefix in ['anchor', 'positive', 'negative']:
                    logits = outputs[f'{prefix}_posture_logits']
                    labels = batch[f'{prefix}_posture_class'].to(self.device)
                    pred = torch.argmax(logits, dim=1)
                    classification_correct += (pred == labels).sum().item()
                    classification_total += labels.size(0)

                # Enhanced classification metrics (all triplet samples)
                # Collect all logits and labels from the triplet
                all_logits = torch.cat([
                    outputs['anchor_posture_logits'],
                    outputs['positive_posture_logits'],
                    outputs['negative_posture_logits']
                ], dim=0)
                all_labels = torch.cat([
                    batch['anchor_posture_class'],
                    batch['positive_posture_class'],
                    batch['negative_posture_class']
                ], dim=0).to(self.device)

                # Collect embeddings for silhouette score
                all_embeddings = torch.cat([
                    anchor_emb,
                    positive_emb,
                    negative_emb
                ], dim=0)

                # Update enhanced classification metrics calculator
                cls_metrics_calculator.update_batch(
                    all_logits,
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

                # SSIM calculation
                ssim = self._calculate_ssim(reconstructed, target)
                ssim_scores.extend(ssim.cpu().numpy())

                # PSNR calculation
                psnr = self._calculate_psnr(reconstructed, target)
                psnr_scores.extend(psnr.cpu().numpy())

                # Update reconstruction metrics calculator
                recon_metrics_calculator.update_batch(
                    ssim,
                    psnr,
                    batch['anchor_posture_class']
                )

        # After the validation loop, generate the augmentation plot if needed
        if is_vis_epoch:
            self._plot_augmentation_samples()

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
            f.write(f"Epoch {self.current_epoch + 1} - Comprehensive Metrics Report (THz-Only Model)\n")
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
                avg_losses.get('verification_loss', 0),
                avg_losses.get('classification_loss', 0),
                avg_losses.get('recon_total', 0),
                avg_losses.get('recon_charbonnier', 0),
                avg_losses.get('recon_ssim', 0),
                avg_losses.get('recon_gradient', 0),
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
        return calculate_ssim(img1, img2)

    def _calculate_psnr(self, img1, img2):
        """Calculate PSNR between two images"""
        return calculate_psnr(img1, img2)

    def _tensor_to_np(self, tensor):
        """Convert a CHW tensor to a HWC numpy array for plotting"""
        return tensor_to_numpy(tensor)

    def _plot_and_save_metrics(self):
        """Generate and save plots for batch-level losses"""
        plot_dir = f"{self.run_dir}/plots"
        current_lr = self.optimizer.param_groups[0]["lr"]
        plot_training_losses(self.batch_losses, self.global_step, current_lr,
                            f"{plot_dir}/latest.png")

    def _plot_train_metrics(self):
        """Generate and save train evaluation metrics plot"""
        plot_dir = f"{self.run_dir}/plots"
        plot_training_metrics(self.batch_metrics, self.global_step,
                             f"{plot_dir}/latest_train.png")

    def _plot_validation_metrics(self):
        """Generate and save validation plots"""
        plot_dir = f"{self.run_dir}/plots"
        plot_validation_metrics(dict(self.train_losses), dict(self.val_losses),
                               dict(self.val_metrics), f"{plot_dir}/latest_val.png")

    def _visualize_reconstructions(self, batch, outputs):
        """Visualize reconstruction results"""
        plot_dir = f"{self.run_dir}/plots"
        visualize_reconstructions(batch['anchor'], batch['positive'],
                                 outputs['reconstructed_image'], self.global_step,
                                 f"{plot_dir}/latest_reconstruction.png")

    def _visualize_augmentations(self, batch):
        """Visualize augmentations on triplets"""
        plot_dir = f"{self.run_dir}/plots"
        visualize_augmentations(batch, self.global_step,
                               f"{plot_dir}/latest_augmentations.png")

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

    def _generate_and_save_visualizations(self, batch, outputs):
        """Generates and saves visualizations for all three tasks."""
        epoch = self.current_epoch

        # --- Task 1: Reconstruction Visualization ---
        recon_vis_needed = any(len(self.visualizations_saved_this_epoch['recon'][pid]) < 5 for pid in self.config.get('posture_ids', []))
        if recon_vis_needed:
            vis_dir = f"{self.run_dir}/visualizations/epoch_{epoch+1:03d}/reconstruction"
            for i in range(len(batch['anchor'])):
                posture_id = batch['anchor_posture_class'][i].item()
                if len(self.visualizations_saved_this_epoch['recon'][posture_id]) < 5:
                    # Get images
                    concealed_input = self._tensor_to_np(batch['anchor'][i])
                    reconstructed_output = self._tensor_to_np(outputs['reconstructed_image'][i])
                    ground_truth = self._tensor_to_np(batch['positive'][i])

                    # Plot
                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    axes[0].imshow(concealed_input, cmap='gray')
                    axes[0].set_title('Input (Concealed)')
                    axes[1].imshow(reconstructed_output, cmap='gray')
                    axes[1].set_title('Reconstructed')
                    axes[2].imshow(ground_truth, cmap='gray')
                    axes[2].set_title('Ground Truth')
                    for ax in axes: ax.axis('off')

                    sample_num = len(self.visualizations_saved_this_epoch['recon'][posture_id]) + 1
                    fig.suptitle(f'Reconstruction - Expert {posture_id} - Sample {sample_num}')

                    # Save
                    save_path = f"{vis_dir}/expert_{posture_id}_sample_{sample_num}.png"
                    plt.savefig(save_path)
                    plt.close(fig)

                    self.visualizations_saved_this_epoch['recon'][posture_id].append(i)

        # --- Task 2: Face Verification Visualization ---
        if len(self.visualizations_saved_this_epoch['ver']) < 20:
            vis_dir = f"{self.run_dir}/visualizations/epoch_{epoch+1:03d}/verification"
            for i in range(len(batch['anchor'])):
                identity_id = batch['identity'][i]
                if len(self.visualizations_saved_this_epoch['ver']) < 20 and identity_id not in self.visualizations_saved_this_epoch['ver_ids']:
                    # Get images
                    anchor_img = self._tensor_to_np(batch['anchor'][i])
                    pos_img = self._tensor_to_np(batch['positive'][i])
                    neg_img = self._tensor_to_np(batch['negative'][i])

                    # Get distances
                    pos_dist = torch.norm(outputs['anchor_embedding'][i] - outputs['positive_embedding'][i])
                    neg_dist = torch.norm(outputs['anchor_embedding'][i] - outputs['negative_embedding'][i])

                    # Plot
                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    axes[0].imshow(anchor_img, cmap='gray')
                    axes[0].set_title('Anchor')
                    axes[1].imshow(pos_img, cmap='gray')
                    axes[1].set_title(f'Positive (Dist: {pos_dist:.2f})')
                    axes[2].imshow(neg_img, cmap='gray')
                    axes[2].set_title(f'Negative (Dist: {neg_dist:.2f})')
                    for ax in axes: ax.axis('off')

                    fig.suptitle(f'Verification - ID {identity_id}')

                    # Save
                    save_path = f"{vis_dir}/id_{identity_id}.png"
                    plt.savefig(save_path)
                    plt.close(fig)

                    self.visualizations_saved_this_epoch['ver'].append(identity_id)
                    self.visualizations_saved_this_epoch['ver_ids'].add(identity_id)

        # --- Task 3: Head Posture Classification Visualization ---
        if len(self.visualizations_saved_this_epoch['cls']) < 20:
            vis_dir = f"{self.run_dir}/visualizations/epoch_{epoch+1:03d}/classification"
            for i in range(len(batch['anchor'])):
                identity_id = batch['identity'][i]
                if len(self.visualizations_saved_this_epoch['cls']) < 20 and identity_id not in self.visualizations_saved_this_epoch['cls_ids']:
                    # Get data
                    img = self._tensor_to_np(batch['anchor'][i])
                    gt_label = batch['anchor_posture_class'][i].item()
                    pred_label = torch.argmax(outputs['anchor_posture_logits'][i]).item()

                    # Plot
                    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                    ax.imshow(img, cmap='gray')
                    is_correct = (gt_label == pred_label)
                    title_color = 'green' if is_correct else 'red'
                    ax.set_title(f'ID {identity_id}\nGT: {gt_label}, Pred: {pred_label}', color=title_color)
                    ax.axis('off')

                    # Save
                    save_path = f"{vis_dir}/id_{identity_id}.png"
                    plt.savefig(save_path)
                    plt.close(fig)

                    self.visualizations_saved_this_epoch['cls'].append(identity_id)
                    self.visualizations_saved_this_epoch['cls_ids'].add(identity_id)

    def _plot_augmentation_samples(self):
        """Plots the collected augmentation samples in a grid."""
        if not hasattr(self, 'augmentation_samples_to_plot') or not self.augmentation_samples_to_plot:
            self.logger.warning("No augmentation samples were collected for plotting.")
            return

        self.logger.info(f"Generating augmentation comparison plot for {len(self.augmentation_samples_to_plot)} samples...")

        num_samples = len(self.augmentation_samples_to_plot)
        fig, axes = plt.subplots(num_samples, 6, figsize=(18, num_samples * 3))
        if num_samples == 1: # Handle case of single sample
            axes = axes.reshape(1, -1)

        column_titles = ['Orig Anchor', 'Aug Anchor', 'Orig Positive', 'Aug Positive', 'Orig Negative', 'Aug Negative']
        for ax, title in zip(axes[0], column_titles):
            ax.set_title(title, fontsize=10)

        for i, sample in enumerate(self.augmentation_samples_to_plot):
            images = [
                sample['original_anchor'], sample['anchor'],
                sample['original_positive'], sample['positive'],
                sample['original_negative'], sample['negative']
            ]

            for j, img_tensor in enumerate(images):
                ax = axes[i, j]
                ax.imshow(self._tensor_to_np(img_tensor), cmap='gray')
                ax.axis('off')

            # Label the row with the Identity ID
            axes[i, 0].text(-0.2, 0.5, f"ID: {sample['identity']}", va='center', ha='center',
                            transform=axes[i, 0].transAxes, fontsize=10, rotation=90)

        plt.tight_layout(pad=0.5, h_pad=1.0)

        # Save the final plot
        vis_dir = f"{self.run_dir}/visualizations/epoch_{self.current_epoch+1:03d}"
        os.makedirs(vis_dir, exist_ok=True)
        save_path = f"{vis_dir}/augmentation_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.logger.info(f"Saved augmentation comparison plot to {save_path}")

    def train(self):
        """Main training loop"""
        self.logger.info("Starting unified MTL training...")
        
        for epoch in range(self.config['epochs']):
            self.current_epoch = epoch
            
            # Train epoch
            train_losses = self.train_epoch()
            
            # Validate epoch
            val_losses, val_metrics = self.validate_epoch()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step(val_losses['total_loss'])
            
            # Store losses and metrics
            for key, value in train_losses.items():
                self.train_losses[key].append(value)
            for key, value in val_losses.items():
                self.val_losses[key].append(value)
            for key, value in val_metrics.items():
                self.val_metrics[key].append(value)
            
            # Check if best model
            is_best = val_metrics['ssim'] > self.best_val_ssim
            if is_best:
                self.best_val_ssim = val_metrics['ssim']
                self.best_val_loss = val_losses['total_loss']
            
            # Save checkpoint
            self.save_checkpoint(is_best)
            
            # Log progress
            self.logger.info(f"Epoch {epoch+1}/{self.config['epochs']} - "
                           f"Train Loss: {train_losses['total_loss']:.4f}, "
                           f"Val Loss: {val_losses['total_loss']:.4f}, "
                           f"Ver Acc: {val_metrics['verification_accuracy']:.4f}, "
                           f"Cls Acc: {val_metrics['classification_accuracy']:.4f}, "
                           f"SSIM: {val_metrics['ssim']:.4f}, "
                           f"PSNR: {val_metrics['psnr']:.2f}")

            # Periodically save plots (e.g., every 10 epochs)
            if (epoch + 1) % 10 == 0:
                self._plot_and_save_metrics()
        
        # Final plot generation at the end of training
        self._plot_and_save_metrics()
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation SSIM: {self.best_val_ssim:.4f}")

def create_config():
    """Create config from config.py or return defaults"""
    return get_full_config()

def main():
    parser = argparse.ArgumentParser(description='Unified MTL Training')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    args = parser.parse_args()

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
    trainer = UnifiedTrainer(config)
    trainer.setup_model_and_optimizer()
    trainer.setup_data_loaders()
    trainer.train()

if __name__ == "__main__":
    main()