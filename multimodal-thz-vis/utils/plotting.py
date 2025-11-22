#!/usr/bin/env python3
"""
Plotting and visualization utilities for training
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
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


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Calculate PSNR between two images"""
    mse = ((img1 - img2) ** 2).mean(dim=[1, 2, 3])
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a CHW tensor to a HWC numpy array for plotting"""
    return tensor.squeeze(0).cpu().numpy()


def plot_training_losses(batch_losses: Dict[str, List[float]], global_step: int,
                         current_lr: float, save_path: str) -> None:
    """Plot training losses"""
    if len(batch_losses.get('total_loss', [])) < 10:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Training Progress (Step {global_step})', fontsize=16)

    # Plot 1: Total Loss
    ax1 = axes[0, 0]
    if 'total_loss' in batch_losses and batch_losses['total_loss']:
        ax1.plot(batch_losses['total_loss'], label='Total Loss', color='blue', alpha=0.3)
        window = min(100, len(batch_losses['total_loss']) // 10)
        if window > 1:
            smoothed = np.convolve(batch_losses['total_loss'], np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(batch_losses['total_loss'])), smoothed,
                    label='Smoothed', color='blue', linewidth=2)
    ax1.set_title('Total Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Task-specific Losses
    ax2 = axes[0, 1]
    if 'verification_loss' in batch_losses:
        ax2.plot(batch_losses['verification_loss'], label='Verification', color='green', alpha=0.5)
    if 'classification_loss' in batch_losses:
        ax2.plot(batch_losses['classification_loss'], label='Classification', color='red', alpha=0.5)
    if 'recon_total' in batch_losses:
        ax2.plot(batch_losses['recon_total'], label='Reconstruction', color='purple', alpha=0.5)
    ax2.set_title('Task-specific Losses')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Reconstruction Sub-losses
    ax3 = axes[1, 0]
    if 'recon_charbonnier' in batch_losses:
        ax3.plot(batch_losses['recon_charbonnier'], label='Charbonnier', color='blue', alpha=0.5)
    if 'recon_ssim' in batch_losses:
        ax3.plot(batch_losses['recon_ssim'], label='SSIM', color='green', alpha=0.5)
    if 'recon_lpips' in batch_losses:
        ax3.plot(batch_losses['recon_lpips'], label='LPIPS', color='red', alpha=0.5)
    ax3.set_title('Reconstruction Sub-losses')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Learning Rate
    ax4 = axes[1, 1]
    ax4.text(0.5, 0.5, f'Current LR: {current_lr:.6f}',
            ha='center', va='center', transform=ax4.transAxes, fontsize=14)
    ax4.set_title('Learning Rate')
    ax4.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def plot_training_metrics(batch_metrics: Dict[str, List[float]], global_step: int,
                          save_path: str) -> None:
    """Plot training evaluation metrics"""
    if len(batch_metrics.get('verification_accuracy', [])) < 10:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Training Evaluation Metrics (Step {global_step})', fontsize=16)

    def plot_metric_with_smoothing(ax, data, label, color, ylabel='Accuracy', ylim=None):
        if data:
            ax.plot(data, label=label, color=color, alpha=0.3)
            window = min(100, len(data) // 10)
            if window > 1:
                smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(data)), smoothed, label='Smoothed', color=color, linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(ylim)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot metrics
    plot_metric_with_smoothing(axes[0, 0], batch_metrics.get('verification_accuracy', []),
                               'Verification Accuracy', 'green', ylim=[0, 1])
    axes[0, 0].set_title('Verification Accuracy')

    plot_metric_with_smoothing(axes[0, 1], batch_metrics.get('classification_accuracy', []),
                               'Classification Accuracy', 'red', ylim=[0, 1])
    axes[0, 1].set_title('Classification Accuracy')

    plot_metric_with_smoothing(axes[1, 0], batch_metrics.get('ssim', []),
                               'SSIM', 'blue', 'SSIM', ylim=[0, 1])
    axes[1, 0].set_title('Reconstruction SSIM')

    plot_metric_with_smoothing(axes[1, 1], batch_metrics.get('psnr', []),
                               'PSNR', 'orange', 'PSNR (dB)')
    axes[1, 1].set_title('Reconstruction PSNR')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def plot_validation_metrics(train_losses: Dict[str, List[float]], val_losses: Dict[str, List[float]],
                            val_metrics: Dict[str, List[float]], save_path: str) -> None:
    """Plot validation metrics comparing train vs validation"""
    if len(val_losses.get('total_loss', [])) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    num_validations = len(val_losses['total_loss'])
    val_x_axis = range(1, num_validations + 1)
    fig.suptitle(f'Train vs Validation Metrics ({num_validations} validations)', fontsize=16)

    # Plot 1: Total Loss
    ax1 = axes[0, 0]
    if 'total_loss' in train_losses and train_losses['total_loss']:
        train_epochs = range(1, len(train_losses['total_loss']) + 1)
        ax1.plot(train_epochs, train_losses['total_loss'], label='Train Total Loss',
                color='blue', linewidth=2, marker='o', markersize=4, alpha=0.7)
    if 'total_loss' in val_losses and val_losses['total_loss']:
        ax1.plot(val_x_axis, val_losses['total_loss'], label='Val Total Loss',
                color='orange', linewidth=2, marker='s', markersize=3)
    ax1.set_title('Total Loss')
    ax1.set_xlabel('Validation Run')
    ax1.set_ylabel('Loss')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Task-specific Losses
    ax2 = axes[0, 1]
    if 'verification_loss' in val_losses:
        ax2.plot(val_x_axis, val_losses['verification_loss'], label='Val Verification',
                color='green', linewidth=2, marker='o', markersize=3)
    if 'classification_loss' in val_losses:
        ax2.plot(val_x_axis, val_losses['classification_loss'], label='Val Classification',
                color='red', linewidth=2, marker='s', markersize=3)
    if 'recon_total' in val_losses:
        ax2.plot(val_x_axis, val_losses['recon_total'], label='Val Reconstruction',
                color='purple', linewidth=2, marker='^', markersize=3)
    ax2.set_title('Task-specific Losses (Validation)')
    ax2.set_xlabel('Validation Run')
    ax2.set_ylabel('Loss')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Validation Accuracy
    ax3 = axes[1, 0]
    if 'verification_accuracy' in val_metrics:
        ax3.plot(val_x_axis, val_metrics['verification_accuracy'], label='Verification Accuracy',
                color='green', linewidth=2, marker='o', markersize=3)
    if 'classification_accuracy' in val_metrics:
        ax3.plot(val_x_axis, val_metrics['classification_accuracy'], label='Classification Accuracy',
                color='red', linewidth=2, marker='s', markersize=3)
    ax3.set_title('Validation Accuracy Metrics')
    ax3.set_xlabel('Validation Run')
    ax3.set_ylabel('Accuracy')
    ax3.set_ylim([0, 1])
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Image Quality
    ax4 = axes[1, 1]
    if 'ssim' in val_metrics:
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(val_x_axis, val_metrics['ssim'], label='SSIM',
                        color='blue', linewidth=2, marker='o', markersize=3)
        if 'psnr' in val_metrics:
            line2 = ax4_twin.plot(val_x_axis, val_metrics['psnr'], label='PSNR (dB)',
                                 color='orange', linewidth=2, marker='s', markersize=3)
        ax4.set_xlabel('Validation Run')
        ax4.set_ylabel('SSIM', color='blue')
        ax4.tick_params(axis='y', labelcolor='blue')
        ax4_twin.set_ylabel('PSNR (dB)', color='orange')
        ax4_twin.tick_params(axis='y', labelcolor='orange')
        ax4.set_title('Reconstruction Quality')
        lines = line1 + (line2 if 'psnr' in val_metrics else [])
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='lower right')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def visualize_reconstructions(anchor: torch.Tensor, positive: torch.Tensor,
                              reconstructed: torch.Tensor, global_step: int,
                              save_path: str, num_samples: int = 8) -> None:
    """Visualize reconstruction results"""
    anchor = anchor.detach().cpu()
    positive = positive.detach().cpu()
    reconstructed = reconstructed.detach().cpu()

    num_samples = min(num_samples, anchor.size(0))
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'Reconstruction Results (Step {global_step})', fontsize=14)

    for i in range(num_samples):
        axes[i, 0].imshow(anchor[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title('Anchor (Concealed)')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(positive[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('Target (Unconcealed)')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(reconstructed[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title('Reconstructed')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def visualize_augmentations(batch: Dict[str, torch.Tensor], global_step: int,
                            save_path: str, num_samples: int = 10) -> None:
    """Visualize augmentations on triplets"""
    anchor_aug = batch['anchor'].detach().cpu()
    positive_aug = batch['positive'].detach().cpu()
    negative_aug = batch['negative'].detach().cpu()

    anchor_orig = batch['original_anchor'].detach().cpu()
    positive_orig = batch['original_positive'].detach().cpu()
    negative_orig = batch['original_negative'].detach().cpu()

    num_samples = min(num_samples, anchor_aug.size(0))
    fig, axes = plt.subplots(num_samples, 6, figsize=(18, 3*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'Augmentation Visualization (Step {global_step})', fontsize=14)

    col_titles = ['Anchor\nOriginal', 'Anchor\nAugmented',
                 'Positive\nOriginal', 'Positive\nAugmented',
                 'Negative\nOriginal', 'Negative\nAugmented']

    for i in range(num_samples):
        images = [anchor_orig[i, 0], anchor_aug[i, 0],
                 positive_orig[i, 0], positive_aug[i, 0],
                 negative_orig[i, 0], negative_aug[i, 0]]

        for j, img in enumerate(images):
            axes[i, j].imshow(img, cmap='gray', vmin=0, vmax=1)
            if i == 0:
                axes[i, j].set_title(col_titles[j])
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
