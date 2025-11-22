#!/usr/bin/env python3
"""
Privileged Information Loss Functions (V2)

Implements teacher-student distillation loss with:
1. Student task losses (verification, classification, reconstruction)
2. Teacher task losses (optional, for monitoring)
3. Feature-level distillation (MSE between student and teacher latents)
4. Logit-level distillation (KL divergence between student and teacher predictions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

# Loss components (simplified, self-contained)
class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Loss"""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos = mask * (1 - torch.eye(batch_size).to(device))
        mean_log_prob_pos = (mask_pos * log_prob).sum(1) / (mask_pos.sum(1) + 1e-8)

        loss = -mean_log_prob_pos.mean()
        return loss

class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross-entropy"""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)
        loss = -log_preds.sum(dim=-1).mean()
        nll = F.nll_loss(log_preds, target, reduction='mean')
        return self.smoothing * (loss / n_classes) + (1 - self.smoothing) * nll

class CharbonnierLoss(nn.Module):
    """Charbonnier loss (smooth L1)"""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps)
        return loss.mean()

class SSIMLoss(nn.Module):
    """SSIM loss"""
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mu1 = pred.mean(dim=[2, 3], keepdim=True)
        mu2 = target.mean(dim=[2, 3], keepdim=True)
        sigma1_sq = ((pred - mu1) ** 2).mean(dim=[2, 3], keepdim=True)
        sigma2_sq = ((target - mu2) ** 2).mean(dim=[2, 3], keepdim=True)
        sigma12 = ((pred - mu1) * (target - mu2)).mean(dim=[2, 3], keepdim=True)

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim.mean()

class ReconstructionLoss(nn.Module):
    """Combined reconstruction loss"""
    def __init__(self, weights: Dict[str, float], device: torch.device):
        super().__init__()
        self.weights = weights
        self.charbonnier = CharbonnierLoss()
        self.ssim = SSIMLoss()
        if LPIPS_AVAILABLE:
            self.lpips = lpips.LPIPS(net='vgg').to(device)
        else:
            self.lpips = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = {}
        losses['charbonnier'] = self.charbonnier(pred, target)
        losses['ssim'] = self.ssim(pred, target)

        if self.lpips and self.weights.get('lpips', 0) > 0:
            pred_rgb = pred.repeat(1, 3, 1, 1) if pred.size(1) == 1 else pred
            target_rgb = target.repeat(1, 3, 1, 1) if target.size(1) == 1 else target
            losses['lpips'] = self.lpips(pred_rgb, target_rgb).mean()
        else:
            losses['lpips'] = torch.tensor(0.0, device=pred.device)

        total = sum(self.weights.get(k, 0) * v for k, v in losses.items())
        return total, losses


class PrivilegedDistillationLoss(nn.Module):
    """
    Unified loss for privileged information learning via teacher-student distillation.

    Combines:
    1. Student verification loss (SupCon on THz embeddings)
    2. Student classification loss (cross-entropy on THz posture predictions)
    3. Teacher verification loss (SupCon on fused embeddings)
    4. Teacher classification loss (for teacher supervision)
    5. Feature-level distillation (MSE between student and teacher latent projections)
    6. Logit-level distillation (KL divergence between student and teacher logits)
    7. Reconstruction loss (pixel-level and perceptual)
    """

    def __init__(
        self,
        weights: Dict[str, float],
        device: torch.device,
        supcon_temperature: float = 0.07,
        cls_smoothing: float = 0.1,
        distill_temperature: float = 3.0,
        reconstruction_weights: Dict[str, float] = None,
        distill_full_triplet: bool = False
    ):
        """
        Args:
            weights: Dictionary of loss weights for each component
            device: Device for computation
            supcon_temperature: Temperature for supervised contrastive loss
            cls_smoothing: Label smoothing factor for classification
            distill_temperature: Temperature for logit distillation
            reconstruction_weights: Weights for reconstruction loss components
            distill_full_triplet: If True, distill logits for anchor+positive+negative.
                                 If False, distill only anchor logits (default for backward compatibility)
        """
        super().__init__()
        self.weights = weights
        self.device = device
        self.distill_temperature = distill_temperature
        self.distill_full_triplet = distill_full_triplet

        # --- Task Loss Components ---
        self.verification_loss = SupervisedContrastiveLoss(temperature=supcon_temperature)
        self.classification_loss = LabelSmoothingCrossEntropy(smoothing=cls_smoothing)

        # Reconstruction loss
        if reconstruction_weights is None:
            reconstruction_weights = {
                'charbonnier': 1.0,
                'ssim': 0.6,
                'gradient': 0.0,
                'lpips': 0.4
            }
        self.reconstruction_loss = ReconstructionLoss(
            weights=reconstruction_weights,
            device=device
        )

        # --- Distillation Loss Components ---
        # Feature-level distillation: MSE between projected latents
        self.feature_distill_loss = nn.MSELoss()

        # Logit-level distillation: KL divergence
        self.logit_distill_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(
        self,
        network_outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss with distillation.

        Args:
            network_outputs: Dictionary of model outputs
            batch: Dictionary of batch data

        Returns:
            total_loss: Combined weighted loss
            loss_log: Dictionary of individual loss values for logging
        """
        loss_log = {}
        total_loss = 0.0

        batch_size = network_outputs['anchor_embedding'].shape[0]

        # --- 1. Student Verification Loss (SupCon) ---
        # Concatenate anchor, positive, negative embeddings
        student_embeddings = torch.cat([
            network_outputs['anchor_embedding'],
            network_outputs['positive_embedding'],
            network_outputs['negative_embedding']
        ], dim=0)  # [3B, embed_dim]

        # Create identity labels for SupCon
        # anchor and positive share same identity, negative has different identity
        identity_labels = torch.cat([
            torch.arange(batch_size),  # anchor: 0,1,2,...,B-1
            torch.arange(batch_size),  # positive: 0,1,2,...,B-1 (same as anchor)
            torch.arange(batch_size, 2 * batch_size)  # negative: B,B+1,...,2B-1 (all different)
        ], dim=0).to(self.device)

        ver_loss_student = self.verification_loss(student_embeddings, identity_labels)
        loss_log['verification_loss_student'] = ver_loss_student.item()
        total_loss += self.weights.get('verification_student', 1.0) * ver_loss_student

        # --- 2. Student Classification Loss ---
        # Compute on all three images (anchor, positive, negative) for richer gradients
        student_logits = torch.cat([
            network_outputs['anchor_posture_logits'],
            network_outputs['positive_posture_logits'],
            network_outputs['negative_posture_logits']
        ], dim=0)
        labels = torch.cat([
            batch['anchor_posture_class'],
            batch['positive_posture_class'],
            batch['negative_posture_class']
        ], dim=0).to(self.device)

        cls_loss_student = self.classification_loss(student_logits, labels)
        loss_log['classification_loss_student'] = cls_loss_student.item()
        total_loss += self.weights.get('classification_student', 1.0) * cls_loss_student

        # --- 3. Reconstruction Loss ---
        target_positive = batch['positive'].to(self.device)

        # Handle potential channel mismatch
        if target_positive.shape[1] == 3 and network_outputs['reconstructed_image'].shape[1] == 1:
            target_positive = target_positive.mean(dim=1, keepdim=True)

        recon_loss, recon_loss_dict = self.reconstruction_loss(
            network_outputs['reconstructed_image'],
            target_positive
        )
        loss_log.update({f'recon_{k}': v.item() for k, v in recon_loss_dict.items()})
        total_loss += self.weights.get('reconstruction', 1.0) * recon_loss

        # --- Teacher Path (only if visible images were provided) ---
        if network_outputs.get('has_teacher', False):
            # --- 4. Teacher Verification Loss (SupCon on fused embeddings) ---
            if all(k in network_outputs for k in ['teacher_anchor_embedding', 'teacher_positive_embedding', 'teacher_negative_embedding']):
                # Concatenate teacher embeddings
                teacher_embeddings = torch.cat([
                    network_outputs['teacher_anchor_embedding'],
                    network_outputs['teacher_positive_embedding'],
                    network_outputs['teacher_negative_embedding']
                ], dim=0)  # [3B, embed_dim]

                # Use same identity labels as student (already created above)
                ver_loss_teacher = self.verification_loss(teacher_embeddings, identity_labels)
                loss_log['verification_loss_teacher'] = ver_loss_teacher.item()
                total_loss += self.weights.get('verification_teacher', 0.5) * ver_loss_teacher
            else:
                loss_log['verification_loss_teacher'] = 0.0

            # --- 5. Teacher Classification Loss ---
            # Compute on all three images (anchor, positive, negative) for richer gradients
            if all(k in network_outputs for k in ['teacher_anchor_posture_logits', 'teacher_positive_posture_logits', 'teacher_negative_posture_logits']):
                teacher_logits = torch.cat([
                    network_outputs['teacher_anchor_posture_logits'],
                    network_outputs['teacher_positive_posture_logits'],
                    network_outputs['teacher_negative_posture_logits']
                ], dim=0)
                labels = torch.cat([
                    batch['anchor_posture_class'],
                    batch['positive_posture_class'],
                    batch['negative_posture_class']
                ], dim=0).to(self.device)

                cls_loss_teacher = self.classification_loss(teacher_logits, labels)
                loss_log['classification_loss_teacher'] = cls_loss_teacher.item()
                total_loss += self.weights.get('classification_teacher', 0.5) * cls_loss_teacher
            else:
                # Fallback: only anchor if full triplet not available
                teacher_logits_anchor = network_outputs['teacher_anchor_posture_logits']
                labels_anchor = batch['anchor_posture_class'].to(self.device)

                cls_loss_teacher = self.classification_loss(teacher_logits_anchor, labels_anchor)
                loss_log['classification_loss_teacher'] = cls_loss_teacher.item()
                total_loss += self.weights.get('classification_teacher', 0.5) * cls_loss_teacher

            # --- 6. Feature-Level Distillation ---
            # MSE between student and teacher projected latents (anchor only)
            student_proj = network_outputs['student_distill_proj']
            teacher_proj = network_outputs['teacher_distill_proj'].detach()  # Stop gradient on teacher

            feat_distill_loss = self.feature_distill_loss(student_proj, teacher_proj)
            loss_log['feature_distill_loss'] = feat_distill_loss.item()
            total_loss += self.weights.get('feature_distillation', 1.0) * feat_distill_loss

            # --- 7. Logit-Level Distillation ---
            # KL divergence between student and teacher classification logits
            # Controlled by self.distill_full_triplet flag
            if self.distill_full_triplet and all(k in network_outputs for k in ['teacher_positive_posture_logits', 'teacher_negative_posture_logits']):
                # Distill logits for anchor + positive + negative (3x stronger signal)
                student_logits_all = torch.cat([
                    network_outputs['anchor_posture_logits'],
                    network_outputs['positive_posture_logits'],
                    network_outputs['negative_posture_logits']
                ], dim=0)
                teacher_logits_all = torch.cat([
                    network_outputs['teacher_anchor_posture_logits'],
                    network_outputs['teacher_positive_posture_logits'],
                    network_outputs['teacher_negative_posture_logits']
                ], dim=0).detach()  # Stop gradient on teacher

                # Apply temperature scaling
                T = self.distill_temperature
                student_log_probs = F.log_softmax(student_logits_all / T, dim=1)
                teacher_probs = F.softmax(teacher_logits_all / T, dim=1)

                logit_distill_loss = self.logit_distill_loss(student_log_probs, teacher_probs) * (T ** 2)
            else:
                # Distill only anchor logits (default for backward compatibility)
                student_logits_anchor = network_outputs['anchor_posture_logits']
                teacher_logits_anchor = network_outputs['teacher_anchor_posture_logits'].detach()

                # Apply temperature scaling
                T = self.distill_temperature
                student_log_probs = F.log_softmax(student_logits_anchor / T, dim=1)
                teacher_probs = F.softmax(teacher_logits_anchor / T, dim=1)

                logit_distill_loss = self.logit_distill_loss(student_log_probs, teacher_probs) * (T ** 2)

            loss_log['logit_distill_loss'] = logit_distill_loss.item()
            total_loss += self.weights.get('logit_distillation', 0.5) * logit_distill_loss

        else:
            # No teacher available (no visible images in this batch)
            loss_log['verification_loss_teacher'] = 0.0
            loss_log['classification_loss_teacher'] = 0.0
            loss_log['feature_distill_loss'] = 0.0
            loss_log['logit_distill_loss'] = 0.0

        # Add total loss to log
        loss_log['total_loss'] = total_loss.item()

        return total_loss, loss_log


def create_privileged_distillation_loss(
    weights: Dict[str, float],
    device: torch.device,
    supcon_temperature: float = 0.07,
    cls_smoothing: float = 0.1,
    distill_temperature: float = 3.0,
    reconstruction_weights: Dict[str, float] = None,
    distill_full_triplet: bool = False
) -> PrivilegedDistillationLoss:
    """Factory function to create PrivilegedDistillationLoss"""
    return PrivilegedDistillationLoss(
        weights=weights,
        device=device,
        supcon_temperature=supcon_temperature,
        cls_smoothing=cls_smoothing,
        distill_temperature=distill_temperature,
        reconstruction_weights=reconstruction_weights,
        distill_full_triplet=distill_full_triplet
    )


if __name__ == "__main__":
    # Test the Privileged Distillation Loss
    print("Testing PrivilegedDistillationLoss...")

    batch_size = 4
    embed_dim = 128
    num_classes = 5
    distill_dim = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create loss function
    loss_weights = {
        'verification_student': 1.0,
        'verification_teacher': 0.5,
        'classification_student': 1.0,
        'classification_teacher': 0.5,
        'feature_distillation': 1.0,
        'logit_distillation': 0.5,
        'reconstruction': 1.0
    }

    reconstruction_weights = {
        'charbonnier': 1.0,
        'ssim': 0.6,
        'gradient': 0.0,
        'lpips': 0.4
    }

    loss_fn = create_privileged_distillation_loss(
        weights=loss_weights,
        device=device,
        supcon_temperature=0.07,
        cls_smoothing=0.1,
        distill_temperature=3.0,
        reconstruction_weights=reconstruction_weights
    )

    # Test Case 1: With teacher (visible images available)
    print("\n1. Testing with teacher (visible images available):")

    network_outputs = {
        'anchor_embedding': F.normalize(torch.randn(batch_size, embed_dim), p=2, dim=1).to(device),
        'positive_embedding': F.normalize(torch.randn(batch_size, embed_dim), p=2, dim=1).to(device),
        'negative_embedding': F.normalize(torch.randn(batch_size, embed_dim), p=2, dim=1).to(device),
        'anchor_posture_logits': torch.randn(batch_size, num_classes).to(device),
        'positive_posture_logits': torch.randn(batch_size, num_classes).to(device),
        'negative_posture_logits': torch.randn(batch_size, num_classes).to(device),
        'reconstructed_image': torch.rand(batch_size, 1, 64, 64).to(device),
        'student_distill_proj': torch.randn(batch_size, distill_dim).to(device),
        'has_teacher': True,
        'teacher_anchor_embedding': F.normalize(torch.randn(batch_size, embed_dim), p=2, dim=1).to(device),
        'teacher_positive_embedding': F.normalize(torch.randn(batch_size, embed_dim), p=2, dim=1).to(device),
        'teacher_negative_embedding': F.normalize(torch.randn(batch_size, embed_dim), p=2, dim=1).to(device),
        'teacher_anchor_posture_logits': torch.randn(batch_size, num_classes).to(device),
        'teacher_positive_posture_logits': torch.randn(batch_size, num_classes).to(device),
        'teacher_negative_posture_logits': torch.randn(batch_size, num_classes).to(device),
        'teacher_distill_proj': torch.randn(batch_size, distill_dim).to(device)
    }

    batch = {
        'anchor_posture_class': torch.randint(0, num_classes, (batch_size,)).to(device),
        'positive_posture_class': torch.randint(0, num_classes, (batch_size,)).to(device),
        'negative_posture_class': torch.randint(0, num_classes, (batch_size,)).to(device),
        'positive': torch.rand(batch_size, 1, 64, 64).to(device)
    }

    total_loss, loss_log = loss_fn(network_outputs, batch)

    print(f"  Total loss: {total_loss.item():.4f}")
    print("  Component losses:")
    for key, value in loss_log.items():
        if key != 'total_loss':
            print(f"    {key}: {value:.4f}")

    # Test Case 2: Without teacher (no visible images)
    print("\n2. Testing without teacher (no visible images):")

    network_outputs_no_teacher = {
        'anchor_embedding': F.normalize(torch.randn(batch_size, embed_dim), p=2, dim=1).to(device),
        'positive_embedding': F.normalize(torch.randn(batch_size, embed_dim), p=2, dim=1).to(device),
        'negative_embedding': F.normalize(torch.randn(batch_size, embed_dim), p=2, dim=1).to(device),
        'anchor_posture_logits': torch.randn(batch_size, num_classes).to(device),
        'positive_posture_logits': torch.randn(batch_size, num_classes).to(device),
        'negative_posture_logits': torch.randn(batch_size, num_classes).to(device),
        'reconstructed_image': torch.rand(batch_size, 1, 64, 64).to(device),
        'student_distill_proj': torch.randn(batch_size, distill_dim).to(device),
        'has_teacher': False
    }

    total_loss_no_teacher, loss_log_no_teacher = loss_fn(network_outputs_no_teacher, batch)

    print(f"  Total loss: {total_loss_no_teacher.item():.4f}")
    print("  Component losses:")
    for key, value in loss_log_no_teacher.items():
        if key != 'total_loss':
            print(f"    {key}: {value:.4f}")

    print("\nPrivilegedDistillationLoss test complete!")
    print("\nNote: When teacher is not available, distillation losses are zero.")
    print("The model gracefully degrades to THz-only training in such cases.")
