#!/usr/bin/env python3
"""
Task-Specific Triplets Loss Functions
Separate loss classes for each training stage to replace flawed static weighting

Stage 1 (DiscriminativeLoss):
1. Face verification (triplet loss)  
2. Head posture classification (label smoothing cross-entropy)

Stage 2 (ReconstructionLoss):
3. Image reconstruction (Charbonnier + SSIM + LPIPS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

# Try to import LPIPS, fall back if not available
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available. Using only Charbonnier + SSIM.")

class SimpleTripletLoss(nn.Module):
    """Simple triplet loss using PyTorch built-in implementation"""
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchor, positive, negative: L2-normalized embeddings [B, embed_dim]
        """
        return self.triplet_loss(anchor, positive, negative)


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon)

    Reference: "Supervised Contrastive Learning" (Khosla et al., NeurIPS 2020)
    https://arxiv.org/abs/2004.11362

    Key advantages over triplet loss:
    - Uses all positives and negatives in batch, not just one pair
    - Better sample efficiency and generalization
    - More stable training with less overfitting

    Usage:
        - Embeddings MUST be L2-normalized before passing to this loss
        - Requires identity labels for each sample
        - Works best with batch sizes >= 32 for sufficient positive/negative pairs
    """

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        """
        Args:
            temperature: Temperature parameter for scaling (default: 0.07)
            base_temperature: Base temperature for normalization (default: 0.07)
        """
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute supervised contrastive loss.

        Args:
            embeddings: L2-normalized embeddings [N, embed_dim] where N = batch_size
                       MUST be normalized: F.normalize(embeddings, p=2, dim=1)
            labels: Identity labels [N] - samples with same label are positives

        Returns:
            Scalar loss value
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        # Ensure embeddings are normalized (safety check)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity matrix: [N, N]
        # similarity[i,j] = embeddings[i] · embeddings[j]
        similarity_matrix = torch.matmul(embeddings, embeddings.T)

        # Create mask for positive pairs (same identity, different sample)
        # labels: [N] -> labels_eq: [N, N] where labels_eq[i,j] = (labels[i] == labels[j])
        labels = labels.contiguous().view(-1, 1)
        mask_positives = torch.eq(labels, labels.T).float().to(device)

        # Remove self-contrast (diagonal): don't compare sample with itself
        mask_self = torch.eye(batch_size, dtype=torch.float32, device=device)
        mask_positives = mask_positives * (1 - mask_self)

        # For numerical stability, subtract max from similarity before exp
        logits = similarity_matrix / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # Compute exp(logits)
        exp_logits = torch.exp(logits)

        # Mask out self-contrast from denominator
        exp_logits = exp_logits * (1 - mask_self)

        # Sum of exp(similarity) for all samples (denominator)
        sum_exp_logits = exp_logits.sum(dim=1, keepdim=True)

        # Compute log probability: log(exp(z_i·z_p) / sum(exp(z_i·z_j)))
        log_prob = logits - torch.log(sum_exp_logits + 1e-8)

        # Compute mean of log-likelihood over positive pairs
        # Only consider samples that have at least one positive pair
        num_positives_per_sample = mask_positives.sum(dim=1)

        # Avoid division by zero for samples with no positives
        # This can happen if a batch has only one sample per identity
        mask_has_positives = (num_positives_per_sample > 0).float()

        # Sum log probabilities over positive pairs and normalize
        mean_log_prob_pos = (mask_positives * log_prob).sum(dim=1) / (num_positives_per_sample + 1e-8)

        # Final loss: negative mean of log probabilities
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos

        # Only average over samples that have positive pairs
        loss = (loss * mask_has_positives).sum() / (mask_has_positives.sum() + 1e-8)

        return loss

class LabelSmoothingCrossEntropy(nn.Module):
    """Simple cross-entropy loss with label smoothing"""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Logits [B, num_classes]
            target: Class indices [B]
        """
        confidence = 1.0 - self.smoothing
        log_probs = F.log_softmax(pred, dim=-1)
        
        # True class probability
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        
        # Uniform distribution over all classes
        smooth_loss = -log_probs.mean(dim=-1)
        
        # Combine label smoothing
        final_loss = confidence * nll_loss + self.smoothing * smooth_loss
            
        return final_loss.mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier loss (smooth L1) for better edge preservation"""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps)
        return loss.mean()

class SSIMLoss(nn.Module):
    """SSIM loss for structural similarity"""
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.register_buffer('window', self._create_window(window_size, sigma))
        
    def _gaussian(self, window_size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(window_size, dtype=torch.float32)
        coords = coords - (window_size - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.outer(g)
    
    def _create_window(self, window_size: int, sigma: float) -> torch.Tensor:
        window = self._gaussian(window_size, sigma)
        return window.unsqueeze(0).unsqueeze(0)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.size(1) == 1:  # Grayscale
            window = self.window.to(pred.device)
        else:  # Multi-channel
            window = self.window.expand(pred.size(1), 1, -1, -1).to(pred.device)
            
        mu1 = F.conv2d(pred, window, padding=self.window_size//2, groups=pred.size(1))
        mu2 = F.conv2d(target, window, padding=self.window_size//2, groups=target.size(1))
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size//2, groups=pred.size(1)) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size//2, groups=target.size(1)) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size//2, groups=pred.size(1)) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()

class DiscriminativeLoss(nn.Module):
    """
    Loss for Stage 1: Expert encoder training.
    This version is simplified to only compute the verification loss.
    The classification loss will be handled separately in the training loop.
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.triplet_loss = SimpleTripletLoss(margin=margin)

    def forward(self, anchor_embedding: torch.Tensor, positive_embedding: torch.Tensor, 
                negative_embedding: torch.Tensor) -> torch.Tensor:
        """
        Calculates the verification (triplet) loss.
        
        Args:
            anchor_embedding: Anchor embeddings [B, embed_dim]
            positive_embedding: Positive embeddings [B, embed_dim] 
            negative_embedding: Negative embeddings [B, embed_dim]
        
        Returns:
            verification_loss: Triplet loss
        """
        return self.triplet_loss(anchor_embedding, positive_embedding, negative_embedding)

class GradientLoss(nn.Module):
    """Computes the L1 loss between the gradients of the prediction and target."""
    def __init__(self, device: torch.device):
        super().__init__()
        # Define Sobel filters for X and Y gradients
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        self.sobel_x = nn.Parameter(sobel_x, requires_grad=False)
        self.sobel_y = nn.Parameter(sobel_y, requires_grad=False)
        self.loss = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.size(1) != 1: # Only for grayscale
            pred = pred.mean(dim=1, keepdim=True)
            target = target.mean(dim=1, keepdim=True)

        pred_grad_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.sobel_y, padding=1)
        target_grad_x = F.conv2d(target, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1)

        grad_loss_x = self.loss(pred_grad_x, target_grad_x)
        grad_loss_y = self.loss(pred_grad_y, target_grad_y)

        return grad_loss_x + grad_loss_y

class ReconstructionLoss(nn.Module):
    """
    Pixel-perfect and perceptual loss for Stage 2: Reconstruction training.
    Configurable combination of Charbonnier, SSIM, LPIPS, and Gradient loss.
    """
    def __init__(self, weights: Dict[str, float], device: torch.device, lpips_ema_beta: float = 0.9):
        super().__init__()
        self.weights = weights
        self.device = device
        self.lpips_ema_beta = lpips_ema_beta

        # Initialize core loss components
        self.charbonnier = CharbonnierLoss()
        self.ssim = SSIMLoss()

        # Only initialize gradient loss if weight > 0 (deprecated, for backward compatibility)
        self.use_gradient = self.weights.get('gradient', 0.0) > 0
        if self.use_gradient:
            self.gradient = GradientLoss(device=self.device)
            print("Warning: Gradient loss is deprecated and should not be used")

        # Initialize LPIPS only if it has weight and is available
        self.use_lpips = self.weights.get('lpips', 0.0) > 0 and LPIPS_AVAILABLE
        if self.use_lpips:
            self.lpips_net = lpips.LPIPS(net='vgg', verbose=False).to(device)
            for param in self.lpips_net.parameters():
                param.requires_grad = False
            # EMA buffer for LPIPS smoothing
            self.register_buffer('lpips_ema', None)
            print(f"LPIPS EMA smoothing enabled with beta={lpips_ema_beta}")
        elif self.weights.get('lpips', 0.0) > 0:
            print("Warning: LPIPS weight > 0 but library not found. LPIPS will be disabled.")

    def _normalize_for_lpips(self, image: torch.Tensor) -> torch.Tensor:
        return image * 2 - 1

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if pred.shape != target.shape:
            target = F.interpolate(target, size=pred.shape[-2:], mode='bilinear', align_corners=False)

        # --- Calculate core losses ---
        char_loss = self.charbonnier(pred, target)
        ssim_loss = self.ssim(pred, target)

        loss_dict = {
            'charbonnier': char_loss,
            'ssim': ssim_loss,
        }

        total_loss = (
            self.weights.get('charbonnier', 0.0) * char_loss +
            self.weights.get('ssim', 0.0) * ssim_loss
        )

        # Gradient loss (deprecated, only computed if explicitly requested)
        if self.use_gradient:
            grad_loss = self.gradient(pred, target)
            loss_dict['gradient'] = grad_loss
            total_loss += self.weights.get('gradient', 0.0) * grad_loss
        else:
            loss_dict['gradient'] = torch.tensor(0.0, device=pred.device)

        # LPIPS loss (perceptual) with EMA smoothing
        if self.use_lpips:
            pred_norm = self._normalize_for_lpips(pred.repeat(1, 3, 1, 1))
            target_norm = self._normalize_for_lpips(target.repeat(1, 3, 1, 1))
            lpips_raw = self.lpips_net(pred_norm, target_norm).mean()

            # Apply EMA smoothing during training
            if self.training:
                if self.lpips_ema is None:
                    # Initialize EMA with first value
                    self.lpips_ema = lpips_raw.detach()
                else:
                    # Update EMA: ema = beta * ema + (1 - beta) * current
                    self.lpips_ema = self.lpips_ema_beta * self.lpips_ema + (1 - self.lpips_ema_beta) * lpips_raw.detach()

                # Use smoothed value for loss computation
                lpips_loss = self.lpips_ema.clone()
            else:
                # During eval, use raw value
                lpips_loss = lpips_raw

            total_loss += self.weights['lpips'] * lpips_loss
            loss_dict['lpips'] = lpips_loss
        else:
            loss_dict['lpips'] = torch.tensor(0.0, device=pred.device)

        loss_dict['total'] = total_loss
        return total_loss, loss_dict


def create_discriminative_loss(margin: float = 0.2) -> DiscriminativeLoss:
    """Create discriminative loss function for Stage 1 training"""
    return DiscriminativeLoss(margin=margin)

def create_reconstruction_loss(weights: Dict[str, float], device: torch.device) -> ReconstructionLoss:
    """Creates the reconstruction loss function for Stage 2 training with configurable weights."""
    return ReconstructionLoss(weights=weights, device=device)

class UnifiedMTLLoss(nn.Module):
    """
    Unified Multi-Task Loss that computes all losses and combines them.
    Uses Supervised Contrastive Loss for face verification instead of triplet loss.
    """
    def __init__(self, weights: Dict[str, float], device: torch.device, supcon_temperature: float = 0.07, cls_smoothing: float = 0.1):
        super().__init__()
        self.weights = weights
        self.device = device

        # Instantiate individual loss components
        self.verification_loss = SupervisedContrastiveLoss(temperature=supcon_temperature)
        self.classification_loss = LabelSmoothingCrossEntropy(smoothing=cls_smoothing)
        self.reconstruction_loss = ReconstructionLoss(weights=weights.get('reconstruction_weights', {}), device=device)

    def forward(self, network_outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = network_outputs['anchor_embedding'].shape[0]

        # 1. Verification Loss (Supervised Contrastive)
        # Concatenate anchor, positive, negative embeddings
        embeddings = torch.cat([
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

        ver_loss = self.verification_loss(embeddings, identity_labels)

        # 2. Classification Loss (calculated on all three images in the triplet for richer gradients)
        logits = torch.cat([
            network_outputs['anchor_posture_logits'], network_outputs['positive_posture_logits'], network_outputs['negative_posture_logits']
        ], dim=0)
        labels = torch.cat([
            batch['anchor_posture_class'], batch['positive_posture_class'], batch['negative_posture_class']
        ], dim=0).to(logits.device)
        cls_loss = self.classification_loss(logits, labels)

        # 3. Reconstruction Loss
        target_positive = batch['positive'].to(network_outputs['reconstructed_image'].device)  # Target is the unconcealed positive

        # If positive is 3-channel (padded for cross-modal batching) but reconstruction is 1-channel, convert back
        if target_positive.shape[1] == 3 and network_outputs['reconstructed_image'].shape[1] == 1:
            target_positive = target_positive.mean(dim=1, keepdim=True)

        recon_loss, recon_loss_dict = self.reconstruction_loss(
            network_outputs['reconstructed_image'], target_positive
        )

        # 4. Weighted Sum for a Single, Unified Loss
        total_loss = (self.weights.get('verification', 1.0) * ver_loss +
                      self.weights.get('classification', 1.0) * cls_loss +
                      self.weights.get('reconstruction', 1.0) * recon_loss)

        loss_log = {
            'total_loss': total_loss.item(), 'verification_loss': ver_loss.item(), 'classification_loss': cls_loss.item(),
            **{f'recon_{k}': v.item() for k, v in recon_loss_dict.items()}
        }

        return total_loss, loss_log

if __name__ == "__main__":
    # Test the task-specific loss functions
    print("Testing DiscriminativeLoss and ReconstructionLoss...")
    
    batch_size = 4
    embed_dim = 512
    
    # Test DiscriminativeLoss
    print("\n1. Testing DiscriminativeLoss:")
    discriminative_loss = create_discriminative_loss()
    
    # Mock discriminative inputs
    anchor_embedding = F.normalize(torch.randn(batch_size, embed_dim), p=2, dim=1)
    positive_embedding = F.normalize(torch.randn(batch_size, embed_dim), p=2, dim=1)
    negative_embedding = F.normalize(torch.randn(batch_size, embed_dim), p=2, dim=1)
    posture_logits = torch.randn(batch_size, 5)
    posture_class = torch.randint(0, 5, (batch_size,))
    
    # Compute discriminative loss (verification only now)
    verification_loss = discriminative_loss(
        anchor_embedding, positive_embedding, negative_embedding
    )
    
    print(f"  Verification loss: {verification_loss.item():.4f}")
    
    # Test classification loss separately
    classification_loss = LabelSmoothingCrossEntropy(smoothing=0.1)
    cls_loss = classification_loss(posture_logits, posture_class)
    print(f"  Classification loss: {cls_loss.item():.4f}")
    
    # Test ReconstructionLoss
    print("\n2. Testing ReconstructionLoss:")
    reconstruction_loss = create_reconstruction_loss()
    
    # Mock reconstruction inputs - both in [0,1] range for THz data
    pred_image = torch.rand(batch_size, 1, 56, 56)
    target_image = torch.rand(batch_size, 1, 56, 56)
    
    # Compute reconstruction loss
    total_loss, loss_dict = reconstruction_loss(pred_image, target_image)
    
    print(f"  Total loss: {total_loss.item():.4f}")
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.4f}")
    
    print("\nTask-specific loss testing complete!")