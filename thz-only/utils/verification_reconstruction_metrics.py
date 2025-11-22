#!/usr/bin/env python3
"""
Enhanced Verification and Reconstruction Metrics

Provides comprehensive metrics to measure:
1. Face Verification Quality: Distance distributions, margin analysis, failure cases
2. Reconstruction Quality: Per-class SSIM/PSNR, perceptual quality, detail preservation

These metrics help quantify the superiority of distillation in learning better
embeddings for face verification and reconstructing fine-grained details.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from collections import defaultdict


class VerificationMetricsCalculator:
    """
    Calculator for enhanced face verification metrics.

    Tracks embedding distance distributions and computes:
    - Average positive/negative distances and their separation
    - Distance margin statistics
    - Hard negative/positive analysis
    - Embedding space quality metrics
    """

    def __init__(self):
        """Initialize the verification metrics calculator"""
        self.reset()

    def reset(self):
        """Reset all accumulators for a new epoch"""
        self.positive_distances = []
        self.negative_distances = []
        self.all_anchor_embeddings = []
        self.all_positive_embeddings = []
        self.all_negative_embeddings = []
        self.all_identities = []

    def update_batch(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor,
        identities: Optional[torch.Tensor] = None
    ):
        """
        Update metrics with a batch of embeddings.

        Args:
            anchor_embeddings: Anchor embeddings [batch_size, embed_dim]
            positive_embeddings: Positive embeddings [batch_size, embed_dim]
            negative_embeddings: Negative embeddings [batch_size, embed_dim]
            identities: Optional identity labels [batch_size]
        """
        # Move to CPU for metric computation
        anchor_emb = anchor_embeddings.detach().cpu()
        positive_emb = positive_embeddings.detach().cpu()
        negative_emb = negative_embeddings.detach().cpu()

        # Compute distances
        pos_dist = torch.norm(anchor_emb - positive_emb, p=2, dim=1)
        neg_dist = torch.norm(anchor_emb - negative_emb, p=2, dim=1)

        # Store distances
        self.positive_distances.extend(pos_dist.numpy().tolist())
        self.negative_distances.extend(neg_dist.numpy().tolist())

        # Store embeddings for global analysis
        self.all_anchor_embeddings.append(anchor_emb.numpy())
        self.all_positive_embeddings.append(positive_emb.numpy())
        self.all_negative_embeddings.append(negative_emb.numpy())

        if identities is not None:
            # Handle both tensor and list inputs
            if torch.is_tensor(identities):
                self.all_identities.extend(identities.cpu().numpy().tolist())
            else:
                self.all_identities.extend(identities)

    def compute_epoch_metrics(self) -> Dict[str, float]:
        """
        Compute aggregated verification metrics for the entire epoch.

        Returns:
            Dictionary of metric names to values
        """
        metrics = {}

        if not self.positive_distances or not self.negative_distances:
            return metrics

        pos_dist_array = np.array(self.positive_distances)
        neg_dist_array = np.array(self.negative_distances)

        # 1. Average distances
        metrics['ver_pos_dist_mean'] = np.mean(pos_dist_array)
        metrics['ver_pos_dist_std'] = np.std(pos_dist_array)
        metrics['ver_neg_dist_mean'] = np.mean(neg_dist_array)
        metrics['ver_neg_dist_std'] = np.std(neg_dist_array)

        # 2. Distance margin (separation between positive and negative)
        # Higher margin = better separation
        metrics['ver_distance_margin'] = metrics['ver_neg_dist_mean'] - metrics['ver_pos_dist_mean']

        # 3. Normalized margin (accounts for spread)
        # Higher = more confident separation
        margin_denominator = metrics['ver_pos_dist_std'] + metrics['ver_neg_dist_std']
        if margin_denominator > 0:
            metrics['ver_normalized_margin'] = metrics['ver_distance_margin'] / margin_denominator
        else:
            metrics['ver_normalized_margin'] = 0.0

        # 4. Hard cases analysis
        # Hard positives: positive pairs with large distances (top 10%)
        hard_pos_threshold = np.percentile(pos_dist_array, 90)
        metrics['ver_hard_pos_dist'] = np.mean(pos_dist_array[pos_dist_array >= hard_pos_threshold])
        metrics['ver_hard_pos_ratio'] = np.sum(pos_dist_array >= hard_pos_threshold) / len(pos_dist_array)

        # Hard negatives: negative pairs with small distances (bottom 10%)
        hard_neg_threshold = np.percentile(neg_dist_array, 10)
        metrics['ver_hard_neg_dist'] = np.mean(neg_dist_array[neg_dist_array <= hard_neg_threshold])
        metrics['ver_hard_neg_ratio'] = np.sum(neg_dist_array <= hard_neg_threshold) / len(neg_dist_array)

        # 5. Error analysis
        # Count verification failures (pos_dist >= neg_dist)
        failures = pos_dist_array >= neg_dist_array
        metrics['ver_failure_rate'] = np.sum(failures) / len(pos_dist_array)

        # 6. Distance distribution overlap
        # Measure overlap between positive and negative distance distributions
        pos_max = np.max(pos_dist_array)
        neg_min = np.min(neg_dist_array)
        overlap_region = max(0, pos_max - neg_min)
        metrics['ver_dist_overlap'] = overlap_region

        # 7. Embedding space quality (if we have all embeddings)
        if self.all_anchor_embeddings:
            all_anchor = np.concatenate(self.all_anchor_embeddings, axis=0)
            all_positive = np.concatenate(self.all_positive_embeddings, axis=0)

            # Embedding norm statistics
            anchor_norms = np.linalg.norm(all_anchor, axis=1)
            metrics['ver_emb_norm_mean'] = np.mean(anchor_norms)
            metrics['ver_emb_norm_std'] = np.std(anchor_norms)

            # Embedding space utilization (dimensionality)
            # Use PCA to estimate effective dimensionality
            try:
                from sklearn.decomposition import PCA
                pca = PCA()
                pca.fit(all_anchor)
                # Effective dimensionality: number of components explaining 95% variance
                cumsum = np.cumsum(pca.explained_variance_ratio_)
                effective_dim = np.argmax(cumsum >= 0.95) + 1
                metrics['ver_effective_dim'] = effective_dim
                metrics['ver_variance_explained_95'] = cumsum[effective_dim - 1] if effective_dim > 0 else 0
            except Exception as e:
                print(f"Warning: Could not compute embedding dimensionality: {e}")
                metrics['ver_effective_dim'] = 0
                metrics['ver_variance_explained_95'] = 0

        return metrics

    def get_distance_statistics(self) -> str:
        """
        Get detailed distance statistics for logging.

        Returns:
            String containing distance statistics
        """
        if not self.positive_distances or not self.negative_distances:
            return "No verification data available"

        pos_dist_array = np.array(self.positive_distances)
        neg_dist_array = np.array(self.negative_distances)

        report = "Verification Distance Statistics\n"
        report += "="*60 + "\n"
        report += f"Positive Distances:\n"
        report += f"  Mean: {np.mean(pos_dist_array):.4f}\n"
        report += f"  Std: {np.std(pos_dist_array):.4f}\n"
        report += f"  Min: {np.min(pos_dist_array):.4f}\n"
        report += f"  Max: {np.max(pos_dist_array):.4f}\n"
        report += f"  Median: {np.median(pos_dist_array):.4f}\n"
        report += f"  P90: {np.percentile(pos_dist_array, 90):.4f}\n\n"

        report += f"Negative Distances:\n"
        report += f"  Mean: {np.mean(neg_dist_array):.4f}\n"
        report += f"  Std: {np.std(neg_dist_array):.4f}\n"
        report += f"  Min: {np.min(neg_dist_array):.4f}\n"
        report += f"  Max: {np.max(neg_dist_array):.4f}\n"
        report += f"  Median: {np.median(neg_dist_array):.4f}\n"
        report += f"  P10: {np.percentile(neg_dist_array, 10):.4f}\n\n"

        report += f"Separation Margin: {np.mean(neg_dist_array) - np.mean(pos_dist_array):.4f}\n"

        return report


class ReconstructionMetricsCalculator:
    """
    Calculator for enhanced reconstruction metrics.

    Tracks per-class and per-metric reconstruction quality to identify:
    - Which postures are harder to reconstruct
    - Which quality metrics benefit most from distillation
    - Detail preservation vs. overall quality trade-offs
    """

    def __init__(self, num_classes: int = 5, class_names: Optional[list] = None):
        """
        Args:
            num_classes: Number of posture classes
            class_names: Names of classes (e.g., ['Front', 'Up', 'Down', 'Left', 'Right'])
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        self.reset()

    def reset(self):
        """Reset all accumulators for a new epoch"""
        # Per-class metrics
        self.ssim_per_class = defaultdict(list)
        self.psnr_per_class = defaultdict(list)

        # Global metrics
        self.all_ssim = []
        self.all_psnr = []

    def update_batch(
        self,
        ssim_scores: torch.Tensor,
        psnr_scores: torch.Tensor,
        posture_classes: torch.Tensor
    ):
        """
        Update metrics with a batch of reconstruction quality scores.

        Args:
            ssim_scores: SSIM scores [batch_size]
            psnr_scores: PSNR scores [batch_size]
            posture_classes: Posture class labels [batch_size]
        """
        # Move to CPU
        ssim = ssim_scores.detach().cpu().numpy()
        psnr = psnr_scores.detach().cpu().numpy()
        classes = posture_classes.detach().cpu().numpy()

        # Store global metrics
        self.all_ssim.extend(ssim.tolist())
        self.all_psnr.extend(psnr.tolist())

        # Store per-class metrics
        for i in range(len(classes)):
            class_id = int(classes[i])
            self.ssim_per_class[class_id].append(ssim[i])
            self.psnr_per_class[class_id].append(psnr[i])

    def compute_epoch_metrics(self) -> Dict[str, float]:
        """
        Compute aggregated reconstruction metrics for the entire epoch.

        Returns:
            Dictionary of metric names to values
        """
        metrics = {}

        if not self.all_ssim or not self.all_psnr:
            return metrics

        ssim_array = np.array(self.all_ssim)
        psnr_array = np.array(self.all_psnr)

        # 1. Global statistics
        metrics['recon_ssim_mean'] = np.mean(ssim_array)
        metrics['recon_ssim_std'] = np.std(ssim_array)
        metrics['recon_ssim_min'] = np.min(ssim_array)
        metrics['recon_ssim_max'] = np.max(ssim_array)

        metrics['recon_psnr_mean'] = np.mean(psnr_array)
        metrics['recon_psnr_std'] = np.std(psnr_array)
        metrics['recon_psnr_min'] = np.min(psnr_array)
        metrics['recon_psnr_max'] = np.max(psnr_array)

        # 2. Quality consistency (lower std = more consistent)
        metrics['recon_ssim_cv'] = metrics['recon_ssim_std'] / metrics['recon_ssim_mean'] if metrics['recon_ssim_mean'] > 0 else 0
        metrics['recon_psnr_cv'] = metrics['recon_psnr_std'] / metrics['recon_psnr_mean'] if metrics['recon_psnr_mean'] > 0 else 0

        # 3. Per-class SSIM
        for class_id, class_name in enumerate(self.class_names):
            if class_id in self.ssim_per_class and self.ssim_per_class[class_id]:
                metrics[f'recon_ssim_{class_name}'] = np.mean(self.ssim_per_class[class_id])
            else:
                metrics[f'recon_ssim_{class_name}'] = 0.0

        # 4. Per-class PSNR
        for class_id, class_name in enumerate(self.class_names):
            if class_id in self.psnr_per_class and self.psnr_per_class[class_id]:
                metrics[f'recon_psnr_{class_name}'] = np.mean(self.psnr_per_class[class_id])
            else:
                metrics[f'recon_psnr_{class_name}'] = 0.0

        # 5. Identify hardest postures to reconstruct
        ssim_by_class = {class_id: np.mean(scores) for class_id, scores in self.ssim_per_class.items() if scores}
        if ssim_by_class:
            hardest_class_id = min(ssim_by_class, key=ssim_by_class.get)
            metrics['recon_hardest_posture_ssim'] = ssim_by_class[hardest_class_id]
            metrics['recon_hardest_posture_id'] = hardest_class_id

            easiest_class_id = max(ssim_by_class, key=ssim_by_class.get)
            metrics['recon_easiest_posture_ssim'] = ssim_by_class[easiest_class_id]
            metrics['recon_easiest_posture_id'] = easiest_class_id

            # Range across postures
            metrics['recon_ssim_class_range'] = metrics['recon_easiest_posture_ssim'] - metrics['recon_hardest_posture_ssim']

        # 6. Failure analysis (samples with poor reconstruction)
        poor_ssim_threshold = 0.7  # Typically, SSIM < 0.7 is considered poor
        poor_reconstructions = ssim_array < poor_ssim_threshold
        metrics['recon_failure_rate_ssim'] = np.sum(poor_reconstructions) / len(ssim_array)

        # High SSIM samples (excellent reconstruction)
        excellent_ssim_threshold = 0.95
        excellent_reconstructions = ssim_array >= excellent_ssim_threshold
        metrics['recon_excellent_rate_ssim'] = np.sum(excellent_reconstructions) / len(ssim_array)

        return metrics

    def get_reconstruction_report(self) -> str:
        """
        Get detailed reconstruction statistics for logging.

        Returns:
            String containing reconstruction statistics
        """
        if not self.all_ssim or not self.all_psnr:
            return "No reconstruction data available"

        report = "Reconstruction Quality Statistics\n"
        report += "="*60 + "\n"

        # Global statistics
        report += f"Global Metrics:\n"
        report += f"  SSIM: {np.mean(self.all_ssim):.4f} ± {np.std(self.all_ssim):.4f}\n"
        report += f"  PSNR: {np.mean(self.all_psnr):.4f} ± {np.std(self.all_psnr):.4f}\n\n"

        # Per-class statistics
        report += f"Per-Class SSIM:\n"
        for class_id, class_name in enumerate(self.class_names):
            if class_id in self.ssim_per_class and self.ssim_per_class[class_id]:
                ssim_values = self.ssim_per_class[class_id]
                report += f"  {class_name}: {np.mean(ssim_values):.4f} ± {np.std(ssim_values):.4f} (n={len(ssim_values)})\n"

        report += f"\nPer-Class PSNR:\n"
        for class_id, class_name in enumerate(self.class_names):
            if class_id in self.psnr_per_class and self.psnr_per_class[class_id]:
                psnr_values = self.psnr_per_class[class_id]
                report += f"  {class_name}: {np.mean(psnr_values):.4f} ± {np.std(psnr_values):.4f} (n={len(psnr_values)})\n"

        return report


if __name__ == "__main__":
    # Test the metrics calculators
    print("Testing VerificationMetricsCalculator...")

    torch.manual_seed(42)

    # Simulate verification data
    batch_size = 50
    embed_dim = 128

    # Create embeddings (simulate good vs poor separation)
    anchor_emb = torch.randn(batch_size, embed_dim)
    positive_emb = anchor_emb + torch.randn(batch_size, embed_dim) * 0.3  # Close to anchor
    negative_emb = anchor_emb + torch.randn(batch_size, embed_dim) * 1.5  # Far from anchor

    # Test verification metrics
    ver_calc = VerificationMetricsCalculator()
    ver_calc.update_batch(anchor_emb, positive_emb, negative_emb)
    ver_metrics = ver_calc.compute_epoch_metrics()

    print("\n" + "="*60)
    print("Verification Metrics:")
    print("="*60)
    for key, value in ver_metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\n" + ver_calc.get_distance_statistics())

    # Test reconstruction metrics
    print("\n" + "="*60)
    print("Testing ReconstructionMetricsCalculator...")
    print("="*60)

    recon_calc = ReconstructionMetricsCalculator(
        num_classes=5,
        class_names=['Front', 'Up', 'Down', 'Left', 'Right']
    )

    # Simulate reconstruction scores with some postures harder to reconstruct
    ssim_scores = torch.rand(batch_size) * 0.3 + 0.7  # SSIM in [0.7, 1.0]
    psnr_scores = torch.rand(batch_size) * 10 + 20  # PSNR in [20, 30]
    posture_classes = torch.randint(0, 5, (batch_size,))

    # Make "Up" posture (class 1) harder to reconstruct
    up_mask = (posture_classes == 1)
    ssim_scores[up_mask] -= 0.15
    psnr_scores[up_mask] -= 5

    recon_calc.update_batch(ssim_scores, psnr_scores, posture_classes)
    recon_metrics = recon_calc.compute_epoch_metrics()

    print("\nReconstruction Metrics:")
    print("="*60)
    for key, value in recon_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\n" + recon_calc.get_reconstruction_report())

    print("\n✓ All metrics calculators tested successfully!")
