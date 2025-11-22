#!/usr/bin/env python3
"""
Enhanced Classification Metrics for Posture Recognition

Provides comprehensive metrics to measure:
1. Decisiveness: How confident the model is in its predictions
2. Separability: How well-separated the latent space clusters are

These metrics help quantify the "certainty gap" between models trained with
and without privileged information distillation.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from scipy.stats import entropy
from sklearn.metrics import silhouette_score, f1_score, classification_report


class ClassificationMetricsCalculator:
    """
    Calculator for enhanced classification metrics.

    Tracks batch-level statistics and computes epoch-level aggregations for:
    - Confidence on correct/incorrect predictions
    - Prediction entropy (uncertainty)
    - Silhouette score (latent space separability)
    - Per-class F1 scores
    """

    def __init__(self, num_classes: int = 5, class_names: Optional[list] = None):
        """
        Args:
            num_classes: Number of classification classes
            class_names: Names of classes (e.g., ['Front', 'Up', 'Down', 'Left', 'Right'])
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]

        # Batch-level accumulators
        self.reset()

    def reset(self):
        """Reset all accumulators for a new epoch"""
        self.correct_confidences = []
        self.incorrect_confidences = []
        self.all_entropies = []
        self.all_embeddings = []
        self.all_targets = []
        self.all_predictions = []

    def update_batch(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None
    ):
        """
        Update metrics with a batch of predictions.

        Args:
            logits: Model output logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            embeddings: Optional embeddings for silhouette computation [batch_size, embed_dim]
        """
        # Move to CPU and convert to numpy for metric computation
        logits = logits.detach().cpu()
        targets = targets.detach().cpu()

        # Get predictions and probabilities
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        # Get confidence (probability of predicted class)
        pred_confidences = probs[torch.arange(probs.size(0)), preds]

        # Separate correct and incorrect predictions
        correct_mask = (preds == targets)

        # A. Decisiveness Metrics
        # 1. Confidence on correct predictions
        if correct_mask.any():
            correct_confs = pred_confidences[correct_mask].numpy()
            self.correct_confidences.extend(correct_confs.tolist())

        # 2. Confidence on incorrect predictions
        if (~correct_mask).any():
            incorrect_confs = pred_confidences[~correct_mask].numpy()
            self.incorrect_confidences.extend(incorrect_confs.tolist())

        # 3. Prediction entropy (uncertainty)
        # Shannon entropy: -sum(p * log(p))
        probs_np = probs.numpy()
        batch_entropies = entropy(probs_np.T, base=2)  # base=2 for bits
        self.all_entropies.extend(batch_entropies.tolist())

        # Store predictions and targets for per-class F1
        self.all_predictions.extend(preds.numpy().tolist())
        self.all_targets.extend(targets.numpy().tolist())

        # B. Separability Metrics (collect embeddings for epoch-level computation)
        if embeddings is not None:
            embeddings = embeddings.detach().cpu().numpy()
            self.all_embeddings.append(embeddings)
            # Note: Silhouette score computed at epoch level, not batch level

    def compute_epoch_metrics(self) -> Dict[str, float]:
        """
        Compute aggregated metrics for the entire epoch.

        Returns:
            Dictionary of metric names to values
        """
        metrics = {}

        # A. Decisiveness Metrics
        # 1. Average confidence on correct predictions
        if self.correct_confidences:
            metrics['cls_conf_correct'] = np.mean(self.correct_confidences)
            metrics['cls_conf_correct_std'] = np.std(self.correct_confidences)
        else:
            metrics['cls_conf_correct'] = 0.0
            metrics['cls_conf_correct_std'] = 0.0

        # 2. Average confidence on incorrect predictions
        if self.incorrect_confidences:
            metrics['cls_conf_error'] = np.mean(self.incorrect_confidences)
            metrics['cls_conf_error_std'] = np.std(self.incorrect_confidences)
        else:
            metrics['cls_conf_error'] = 0.0
            metrics['cls_conf_error_std'] = 0.0

        # 3. Average prediction entropy
        if self.all_entropies:
            metrics['cls_entropy'] = np.mean(self.all_entropies)
            metrics['cls_entropy_std'] = np.std(self.all_entropies)
        else:
            metrics['cls_entropy'] = 0.0
            metrics['cls_entropy_std'] = 0.0

        # B. Separability Metrics
        # 4. Silhouette score (requires all embeddings from epoch)
        if self.all_embeddings and len(self.all_embeddings) > 0:
            # Concatenate all embeddings
            all_emb = np.concatenate(self.all_embeddings, axis=0)
            all_tgt = np.array(self.all_targets)

            # Silhouette score requires at least 2 samples per class
            unique_classes = np.unique(all_tgt)
            if len(unique_classes) > 1 and all_emb.shape[0] > len(unique_classes):
                try:
                    silhouette = silhouette_score(all_emb, all_tgt, metric='euclidean')
                    metrics['cls_silhouette'] = silhouette
                except Exception as e:
                    print(f"Warning: Could not compute silhouette score: {e}")
                    metrics['cls_silhouette'] = 0.0
            else:
                metrics['cls_silhouette'] = 0.0
        else:
            metrics['cls_silhouette'] = 0.0

        # 5. Per-class F1 scores
        if self.all_predictions and self.all_targets:
            all_preds = np.array(self.all_predictions)
            all_tgts = np.array(self.all_targets)

            # Compute F1 per class
            f1_per_class = f1_score(all_tgts, all_preds, average=None, zero_division=0)

            # Add per-class F1 to metrics
            for i, class_name in enumerate(self.class_names):
                if i < len(f1_per_class):
                    metrics[f'cls_f1_{class_name}'] = f1_per_class[i]

            # Also add macro and weighted F1
            metrics['cls_f1_macro'] = f1_score(all_tgts, all_preds, average='macro', zero_division=0)
            metrics['cls_f1_weighted'] = f1_score(all_tgts, all_preds, average='weighted', zero_division=0)

        return metrics

    def get_classification_report(self) -> str:
        """
        Get detailed classification report for logging.

        Returns:
            String containing classification report
        """
        if self.all_predictions and self.all_targets:
            all_preds = np.array(self.all_predictions)
            all_tgts = np.array(self.all_targets)

            report = classification_report(
                all_tgts,
                all_preds,
                target_names=self.class_names,
                zero_division=0
            )
            return report
        return "No predictions available"


def compute_classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    embeddings: Optional[torch.Tensor] = None,
    num_classes: int = 5,
    class_names: Optional[list] = None
) -> Dict[str, float]:
    """
    Convenience function to compute all classification metrics in one call.

    Args:
        logits: Model output logits [batch_size, num_classes]
        targets: Ground truth labels [batch_size]
        embeddings: Optional embeddings [batch_size, embed_dim]
        num_classes: Number of classes
        class_names: Names of classes

    Returns:
        Dictionary of metrics
    """
    calculator = ClassificationMetricsCalculator(num_classes, class_names)
    calculator.update_batch(logits, targets, embeddings)
    return calculator.compute_epoch_metrics()


if __name__ == "__main__":
    # Test the metrics calculator
    print("Testing ClassificationMetricsCalculator...")

    # Simulate some predictions
    torch.manual_seed(42)

    # Create test data
    num_samples = 100
    num_classes = 5
    embed_dim = 128

    # Simulate logits (some confident, some not)
    logits = torch.randn(num_samples, num_classes) * 2
    logits[:50, 0] += 3  # Make first 50 samples confident about class 0

    # Create targets (first 45 correct, next 5 incorrect, rest random)
    targets = torch.zeros(num_samples, dtype=torch.long)
    targets[:45] = 0  # Correct predictions
    targets[45:50] = 1  # Incorrect predictions (model predicts 0, truth is 1)
    targets[50:] = torch.randint(0, num_classes, (50,))

    # Create embeddings (simulating well-separated clusters)
    embeddings = torch.randn(num_samples, embed_dim)
    for i in range(num_classes):
        mask = targets == i
        embeddings[mask] += torch.randn(1, embed_dim) * 5  # Shift each class

    # Compute metrics
    calculator = ClassificationMetricsCalculator(
        num_classes=5,
        class_names=['Front', 'Up', 'Down', 'Left', 'Right']
    )

    # Simulate batch updates
    batch_size = 20
    for i in range(0, num_samples, batch_size):
        batch_logits = logits[i:i+batch_size]
        batch_targets = targets[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]
        calculator.update_batch(batch_logits, batch_targets, batch_embeddings)

    # Compute epoch metrics
    metrics = calculator.compute_epoch_metrics()

    print("\n" + "="*60)
    print("Computed Metrics:")
    print("="*60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "="*60)
    print("Classification Report:")
    print("="*60)
    print(calculator.get_classification_report())

    print("\n✓ Metrics calculator test complete!")
