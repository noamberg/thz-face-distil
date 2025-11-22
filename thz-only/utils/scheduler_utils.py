#!/usr/bin/env python3
"""
Learning Rate Schedulers with Warmup
Provides warmup + cosine annealing for stable training
"""

import torch
import math
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineAnnealingLR(_LRScheduler):
    """
    Cosine Annealing with Linear Warmup.

    Learning rate schedule:
    1. Linear warmup from 0 to base_lr over warmup_epochs
    2. Cosine annealing from base_lr to min_lr over remaining epochs

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of epochs for warmup
        total_epochs: Total number of training epochs
        min_lr: Minimum learning rate (default: 1e-6)
        last_epoch: The index of last epoch (default: -1)
    """

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class WarmupMultiStepLR(_LRScheduler):
    """
    Multi-step LR decay with Linear Warmup.

    Learning rate schedule:
    1. Linear warmup from 0 to base_lr over warmup_epochs
    2. Multiply LR by gamma at each milestone

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of epochs for warmup
        milestones: List of epoch indices for LR decay
        gamma: Multiplicative factor of learning rate decay (default: 0.1)
        last_epoch: The index of last epoch (default: -1)
    """

    def __init__(self, optimizer, warmup_epochs, milestones, gamma=0.1, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.milestones = sorted(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Multi-step decay
            decay_factor = self.gamma ** sum([self.last_epoch >= m for m in self.milestones])
            return [base_lr * decay_factor for base_lr in self.base_lrs]


def create_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    """
    Factory function to create warmup + cosine annealing scheduler.

    Example usage:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = create_warmup_cosine_scheduler(
            optimizer,
            warmup_epochs=10,
            total_epochs=200,
            min_lr=1e-6
        )

        # In training loop:
        for epoch in range(total_epochs):
            train_epoch()
            scheduler.step()
    """
    return WarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=total_epochs,
        min_lr=min_lr
    )


def create_warmup_multistep_scheduler(optimizer, warmup_epochs, milestones, gamma=0.1):
    """
    Factory function to create warmup + multi-step scheduler.

    Example usage:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = create_warmup_multistep_scheduler(
            optimizer,
            warmup_epochs=10,
            milestones=[100, 150, 200],
            gamma=0.5
        )
    """
    return WarmupMultiStepLR(
        optimizer,
        warmup_epochs=warmup_epochs,
        milestones=milestones,
        gamma=gamma
    )


if __name__ == "__main__":
    # Test the schedulers
    import matplotlib.pyplot as plt
    import numpy as np

    # Create dummy optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Test Warmup + Cosine Annealing
    print("Testing Warmup + Cosine Annealing Scheduler...")
    scheduler = create_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=10,
        total_epochs=200,
        min_lr=1e-6
    )

    lrs_cosine = []
    for epoch in range(200):
        lrs_cosine.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    # Test Warmup + Multi-Step
    optimizer2 = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler2 = create_warmup_multistep_scheduler(
        optimizer2,
        warmup_epochs=10,
        milestones=[50, 100, 150],
        gamma=0.5
    )

    lrs_multistep = []
    for epoch in range(200):
        lrs_multistep.append(optimizer2.param_groups[0]['lr'])
        scheduler2.step()

    # Plot both schedulers
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(lrs_cosine, linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Learning Rate')
    ax1.set_title('Warmup (10 epochs) + Cosine Annealing')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    ax2.plot(lrs_multistep, linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Warmup (10 epochs) + Multi-Step [50,100,150]')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig('scheduler_comparison.png', dpi=150)
    print("Saved scheduler visualization to scheduler_comparison.png")

    print("\nScheduler values at key epochs:")
    for epoch in [0, 5, 10, 50, 100, 150, 199]:
        print(f"Epoch {epoch:3d}: Cosine={lrs_cosine[epoch]:.2e}, MultiStep={lrs_multistep[epoch]:.2e}")