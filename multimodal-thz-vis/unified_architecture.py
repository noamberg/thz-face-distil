#!/usr/bin/env python3
"""
Privileged Information Multi-Task Architecture (V2)

Teacher-Student Distillation Framework with Dual Cross-Attention:
- Student (THz-only): Used for inference
- Teacher (THz + Visible fusion): Provides privileged supervision during training
- Dual cross-attention for bidirectional multimodal fusion:
  * Visible-guided: Q=Visible, K/V=THz (visible attends to THz)
  * THz-guided: Q=THz, K/V=Visible (THz attends to visible)
- Feature-level and logit-level distillation
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

# Import existing building blocks from parent module
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from mtl.unified_architecture import (
    BackboneEncoder,
    UNetDecoder,
    ResNetDownsampleBlock,
    ResidualDenseBlock,
    PixelShuffleUpsampler,
    BasicResidualBlock,
    MultiScaleResNetEncoder
)


class PrivilegedDistillationModel(nn.Module):
    """
    Multi-task model with privileged information learning via teacher-student distillation.

    Architecture:
        - THz Encoder (Student): Processes THz images alone (used at inference)
        - Visible Encoder (Teacher): Processes visible images (training only)
        - Dual Cross-Attention: Bidirectional fusion of THz and visible features
          * Visible-guided attention: Visible queries attend to THz keys/values
          * THz-guided attention: THz queries attend to visible keys/values
          * Fusion layer: Combines both attention outputs for richer multimodal representation
        - Student Heads: Operate on THz features only
        - Teacher Heads: Operate on fused features
        - Distillation: Teacher supervises student via feature and logit matching
    """

    def __init__(
        self,
        num_classes: int = 5,
        embed_dim: int = 128,
        backbone_type: str = "custom",
        posture_ids: list[int] = [0, 1, 2, 3, 4],
        input_size: int = 64,
        dropout_rate: float = 0.5,
        pretrained: bool = True,
        cross_attention_heads: int = 8,
        distill_dim: int = 256
    ):
        """
        Args:
            num_classes: Number of posture classes
            embed_dim: Embedding dimension for verification task
        backbone_type: Type of backbone encoder ('custom' or 'resnet18')
            posture_ids: List of posture class IDs
            input_size: Input image size
            dropout_rate: Dropout rate
            pretrained: Whether to use pretrained weights for backbone
            cross_attention_heads: Number of heads for cross-attention
            distill_dim: Dimension of distillation projection space
        """
        super().__init__()

        # --- Dual Encoders ---
        self.thz_encoder = BackboneEncoder(backbone_type=backbone_type, pretrained=pretrained)
        self.vis_encoder = BackboneEncoder(backbone_type=backbone_type, pretrained=pretrained)

        # Get feature dimension from backbone (512 for custom, 512 for resnet18)
        self.feature_dim = 512

        # Store config for logging
        self.cross_attention_heads = cross_attention_heads

        # --- Cross-Attention Blocks ---
        # Visible-guided: Q=Visible, K/V=THz (visible features attend to THz features)
        self.visible_guided_cross_attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=cross_attention_heads,
            dropout=0.1,
            batch_first=True
        )

        # THz-guided: Q=THz, K/V=Visible (THz features attend to visible features)
        self.thz_guided_cross_attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=cross_attention_heads,
            dropout=0.1,
            batch_first=True
        )

        # Fusion layer to combine both attention outputs
        self.attention_fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(inplace=True)
        )

        # --- Student Path (THz-only) ---
        # Spatial reduction for student: [N, 512, 4, 4] -> [N, 512, 1, 1]
        self.student_spatial_reduction = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.student_dropout = nn.Dropout(dropout_rate)

        # Student verification head
        hidden_dim = 256
        self.student_verification_head = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, embed_dim)
        )

        # Student classification head
        self.student_classification_head = nn.Linear(512, num_classes)

        # --- Teacher Path (Fused) ---
        # Spatial reduction for teacher: [N, 512, 4, 4] -> [N, 512, 1, 1]
        self.teacher_spatial_reduction = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.teacher_dropout = nn.Dropout(dropout_rate)

        # Teacher verification head
        self.teacher_verification_head = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, embed_dim)
        )

        # Teacher classification head
        self.teacher_classification_head = nn.Linear(512, num_classes)

        # --- Distillation Projectors ---
        # Project student and teacher latent vectors into a common space
        self.student_distill_projection = nn.Linear(512, distill_dim)
        self.teacher_distill_projection = nn.Linear(512, distill_dim)

        # --- Reconstruction Decoders (Mixture-of-Experts) ---
        add_final_upsample_stage = (backbone_type == 'resnet18')
        self.reconstruction_decoders = nn.ModuleDict({
            str(pid): UNetDecoder(
                out_channels=1,
                num_rdb=1,
                add_final_upsample_stage=add_final_upsample_stage
            ) for pid in posture_ids
        })

    def freeze_teacher(self):
        """
        Freeze teacher model parameters to prevent gradient updates.
        This includes: visible encoder, cross-attention modules, fusion layer, and teacher heads.
        Call this from the training script when config specifies freeze_teacher=True.
        """
        # Freeze visible encoder
        for param in self.vis_encoder.parameters():
            param.requires_grad = False

        # Freeze cross-attention modules
        for param in self.visible_guided_cross_attention.parameters():
            param.requires_grad = False
        for param in self.thz_guided_cross_attention.parameters():
            param.requires_grad = False

        # Freeze attention fusion
        for param in self.attention_fusion.parameters():
            param.requires_grad = False

        # Freeze teacher heads
        for param in self.teacher_spatial_reduction.parameters():
            param.requires_grad = False
        for param in self.teacher_verification_head.parameters():
            param.requires_grad = False
        for param in self.teacher_classification_head.parameters():
            param.requires_grad = False

        # Freeze teacher distillation projection
        for param in self.teacher_distill_projection.parameters():
            param.requires_grad = False

    def unfreeze_teacher(self):
        """
        Unfreeze teacher model parameters to enable gradient updates.
        This reverses the effect of freeze_teacher().
        Call this from the training script when config specifies freeze_teacher=False.
        """
        # Unfreeze visible encoder
        for param in self.vis_encoder.parameters():
            param.requires_grad = True

        # Unfreeze cross-attention modules
        for param in self.visible_guided_cross_attention.parameters():
            param.requires_grad = True
        for param in self.thz_guided_cross_attention.parameters():
            param.requires_grad = True

        # Unfreeze attention fusion
        for param in self.attention_fusion.parameters():
            param.requires_grad = True

        # Unfreeze teacher heads
        for param in self.teacher_spatial_reduction.parameters():
            param.requires_grad = True
        for param in self.teacher_verification_head.parameters():
            param.requires_grad = True
        for param in self.teacher_classification_head.parameters():
            param.requires_grad = True

        # Unfreeze teacher distillation projection
        for param in self.teacher_distill_projection.parameters():
            param.requires_grad = True

    def _tokenize_features(self, feat4: torch.Tensor) -> torch.Tensor:
        """
        Convert feature map to sequence of tokens for attention.
        Args:
            feat4: Feature map [N, C, H, W]
        Returns:
            tokens: Token sequence [N, H*W, C]
        """
        N, C, H, W = feat4.shape
        tokens = feat4.view(N, C, H * W).permute(0, 2, 1)  # [N, H*W, C]
        return tokens

    def _pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Pool tokens to get a single latent vector.
        Args:
            tokens: Token sequence [N, num_tokens, C]
        Returns:
            latent: Latent vector [N, C]
        """
        return tokens.mean(dim=1)  # Average pooling over spatial tokens

    def forward(
        self,
        anchor_thz: torch.Tensor,
        positive_thz: torch.Tensor,
        negative_thz: torch.Tensor,
        anchor_posture_ids: torch.Tensor,
        visible_concealed: Optional[torch.Tensor] = None,
        visible_unconcealed: Optional[torch.Tensor] = None,
        visible_negative: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with teacher-student distillation.

        Args:
            anchor_thz: THz anchor images [B, 1, H, W]
            positive_thz: THz positive images [B, 1, H, W]
            negative_thz: THz negative images [B, 1, H, W]
            anchor_posture_ids: Posture class IDs for anchors [B]
            visible_concealed: Visible anchor images (privileged) [B, 1, H, W] (optional, training only)
            visible_unconcealed: Visible positive images (privileged) [B, 1, H, W] (optional, training only)
            visible_negative: Visible negative images (privileged) [B, 1, H, W] (optional, training only)

        Returns:
            Dictionary with student and teacher outputs + distillation targets
        """
        batch_size = anchor_thz.size(0)

        # Concatenate triplets for efficient processing
        combined_thz = torch.cat([anchor_thz, positive_thz, negative_thz], dim=0)

        # --- Student Path (THz-only) ---
        # Extract THz features through student encoder
        thz_feat1, thz_feat2, thz_feat3, thz_feat4 = self.thz_encoder(combined_thz)

        # Tokenize THz features
        thz_tokens = self._tokenize_features(thz_feat4)  # [3B, H*W, 512]

        # Pool THz tokens to get student latent vectors
        thz_latent_combined = self._pool_tokens(thz_tokens)  # [3B, 512]

        # Apply spatial reduction and dropout
        thz_feat4_reduced = self.student_spatial_reduction(thz_feat4).flatten(1)  # [3B, 512]
        thz_feat4_reduced = self.student_dropout(thz_feat4_reduced)

        # Student verification embeddings
        student_embeddings = F.normalize(
            self.student_verification_head(thz_feat4_reduced), p=2, dim=1
        )  # [3B, embed_dim]

        # Student classification logits
        student_posture_logits = self.student_classification_head(thz_feat4_reduced)  # [3B, num_classes]

        # Split back into anchor, positive, negative
        student_anchor_emb = student_embeddings[:batch_size]
        student_positive_emb = student_embeddings[batch_size:2*batch_size]
        student_negative_emb = student_embeddings[2*batch_size:]

        student_anchor_logits = student_posture_logits[:batch_size]
        student_positive_logits = student_posture_logits[batch_size:2*batch_size]
        student_negative_logits = student_posture_logits[2*batch_size:]

        # Get student latent for distillation (anchor only)
        student_anchor_latent = thz_latent_combined[:batch_size]  # [B, 512]

        # --- Reconstruction (using student encoder features) ---
        # Extract anchor features for reconstruction
        anchor_thz_feat1 = thz_feat1[:batch_size]
        anchor_thz_feat2 = thz_feat2[:batch_size]
        anchor_thz_feat3 = thz_feat3[:batch_size]
        anchor_thz_feat4 = thz_feat4[:batch_size]

        # MoE Reconstruction: route to expert decoder based on posture
        reconstructed_output = torch.zeros_like(anchor_thz)
        for pid in torch.unique(anchor_posture_ids):
            mask = (anchor_posture_ids == pid)
            decoder = self.reconstruction_decoders[str(pid.item())]
            recon_result = decoder(
                anchor_thz_feat4[mask],
                anchor_thz_feat3[mask],
                anchor_thz_feat2[mask],
                anchor_thz_feat1[mask]
            )
            reconstructed_output[mask] = recon_result

        # --- Teacher Path (only if visible images are provided) ---
        teacher_outputs = {}
        has_any_visible = (visible_concealed is not None or visible_unconcealed is not None or visible_negative is not None)

        if has_any_visible:
            # Combine all three visible images for efficient processing
            combined_visible = torch.cat([
                visible_concealed if visible_concealed is not None else torch.zeros_like(anchor_thz),
                visible_unconcealed if visible_unconcealed is not None else torch.zeros_like(positive_thz),
                visible_negative if visible_negative is not None else torch.zeros_like(negative_thz)
            ], dim=0)  # [3B, 1, H, W]

            # Extract visible features through teacher encoder for all three
            vis_feat1, vis_feat2, vis_feat3, vis_feat4 = self.vis_encoder(combined_visible)  # [3B, 512, H', W']

            # Split features back into anchor, positive, negative
            vis_anchor_feat4 = vis_feat4[:batch_size]
            vis_positive_feat4 = vis_feat4[batch_size:2*batch_size]
            vis_negative_feat4 = vis_feat4[2*batch_size:]

            # Split THz features (already extracted earlier)
            thz_anchor_feat4 = thz_feat4[:batch_size]
            thz_positive_feat4 = thz_feat4[batch_size:2*batch_size]
            thz_negative_feat4 = thz_feat4[2*batch_size:]

            # Process each triplet element separately with DUAL cross-attention
            def process_teacher_branch(vis_feat, thz_feat):
                """
                Helper to process one visible-THz pair through teacher branch with dual cross-attention.
                Uses both visible-guided and THz-guided cross-attention for richer fusion.
                """
                # Tokenize features
                vis_tokens = self._tokenize_features(vis_feat)  # [B, H*W, 512]
                thz_tokens = self._tokenize_features(thz_feat)  # [B, H*W, 512]

                # Visible-guided cross-attention: Q=Visible, K/V=THz
                # (Visible features attend to THz features)
                vis_guided_tokens, _ = self.visible_guided_cross_attention(
                    query=vis_tokens,
                    key=thz_tokens,
                    value=thz_tokens
                )  # [B, H*W, 512]

                # THz-guided cross-attention: Q=THz, K/V=Visible
                # (THz features attend to visible features)
                thz_guided_tokens, _ = self.thz_guided_cross_attention(
                    query=thz_tokens,
                    key=vis_tokens,
                    value=vis_tokens
                )  # [B, H*W, 512]

                # Fuse both attention outputs
                # Concatenate and project to combine complementary information
                dual_attention_tokens = torch.cat([vis_guided_tokens, thz_guided_tokens], dim=-1)  # [B, H*W, 1024]
                fused_tokens = self.attention_fusion(dual_attention_tokens)  # [B, H*W, 512]

                # Pool and reduce
                teacher_latent = self._pool_tokens(fused_tokens)  # [B, 512]

                # Reshape for spatial reduction
                H = W = int((fused_tokens.shape[1]) ** 0.5)
                fused_feat = fused_tokens.permute(0, 2, 1).view(batch_size, 512, H, W)
                fused_feat_reduced = self.teacher_spatial_reduction(fused_feat).flatten(1)
                fused_feat_reduced = self.teacher_dropout(fused_feat_reduced)

                # Generate embeddings and logits
                embedding = F.normalize(self.teacher_verification_head(fused_feat_reduced), p=2, dim=1)
                logits = self.teacher_classification_head(fused_feat_reduced)

                return embedding, logits, teacher_latent

            # Process all three branches
            teacher_anchor_emb, teacher_anchor_logits, teacher_anchor_latent = process_teacher_branch(
                vis_anchor_feat4, thz_anchor_feat4
            )
            teacher_positive_emb, teacher_positive_logits, teacher_positive_latent = process_teacher_branch(
                vis_positive_feat4, thz_positive_feat4
            )
            teacher_negative_emb, teacher_negative_logits, teacher_negative_latent = process_teacher_branch(
                vis_negative_feat4, thz_negative_feat4
            )

            teacher_outputs = {
                # Teacher embeddings for verification (triplet)
                'teacher_anchor_embedding': teacher_anchor_emb,
                'teacher_positive_embedding': teacher_positive_emb,
                'teacher_negative_embedding': teacher_negative_emb,
                # Teacher classification logits
                'teacher_anchor_posture_logits': teacher_anchor_logits,
                'teacher_positive_posture_logits': teacher_positive_logits,
                'teacher_negative_posture_logits': teacher_negative_logits,
                # Teacher latent (for distillation - use anchor)
                'teacher_latent': teacher_anchor_latent
            }

        # --- Distillation Projections ---
        # Project student and teacher latents into distillation space
        student_distill_proj = self.student_distill_projection(student_anchor_latent)  # [B, distill_dim]

        if has_any_visible and 'teacher_latent' in teacher_outputs:
            teacher_distill_proj = self.teacher_distill_projection(teacher_outputs['teacher_latent'])  # [B, distill_dim]
            teacher_outputs['teacher_distill_proj'] = teacher_distill_proj

        # --- Combine Outputs ---
        outputs = {
            # Student outputs (always available)
            'anchor_embedding': student_anchor_emb,
            'positive_embedding': student_positive_emb,
            'negative_embedding': student_negative_emb,
            'anchor_posture_logits': student_anchor_logits,
            'positive_posture_logits': student_positive_logits,
            'negative_posture_logits': student_negative_logits,
            'reconstructed_image': reconstructed_output,
            'student_distill_proj': student_distill_proj,

            # Metadata
            'has_teacher': has_any_visible
        }

        # Add teacher outputs if available
        outputs.update(teacher_outputs)

        return outputs


class PrivilegedMultiTaskModel(nn.Module):
    """
    Wrapper class for backward compatibility with existing training scripts.
    Can be used as a drop-in replacement for UnifiedMultiTaskModel.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.model = PrivilegedDistillationModel(**kwargs)

    def forward(self, anchor, positive, negative, anchor_posture_ids,
                visible_concealed=None, visible_unconcealed=None, visible_negative=None):
        return self.model(anchor, positive, negative, anchor_posture_ids,
                         visible_concealed, visible_unconcealed, visible_negative)


if __name__ == "__main__":
    # Test the Privileged Distillation Model
    print("Testing PrivilegedDistillationModel...")

    # Create model
    model = PrivilegedDistillationModel(
        num_classes=5,
        embed_dim=128,
        backbone_type='custom',
        posture_ids=[0, 1, 2, 3, 4],
        input_size=64,
        dropout_rate=0.5,
        pretrained=False,
        cross_attention_heads=8,
        distill_dim=256
    )

    # Test inputs
    batch_size = 4
    anchor_thz = torch.randn(batch_size, 1, 64, 64)
    positive_thz = torch.randn(batch_size, 1, 64, 64)
    negative_thz = torch.randn(batch_size, 1, 64, 64)
    anchor_posture_ids = torch.randint(0, 5, (batch_size,))
    visible_unconcealed = torch.randn(batch_size, 1, 64, 64)

    # Test forward pass with visible images (training mode)
    print("\n1. Testing forward pass WITH visible images (training mode):")
    outputs = model(anchor_thz, positive_thz, negative_thz, anchor_posture_ids, visible_unconcealed)

    print(f"  Student anchor embedding shape: {outputs['anchor_embedding'].shape}")
    print(f"  Student classification logits shape: {outputs['anchor_posture_logits'].shape}")
    print(f"  Reconstructed image shape: {outputs['reconstructed_image'].shape}")
    print(f"  Student distill projection shape: {outputs['student_distill_proj'].shape}")
    print(f"  Has teacher: {outputs['has_teacher']}")
    if outputs['has_teacher']:
        print(f"  Teacher anchor embedding shape: {outputs['teacher_anchor_embedding'].shape}")
        print(f"  Teacher classification logits shape: {outputs['teacher_anchor_posture_logits'].shape}")
        print(f"  Teacher distill projection shape: {outputs['teacher_distill_proj'].shape}")

    # Test forward pass without visible images (inference mode)
    print("\n2. Testing forward pass WITHOUT visible images (inference mode):")
    outputs_inf = model(anchor_thz, positive_thz, negative_thz, anchor_posture_ids, visible_unconcealed=None)

    print(f"  Student anchor embedding shape: {outputs_inf['anchor_embedding'].shape}")
    print(f"  Student classification logits shape: {outputs_inf['anchor_posture_logits'].shape}")
    print(f"  Reconstructed image shape: {outputs_inf['reconstructed_image'].shape}")
    print(f"  Has teacher: {outputs_inf['has_teacher']}")

    print("\nModel parameter count:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    print("\nPrivilegedDistillationModel test complete!")
