#!/usr/bin/env python3
"""
Triplets-Adapted Breakthrough Architecture
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import Dict, Tuple

# ============================
# Utility / building blocks
# ============================

class ResNetDownsampleBlock(nn.Module):
    """A ResNet-style block that downsamples the input by a factor of 2."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # First convolution uses stride=2 to downsample the input
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        # Second convolution processes features at the new, smaller resolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # This block is used to match the dimensions of the input (x) to the dimensions of the main path's output.
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            # If channels change, we need a 1x1 convolution to match them.
            # We also use stride=2 to match the spatial downsampling.
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        elif in_channels == out_channels:
            # If channels are the same, we can use pooling to downsample.
            # AvgPool is often preferred in the shortcut over MaxPool.
            self.shortcut = nn.AvgPool2d(kernel_size=2, stride=2)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass input through the main path
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Pass input through the shortcut path and add it to the main path's output
        out += self.shortcut(x)

        # Final activation
        return self.relu(out)

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block."""
    def __init__(self, channels: int, growth: int = 32, rdb_weight: float = 0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(5):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(channels + i * growth, growth, 3, 1, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
        self.local_fusion = nn.Conv2d(channels + 5 * growth, channels, 1)
        self.rdb_weight = rdb_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x]
        for layer in self.layers:
            out = layer(torch.cat(feats, dim=1))
            feats.append(out)
        fused = self.local_fusion(torch.cat(feats, dim=1))
        return (fused * self.rdb_weight) + (x * self.rdb_weight)

class PixelShuffleUpsampler(nn.Module):
    """Upsample by 2 using PixelShuffle."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, 3, 1, 1)
        self.pixelshuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pixelshuffle(self.conv(x))

class BasicResidualBlock(nn.Module):
    """A residual block with two convolutional layers and a skip connection."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        return F.relu(out)

class MultiScaleResNetEncoder(nn.Module):
    """A custom ResNet-style encoder backbone for 64x64 images."""
    def __init__(self):
        super().__init__()
        base_channels = 64

        # Initial convolution layer remains the same
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Corrected encoder stages for the desired channel flow
        self.stage1 = ResNetDownsampleBlock(base_channels, base_channels)          # 64 -> 64
        self.stage2 = ResNetDownsampleBlock(base_channels, base_channels * 2)      # 64 -> 128
        self.stage3 = ResNetDownsampleBlock(base_channels * 2, base_channels * 4)  # 128 -> 256
        self.stage4 = ResNetDownsampleBlock(base_channels * 4, base_channels * 8)  # 256 -> 512

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Pass input through the initial convolution first
        x = self.initial_conv(x)

        # Then pass it through the encoder stages
        feat1 = self.stage1(x)
        feat2 = self.stage2(feat1)
        feat3 = self.stage3(feat2)
        feat4 = self.stage4(feat3)

        return feat1, feat2, feat3, feat4

class UNetDecoder(nn.Module):
    """Simplified decoder for reconstruction."""
    def __init__(self, out_channels: int = 1, num_rdb: int = 1, add_final_upsample_stage: bool = False):
        super().__init__()

        # Bottleneck: reduce channels and enrich features
        self.channel_reduction = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.bottleneck_rdb = ResidualDenseBlock(256)

        self.up4 = PixelShuffleUpsampler(256, 128)
        self.conv4 = nn.Sequential(nn.Conv2d(384, 128, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True))
        self.up3 = PixelShuffleUpsampler(128, 64)
        self.conv3 = nn.Sequential(nn.Conv2d(192, 64, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True))
        self.up2 = PixelShuffleUpsampler(64, 32)
        self.conv2 = nn.Sequential(nn.Conv2d(96, 32, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True))
        self.up1 = PixelShuffleUpsampler(32, 16)

        # Add conditional logic for final layers
        self.final_conv = nn.Sequential(nn.Conv2d(16, out_channels, 3, 1, 1), nn.Sigmoid())

    def forward(self, feat4: torch.Tensor, feat3: torch.Tensor, feat2: torch.Tensor, feat1: torch.Tensor) -> torch.Tensor:
        # Apply bottleneck: channel reduction + RDB enrichment
        x = self.channel_reduction(feat4)
        x = self.bottleneck_rdb(x)

        x = self.up4(x)
        x = torch.cat([x, feat3], dim=1)
        x = self.conv4(x)

        x = self.up3(x)
        x = torch.cat([x, feat2], dim=1)
        x = self.conv3(x)

        x = self.up2(x)
        x = torch.cat([x, feat1], dim=1)
        x = self.conv2(x)

        x = self.up1(x)

        return self.final_conv(x)

class BackboneEncoder(nn.Module):
    """A unified encoder that supports multiple backbones."""
    def __init__(self, backbone_type: str = "custom", pretrained: bool = True):
        super().__init__()
        self.backbone_type = backbone_type

        if backbone_type == "custom":
            self.backbone = MultiScaleResNetEncoder()
        elif backbone_type == "resnet18":
            resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self.stage0 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False), resnet.bn1, resnet.relu, resnet.maxpool)
            self.stage1 = resnet.layer1
            self.stage2 = resnet.layer2
            self.stage3 = resnet.layer3
            self.stage4 = resnet.layer4
        else:
            raise ValueError(f"Unsupported backbone: {backbone_type}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.backbone_type == "custom":
            # The CustomResNetEncoder already returns the 4 feature maps
            return self.backbone(x)

        # Logic for resnet18 to extract and return the 4 feature maps
        x = self.stage0(x)
        feat1 = self.stage1(x)
        feat2 = self.stage2(feat1)
        feat3 = self.stage3(feat2)
        feat4 = self.stage4(feat3)
        return feat1, feat2, feat3, feat4

class TripletMultiTaskNetwork(nn.Module):
    """Efficient multi-task network."""
    def __init__(self, num_classes: int = 5, base_channels: int = 64, backbone_type: str = "custom", input_size: int = 64, embed_dim: int = 128, dropout_rate: float = 0.5, pretrained: bool = True):
        super().__init__()

        # The backbone logic is now encapsulated in the Encoder class
        self.backbone = BackboneEncoder(backbone_type=backbone_type, pretrained=pretrained)

        self.reconstruction_decoder = UNetDecoder(out_channels=1)

        # Learnable spatial reduction: [N, 512, 4, 4] -> [N, 512, 1, 1]
        self.spatial_reduction = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dropout = nn.Dropout(dropout_rate)

        # Multi-layer verification head for better embeddings
        hidden_dim = 256
        self.verification_head = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.classification_head = nn.Linear(512, num_classes)

    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Feature extraction is now a simple call to the backbone
        return self.backbone(x)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> Dict[str, torch.Tensor]:
        combined_input = torch.cat([anchor, positive, negative], dim=0)
        feat1, feat2, feat3, feat4 = self.extract_features(combined_input)

        reduced_feat = self.spatial_reduction(feat4).flatten(1)
        reduced_feat = self.dropout(reduced_feat)

        embeddings = F.normalize(self.verification_head(reduced_feat), p=2, dim=1)
        posture_logits = self.classification_head(reduced_feat)

        batch_size = anchor.size(0)
        return {
            'anchor_embedding': embeddings[:batch_size],
            'positive_embedding': embeddings[batch_size:2*batch_size],
            'negative_embedding': embeddings[2*batch_size:],
            'anchor_posture_logits': posture_logits[:batch_size],
            'positive_posture_logits': posture_logits[batch_size:2*batch_size],
            'negative_posture_logits': posture_logits[2*batch_size:],
            'anchor_features': (feat1[:batch_size], feat2[:batch_size], feat3[:batch_size], feat4[:batch_size])
        }

class TripletDiscriminativeModel(nn.Module):
    """Dedicated model for Stage 1: Expert encoder training"""
    def __init__(self, **kwargs):
        super().__init__()
        self.base_network = TripletMultiTaskNetwork(**kwargs)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.base_network(anchor, positive, negative)

class PostureSpecificReconstructionModel(nn.Module):
    """Model for Stage 2: Mixture-of-Experts Reconstruction task."""
    def __init__(self, pretrained_discriminative_network: TripletDiscriminativeModel, posture_ids: list[int]):
        super().__init__()
        self.base_network = pretrained_discriminative_network.base_network
        
        # for param in self.base_network.backbone.parameters():
        for param in self.base_network.parameters():
            param.requires_grad = False
        
        # Determine if the decoder needs an extra upsampling stage based on the backbone
        backbone_type = self.base_network.backbone.backbone_type
        add_final_upsample_stage = (backbone_type == 'resnet18')
            
        self.reconstruction_decoders = nn.ModuleDict({
            str(pid): UNetDecoder(
                out_channels=1,
                num_rdb=1,
                add_final_upsample_stage=add_final_upsample_stage
            ) for pid in posture_ids
        })

    def forward(self, concealed_anchor: torch.Tensor, posture_id: int) -> torch.Tensor:
        with torch.no_grad():
            feat1, feat2, feat3, feat4 = self.base_network.extract_features(concealed_anchor)
        
        decoder = self.reconstruction_decoders[str(posture_id)]
        return decoder(feat4, feat3, feat2, feat1)

class UnifiedMultiTaskModel(nn.Module):
    """Unified Multi-Task Learning Network that trains all tasks simultaneously."""
    def __init__(self, num_classes: int = 5, embed_dim: int = 128, backbone_type: str = "resnet18", posture_ids: list[int] = [1, 2, 3, 4, 5], **kwargs):
        super().__init__()
        
        # 1. Shared Encoder: Instantiate the base network. Its weights will be trained by all tasks.
        self.encoder = TripletMultiTaskNetwork(
            num_classes=num_classes, embed_dim=embed_dim, backbone_type=backbone_type, **kwargs
        )
        
        # Ensure the encoder's original reconstruction decoder is not present
        if hasattr(self.encoder, 'reconstruction_decoder'):
            del self.encoder.reconstruction_decoder

        # 2. Mixture-of-Experts Reconstruction Head
        add_final_upsample_stage = (self.encoder.backbone.backbone_type == 'resnet18')
        self.reconstruction_decoders = nn.ModuleDict({
            str(pid): UNetDecoder(
                out_channels=1,
                num_rdb=1,
                add_final_upsample_stage=add_final_upsample_stage
            ) for pid in posture_ids
        })

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, anchor_posture_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        # --- Step 1: Shared Encoder Pass for Discriminative Tasks ---
        discriminative_outputs = self.encoder(anchor, positive, negative)

        # --- Step 2: Shared Encoder Pass for Reconstruction Task ---
        # Extract multi-level features from the anchor image (the input to be reconstructed)
        feat1, feat2, feat3, feat4 = self.encoder.extract_features(anchor)

        # --- Step 3: MoE Reconstruction ---
        # Route the anchor features to the correct expert decoder based on posture
        reconstructed_output = torch.zeros_like(anchor)
        for pid in torch.unique(anchor_posture_ids):
            mask = (anchor_posture_ids == pid)
            decoder = self.reconstruction_decoders[str(pid.item())]
            recon_result = decoder(feat4[mask], feat3[mask], feat2[mask], feat1[mask])
            reconstructed_output[mask] = recon_result

        # --- Step 4: Combine All Outputs ---
        final_outputs = discriminative_outputs
        final_outputs['reconstructed_image'] = reconstructed_output
        
        return final_outputs
