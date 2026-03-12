"""
Visual Encoder Module
=====================
[Proposed Implementation]

The paper describes a visual encoder that extracts features from video frames.
This implementation uses a Vision Transformer (ViT-B/16) backbone from the
timm library, pretrained on ImageNet and projectable onto a shared embedding
space with the text encoder.

For downstream emotion tasks, the backbone should be fine-tuned on AffectNet
or FER2013 before MSPN training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.transforms as T
from omegaconf import DictConfig

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class VisualEncoder(nn.Module):
    """
    ViT-based visual feature extractor.

    Accepts a batch of video frame tensors and returns a projected
    embedding of shape (B, embed_dim).

    Args:
        cfg: visual_encoder section of the config.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        if not HAS_TIMM:
            raise ImportError("timm is required for VisualEncoder. Install with: pip install timm")

        self.embed_dim = cfg.embed_dim
        self.frames_per_sample = cfg.get("frames_per_sample", 8)

        # Load backbone
        self.backbone = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            num_classes=0,  # Remove classification head; output is feature vector
        )
        backbone_dim = self.backbone.num_features

        if cfg.get("freeze_backbone", False):
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection head: maps backbone features -> shared embed_dim
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

        # Temporal aggregation: mean-pool across frames
        # [Proposed]: Simple temporal mean pooling. Could be replaced with
        # a temporal transformer for richer temporal modeling.
        self.temporal_pool = "mean"

        # Standard ImageNet normalization
        self.normalize = T.Normalize(
            mean=cfg.get("normalize_mean", [0.485, 0.456, 0.406]),
            std=cfg.get("normalize_std", [0.229, 0.224, 0.225]),
        )

    def preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Normalize a batch of frame tensors.

        Args:
            frames: Float tensor of shape (B, T, C, H, W) in [0, 1].

        Returns:
            Normalized tensor of the same shape.
        """
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W)
        frames = torch.stack([self.normalize(f) for f in frames])
        return frames.view(B, T, C, H, W)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features from a batch of video frames.

        Args:
            frames: Float tensor of shape (B, T, C, H, W) in [0, 1].
                    T is the number of frames per sample.

        Returns:
            Projected feature tensor of shape (B, embed_dim).
        """
        B, T, C, H, W = frames.shape

        frames = self.preprocess(frames)
        frames_flat = frames.view(B * T, C, H, W)

        # Extract per-frame features
        features = self.backbone(frames_flat)           # (B*T, backbone_dim)
        features = features.view(B, T, -1)              # (B, T, backbone_dim)

        # Temporal aggregation
        if self.temporal_pool == "mean":
            features = features.mean(dim=1)             # (B, backbone_dim)
        else:
            features = features[:, 0, :]                # Use first frame only

        # Project to shared embedding space
        embeddings = self.projection(features)          # (B, embed_dim)

        return embeddings
