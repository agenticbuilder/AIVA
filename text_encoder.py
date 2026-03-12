"""
Text Encoder Module
===================
[Proposed Implementation]

The paper describes a textual encoder that extracts sentiment-relevant
features from the user's utterance. This implementation uses BERT-base-uncased
from HuggingFace Transformers, with a learned projection head to align text
features with visual features in a shared embedding space.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    """
    BERT-based text feature extractor.

    Encodes a batch of text strings and returns a projected embedding
    of shape (B, embed_dim).

    Args:
        cfg: text_encoder section of the config.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.embed_dim = cfg.embed_dim
        self.max_length = cfg.get("max_length", 128)
        model_name = cfg.model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        backbone_dim = self.backbone.config.hidden_size

        if cfg.get("freeze_backbone", False):
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection head: [CLS] token -> shared embed_dim
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

    def tokenize(self, texts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
        """Tokenize a list of strings and move tensors to device."""
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in encoding.items()}

    def forward(self, texts: list[str] | dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode a batch of text inputs.

        Args:
            texts: Either a list of raw strings or a pre-tokenized
                   dict of input_ids / attention_mask tensors.

        Returns:
            Projected text embedding of shape (B, embed_dim).
        """
        if isinstance(texts, list):
            device = next(self.parameters()).device
            encoding = self.tokenize(texts, device)
        else:
            encoding = texts

        outputs = self.backbone(**encoding)

        # Use [CLS] token representation as the sentence embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]   # (B, backbone_dim)

        embeddings = self.projection(cls_embedding)           # (B, embed_dim)

        return embeddings
