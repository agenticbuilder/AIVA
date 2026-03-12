"""
Cross-Modal Attention Module
=============================
Implements bidirectional cross-attention between visual and text embeddings,
as described in arXiv:2509.03212.

Each modality attends to the other:
  - Visual features attend over text token embeddings
  - Text features attend over visual patch/frame embeddings

[Proposed Implementation]: The paper specifies cross-attention fusion between
modalities. This implements a standard multi-head cross-attention layer following
the architecture used in multimodal fusion literature (e.g., ViLBERT, FLAVA).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    """
    Bidirectional cross-modal attention.

    Given query features from one modality and key-value features from another,
    computes attention-weighted representations for both modalities.

    Args:
        embed_dim: Dimensionality of query, key, and value vectors.
        num_heads: Number of attention heads.
        dropout: Dropout probability on attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Visual attends to text
        self.v_to_t = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Text attends to visual
        self.t_to_v = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_v = nn.LayerNorm(embed_dim)
        self.norm_t = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        visual: torch.Tensor,
        text: torch.Tensor,
        visual_mask: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute bidirectional cross-modal attention.

        Args:
            visual: Visual embeddings of shape (B, Nv, embed_dim).
                    If 2D (B, embed_dim), unsqueezes a token dimension.
            text:   Text embeddings of shape (B, Nt, embed_dim).
                    If 2D (B, embed_dim), unsqueezes a token dimension.
            visual_mask: Optional key padding mask for visual tokens.
            text_mask:   Optional key padding mask for text tokens.

        Returns:
            Tuple of (visual_out, text_out), each of shape matching input.
        """
        # Handle 2D inputs (single-vector per sample)
        squeeze_v = visual.dim() == 2
        squeeze_t = text.dim() == 2

        if squeeze_v:
            visual = visual.unsqueeze(1)    # (B, 1, embed_dim)
        if squeeze_t:
            text = text.unsqueeze(1)        # (B, 1, embed_dim)

        # Visual attends to text
        v_attended, _ = self.v_to_t(
            query=visual,
            key=text,
            value=text,
            key_padding_mask=text_mask,
        )
        visual_out = self.norm_v(visual + self.dropout(v_attended))

        # Text attends to visual
        t_attended, _ = self.t_to_v(
            query=text,
            key=visual,
            value=visual,
            key_padding_mask=visual_mask,
        )
        text_out = self.norm_t(text + self.dropout(t_attended))

        # Restore original dimensions
        if squeeze_v:
            visual_out = visual_out.squeeze(1)
        if squeeze_t:
            text_out = text_out.squeeze(1)

        return visual_out, text_out
