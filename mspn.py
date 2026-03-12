"""
Multimodal Sentiment Perception Network (MSPN)
===============================================
Core fusion module from arXiv:2509.03212.

Architecture (from the paper + proposed implementation):
  1. Visual Encoder: extracts frame-level features
  2. Text Encoder: extracts utterance-level features
  3. Cross-Attention Fusion: bidirectional cross-modal attention
  4. Cross-Modal Fusion Transformer (CMFT): produces unified sentiment embedding
  5. Sentiment Head: outputs emotion label + VAD regression scores

[Proposed Implementation]: The exact CMFT depth, head count, and training
procedure are not specified in the paper. The implementation below follows
common practice in multimodal fusion literature.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.encoders.text_encoder import TextEncoder
from src.encoders.visual_encoder import VisualEncoder
from src.fusion.cross_attention import CrossModalAttention

EMOTION_LABELS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
]


@dataclass
class SentimentCue:
    """Structured output of MSPN inference."""
    emotion_label: str
    emotion_logits: torch.Tensor       # (num_classes,)
    emotion_probs: torch.Tensor        # (num_classes,)
    valence: float                     # [-1, 1]
    arousal: float                     # [-1, 1]
    dominance: float                   # [-1, 1]
    confidence: float                  # max probability of emotion prediction


class CrossModalFusionTransformer(nn.Module):
    """
    Small transformer that fuses concatenated cross-attended embeddings
    into a single holistic sentiment representation.

    [Proposed Implementation]
    """

    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,       # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, visual: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual: (B, embed_dim) visual features after cross-attention.
            text:   (B, embed_dim) text features after cross-attention.

        Returns:
            (B, embed_dim) unified sentiment embedding.
        """
        B = visual.size(0)

        # Concatenate modality tokens + learnable CLS token
        v = visual.unsqueeze(1)                             # (B, 1, D)
        t = text.unsqueeze(1)                               # (B, 1, D)
        cls = self.cls_token.expand(B, -1, -1)              # (B, 1, D)

        sequence = torch.cat([cls, v, t], dim=1)            # (B, 3, D)
        out = self.transformer(sequence)                    # (B, 3, D)

        return out[:, 0, :]                                 # CLS token = fused representation


class SentimentHead(nn.Module):
    """
    Joint classification + regression head.
    Outputs emotion class logits and VAD regression scores.

    [Proposed Implementation]
    """

    def __init__(self, embed_dim: int, num_classes: int, predict_vad: bool = True) -> None:
        super().__init__()
        self.predict_vad = predict_vad

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes),
        )

        if predict_vad:
            self.vad_regressor = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim // 2, 3),   # valence, arousal, dominance
                nn.Tanh(),                      # maps to [-1, 1]
            )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, embed_dim) fused sentiment embedding.

        Returns:
            dict with 'logits' (B, num_classes) and optionally 'vad' (B, 3).
        """
        out: dict[str, torch.Tensor] = {"logits": self.classifier(x)}
        if self.predict_vad:
            out["vad"] = self.vad_regressor(x)
        return out


class MSPN(nn.Module):
    """
    Multimodal Sentiment Perception Network.

    Full end-to-end module from frame + text inputs to structured sentiment cue.

    Args:
        cfg: Full system config (DictConfig). Uses cfg.visual_encoder,
             cfg.text_encoder, cfg.fusion, and cfg.sentiment_head sections.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.visual_encoder = VisualEncoder(cfg.visual_encoder)
        self.text_encoder = TextEncoder(cfg.text_encoder)

        embed_dim = cfg.fusion.embed_dim
        assert embed_dim == cfg.visual_encoder.embed_dim == cfg.text_encoder.embed_dim, (
            "embed_dim must match across visual_encoder, text_encoder, and fusion config."
        )

        self.cross_attention = CrossModalAttention(
            embed_dim=embed_dim,
            num_heads=cfg.fusion.num_heads,
            dropout=cfg.fusion.dropout,
        )

        self.cmft = CrossModalFusionTransformer(
            embed_dim=embed_dim,
            num_heads=cfg.fusion.num_heads,
            num_layers=cfg.fusion.num_layers,
            ffn_dim=cfg.fusion.ffn_dim,
            dropout=cfg.fusion.dropout,
        )

        self.sentiment_head = SentimentHead(
            embed_dim=embed_dim,
            num_classes=cfg.sentiment_head.num_emotion_classes,
            predict_vad=cfg.sentiment_head.predict_vad,
        )

        self.emotion_labels = EMOTION_LABELS[: cfg.sentiment_head.num_emotion_classes]

    def forward(
        self,
        frames: torch.Tensor | None,
        texts: list[str] | dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the full MSPN.

        Args:
            frames: Video frame tensor (B, T, C, H, W) or None (text-only mode).
            texts:  List of text strings or pre-tokenized dict.

        Returns:
            Dict containing 'logits', and optionally 'vad'.
        """
        # Text encoding
        text_emb = self.text_encoder(texts)                     # (B, D)

        # Visual encoding (or zero vector if no video provided)
        if frames is not None:
            visual_emb = self.visual_encoder(frames)            # (B, D)
        else:
            visual_emb = torch.zeros_like(text_emb)

        # Cross-modal attention
        visual_attended, text_attended = self.cross_attention(visual_emb, text_emb)

        # Cross-modal fusion transformer
        fused = self.cmft(visual_attended, text_attended)       # (B, D)

        # Sentiment prediction
        return self.sentiment_head(fused)

    @torch.no_grad()
    def predict(
        self,
        frames: torch.Tensor | None,
        texts: list[str],
    ) -> list[SentimentCue]:
        """
        Run inference and return structured SentimentCue objects.

        Args:
            frames: (B, T, C, H, W) float tensor or None.
            texts:  List of B text strings.

        Returns:
            List of SentimentCue, one per sample.
        """
        self.eval()
        outputs = self.forward(frames, texts)
        logits = outputs["logits"]                              # (B, C)
        probs = torch.softmax(logits, dim=-1)
        pred_ids = probs.argmax(dim=-1)

        vad = outputs.get("vad")                               # (B, 3) or None

        cues = []
        for i in range(logits.size(0)):
            pred_id = pred_ids[i].item()
            cue = SentimentCue(
                emotion_label=self.emotion_labels[pred_id],
                emotion_logits=logits[i],
                emotion_probs=probs[i],
                valence=vad[i, 0].item() if vad is not None else 0.0,
                arousal=vad[i, 1].item() if vad is not None else 0.0,
                dominance=vad[i, 2].item() if vad is not None else 0.0,
                confidence=probs[i, pred_id].item(),
            )
            cues.append(cue)

        return cues
