"""
MSPN Evaluator
===============
[Proposed Implementation]

Evaluation utilities for the Multimodal Sentiment Perception Network.
Computes standard metrics used in multimodal sentiment analysis literature.

Metrics:
  - Accuracy (emotion classification)
  - Weighted F1 (emotion classification)
  - MAE (valence regression)
  - MAE (arousal regression)
  - Pearson correlation (valence, arousal)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.fusion.mspn import MSPN
from src.utils.logging import get_logger

logger = get_logger(__name__)


class MSPNEvaluator:
    """
    Evaluation runner for MSPN on a held-out dataset split.

    Args:
        cfg:         Full system config.
        checkpoint:  Path to model checkpoint (.pt file).
    """

    def __init__(self, cfg: DictConfig, checkpoint: str | None = None) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.system.get("device", "cpu"))
        self.model = MSPN(cfg).to(self.device)

        if checkpoint:
            self._load_checkpoint(checkpoint)

        self.model.eval()

    def _load_checkpoint(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        state = torch.load(p, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        logger.info("Loaded checkpoint: %s", p)

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> dict[str, float]:
        """
        Run evaluation over a full DataLoader.

        Returns:
            Dict of metric names to float values.
        """
        all_preds = []
        all_labels = []
        all_vad_preds = []
        all_vad_targets = []

        for batch in data_loader:
            frames = batch.get("frames")
            texts = batch["texts"]
            labels = batch["labels"]

            if frames is not None:
                frames = frames.to(self.device)

            outputs = self.model(frames=frames, texts=texts)

            preds = outputs["logits"].argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

            if "vad" in outputs and "vad_targets" in batch:
                all_vad_preds.append(outputs["vad"].cpu().numpy())
                all_vad_targets.append(batch["vad_targets"].numpy())

        metrics: dict[str, float] = {}

        # Classification metrics
        metrics["accuracy"] = float(accuracy_score(all_labels, all_preds))
        metrics["f1_weighted"] = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))
        metrics["f1_macro"] = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))

        # Regression metrics (if available)
        if all_vad_preds:
            vad_preds = np.vstack(all_vad_preds)       # (N, 3)
            vad_targets = np.vstack(all_vad_targets)   # (N, 3)

            metrics["mae_valence"] = float(np.mean(np.abs(vad_preds[:, 0] - vad_targets[:, 0])))
            metrics["mae_arousal"] = float(np.mean(np.abs(vad_preds[:, 1] - vad_targets[:, 1])))
            metrics["mae_dominance"] = float(np.mean(np.abs(vad_preds[:, 2] - vad_targets[:, 2])))

            # Pearson correlations
            for i, name in enumerate(["valence", "arousal", "dominance"]):
                corr = float(np.corrcoef(vad_preds[:, i], vad_targets[:, i])[0, 1])
                metrics[f"pearson_{name}"] = corr

        return metrics

    def report(self, metrics: dict[str, float], output_path: str | None = None) -> None:
        """Print metrics and optionally save to JSON."""
        logger.info("Evaluation Results:")
        for key, val in metrics.items():
            logger.info("  %-25s %.4f", key, val)

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info("Results saved to %s", path)
