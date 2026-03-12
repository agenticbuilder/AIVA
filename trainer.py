"""
MSPN Trainer
=============
[Proposed Implementation]

Training loop for the Multimodal Sentiment Perception Network.
The paper does not specify training details. This trainer follows
standard practices for multimodal sentiment models trained on CMU-MOSI
or IEMOCAP.

Loss:
  - Cross-entropy for emotion classification
  - MSE for VAD regression (if enabled)
  - Combined as: loss = alpha * cls_loss + (1 - alpha) * vad_loss
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.fusion.mspn import MSPN
from src.utils.logging import get_logger

logger = get_logger(__name__)


class MSPNTrainer:
    """
    Training manager for MSPN.

    Args:
        cfg: Full system config.
        train_loader: DataLoader yielding (frames, texts, labels, vad_targets).
        val_loader:   Validation DataLoader.
    """

    def __init__(
        self,
        cfg: DictConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = torch.device(cfg.system.get("device", "cpu"))
        self.model = MSPN(cfg).to(self.device)

        train_cfg = cfg.training
        self.epochs = train_cfg.num_epochs
        self.val_interval = train_cfg.get("val_interval", 1)
        self.save_interval = train_cfg.get("save_interval", 5)
        self.checkpoint_dir = Path(train_cfg.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.vad_loss_fn = nn.MSELoss()
        self.alpha = 0.7    # Weight for classification vs. regression loss

        self.grad_clip = train_cfg.get("grad_clip", 1.0)
        self.best_val_loss = float("inf")

        # Optional: Weights & Biases
        self.use_wandb = train_cfg.get("use_wandb", False)
        if self.use_wandb:
            import wandb
            wandb.init(project=train_cfg.get("wandb_project", "aiva-mspn"), config=dict(cfg))

    def _step(self, batch: dict) -> dict[str, float]:
        """Run a single training step and return loss components."""
        frames = batch.get("frames")
        texts = batch["texts"]
        labels = batch["labels"].to(self.device)

        if frames is not None:
            frames = frames.to(self.device)

        outputs = self.model(frames=frames, texts=texts)

        cls_loss = self.cls_loss_fn(outputs["logits"], labels)
        total_loss = cls_loss
        loss_dict = {"cls_loss": cls_loss.item()}

        if "vad" in outputs and "vad_targets" in batch:
            vad_targets = batch["vad_targets"].to(self.device)
            vad_loss = self.vad_loss_fn(outputs["vad"], vad_targets)
            total_loss = self.alpha * cls_loss + (1 - self.alpha) * vad_loss
            loss_dict["vad_loss"] = vad_loss.item()

        loss_dict["total_loss"] = total_loss.item()

        self.optimizer.zero_grad()
        total_loss.backward()

        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()
        return loss_dict

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in self.val_loader:
            frames = batch.get("frames")
            texts = batch["texts"]
            labels = batch["labels"].to(self.device)

            if frames is not None:
                frames = frames.to(self.device)

            outputs = self.model(frames=frames, texts=texts)
            loss = self.cls_loss_fn(outputs["logits"], labels)
            total_loss += loss.item()

            preds = outputs["logits"].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        self.model.train()
        return {
            "val_loss": total_loss / max(len(self.val_loader), 1),
            "val_accuracy": correct / max(total, 1),
        }

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }
        path = self.checkpoint_dir / f"mspn_epoch{epoch:03d}.pt"
        torch.save(state, path)
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(state, best_path)
            logger.info("New best checkpoint saved: val_loss=%.4f", val_loss)

    def train(self) -> None:
        logger.info("Starting MSPN training for %d epochs.", self.epochs)
        self.model.train()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.perf_counter()
            running_loss = 0.0

            for batch_idx, batch in enumerate(self.train_loader):
                loss_dict = self._step(batch)
                running_loss += loss_dict["total_loss"]

                if batch_idx % 50 == 0:
                    logger.info(
                        "Epoch %d | Step %d | Loss: %.4f",
                        epoch, batch_idx, loss_dict["total_loss"],
                    )

            epoch_loss = running_loss / max(len(self.train_loader), 1)
            epoch_time = time.perf_counter() - epoch_start
            logger.info("Epoch %d complete. Loss=%.4f | Time=%.1fs", epoch, epoch_loss, epoch_time)

            self.scheduler.step()

            # Validation
            if epoch % self.val_interval == 0:
                val_metrics = self._validate()
                logger.info(
                    "Validation | Loss=%.4f | Acc=%.3f",
                    val_metrics["val_loss"], val_metrics["val_accuracy"],
                )
                is_best = val_metrics["val_loss"] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics["val_loss"]

            # Checkpoint
            if epoch % self.save_interval == 0 or epoch == self.epochs:
                self._save_checkpoint(epoch, val_metrics.get("val_loss", 0.0), is_best=is_best)

            if self.use_wandb:
                import wandb
                wandb.log({"epoch": epoch, "train_loss": epoch_loss, **val_metrics})

        logger.info("Training complete. Best val loss: %.4f", self.best_val_loss)
