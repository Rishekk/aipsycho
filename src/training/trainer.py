import os
import torch
import numpy as np
from typing import Dict, Optional
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

from .metrics import MetricTracker
from .validator import Validator
from src.utils.logger import get_logger
from src.utils.model_utils import save_checkpoint

logger = get_logger("trainer")


class Trainer:
    """
    Generic training loop for all models (body, face, speech, fusion).
    Handles:
        - Training + validation epochs
        - LR scheduling (cosine or plateau)
        - Best-model checkpointing
        - Early stopping
        - Per-epoch metric logging
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        checkpoint_dir: str = "models/",
        model_name: str = "model",
        epochs: int = 50,
        scheduler_type: str = "cosine",   # "cosine" | "plateau" | None
        early_stop_patience: int = 10,
        save_every: int = 5,
    ):
        self.model          = model.to(device)
        self.train_loader   = train_loader
        self.criterion      = criterion
        self.optimizer      = optimizer
        self.device         = device
        self.checkpoint_dir = checkpoint_dir
        self.model_name     = model_name
        self.epochs         = epochs
        self.early_stop_patience = early_stop_patience
        self.save_every     = save_every

        # Validator
        self.validator = Validator(model, val_loader, criterion, device)

        # Scheduler
        self.scheduler = self._build_scheduler(scheduler_type)

        # State
        self.best_val_f1   = 0.0
        self.no_improve    = 0
        self.history: Dict = {"train_loss": [], "val_loss": [],
                               "train_acc": [],  "val_acc": [],
                               "val_f1": []}

        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(
            f"Trainer initialized — model: {model_name}, epochs: {epochs}, "
            f"device: {device}, scheduler: {scheduler_type}"
        )

    def train(self) -> Dict:
        """
        Run full training loop.

        Returns:
            Training history dict.
        """
        for epoch in range(1, self.epochs + 1):
            train_metrics = self._train_epoch(epoch)
            val_metrics   = self.validator.evaluate()

            # Update history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_loss"].append(val_metrics.get("loss", 0.0))
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["val_f1"].append(val_metrics["f1_weighted"])

            # Scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["f1_weighted"])
                else:
                    self.scheduler.step()

            # Checkpoint
            if epoch % self.save_every == 0:
                self._save(epoch, tag=f"epoch{epoch}")

            # Best model
            if val_metrics["f1_weighted"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1_weighted"]
                self._save(epoch, tag="best")
                self.no_improve = 0
                logger.info(f"  → New best val F1: {self.best_val_f1:.4f}")
            else:
                self.no_improve += 1

            self._log_epoch(epoch, train_metrics, val_metrics)

            # Early stopping
            if self.no_improve >= self.early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch} (no improvement for {self.no_improve} epochs)")
                break

        logger.info(f"Training complete. Best val F1: {self.best_val_f1:.4f}")
        return self.history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        tracker = MetricTracker()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:03d}/{self.epochs}",
                    unit="batch", leave=False, dynamic_ncols=True)

        for batch in pbar:
            inputs, labels = self._unpack_batch(batch)

            self.optimizer.zero_grad()
            logits = self.model(*inputs)
            loss   = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            with torch.no_grad():
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                tracker.update(labels.cpu().numpy(), preds, probs, loss.item())

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return tracker.compute()

    def _unpack_batch(self, batch):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, labels = batch
            if isinstance(inputs, (list, tuple)):
                inputs = tuple(x.to(self.device) for x in inputs)
            else:
                inputs = (inputs.to(self.device),)
            labels = labels.to(self.device)
        else:
            raise ValueError(f"Unexpected batch format: {type(batch)}")
        return inputs, labels

    def _save(self, epoch: int, tag: str):
        path = os.path.join(self.checkpoint_dir, f"{self.model_name}_{tag}.pth")
        save_checkpoint(self.model, self.optimizer, epoch, path,
                        extra={"val_f1": self.best_val_f1})
        logger.info(f"  Saved checkpoint: {path}")

    def _build_scheduler(self, scheduler_type: Optional[str]):
        if scheduler_type == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)
        elif scheduler_type == "plateau":
            return ReduceLROnPlateau(self.optimizer, mode="max", patience=5,
                                     factor=0.5, verbose=True)
        return None

    def _log_epoch(self, epoch: int, train: Dict, val: Dict):
        lr = self.optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch [{epoch:03d}/{self.epochs}] "
            f"train_loss: {train['loss']:.4f}  train_acc: {train['accuracy']:.4f}  "
            f"val_loss: {val.get('loss', 0.0):.4f}  val_acc: {val['accuracy']:.4f}  "
            f"val_f1: {val['f1_weighted']:.4f}  lr: {lr:.2e}"
        )
