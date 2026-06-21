from pathlib import Path
import numpy as np
import torch
from torch.amp import GradScaler

from core.logger import logger
from vision.strategies import BaseVisionModel
from training.utils.optimizer import build_optimizer
from training.utils.loss_fn import build_criterion
from training.utils.engine import run_epoch, run_validate, CheckpointManager

class Trainer:
    def __init__(self, model: BaseVisionModel, class_counts: np.ndarray, config):
        self.config = config
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        
        self.model = model
        self.scaler = GradScaler()
        self.optimizer, self.scheduler = build_optimizer(model.model, config.metrics.lr, freeze=True, config=config)
        self.criterion = build_criterion(class_counts, model.num_classes, model.device, config=config)
        self.checkpoint_mgr = CheckpointManager(model.checkpoint_dir, top_k=config.metrics.top_k_checkpoints)

    def _unfreeze_backbone(self):
        for p in self.model.model.features.parameters():
            p.requires_grad = True
        self.optimizer, self.scheduler = build_optimizer(self.model.model, self.config.metrics.lr, freeze=False, config=self.config)

    def fit(self, train_loader, val_loader) -> None:
        start_epoch = 1

        if self.config.resume_dir and self.config.resume_dir.exists():
            last_checkpoint = self.config.resume_dir / "last_checkpoint.pth"
            if last_checkpoint.exists():
                loaded_epoch, self.best_val_acc = self.load_checkpoint(last_checkpoint)
                start_epoch = loaded_epoch + 1
                logger.info(f"Resumed from epoch {start_epoch-1}, best_val_acc={self.best_val_acc:.4f}")

        for epoch in range(start_epoch, self.config.metrics.epochs + 1):
            if epoch == self.config.metrics.unfreeze_epoch:
                self._unfreeze_backbone()
                print("!! Backbone unfrozen !!")

            train_m = run_epoch(self.model.model, train_loader, self.criterion, self.optimizer, self.model.device, train=True, scaler=self.scaler)
            self.scheduler.step()
            val_m = run_validate(self.model.model, val_loader, self.criterion, self.model.device)

            self.checkpoint_mgr.save_if_best(self, val_m["acc"], epoch)
            self.save_checkpoint(self.model.checkpoint_dir / "last_checkpoint.pth", epoch)

            print(f"Epoch {epoch:02d}/{self.config.metrics.epochs}  loss={train_m['loss']:.4f}  train_acc={train_m['acc']:.4f}  val_acc={val_m['acc']:.4f}")

            # -------- Early Stopping --------
            if val_m["acc"] > self.best_val_acc:
                self.best_val_acc = val_m["acc"]
                self.epochs_without_improvement = 0
                print(f"  ^^^ New best! val_acc={val_m['acc']:.4f} ^^^")
            else:
                self.epochs_without_improvement += 1
            
            if self.epochs_without_improvement >= self.config.metrics.early_stopping:
                logger.info(f"Early stopping triggered at epoch {epoch} (no improvement for {self.config.metrics.early_stopping} epochs)")
                break

    def save_checkpoint(self, path: Path, epoch: int) -> None:
        torch.save({
            "epoch": epoch,
            "model_state": self.model.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
        }, path)

    def load_checkpoint(self, path: Path) -> tuple[int, float]:
        checkpoint = torch.load(path, map_location=self.model.device)
        self.model.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        return checkpoint["epoch"], checkpoint["best_val_acc"]