from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class TrainMetrics:
    model_name:         str
    image_size:         int
    epochs:             int
    batch_size:         int
    lr:                 float
    unfreeze_epoch:     int
    weight_decay:       float
    top_k_checkpoints:  int
    early_stopping:     int

@dataclass(frozen=True)
class TrainConfig:
    train_dir:  Path
    valid_dir:  Path
    output_dir: Path
    metrics:    TrainMetrics
    resume_dir: Path | None = None