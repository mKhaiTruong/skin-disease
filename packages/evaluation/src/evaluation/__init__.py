from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class EvaluationMetrics:
    model_name: str
    batch_size: int

@dataclass(frozen=True)
class EvaluationConfig:
    train_dir:  Path
    output_dir: Path
    data_dir:   Path
    metrics:    EvaluationMetrics