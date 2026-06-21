from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class TransformationMetrics:
    image_size:  int
    train_ratio: float
    valid_ratio: float
    
@dataclass(frozen=True)
class TransformationConfig:
    root_dir:       Path
    data_dirs:      tuple
    out_train_dir:  Path
    out_valid_dir:  Path
    out_infer_dir:  Path
    metrics:        TransformationMetrics