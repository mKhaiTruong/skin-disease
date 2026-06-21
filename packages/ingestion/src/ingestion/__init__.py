from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class IngestionConfig:
    root_dir: Path
    src_type: str
    source:   str
    name:     str