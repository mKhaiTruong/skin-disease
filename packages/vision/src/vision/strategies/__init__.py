from abc import ABC, abstractmethod
from pathlib import Path

class BaseVisionModel(ABC):
    @abstractmethod
    def predict(self, imgs): ...
    
    @abstractmethod
    def save(self, path: Path) -> None: ...
    
    @abstractmethod
    def load(self, path: Path) -> None: ...
    
    @abstractmethod
    def save_onnx(self, path: Path, sample_input) -> None: ...