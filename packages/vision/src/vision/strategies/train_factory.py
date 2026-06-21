from pathlib import Path

from vision.strategies import BaseVisionModel
from vision.strategies.effnet import EfficientNetModel

MODEL_REGISTRY = {
    "efficientnet": EfficientNetModel,
    # "resnet": ResNetModel,
}

class ModelFactory:
    @staticmethod
    def create(model_name: str, config, num_classes: int) -> BaseVisionModel:
        cls = MODEL_REGISTRY.get(model_name)
        if not cls:
            raise ValueError(f"No model registered for: '{model_name}'. Add it to MODEL_REGISTRY.")
        
        checkpoint_dir = Path(config.output_dir) / model_name
        return cls(num_classes, checkpoint_dir)
    
class EvalModelFactory:
    @staticmethod
    def create(model_name: str, num_classes: int) -> BaseVisionModel:
        cls = MODEL_REGISTRY.get(model_name)
        if not cls:
            raise ValueError(f"No model registered for: '{model_name}'.")
        return cls(num_classes)