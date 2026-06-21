from vision.strategies import BaseVisionModel
from vision.strategies.effnet_onnx import EfficientNetONNXModel

ONNX_MODEL_REGISTRY = {
    "efficientnet": EfficientNetONNXModel,
}

class ONNXModelFactory:
    @staticmethod
    def create(model_name: str, num_classes: int) -> BaseVisionModel:
        cls = ONNX_MODEL_REGISTRY.get(model_name)
        if not cls:
            raise ValueError(f"No ONNX model registered for: '{model_name}'.")
        return cls(num_classes)