from pathlib import Path
import numpy as np
import onnxruntime as ort
import numpy as np
from pathlib import Path

from vision.strategies import BaseVisionModel
        
class EfficientNetONNXModel(BaseVisionModel):
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.session = None
        self.input_name = None

    def predict(self, imgs):
        imgs_np     = imgs.cpu().numpy() if hasattr(imgs, "cpu") else imgs
        logits      = self.session.run(None, {self.input_name: imgs_np})[0]
        exp_logits  = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs       = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        pred_idx    = int(np.argmax(probs, axis=1)[0])
        confidence  = float(probs[0, pred_idx])
        return pred_idx, confidence

    def save(self, path: Path) -> None:
        raise NotImplementedError("ONNX models are exported via save_onnx(), not saved directly.")

    def load(self, path: Path) -> None:
        self.session = ort.InferenceSession(str(path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def save_onnx(self, path: Path, sample_input) -> None:
        raise NotImplementedError("Already in ONNX format, cannot re-export.")