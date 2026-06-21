import torch
from pathlib import Path

from vision.strategies import BaseVisionModel
from vision.utils.load_model import _build_efficientnet
from vision.utils.helpers import _export_onnx

class EfficientNetModel(BaseVisionModel):
    def __init__(self, num_classes: int, checkpoint_dir: Path = None):
        self.num_classes = num_classes
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = _build_efficientnet(num_classes, freeze_backbone=True).to(self.device)

    def predict(self, imgs):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(imgs.to(self.device))
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
        return pred_idx.item(), confidence.item()

    def save(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        state = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
        self.model.load_state_dict(state)

    def save_onnx(self, path: Path, sample_input) -> None:
        _export_onnx(self.model, sample_input, path)