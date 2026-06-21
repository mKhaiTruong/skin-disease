import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def compute_full_metrics(y_true, y_pred, class_names: list[str]) -> dict:
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    return {
        "report": report,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }
    
def collect_predictions(model, loader) -> tuple[np.ndarray, np.ndarray]:
    model.model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(model.device)
            logits = model.model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    return np.array(y_true), np.array(y_pred)