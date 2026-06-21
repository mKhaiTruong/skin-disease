import os
import json
import torch
from tqdm import tqdm
import heapq
import boto3

from torch.amp import autocast

from core.logger import logger
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

def run_epoch(model, loader, criterion, optimizer, device, train: bool, scaler=None) -> dict:
    model.train() if train else model.eval()
    total_loss, total_correct, total_n = 0, 0, 0

    with torch.set_grad_enabled(train):
        for imgs, labels in tqdm(loader, desc="Train" if train else "Valid", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            with autocast(device_type="cuda", dtype=torch.float16):
                out  = model(imgs)
                loss = criterion(out, labels)

            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss    += loss.item() * imgs.size(0)
            total_correct += (out.argmax(1) == labels).sum().item()
            total_n       += imgs.size(0)

    return {"loss": total_loss / total_n, "acc": total_correct / total_n}

@torch.no_grad()
def run_validate(model, loader, criterion, device) -> dict:
    return run_epoch(model, loader, criterion, optimizer=None, device=device, train=False)

def save_model_metadata(output_dir: Path, class_names: list[str], image_size: int):
    metadata = {
        "class_names": class_names,
        "num_classes": len(class_names),
        "image_size": image_size,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

def push_to_s3(local_dir: Path, prefix: str = "train"):
    s3 = boto3.client(
        "s3",
        aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name           = os.getenv("AWS_REGION"),
    )
    bucket = os.getenv("S3_BUCKET")

    for filepath in local_dir.rglob("*"):
        if filepath.is_file():
            key = f"{prefix}/{filepath.relative_to(local_dir)}"
            s3.upload_file(str(filepath), bucket, key)
            logger.info(f"Uploaded: {key}")

class CheckpointManager:
    def __init__(self, output_dir: Path, top_k: int = 3):
        self.output_dir = output_dir
        self.top_k = top_k
        self.heap = []
        os.makedirs(self.output_dir, exist_ok=True)

    def save_if_best(self, trainer, val_acc: float, epoch: int):
        filepath = self.output_dir / f"epoch{epoch:03d}_acc{val_acc:.4f}.pth"

        if len(self.heap) < self.top_k:
            trainer.save_checkpoint(filepath, epoch)
            heapq.heappush(self.heap, (val_acc, epoch, filepath))
            logger.info(f"  ✓ Saved checkpoint ({len(self.heap)}/{self.top_k}): {filepath.name}")
            return

        if val_acc > self.heap[0][0]:
            trainer.save_checkpoint(filepath, epoch)
            heapq.heappush(self.heap, (val_acc, epoch, filepath))

            _, _, worst_path = heapq.heappop(self.heap)
            if worst_path.exists():
                worst_path.unlink()
            logger.info(f"  ✓ New checkpoint kept, removed: {worst_path.name}")

    def get_best_checkpoint(self) -> Path:
        _, _, best_path = max(self.heap)
        return best_path