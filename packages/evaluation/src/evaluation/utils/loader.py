import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import datasets

from evaluation.utils.augment import get_eval_transforms

def build_eval_loaders(eval_dir, batch_size: int, num_workers: int = 2):
    eval_ds = datasets.ImageFolder(eval_dir, transform=get_eval_transforms())
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return eval_loader, eval_ds.classes