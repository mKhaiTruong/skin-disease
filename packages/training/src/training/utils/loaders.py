import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets

from training.utils.augment import get_transforms

def build_dataloaders(train_dir, valid_dir, batch_size: int, num_workers: int = 2):
    train_ds = datasets.ImageFolder(train_dir, transform=get_transforms(augment=True))
    val_ds   = datasets.ImageFolder(valid_dir, transform=get_transforms(augment=False))

    targets      = np.array(train_ds.targets)
    class_counts = np.bincount(targets)
    weights      = 1.0 / class_counts[targets]
    sampler      = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,   num_workers=num_workers)

    return train_loader, val_loader, len(train_ds.classes), class_counts