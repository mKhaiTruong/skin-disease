import torch
import torch.nn as nn

def build_criterion(class_counts, num_classes: int, device, config):
    cw = torch.FloatTensor(1.0 / class_counts).to(device)
    cw = cw / cw.sum() * num_classes
    return nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)