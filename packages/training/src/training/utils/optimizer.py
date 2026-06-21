from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def build_optimizer(model, lr: float, freeze: bool, config):
    params = filter(lambda p: p.requires_grad, model.parameters()) if freeze else model.parameters()
    actual_lr = lr if freeze else lr * 0.1
    optimizer = AdamW(params, lr=actual_lr, weight_decay=config.metrics.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    return optimizer, scheduler