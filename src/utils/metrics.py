import torch

@torch.no_grad()
def accuracy(logits, targets):
    preds = torch.argmax(logits, dim=-1)
    return (preds == targets).float().mean().item()
