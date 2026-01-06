import torch
import torch.nn.functional as F

def siglip_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Sigmoid contrastive loss (SigLIP style).

    z_a, z_b: (B, D)
    - positives: diagonal pairs
    - negatives: off-diagonal

    loss = mean( softplus( -y * logits ) )
      y = +1 for positives, -1 for negatives
    """
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)

    logits = (z_a @ z_b.t()) / temperature        # (B,B)
    B = logits.size(0)

    y = -torch.ones((B, B), device=logits.device)
    y.fill_(-1.0)
    y.view(-1)[::B + 1] = 1.0                     # diagonal +1

    loss = F.softplus(-y * logits).mean()
    return loss
