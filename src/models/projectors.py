import torch.nn as nn

class ProjectionHead(nn.Module):
    """
    2-layer MLP projection head with normalization.
    Typical in contrastive learning.
    """
    def __init__(self, in_dim: int, proj_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        return self.net(x)
