import torch
import torch.nn as nn

class PatchTSTIMUEncoder(nn.Module):
    """
    PatchTST-like encoder for IMU.

    Input:  (B, L, C) e.g. (B, 250, 30)

    Steps:
      1) Patchify along time: patches of length patch_len with stride patch_stride
      2) Flatten each patch (patch_len * C) and project to d_model
      3) Add CLS token + learned positional embeddings
      4) Transformer encoder over tokens
      5) Return CLS embedding -> out_dim

    Output: (B, out_dim)
    """
    def __init__(   
        self,
        in_ch: int = 30,
        out_dim: int = 256,
        d_model: int = 192,
        nhead: int = 6,
        num_layers: int = 4,
        patch_len: int = 10,
        patch_stride: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_dim = out_dim
        self.patch_len = patch_len
        self.patch_stride = patch_stride

        # Patch embedding: (patch_len * in_ch) -> d_model
        self.patch_proj = nn.Linear(patch_len * in_ch, d_model)

        # CLS token + positional embedding
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = None  # lazy init once we know #patches

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head = nn.Linear(d_model, out_dim)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, C)
        returns: (B, N, patch_len*C)
        """
        B, L, C = x.shape
        if C != self.in_ch:
            raise ValueError(f"Expected C={self.in_ch}, got C={C}")

        # unfold along time dimension
        x_t = x.transpose(1, 2)  # (B, C, L)
        patches = x_t.unfold(dimension=2, size=self.patch_len, step=self.patch_stride)  # (B,C,N,patch_len)
        patches = patches.transpose(1, 2)  # (B,N,C,patch_len)
        patches = patches.reshape(B, patches.size(1), C * self.patch_len)  # (B,N,patch_len*C)
        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self._patchify(x)            # (B, N, patch_len*C)
        tok = self.patch_proj(patches)         # (B, N, d_model)

        B, N, D = tok.shape

        # create pos embedding once we know N
        if (self.pos is None) or (self.pos.size(1) != (N + 1)):
            self.pos = nn.Parameter(torch.zeros(1, N + 1, D, device=tok.device))

        cls = self.cls.expand(B, -1, -1)       # (B,1,D)
        tok = torch.cat([cls, tok], dim=1)     # (B,1+N,D)
        tok = tok + self.pos

        z = self.enc(tok)                      # (B,1+N,D)
        emb = z[:, 0]                          # CLS
        return self.head(emb)                  # (B,out_dim)
