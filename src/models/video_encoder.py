import torch
import torch.nn as nn
from transformers import VideoMAEModel, VideoMAEImageProcessor


class FrozenVideoMAEEncoder(nn.Module):
    """
    Frozen VideoMAE encoder (feature extractor only).

    Input:  (B, T, 3, H, W)   float32 in [0,1]
    Output: (B, D)
    """
    def __init__(self, model_name="MCG-NJU/videomae-base"):
        super().__init__()

        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        self.model = VideoMAEModel.from_pretrained(model_name)

        # 🔒 Freeze everything
        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, 3, H, W)
        """
        B, T, C, H, W = x.shape

        # VideoMAE expects list of videos, each (T,H,W,C)
        x = x.permute(0, 1, 3, 4, 2)  # (B,T,H,W,C)
        videos = list(x)

        inputs = self.processor(videos, return_tensors="pt")
        inputs = {k: v.to(x.device) for k, v in inputs.items()}

        out = self.model(**inputs)

        # Average token embeddings → video-level embedding
        return out.last_hidden_state.mean(dim=1)
