import torch
import torch.nn as nn
from transformers import VideoMAEModel, VideoMAEImageProcessor


class FrozenVideoMAEEncoder(nn.Module):
    """
    Frozen VideoMAE encoder (feature extractor only).

    Input:
        x: (B, T, 3, H, W)   float32 in [0,1]

    Output:
        (B, D) video-level embedding
    """
    def __init__(self, model_name: str = "MCG-NJU/videomae-base"):
        super().__init__()

        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        self.model = VideoMAEModel.from_pretrained(model_name)

        # 🔒 Freeze VideoMAE parameters
        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, 3, H, W)
        Returns: (B, D)
        """
        x = x.float().cpu()  # move to CPU for PIL / numpy

        # If normalized, bring back to [0,1]
        if x.min() < 0 or x.max() > 1:
            x = x - x.min()
            x = x / (x.max() + 1e-6)

        B, T, C, H, W = x.shape

        # Build list-of-lists: videos -> frames
        videos = []
        for b in range(B):
            frames = []
            for t in range(T):
                frame = x[b, t].permute(1, 2, 0).numpy()  # (H,W,3)
                frames.append(frame)
            videos.append(frames)

        # Let VideoMAE handle resize + normalization
        inputs = self.processor(
            videos,
            return_tensors="pt",
            do_rescale=False,  # VERY IMPORTANT
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        # Average token embeddings → video-level embedding
        return outputs.last_hidden_state.mean(dim=1)
