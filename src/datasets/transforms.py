import numpy as np
import torch
import cv2

# Standard ImageNet normalization (works as a reasonable default for pretrained vision backbones)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_frame_rgb(frame_rgb: np.ndarray, image_size: int) -> torch.Tensor:
    """
    frame_rgb: (H,W,3) uint8, RGB
    returns: (3,image_size,image_size) float32 normalized
    """
    frame_resized = cv2.resize(frame_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    x = frame_resized.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x).permute(2, 0, 1).contiguous()  # (3,H,W)
    return x
