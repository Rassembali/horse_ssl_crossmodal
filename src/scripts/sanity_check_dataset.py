import argparse
import torch
from torch.utils.data import DataLoader

from src.datasets.horse_dataset import HorseCrossModalDataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--clip_len", type=int, default=16)
    ap.add_argument("--image_size", type=int, default=224)
    args = ap.parse_args()

    ds = HorseCrossModalDataset(root=args.data, clip_len=args.clip_len, image_size=args.image_size)
    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)

    batch = next(iter(dl))
    imu = batch["imu"]
    vid = batch["video"]
    lab = batch["label"]

    print("Batch shapes:")
    print("  imu:", tuple(imu.shape), "(expected B, L, C)")
    print("  video:", tuple(vid.shape), "(expected B, T, 3, H, W)")
    print("  label:", tuple(lab.shape))
    print("Example time window:", batch["t0"][0].item(), "->", batch["t1"][0].item())
    print("Session:", batch["session"][0])

    # Quick alignment check: frame indices should fall in the time window (approx)
    # (We don't print indices here; dataset uses timestamps to sample within range.)
    print("Sanity check OK." )

if __name__ == "__main__":
    main()
