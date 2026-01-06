import argparse
import os
import yaml
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from src.utils.seed import set_seed
from src.utils.logging import make_run_dir, save_json
from src.datasets.horse_dataset import HorseCrossModalDataset

from src.models.imu_encoder import PatchTSTIMUEncoder
from src.models.videomae_encoder import FrozenVideoMAEEncoder
from src.models.projectors import ProjectionHead
from src.losses.contrastive import siglip_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(int(cfg.get("seed", 123)))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================
    # Dataset & DataLoader
    # =========================
    ds = HorseCrossModalDataset(
        root=cfg["data_root"],
        imu_hz=int(cfg.get("imu_hz", 50)),
        window_s=float(cfg.get("window_s", 5.0)),
        clip_len=int(cfg.get("clip_len", 16)),
        image_size=int(cfg.get("image_size", 224)),
        use_mp4=True,
    )

    dl = DataLoader(
        ds,
        batch_size=int(cfg.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 2)),
        pin_memory=True,
    )

    embed_dim = int(cfg.get("embed_dim", 256))

    # =========================
    # IMU Encoder (PatchTST-like)
    # =========================
    imu_enc = PatchTSTIMUEncoder(
        in_ch=30,
        out_dim=embed_dim,
        d_model=int(cfg.get("imu_d_model", 192)),
        nhead=int(cfg.get("imu_nhead", 6)),
        num_layers=int(cfg.get("imu_layers", 4)),
        patch_len=int(cfg.get("patch_len", 10)),
        patch_stride=int(cfg.get("patch_stride", 10)),
        dropout=float(cfg.get("imu_dropout", 0.1)),
    ).to(device)

    # =========================
    # Video Encoder (Frozen VideoMAE)
    # =========================
    vid_enc = FrozenVideoMAEEncoder().to(device)

    # VideoMAE-base hidden size is 768 (this is what your encoder returns)
    video_backbone_dim = 768

    # =========================
    # Projection Heads
    # =========================
    proj_imu = ProjectionHead(
        in_dim=embed_dim,              # IMU encoder output: 256
        proj_dim=embed_dim,            # Shared embedding: 256
        hidden_dim=int(cfg.get("proj_hidden", 512)),
    ).to(device)

    proj_vid = ProjectionHead(
        in_dim=video_backbone_dim,     # ✅ VideoMAE output: 768
        proj_dim=embed_dim,            # Shared embedding: 256
        hidden_dim=int(cfg.get("proj_hidden", 512)),
    ).to(device)

    # =========================
    # Optimizer (NO video params)
    # =========================
    params = (
        list(imu_enc.parameters()) +
        list(proj_imu.parameters()) +
        list(proj_vid.parameters())
    )

    opt = torch.optim.AdamW(
        params,
        lr=float(cfg.get("lr", 1e-4))
    )

    # =========================
    # Output directory
    # =========================
    run_dir = make_run_dir(cfg.get("out_dir", "outputs/runs"), "pretrain")
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

    open(
        os.path.join(run_dir, "config.yaml"),
        "w",
        encoding="utf-8"
    ).write(yaml.safe_dump(cfg))

    imu_enc.train()
    proj_imu.train()
    proj_vid.train()

    # =========================
    # Training loop
    # =========================
    losses = []
    epochs = int(cfg.get("epochs", 5))

    for ep in range(epochs):
        pbar = tqdm(dl, desc=f"epoch {ep+1}/{epochs}")
        for batch in pbar:
            imu = batch["imu"].to(device)        # (B, L, C)
            video = batch["video"]               # keep on CPU for processor if needed

            # Forward
            z_imu = imu_enc(imu)                 # (B, 256)
            z_vid = vid_enc(video)               # (B, 768)

            # Project to shared space
            z_imu = proj_imu(z_imu)              # (B, 256)
            z_vid = proj_vid(z_vid)              # (B, 256)

            # Contrastive loss
            loss = siglip_loss(
                z_imu,
                z_vid,
                temperature=float(cfg.get("temperature", 1.0))
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Save checkpoint
        torch.save(
            {
                "imu_enc": imu_enc.state_dict(),
                "proj_imu": proj_imu.state_dict(),
                "proj_vid": proj_vid.state_dict(),
                "epoch": ep + 1,
            },
            os.path.join(run_dir, "checkpoints", f"epoch_{ep+1}.pt")
        )

    save_json(
        os.path.join(run_dir, "train_summary.json"),
        {
            "final_loss": losses[-1] if losses else None,
            "loss_curve_first_2000": losses[:2000],
        }
    )

    print(f"✅ Training finished. Run saved to: {run_dir}")


if __name__ == "__main__":
    main()
