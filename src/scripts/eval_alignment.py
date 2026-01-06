import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.horse_dataset import HorseCrossModalDataset
from src.models.imu_encoder import PatchTSTIMUEncoder
from src.models.video_encoder import FrozenVideoEncoder
from src.models.projectors import ProjectionHead


@torch.no_grad()
def main():
    print("eval_alignment is running...")

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--batches", type=int, default=20, help="How many batches to evaluate")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = HorseCrossModalDataset(
        root=cfg["data_root"],
        imu_hz=int(cfg.get("imu_hz", 50)),
        window_s=float(cfg.get("window_s", 5.0)),
        clip_len=int(cfg.get("clip_len", 16)),
        image_size=int(cfg.get("image_size", 224)),
        use_mp4=True,
    )
    dl = DataLoader(ds, batch_size=int(cfg.get("batch_size", 8)), shuffle=True, num_workers=0)

    embed_dim = int(cfg.get("embed_dim", 256))

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

    vid_enc = FrozenVideoEncoder(out_dim=embed_dim).to(device)

    proj_imu = ProjectionHead(in_dim=embed_dim, proj_dim=embed_dim, hidden_dim=int(cfg.get("proj_hidden", 512))).to(device)
    proj_vid = ProjectionHead(in_dim=embed_dim, proj_dim=embed_dim, hidden_dim=int(cfg.get("proj_hidden", 512))).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    imu_enc.load_state_dict(ckpt["imu_enc"], strict=True)
    proj_imu.load_state_dict(ckpt["proj_imu"], strict=True)
    proj_vid.load_state_dict(ckpt["proj_vid"], strict=True)

    imu_enc.eval(); vid_enc.eval(); proj_imu.eval(); proj_vid.eval()

    pos_sims = []
    neg_sims = []

    it = iter(dl)
    for _ in range(args.batches):
        try:
            batch = next(it)
        except StopIteration:
            break

        imu = batch["imu"].to(device)
        video = batch["video"].to(device)

        z_imu = proj_imu(imu_enc(imu))
        z_vid = proj_vid(vid_enc(video))

        z_imu = F.normalize(z_imu, dim=-1)
        z_vid = F.normalize(z_vid, dim=-1)

        # Positive: diagonal
        sim_pos = (z_imu * z_vid).sum(dim=-1)  # (B,)

        # Negative: shuffle video embeddings
        perm = torch.randperm(z_vid.size(0), device=device)
        sim_neg = (z_imu * z_vid[perm]).sum(dim=-1)

        pos_sims.append(sim_pos.cpu())
        neg_sims.append(sim_neg.cpu())

    pos = torch.cat(pos_sims).numpy()
    neg = torch.cat(neg_sims).numpy()

    print("Alignment evaluation")
    print(f"  mean(sim_pos) = {pos.mean():.4f}  std = {pos.std():.4f}")
    print(f"  mean(sim_neg) = {neg.mean():.4f}  std = {neg.std():.4f}")
    print("  (Goal: mean(sim_pos) > mean(sim_neg))")


if __name__ == "__main__":
    main()
