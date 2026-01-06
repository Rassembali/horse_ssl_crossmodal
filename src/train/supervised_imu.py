import argparse
import os
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.utils.seed import set_seed
from src.utils.logging import make_run_dir, save_json
from src.utils.metrics import accuracy
from src.datasets.horse_dataset import HorseCrossModalDataset
from src.models.imu_encoder import PatchTSTIMUEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(int(cfg.get("seed", 123)))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = HorseCrossModalDataset(
        root=cfg["data_root"],
        imu_hz=int(cfg.get("imu_hz", 50)),
        window_s=float(cfg.get("window_s", 5.0)),
        clip_len=int(cfg.get("clip_len", 16)),
        image_size=int(cfg.get("image_size", 224)),
        use_mp4=True,
    )

    n = len(ds)
    n_train = int(0.8 * n)
    n_val = n - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    dl_tr = DataLoader(train_ds, batch_size=int(cfg.get("batch_size", 16)), shuffle=True, num_workers=0, pin_memory=True)
    dl_va = DataLoader(val_ds, batch_size=int(cfg.get("batch_size", 16)), shuffle=False, num_workers=0, pin_memory=True)

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

    head = nn.Linear(embed_dim, int(cfg.get("num_classes", 4))).to(device)

    opt = torch.optim.AdamW(list(imu_enc.parameters()) + list(head.parameters()), lr=float(cfg.get("lr", 1e-4)))
    loss_fn = nn.CrossEntropyLoss()

    run_dir = make_run_dir(cfg.get("out_dir", "outputs/runs"), "supervised_imu")
    open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8").write(yaml.safe_dump(cfg))

    epochs = int(cfg.get("epochs", 10))
    for ep in range(epochs):
        imu_enc.train(); head.train()
        tr_losses = []
        for batch in tqdm(dl_tr, desc=f"train {ep+1}/{epochs}"):
            imu = batch["imu"].to(device)
            y = batch["label"].to(device)

            z = imu_enc(imu)
            logits = head(z)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_losses.append(float(loss.item()))

        imu_enc.eval(); head.eval()
        accs = []
        for batch in dl_va:
            imu = batch["imu"].to(device)
            y = batch["label"].to(device)
            with torch.no_grad():
                z = imu_enc(imu)
                logits = head(z)
            accs.append(accuracy(logits, y))

        print(f"epoch {ep+1}: train_loss={sum(tr_losses)/max(1,len(tr_losses)):.4f} val_acc={sum(accs)/max(1,len(accs)):.4f}")

    save_json(os.path.join(run_dir, "results.json"), {
        "val_acc": float(sum(accs)/max(1, len(accs))) if accs else None
    })
    print(f"Saved supervised baseline to: {run_dir}")


if __name__ == "__main__":
    main()
