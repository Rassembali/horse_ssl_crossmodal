import argparse
import os
import yaml
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.utils.seed import set_seed
from src.datasets.horse_dataset import HorseCrossModalDataset
from src.models.imu_encoder import PatchTSTIMUEncoder


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Balanced accuracy = mean recall over classes.
    """
    classes = np.unique(y_true)
    recalls = []
    for c in classes:
        mask = (y_true == c)
        if mask.sum() == 0:
            continue
        recalls.append((y_pred[mask] == c).mean())
    return float(np.mean(recalls)) if len(recalls) > 0 else 0.0


@torch.no_grad()
def extract_embeddings(
    imu_enc: nn.Module,
    dl: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X: (N, D) embeddings
      y: (N,) int labels
    """
    imu_enc.eval()
    X_list, y_list = [], []

    for batch in dl:
        imu = batch["imu"].to(device)          # (B, L, C)
        y = batch["label"].cpu().numpy()       # (B,)

        z = imu_enc(imu).detach().cpu().numpy()  # (B, D)

        X_list.append(z)
        y_list.append(y)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0).astype(np.int64)
    return X, y


def train_linear_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    epochs: int = 50,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Train a simple linear layer on frozen embeddings.
    Returns:
      val_pred, val_true, stats
    """
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)

    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)

    D = X_train.shape[1]
    clf = nn.Linear(D, num_classes).to(device)

    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    best_bal = -1.0
    best_state = None

    for ep in range(epochs):
        clf.train()
        opt.zero_grad()
        logits = clf(X_train_t)
        loss = crit(logits, y_train_t)
        loss.backward()
        opt.step()

        clf.eval()
        with torch.no_grad():
            val_logits = clf(X_val_t)
            val_pred = val_logits.argmax(dim=1).cpu().numpy()
            bal = balanced_accuracy(y_val, val_pred)
            acc = float((val_pred == y_val).mean())

        if bal > best_bal:
            best_bal = bal
            best_state = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[ep {ep+1:03d}] loss={loss.item():.4f}  val_acc={acc:.4f}  val_bal_acc={bal:.4f}")

    # load best
    if best_state is not None:
        clf.load_state_dict(best_state)

    clf.eval()
    with torch.no_grad():
        val_logits = clf(X_val_t)
        val_pred = val_logits.argmax(dim=1).cpu().numpy()

    stats = {"best_val_balanced_accuracy": float(best_bal)}
    return val_pred, y_val, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to configs/pretrain.yaml")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to pretrained checkpoint epoch_X.pt")
    ap.add_argument("--train_frac", type=float, default=0.8, help="Fraction for train split")
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size for embedding extraction")
    ap.add_argument("--epochs", type=int, default=50, help="Training epochs for linear probe")
    ap.add_argument("--lr", type=float, default=1e-2, help="Learning rate for linear probe")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # -------------------------
    # Dataset
    # -------------------------
    ds = HorseCrossModalDataset(
        root=cfg["data_root"],
        imu_hz=int(cfg.get("imu_hz", 50)),
        window_s=float(cfg.get("window_s", 5.0)),
        clip_len=int(cfg.get("clip_len", 16)),      # not used here but dataset returns it
        image_size=int(cfg.get("image_size", 224)), # not used here but dataset returns it
        use_mp4=True,
    )

    N = len(ds)
    idx = np.arange(N)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)

    n_train = int(args.train_frac * N)
    train_idx = idx[:n_train].tolist()
    val_idx = idx[n_train:].tolist()

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # -------------------------
    # IMU encoder (same config)
    # -------------------------
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

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    imu_enc.load_state_dict(ckpt["imu_enc"], strict=False)


    # Freeze encoder
    for p in imu_enc.parameters():
        p.requires_grad = False

    # -------------------------
    # Extract embeddings
    # -------------------------
    print("Extracting embeddings...")
    X_train, y_train = extract_embeddings(imu_enc, train_dl, device)
    X_val, y_val = extract_embeddings(imu_enc, val_dl, device)

    num_classes = int(np.max(np.concatenate([y_train, y_val])) + 1)
    print(f"Embeddings: train {X_train.shape}, val {X_val.shape}, classes={num_classes}")

    # -------------------------
    # Train linear probe
    # -------------------------
    val_pred, val_true, stats = train_linear_classifier(
        X_train, y_train,
        X_val, y_val,
        num_classes=num_classes,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )

    acc = float((val_pred == val_true).mean())
    bal = balanced_accuracy(val_true, val_pred)

    print("\n==== Linear probe results ====")
    print("Val Accuracy:", acc)
    print("Val Balanced Accuracy:", bal)
    print("Best Val Balanced Accuracy (during training):", stats["best_val_balanced_accuracy"])

    # Optionally save results
    out_dir = os.path.join("outputs", "linear_probe")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "val_pred.npy"), val_pred)
    np.save(os.path.join(out_dir, "val_true.npy"), val_true)
    with open(os.path.join(out_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"val_accuracy={acc}\n")
        f.write(f"val_balanced_accuracy={bal}\n")
        f.write(f"best_val_balanced_accuracy={stats['best_val_balanced_accuracy']}\n")

    print(f"\nSaved predictions + metrics to: {out_dir}")


if __name__ == "__main__":
    main()
