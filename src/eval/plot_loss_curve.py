import json
import matplotlib.pyplot as plt
import os

RUN_DIR = r"outputs\runs\pretrain_20251229_003430"  # change if needed
SUMMARY_PATH = os.path.join(RUN_DIR, "train_summary.json")
OUT_DIR = RUN_DIR

with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
    summary = json.load(f)

losses = summary.get("loss_curve_first_2000", [])
if not losses:
    raise RuntimeError("No loss curve found in train_summary.json")

plt.figure(figsize=(7,4))
plt.plot(losses)
plt.title("Pretraining Loss Curve (first 2000 steps)")
plt.xlabel("Step")
plt.ylabel("Sigmoid contrastive loss")
out_path = os.path.join(OUT_DIR, "loss_curve.png")
plt.tight_layout()
plt.savefig(out_path, dpi=200)
print("Saved:", out_path)
