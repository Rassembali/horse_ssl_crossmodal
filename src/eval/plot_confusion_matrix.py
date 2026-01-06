import numpy as np
import matplotlib.pyplot as plt
import os

# Paths (edit if needed)
TRUE_PATH = r"outputs\linear_probe\val_true.npy"
PRED_PATH = r"outputs\linear_probe\val_pred.npy"
OUT_DIR = r"outputs\linear_probe"

os.makedirs(OUT_DIR, exist_ok=True)

y_true = np.load(TRUE_PATH)
y_pred = np.load(PRED_PATH)

classes = np.unique(np.concatenate([y_true, y_pred]))
K = int(classes.max() + 1)

cm = np.zeros((K, K), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[int(t), int(p)] += 1

# Normalize by true class (row)
cm_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

plt.figure(figsize=(6, 5))
plt.imshow(cm_norm, interpolation="nearest")
plt.title("Confusion Matrix (Normalized by True Class)")
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.colorbar()

plt.xticks(range(K), [str(i) for i in range(K)])
plt.yticks(range(K), [str(i) for i in range(K)])

for i in range(K):
    for j in range(K):
        plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")

out_path = os.path.join(OUT_DIR, "confusion_matrix.png")
plt.tight_layout()
plt.savefig(out_path, dpi=200)
print("Saved:", out_path)
