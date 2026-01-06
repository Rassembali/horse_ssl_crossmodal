# 🐎 Horse Locomotion: Cross-Modal Alignment via Self-Supervised Learning

This repository presents a partial implementation of a self-supervised cross-modal learning framework designed to align multi-sensor IMU time series with video representations for equine gait analysis.  
The system learns a robust sensor-only encoder by distilling knowledge from a frozen visual “teacher” model.

---

## 📌 Objectives

- Synchronize and align data from 5 IMU sensors with visual features extracted from VideoMAE-base  
- Learn a shared latent space using Sigmoid Contrastive Loss (SigLIP)  
- Enable accurate classification of horse gaits — Walk / Trot / Canter — using wearable data only  
- Provide a modular research codebase demonstrating:
  - Time-series Transformers  
  - Cross-modal representation learning  
  - Knowledge distillation from foundation models

---

## 🛠 Key Technical Features

- Sensor Fusion from 5 IMUs (Accelerometer + Gyroscope) creating a 30-channel input  
- PatchTST-like IMU Transformer Encoder with 25 patches × 10 steps  
- Frozen VideoMAE-base (768-d) teacher backbone  
- SigLIP projection heads to 256-d shared space  
- Two-stage training: self-supervised pretraining + linear probe

---

## 📂 Repository Structure

configs/        → YAML configuration files  
src/datasets/   → IMU/video loaders & synchronization logic  
src/models/     → PatchTST encoder, VideoMAE wrapper, projections  
src/losses/     → SigLIP contrastive objective  
src/train/      → pretrain.py / probe.py  
src/eval/       → metrics, confusion matrix, curves

| Directory | Description |
|---|---|
| configs | Hyperparameters for pretraining & probing |
| datasets | Cross-modal pair construction |
| models | Encoders and projection heads |
| losses | Alignment objectives |
| train | Training entry points |
| eval | Evaluation utilities |

---

## 🔬 Architecture in Detail

### 1. IMU Branch — Patch Transformer

- Input aggregation from:
  - Accelerometer (x, y, z)  
  - Gyroscope (x, y, z)  
- Total = 30 channels  
- 250 timesteps @ 50Hz

Processing pipeline:

IMU (30×250)  
→ Patching (10,10) → 25 patches  
→ CLS Token + Positional Encoding  
→ Transformer Encoder  
→ 256-d sensor embedding

---

### 2. Video Branch — Frozen Foundation Teacher

- Backbone: VideoMAE-base  
- Output: mean over tokens → 768-d  
- Backbone remains frozen to prevent overfitting on limited datasets

---

### 3. Cross-Modal Alignment — SigLIP

- Both modalities projected to 256-d  
- Similarity matrix learned via sigmoid binary objectives  
- Diagonal → positive pairs  
- Off-diagonal → negatives

---

## 📈 Training Strategy

1. Stage A — Self-Supervised Pretraining  
   - Train IMU encoder + projections  
   - Align with frozen VideoMAE teacher

2. Stage B — Linear Probe  
   - Freeze IMU encoder  
   - Train lightweight classifier to validate embedding quality

