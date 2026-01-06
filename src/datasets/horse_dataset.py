import os
import csv
import glob
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

from src.utils.time_alignment import sample_frame_indices
from src.datasets.transforms import preprocess_frame_rgb

GAIT_TO_ID = {"stop": 0, "walk": 1, "trot": 2, "canter": 3}

def _read_imu_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (t, data) where data is (T, 6)."""
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append([
                float(r["t"]),
                float(r["ax"]), float(r["ay"]), float(r["az"]),
                float(r["gx"]), float(r["gy"]), float(r["gz"]),
            ])
    arr = np.array(rows, dtype=np.float32)
    return arr[:, 0], arr[:, 1:]

def _read_labels(path: str) -> List[Tuple[float, float, str]]:
    segs = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            segs.append((float(r["start_s"]), float(r["end_s"]), str(r["gait"])))
    return segs

def _label_for_window(segs, t0, t1) -> str:
    best_g, best_overlap = "stop", 0.0
    for s, e, g in segs:
        overlap = max(0.0, min(e, t1) - max(s, t0))
        if overlap > best_overlap:
            best_g, best_overlap = g, overlap
    return best_g

class HorseCrossModalDataset(Dataset):
    """
    Returns:
      imu:   (L, 30)  where L=window_s*imu_hz (250)
      video: (T, 3, H, W) where T=clip_len (e.g. 16), H=W=image_size (e.g. 224)
      label: int
    """
    def __init__(
        self,
        root: str,
        imu_hz: int = 50,
        window_s: float = 5.0,
        clip_len: int = 16,
        image_size: int = 224,
        use_mp4: bool = True,
    ):
        self.root = root
        self.imu_hz = imu_hz
        self.window_s = window_s
        self.win_len = int(round(window_s * imu_hz))
        self.clip_len = clip_len
        self.image_size = image_size
        self.use_mp4 = use_mp4

        self.sessions = sorted(glob.glob(os.path.join(root, "session_*")))
        if len(self.sessions) == 0:
            raise FileNotFoundError(f"No sessions found under {root}")

        # Pre-index non-overlapping windows across all sessions
        self.index = []
        for sess in self.sessions:
            imu_files = sorted(glob.glob(os.path.join(sess, "imu", "imu_*.csv")))
            if len(imu_files) < 5:
                continue
            t, _ = _read_imu_csv(imu_files[0])
            total_T = t.shape[0]
            step = self.win_len
            for start in range(0, total_T - self.win_len + 1, step):
                self.index.append((sess, start))

    def __len__(self):
        return len(self.index)

    def _read_frame_from_mp4(self, mp4_path: str, frame_idx: int) -> np.ndarray:
        cap = cv2.VideoCapture(mp4_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame_bgr = cap.read()
        cap.release()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"Failed reading frame {frame_idx} from {mp4_path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def _read_frame_from_frames_dir(self, frames_dir: str, frame_idx: int) -> np.ndarray:
        png = os.path.join(frames_dir, f"frame_{frame_idx:06d}.png")
        npy = os.path.join(frames_dir, f"frame_{frame_idx:06d}.npy")
        if os.path.exists(png):
            import imageio.v2 as imageio
            frame = imageio.imread(png)
            return frame
        if os.path.exists(npy):
            return np.load(npy)
        raise FileNotFoundError(f"No frame file found for idx={frame_idx} in {frames_dir}")

    def __getitem__(self, i: int):
        sess, start = self.index[i]

        imu_files = sorted(glob.glob(os.path.join(sess, "imu", "imu_*.csv")))[:5]

        imu_stack = []
        t = None
        for p in imu_files:
            tt, dd = _read_imu_csv(p)  # dd: (T,6)
            if t is None:
                t = tt
            imu_stack.append(dd[start:start + self.win_len])  # (L,6)

        imu_window = np.concatenate(imu_stack, axis=-1)  # (L,30)
        imu_window = torch.from_numpy(imu_window).float()

        t0 = float(t[start])
        t1 = float(t[start + self.win_len - 1] + (1.0 / self.imu_hz))

        segs = _read_labels(os.path.join(sess, "labels.csv"))
        gait = _label_for_window(segs, t0, t1)
        label = GAIT_TO_ID[gait]

        frame_ts = np.load(os.path.join(sess, "video", "cam_0_timestamps.npy"))
        frame_idx = sample_frame_indices(frame_ts, t0, t1, self.clip_len)

        mp4 = os.path.join(sess, "video", "cam_0.mp4")
        frames_dir = os.path.join(sess, "video", "frames_cam_0")

        frames_t = []
        for fi in frame_idx.tolist():
            if self.use_mp4 and os.path.exists(mp4):
                frame_rgb = self._read_frame_from_mp4(mp4, fi)
            else:
                frame_rgb = self._read_frame_from_frames_dir(frames_dir, fi)
            frames_t.append(preprocess_frame_rgb(frame_rgb, self.image_size))

        video_clip = torch.stack(frames_t, dim=0)  # (T,3,H,W)

        return {
            "imu": imu_window,  # (L,30)
            "video": video_clip,  # (T,3,H,W)
            "label": torch.tensor(label, dtype=torch.long),
            "session": os.path.basename(sess),
            "t0": torch.tensor(t0, dtype=torch.float32),
            "t1": torch.tensor(t1, dtype=torch.float32),
        }
