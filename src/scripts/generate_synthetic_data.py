# Copy of the synthetic dataset generator (lightly adapted).
import os
import json
import csv
import argparse
import numpy as np

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)

class GaitSpec:
    def __init__(self, name, stride_hz, accel_amp, gyro_amp):
        self.name = name
        self.stride_hz = stride_hz
        self.accel_amp = accel_amp
        self.gyro_amp = gyro_amp

GAITS = {
    "stop":   GaitSpec("stop",   0.2, 0.05, 0.02),
    "walk":   GaitSpec("walk",   1.5, 1.0,  0.6),
    "trot":   GaitSpec("trot",   2.6, 1.8,  1.2),
    "canter": GaitSpec("canter", 3.2, 2.4,  1.6),
}

def generate_gait_schedule(total_s, min_seg_s=6.0, max_seg_s=18.0):
    gaits = list(GAITS.keys())
    t = 0.0
    segs = []
    last = None
    while t < total_s - 1e-6:
        dur = float(np.random.uniform(min_seg_s, max_seg_s))
        end = min(total_s, t + dur)
        options = [g for g in gaits if g != last] if last is not None else gaits
        gait = str(np.random.choice(options))
        segs.append((t, end, gait))
        last = gait
        t = end
    return segs

def make_transition_envelope(t, segs, ramp_s=0.6):
    weights = {k: np.zeros_like(t, dtype=np.float32) for k in GAITS.keys()}
    for s, e, g in segs:
        w = np.ones_like(t, dtype=np.float32)
        w *= smoothstep((t - s) / ramp_s)
        w *= smoothstep((e - t) / ramp_s)
        w[(t < s) | (t >= e)] = 0.0
        weights[g] += w
    total = np.zeros_like(t, dtype=np.float32)
    for g in weights:
        total += weights[g]
    total = np.maximum(total, 1e-6)
    for g in weights:
        weights[g] = weights[g] / total
    return weights

def generate_imu_signals(t, weights, n_imus=5, noise_std_acc=0.12, noise_std_gyro=0.08):
    T = t.shape[0]
    imu = np.zeros((T, n_imus, 6), dtype=np.float32)

    stride = np.zeros(T, dtype=np.float32)
    aamp = np.zeros(T, dtype=np.float32)
    gamp = np.zeros(T, dtype=np.float32)
    for gname, w in weights.items():
        spec = GAITS[gname]
        stride += w * spec.stride_hz
        aamp += w * spec.accel_amp
        gamp += w * spec.gyro_amp

    drift = np.cumsum(np.random.normal(0, 0.002, size=T)).astype(np.float32)
    drift = drift - drift.mean()

    phase = 2 * np.pi * np.cumsum(stride / max(stride.mean(), 1e-3)) * (t[1] - t[0])
    base1 = np.sin(phase)
    base2 = np.sin(2 * phase + 0.7)
    base3 = np.cos(phase + 1.2)

    gravity = 9.81
    base_acc = np.stack([
        0.35 * aamp * base1 + 0.10 * aamp * base2,
        0.25 * aamp * base3 + 0.08 * aamp * base2,
        gravity + 0.45 * aamp * np.abs(base1) + 0.15 * aamp * base2
    ], axis=-1).astype(np.float32)

    base_gyro = np.stack([
        0.40 * gamp * base3 + 0.10 * gamp * base2,
        0.30 * gamp * base1 + 0.12 * gamp * base2,
        0.20 * gamp * base1 + 0.08 * gamp * base3
    ], axis=-1).astype(np.float32)

    for i in range(n_imus):
        phi = np.random.uniform(-np.pi, np.pi)
        amp_scale_a = np.random.uniform(0.85, 1.20)
        amp_scale_g = np.random.uniform(0.85, 1.25)
        bias_acc = np.random.normal(0, 0.08, size=3).astype(np.float32)
        bias_gyro = np.random.normal(0, 0.05, size=3).astype(np.float32)

        extra = np.stack([np.sin(phase + phi), np.cos(phase + 0.5 * phi), np.sin(2 * phase - 0.3 * phi)], axis=-1).astype(np.float32)
        acc = amp_scale_a * base_acc + 0.12 * aamp[:, None].astype(np.float32) * extra
        gyro = amp_scale_g * base_gyro + 0.10 * gamp[:, None].astype(np.float32) * extra
        acc[:, 0] += 0.06 * drift
        gyro[:, 2] += 0.04 * drift
        acc += np.random.normal(0, noise_std_acc, size=acc.shape).astype(np.float32)
        gyro += np.random.normal(0, noise_std_gyro, size=gyro.shape).astype(np.float32)
        acc += bias_acc
        gyro += bias_gyro
        imu[:, i, :3] = acc
        imu[:, i, 3:] = gyro

    return imu

def save_imu_csv(path, t, data_6):
    header = ["t", "ax", "ay", "az", "gx", "gy", "gz"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(t.shape[0]):
            w.writerow([f"{t[i]:.6f}"] + [f"{x:.6f}" for x in data_6[i].tolist()])

def save_labels_csv(path, segs):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_s", "end_s", "gait"])
        for s, e, g in segs:
            w.writerow([f"{s:.3f}", f"{e:.3f}", g])

def write_calibration(calib_dir, width, height):
    ensure_dir(calib_dir)
    fx = 0.9 * width
    fy = 0.9 * width
    cx = width / 2
    cy = height / 2
    intr = {"model": "pinhole", "width": width, "height": height,
            "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]}
    extr = {"R": [[1,0,0],[0,1,0],[0,0,1]], "t": [0,0,0]}
    with open(os.path.join(calib_dir, "cam_intrinsics.json"), "w") as f:
        json.dump(intr, f, indent=2)
    with open(os.path.join(calib_dir, "cam_extrinsics.json"), "w") as f:
        json.dump(extr, f, indent=2)

def render_frame(width, height, center_x, center_y, gait_name):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    yy = np.linspace(0, 1, height)[:, None]
    img[..., 1] = (30 + 40 * yy).astype(np.uint8)
    img[..., 2] = (20 + 25 * yy).astype(np.uint8)
    gait_color = {
        "stop": (200, 200, 200),
        "walk": (120, 220, 120),
        "trot": (220, 180, 80),
        "canter": (220, 120, 200),
    }[gait_name]
    cx = int(np.clip(center_x, 20, width - 20))
    cy = int(np.clip(center_y, 20, height - 20))
    rr_y, rr_x = 12, 22
    y0, y1 = max(0, cy - rr_y), min(height, cy + rr_y)
    x0, x1 = max(0, cx - rr_x), min(width, cx + rr_x)
    patch = img[y0:y1, x0:x1]
    if patch.size > 0:
        ys = np.arange(y0, y1)[:, None] - cy
        xs = np.arange(x0, x1)[None, :] - cx
        mask = (xs.astype(np.float32)**2) / (rr_x**2) + (ys.astype(np.float32)**2) / (rr_y**2) <= 1.0
        patch[mask] = gait_color
    return img

def try_get_video_writer(path, fps):
    try:
        import imageio.v2 as imageio
        return imageio.get_writer(path, fps=fps, codec="libx264", quality=8), True
    except Exception:
        return None, False

def generate_session(out_dir, session_id, duration_s, imu_hz, video_fps, width, height, n_imus):
    sess_dir = os.path.join(out_dir, f"session_{session_id:02d}")
    imu_dir = os.path.join(sess_dir, "imu")
    vid_dir = os.path.join(sess_dir, "video")
    calib_dir = os.path.join(sess_dir, "calib")
    ensure_dir(imu_dir); ensure_dir(vid_dir); ensure_dir(calib_dir)

    write_calibration(calib_dir, width, height)

    segs = generate_gait_schedule(duration_s)
    t_imu = np.arange(0, duration_s, 1.0 / imu_hz, dtype=np.float32)
    t_vid = np.arange(0, duration_s, 1.0 / video_fps, dtype=np.float32)

    weights_imu = make_transition_envelope(t_imu, segs, ramp_s=0.6)
    imu = generate_imu_signals(t_imu, weights_imu, n_imus=n_imus)

    for i in range(n_imus):
        save_imu_csv(os.path.join(imu_dir, f"imu_{i}.csv"), t_imu, imu[:, i, :])

    save_labels_csv(os.path.join(sess_dir, "labels.csv"), segs)

    # video
    np.save(os.path.join(vid_dir, "cam_0_timestamps.npy"), t_vid)

    mp4_path = os.path.join(vid_dir, "cam_0.mp4")
    writer, ok = try_get_video_writer(mp4_path, video_fps)
    frames_dir = os.path.join(vid_dir, "frames_cam_0")
    if not ok:
        ensure_dir(frames_dir)

    # simple motion
    gait_seq_vid = np.array([g for _, _, g in segs for g in []], dtype=object)  # unused
    # linear x motion + gait-driven bobbing (by nearest segment gait)
    def gait_for_time(tt):
        for s, e, g in segs:
            if s <= tt < e:
                return g
        return segs[-1][2]

    stride_vid = np.array([GAITS[gait_for_time(tt)].stride_hz for tt in t_vid], dtype=np.float32)
    bob = 10.0 * np.sin(2 * np.pi * np.cumsum(stride_vid) / max(video_fps, 1))

    center_x = (width * 0.2) + (width * 0.6) * (t_vid / max(duration_s, 1e-6))
    center_y = (height * 0.55) + bob

    for k, tt in enumerate(t_vid):
        g = gait_for_time(float(tt))
        frame = render_frame(width, height, center_x[k], center_y[k], g)
        if ok:
            writer.append_data(frame)
        else:
            # save png
            try:
                import imageio.v2 as imageio
                imageio.imwrite(os.path.join(frames_dir, f"frame_{k:06d}.png"), frame)
            except Exception:
                np.save(os.path.join(frames_dir, f"frame_{k:06d}.npy"), frame)

    if ok:
        writer.close()

    meta = {
        "session_id": session_id,
        "duration_s": duration_s,
        "imu_hz": imu_hz,
        "video_fps": video_fps,
        "n_imus": n_imus,
        "video": {"mp4": "video/cam_0.mp4", "timestamps": "video/cam_0_timestamps.npy", "fallback_frames_dir": "video/frames_cam_0"},
        "imu": {"files": [f"imu/imu_{i}.csv" for i in range(n_imus)]},
        "labels": "labels.csv",
        "calib": {"intrinsics": "calib/cam_intrinsics.json", "extrinsics": "calib/cam_extrinsics.json"}
    }
    with open(os.path.join(sess_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/synthetic_horse_dataset")
    ap.add_argument("--sessions", type=int, default=10)
    ap.add_argument("--duration_s", type=float, default=90.0)
    ap.add_argument("--imu_hz", type=int, default=50)
    ap.add_argument("--video_fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--imus", type=int, default=5)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    np.random.seed(args.seed)
    ensure_dir(args.out)

    for s in range(args.sessions):
        np.random.seed(args.seed + 1000 * s)
        generate_session(args.out, s, args.duration_s, args.imu_hz, args.video_fps, args.width, args.height, args.imus)

    print(f"Generated dataset at: {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
