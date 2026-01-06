import numpy as np

def sample_frame_indices(frame_ts: np.ndarray, t0: float, t1: float, clip_len: int):
    """Return `clip_len` indices evenly spaced within [t0, t1)."""
    mask = (frame_ts >= t0) & (frame_ts < t1)
    idx = np.where(mask)[0]
    if idx.size == 0:
        # fallback: nearest frame to t0
        nearest = int(np.argmin(np.abs(frame_ts - t0)))
        return np.array([nearest] * clip_len, dtype=np.int64)
    if idx.size < clip_len:
        # repeat to fill
        reps = int(np.ceil(clip_len / idx.size))
        idx_rep = np.tile(idx, reps)[:clip_len]
        return idx_rep.astype(np.int64)
    # evenly spaced selection
    lin = np.linspace(0, idx.size - 1, clip_len).round().astype(np.int64)
    return idx[lin].astype(np.int64)
