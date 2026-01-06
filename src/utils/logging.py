import os
import json
import time

def make_run_dir(out_dir: str, prefix: str):
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"{prefix}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    return run_dir

def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
