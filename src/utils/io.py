import os
import json
import numpy as np

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_npy(path):
    return np.load(path)
