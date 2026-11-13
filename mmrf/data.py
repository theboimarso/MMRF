"""Data handling: 4D sequences stored as one file per frame, {subject}_frame{t:02d}_ct.npz (key 'ct', a 96^3 volume, any leading singleton dims).
Frame 0 is the reference. Every volume is clipped to its 1st-99th intensity percentiles and rescaled to [0, 1] when loaded."""
import json
from pathlib import Path

import numpy as np
import torch

SIZE = 96
CENTRE = SIZE // 2                                                           # index of the surrogate plane along its axis


def load_volume(path):
    """(96, 96, 96) float32 in [0, 1]: clipped to the volume's 1st-99th percentiles, then rescaled."""
    d = np.load(path)
    key = next((k for k in ("ct", "vol", "image", "data", "arr_0") if k in d.files), d.files[0])
    v = d[key].astype(np.float32).squeeze()
    p1, p99 = np.percentile(v, 1), np.percentile(v, 99)
    v = np.clip(v, p1, p99)
    return (v - p1) / (p99 - p1) if p99 > p1 else v


def frame_files(data_dir, subject):
    fs = Path(data_dir).glob(f"{subject}_frame*_ct.npz")
    return sorted(fs, key=lambda f: int(f.stem.split("frame")[1].split("_")[0]))


def load_sequence(data_dir, subject):
    """(T, 96, 96, 96) float32, frame 0 = reference."""
    return np.stack([load_volume(f) for f in frame_files(data_dir, subject)])


def load_split(path):
    """{"train": [...], "val": [...], "test": [...]} subject identifiers."""
    return json.load(open(path))


def subjects(data_dir, ids, min_frames=3):
    """the subjects of ids that have at least min_frames frames on disk"""
    return [s for s in ids if len(frame_files(data_dir, s)) >= min_frames]


def plane(vol, axis, index=CENTRE):
    """the surrogate plane: the slice at index along axis (lung / liver navigator: coronal, axis 1; cardiac: short axis, axis 0).
    vol: numpy (..., D, H, W) or torch tensor with the same trailing layout."""
    if isinstance(vol, torch.Tensor): return vol.select(vol.dim() - 3 + axis, index)
    return np.take(vol, index, axis=vol.ndim - 3 + axis)
