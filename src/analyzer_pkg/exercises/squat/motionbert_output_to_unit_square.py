#!/usr/bin/env python3
"""
motionbert_output_to_unit_square.py  – reference‑frame version
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Union, Optional, List

import numpy as np
from scipy.ndimage import median_filter

EPS = 1e-8

# ── bbox utilities ───────────────────────────────────────────────
def _adaptive_bbox(kps: np.ndarray) -> Tuple[float, float, float]:
    valid = (kps[:, 2] > 0.05) if kps.shape[1] == 3 else ~np.isnan(kps[:, 0])
    if not np.any(valid):
        return 0.0, 0.0, 1.0
    xs, ys = kps[valid, 0], kps[valid, 1]
    xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()
    side = max(xmax - xmin, ymax - ymin) or 1.0
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    return cx - side / 2.0, cy - side / 2.0, side

def _normalise(kps: np.ndarray, bbox: Tuple[float, float, float]) -> np.ndarray:
    xmin, ymin, side = bbox
    out = kps.copy()
    if side < EPS:
        return out
    valid = (out[:, 2] > 0.05) if out.shape[1] == 3 else ~np.isnan(out[:, 0])
    out[valid, 0] = (out[valid, 0] - xmin) / side
    out[valid, 1] = 1.0 - (out[valid, 1] - ymin) / side
    return out

# ── public API ───────────────────────────────────────────────────
def pipeline(
    x3d: str | Path | np.ndarray,
    *,
    ref_idx: int,
    out_stable: str | Path | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (stable_array, adaptive_array).

    • stable side‑length = side of ref_idx frame
    • adaptive is still computed (may be handy for debugging)
    """
    arr_in = np.load(x3d) if not isinstance(x3d, np.ndarray) else x3d
    arr_in = arr_in.astype(np.float32, copy=True)
    F      = arr_in.shape[0]

    # pass 1 – adaptive arrays / boxes
    adaptive_bb: List[Tuple[float, float, float]] = []
    adaptive = arr_in.copy()
    for f in range(F):
        bb = _adaptive_bbox(arr_in[f])
        adaptive_bb.append(bb)
        adaptive[f] = _normalise(arr_in[f], bb)

    # side length of reference frame
    S_ref = adaptive_bb[ref_idx][2] or 1.0

    # pass 2 – stable array
    stable = arr_in.copy()
    for f, (xmin, ymin, _) in enumerate(adaptive_bb):
        stable[f] = _normalise(arr_in[f], (xmin, ymin, S_ref))

    if out_stable:
        p = Path(out_stable)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, stable)
        print(f"✅ stable array (ref {ref_idx}) → {p}  shape={stable.shape}")

    return stable, adaptive
