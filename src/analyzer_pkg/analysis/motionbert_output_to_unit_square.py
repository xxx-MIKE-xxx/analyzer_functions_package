#!/usr/bin/env python3
"""
motionbert_output_to_unit_square.py
───────────────────────────────────
Normalises the *(x, y)* part of a MotionBERT **(F, 17, 3)** array so every
coordinate lives in the unit square:

    (0, 0) ─ bottom-left      (1, 1) ─ top-right

• **x** : left → right  → divide by global *max x*  
• **y** : top → bottom → divide by global *max y* **and** flip so bottom = 0  
• **z** : unchanged (still in metres)

API mirrors `json_to_npy.pipeline` for consistency.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Union

import numpy as np


# ────────────────────────── public helper ──────────────────────────
def pipeline(
    x3d: Union[str, Path, np.ndarray],
    out_path: Union[str, Path, None] = None,
) -> np.ndarray:
    """
    Parameters
    ----------
    x3d       : (F, 17, 3) array **or** path to an .npy file.
    out_path  : optional destination to save the scaled array.

    Returns
    -------
    ndarray of shape (F, 17, 3) with x,y ∈ [0, 1]; z unchanged.
    """
    # 1) load / copy
    arr = np.load(x3d) if not isinstance(x3d, np.ndarray) else x3d
    arr = arr.astype(np.float32, copy=True)          # avoid mutating caller data

    # 2) global bounds (ignore NaNs)
    max_x = float(np.nanmax(arr[..., 0]))
    max_y = float(np.nanmax(arr[..., 1]))
    fx, fy = (1.0 / max_x if max_x else 1.0,
              1.0 / max_y if max_y else 1.0)

    # 3) scale in-place
    valid = ~np.isnan(arr[..., 0]) & ~np.isnan(arr[..., 1])
    arr[..., 0][valid] *= fx
    arr[..., 1][valid] = 1.0 - (arr[..., 1][valid] * fy)   # flip Y

    # 4) optional save
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, arr)
        print(f"✅ wrote {out_path}  shape={arr.shape}")

    return arr


# ────────────────────────── CLI entry ──────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit("Usage:\n  python motionbert_output_to_unit_square.py <in-npy> [out-npy]")
    _inp  = sys.argv[1]
    _out  = sys.argv[2] if len(sys.argv) == 3 else None
    pipeline(_inp, _out)
