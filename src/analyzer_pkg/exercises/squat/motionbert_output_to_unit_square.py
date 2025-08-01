#!/usr/bin/env python3
"""
motionbert_output_to_unit_square.py  – reference‑frame version (2D/3D unified)
------------------------------------------------------------------------------

This version scales **either** a 2D (F, K, 2) or 3D (F, K, 3) skeleton sequence
to the same unit square, using the same reference logic, for perfect alignment
between modalities.

• "Stable" mode: all frames are scaled to the *side* and *origin* of the reference frame bbox.
• "Adaptive" mode: each frame is scaled to its own bbox (useful for visualisation/debug).

Public API:
    pipeline(arr: ndarray, ref_idx: int, ...) -> (stable, adaptive)
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Union, Optional, List
import numpy as np

EPS = 1e-8

def _adaptive_bbox(kps: np.ndarray) -> Tuple[float, float, float]:
    """Calculate a square bounding box around valid keypoints (2D or 3D)."""
    # Accept both (K,2) and (K,3); use 3rd channel if present for visibility
    if kps.shape[1] >= 3:
        valid = (kps[:, 2] > 0.05)
    else:
        valid = ~np.isnan(kps[:, 0]) & ~np.isnan(kps[:, 1])
    if not np.any(valid):
        return 0.0, 0.0, 1.0  # default box if all missing
    xs, ys = kps[valid, 0], kps[valid, 1]
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    side = max(xmax - xmin, ymax - ymin) or 1.0
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    return cx - side / 2.0, cy - side / 2.0, side

def _normalise_and_crop(kps: np.ndarray, bbox: Tuple[float, float, float]) -> np.ndarray:
    """Normalize to unit square using bbox; discard points outside the square."""
    xmin, ymin, side = bbox
    out = kps.copy()
    if side < EPS:
        return out
    is_3d = out.shape[1] >= 3
    valid = ~np.isnan(out[:, 0]) & ~np.isnan(out[:, 1])
    xnorm = (out[valid, 0] - xmin) / side
    ynorm = 1.0 - (out[valid, 1] - ymin) / side  # always flip Y for y-up!
    inside = (xnorm >= 0) & (xnorm <= 1) & (ynorm >= 0) & (ynorm <= 1)
    out_valid_idx = np.where(valid)[0]
    for i, v in enumerate(out_valid_idx):
        if inside[i]:
            out[v, 0] = xnorm[i]
            out[v, 1] = ynorm[i]
        else:
            out[v, 0] = np.nan
            out[v, 1] = np.nan
            # In 3D: preserve z as-is (do not touch out[v, 2])
    return out

def pipeline(
    arr: str | Path | np.ndarray,
    *,
    ref_idx: int,
    out_stable: str | Path | None = None,
    out_adaptive: str | Path | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale (F, K, 2|3) skeleton sequence into unit square, y-up.
    Returns (stable, adaptive).

    Parameters
    ----------
    arr         : ndarray or path to .npy (F, K, 2|3)
    ref_idx     : int, index of reference frame for "stable" scaling
    out_stable  : optional, path to save stable result
    out_adaptive: optional, path to save adaptive result

    Returns
    -------
    stable   : ndarray, (F, K, 2|3), all frames scaled to reference bbox
    adaptive : ndarray, (F, K, 2|3), each frame scaled to its own bbox
    """
    arr_in = np.load(arr) if not isinstance(arr, np.ndarray) else arr
    arr_in = arr_in.astype(np.float32, copy=True)
    F = arr_in.shape[0]

    # pass 1 – adaptive arrays / boxes
    adaptive_bb: List[Tuple[float, float, float]] = []
    adaptive = arr_in.copy()
    for f in range(F):
        bb = _adaptive_bbox(arr_in[f])
        adaptive_bb.append(bb)
        adaptive[f] = _normalise_and_crop(arr_in[f], bb)

    # side length of reference frame
    ref_bbox = adaptive_bb[ref_idx]
    S_ref = ref_bbox[2] or 1.0

    # pass 2 – stable array
    stable = arr_in.copy()
    for f in range(F):
        xmin, ymin, _ = adaptive_bb[ref_idx]  # always use ref frame
        stable[f] = _normalise_and_crop(arr_in[f], (xmin, ymin, S_ref))

    # Optionally write output
    if out_stable:
        p = Path(out_stable)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, stable)
        print(f"✅ stable array (ref {ref_idx}) → {p}  shape={stable.shape}")

    if out_adaptive:
        p = Path(out_adaptive)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, adaptive)
        print(f"✅ adaptive array → {p}  shape={adaptive.shape}")

    return stable, adaptive

# ---------------- CLI (optional for direct use/testing) -------------------- #
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Scale skeletons (2D/3D) to unit square (y-up, ref frame logic)")
    p.add_argument("--input", required=True, help=".npy file (F,K,2|3)")
    p.add_argument("--ref-frame", type=int, required=True, help="Reference frame for scaling")
    p.add_argument("--out-stable", required=True, help="Output .npy for stable (ref-aligned)")
    p.add_argument("--out-adaptive", default=None, help="Optional output .npy for adaptive (per-frame) scaling")
    args = p.parse_args()
    stable, adaptive = pipeline(args.input, ref_idx=args.ref_frame, out_stable=args.out_stable, out_adaptive=args.out_adaptive)
    print(f"Done. stable: {stable.shape}, adaptive: {adaptive.shape}")
