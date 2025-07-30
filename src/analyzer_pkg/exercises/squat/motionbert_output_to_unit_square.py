#!/usr/bin/env python3
"""
motionbert_output_to_unit_square.py  – dual normalisers (adaptive & stable)
===========================================================================

Produces **two** MotionBERT‑scaled arrays:

1. *adaptive* – per‑frame square boxing (same as before)
2. *stable*   – one fixed side‑length S* found automatically

The function `pipeline()` returns **(stable, adaptive)** and is
back‑compatible with older callers that still pass `out_path=`.

---------------------------------------------------------------------------
Stable‑side detection
---------------------------------------------------------------------------
• Compute the per‑frame square side *s(t)* from the adaptive pass  
• Median‑filter (width=9) → *smooth_s*  
• Keep frames within the 5‑th … 95‑th percentiles of *smooth_s*  
• Find the **longest contiguous run** among those frames  
• **S\*** = median side‑length of that run
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union, Optional, List

import numpy as np
from scipy.ndimage import median_filter


EPS = 1e-8


# ────────────────────────────── utilities ──────────────────────────────
def _adaptive_bbox(kps: np.ndarray) -> Tuple[float, float, float]:
    """xmin, ymin, side  for the per‑frame adaptive square."""
    if kps.shape[1] == 3:
        valid = kps[:, 2] > 0.05
    else:
        valid = ~np.isnan(kps[:, 0]) & ~np.isnan(kps[:, 1])

    if not np.any(valid):
        return 0.0, 0.0, 1.0

    xs, ys = kps[valid, 0], kps[valid, 1]
    xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()
    side = max(xmax - xmin, ymax - ymin) or 1.0
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    return cx - side / 2.0, cy - side / 2.0, side


def _normalise(kps: np.ndarray, bbox: Tuple[float, float, float]) -> np.ndarray:
    """Apply one bbox (xmin, ymin, side) to a frame copy (Z unchanged)."""
    xmin, ymin, side = bbox
    out = kps.copy()
    if side < EPS:
        return out

    if out.shape[1] == 3:
        valid = out[:, 2] > 0.05
    else:
        valid = ~np.isnan(out[:, 0]) & ~np.isnan(out[:, 1])

    if not np.any(valid):
        return out

    out[valid, 0] = (out[valid, 0] - xmin) / side
    out[valid, 1] = (out[valid, 1] - ymin) / side
    return out


def _stable_side(adapt_sides: np.ndarray) -> float:
    """Derive the global stable side‑length S* from adaptive sides."""
    smooth = median_filter(adapt_sides, size=9)
    p5, p95 = np.percentile(smooth, [5, 95])
    good = (smooth >= p5) & (smooth <= p95)

    best_len = best_start = 0
    cur_start = None
    for i, ok in enumerate(good):
        if ok and cur_start is None:
            cur_start = i
        if (not ok) and cur_start is not None:
            length = i - cur_start
            if length > best_len:
                best_len, best_start = length, cur_start
            cur_start = None
    if cur_start is not None and len(good) - cur_start > best_len:
        best_start, best_len = cur_start, len(good) - cur_start

    return float(np.median(smooth[best_start:best_start + best_len])) or 1.0


# ─────────────────────── back‑compat shim ────────────────────────
def _reconcile_kwargs(kwargs: dict) -> dict:
    """
    Map legacy names → new ones so old driver.py keeps working.
    (out_path → out_stable)
    """
    if "out_path" in kwargs and "out_stable" not in kwargs:
        kwargs["out_stable"] = kwargs.pop("out_path")
    return kwargs


# ─────────────────────────── PUBLIC API ───────────────────────────
def pipeline(
    x3d: Union[str, Path, np.ndarray],
    *,
    out_stable: Optional[Union[str, Path]] = None,
    out_adaptive: Optional[Union[str, Path]] = None,
    **_legacy,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    x3d          : .npy file or ndarray (F,17,3)
    out_stable   : optional file for stable array
    out_adaptive : optional file for per‑frame adaptive array

    Returns
    -------
    (stable_array, adaptive_array)
    """
    _reconcile_kwargs(_legacy)          # make sure old kw names work

    arr_in = np.load(x3d) if not isinstance(x3d, np.ndarray) else x3d
    arr_in = arr_in.astype(np.float32, copy=True)
    F = arr_in.shape[0]

    adaptive_bb: List[Tuple[float, float, float]] = []
    adaptive = arr_in.copy()
    for f in range(F):
        bb = _adaptive_bbox(arr_in[f])
        adaptive_bb.append(bb)
        adaptive[f] = _normalise(arr_in[f], bb)

    S_star = _stable_side(np.array([b[2] for b in adaptive_bb], np.float32))

    stable = arr_in.copy()
    for f in range(F):
        xmin, ymin, _ = adaptive_bb[f]   # keep per‑frame centre, fix side
        stable[f] = _normalise(arr_in[f], (xmin, ymin, S_star))

    # optional saves
    def _np_save(path: Optional[Union[str, Path]], arr: np.ndarray, tag: str):
        if path is None:
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, arr)
        print(f"✅ {tag} array → {p}  shape={arr.shape}")

    _np_save(out_stable,   stable,   "stable")
    _np_save(out_adaptive, adaptive, "adaptive")

    return stable, adaptive


# ───────────────────────────── CLI ────────────────────────────────
if __name__ == "__main__":
    import argparse, matplotlib.pyplot as plt

    ap = argparse.ArgumentParser("dual MotionBERT normaliser")
    ap.add_argument("inp",  help=".npy with raw MotionBERT keypoints")
    ap.add_argument("--stable-out",   help="save stable array here")
    ap.add_argument("--adaptive-out", help="save adaptive array here")
    ap.add_argument("--plot", action="store_true",
                    help="show adaptive side‑length series for debugging")
    args = ap.parse_args()

    stab, adapt = pipeline(
        args.inp,
        out_stable=args.stable_out,
        out_adaptive=args.adaptive_out,
    )

    if args.plot:
        sides = [np.max(f[..., :2]) for f in adapt]
        plt.plot(sides)
        plt.title("Adaptive side length across frames")
        plt.xlabel("frame"); plt.ylabel("pixels")
        plt.show()
