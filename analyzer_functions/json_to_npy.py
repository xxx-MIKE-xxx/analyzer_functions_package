#!/usr/bin/env python3
"""
json_to_npy.py  –  scale & *flip-Y*           (2025-07-06 hot-fix)
---------------------------------------------------------------
Convert one AlphaPose result JSON to a NumPy ``.npy`` file with
coordinates normalised to the unit square **and the Y-axis kept
in image order**:

    (0, 0) = **top-left**      ·     (1, 1) = bottom-right

Why the change?
---------------
Down-stream visualisers (e.g. ``visualize_halpe26.py``) draw points by
``(int(x * W), int(y * H))`` where the canvas origin is the top-left
pixel.  The previous version inverted Y (bottom-left origin), which
made every skeleton appear upside-down.

*Left ↔ right* mapping is untouched, so the left side of the image is
still the subject’s right side (standard camera view).

Public API is **unchanged** – existing driver code continues to call

    from . import json_to_npy as jtn
    arr = jtn.pipeline(json_path, outdir)

CLI
~~~
    python json_to_npy.py <alphapose.json> <out.npy>
"""
from __future__ import annotations

import json, sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


# ───────────────────────── helpers ──────────────────────────────────────────
def _load_keypoints(js: Path) -> List[np.ndarray]:
    """Return list of (K,3) arrays – one per frame (missing → NaNs)."""
    with open(js, "r") as fh:
        data = json.load(fh)

    skels: List[np.ndarray] = []
    for frame in data:
        kps = frame["keypoints"] if frame else []
        if not kps:
            skels.append(np.full((26, 3), np.nan, dtype=np.float32))
        else:
            skels.append(np.asarray(kps, dtype=np.float32).reshape(-1, 3))
    return skels


def _frame_bounds(skels: List[np.ndarray]) -> Tuple[float, float]:
    xs = np.concatenate([s[:, 0] for s in skels])
    ys = np.concatenate([s[:, 1] for s in skels])
    return float(np.nanmax(xs)), float(np.nanmax(ys))


def _scale_unit_square(
    skels: List[np.ndarray],
    max_x: float,
    max_y: float,
) -> np.ndarray:
    """
    Scale (x,y) into [0,1]×[0,1] with origin at top-left.

    Note: **no Y inversion** – preserves image coordinate system.
    """
    fx, fy = 1.0 / max_x, 1.0 / max_y
    out: List[np.ndarray] = []

    for s in skels:
        sc = s.copy()
        valid = ~np.isnan(sc[:, 0]) & ~np.isnan(sc[:, 1])
        sc[valid, 0] *= fx        # left → right
        sc[valid, 1] *= fy        # top    stays   top
        out.append(sc)
    return np.stack(out, dtype=np.float32)          # (F,26,3)


# ───────────────────────── public API ───────────────────────────────────────
def pipeline(
    json_file: str | Path,
    npy_file_or_dir: str | Path,
    *,
    outfile: str | Path | None = None,
) -> np.ndarray:
    """
    Parameters
    ----------
    json_file        : AlphaPose JSON path
    npy_file_or_dir  : output directory **or** .npy filename
    outfile          : explicit override (optional)

    Returns
    -------
    ndarray  (F,26,3)  – unit-square coords, origin top-left
    """
    json_file = Path(json_file)

    # decide output path
    if outfile is not None:
        npy_path = Path(outfile)
    else:
        target = Path(npy_file_or_dir)
        npy_path = target if target.suffix == ".npy" else target / (json_file.stem + ".npy")

    # convert
    skels = _load_keypoints(json_file)
    max_x, max_y = _frame_bounds(skels)
    arr = _scale_unit_square(skels, max_x, max_y)

    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, arr)
    print(f"✅ wrote {npy_path}  shape={arr.shape}  "
          f"(max_x={max_x:.0f}, max_y={max_y:.0f})")
    return arr


# ───────────────────────── CLI entry-point ──────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage:\n  python json_to_npy.py <alphapose-json> <out-npy>")
    pipeline(sys.argv[1], sys.argv[2])
