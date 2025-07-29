#!/usr/bin/env python3
"""
compute_reference_lengths.py
============================

Derive per-bone reference lengths from a *calibration pose* (or any clean
sequence of frames) and optionally return them as JSON.

This module provides:
  - `pipeline_reference_lengths`: programmatic API that reads inputs (file paths or arrays),
    computes 3D and 2D reference lengths, and returns a dict (serializable to JSON).
  - CLI (`main`) that wraps around the same functionality, writing JSON files.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# 3-D SKEL: Human3.6M / COCO-17  (UNCHANGED)
# ---------------------------------------------------------------------------
CONNECTIONS_3D: List[Tuple[int, int]] = [
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (8, 14),
    (11, 12), (12, 13), (14, 15), (15, 16),
    (0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6),
]

SYMMETRIC_PAIRS_3D: List[Tuple[Tuple[int, int], Tuple[int, int]]] = [
    ((0, 1), (0, 4)), ((1, 2), (4, 5)), ((2, 3), (5, 6)),
    ((8, 11), (8, 14)), ((11, 12), (14, 15)), ((12, 13), (15, 16)),
]

# ---------------------------------------------------------------------------
# 2-D SKEL: HALPE-26 (body subset only – no face/hands)
# ---------------------------------------------------------------------------
CONNECTIONS_2D: List[Tuple[int, int]] = [
    (0, 17), (17, 18), (18, 19), (18, 5), (18, 6), (19, 11), (19, 12),
    (11, 5), (12, 6), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (12, 14), (13, 15), (14, 16),
    (15, 20), (15, 22), (15, 24), (16, 21), (16, 23), (16, 25),
]

SYMMETRIC_PAIRS_2D: List[Tuple[Tuple[int, int], Tuple[int, int]]] = [
    ((19, 11), (19, 12)), ((11, 13), (12, 14)), ((13, 15), (14, 16)),
    ((18, 5),  (18, 6)), ((5, 7),   (6, 8)), ((7, 9),   (8, 10)),
    ((15, 20), (16, 21)), ((15, 22), (16, 23)), ((15, 24), (16, 25)),
    ((11, 5),  (12, 6)),
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def parse_frame_spec(spec: str, n_frames: int) -> List[int]:
    if not spec:
        return list(range(n_frames))
    indices: List[int] = []
    for token in spec.split(','):
        token = token.strip()
        if re.fullmatch(r"\d+", token):
            indices.append(int(token))
        elif re.fullmatch(r"\d*\-\d*", token):
            start_str, end_str = token.split('-')
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else (n_frames - 1)
            indices.extend(range(start, end + 1))
        else:
            raise ValueError(f"Cannot parse frame spec: '{token}'")
    indices = sorted({i for i in indices if 0 <= i < n_frames})
    if not indices:
        raise ValueError("No valid frame indices – check your spec.")
    return indices


def compute_lengths(
    skeleton: np.ndarray,
    frames: Iterable[int],
    connections: List[Tuple[int, int]],
) -> Dict[str, float]:
    dists: Dict[Tuple[int, int], List[float]] = {c: [] for c in connections}
    for f in frames:
        pts = skeleton[f]
        for start, end in connections:
            d = np.linalg.norm(pts[end] - pts[start])
            dists[(start, end)].append(float(d))
    # median per-bone
    return {f"({i},{j})": float(np.median(vals)) for (i, j), vals in dists.items()}


def enforce_symmetry(
    lengths: Dict[str, float],
    symmetric_pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]]
) -> None:
    for (a, b) in symmetric_pairs:
        key_a, key_b = f"({a[0]},{a[1]})", f"({b[0]},{b[1]})"
        if key_a in lengths and key_b in lengths:
            avg = (lengths[key_a] + lengths[key_b]) / 2.0
            lengths[key_a] = lengths[key_b] = avg

# ---------------------------------------------------------------------------
# Pipeline function
# ---------------------------------------------------------------------------

def pipeline_reference_lengths(
    input_3d: Union[str, np.ndarray],
    input_2d: Optional[Union[str, np.ndarray]] = None,
    frames: Optional[str] = None
) -> Tuple[Dict[str, float], Optional[Dict[str, float]]]:
    """
    Compute reference lengths for 3D and optional 2D skeleton.
    Returns a tuple: (lengths3d, lengths2d_or_None).
    """
    # load 3D
    sk3 = np.load(input_3d) if isinstance(input_3d, str) else input_3d
    if sk3.ndim != 3 or sk3.shape[1] != 17 or sk3.shape[2] != 3:
        raise ValueError("3D input must be (F,17,3)")
    frame_list = parse_frame_spec(frames or "", sk3.shape[0])
    lengths3d = compute_lengths(sk3, frame_list, CONNECTIONS_3D)
    enforce_symmetry(lengths3d, SYMMETRIC_PAIRS_3D)

    lengths2d = None
    # optional 2D
    if input_2d is not None:
        sk2_raw = np.load(input_2d) if isinstance(input_2d, str) else input_2d
        if sk2_raw.ndim != 3 or sk2_raw.shape[1] != 26:
            raise ValueError("2D input must be (F,26,2) or (F,26,3)")
        sk2 = sk2_raw[..., :2]
        frames2 = [f for f in frame_list if f < sk2.shape[0]]
        lengths2d = compute_lengths(sk2, frames2, CONNECTIONS_2D)
        enforce_symmetry(lengths2d, SYMMETRIC_PAIRS_2D)

    return lengths3d, lengths2d

# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Derive per-bone reference lengths and optionally output JSON"
    )
    ap.add_argument("input_3d", help="3D .npy sequence (F×17×3) or array")
    ap.add_argument("output_3d", help="Output JSON for 3D reference lengths")
    ap.add_argument("--input-2d", default=None,
                    help="Optional 2D (F×26×2/3) .npy sequence or array")
    ap.add_argument("--output-2d", default=None,
                    help="Output JSON for 2D reference lengths")
    ap.add_argument("--frames", default=None,
                    help="Frame spec (e.g. '0-30,50-') to subset frames")
    args = ap.parse_args()

    lengths3d, lengths2d = pipeline_reference_lengths(
        input_3d=args.input_3d,
        input_2d=args.input_2d,
        frames=args.frames
    )

    # Write out 3D JSON
    Path(args.output_3d).write_text(json.dumps(lengths3d, indent=4))
    print(f"Wrote 3D lengths → {args.output_3d}")

    # Write out 2D JSON if available(f"Wrote 3D lengths → {args.output_3d}")

    # Write out 2D JSON if available
    if "lengths2d" in data:
        out2 = args.output_2d or Path(args.output_3d).with_stem(
            Path(args.output_3d).stem + "_2d"
        ).as_posix()
        Path(out2).write_text(json.dumps(data["lengths2d"], indent=4))
        print(f"Wrote 2D lengths → {out2}")

if __name__ == "__main__":
    main()
