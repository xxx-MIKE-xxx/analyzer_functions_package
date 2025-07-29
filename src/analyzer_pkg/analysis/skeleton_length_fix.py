#!/usr/bin/env python3
"""
skeleton_length_fix.py  – rigid top‑down bone‑length correction
===============================================================

This supersedes *skeleton_lengthen_fix.py*. It now **expands *or* contracts**
any bone whose length differs from the reference by a small
... [existing doc omitted for brevity] ...
"""

from __future__ import annotations

import json
import sys
from typing import Dict, List, Tuple

import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Connection = Tuple[int, int]
Hierarchy = Dict[int, List[int]]
RefLengths = Dict[Connection, float]

# H36M hierarchy with 17 joints (root 0)
HIERARCHY: Hierarchy = {
    0: [1, 4, 7],
    1: [2], 2: [3],
    4: [5], 5: [6],
    7: [8],
    8: [9, 11, 14],
    9: [10],
    11: [12], 12: [13],
    14: [15], 15: [16],
}

# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------
def correct_skeleton_rigid(
    skeleton: np.ndarray,
    reference_lengths: RefLengths,
    hierarchy: Hierarchy,
    tolerance: float = 1e-3,
) -> np.ndarray:
    """Return a new sequence in which every bone length matches the supplied
    reference within *tolerance*."""
    corrected = skeleton.copy()
    for f in range(corrected.shape[0]):
        _dfs_correct(
            corrected[f],
            parent=0,
            hierarchy=hierarchy,
            reference_lengths=reference_lengths,
            tolerance=tolerance,
        )
    return corrected


def _dfs_correct(
    frame: np.ndarray,
    parent: int,
    hierarchy: Hierarchy,
    reference_lengths: RefLengths,
    tolerance: float,
):
    for child in hierarchy.get(parent, []):
        ref = reference_lengths.get((parent, child)) or reference_lengths.get((child, parent))
        if ref is None:
            _dfs_correct(frame, child, hierarchy, reference_lengths, tolerance)
            continue

        vec = frame[child] - frame[parent]
        curr_len = float(np.linalg.norm(vec))
        if curr_len < 1e-9:
            _dfs_correct(frame, child, hierarchy, reference_lengths, tolerance)
            continue

        if abs(curr_len - ref) > tolerance:
            unit = vec / curr_len
            new_pos = frame[parent] + unit * ref
            shift = new_pos - frame[child]
            frame[child] = new_pos
            _propagate(frame, child, hierarchy, shift)

        _dfs_correct(frame, child, hierarchy, reference_lengths, tolerance)


def _propagate(frame: np.ndarray, joint: int, hierarchy: Hierarchy, shift: np.ndarray):
    for ch in hierarchy.get(joint, []):
        frame[ch] += shift
        _propagate(frame, ch, hierarchy, shift)

# ---------------------------------------------------------------------------
# Pipeline function
# ---------------------------------------------------------------------------
def pipeline(
    skeleton: str | Path | np.ndarray,
    lengths: str | Path | RefLengths,
    hierarchy: Hierarchy | None = None,
    tolerance: float = 1e-3,
) -> np.ndarray:
    """
    Load inputs (if needed), apply rigid skeleton length correction,
    and return the corrected skeleton sequence without saving to disk.

    Args:
        skeleton: path to a (F, J, D) .npy file or an ndarray.
        lengths: path to a JSON lengths file or a dict of reference lengths.
        hierarchy: bone hierarchy (defaults to HIERARCHY above).
        tolerance: tolerance for length matching.

    Returns:
        A numpy array of the corrected skeletons, same shape as input.
    """
    # 1) load skeleton array
    if not isinstance(skeleton, np.ndarray):
        skeleton = np.load(skeleton)

    # 2) load or accept reference lengths
    if isinstance(lengths, (str, Path)):
        with open(lengths, 'r') as f:
            data = json.load(f)
        ref_lengths = {
            tuple(map(int, k.strip('() ').split(','))): v
            for k, v in data.items()
        }
    else:
        ref_lengths = lengths

    # 3) set default hierarchy
    hier = hierarchy if hierarchy is not None else HIERARCHY

    # 4) apply correction and return
    return correct_skeleton_rigid(skeleton, ref_lengths, hier, tolerance)

# ---------------------------------------------------------------------------
# Minimal CLI (unchanged)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python skeleton_length_fix.py <input.npy> <lengths.json> <output.npy>")
        sys.exit(1)

    npy_in, json_lengths, npy_out = sys.argv[1:4]

    seq = np.load(npy_in)
    with open(json_lengths) as f:
        ref_lengths: RefLengths = {
            tuple(map(int, k.strip("() ").split("",""))): v
            for k, v in json.load(f).items()
        }

    fixed = pipeline(seq, ref_lengths)
    np.save(npy_out, fixed)
    print(f"Corrected skeleton saved to {npy_out}")
