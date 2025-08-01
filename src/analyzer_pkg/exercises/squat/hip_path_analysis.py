#!/usr/bin/env python3
"""
hip_path_analysis.py  🏃‍♀️
===========================
Compute lateral hip-midpoint deviation for each squat rep.

Output CSV columns:
  rep_id : int
  severity: list[str, str]        # [left, right]
  frames  : list[[int,int],[int,int]]  # [[left_peak,left_return],…]
  value   : list[float, float]    # [left_ratio, right_ratio]

Severity bands (peak deviation ratio to leg length):
  ≤ 0.04 → none
  ≤ 0.08 → mild
   > 0.08 → severe
"""

from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Union, Tuple, Mapping
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# HALPE-26 indices
L_HIP, R_HIP   = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANK,  R_ANK  = 15, 16

# Relative thresholds (unit coords)
END_FRAME_TOL   = 0.02   # hip_y within 2% of start ⇒ rep end
RETURN_TOL      = 0.02   # dev_ratio within 2% ⇒ “returned” to line

# Severity bands
NONE_TH    = 0.04
MILD_TH    = 0.08

# ---------------------------------------------------------------------
# Helpers


def plot_hip_path(
    keypoints: np.ndarray,
    reps: pd.DataFrame,
    out_path: str | Path,
    lengths_json: str | Path | dict = None,
):
    """
    Plot hip-midpoint deviation ratio (absolute) over frames, with severity thresholds and rep boundaries.
    """
    # Severity bands (must match core)
    NONE_TH = 0.04
    MILD_TH = 0.08

    # Load leg length (same logic as pipeline)
    if lengths_json is None:
        raise ValueError("`lengths_json` is required (leg-length reference).")
    if isinstance(lengths_json, dict):
        raw = lengths_json
        leg_len = (
            sum((_get_len(raw, i, j) or _get_len(raw, j, i) or 0.0)
                for i, j in [(L_HIP, L_KNEE), (R_HIP, R_KNEE), (L_KNEE, L_ANK), (R_KNEE, R_ANK)]) / 4.0 or 1.0
        )
    else:
        txt = str(lengths_json).strip()
        if txt.startswith("{"):
            raw = json.loads(txt)
            leg_len = (
                sum((_get_len(raw, i, j) or _get_len(raw, j, i) or 0.0)
                    for i, j in [(L_HIP, L_KNEE), (R_HIP, R_KNEE), (L_KNEE, L_ANK), (R_KNEE, R_ANK)]) / 4.0 or 1.0
            )
        else:
            leg_len = load_leg_len(Path(txt))

    # Prepare hip and ankle midpoints
    hip_x = (keypoints[:, L_HIP, 0] + keypoints[:, R_HIP, 0]) / 2.0
    hip_y = (keypoints[:, L_HIP, 1] + keypoints[:, R_HIP, 1]) / 2.0

    frame_nums = []
    dev_ratios = []

    # Rep boundaries
    if {"rep_start", "rep_end"}.issubset(reps.columns):
        reps = reps.rename(columns={"rep_start": "start", "rep_end": "end"})

    for _, row in reps.iterrows():
        start = int(row.start)
        end = int(row.end)
        rng = np.arange(start, end + 1)
        # ankle midpoint at start
        A = (
            (keypoints[start, L_ANK, 0] + keypoints[start, R_ANK, 0]) / 2.0,
            (keypoints[start, L_ANK, 1] + keypoints[start, R_ANK, 1]) / 2.0
        )
        B = (hip_x[start], hip_y[start])

        pts = np.column_stack((hip_x[rng], hip_y[rng]))
        dev = signed_dist(pts, A, B)         # signed
        dev_ratio = np.abs(dev / leg_len)    # absolute value for plotting

        frame_nums.extend(rng.tolist())
        dev_ratios.extend(dev_ratio.tolist())

    plt.figure(figsize=(12, 5))
    plt.plot(frame_nums, dev_ratios, label="|Hip-mid deviation / leg length|", lw=1.5)
    plt.axhline(NONE_TH, color="orange", ls="--", lw=1.5, label="Mild threshold (0.04)")
    plt.axhline(MILD_TH, color="red", ls="--", lw=1.5, label="Severe threshold (0.08)")

    # ---- Add vertical lines for start/end of each rep ----
    for idx, row in reps.iterrows():
        start, end = int(row['start']), int(row['end'])
        plt.axvline(start, color='green', ls=':', lw=1, alpha=0.7)
        plt.axvline(end, color='purple', ls=':', lw=1, alpha=0.7)
        # Optional: Label (slightly offset vertically for clarity)
        plt.text(start, plt.ylim()[1]*0.95, f"Rep {idx+1} start", color='green', rotation=90, va='top', ha='right', fontsize=8)
        plt.text(end, plt.ylim()[1]*0.95, f"Rep {idx+1} end", color='purple', rotation=90, va='top', ha='left', fontsize=8)

    plt.xlabel("Frame")
    plt.ylabel("Absolute deviation ratio")
    plt.title("Hip Path Deviation Ratio Over Frames")
    plt.legend()
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"📈  Saved hip-path plot → {out_path}")



def _get_len(pairs: Mapping[str, float], i: int, j: int) -> float | None:
    """Fetch raw-length under any '(i,j)' permutation."""
    for k in (f"({i},{j})", f"({i}, {j})", f"({j},{i})", f"({j}, {i})"):
        if k in pairs:
            return float(pairs[k])
    return None

def load_leg_len(path: Path) -> float:
    """
    Average four segment lengths (hip→knee + knee→ankle on both sides).
    JSON keys must be in the same unit scale as keypoints.
    """
    raw = json.load(open(path))
    def gl(a,b): return _get_len(raw, a, b) or 0.0
    avg = (gl(L_HIP,L_KNEE) + gl(R_HIP,R_KNEE) +
           gl(L_KNEE,L_ANK)  + gl(R_KNEE,R_ANK)) / 4.0
    return avg or 1.0

def signed_dist(
    pts: np.ndarray,   # shape (N,2)
    A: Tuple[float,float],
    B: Tuple[float,float],
) -> np.ndarray:
    """
    Vectorised signed distance from each point in `pts` to line AB.
    Uses right-hand normal. Returns array length N.
    """
    px, py = pts[:,0], pts[:,1]
    x1, y1 = A
    x2, y2 = B
    nx, ny = (y2 - y1), -(x2 - x1)
    denom = np.hypot(nx, ny) or 1.0
    return ((px - x1)*nx + (py - y1)*ny) / denom

def severity(ratio: float) -> str:
    """Map a deviation ratio to severity string."""
    if math.isnan(ratio):
        return "unknown"
    if ratio > MILD_TH:
        return "severe"
    if ratio > NONE_TH:
        return "mild"
    return "none"

def peak_and_return(
    dev: np.ndarray,
    side: str
) -> Tuple[int, int, float]:
    """
    For 'left', find the minimum (negative) peak idx; for 'right', the max.
    Then scan forward until `dev_ratio` crosses back within ±RETURN_TOL.
    Returns (peak_rel_idx, return_rel_idx, peak_value).
    """
    if side == "left":
        peak_rel = int(np.argmin(dev))
        peak_val = dev[peak_rel]
        # return when dev >= -RETURN_TOL
        mask = dev[peak_rel:] >= -RETURN_TOL
    else:
        peak_rel = int(np.argmax(dev))
        peak_val = dev[peak_rel]
        # return when dev <= +RETURN_TOL
        mask = dev[peak_rel:] <= +RETURN_TOL

    # locate first True in mask
    rels = np.nonzero(mask)[0]
    ret_rel = int(rels[0] + peak_rel) if rels.size else len(dev) - 1
    return peak_rel, ret_rel, peak_val

# ---------------------------------------------------------------------
# Core report generator

def generate_report(
    kps: np.ndarray,           # (F, 26, 3) unit-square coords
    reps: pd.DataFrame,
    leg_len: float,
    *,
    out_csv: str = "hip_path_report.csv"
) -> pd.DataFrame:
    # --- hip midpoints over time --------------------------------------
    hip_x = (kps[:, L_HIP, 0] + kps[:, R_HIP, 0]) / 2.0
    hip_y = (kps[:, L_HIP, 1] + kps[:, R_HIP, 1]) / 2.0

    records: List[Dict[str, Union[int, List, float]]] = []

    for _, row in reps.iterrows():
        rep_id  = int(row.rep_id)
        start   = int(row.start)
        end_nom = int(row.end)

        # --- find real end based on hip_y returning near start
        y0 = hip_y[start]
        real_end = end_nom
        for f in range(max(start, end_nom - 10), end_nom + 1):
            if abs(hip_y[f] - y0) < END_FRAME_TOL:
                real_end = f
                break

        # --- build dev_ratios over the rep ------------------------------
        rng = np.arange(start, real_end + 1)
        # ankle midpoint at start
        A = ((kps[start, L_ANK, 0] + kps[start, R_ANK, 0]) / 2.0,
             (kps[start, L_ANK, 1] + kps[start, R_ANK, 1]) / 2.0)
        # hip midpoint at start
        B = (hip_x[start], hip_y[start])

        pts = np.column_stack((hip_x[rng], hip_y[rng]))
        dev = signed_dist(pts, A, B)            # in unit coords
        dev_ratio = dev / leg_len               # normalise by leg length

        # --- find peaks and returns -------------------------------------
        lp, lr, lval = peak_and_return(dev_ratio, "left")
        rp, rr, rval = peak_and_return(dev_ratio, "right")

         # Use MILD_TH as the "problem" threshold (you can adjust)
        l_start, l_end = find_problem_segment(dev_ratio, "left", MILD_TH)
        r_start, r_end = find_problem_segment(dev_ratio, "right", MILD_TH)

        records.append({
            "rep_id": rep_id,
            "severity": [severity(abs(lval)), severity(abs(rval))],
            "frames": [
                {"start": int(rng[l_start]) if l_start != -1 else None,
                 "peak": int(rng[lp]),
                 "end": int(rng[l_end]) if l_end != -1 else None},
                {"start": int(rng[r_start]) if r_start != -1 else None,
                 "peak": int(rng[rp]),
                 "end": int(rng[r_end]) if r_end != -1 else None},
            ],
            "value": [abs(lval), abs(rval)],
        })

    df = pd.DataFrame(records)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✅ hip-path report → {os.path.abspath(out_csv)}")
    return df

# ---------------------------------------------------------------------
# In-memory pipeline

def pipeline(
    keypoints: str | Path | np.ndarray,
    reps: str | Path | pd.DataFrame,
    *,
    lengths_json: str | Path | Dict[str, float] | None = None,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Hip-path analysis – return DataFrame (and optionally write CSV).

    Parameters
    ----------
    keypoints    : .npy path or ndarray (F,26,3) – unit-square coords
    reps         : CSV path or DataFrame with rep_id, start/end or rep_start/rep_end
    lengths_json : dict or raw JSON string or path – average leg lengths in unit scale
    output_csv   : optional CSV path
    """
    if lengths_json is None:
        raise ValueError("`lengths_json` is required (leg-length reference).")

    # 1) load keypoints
    kps = np.load(keypoints) if isinstance(keypoints, (str, Path)) else keypoints

    # 2) load reps
    if isinstance(reps, pd.DataFrame):
        reps_df = reps.copy()
    else:
        reps_df = pd.read_csv(reps)
    # rename rep_start/end if present
    if {"rep_start", "rep_end"}.issubset(reps_df.columns):
        reps_df = reps_df.rename(columns={"rep_start": "start", "rep_end": "end"})

    # 3) load lengths
    if isinstance(lengths_json, dict):
        raw = lengths_json
    else:
        txt = str(lengths_json).strip()
        raw = json.loads(txt) if txt.startswith("{") else json.load(open(txt))
    leg_len = load_leg_len(Path(raw) if isinstance(raw, str) else Path("<in-memory>")) \
              if not isinstance(raw, dict) else (
                  sum((_get_len(raw, i, j) or _get_len(raw, j, i) or 0.0)
                      for i,j in [(L_HIP,L_KNEE),(R_HIP,R_KNEE),
                                  (L_KNEE,L_ANK),(R_KNEE,R_ANK)]) / 4.0 or 1.0
              )

    # 4) analyse & optionally write
    outpath = output_csv or Path(os.devnull)
    df = generate_report(kps, reps_df, leg_len, out_csv=str(outpath))

    if output_csv is None:
        df.attrs.pop("filepath_or_buffer", None)
    return df




def find_problem_segment(dev: np.ndarray, side: str, th: float) -> Tuple[int, int]:
    """
    Find first (start) and last (end) frame where abs(dev) exceeds threshold.
    Returns relative indices (for slicing rep range).
    """
    if side == "left":
        mask = dev < -th
    else:
        mask = dev > th
    indices = np.where(mask)[0]
    if indices.size == 0:
        return -1, -1
    return int(indices[0]), int(indices[-1])

# ---------------------------------------------------------------------
# CLI entry-point

def main():
    p = argparse.ArgumentParser("Hip-path deviation report")
    p.add_argument("--keypoints", type=Path, default="imputed_ma.npy")
    p.add_argument("--reps",      type=Path, default="repetition_data.csv")
    p.add_argument("--lengths",   type=Path, required=True,
                   help="JSON with (i,j) leg lengths in unit scale")
    p.add_argument("--output",    type=Path, default="hip_path_report.csv")
    args = p.parse_args()

    kps  = np.load(args.keypoints)
    reps = pd.read_csv(args.reps)
    leg  = load_leg_len(args.lengths)
    generate_report(kps, reps, leg, out_csv=str(args.output))


if __name__ == "__main__":
    main()
