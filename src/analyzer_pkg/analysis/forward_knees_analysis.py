#!/usr/bin/env python3
"""
forward_knees_analysis.py  🦵
============================
Estimate forward-knee angle per squat repetition (HALPE-26 keypoints).

► **NEW (2025-07-06): `lengths_json` is now mandatory.**
  Calling `build_report` or `pipeline` without it raises `ValueError`.

Public API
----------
build_report(keypoints, reps_csv_or_df, out_csv, lengths_json=...) -> pd.DataFrame
pipeline(...)                                   # lightweight in-memory helper
"""
from __future__ import annotations
import math, json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

CONF_TH = 0.0                     # keypoint confidence threshold
SEVERE_TH, MILD_TH = 20.0, 10.0   # angle thresholds (deg)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _load_lengths(path: Path) -> dict[tuple[int, int], float]:
    """Load shin lengths JSON → dict[(idx,idx)] → float."""
    with open(path) as f:
        raw = json.load(f)
    return {tuple(map(int, k.strip("()").split(","))): float(v)
            for k, v in raw.items()}


def _shin_len(lengths: Mapping[tuple[int, int], float],
              knee: int, ankle: int) -> float:
    """Fetch shin length (0 if missing/invalid)."""
    val = lengths.get((knee, ankle)) or lengths.get((ankle, knee), 0.0)
    return float(val) if val and val > 0 else 0.0


def _knee_forward_angle(kp: np.ndarray,
                        knee_idx: int,
                        ankle_idx: int,
                        shin_len: float) -> tuple[float, bool]:
    """
    3-D forward-knee angle from 2-D projection.

    Returns (theta_deg, valid_flag).
    """
    x1, y1, c1 = kp[ankle_idx]
    x2, y2, c2 = kp[knee_idx]

    if min(c1, c2) < CONF_TH or np.isnan([x1, y1, x2, y2]).any():
        return 0.0, False

    planar = math.hypot(x2 - x1, y2 - y1)
    shin   = shin_len

    if planar <= 0 or planar >= shin:
        return 0.0, False

    z = math.sqrt(shin ** 2 - planar ** 2)
    theta = math.degrees(math.atan2(z, planar))
    return theta, True


# --------------------------------------------------------------------------
# Main disk-writing function
# --------------------------------------------------------------------------
def build_report(
    keypoints: np.ndarray,
    reps_csv_or_df: str | Path | pd.DataFrame,
    out_csv: str | Path,
    lengths_json: str | Path | Mapping,
) -> pd.DataFrame:
    """
    Write the forward-knee report and return it.

    `lengths_json` **must be provided**; otherwise a `ValueError` is raised.
    """
    if lengths_json is None:
        raise ValueError(
            "The `lengths_json` argument is required (shin-length reference)."
        )

    # ---------------- reps table ----------------
    reps_df = (
        reps_csv_or_df.copy()
        if isinstance(reps_csv_or_df, pd.DataFrame)
        else pd.read_csv(reps_csv_or_df)
    )
    if {"rep_start", "rep_end"}.issubset(reps_df.columns):
        reps_df = reps_df.rename(columns={"rep_start": "start",
                                          "rep_end":   "end"})
    required = {"rep_id", "start", "end"}
    if not required.issubset(reps_df.columns):
        raise ValueError(f"Reps table must contain {sorted(required)}")

    # ---------------- shin lengths -------------
    if isinstance(lengths_json, Mapping):
        raw = lengths_json
    else:  # path or raw JSON string
        txt = str(lengths_json).strip()
        raw = _load_lengths(Path(txt)) if not (txt.startswith("{") and txt.endswith("}")) \
              else json.loads(txt)

    lengths: dict[tuple[int, int], float] = {
        (int(a), int(b)): float(v)
        for k, v in raw.items()
        for a, b in [k if isinstance(k, (list, tuple))
                     else map(int, str(k).strip("()").split(","))]
    }

    # ---------------- analyse reps -------------
    rows = []
    for _, rep in reps_df.iterrows():
        rep_id, start, end = int(rep.rep_id), int(rep.start), int(rep.end)
        thetas: list[float] = []

        for f in range(start, end + 1):
            kp = keypoints[f]
            θL, okL = _knee_forward_angle(kp, L_KNEE, L_ANKLE,
                                          _shin_len(lengths, L_KNEE, L_ANKLE))
            θR, okR = _knee_forward_angle(kp, R_KNEE, R_ANKLE,
                                          _shin_len(lengths, R_KNEE, R_ANKLE))
            if okL or okR:
                thetas.append(max(θL, θR))

        maxθ = max(thetas) if thetas else 0.0
        avgθ = float(np.mean(thetas)) if thetas else 0.0
        severity = ("none", "mild", "severe")[2 if maxθ >= SEVERE_TH
                                              else 1 if maxθ >= MILD_TH else 0]

        rows.append({
            "rep_id": rep_id,
            "avg_forward_angle_deg": round(avgθ, 2),
            "max_forward_angle_deg": round(maxθ, 2),
            "severity": severity,
        })

    df = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✅ forward-knee report → {out_csv}")
    return df


# --------------------------------------------------------------------------
# Lightweight in-memory helper
# --------------------------------------------------------------------------
def pipeline(
    keypoints: str | Path | np.ndarray,
    reps: str | Path | pd.DataFrame,
    *,
    lengths_json: str | Path | Mapping,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Forward-knee analysis that returns a DataFrame.

    `lengths_json` **must be supplied**; omission raises `ValueError`.
    """
    if lengths_json is None:
        raise ValueError(
            "The `lengths_json` argument is required; shin lengths are essential."
        )

    kps = np.load(keypoints) if isinstance(keypoints, (str, Path)) else keypoints
    reps_df = reps.copy() if isinstance(reps, pd.DataFrame) else pd.read_csv(reps)
    if {"rep_start", "rep_end"}.issubset(reps_df.columns):
        reps_df = reps_df.rename(columns={"rep_start": "start",
                                          "rep_end":   "end"})

    df = build_report(
        kps, reps_df,
        out_csv=output_csv or Path("/dev/null"),
        lengths_json=lengths_json,
    )

    if output_csv is None:                     # silence dummy path print
        df.attrs.pop("filepath_or_buffer", None)
    return df


# --------------------------------------------------------------------------
# CLI entry-point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        "Forward-knee angle reporter (unit-square coords)"
    )
    ap.add_argument("--keypoints", required=True,
                    help="NumPy .npy with (F,26,3) keypoints in unit coords")
    ap.add_argument("--reps",      required=True,
                    help="CSV/feather with rep_id,start,end (or rep_* variants)")
    ap.add_argument("--lengths",   required=True,
                    help="Shin-lengths JSON file or raw JSON string")
    ap.add_argument("--out",       default="forward_knee_angle_report.csv",
                    help="Output CSV path")
    args = ap.parse_args()

    kps = np.load(args.keypoints)
    build_report(kps, args.reps, args.out, args.lengths)
