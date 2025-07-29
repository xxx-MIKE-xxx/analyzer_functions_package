#!/usr/bin/env python3
"""
forward_lean_analysis.py  📏
===========================

Compute *deviation from vertical* for each squat repetition.

Definitions
-----------
Hip              = (hx,  hy,    0)
Shoulder         = (sx,  sy,   ±z)      # depth reconstructed from torso length
Vertical unit v  = (0,  sign,  0)       # sign = +1 if y grows upward, –1 if y grows downward
Torso vector t   = Shoulder − Hip

Angle = arccos( ⟨t, v⟩ / ||t|| ).
• Small angle ⇒ upright torso
• Large angle ⇒ forward-lean

Output
------
forward_lean_report.csv with columns
    rep_id, severity, frame, angle_deg

Changes (2025-07-06)
--------------------
* Works with **unit-square coordinates** by scaling xyz ×1000 internally.
* Auto-detects y-axis orientation (image vs Cartesian).
* `lengths_json` is **mandatory**; ValueError if omitted.
"""
from __future__ import annotations
import argparse, json, math, os
from pathlib import Path
from typing import Dict, List, Mapping, Union

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ constants & indices
# HALPE-26 indices
L_HIP, R_HIP   = 11, 12
L_KNEE, R_KNEE = 13, 14
L_SHO, R_SHO   = 5,  6

CONF_TH_DEF = 0.20       # default confidence
SCALE       = 1000.0     # up-scale to kill 0-1 rounding errors
MILD_TH     = 45.0       # >45° ⇒ mild lean
SEV_TH      = 55.0       # >55° ⇒ severe lean

# ------------------------------------------------------------------ helpers
def _get_len(pairs: Mapping[str, float], i: int, j: int) -> float | None:
    """Fetch length under any key permutation "(i,j)" etc."""
    for k in (f"({i},{j})", f"({i}, {j})", f"({j},{i})", f"({j}, {i})"):
        if k in pairs:
            return float(pairs[k])
    return None


def _normalize_lengths(raw: Mapping[str, float]) -> Dict[str, float]:
    """Return dict with torsoL, torsoR … in unit-square scale."""
    if any(k.startswith("(") for k in raw):       # "(i,j)" style keys
        return {
            "torsoL": _get_len(raw, L_HIP, L_SHO) or 0.0,
            "torsoR": _get_len(raw, R_HIP, R_SHO) or 0.0,
            # thighs unused but kept for completeness
            "thighL": _get_len(raw, L_HIP, L_KNEE) or 0.0,
            "thighR": _get_len(raw, R_HIP, R_KNEE) or 0.0,
        }
    # already in flat form
    return {k: float(v) for k, v in raw.items()}


def _rel_z(x1: float, y1: float,
           x2: float, y2: float,
           L: float) -> float:
    """Depth from 2-D projection (positive)."""
    d2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
    if d2 < L * L:
        return math.sqrt(L * L - d2)
    return math.sqrt(d2 - L * L)


def _orientation_sign(kps: np.ndarray, mid: int) -> int:
    """
    Detect whether y grows upward (+1) or downward (−1).
    Hip descends at 'mid' in image coords.
    """
    hip_y0 = kps[max(0, mid - 3), L_HIP, 1]
    hip_y1 = kps[mid, L_HIP, 1]
    return -1 if hip_y1 > hip_y0 else +1


def _vert_torso_angle(
    kp: np.ndarray,
    hip_i: int,
    sho_i: int,
    torso_len: float,
    sign: int,
) -> float:
    """Angle (deg) between torso vector and vertical axis."""
    hx, hy, hc = kp[hip_i]
    sx, sy, sc = kp[sho_i]
    if min(hc, sc) < CONF_TH_DEF or math.isnan(hx+hy+sx+sy) or torso_len == 0:
        return float("nan")

    # scale coords and length to avoid precision issues
    hx, hy, sx, sy, L = (hx * SCALE, hy * SCALE, sx * SCALE, sy * SCALE, torso_len * SCALE)
    dz = _rel_z(hx, hy, sx, sy, L)

    v_t = np.array([sx - hx, sy - hy, dz])
    norm = np.linalg.norm(v_t)
    if norm == 0.0:
        return float("nan")

    v_v = np.array([0.0, sign, 0.0])     # vertical reference
    cosang = np.clip(np.dot(v_t, v_v) / norm, -1.0, 1.0)
    return math.degrees(math.acos(cosang))


def _frame_deviation(kp: np.ndarray,
                     lens: Mapping[str, float],
                     sign: int) -> float:
    """Return the larger deviation angle of the two sides for one frame."""
    left  = _vert_torso_angle(kp, L_HIP, R_SHO, lens["torsoL"], sign)
    right = _vert_torso_angle(kp, R_HIP, L_SHO, lens["torsoR"], sign)
    vals = [a for a in (left, right) if not math.isnan(a)]
    return max(vals) if vals else float("nan")


def _severity(a: float) -> str:
    if math.isnan(a):
        return "unknown"
    if a > SEV_TH:
        return "severe"
    if a > MILD_TH:
        return "mild"
    return "none"


# ------------------------------------------------------------------ core report generator
def generate_report(
    kps: np.ndarray,
    reps: pd.DataFrame,
    lens: Mapping[str, float],
    *,
    window: int = 5,
    out_csv: str = "forward_lean_report.csv",
) -> pd.DataFrame:
    recs: List[Dict[str, Union[int, float, str]]] = []
    F = kps.shape[0]

    for _, row in reps.iterrows():
        rep_id = int(row.rep_id)
        mid    = int(row.mid)
        sign   = _orientation_sign(kps, mid)

        frames = range(max(0, mid - window), min(F, mid + window + 1))
        vals = [
            _frame_deviation(kps[f], lens, sign)
            for f in frames
            if not math.isnan(_frame_deviation(kps[f], lens, sign))
        ]
        val = max(vals) if vals else float("nan")

        recs.append({
            "rep_id":   rep_id,
            "severity": _severity(val),
            "frame":    mid,
            "angle_deg": round(val, 2) if not math.isnan(val) else float("nan"),
        })

    df = pd.DataFrame(recs)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✅ forward-lean report → {os.path.abspath(out_csv)}")
    return df


# ------------------------------------------------------------------ CLI (lengths required)
def main() -> None:                # pragma: no cover
    ap = argparse.ArgumentParser("Deviation from vertical per squat rep")
    ap.add_argument("--keypoints", type=Path, required=True)
    ap.add_argument("--reps",      type=Path, required=True)
    ap.add_argument("--lengths",   type=Path, required=True,
                    help="Reference lengths JSON in unit-square scale")
    ap.add_argument("--window",    type=int,  default=5)
    ap.add_argument("--output",    type=Path, default="forward_lean_report.csv")
    args = ap.parse_args()

    kps  = np.load(args.keypoints)
    reps = pd.read_csv(args.reps)
    lens = _normalize_lengths(json.load(open(args.lengths)))

    generate_report(kps, reps, lens,
                    window=args.window,
                    out_csv=str(args.output))


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------ pipeline helper
def pipeline(
    keypoints: str | Path | np.ndarray,
    reps: str | Path | pd.DataFrame,
    *,
    lengths_json: str | Path | Mapping,
    window: int = 5,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    In-memory forward-lean analysis.

    Parameters
    ----------
    keypoints     : .npy path or ndarray (F,26,3) – unit-square coords
    reps          : CSV path or DataFrame with ['rep_id','mid'] or ['rep_mid']
    lengths_json  : required – torso lengths in the same unit scale
    window        : ±frames around each mid to search
    output_csv    : optional CSV path
    """
    # --- keypoints --------------------------------------------------------
    kps = np.load(keypoints) if isinstance(keypoints, (str, Path)) else keypoints

    # --- reps -------------------------------------------------------------
    reps_df = reps.copy() if isinstance(reps, pd.DataFrame) else pd.read_csv(reps)
    if "rep_mid" in reps_df.columns and "mid" not in reps_df.columns:
        reps_df = reps_df.rename(columns={"rep_mid": "mid"})

    # --- lengths ----------------------------------------------------------
    if lengths_json is None:
        raise ValueError("`lengths_json` is required (torso reference).")

    if isinstance(lengths_json, Mapping):
        raw = lengths_json
    else:
        txt = str(lengths_json).strip()
        raw = json.loads(txt) if txt.startswith("{") else json.load(open(txt))

    lens = _normalize_lengths(raw)

    # --- compute ----------------------------------------------------------
    tmp_out = output_csv or Path(os.devnull)
    df = generate_report(kps, reps_df, lens, window=window, out_csv=tmp_out)

    if output_csv is None:          # silence dummy file path in attrs
        df.attrs.pop("filepath_or_buffer", None)
    return df
