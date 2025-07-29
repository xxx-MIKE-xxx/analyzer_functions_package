#!/usr/bin/env python3
"""
foot_orientation_analysis.py
---------------------------
Analyse foot pointing angles for every squat repetition using HALPE‑26
keypoints.

For each rep we compute, for **left** and **right** side:
* ankle → mid‑toes vector in the ground (X‑Y) plane
* angle w.r.t. global +X axis (right).  Positive = toes outward, negative = inward.
* Adjust angle by an optional **heading** (athlete orientation) so that
  0 ° means toes straight forward relative to their body.

Report columns
~~~~~~~~~~~~~~
rep_id  int
severity list[3]  [left_sev, right_sev, symmetry_bool]
frames   list[ ]   empty placeholder (per requirement)
value    list[2]  [angle_left_deg, angle_right_deg]

Severity encoding (numeric, signed):
    0   = none (|angle| ≤ sym_tol)
    +1  = mild outward   ( out_mild  ≤ angle <  out_severe )
    +2  = severe outward ( angle ≥ out_severe )
    −1  = mild inward    ( in_mild_neg ≥ angle >  in_severe_neg )
    −2  = severe inward  ( angle ≤ in_severe_neg )
Default thresholds:
    out_mild   = +35°,  out_severe = +45°
    in_mild    = −10°,  in_severe  = −20°
    symmetry_tolerance = 5°
    heading = 0° (front‑facing)
All thresholds can be modified via CLI args.
"""
from __future__ import annotations
import argparse, json, math, os
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ HALPE-26 indices
L_ANK, R_ANK   = 15, 16
L_BIG, R_BIG   = 20, 21
L_SML, R_SML   = 22, 23

# ------------------------------------------------------------------ angle helper

def foot_angle(ank: np.ndarray, big: np.ndarray, sml: np.ndarray, heading_deg: float) -> float:
    """Return foot pointing angle (deg) corrected by heading."""
    # mid‑toes point
    mid = (big + sml) / 2.0
    vec = mid - ank
    ang = math.degrees(math.atan2(vec[1], vec[0]))  # +(x right, y down)
    # convert to body‑centric by subtracting heading
    return ang - heading_deg

# ------------------------------------------------------------------ severity helper

def classify(angle: float,
             out_mild: float, out_sev: float,
             in_mild: float,  in_sev: float) -> int:
    """
    Returns:
      +2 = severe outward (angle ≥ out_sev)
      +1 = mild outward   (angle ≥ out_mild)
      -1 = mild inward    (angle ≤ in_mild)
      -2 = severe inward  (angle ≤ in_sev)
       0 = no issue       (else)
    """
    if math.isnan(angle):
        return 0
    # outward: positive
    if angle >= out_sev:
        return 2
    if angle >= out_mild:
        return 1
    # inward: negative
    if angle <= in_mild:
        return -1
    if angle <= in_sev:
        return -2
    return 0
    if angle >= out_sev:
        return 2
    if angle >= out_mild:
        return 1
    if angle <= in_sev:
        return -2
    if angle <= in_mild:
        return -1
    return 0

# ------------------------------------------------------------------ main routine

def generate_report(kps: np.ndarray,
                    reps: pd.DataFrame,
                    *,
                    heading: float = 0.0,
                    out_mild: float = 35.0,
                    out_sev:  float = 45.0,
                    in_mild:  float = -10.0,
                    in_sev:   float = -20.0,
                    sym_tol:  float = 5.0,
                    out_csv: str = "foot_orientation_report.csv") -> pd.DataFrame:

    records: List[Dict[str, Union[int, List]]] = []

    for _, row in reps.iterrows():
        start, end = int(row.start), int(row.end)
        # use full rep window → mean angle over frames
        left_angles  = []
        right_angles = []
        for f in range(start, end+1):
            if np.isnan(kps[f,[L_ANK,L_BIG,L_SML,R_ANK,R_BIG,R_SML],:2]).any():
                continue  # skip frame with NaNs
            la = foot_angle(kps[f,L_ANK,:2], kps[f,L_BIG,:2], kps[f,L_SML,:2], heading)
            ra = foot_angle(kps[f,R_ANK,:2], kps[f,R_BIG,:2], kps[f,R_SML,:2], heading)
            left_angles.append(la)
            right_angles.append(ra)

        # mean angles over rep (fallback NaN if no valid frames)
        angL = float(np.nanmean(left_angles)) if left_angles else float("nan")
        angR = float(np.nanmean(right_angles)) if right_angles else float("nan")

        sevL = classify(angL, out_mild, out_sev, in_mild, in_sev)
        sevR = classify(angR, out_mild, out_sev, in_mild, in_sev)
        sym  = abs(angL - angR) <= sym_tol if not (math.isnan(angL) or math.isnan(angR)) else False

        records.append(dict(
            rep_id   = int(row.rep_id),
            severity = [sevL, sevR, sym],
            frames   = [],
            value    = [angL, angR],
        ))

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print("saved →", os.path.abspath(out_csv))
    return df

# ------------------------------------------------------------------ CLI

def main():
    ap = argparse.ArgumentParser(description="Foot pointing (outward/inward) report")
    ap.add_argument("--keypoints", type=Path, default="imputed_ma.npy")
    ap.add_argument("--reps",      type=Path, default="repetition_data.csv")
    ap.add_argument("--heading",   type=float, default=0.0, help="athlete heading +deg=right")
    ap.add_argument("--out-mild",  type=float, default=35.0)
    ap.add_argument("--out-severe",type=float, default=45.0)
    ap.add_argument("--in-mild",   type=float, default=10.0)
    ap.add_argument("--in-severe", type=float, default=5.0)
    ap.add_argument("--sym-tol",   type=float, default=5.0)
    ap.add_argument("--output",    type=Path, default="foot_orientation_report.csv")
    args = ap.parse_args()

    kps  = np.load(args.keypoints)
    reps = pd.read_csv(args.reps)

    generate_report(kps, reps,
                    heading=args.heading,
                    out_mild=args.out_mild,
                    out_sev=args.out_severe,
                    in_mild=args.in_mild,
                    in_sev=args.in_severe,
                    sym_tol=args.sym_tol,
                    out_csv=str(args.output))

if __name__ == "__main__":
    main()
