#!/usr/bin/env python3
"""
inward_knees_analysis.py  🦵
===========================
FPPA and line-crossing analytics for squat reps (2-D unit-square coordinates).

Public API
----------
generate_ffpa_report(...)
generate_line_crossing_report(...)
pipeline(...)                     # convenience – returns the FPPA report as a DataFrame

Change log (2025-07-06)
-----------------------
* Added dual support for ['start','end'] **and** ['rep_start','rep_end'] rep tables.
* Normalised default `crossing_thresh` to 0.05 (was 10.0 in pixel space).
* `generate_line_crossing_report` now accepts a DataFrame as well as a CSV path.
* Guarded `_signed_dist` against zero-length hip-ankle vectors.
* Updated CLI defaults/help accordingly.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


# ------------------ FPPA math helpers ------------------
def _angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Return the interior angle (deg) between two 2-D vectors."""
    num = float(np.dot(v1, v2))
    den = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.degrees(np.arccos(np.clip(num / den, -1.0, 1.0))))


def _signed_dist(pt: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Signed perpendicular distance from point *pt* to the line through *p1*→*p2*.
    Positive = medial side (hip-midpoint test later).
    """
    v = p2 - p1
    norm = np.linalg.norm(v)
    if norm < 1e-8:        # hip and ankle coincide – avoid divide-by-zero
        return 0.0
    perp = np.array([-v[1], v[0]]) / norm
    return float(np.dot(pt - p1, perp))


def _fppa_outside(kp2: np.ndarray, side: str = "L") -> float:
    """
    Outside-knee frontal-plane projection angle (FPPA) for one frame.

    Parameters
    ----------
    kp2  : ndarray (17, 2)  – 2-D keypoints in unit square
    side : {"L","R"}        – left or right leg
    """
    h, k, a = (11, 13, 15) if side == "L" else (12, 14, 16)  # hip, knee, ankle
    hip, knee, ankle = kp2[h], kp2[k], kp2[a]

    # Interior (anatomical) knee angle
    interior = _angle(hip - knee, ankle - knee)

    # Determine if knee is medial to hip-ankle line
    midhip = (kp2[11] + kp2[12]) / 2.0
    medial = _signed_dist(midhip, hip, ankle) * _signed_dist(knee, hip, ankle) > 0

    return interior if medial else 360.0 - interior


# ---------------------------------------------------------------------
# ❶  FPPA report generator
# ---------------------------------------------------------------------
def generate_ffpa_report(
    keypoints_array: np.ndarray,
    reps: str | Path | pd.DataFrame,
    mild_thresh: float = 170.0,
    severe_thresh: float = 160.0,
    output_csv: str | Path | None = "ffpa_report.csv",
) -> pd.DataFrame:
    """
    Compute outside-knee FPPA (deg) for each rep.

    Parameters
    ----------
    keypoints_array : ndarray (F, 17, 3)  – imputed 2-D+conf keypoints
    reps            : CSV path or DataFrame with rep boundaries
    mild_thresh     : FPPA < mild_thresh ⇒ "mild" valgus
    severe_thresh   : FPPA < severe_thresh ⇒ "severe" valgus
    output_csv      : where to write the report (None = don’t write)

    Returns
    -------
    pd.DataFrame with per-rep minima and severity labels.
    """
    # --- load reps table --------------------------------------------------
    reps_df = reps if isinstance(reps, pd.DataFrame) else pd.read_csv(reps)

    # Detect column names
    if {"start", "end"}.issubset(reps_df.columns):
        start_col, end_col = "start", "end"
    elif {"rep_start", "rep_end"}.issubset(reps_df.columns):
        start_col, end_col = "rep_start", "rep_end"
    else:
        raise ValueError(
            "Reps table must contain either ['start','end'] or ['rep_start','rep_end'] columns"
        )

    # --- iterate reps -----------------------------------------------------
    rows: list[dict] = []
    for _, rep in reps_df.iterrows():
        rep_id = int(rep["rep_id"])
        start, end = int(rep[start_col]), int(rep[end_col])

        left_vals, right_vals = [], []
        for f in range(start, end + 1):
            kp2 = keypoints_array[f, :, :2]
            left_vals.append(_fppa_outside(kp2, "L"))
            right_vals.append(_fppa_outside(kp2, "R"))

        def _class(vals: list[float]) -> tuple[str, float]:
            if not vals:
                return "none", float("nan")
            mn = min(vals)
            if mn < severe_thresh:
                return "severe", mn
            if mn < mild_thresh:
                return "mild", mn
            return "none", mn

        lsev, lmin = _class(left_vals)
        rsev, rmin = _class(right_vals)
        rows.append(
            {
                "rep_id": rep_id,
                "left_min_FPPA": lmin,
                "left_severity": lsev,
                "right_min_FPPA": rmin,
                "right_severity": rsev,
            }
        )

    df = pd.DataFrame(rows)

    if output_csv is not None:
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"✅ FPPA report → {out}")

    return df


# ---------------------------------------------------------------------
# ❷  Line-crossing report (normalised coords)
# ---------------------------------------------------------------------
def generate_line_crossing_report(
    keypoints_array: np.ndarray,
    reps: str | Path | pd.DataFrame,
    crossing_thresh: float = 0.05,
    output_csv: str | Path = "line_crossing_report.csv",
) -> pd.DataFrame:
    """
    Detect knee crossings of the hip-to-ankle line for each rep.

    Parameters
    ----------
    crossing_thresh : float
        Minimum perpendicular distance (unit coords) to count as a crossing.
        0.05 ≈ 5 % of frame height/width.
    """
    reps_df = reps if isinstance(reps, pd.DataFrame) else pd.read_csv(reps)

    # Column detection
    if {"start", "end"}.issubset(reps_df.columns):
        start_col, end_col = "start", "end"
    elif {"rep_start", "rep_end"}.issubset(reps_df.columns):
        start_col, end_col = "rep_start", "rep_end"
    else:
        raise ValueError(
            "Reps table must contain either ['start','end'] or ['rep_start','rep_end'] columns"
        )

    rows = []
    for _, rep in reps_df.iterrows():
        rep_id = int(rep["rep_id"])
        start, end = int(rep[start_col]), int(rep[end_col])

        ldist, rdist = [], []
        for f in range(start, end + 1):
            kp2 = keypoints_array[f, :, :2]

            # ---------- left side ----------
            medial_l = _signed_dist((kp2[11] + kp2[12]) / 2, kp2[11], kp2[15]) * \
                       _signed_dist(kp2[13], kp2[11], kp2[15]) > 0
            if medial_l:
                dist_l = abs(_signed_dist(kp2[13], kp2[11], kp2[15]))
                if dist_l >= crossing_thresh:
                    ldist.append(dist_l)

            # ---------- right side ----------
            medial_r = _signed_dist((kp2[11] + kp2[12]) / 2, kp2[12], kp2[16]) * \
                       _signed_dist(kp2[14], kp2[12], kp2[16]) > 0
            if medial_r:
                dist_r = abs(_signed_dist(kp2[14], kp2[12], kp2[16]))
                if dist_r >= crossing_thresh:
                    rdist.append(dist_r)

        def _class(ds):
            if not ds:
                return "none", 0.0
            mx = max(ds)
            return ("severe" if mx >= 2 * crossing_thresh else "mild"), mx

        lsev, lmax = _class(ldist)
        rsev, rmax = _class(rdist)
        rows.append(
            {
                "rep_id": rep_id,
                "left_max_dist": lmax,
                "left_severity": lsev,
                "right_max_dist": rmax,
                "right_severity": rsev,
            }
        )

    df = pd.DataFrame(rows)
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"✅ Line-crossing report → {out}")
    return df


# ---------------------------------------------------------------------
# ❸  Convenience wrapper – FPPA only
# ---------------------------------------------------------------------
def pipeline(
    keypoints_array: np.ndarray,
    reps: str | Path | pd.DataFrame,
    *,
    mild_thresh: float = 175.0,
    severe_thresh: float = 170.0,
    output_csv: str | Path | None = "ffpa_report.csv",
) -> pd.DataFrame:
    """Run FPPA analysis end-to-end and return the report."""
    return generate_ffpa_report(
        keypoints_array,
        reps,
        mild_thresh=mild_thresh,
        severe_thresh=severe_thresh,
        output_csv=output_csv,
    )


# ---------------------------------------------------------------------
# ❹  CLI entry-point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser("FFPA & line-crossing reporter")
    ap.add_argument("--keypoints",       default="imputed_ma.npy",
                    help="NumPy npy file with (F,17,3) keypoints")
    ap.add_argument("--reps",            default="repetition_data.csv",
                    help="CSV or feather file with rep boundaries")
    ap.add_argument("--out-fppa",        default="ffpa_report.csv",
                    help="Output CSV for FPPA report")
    ap.add_argument("--out-line",        default="line_crossing_report.csv",
                    help="Output CSV for line-crossing report")
    ap.add_argument("--crossing-thresh", type=float, default=0.05,
                    help="Distance threshold for line-cross detection (unit coords)")

    args = ap.parse_args()

    kps = np.load(args.keypoints)
    generate_ffpa_report(kps, args.reps, output_csv=args.out_fppa)
    generate_line_crossing_report(
        kps, args.reps,
        crossing_thresh=args.crossing_thresh,
        output_csv=args.out_line
    )
