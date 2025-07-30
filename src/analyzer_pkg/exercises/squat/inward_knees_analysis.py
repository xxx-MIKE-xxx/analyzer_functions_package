#!/usr/bin/env python3
"""
inward_knees_analysis.py  🦵   (hot‑fix 2025‑07‑30)
────────────────────────────────────────────────────
Zero‑division–safe FPPA calculation.

The only changes vs. the 2025‑07‑06 revision are:

1. `_angle`   – early‑return *nan* when either vector is (almost) zero.
2. `_fppa_outside`
   * propagates the *nan* if the interior angle is undefined.
3. `generate_ffpa_report`
   * ignores NaNs when computing per‑rep minima / severity.
Nothing else was modified.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


# ────────────────── FPPA math helpers ──────────────────
_EPS = 1e-6   # tolerance for “zero” vector length


def _angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Interior angle (deg) between two 2‑D vectors – NaN if undefined."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < _EPS or n2 < _EPS:          # one vector collapsed → undefined
        return float('nan')
    num = float(np.dot(v1, v2))
    den = float(n1 * n2)
    return float(np.degrees(np.arccos(np.clip(num / den, -1.0, 1.0))))


def _signed_dist(pt: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Signed perpendicular distance from *pt* to the line p1→p2."""
    v = p2 - p1
    norm = np.linalg.norm(v)
    if norm < _EPS:
        return 0.0
    perp = np.array([-v[1], v[0]]) / norm
    return float(np.dot(pt - p1, perp))


def _fppa_outside(kp2: np.ndarray, side: str = "L") -> float:
    """Outside‑knee FPPA for one frame – NaN if hip/ankle coincide with knee."""
    h, k, a = (11, 13, 15) if side == "L" else (12, 14, 16)
    hip, knee, ankle = kp2[h], kp2[k], kp2[a]

    interior = _angle(hip - knee, ankle - knee)
    if not np.isfinite(interior):       # bad geometry this frame
        return float('nan')

    midhip = (kp2[11] + kp2[12]) / 2.0
    medial = _signed_dist(midhip, hip, ankle) * _signed_dist(knee, hip, ankle) > 0
    return interior if medial else 360.0 - interior


# ───────────────────── FPPA report ─────────────────────
def generate_ffpa_report(
    keypoints_array: np.ndarray,
    reps: str | Path | pd.DataFrame,
    mild_thresh: float = 170.0,
    severe_thresh: float = 160.0,
    output_csv: str | Path | None = "ffpa_report.csv",
) -> pd.DataFrame:
    reps_df = reps if isinstance(reps, pd.DataFrame) else pd.read_csv(reps)

    if {"start", "end"}.issubset(reps_df.columns):
        start_col, end_col = "start", "end"
    elif {"rep_start", "rep_end"}.issubset(reps_df.columns):
        start_col, end_col = "rep_start", "rep_end"
    else:
        raise ValueError("Reps table must contain either "
                         "['start','end'] or ['rep_start','rep_end']")

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
            finite = [v for v in vals if np.isfinite(v)]
            if not finite:
                return "none", float('nan')
            mn = min(finite)
            if mn < severe_thresh:
                return "severe", mn
            if mn < mild_thresh:
                return "mild", mn
            return "none", mn

        lsev, lmin = _class(left_vals)
        rsev, rmin = _class(right_vals)
        rows.append(
            dict(rep_id=rep_id,
                 left_min_FPPA=lmin,  left_severity=lsev,
                 right_min_FPPA=rmin, right_severity=rsev)
        )

    df = pd.DataFrame(rows)
    if output_csv is not None:
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"✅ FPPA report → {out}")
    return df


# ──────────────────── Line crossing (unchanged) ────────────────────
def generate_line_crossing_report(
    keypoints_array: np.ndarray,
    reps: str | Path | pd.DataFrame,
    crossing_thresh: float = 0.05,
    output_csv: str | Path = "line_crossing_report.csv",
) -> pd.DataFrame:
    # ... **identical to previous version** ...
    # (omitted for brevity – no functional changes required)
    # ----------------------------------------------------------------
    reps_df = reps if isinstance(reps, pd.DataFrame) else pd.read_csv(reps)
    if {"start", "end"}.issubset(reps_df.columns):
        start_col, end_col = "start", "end"
    elif {"rep_start", "rep_end"}.issubset(reps_df.columns):
        start_col, end_col = "rep_start", "rep_end"
    else:
        raise ValueError("Reps table must contain either "
                         "['start','end'] or ['rep_start','rep_end']")

    rows = []
    for _, rep in reps_df.iterrows():
        rep_id = int(rep["rep_id"])
        start, end = int(rep[start_col]), int(rep[end_col])

        ldist, rdist = [], []
        for f in range(start, end + 1):
            kp2 = keypoints_array[f, :, :2]

            # left
            medial_l = _signed_dist((kp2[11] + kp2[12]) / 2, kp2[11], kp2[15]) * \
                       _signed_dist(kp2[13], kp2[11], kp2[15]) > 0
            if medial_l:
                d = abs(_signed_dist(kp2[13], kp2[11], kp2[15]))
                if d >= crossing_thresh:
                    ldist.append(d)

            # right
            medial_r = _signed_dist((kp2[11] + kp2[12]) / 2, kp2[12], kp2[16]) * \
                       _signed_dist(kp2[14], kp2[12], kp2[16]) > 0
            if medial_r:
                d = abs(_signed_dist(kp2[14], kp2[12], kp2[16]))
                if d >= crossing_thresh:
                    rdist.append(d)

        def _class(ds):
            if not ds:
                return "none", 0.0
            mx = max(ds)
            return ("severe" if mx >= 2 * crossing_thresh else "mild"), mx

        lsev, lmax = _class(ldist)
        rsev, rmax = _class(rdist)
        rows.append(dict(rep_id=rep_id,
                         left_max_dist=lmax,  left_severity=lsev,
                         right_max_dist=rmax, right_severity=rsev))

    df = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Line‑crossing report → {output_csv}")
    return df


# ───────────────── Convenience wrapper (unchanged) ─────────────────
def pipeline(
    keypoints_array: np.ndarray,
    reps: str | Path | pd.DataFrame,
    *,
    mild_thresh: float = 175.0,
    severe_thresh: float = 170.0,
    output_csv: str | Path | None = "ffpa_report.csv",
) -> pd.DataFrame:
    return generate_ffpa_report(
        keypoints_array, reps,
        mild_thresh=mild_thresh,
        severe_thresh=severe_thresh,
        output_csv=output_csv,
    )


# ────────────────────────── CLI entry ‑ unchanged ──────────────────
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser("FFPA & line‑crossing reporter (safe)")
    ap.add_argument("--keypoints",       default="imputed_ma.npy",
                    help="NumPy npy file with (F,17,3) keypoints")
    ap.add_argument("--reps",            default="repetition_data.csv",
                    help="CSV or feather file with rep boundaries")
    ap.add_argument("--out-fppa",        default="ffpa_report.csv",
                    help="Output CSV for FPPA report")
    ap.add_argument("--out-line",        default="line_crossing_report.csv",
                    help="Output CSV for line‑crossing report")
    ap.add_argument("--crossing-thresh", type=float, default=0.05,
                    help="Distance threshold for line‑cross detection (unit coords)")

    args = ap.parse_args()

    kps = np.load(args.keypoints)
    generate_ffpa_report(kps, args.reps, output_csv=args.out_fppa)
    generate_line_crossing_report(
        kps, args.reps,
        crossing_thresh=args.crossing_thresh,
        output_csv=args.out_line
    )
