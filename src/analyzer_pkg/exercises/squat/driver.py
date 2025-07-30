#!/usr/bin/env python3
"""
full_pipeline_driver.py – single-entry script  (2025-07 hot-fix)

• Stores every artefact **both** in the scratch folder (`--out`) *and*
  in `/opt/analyzer/results/<exercise>_<rand>` for offline triage.

• Skips the Random-Forest bad-frame detector – all frames are treated as
  “good”, so the moving-average imputer works on raw key-points.

• The final JSON report is now an **object** instead of a bare list
  (required by the Flutter feedback screen).
"""
from __future__ import annotations

import argparse, json, secrets, shutil
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify

# ── local imports ────────────────────────────────────────────────
from . import (
    json_to_npy            as jtn,
    ma_impute              as ma,
    convert,
    keypoint_filter        as kpf,
    motionbert_output_to_unit_square as m3u,
    ref_frame_find         as rff,
    skeleton_length_fix    as slf,
    squat_detector         as sd,
    inward_knees_analysis  as ikn,
    heel_raise_analysis_26 as hra,
    forward_knees_analysis as fka,
    squat_depth_analysis   as sda,
    forward_lean_analysis  as fla,
    hip_path_analysis      as hpr,
    feet_width_analysis    as fwa,
)
from .compute_reference_lengths import pipeline_reference_lengths

app = Flask(__name__)   # only so we can return jsonify() in CLI mode
# ════════════════════════════════════════════════════════════════
def _mirror(src: Path, dst_dir: Path) -> None:
    """Copy *src* into *dst_dir* (idempotent)."""
    if src.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)
# ════════════════════════════════════════════════════════════════
def run_pipeline(
    json_pose_path: str | Path,
    X3D_pose_path : str | Path,
    reference_skeleton_path: str | Path,
    model_path    : str | Path,   # kept for API stability – unused
    outdir        : str | Path,
    exercise      : str,
):
    # ────────────── set-up folders ───────────────────────────────
    json_pose_path     = Path(json_pose_path)
    X3D_pose_path      = Path(X3D_pose_path)
    reference_skeleton = np.load(reference_skeleton_path)
    outdir             = Path(outdir)                # == jobdir/analysis
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] scratch  → {outdir.resolve()}")

    run_dir = (
        Path("/opt/analyzer/results")
        / f"{exercise}_{secrets.token_hex(4)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] persist  → {run_dir}")

    # ────────────── 1. AlphaPose JSON → NPY ──────────────────────
    alphapose = jtn.pipeline(json_pose_path, outdir)
    np.save(run_dir / "alphapose_results.npy", alphapose)
    shutil.copy2(json_pose_path, run_dir / "alphapose-results.json")
    _mirror(run_dir / "alphapose_results.npy", outdir)
    _mirror(run_dir / "alphapose-results.json", outdir)

    # ────────────── 2. bad-frame mask (all good) ─────────────────
    bad_csv = run_dir / "bad_frames.csv"
    pd.DataFrame(
        {"frame": np.arange(alphapose.shape[0]), "is_bad": 0}
    ).to_csv(bad_csv, index=False)
    _mirror(bad_csv, outdir)

    # ────────────── 3. MA imputation ─────────────────────────────
    imputed = ma.pipeline(
        skeletons=alphapose,
        df_bad=bad_csv,
        window="auto",
        imputed_out_path=run_dir / f"{exercise}_imputed_ma.npy",
    )
    _mirror(run_dir / f"{exercise}_imputed_ma.npy", outdir)

    # ────────────── 4. remaining stack ───────────────────────────
    skel_2d = convert.pipeline(alphapose)
    skel_3d = kpf.pipeline(X3D_pose_path, kernel_size=5)
    skel_3d, skel_3d_adapt = m3u.pipeline(
    skel_3d,
    out_stable   = run_dir / "motionbert_scaled.npy",
    out_adaptive = run_dir / "motionbert_scaled_adapt.npy",
)
# keep both files for triage
    _mirror(run_dir / "motionbert_scaled.npy",          outdir)
    _mirror(run_dir / "motionbert_scaled_adapt.npy",    outdir)

    ref_frame = rff.run_exercise_analysis(
        skel_2d, skel_3d, reference_value=reference_skeleton
    )

    len3d, len2d = pipeline_reference_lengths(
        input_3d=skel_3d, input_2d=imputed, frames=str(ref_frame)
    )
    json_3d, json_2d = json.dumps(len3d), json.dumps(len2d)

    slf.pipeline(skel_3d, json.loads(json_3d))

    df_reps  = sd.pipeline(keypoints=imputed, start_frame=ref_frame)
    df_ffpa  = ikn.pipeline(imputed, df_reps, output_csv=outdir / "ffpa_report.csv")
    df_heel  = hra.pipeline(imputed, df_reps)
    df_fk    = fka.pipeline(imputed, df_reps, lengths_json=json_2d)
    df_depth = sda.pipeline(imputed, df_reps, lengths_json=json_2d)
    df_lean  = fla.pipeline(imputed, df_reps, lengths_json=json_2d)
    df_hip   = hpr.pipeline(imputed, df_reps, lengths_json=json_2d)
    df_feet  = fwa.pipeline(imputed, df_reps)

    def _tag(df: pd.DataFrame, tag: str) -> pd.DataFrame:
        return df.rename(columns={c: f"{tag}_{c}" for c in df.columns if c != "rep_id"})

    merged_df = reduce(
        lambda l, r: pd.merge(l, r, on="rep_id", how="outer"),
        [
            _tag(d, t)
            for d, t in [
                (df_reps,  "reps"),
                (df_ffpa,  "ffpa"),
                (df_heel,  "heel"),
                (df_fk,    "fk"),
                (df_depth, "depth"),
                (df_lean,  "flean"),
                (df_hip,   "hip"),
                (df_feet,  "feet"),
            ]
        ],
    )

    # ────────────── 5. build FINAL report object ─────────────────
    report = {
        "mistakes": merged_df.to_dict(orient="records"),
        "fps"     : 30,                # ↔ replace with real fps if needed
        "version" : "2025-07-06",
    }

    # main triage copy
    merged_json = run_dir / "merged_report.json"
    merged_json.write_text(json.dumps(report, indent=2))

    # copy inside scratch → Celery will upload it
    squat_json = outdir / "squat_analysis.json"
    squat_json.write_text(json.dumps(report, indent=2))

    # mirror both into scratch/persist dirs
    _mirror(merged_json, outdir)  # convenience duplicate
    _mirror(squat_json,   run_dir)

    return jsonify(report)

# ════════════════════════════════════════════════════════════════
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser("Full squat-pipeline driver")
    p.add_argument("--json",   required=True, type=Path)
    p.add_argument("--x3d",    required=True, type=Path)
    p.add_argument("--ref3d",  required=True, type=Path)
    p.add_argument("--model",  default="rf_badframe_detector.joblib", type=Path)
    p.add_argument("--out",    default="outputs", type=Path)
    p.add_argument("--exercise", default="squat")
    args = p.parse_args(argv)

    run_pipeline(
        json_pose_path=args.json,
        X3D_pose_path=args.x3d,
        reference_skeleton_path=args.ref3d,
        model_path=args.model,
        outdir=args.out,
        exercise=args.exercise,
    )

if __name__ == "__main__":
    main()
