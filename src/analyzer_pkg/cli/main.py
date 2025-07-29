#!/usr/bin/env python3
"""
full_video_analysis.py
Runs the complete AlphaPose → MotionBERT → analytics pipeline.

Supports
  • Local files        →  /path/to/video.mp4
  • S3 URIs            →  s3://bucket/key/clip.mp4

Typical use
------------
$ full_video_analysis.py s3://my-bucket/raw/demo.mp4                \
        --exercise squat                                            \
        --jobdir   /tmp/job_001

Flags
-----
--jobdir    : where *all* artefacts go (defaults /tmp/video_pose_<ts>)
--exercise  : squat | lunge | … (default: squat)
--no-copy   : skip the initial copy if the source video is already
              located at <jobdir>/src.mp4   (handy for repeated runs)
"""

from __future__ import annotations

import argparse, json, logging, os, re, shutil, subprocess, sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

import boto3
import numpy as np
import pandas as pd
from flask import Flask

# ────────────────────────── configuration ──────────────────────────
ALPHAPOSE_ROOT  = Path(os.environ["ALPHAPOSE_ROOT"])
MOTIONBERT_ROOT = Path(os.environ["MOTIONBERT_ROOT"])
REF_DIR         = (
    Path.home() / "analyzer_functions_final" / "analyzer_functions"
)
os.environ["PYTHONPATH"] = (
    f"{ALPHAPOSE_ROOT}:{MOTIONBERT_ROOT}:"
    f"{os.environ.get('PYTHONPATH', '')}"
)

LOG = logging.getLogger("full_video_analysis")
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)

# ────────────────────────── helpers ────────────────────────────────
S3_RX = re.compile(r"^s3://([^/]+)/(.+)$")


def run(cmd, *, cwd=None):
    """
    Convenience wrapper around ``subprocess.check_call`` that always
    prints the command.  If *cwd* is omitted, we automatically choose
    a sensible default for AlphaPose / MotionBERT.
    """
    if cwd is None:
        p = Path(cmd[0]).resolve()
        if str(p).startswith(str(ALPHAPOSE_ROOT)):
            cwd = ALPHAPOSE_ROOT
        elif str(p).startswith(str(MOTIONBERT_ROOT)):
            cwd = MOTIONBERT_ROOT
    print("→", " ".join(map(str, cmd)), f"(cwd={cwd})")
    subprocess.check_call(cmd, cwd=cwd)

def fetch_video(src: str, dst: Path, skip_if_same: bool) -> None:
    """
    • If *src* looks like s3://bucket/key  → download via boto3.
    • Otherwise treat as local file and shutil.copy2.

    When *skip_if_same* is True and src == dst, the copy is silently skipped.
    """
    if skip_if_same and Path(src).resolve() == dst.resolve():
        LOG.info("⚠️  source and destination are identical – skipping copy")
        return

    m = S3_RX.match(src)
    if m:  # ---------- S3 ----------
        bucket, key = m.groups()
        LOG.info("⬇️  downloading   s3://%s/%s → %s", bucket, key, dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        boto3.client("s3").download_file(bucket, key, str(dst))
    else:  # ---------- local ----------
        LOG.info("📥 copying        %s → %s", src, dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(src).expanduser(), dst)


def save_x3d_npy(mb_out_dir: Path) -> Path:
    """
    MotionBERT (our fork) now produces *directory* 3d-pose-results.npz/
    containing X3D.npy.  Older revs produced a single .npz file.

    This helper returns the path to a valid *.npy* file, converting if needed.
    """
    target = mb_out_dir / "3d-pose-results.npz"
    if target.is_dir():  # new layout (already *.npy*)
        npy = target / "X3D.npy"
        if not npy.exists():
            raise FileNotFoundError(f"{npy} missing inside {target}")
        return npy

    if target.is_file():  # legacy layout – convert
        npy = target.with_suffix(".npy")
        LOG.info("🔄 extracting X3D → %s", npy)
        np.save(npy, np.load(target)["X3D"])
        return npy

    raise FileNotFoundError(f"{target} (file *or* dir) not found")


# ────────────────────────── CLI ────────────────────────────────────
ap = argparse.ArgumentParser(
    description="Run AlphaPose + MotionBERT + analytics in one shot"
)
ap.add_argument("video", help="local file or s3://bucket/key.mp4")
ap.add_argument("--exercise", default="squat", help="exercise name (default: squat)")
ap.add_argument("--jobdir", help="where to store artefacts")
ap.add_argument(
    "--no-copy",
    action="store_true",
    help="assume <jobdir>/src.mp4 already exists – don't copy/download",
)
args = ap.parse_args()

jobdir = Path(
    args.jobdir or f"/tmp/video_pose_{int(datetime.now().timestamp())}"
).expanduser()
jobdir.mkdir(parents=True, exist_ok=True)

src_local = jobdir / "src.mp4"
if not args.no_copy:
    fetch_video(args.video, src_local, skip_if_same=True)
elif not src_local.exists():
    LOG.error("--no-copy specified but %s is missing", src_local)
    sys.exit(2)

# ─────────────────────── AlphaPose & MotionBERT ────────────────────
run([ALPHAPOSE_ROOT / "run_alphapose.sh",  jobdir])
run([MOTIONBERT_ROOT / "run_motionbert.sh", jobdir])

# ─────────────────────── Convert + analytics ───────────────────────
x3d_npy = save_x3d_npy(jobdir / "motionbert")

from analyzer_pkg.analysis import run_pipeline  # noqa:  E402  (lazy import)

app = Flask(__name__)
with app.app_context():
    report_json = run_pipeline(
        json_pose_path=jobdir / "alphapose" / "alphapose-results.json",
        X3D_pose_path=x3d_npy,
        reference_skeleton_path=REF_DIR / "reference_frame.npy",
        model_path=REF_DIR / "models" / "rf_badframe_detector.joblib",
        outdir=jobdir / "analysis",
        exercise=args.exercise,
    ).get_json()

# ─────────────────────── Persist results ───────────────────────────
analysis_dir = jobdir / "analysis"
analysis_dir.mkdir(exist_ok=True)

json_path = analysis_dir / f"{args.exercise}_analysis.json"
csv_path  = analysis_dir / f"{args.exercise}_analysis.csv"

with open(json_path, "w") as f:
    json.dump(report_json, f, indent=2)

pd.DataFrame(report_json).to_csv(csv_path, index=False)

npz_path = analysis_dir / f"{args.exercise}_analysis.npz"
np.savez(npz_path, X3D=np.load(x3d_npy))


LOG.info("✅ finished – artefacts in %s", jobdir)
LOG.info("   JSON : %s", json_path)
LOG.info("   CSV  : %s", csv_path)
LOG.info("   NPZ  : %s", npz_path)
