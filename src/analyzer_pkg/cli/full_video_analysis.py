#!/usr/bin/env python3
"""
full_video_analysis.py
Runs the complete AlphaPose → MotionBERT → analytics pipeline.

(unchanged doc‑string omitted for brevity)
"""
from __future__ import annotations

import argparse, json, logging, os, re, shutil, subprocess, sys
from datetime import datetime
from pathlib import Path

import boto3
import importlib
import importlib.resources as res
import numpy as np
import pandas as pd
from flask import Flask
import subprocess


# ───────────────────────── debugging ─────────────────────────
def debug_tools_pipeline(jobdir, analysis_dir, exercise):
    """
    Run the debugging tools and save their outputs to analysis_dir/debug_plots.
    """
    debug_dir = analysis_dir / "debug_plots"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # -------- 1. alphapose_bbox_overlay.py --------------
    alphapose_json = jobdir / "alphapose" / "alphapose-results.json"
    video_mp4      = jobdir / "src.mp4"
    out_overlay    = debug_dir / "alphapose_overlay.mp4"
    out_stickman   = debug_dir / "alphapose_stickman.mp4"
    out_plot       = debug_dir / "alphapose_sidelengths.png"
    alphapose_overlay_cmd = [
        "python", "debug_tools/alphapose_bbox_overlay.py",
        "--alphapose", str(alphapose_json),
        "--video",     str(video_mp4),
        "--out",       str(out_overlay),
        "--stickman",  str(out_stickman),
        "--plot",      str(out_plot)
    ]
    subprocess.run(alphapose_overlay_cmd, check=True)

    # -------- 2. skeleton_overlay.py --------------------
    skeleton_overlay_cmd = [
        "python", "debug_tools/skeleton_overlay.py",
        "--jobdir", str(jobdir),
        "--out", str(debug_dir / "overlay_skeleton.mp4"),
    ]
    subprocess.run(skeleton_overlay_cmd, check=True)

    # -------- 3. debug_skeleton_hip_y_overlay.py --------
    alphapose_scaled = jobdir / "analysis" / f"{exercise}_imputed_ma.npy"
    debug_hip_y_cmd = [
        "python", "debug_tools/debug_skeleton_hip_y_overlay.py",
        "--input", str(alphapose_scaled),
        "--out",   str(debug_dir / "hip_y_overlay.avi"),
    ]
    subprocess.run(debug_hip_y_cmd, check=True)
# ───────────────────────── configuration ─────────────────────────
def _tool_dir(name: str) -> Path:
    here = Path(__file__).resolve().parent.parent  # …/analyzer_pkg/cli
    return here / "tools" / name

ALPHAPOSE_ROOT  = Path(os.getenv("ALPHAPOSE_ROOT",  _tool_dir("AlphaPose")))
MOTIONBERT_ROOT = Path(os.getenv("MOTIONBERT_ROOT", _tool_dir("MotionBERT")))

# make sure the shell‑scripts see them
os.environ.setdefault("ALPHAPOSE_ROOT",  str(ALPHAPOSE_ROOT))
os.environ.setdefault("MOTIONBERT_ROOT", str(MOTIONBERT_ROOT))

REF_DIR = Path.home() / "analyzer_functions_final" / "analyzer_functions"

# prepend tools to PYTHONPATH for the subprocesses
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

# ───────────────────────── helpers ───────────────────────────────
S3_RX = re.compile(r"^s3://([^/]+)/(.+)$")


def run(cmd, *, cwd=None) -> None:
    """Print & exec a subprocess, defaulting cwd to the tool root."""
    if cwd is None:
        p = Path(cmd[0]).resolve()
        if str(p).startswith(str(ALPHAPOSE_ROOT)):
            cwd = ALPHAPOSE_ROOT
        elif str(p).startswith(str(MOTIONBERT_ROOT)):
            cwd = MOTIONBERT_ROOT
    print("→", " ".join(map(str, cmd)), f"(cwd={cwd})")
    subprocess.check_call(cmd, cwd=cwd)


def fetch_video(src: str, dst: Path, skip_if_same: bool) -> None:
    if skip_if_same and Path(src).resolve() == dst.resolve():
        LOG.info("⚠️  source and destination are identical – skipping copy")
        return

    m = S3_RX.match(src)
    if m:
        bucket, key = m.groups()
        LOG.info("⬇️  downloading   s3://%s/%s → %s", bucket, key, dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        boto3.client("s3").download_file(bucket, key, str(dst))
    else:
        LOG.info("📥 copying        %s → %s", src, dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(src).expanduser(), dst)


def save_x3d_npy(mb_out_dir: Path) -> Path:
    target = mb_out_dir / "3d-pose-results.npz"
    if target.is_dir():                   # new layout
        npy = target / "X3D.npy"
        if not npy.exists():
            raise FileNotFoundError(f"{npy} missing inside {target}")
        return npy

    if target.is_file():                  # legacy → convert
        npy = target.with_suffix(".npy")
        LOG.info("🔄 extracting X3D → %s", npy)
        np.save(npy, np.load(target)["X3D"])
        return npy

    raise FileNotFoundError(target)


# ────────────────────────── CLI ──────────────────────────────────
ap = argparse.ArgumentParser(
    description="Run AlphaPose + MotionBERT + analytics in one shot"
)
ap.add_argument("video", help="local file or s3://bucket/key.mp4")
ap.add_argument("--exercise", default="squat", help="exercise name")
ap.add_argument("--jobdir", help="where to store artefacts")
ap.add_argument("--no-copy", action="store_true",
                help="assume <jobdir>/src.mp4 already exists")
ap.add_argument("--debug", action="store_true")
args = ap.parse_args()

jobdir = Path(args.jobdir or f"/tmp/video_pose_{int(datetime.now().timestamp())}"
              ).expanduser()
jobdir.mkdir(parents=True, exist_ok=True)

src_local = jobdir / "src.mp4"
if not args.no_copy:
    fetch_video(args.video, src_local, skip_if_same=True)
elif not src_local.exists():
    LOG.error("--no-copy specified but %s is missing", src_local)
    sys.exit(2)

# ───────────── AlphaPose & MotionBERT ───────────────────────────
run([ALPHAPOSE_ROOT / "run_alphapose.sh",  jobdir])
run([MOTIONBERT_ROOT / "run_motionbert.sh", jobdir])

# ───────────── Analytics driver (per exercise) ───────────────────
driver = importlib.import_module(
    f"analyzer_pkg.exercises.{args.exercise}.driver"
)
exercise_pkg = importlib.import_module(
    f"analyzer_pkg.exercises.{args.exercise}"
)
EXER_DIR = Path(res.files(exercise_pkg))

x3d_npy = save_x3d_npy(jobdir / "motionbert")

app = Flask(__name__)
with app.app_context():
    report_json = driver.run_pipeline(
        json_pose_path          = jobdir / "alphapose" / "alphapose-results.json",
        X3D_pose_path           = x3d_npy,
        reference_skeleton_path = EXER_DIR / "data" / "reference_frame.npy",
        model_path              = Path(__file__).resolve().parent.parent
                                  / "models" / "rf_badframe_detector.joblib",
        outdir                  = jobdir / "analysis",
        exercise                = args.exercise,
        debug                 = args.debug,
    ).get_json()

# ───────────── Persist results ───────────────────────────────────
analysis_dir = jobdir / "analysis"
analysis_dir.mkdir(exist_ok=True)

json_path = analysis_dir / f"{args.exercise}_analysis.json"
csv_path  = analysis_dir / f"{args.exercise}_analysis.csv"
npz_path  = analysis_dir / f"{args.exercise}_analysis.npz"

with open(json_path, "w") as f:
    json.dump(report_json, f, indent=2)

pd.DataFrame(report_json).to_csv(csv_path, index=False)
np.savez(npz_path, X3D=np.load(x3d_npy))

LOG.info("✅ finished – artefacts in %s", jobdir)
LOG.info("   JSON : %s", json_path)
LOG.info("   CSV  : %s", csv_path)
LOG.info("   NPZ  : %s", npz_path)

if args.debug:
    LOG.info("🔍 running debugging tools")
    debug_tools_pipeline(jobdir, analysis_dir, args.exercise)

    LOG.info("📹  video overlay saved to %s", analysis_dir / "debug_plots")

def main() -> None:           # required by setup.cfg / pyproject.toml
    """Wrapper so `full‑video‑analysis` can `import main`."""
    pass 