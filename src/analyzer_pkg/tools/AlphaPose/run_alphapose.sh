#!/usr/bin/env bash
set -euo pipefail



FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if   (( FREE < 4000 ));  then DET=16;  POSE=32;  Q=16;
elif (( FREE < 6000 ));  then DET=24;  POSE=48;  Q=32;
elif (( FREE < 10000 )); then DET=32;  POSE=64;  Q=32;
elif (( FREE < 14000 )); then DET=48;  POSE=96;  Q=48;
else                        DET=64;  POSE=128; Q=64;
fi
echo "ℹ️  Using detbatch=$DET  posebatch=$POSE  qsize=$Q (free VRAM: ${FREE} MiB)"
###############################################################################
# run_alphapose.sh  –  AlphaPose runner that always leaves breadcrumbs
#                     (meta + 10 JPG frames) in /home/ubuntu/video_meta
#
# Accepts the same call forms as the original:
#   1)  run_alphapose.sh  <job_dir> <video>        # legacy, two args
#   2)  run_alphapose.sh  <job_dir>                # legacy, video = <job_dir>/src.mp4
#   3)  run_alphapose.sh  <video>                  # new, auto-creates job dir
###############################################################################

# ─────────────────────────────── persistent artefacts ────────────────────────
PERSIST_BASE="$HOME/video_meta"        # <── change if you wish
mkdir -p "$PERSIST_BASE"

# ──────────────────────────────── parse arguments ────────────────────────────
if [[ $# -eq 2 && -d $1 ]]; then            # mode 1
    JOB_DIR="$(realpath "$1")"
    VIDEO="$(realpath "$2")"
    RUN_ID="$(basename "$JOB_DIR")"

elif [[ $# -eq 1 && -d $1 ]]; then          # mode 2
    JOB_DIR="$(realpath "$1")"
    VIDEO="$JOB_DIR/src.mp4"
    RUN_ID="$(basename "$JOB_DIR")"

elif [[ $# -eq 1 ]]; then                   # mode 3
    VIDEO="$(realpath "$1")"
    RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
    JOB_DIR="$PERSIST_BASE/$RUN_ID"          # we just reuse the same id
    mkdir -p "$JOB_DIR"

else
    echo "Usage:"
    echo "  $0 <job_dir> <video>"
    echo "  $0 <job_dir>"
    echo "  $0 <video>"
    exit 1
fi

mkdir -p "$JOB_DIR/alphapose"

# ─────────────────────────────── 1. meta + 10 frames ─────────────────────────
{
    echo "video_path=${VIDEO}"
    ffprobe -v error -select_streams v:0 \
            -show_entries stream=width,height:stream_tags=rotate \
            -of default=noprint_wrappers=1:nokey=0 "${VIDEO}" || true
} > "$JOB_DIR/video_meta.txt" || true

python - <<'PY' "${VIDEO}" "${JOB_DIR}"
import cv2, sys, pathlib, math
video, outdir = sys.argv[1:3]
outdir = pathlib.Path(outdir)
cap   = cv2.VideoCapture(video)
F     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)

# grab 10 evenly-spaced frames (incl. first & last)
for i in range(10):
    idx = min(int(i*F/9), F-1)          # 0 … F-1
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    if ok:
        fname = outdir / f"debug_{i:02d}.jpg"
        fname.write_bytes(cv2.imencode(".jpg", img)[1])
PY

# ─────────────────────────────── 2. AlphaPose run ────────────────────────────
ROOT="${ALPHAPOSE_ROOT:?ALPHAPOSE_ROOT env var not set}"
CFG="$ROOT/configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml"
CKPT="$ROOT/pretrained_models/halpe26_fast_res50_256x192.pth"

(
  cd "$ROOT"
  python scripts/demo_inference.py \
        --cfg        "$CFG" \
        --checkpoint "$CKPT" \
        --video      "$VIDEO" \
        --outdir     "$JOB_DIR/alphapose" \
        --detbatch 16 --posebatch 64 --qsize 32
)

echo "✅ AlphaPose done   →  $JOB_DIR/alphapose"
echo "ⓘ Meta + frames in  →  $JOB_DIR"
