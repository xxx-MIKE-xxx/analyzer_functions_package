#!/usr/bin/env bash
set -euo pipefail

JOB_DIR=${1:?Usage: $0 <job_dir>}

VIDEO="$JOB_DIR/src.mp4"
JSON="$JOB_DIR/alphapose/alphapose-results.json"
[[ -f "$JSON" ]] || { echo "❌ 2D json missing: $JSON" >&2; exit 1; }

ROOT="$MOTIONBERT_ROOT"
CFG="$ROOT/configs/pose3d/MB_ft_h36m_global_lite.yaml"
CKPT="$ROOT/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin"
OUT_DIR="$JOB_DIR/motionbert"; mkdir -p "$OUT_DIR"

echo "---- MotionBERT -----------------------------------------------------"
echo " video : $VIDEO"
echo " json  : $JSON"
echo " out   : $OUT_DIR"
echo "--------------------------------------------------------------------"

# run from repo root so relative paths work
(
  cd "$ROOT"
  QT_QPA_PLATFORM=offscreen \
  python infer_wild.py \
         --config "$CFG" \
         -e      "$CKPT" \
         -v      "$VIDEO" \
         -j      "$JSON" \
         -o      "$OUT_DIR/3d-pose-results.npz" \
         --pixel \
    # keep the largest person only
)

echo "✅  MotionBERT done – results in $OUT_DIR"
