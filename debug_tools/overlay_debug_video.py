#!/usr/bin/env python3
"""
overlay_debug_video.py
-------------------------------------------------------------
Create a “developer overlay” video that shows, for every
detected mistake, a big label *and* red markers on the
relevant joints.

Usage
-----
$ python overlay_debug_video.py --jobdir <JOBDIR> [--out overlay.mp4]

<JOBDIR> must be the directory that already contains:
  • alphapose/alphapose‑results.json   (original pixel coords)
  • src.mp4                            (original video)
  • analysis/squat_analysis.json       (the report)
"""

from __future__ import annotations

import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ────────────────────────────── TWEAK ME ───────────────────────────
FONT_FACE = cv2.FONT_HERSHEY_DUPLEX            # will fallback if no FreeType
FONT_SCALE = 1.1                               # bigger text
FONT_THICK = 2
FONT_COLOR = (255, 255, 255)                   # white fill
OUTLINE_COLOR = (0, 0, 0)                      # 1‑px black outline
JOINT_RADIUS = 9
JOINT_COLOR = (0, 0, 255)                      # red
FRAMES_BEFORE = 20
FRAMES_AFTER  = 20
# -------------------------------------------------------------------

JOINT_NAMES = [
    "Nose", "LEye", "REye", "LEar", "REar", "LShoulder", "RShoulder",
    "LElbow", "RElbow", "LWrist", "RWrist", "LHip", "RHip", "LKnee",
    "RKnee", "LAnkle", "RAnkle", "Head", "Neck", "Hip", "LBigToe",
    "RBigToe", "LSmallToe", "RSmallToe", "LHeel", "RHeel",
]

# Which joints to highlight per mistake
MISTAKE_JOINTS = {
    "flean":      ["LShoulder", "RShoulder", "Hip"],
    "hip":        ["LHip", "RHip"],
    "fk":         ["LKnee", "RKnee"],
    "ffpa_left":  ["LKnee"],
    "ffpa_right": ["RKnee"],
    "depth":      ["LHip", "RHip", "LKnee", "RKnee"],
    "feet":       ["LAnkle", "RAnkle"],
    "heel":       ["LHeel",  "RHeel"],
}

NAME2IDX = {n: i for i, n in enumerate(JOINT_NAMES)}


def _draw_text(img, text, org):
    """White text with 1‑px black outline."""
    x, y = org
    cv2.putText(img, text, (x, y), FONT_FACE, FONT_SCALE,
                OUTLINE_COLOR, FONT_THICK + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT_FACE, FONT_SCALE,
                FONT_COLOR, FONT_THICK, cv2.LINE_AA)


def load_keypoints(json_path: Path) -> dict[int, np.ndarray]:
    """
    Returns {frame_idx: (26, 3) array} in original pixel space.
    frame_idx is taken from the `image_id` field (e.g. "10.jpg" → 10).
    """
    with open(json_path) as f:
        data = json.load(f)

    frames = {}
    for entry in data:
        frame_no = int(Path(entry["image_id"]).stem)  # "123.jpg" → 123
        kps = np.array(entry["keypoints"], dtype=np.float32).reshape(-1, 3)
        frames[frame_no] = kps
    return frames


def build_mistake_index(report: dict) -> dict[int, list[dict]]:
    """
    Produce {frame_no: [mistake_dict, …]} so we can
    look up quickly which mistakes are active in a given frame.
    """
    idx = defaultdict(list)
    for m in report["mistakes"]:
        for key in ["flean", "fk", "depth", "ffpa_left", "ffpa_right",
                    "feet", "heel", "hip"]:
            frame_field = f"{key}_frame" if f"{key}_frame" in m else None
            if frame_field and m.get(f"{key}_severity", "none") != "none":
                peak = m[frame_field]
                for f in range(peak - FRAMES_BEFORE, peak + FRAMES_AFTER + 1):
                    idx[f].append({"type": key, "peak": peak, "mistake": m})
        # hip is special: it carries ranges in hip_frames
        if "hip_frames" in m:
            for rng, sev, val in zip(m["hip_frames"],
                                     m["hip_severity"],
                                     m["hip_value"]):
                if sev == "none":
                    continue
                start, end = rng
                for f in range(start, end + 1):
                    idx[f].append({"type": "hip", "peak": (start+end)//2,
                                   "mistake": m, "value": val, "severity": sev})
    return idx


def main(args):
    jobdir = Path(args.jobdir).expanduser().resolve()
    vid_path   = jobdir / "src.mp4"
    pose_path  = jobdir / "alphapose" / "alphapose-results.json"
    report_path = jobdir / "analysis" / f"{args.exercise}_analysis.json"
    out_dir    = jobdir / "debug"
    out_dir.mkdir(exist_ok=True)
    out_path   = out_dir / (args.out or "debug_overlay.mp4")

    if not (vid_path.exists() and pose_path.exists() and report_path.exists()):
        sys.exit("✖  Could not locate src.mp4, alphapose-results.json "
                 "or analysis JSON under the given jobdir")

    print("ℹ️  loading keypoints …")
    kps_dict = load_keypoints(pose_path)

    print("ℹ️  loading report …")
    with open(report_path) as f:
        report = json.load(f)
    idx = build_mistake_index(report)

    cap = cv2.VideoCapture(str(vid_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or report.get("fps", 30)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # draw mistakes active at this frame
        active = idx.get(frame_no, [])
        y_ofs = 40
        for a in active:
            mtype = a["type"]
            label = mtype.upper()
            _draw_text(frame, label, (30, y_ofs))
            y_ofs += int(50 * FONT_SCALE)

            # joints
            for jname in MISTAKE_JOINTS.get(mtype, []):
                jidx = NAME2IDX[jname]
                if frame_no in kps_dict:
                    x, y, conf = kps_dict[frame_no][jidx]
                    if conf > 0.05:
                        cv2.circle(frame, (int(x), int(y)),
                                   JOINT_RADIUS, JOINT_COLOR, -1)

        out.write(frame)
        frame_no += 1
        if frame_no % 100 == 0:
            print(f"  processed {frame_no} frames …", end="\r")

    cap.release()
    out.release()
    print(f"\n✅ overlay written to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--jobdir", required=True, help="pipeline job directory")
    p.add_argument("--exercise", default="squat")
    p.add_argument("--out", help="custom output filename (.mp4)")
    main(p.parse_args())
