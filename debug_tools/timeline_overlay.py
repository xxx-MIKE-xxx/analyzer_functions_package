#!/usr/bin/env python3
"""
Timeline‑overlay renderer for quick visual triage ⏱️
===================================================
Given a *raw video* and the JSON report produced by one of the
exercise drivers, this utility generates <video‑stem>_debug.mp4
with an extra bar at the bottom:

* full grey line —— represents the entire video duration
* red segments  —— frames (or frame ranges) where at least one mistake
                     was detected
* moving cyan dot —— current frame while the video plays
* white number  —— current absolute frame index (top‑left corner)

Typical use
-----------
python -m debug_tools.timeline_overlay \
       --video  ~/jobs/001/src.mp4 \
       --report ~/jobs/001/analysis/squat_analysis.json
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

import json, ast
from typing import List, Tuple, Any

def _to_pairs(obj: Any) -> List[Tuple[int, int]]:
    """
    Normalise *obj* into a list of (start,end) integer tuples.

    Accepts:
      • int                       → [(i,i)]
      • [start,end]               → [(start,end)]
      • [[s,e],[s2,e2], …]        → […]
      • "[-1, -1]"  (string repr) → []   (ignored, sentinel for “no event”)
    """
    if obj is None:
        return []

    # 1. strings like "[-1, -1]"  or  "[[12,15],[20,22]]"
    if isinstance(obj, str):
        try:
            obj = ast.literal_eval(obj)
        except (ValueError, SyntaxError):
            return []

    # 2. single int  →  one‑frame event
    if isinstance(obj, (int, float)):
        i = int(obj)
        return [] if i < 0 else [(i, i)]

    # 3. flat 2‑item list  → one span
    if (isinstance(obj, list) and len(obj) == 2
            and all(isinstance(x, (int, float)) for x in obj)):
        a, b = map(int, obj)
        return [] if a < 0 or b < 0 else [(a, b)]

    # 4. list of lists
    if (isinstance(obj, list) and obj and isinstance(obj[0], list)):
        spans = []
        for pair in obj:
            if len(pair) == 2:
                a, b = map(int, pair)
                if a >= 0 and b >= 0:
                    spans.append((a, b))
        return spans

    return []          # anything else → ignore


def _extract_events(report_obj) -> List[Tuple[int, int]]:
    """
    Return **clean, unique** (start,end) tuples no matter how the report encodes
    them.  Works for the sample you pasted and older schemas.
    """
    if isinstance(report_obj, dict):
        mistakes = report_obj.get("mistakes", [])
    else:
        mistakes = report_obj               # already list‑like

    events: list[Tuple[int, int]] = []

    for m in mistakes:
        # legacy keys --------------------------------------------------------
        events += _to_pairs(m.get("frame"))
        events += _to_pairs(m.get("frames"))

        # new per‑mistake keys ----------------------------------------------
        for k, v in m.items():
            if k.endswith("_frame") and not k.endswith("_frames"):
                events += _to_pairs(v)
            elif k.endswith("_frames"):
                events += _to_pairs(v)

    # deduplicate & sort
    events = sorted(set(events), key=lambda t: (t[0], t[1]))
    return events


# ───────────────────────── drawing helpers ──────────────────────────
def _draw_base_timeline(width: int, height: int,
                        events: List[Tuple[int, int]],
                        total_frames: int) -> np.ndarray:
    """Pre‑render the grey bar + red error segments (no moving dot)."""
    base = np.full((height, width, 3), 200, np.uint8)         # light grey
    bar_y = height // 2
    cv2.line(base, (0, bar_y), (width - 1, bar_y), (128, 128, 128), 2)

    for start, end in events:
        x1 = int(start / total_frames * width)
        x2 = int(end   / total_frames * width)
        cv2.line(base, (x1, bar_y), (x2, bar_y), (0, 0, 255), 4)  # red span
    return base


# ───────────────────────── main worker ──────────────────────────────
def render_overlay(video_path : Path,
                   report_path: Path,
                   out_path   : Path | None = None,
                   overlay_h  : int  = 60,
                   font_scale : float = 1.0) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps        = cap.get(cv2.CAP_PROP_FPS) or 30
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── parse report ───────────────────────────────────────────────
    with open(report_path) as f:
        data = json.load(f)
    events = _extract_events(data)

    # ── prepare static overlay part ────────────────────────────────
    base_overlay = _draw_base_timeline(width, overlay_h, events, tot_frames)

    out_path = out_path or video_path.with_name(video_path.stem + "_debug.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(out_path), fourcc, fps,
                               (width, height + overlay_h))

    txt_pos   = (10, 30)      # x,y of frame counter
    txt_color = (255, 255, 255)
    txt_thick = 2
    frame_idx = 0
    dot_r     = 6
    dot_col   = (255, 255, 0)  # cyan

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        overlay = base_overlay.copy()

        # moving dot
        x = int(frame_idx / tot_frames * width)
        cv2.circle(overlay, (x, overlay_h // 2), dot_r, dot_col, -1)

        # frame number
        cv2.putText(frame, f"{frame_idx}", txt_pos, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, txt_color, txt_thick, cv2.LINE_AA)

        writer.write(np.vstack([frame, overlay]))
        frame_idx += 1

    writer.release()
    cap.release()
    return out_path


# ───────────────────────── CLI glue ─────────────────────────────────
def _cli() -> None:
    ap = argparse.ArgumentParser(description="Render timeline‑overlay video")
    ap.add_argument("--video",  required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--out",    type=Path)
    ap.add_argument("--height", type=int, default=60,
                    help="overlay bar height in pixels (default 60)")
    ap.add_argument("--font",   type=float, default=1.0,
                    help="scale of frame‑number font (default 1.0)")
    args = ap.parse_args()

    try:
        out = render_overlay(args.video, args.report, args.out,
                             args.height, args.font)
        print(f"✅ timeline video written to {out}")
    except Exception as e:
        sys.exit(f"✖ {e}")

if __name__ == "__main__":
    _cli()
