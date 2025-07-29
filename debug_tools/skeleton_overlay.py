"""
Draw the full AlphaPose skeleton on every frame of `src.mp4`.

CLI
---
$ python skeleton_overlay.py --jobdir <JOBDIR> [--out MY.mp4]

Python
------
from debug_tools.skeleton_overlay import generate_skeleton_video
generate_skeleton_video("/abs/path/to/jobdir")
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import cv2
import numpy as np

# ─── visual parameters ─────────────────────────────────────────────
DOT_R      = 5
DOT_COLOR  = (0, 255, 0)          # joints   – green
LINE_THICK = 2
LINE_COLOR = (0, 200, 255)        # skeleton – orange
FONT       = cv2.FONT_HERSHEY_SIMPLEX
# -------------------------------------------------------------------

# 26‑joint HALPE edges (0‑based indexes)
EDGES = [
    (17, 18), (18, 19),
    (18,  5), ( 5,  7), ( 7,  9),
    (18,  6), ( 6,  8), ( 8, 10),
    (19, 11), (11, 13), (13, 15), (15, 20),
    (19, 12), (12, 14), (14, 16), (16, 21),
    (15, 24), (16, 25),
]

# ─── helpers ───────────────────────────────────────────────────────
def _load_keypoints(json_path: Path) -> dict[int, np.ndarray]:
    """Return {frame_number: (26,3) array}."""
    with open(json_path) as f:
        raw = json.load(f)
    out: dict[int, np.ndarray] = {}
    for d in raw:
        fno = int(Path(d["image_id"]).stem)
        out[fno] = np.array(d["keypoints"], np.float32).reshape(-1, 3)
    return out


def _draw(frame: np.ndarray, kps: np.ndarray):
    """Overlay joints and limbs on `frame` (in‑place)."""
    # dots
    for x, y, conf in kps:
        if conf < 0.05:
            continue
        cv2.circle(frame, (int(x), int(y)), DOT_R, DOT_COLOR, -1)

    # lines
    for a, b in EDGES:
        if kps[a, 2] < .05 or kps[b, 2] < .05:
            continue
        ax, ay = map(int, kps[a, :2])
        bx, by = map(int, kps[b, :2])
        cv2.line(frame, (ax, ay), (bx, by), LINE_COLOR, LINE_THICK)

# ─── public API ────────────────────────────────────────────────────
def generate_skeleton_video(jobdir: str | Path,
                             out_name: str = "overlay_skeleton.mp4") -> Path:
    """Create the overlay video and return its Path."""
    jobdir = Path(jobdir).expanduser().resolve()
    src  = jobdir / "src.mp4"
    json_path = jobdir / "alphapose" / "alphapose-results.json"
    if not (src.exists() and json_path.exists()):
        raise FileNotFoundError("Expecting src.mp4 and alphapose-results.json in jobdir")

    kps_by_frame = _load_keypoints(json_path)

    cap   = cv2.VideoCapture(str(src))
    w, h  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    fourc = cv2.VideoWriter_fourcc(*"mp4v")

    dbg_dir  = jobdir / "debug"
    dbg_dir.mkdir(exist_ok=True)
    out_mp4  = dbg_dir / out_name
    writer   = cv2.VideoWriter(str(out_mp4), fourc, fps, (w, h))

    fno = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fno in kps_by_frame:
            _draw(frame, kps_by_frame[fno])
        cv2.putText(frame, f"frame {fno}", (10, 28), FONT, 0.7, (255,255,255), 1, cv2.LINE_AA)
        writer.write(frame)
        fno += 1

    cap.release(); writer.release()
    return out_mp4


# ─── CLI wrapper ───────────────────────────────────────────────────
def _main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobdir", required=True, help="pipeline job directory")
    p.add_argument("--out", help="custom name (default overlay_skeleton.mp4)")
    opts = p.parse_args()
    path = generate_skeleton_video(opts.jobdir, opts.out or "overlay_skeleton.mp4")
    print("✅ skeleton overlay written to", path)

if __name__ == "__main__":
    _main()
