#!/usr/bin/env python3
"""
overlay_debug_video.py
-------------------------------------------------------------
Create a “developer overlay” video that shows, for every
detected mistake, a big label *and* red markers on the
relevant joints – **plus the current frame number**.

Usage
-----
python overlay_debug_video.py --jobdir <JOBDIR> [--out overlay.mp4]

<JOBDIR> must contain:
  • src.mp4
  • alphapose/alphapose‑results.json
  • analysis/<exercise>_analysis.json
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ───── tweakables ──────────────────────────────────────────────────
FONT_FACE   = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE  = 1.1
FONT_THICK  = 2
FONT_COLOR  = (255, 255, 255)
OUTLINE_CLR = (0, 0, 0)
JOINT_RAD   = 9
JOINT_CLR   = (0, 0, 255)
FRAME_POS   = (10, 30)          # location of frame counter
# -------------------------------------------------------------------

JOINT_NAMES = ["Nose","LEye","REye","LEar","REar",
               "LShoulder","RShoulder","LElbow","RElbow",
               "LWrist","RWrist","LHip","RHip","LKnee","RKnee",
               "LAnkle","RAnkle","Head","Neck","Hip",
               "LBigToe","RBigToe","LSmallToe","RSmallToe",
               "LHeel","RHeel"]
NAME2IDX = {n:i for i,n in enumerate(JOINT_NAMES)}

MISTAKE_JOINTS = {
    "flean": ["LShoulder","RShoulder","Hip"],
    "hip":   ["LHip","RHip"],
    "fk":    ["LKnee","RKnee"],
    "ffpa_left":  ["LKnee"],
    "ffpa_right": ["RKnee"],
    "depth": ["LHip","RHip","LKnee","RKnee"],
    "feet":  ["LAnkle","RAnkle"],
    "heel":  ["LHeel","RHeel"],
}

FRAMES_BEFORE = 20
FRAMES_AFTER  = 20

# ───── helpers ─────────────────────────────────────────────────────
def _draw_text(img, txt, org):
    x,y = org
    cv2.putText(img, txt, (x,y), FONT_FACE, FONT_SCALE,
                OUTLINE_CLR, FONT_THICK+2, cv2.LINE_AA)
    cv2.putText(img, txt, (x,y), FONT_FACE, FONT_SCALE,
                FONT_COLOR,  FONT_THICK,   cv2.LINE_AA)

def load_keypoints(path: Path) -> dict[int, np.ndarray]:
    data = json.loads(path.read_text())
    frames={}
    for e in data:
        fno = int(Path(e["image_id"]).stem)
        frames[fno] = np.asarray(e["keypoints"], np.float32).reshape(-1,3)
    return frames

def build_index(report: dict) -> dict[int,list[dict]]:
    idx = defaultdict(list)
    for m in report["mistakes"]:
        for key in ["flean","fk","depth",
                    "ffpa_left","ffpa_right",
                    "feet","heel","hip"]:
            if key == "hip" and "hip_frames" in m:
                for rng, sev, val in zip(m["hip_frames"],
                                         m["hip_severity"],
                                         m["hip_value"]):
                    if sev=="none": continue
                    s,e=rng
                    for f in range(s,e+1):
                        idx[f].append({"type":"hip","peak":(s+e)//2})
            else:
                field = f"{key}_frame"
                if field in m and m.get(f"{key}_severity","none")!="none":
                    peak = m[field]
                    for f in range(peak-FRAMES_BEFORE, peak+FRAMES_AFTER+1):
                        idx[f].append({"type":key,"peak":peak})
    return idx

# ───── main ───────────────────────────────────────────────────────
def main(args):
    jobdir   = Path(args.jobdir).expanduser()
    src_mp4  = jobdir/"src.mp4"
    pose_js  = jobdir/"alphapose"/"alphapose-results.json"
    rpt_js   = jobdir/"analysis"/f"{args.exercise}_analysis.json"
    if not (src_mp4.exists() and pose_js.exists() and rpt_js.exists()):
        sys.exit("✖ missing src.mp4 or analysis JSONs under jobdir")

    kps = load_keypoints(pose_js)
    report = json.loads(rpt_js.read_text())
    index  = build_index(report)

    cap = cv2.VideoCapture(str(src_mp4))
    w,h   = int(cap.get(3)), int(cap.get(4))
    fps   = cap.get(cv2.CAP_PROP_FPS) or report.get("fps",30)
    out_p = (jobdir/"debug"/(args.out or "debug_overlay.mp4"))
    out_p.parent.mkdir(exist_ok=True)

    vw = cv2.VideoWriter(str(out_p),
                         cv2.VideoWriter_fourcc(*"mp4v"),
                         fps,(w,h))

    fno=0
    while True:
        ok, frame = cap.read()
        if not ok: break

        # frame number --------------------
        _draw_text(frame, str(fno), FRAME_POS)

        # active mistakes -----------------
        for act in index.get(fno, []):
            mtype = act["type"]
            label = mtype.upper()
            _draw_text(frame, label, (40, 80+30*list(index.get(fno,[])).index(act)))

            for j in MISTAKE_JOINTS.get(mtype, []):
                jidx = NAME2IDX[j]
                if fno in kps:
                    x,y,conf = kps[fno][jidx]
                    if conf>0.05:
                        cv2.circle(frame,(int(x),int(y)),JOINT_RAD,JOINT_CLR,-1)

        vw.write(frame)
        fno+=1
        if fno%200==0: print(f"  processed {fno} frames …",end="\r")

    cap.release(); vw.release()
    print(f"\n✅ overlay written to {out_p}")

if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--jobdir", required=True)
    pa.add_argument("--exercise", default="squat")
    pa.add_argument("--out", help="output filename")
    main(pa.parse_args())
