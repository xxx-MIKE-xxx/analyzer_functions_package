#!/usr/bin/env python3
# hip_path_feet_head_overlay.py

import os, json, cv2, math, numpy as np, pandas as pd

# ------------------------------------------------------------------ user-params
KEYPOINTS_NPY = "imputed_ma.npy"
REPS_CSV      = "repetition_data.csv"
LENGTHS_JSON  = "reference_lengths_2d.json"
VIDEO_IN      = None
VIDEO_OUT     = "hip_path_feet_head_overlay.avi"
FPS           = 5
FRAME_SIZE    = (2000, 2000)
CONF_TH       = 0.0

RETURN_TOL_Y  = 6        # hip-height return tolerance
LOOK_BACK     = 10       # frames before nominal end to look for return

# foot thresholds
OUT_MILD, OUT_SEV = 35, 45
IN_MILD,  IN_SEV  = -10, -20

# head thresholds
PITCH_UP_MILD, PITCH_UP_SEV = 15, 25
PITCH_DN_MILD, PITCH_DN_SEV = -10, -20
YAW_RT_MILD,   YAW_RT_SEV   = 15, 25
YAW_LT_MILD,   YAW_LT_SEV   = -15, -25

# temporary global heading (deg)
GLOBAL_HEADING = 0.0

# ------------------------------------------------------------------ HALPE-26 indices
NOSE            = 0
L_SHO, R_SHO    = 5, 6
L_HIP, R_HIP    = 11, 12
L_KNEE, R_KNEE  = 13, 14
L_ANK,  R_ANK   = 15, 16
L_BIG,  R_BIG   = 20, 21
L_SML,  R_SML   = 22, 23

# ------------------------------------------------------------------ leg length (for hip-shift %)
with open(LENGTHS_JSON) as f:
    raw = json.load(f)
def _len(i, j):
    return raw.get(f"({i}, {j})") or raw.get(f"({j}, {i})") or 0.0

LEG_LEN = (_len(L_HIP, L_KNEE) + _len(R_HIP, R_KNEE) +
           _len(L_KNEE, L_ANK) + _len(R_KNEE, R_ANK)) / 4 or 1
print(f"[info] leg_len ≈ {LEG_LEN:.1f}px")

# ------------------------------------------------------------------ skeleton links
LINKS = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (15,20),(15,22),(16,21),(16,23),
]

def draw_skel(img, kp):
    for i, j in LINKS:
        xi, yi, ci = kp[i]
        xj, yj, cj = kp[j]
        if ci >= CONF_TH and cj >= CONF_TH and not np.isnan([xi, yi, xj, yj]).any():
            cv2.line(img, (int(xi), int(yi)), (int(xj), int(yj)), (255, 0, 0), 1)
    for x, y, c in kp:
        if c >= CONF_TH and not np.isnan([x, y]).any():
            cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)

# ------------------------------------------------------------------ helpers
def signed_dist(pt, a, b):
    px, py = pt
    x1, y1 = a
    x2, y2 = b
    nx, ny = (y2 - y1), -(x2 - x1)
    norm = math.hypot(nx, ny) or 1
    return ((px - x1) * nx + (py - y1) * ny) / norm

def foot_angle(ank, big, sml):
    mid = (big + sml) * 0.5
    vec = mid - ank
    return math.degrees(math.atan2(vec[1], vec[0]))

def foot_flag(a):
    if a >= OUT_SEV or a <= IN_SEV:
        return "⚠︎"
    if a >= OUT_MILD or a <= IN_MILD:
        return "!"
    return ""

def head_sev_pitch(p):
    if math.isnan(p):
        return "0"
    if p >= PITCH_UP_SEV:
        return "+2"
    if p >= PITCH_UP_MILD:
        return "+1"
    if p <= PITCH_DN_SEV:
        return "-2"
    if p <= PITCH_DN_MILD:
        return "-1"
    return "0"

def head_sev_yaw(y):
    if math.isnan(y):
        return "0"
    if y >= YAW_RT_SEV:
        return "+2"
    if y >= YAW_RT_MILD:
        return "+1"
    if y <= YAW_LT_SEV:
        return "-2"
    if y <= YAW_LT_MILD:
        return "-1"
    return "0"

def head_angles(kp, heading_deg):
    # returns (pitch, yaw)
    if np.isnan(kp[[NOSE, L_SHO, R_SHO], :2]).any():
        return float("nan"), float("nan")
    nose = kp[NOSE, :2]
    sho_mid = (kp[L_SHO, :2] + kp[R_SHO, :2]) * 0.5
    vec = nose - sho_mid
    # pitch: up positive (deviation from vertical)
    pitch = -math.degrees(math.atan2(vec[1], abs(vec[0])))
    # yaw: horizontal, corrected by heading
    raw_yaw = math.degrees(math.atan2(vec[1], vec[0]))
    yaw = raw_yaw - heading_deg
    # normalize to [-180,180]
    yaw = (yaw + 180) % 360 - 180
    return pitch, yaw

# ------------------------------------------------------------------ load data
kps      = np.load(KEYPOINTS_NPY)
reps     = pd.read_csv(REPS_CSV)
hip_x    = (kps[:, L_HIP, 0] + kps[:, R_HIP, 0]) * 0.5
hip_y    = (kps[:, L_HIP, 1] + kps[:, R_HIP, 1]) * 0.5
rep_bounds = {int(r.start): (int(r.rep_id), int(r.end)) for _, r in reps.iterrows()}

# ------------------------------------------------------------------ video output
W, H = FRAME_SIZE
out = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
cap = cv2.VideoCapture(VIDEO_IN) if VIDEO_IN else None

# state for hip-path
path_pts = []
ideal_A = ideal_B = None
start_y = None
nom_end = None
peak = {"left": None, "right": None}

for f in range(kps.shape[0]):
    # new rep?
    if f in rep_bounds:
        rid, nom_end = rep_bounds[f]
        ax = (kps[f, L_ANK, 0] + kps[f, R_ANK, 0]) * 0.5
        ay = (kps[f, L_ANK, 1] + kps[f, R_ANK, 1]) * 0.5
        ideal_A, ideal_B = (ax, ay), (hip_x[f], hip_y[f])
        start_y = hip_y[f]
        path_pts.clear()
        peak = {"left": None, "right": None}

    # grab frame
    frame = np.zeros((H, W, 3), np.uint8)
    if cap and cap.isOpened():
        ret, src = cap.read()
        if ret:
            frame = cv2.resize(src, (W, H))

    draw_skel(frame, kps[f])

    # ----- hip path -----
    x, y = hip_x[f], hip_y[f]
    if ideal_A and not np.isnan(x + y):
        path_pts.append((int(x), int(y)))
        dev = signed_dist((x, y), ideal_A, ideal_B)
        side = "left" if dev < 0 else "right"
        if peak[side] is None or abs(dev) > abs(peak[side][1]):
            peak[side] = (f, dev)

    for i in range(1, len(path_pts)):
        cv2.line(frame, path_pts[i-1], path_pts[i], (0, 255, 255), 2)
    if ideal_A:
        cv2.line(frame,
                 (int(ideal_A[0]), int(ideal_A[1])),
                 (int(ideal_B[0]), int(ideal_B[1])),
                 (100, 100, 255), 1)

    # ----- feet -----
    for lab, ank_i, big_i, sml_i, col in [
        ("L", L_ANK, L_BIG, L_SML, (255, 255, 0)),
        ("R", R_ANK, R_BIG, R_SML, (0, 255, 255))
    ]:
        if np.isnan(kps[f, [ank_i, big_i, sml_i], :2]).any():
            continue
        ank = kps[f, ank_i, :2]
        big = kps[f, big_i, :2]
        sml = kps[f, sml_i, :2]
        ang = foot_angle(ank, big, sml)
        mid = (big + sml) * 0.5
        cv2.line(frame,
                 (int(ank[0]), int(ank[1])),
                 (int(mid[0]), int(mid[1])),
                 col, 2)
        cv2.putText(frame,
                    f"{ang:+.1f}°{foot_flag(ang)}",
                    (int(mid[0]) + 5, int(mid[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    col,
                    2)

    # ----- head orientation -----
    pitch, yaw = head_angles(kps[f], GLOBAL_HEADING)
    if not math.isnan(pitch):
        nose   = kps[f, NOSE, :2]
        sho_mid = (kps[f, L_SHO, :2] + kps[f, R_SHO, :2]) * 0.5
        # arrow from shoulder to nose
        cv2.arrowedLine(frame,
                        (int(sho_mid[0]), int(sho_mid[1])),
                        (int(nose[0]), int(nose[1])),
                        (0, 128, 255),
                        2,
                        tipLength=0.15)
        cv2.putText(frame,
                    f"P{pitch:+.1f}°[{head_sev_pitch(pitch)}]  Y{yaw:+.1f}°[{head_sev_yaw(yaw)}]",
                    (int(nose[0]) + 15, int(nose[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 128, 255),
                    2)

    # ----- rep end check -----
    end_now = False
    if ideal_A:
        if f >= nom_end - LOOK_BACK and abs(hip_y[f] - start_y) < RETURN_TOL_Y:
            end_now = True
        elif f == nom_end:
            end_now = True
    if end_now:
        for side, info in peak.items():
            if info is None:
                continue
            pf, dv = info
            px, py = int(hip_x[pf]), int(hip_y[pf])
            ratio = abs(dv) / LEG_LEN
            cv2.circle(frame, (px, py), 6, (0, 0, 255), -1)
            cv2.putText(frame,
                        f"{ratio*100:.1f}% {side}",
                        (px + 10, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2)
        ideal_A = ideal_B = None
        path_pts.clear()
        peak = {"left": None, "right": None}

    # HUD
    cv2.putText(frame,
                f"Frame {f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)

    out.write(frame)

out.release()
if cap:
    cap.release()

print("✅ saved →", os.path.abspath(VIDEO_OUT))
