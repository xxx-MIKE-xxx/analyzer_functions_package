#!/usr/bin/env python3
# hip_path_overlay.py  – visualise hip‑midpoint left/right shifts with dynamic rep end

import os, json, cv2, math, numpy as np, pandas as pd

# ------------------------------------------------------------------ parameters
KEYPOINTS_NPY = "imputed_ma.npy"
REPS_CSV      = "repetition_data.csv"
LENGTHS_JSON  = "reference_lengths_2d.json"
VIDEO_IN      = None
VIDEO_OUT     = "hip_path_overlay.avi"
FPS           = 5
FRAME_SIZE    = (2000, 2000)   # (W, H)
CONF_TH       = 0.0
RETURN_TOL_Y  = 6               # px tolerance: hip y back to start
LOOK_BACK     = 10              # frames before nominal end to start looking

# ------------------------------------------------------------------ HALPE‑26 indices
L_HIP,R_HIP = 11,12
L_KNEE,R_KNEE = 13,14
L_ANK,R_ANK   = 15,16

# ------------------------------------------------------------------ leg length & severity
with open(LENGTHS_JSON) as f: raw=json.load(f)
get_len=lambda i,j: raw.get(f"({i}, {j})") or raw.get(f"({j}, {i})") or 0.0
LEG_LEN=sum([get_len(L_HIP,L_KNEE),get_len(R_HIP,R_KNEE),get_len(L_KNEE,L_ANK),get_len(R_KNEE,R_ANK)])/4 or 1
print(f"[info] normalising by leg_len ≈ {LEG_LEN:.1f}px")

def sev(r:float)->str:
    if math.isnan(r): return "unknown"
    if r>0.08: return "severe"
    if r>0.04: return "mild"
    return "none"

# ------------------------------------------------------------------ drawing helpers
LINKS=[(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
       (11,12),(11,13),(13,15),(12,14),(14,16),(15,20),(15,22),(16,21),(16,23),
       (20,22),(21,23),(24,15),(25,16)]

def draw_skel(img,kp):
    for (i,j) in LINKS:
        xi,yi,ci = kp[i]; xj,yj,cj = kp[j]
        if ci>=CONF_TH and cj>=CONF_TH and not np.isnan([xi,yi,xj,yj]).any():
            #cv2.line(img,(int(xi),int(yi)),(int(xj),int(yj)),(255,0,0),1)
    for x,y,c in kp:
        if c>=CONF_TH and not np.isnan([x,y]).any():
            #cv2.circle(img,(int(x),int(y)),2,(0,255,0),-1)

# ------------------------------------------------------------------ signed distance helper

def signed_dist(pt,a,b):
    px,py=pt; x1,y1=a; x2,y2=b
    nx,ny = (y2-y1), -(x2-x1)
    norm=math.hypot(nx,ny) or 1
    return ((px-x1)*nx + (py-y1)*ny)/norm

# ------------------------------------------------------------------ data
kps=np.load(KEYPOINTS_NPY)
reps=pd.read_csv(REPS_CSV)
hip_x=(kps[:,L_HIP,0]+kps[:,R_HIP,0])/2
hip_y=(kps[:,L_HIP,1]+kps[:,R_HIP,1])/2
rep_start={int(r.start):(int(r.rep_id),int(r.end)) for _,r in reps.iterrows()}

# ------------------------------------------------------------------ video IO
W,H=FRAME_SIZE
#out=cv2.VideoWriter(VIDEO_OUT,cv2.VideoWriter_fourcc(*"MJPG"),FPS,(W,H))
#cap=cv2.VideoCapture(VIDEO_IN) if VIDEO_IN else None

# state vars
path_pts=[]
ideal_A=ideal_B=None
start_y=None
nom_end=None
peak={'left':None,'right':None}

for f in range(kps.shape[0]):
    # --- new rep start
    if f in rep_start:
        rid, nom_end = rep_start[f]
        ax=(kps[f,L_ANK,0]+kps[f,R_ANK,0])/2; ay=(kps[f,L_ANK,1]+kps[f,R_ANK,1])/2
        bx,by=hip_x[f],hip_y[f]
        ideal_A,ideal_B=(ax,ay),(bx,by)
        start_y=by; path_pts.clear(); peak={'left':None,'right':None}
    frame=np.zeros((H,W,3),np.uint8)
    if cap and cap.isOpened():
        ret,src=cap.read()
        if ret: frame=cv2.resize(src,(W,H))
    draw_skel(frame,kps[f])

    x=hip_x[f]; y=hip_y[f]
    if ideal_A and not np.isnan(x+y):
        # accumulate path
        path_pts.append((int(x),int(y)))
        dev=signed_dist((x,y),ideal_A,ideal_B)
        side='left' if dev<0 else 'right'
        if peak[side] is None or abs(dev)>abs(peak[side][1]):
            peak[side]=(f,dev)

    # draw path & ideal line
    for i in range(1,len(path_pts)):
        #cv2.line(frame,path_pts[i-1],path_pts[i],(0,255,255),2)
    if ideal_A:
        #cv2.line(frame,(int(ideal_A[0]),int(ideal_A[1])),(int(ideal_B[0]),int(ideal_B[1])),(100,100,255),1)

    # --- dynamic rep end (return to start height OR nominal end)
    end_now=False
    if ideal_A:
        if f>=nom_end-LOOK_BACK and abs(y-start_y)<RETURN_TOL_Y:
            end_now=True
        elif f==nom_end:
            end_now=True
    if end_now:
        # annotate peaks
        for side,info in peak.items():
            if info is None: continue
            pf,dv=info
            px,py=int(hip_x[pf]),int(hip_y[pf])
            ratio=abs(dv)/LEG_LEN
            #cv2.circle(frame,(px,py),6,(0,0,255),-1)
            #cv2.putText(frame,f"{ratio*100:.1f}% {side} {sev(ratio)}",
                        (px+10,py-10),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,255),2)
        # reset
        ideal_A=ideal_B=start_y=nom_end=None; path_pts.clear(); peak={'left':None,'right':None}

    #cv2.putText(frame,f"Frame {f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    out.write(frame)

out.release()
if cap: cap.release()
print("✅ saved →",os.path.abspath(VIDEO_OUT))