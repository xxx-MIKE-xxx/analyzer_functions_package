#!/usr/bin/env python3
import numpy as np
import joblib
import argparse
from pathlib import Path
from typing import Any
import tempfile
import pandas as pd

# --------------------------------------------------- 26-joint bone list ----
CONNECTIONS_26 = [
    (17, 0), (18, 17), (18, 5), (18, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (19, 11), (19, 12),
    (11, 13), (13, 15), (15, 20), (20, 22), (22, 24),
    (12, 14), (14, 16), (16, 21), (21, 23), (23, 25)
]

def compute_features(skeletons_path, model_path, kernel_size=3):
    """
    Load skeleton data, compute features, load the model, and predict the bad frames.
    Returns the boolean mask of bad frames.
    """
    data = np.load(skeletons_path)  # (F, K, 3)
    F, K, C = data.shape
    if C < 3:
        raise ValueError("Expected at least 3 channels (x, y, conf)")

    # Compute features (returns (F, 20))
    X = []
    for fr in data:
        vec = []
        for i, j in CONNECTIONS_26:
            p, q = fr[i, :2], fr[j, :2]
            vec.append(0. if (np.isnan(p).any() or np.isnan(q).any())
                        else np.linalg.norm(p - q))
        X.append(vec)
    X = np.asarray(X, dtype=float)
    print(f"Computed feature matrix shape: {X.shape}")

    # Load classifier
    clf = joblib.load(model_path)
    print(f"Model loaded successfully from {model_path}")

    # Predict
    y_pred = clf.predict(X)  # array of 0/1, shape (F,)
    mask = y_pred.astype(bool)

    print(f"Detected {mask.sum()} bad frames out of {F}")
    return mask

def run_pipeline(skeletons_path, model_path, kernel_size=3):
    """
    Wrapper function to call compute_features.
    """
    mask = compute_features(skeletons_path, model_path, kernel_size)
    return mask

# --------------------------------------------------------------------------
# NEW: in-memory pipeline → returns a DataFrame of frame vs is_bad
# --------------------------------------------------------------------------
def pipeline(
    skeletons: str | np.ndarray,
    model:     str | Any,
    *,
    kernel_size: int = 3,
    write_csv:   str | Path | None = None
) -> pd.DataFrame:
    """
    Load or accept in-memory skeletons & model, run detection, and return a DataFrame.

    Parameters
    ----------
    skeletons   : path to .npy file or ndarray (F, K, 3)
    model       : path to .joblib RandomForest or already-loaded estimator
    kernel_size : median filter size (unused here, kept for signature)
    write_csv   : if given, path to write the DataFrame out as CSV

    Returns
    -------
    DataFrame with columns ['frame', 'is_bad']
    """
    # 1) load skeletons array if needed
    if isinstance(skeletons, str):
        kps = np.load(skeletons)
    else:
        kps = skeletons

    # 2) if model is in-memory, dump it to a temp file
    if not isinstance(model, str):
        tmp = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
        joblib.dump(model, tmp.name)
        model_path = tmp.name
    else:
        model_path = model

    # 3) run detection (skeletons_path needs to be a path)
    #    if kps is ndarray, save to temp .npy
    if isinstance(kps, np.ndarray):
        tmp_skel = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
        np.save(tmp_skel.name, kps)
        skel_path = tmp_skel.name
    else:
        skel_path = kps

    mask = compute_features(
        skeletons_path=skel_path,
        model_path=model_path,
        kernel_size=kernel_size
    )

    # 4) clean up temp files
    if isinstance(kps, np.ndarray):
        Path(skel_path).unlink()
    if not isinstance(model, str):
        Path(model_path).unlink()

    # 5) build DataFrame
    df = pd.DataFrame({
        "frame":  np.arange(mask.shape[0]),
        "is_bad": mask
    })

    # 6) optionally write CSV
    if write_csv:
        df.to_csv(write_csv, index=False)
        print(f"✅ Written bad-frame report to {write_csv}")

    return df

# --------------------------------------------------------------------------
# CLI entry-point unchanged
# --------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Detect bad frames from skeletons using a RandomForest model"
    )
    parser.add_argument('--skeletons', required=True,
                        help='Path to skeleton .npy (F×K×3)')
    parser.add_argument('--model',     required=True,
                        help='Path to trained RandomForest joblib file')
    args = parser.parse_args()

    mask = run_pipeline(args.skeletons, args.model)
    print(f"Returned mask with {mask.sum()} bad frames")
