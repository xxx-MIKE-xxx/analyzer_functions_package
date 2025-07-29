#!/usr/bin/env python3
import numpy as np
import pandas as pd
import itertools
from pathlib import Path
from typing import Union
import tempfile

# --------------------------------------------------------------------------- #
# basic helpers (unchanged)
# --------------------------------------------------------------------------- #
def ma_impute(arr: np.ndarray, mask: np.ndarray, window: int) -> np.ndarray:
    """Fill NaNs on bad frames with a centred moving-average (x & y only)."""
    F, K, _ = arr.shape
    out = arr.copy().astype(float)
    for kp in range(K):
        for coord in (0, 1):
            s = pd.Series(out[:, kp, coord])
            ma = s.rolling(window, center=True, min_periods=1).mean()
            fill = mask & s.isna()
            out[fill, kp, coord] = ma[fill]
    return out

def choose_window(mask: np.ndarray) -> int:
    """Pick 3/5/7 based on good-ratio and longest bad-streak."""
    good_ratio = 1.0 - mask.mean()
    streaks = [sum(1 for _ in g) for v, g in itertools.groupby(mask) if v]
    max_streak = max(streaks) if streaks else 0
    return 3 if (good_ratio > 0.90 and max_streak <= 1) else 5 if max_streak <= 2 else 7

# --------------------------------------------------------------------------- #
# core imputer (minimal change -- now *optionally* saves cleaned version too)
# --------------------------------------------------------------------------- #
def _impute(
    sk: np.ndarray,
    mask: np.ndarray,
    window: Union[int, str] = "auto",
    clean_out: str | Path | None = None
) -> np.ndarray:
    """Return imputed skeletons; optionally write the NaN-cleaned array."""
    if sk.ndim != 3 or sk.shape[2] < 2:
        raise ValueError("Expected skeletons with shape (F, K, ≥2)")

    cleaned = sk.copy().astype(float)
    cleaned[mask] = np.nan

    if clean_out is not None:
        np.save(clean_out, cleaned)
        print(f"✅ Saved cleaned skeletons → {clean_out}")

    win = choose_window(mask) if window == "auto" else int(window)
    if win < 1 or win % 2 == 0:
        raise ValueError("window must be a positive odd integer")

    return ma_impute(cleaned, mask, win)

# --------------------------------------------------------------------------- #
# PUBLIC PIPELINE
# --------------------------------------------------------------------------- #
def pipeline(
    skeletons: str | np.ndarray,
    df_bad: str | Path | pd.DataFrame,
    *,
    window: int | str = "auto",
    clean_out_path: str | Path | None = None,
    imputed_out_path: str | Path | None = None
) -> np.ndarray:
    """
    End-to-end helper: take raw skeletons + df_bad, produce imputed array.

    Parameters
    ----------
    skeletons        : .npy path or ndarray (F,K,3)
    df_bad           : CSV path or DataFrame with ['frame','is_bad']
    window           : odd window length or 'auto' (default)
    clean_out_path   : if given, save NaN-cleaned array here (optional)
    imputed_out_path : if given, save final imputed array here (optional)

    Returns
    -------
    imputed_arr      : np.ndarray (F,K,3) – ready for downstream use
    """
    # 1️⃣  load skeleton data --------------------------------------------------
    if isinstance(skeletons, (str, Path)):
        sk_arr = np.load(skeletons)
    else:
        sk_arr = skeletons

    # 2️⃣  load / normalise df_bad --------------------------------------------
    if isinstance(df_bad, pd.DataFrame):
        bad_df = df_bad.copy()
    else:
        bad_df = pd.read_csv(df_bad)

    if not {"frame", "is_bad"}.issubset(bad_df.columns):
        raise ValueError("df_bad must have columns ['frame','is_bad']")

    mask = bad_df.sort_values("frame")["is_bad"].to_numpy(bool)

    # 3️⃣  impute --------------------------------------------------------------
    imputed = _impute(
        sk_arr,
        mask=mask,
        window=window,
        clean_out=clean_out_path
    )

    # 4️⃣  optional write-out --------------------------------------------------
    if imputed_out_path is not None:
        np.save(imputed_out_path, imputed)
        print(f"✅ Saved imputed array → {imputed_out_path}")

    return imputed

# --------------------------------------------------------------------------- #
# example run (remove or keep for CLI testing)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # fake example
    F, K = 100, 26
    raw   = np.random.rand(F, K, 3).astype(np.float32)
    mask  = np.random.rand(F) > 0.85
    df_bad_example = pd.DataFrame({"frame": np.arange(F), "is_bad": mask})

    # run pipeline entirely in RAM
    imputed = pipeline(
        skeletons=raw,
        df_bad=df_bad_example,
        window="auto",
        clean_out_path=None,
        imputed_out_path=None
    )
    print("Imputed shape:", imputed.shape)
