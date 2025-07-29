# analyzer_functions/context.py
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
import typing as T

@dataclass
class PipelineContext:
    # raw inputs
    json_pose:   Path | None = None
    npy_pose:    Path | None = None

    # intermediate skeletons
    skel_3d:     np.ndarray | None = None
    skel_2d:     np.ndarray | None = None
    skel_3d_fix: np.ndarray | None = None

    # repetition & reference data
    repetitions: pd.DataFrame | None = None
    ref_len_2d:  dict[str, float] = field(default_factory=dict)
    ref_len_3d:  dict[str, float] = field(default_factory=dict)

    # per-module reports (key = module name)
    reports:     dict[str, pd.DataFrame] = field(default_factory=dict)

