"""
Public surface for the *squat* exercise.

Exports
-------
run_pipeline                 – high‑level end‑to‑end pipeline
parse_frame_spec, …, helpers – individual utilities that other
                               exercises or notebooks may reuse
"""

# ── High‑level entry point ────────────────────────────────────────
from .driver import run_pipeline          # ⬅️  points to driver.py (new)

# ── Low‑level helpers you may want to reuse elsewhere ────────────
from .compute_reference_lengths import (
    parse_frame_spec,
    compute_lengths,
    enforce_symmetry,
)
from .skeleton_length_fix import correct_skeleton_rigid
from .heel_raise_analysis_26 import (
    analyze_heel_raise_report as analyze_heel_raise_detailed,
)

__all__: list[str] = [
    # high‑level
    "run_pipeline",
    # helpers
    "parse_frame_spec",
    "compute_lengths",
    "enforce_symmetry",
    "correct_skeleton_rigid",
    "analyze_heel_raise_detailed",
]
