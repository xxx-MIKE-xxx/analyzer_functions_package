#!/usr/bin/env python3
"""
Console entry point that keeps the classic **full_video_analysis** command alive
after the July‑2025 repo refactor.

Usage
-----
    full_video_analysis <video> [--jobdir DIR] [--exercise squat] [--no-copy]

• *video*      – local path **or** s3://bucket/key.mp4  
• *--exercise* – squat (default), lunge, …  
• *--jobdir*   – scratch directory where all artefacts are stored  
• *--no-copy*  – assume <jobdir>/src.mp4 already exists (skip download / copy)

The script does nothing more than:

1. Parse CLI arguments.
2. Dynamically import the proper exercise driver  
   (`analyzer_pkg.exercises.<exercise>.driver`).
3. Delegate to `driver.run(video, jobdir=None | Path, no_copy=False)`.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import List, Optional


# ────────────────────────── helpers ────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="full_video_analysis")
    p.add_argument("video", help="Local video file or s3://bucket/key.mp4")
    p.add_argument("--exercise", default="squat",
                   help="Exercise name (default: squat)")
    p.add_argument("--jobdir",
                   help="Directory for all artefacts "
                        "(defaults to /tmp/video_pose_<timestamp>)")
    p.add_argument("--no-copy", action="store_true",
                   help="Assume <jobdir>/src.mp4 already exists "
                        "– don't copy / download")
    return p


# ────────────────────────── main entry ─────────────────────────
def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    # 1 – resolve driver module (e.g. analyzer_pkg.exercises.squat.driver)
    mod_name = f"analyzer_pkg.exercises.{args.exercise}.driver"
    try:
        driver = importlib.import_module(mod_name)
    except ModuleNotFoundError:
        sys.exit(f"❌  unknown exercise ‘{args.exercise}’ "
                 f"(no module {mod_name!r})")

    # 2 – dispatch
    if not hasattr(driver, "run"):
        sys.exit(f"❌  driver {mod_name} has no callable ‘run’")

    driver.run(
        video=args.video,
        jobdir=Path(args.jobdir).expanduser() if args.jobdir else None,
        no_copy=args.no_copy,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
