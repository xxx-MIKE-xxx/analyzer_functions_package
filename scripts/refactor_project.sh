#!/usr/bin/env bash
###############################################################################
#  refactor_project.sh
#  Usage:  bash refactor_project.sh
#
#  1. Creates a clean “src/”‑based layout
#  2. Moves your real source code under that tree
#  3. Deletes duplicate folders, build artefacts, __pycache__, *.pyc, etc.
#  4. Commits the result on a new Git branch "refactor-layout"
###############################################################################
set -euo pipefail

echo "🔄  Preparing Git branch …"
git init &>/dev/null || true                # in case repo wasn’t initialised
git add .                                   # stage everything that exists now
git commit -m "checkpoint: pre‑refactor" || true
git switch -c refactor-layout

echo "📂  Creating new layout under src/ …"
mkdir -p src/analyzer_pkg/{analysis,cli,core,data,models,utils}

echo "🚚  Moving primary modules …"
# analysis algorithms
git mv analyzer_pkg/analyzer_functions/*.py              src/analyzer_pkg/analysis/     || true
git mv folder/analyzer_functions/*.py                    src/analyzer_pkg/analysis/     || true
# helper libs
git mv analyzer_pkg/helper_functions.py                  src/analyzer_pkg/utils/__init__.py  || true
git mv folder/helper_functions.py                        src/analyzer_pkg/utils/__init__.py  || true
# CLI entry point
git mv analyzer_pkg/bin/full_video_analysis.py           src/analyzer_pkg/cli/main.py   || true
git mv folder/bin/full_video_analysis.py                 src/analyzer_pkg/cli/main.py   || true
# shared data / models
git mv analyzer_pkg/analyzer_functions/data              src/analyzer_pkg/data           || true
git mv folder/analyzer_functions/data                    src/analyzer_pkg/data           || true
git mv analyzer_pkg/analyzer_functions/models            src/analyzer_pkg/models         || true
git mv folder/analyzer_functions/models                  src/analyzer_pkg/models         || true

echo "🗑️  Removing duplicate & generated trees …"
rm -rf folder                                            # duplicated copy
rm -rf analyzer_pkg/analyzer_functions
rm -rf build dist *.egg-info */*.egg-info
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.py[co]" -delete

echo "🧹  Tidying empty dirs …"
find . -type d -empty -delete

echo "✅  Staging & committing new tree …"
git add .
git commit -m "feat(layout): migrate to src/analyzer_pkg package"

echo "🎉  Refactor completed. Switch back with:  git switch main"
