#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONDA_BIN="${CONDA_BIN:-/home/robot/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-isaaclab}"

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python -u tools/r1a7_build_teach_profile.py "$@"
