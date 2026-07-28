#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

exec ./scripts/archive_r1a7_debug/r1a7_start_body_for_arm_control.sh "$@"
