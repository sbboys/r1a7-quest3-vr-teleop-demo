#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

required_files=(
  PROJECT_INDEX.md
  BRANCH_CATALOG.md
  docs/README.md
  docs/repository_management_zh.md
  docs/r1a7_github_reorganization_2026-10-11_zh.md
  projects/README.md
  r1a7_threading_project/docs/r1a7_visual_servo_control_handoff_2026-10-10_zh.md
  r1a7_threading_project/docs/r1a7_arm_console_direction_switch_correction_2026-09-30_zh.md
)

for file in "${required_files[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "missing required repository entry: $file" >&2
    exit 1
  fi
done

if grep -nRE '^(<<<<<<<|=======|>>>>>>>)' \
  PROJECT_INDEX.md BRANCH_CATALOG.md docs projects r1a7_threading_project/README.md; then
  echo "unresolved merge marker found" >&2
  exit 1
fi

bash -n r1a7_threading_project/run_task6_visual_stack.sh

pycache_root="${TMPDIR:-/tmp}/r1a7_repository_sanity_pycache"
export PYTHONPYCACHEPREFIX="$pycache_root"

mapfile -t python_files < <(
  find \
    r1a7_threading_project/transfer_control/diagnostics \
    r1a7_threading_project/transfer_control/experimental \
    -maxdepth 1 -type f -name '*.py' -print | sort
)
python_files+=(
  r1a7_threading_project/vision_dual_camera/task6_threading/task6_dual_ibvs_fusion.py
  r1a7_threading_project/vision_dual_camera/task6_threading/task6_external_ibvs_v4_aruco_edge.py
  r1a7_threading_project/vision_dual_camera/task6_threading/task6_wrist_ibvs_controller_v2_tip.py
)

python3 -m py_compile "${python_files[@]}"

if [[ -n "${R1A7_DIFF_BASE:-}" ]] && git rev-parse --verify "$R1A7_DIFF_BASE^{commit}" >/dev/null 2>&1; then
  git diff --check "$R1A7_DIFF_BASE"..HEAD

  while IFS= read -r -d '' file; do
    if ! git cat-file -e "HEAD:$file" 2>/dev/null; then
      continue
    fi
    size="$(git cat-file -s "HEAD:$file")"
    if (( size > 20971520 )); then
      echo "new or modified file exceeds 20 MiB: $file ($size bytes)" >&2
      exit 1
    fi
  done < <(git diff --name-only --diff-filter=AM -z "$R1A7_DIFF_BASE"..HEAD)
else
  git diff --check
fi

echo "repository sanity checks passed"
