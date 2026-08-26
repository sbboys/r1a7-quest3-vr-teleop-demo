#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/logs"
OUT="$ROOT/logs/environment_$(date +%Y%m%d_%H%M%S).txt"

{
  echo "timestamp: $(date -Is)"
  echo "cwd: $ROOT"
  echo
  echo "## git"
  git -C "$ROOT" rev-parse HEAD 2>/dev/null || true
  git -C "$ROOT" branch --show-current 2>/dev/null || true
  git -C "$ROOT" status --short --branch 2>/dev/null || true
  echo
  echo "## os"
  uname -a
  lsb_release -a 2>/dev/null || cat /etc/os-release 2>/dev/null || true
  echo
  echo "## compilers"
  gcc --version 2>/dev/null | head -n 1 || true
  g++ --version 2>/dev/null | head -n 1 || true
  echo
  echo "## python"
  python3 --version 2>&1 || true
  /home/robot/miniconda3/bin/conda run -n tv python --version 2>&1 || true
  echo
  echo "## python packages"
  python3 -m pip freeze 2>/dev/null || true
  echo
  echo "## sdk paths"
  python3 - <<'PY' 2>/dev/null || true
import importlib.util
for name in ["unitree_sdk2py", "numpy", "scipy"]:
    spec = importlib.util.find_spec(name)
    print(f"{name}: {spec.origin if spec else 'not found'}")
PY
  echo
  echo "## network"
  ip -br link 2>/dev/null || true
  ip -br addr 2>/dev/null || true
  ip route 2>/dev/null || true
  echo
  echo "## dds env"
  env | grep -E 'CYCLONEDDS|FASTRTPS|RMW|DDS|ROS_DOMAIN|UNITREE' || true
} > "$OUT"

echo "$OUT"
