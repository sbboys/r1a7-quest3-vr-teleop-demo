#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SDK_ROOT="${SDK_ROOT:-/home/robot/unitree_sdk2}"
OUT_DIR="${OUT_DIR:-build/tools}"
OUT="${OUT_DIR}/r1a7_official_arm_sdk_jog"
ARCH="$(uname -m)"

case "${ARCH}" in
  x86_64|amd64)
    SDK_ARCH="x86_64"
    ;;
  aarch64|arm64)
    SDK_ARCH="aarch64"
    ;;
  *)
    echo "[R1-A7 OFFICIAL ARM SDK BUILD] unsupported arch: ${ARCH}" >&2
    exit 2
    ;;
esac

mkdir -p "${OUT_DIR}"

g++ -std=c++17 -O2 \
  tools/r1a7_official_arm_sdk_jog.cpp \
  -I"${SDK_ROOT}/include" \
  -I"${SDK_ROOT}/thirdparty/include" \
  -I"${SDK_ROOT}/thirdparty/include/ddscxx" \
  "${SDK_ROOT}/lib/${SDK_ARCH}/libunitree_sdk2.a" \
  -L"${SDK_ROOT}/thirdparty/lib/${SDK_ARCH}" \
  -Wl,-rpath,"${SDK_ROOT}/thirdparty/lib/${SDK_ARCH}" \
  -lddscxx -lddsc -lspdlog -lfmt -lpthread -ldl \
  -o "${OUT}"

echo "[R1-A7 OFFICIAL ARM SDK BUILD] built ${OUT}"
