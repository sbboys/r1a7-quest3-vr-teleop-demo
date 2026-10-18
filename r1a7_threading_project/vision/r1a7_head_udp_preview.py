#!/usr/bin/env python3
"""Receive and display the official R1-A img stream (UDP 5001 RTP/H.264)."""

from __future__ import annotations

import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import sys
import threading
import time

import cv2
import numpy as np


INTERFACE = os.environ.get("R1A7_INTERFACE", "enp6s0")
RECEIVER_IP = os.environ.get("R1A7_HEAD_RECEIVER_IP", "192.168.123.164")
PORT = int(os.environ.get("R1A7_HEAD_UDP_PORT", "5001"))
WIDTH = int(os.environ.get("R1A7_HEAD_WIDTH", "1280"))
HEIGHT = int(os.environ.get("R1A7_HEAD_HEIGHT", "720"))
FIRST_FRAME_TIMEOUT = float(os.environ.get("R1A7_HEAD_FIRST_FRAME_TIMEOUT", "12"))
DISPLAY_ENABLED = os.environ.get("R1A7_HEAD_DISPLAY", "1") != "0"
RESTORE_SERVICES = os.environ.get("R1A7_HEAD_RESTORE_SERVICES", "1") != "0"
WINDOW = "R1-A Head RGB · official UDP 5001"
SERVICE_HELPER = Path(__file__).with_name("r1a7_head_service_prepare.py")


def pipeline_command() -> list[str]:
    caps = (
        "application/x-rtp,media=video,clock-rate=90000,"
        "encoding-name=H264,payload=96"
    )
    return [
        "gst-launch-1.0",
        "-q",
        "udpsrc",
        f"address={RECEIVER_IP}",
        f"port={PORT}",
        "buffer-size=2097152",
        f"caps={caps}",
        "!",
        "rtpjitterbuffer",
        "latency=100",
        "drop-on-latency=true",
        "!",
        "rtph264depay",
        "!",
        "avdec_h264",
        "!",
        "videoconvert",
        "!",
        "videoscale",
        "!",
        f"video/x-raw,format=BGR,width={WIDTH},height={HEIGHT}",
        "!",
        "fdsink",
        "fd=1",
        "sync=false",
    ]


def restore_robot_services() -> None:
    if not RESTORE_SERVICES or not SERVICE_HELPER.exists():
        return
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            str(SERVICE_HELPER),
            "--interface",
            INTERFACE,
            "--mode",
            "restore",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=35.0,
    )
    for line in (result.stdout + result.stderr).splitlines():
        if line:
            print(line, flush=True)


def stderr_reader(process: subprocess.Popen[bytes]) -> None:
    assert process.stderr is not None
    for raw in iter(process.stderr.readline, b""):
        message = raw.decode("utf-8", errors="replace").strip()
        if message:
            print(f"[HEAD GST] {message}", flush=True)


def main() -> int:
    if not shutil.which("gst-launch-1.0"):
        print("[HEAD UDP ERROR] gst-launch-1.0 not found", flush=True)
        return 2

    print(
        "[HEAD UDP] "
        f"bind={RECEIVER_IP}:{PORT} codec=RTP/H264 target={WIDTH}x{HEIGHT}",
        flush=True,
    )
    process = subprocess.Popen(
        pipeline_command(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert process.stdout is not None
    threading.Thread(target=stderr_reader, args=(process,), daemon=True).start()
    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    frame_size = WIDTH * HEIGHT * 3
    pending = bytearray()
    deadline = time.monotonic() + FIRST_FRAME_TIMEOUT
    ready = False
    try:
        if DISPLAY_ENABLED:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        while not stop_requested:
            if process.poll() is not None:
                print(
                    f"[HEAD UDP ERROR] GStreamer exited with code {process.returncode}",
                    flush=True,
                )
                return 3
            readable, _, _ = select.select([process.stdout], [], [], 0.1)
            if readable:
                chunk = os.read(process.stdout.fileno(), 1024 * 1024)
                if chunk:
                    pending.extend(chunk)
            if not ready and time.monotonic() >= deadline:
                print(
                    "[HEAD UDP ERROR] UDP 5001 在 "
                    f"{FIRST_FRAME_TIMEOUT:.0f} 秒内没有收到 RTP/H.264 图像；"
                    f"接收地址 {RECEIVER_IP} 已就绪且 Stereo patch PC1 已启动。"
                    "请让宇树技术支持确认机器人端 Patch >= 1.0.1.17、"
                    "Stereo patch PC1 >= 0.4.0.1、System_patch_pc4 = 2.0.0.0",
                    flush=True,
                )
                return 4
            latest: np.ndarray | None = None
            while len(pending) >= frame_size:
                frame_bytes = bytes(pending[:frame_size])
                del pending[:frame_size]
                latest = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
                    (HEIGHT, WIDTH, 3)
                )
            if latest is None:
                continue
            if not ready:
                print(f"[HEAD FRAME READY] {WIDTH}x{HEIGHT} UDP {PORT}", flush=True)
                ready = True
            if DISPLAY_ENABLED:
                cv2.imshow(WINDOW, latest)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        return 0
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=3.0)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
        if DISPLAY_ENABLED:
            cv2.destroyAllWindows()
        restore_robot_services()


if __name__ == "__main__":
    raise SystemExit(main())
