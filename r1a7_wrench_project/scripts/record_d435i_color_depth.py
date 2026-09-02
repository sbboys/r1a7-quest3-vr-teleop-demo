#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import signal
import time

import cv2
import numpy as np
import pyrealsense2 as rs


STOP = False


def request_stop(_signum, _frame) -> None:
    global STOP
    STOP = True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record Intel RealSense D435i color video plus depth preview."
    )
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--serial", default="")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    episode_dir = args.episode_dir.expanduser().resolve()
    episode_dir.mkdir(parents=True, exist_ok=True)
    color_path = episode_dir / "d435i_color.mp4"
    depth_path = episode_dir / "d435i_depth_preview.mp4"
    timestamp_path = episode_dir / "d435i_frames.csv"

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)
    device = profile.get_device()
    serial = device.get_info(rs.camera_info.serial_number)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    color_writer = cv2.VideoWriter(
        str(color_path),
        fourcc,
        float(args.fps),
        (args.width, args.height),
        True,
    )
    depth_writer = cv2.VideoWriter(
        str(depth_path),
        fourcc,
        float(args.fps),
        (args.width, args.height),
        True,
    )
    if not color_writer.isOpened():
        raise RuntimeError(f"cannot open color video writer: {color_path}")
    if not depth_writer.isOpened():
        raise RuntimeError(f"cannot open depth video writer: {depth_path}")

    start = time.monotonic()
    frames = 0
    with timestamp_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "frame_index",
                "host_time_monotonic",
                "elapsed_s",
                "color_frame_number",
                "depth_frame_number",
                "color_timestamp_ms",
                "depth_timestamp_ms",
                "serial",
            ]
        )

        try:
            while not STOP and time.monotonic() - start < args.duration_s:
                frameset = pipeline.wait_for_frames(timeout_ms=2000)
                color_frame = frameset.get_color_frame()
                depth_frame = frameset.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                color = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data())
                depth_8u = cv2.convertScaleAbs(depth, alpha=0.03)
                depth_bgr = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)

                color_writer.write(color)
                depth_writer.write(depth_bgr)

                now = time.monotonic()
                writer.writerow(
                    [
                        frames,
                        f"{now:.6f}",
                        f"{now - start:.6f}",
                        color_frame.get_frame_number(),
                        depth_frame.get_frame_number(),
                        f"{color_frame.get_timestamp():.3f}",
                        f"{depth_frame.get_timestamp():.3f}",
                        serial,
                    ]
                )
                frames += 1
        finally:
            pipeline.stop()
            color_writer.release()
            depth_writer.release()

    print(
        f"D435I_OPENCV_DONE frames={frames} color={color_path} depth={depth_path}",
        flush=True,
    )
    return 0 if frames > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
