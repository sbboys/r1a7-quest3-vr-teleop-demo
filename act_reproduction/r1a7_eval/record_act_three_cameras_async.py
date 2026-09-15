#!/usr/bin/env python3

import argparse
import csv
import json
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

from pyorbbecsdk import (
    Config,
    Context,
    OBFormat,
    OBSensorType,
    Pipeline,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "r1a7_wrench_project/config/act_cameras.yaml"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--duration-s",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--queue-size",
        type=int,
        default=180,
    )

    return parser.parse_args()


def load_yaml(path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class StartGate:
    def __init__(self):
        self.event = threading.Event()
        self.start_monotonic = None

    def start(self):
        self.start_monotonic = time.monotonic()
        self.event.set()


class AsyncCameraRecorder(threading.Thread):
    def __init__(
        self,
        role,
        output_dir,
        duration_s,
        queue_size,
        gate,
        width,
        height,
        fps,
    ):
        super().__init__(daemon=True)

        self.role = role
        self.output_dir = output_dir
        self.duration_s = float(duration_s)

        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)

        self.gate = gate

        self.ready = threading.Event()

        self.frame_queue = queue.Queue(
            maxsize=int(queue_size)
        )

        self.capture_pairs = 0
        self.saved_pairs = 0
        self.queue_drops = 0
        self.queue_max_observed = 0

        self.capture_error = None
        self.writer_error = None
        self.stop_error = None

        self.host_times = []
        self.color_device_times = []
        self.depth_device_times = []

        self.writer_thread = None

    def decode_color(self, payload):
        raise NotImplementedError

    def enqueue(
        self,
        capture_index,
        host_time,
        color_device_time,
        depth_device_time,
        color_payload,
        depth_u16,
    ):
        item = (
            capture_index,
            host_time,
            color_device_time,
            depth_device_time,
            color_payload,
            depth_u16,
        )

        try:
            self.frame_queue.put_nowait(item)

            self.queue_max_observed = max(
                self.queue_max_observed,
                self.frame_queue.qsize(),
            )

        except queue.Full:
            self.queue_drops += 1

    def writer_loop(self):
        video_writer = None
        depth_file = None
        csv_file = None

        try:
            self.output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            color_path = (
                self.output_dir
                / f"{self.role}_color.avi"
            )

            depth_path = (
                self.output_dir
                / f"{self.role}_depth.bin"
            )

            csv_path = (
                self.output_dir
                / f"{self.role}_frames.csv"
            )

            fourcc = cv2.VideoWriter_fourcc(
                *"MJPG"
            )

            video_writer = cv2.VideoWriter(
                str(color_path),
                fourcc,
                float(self.fps),
                (self.width, self.height),
            )

            if not video_writer.isOpened():
                raise RuntimeError(
                    f"{self.role}: VideoWriter open failed"
                )

            depth_file = depth_path.open("wb")

            csv_file = csv_path.open(
                "w",
                newline="",
                encoding="utf-8",
            )

            writer = csv.writer(csv_file)

            writer.writerow([
                "saved_frame_index",
                "capture_pair_index",
                "host_time_monotonic",
                "color_device_time_s",
                "depth_device_time_s",
            ])

            while True:
                item = self.frame_queue.get()

                if item is None:
                    break

                (
                    capture_index,
                    host_time,
                    color_device_time,
                    depth_device_time,
                    color_payload,
                    depth_u16,
                ) = item

                color_bgr = self.decode_color(
                    color_payload
                )

                if color_bgr is None:
                    raise RuntimeError(
                        f"{self.role}: color decode failed"
                    )

                if color_bgr.shape[:2] != (
                    self.height,
                    self.width,
                ):
                    raise RuntimeError(
                        f"{self.role}: "
                        f"bad color shape "
                        f"{color_bgr.shape}"
                    )

                if depth_u16.dtype != np.uint16:
                    raise RuntimeError(
                        f"{self.role}: "
                        f"bad depth dtype "
                        f"{depth_u16.dtype}"
                    )

                if depth_u16.shape != (
                    self.height,
                    self.width,
                ):
                    raise RuntimeError(
                        f"{self.role}: "
                        f"bad depth shape "
                        f"{depth_u16.shape}"
                    )

                video_writer.write(color_bgr)

                depth_file.write(
                    np.ascontiguousarray(
                        depth_u16
                    ).tobytes(order="C")
                )

                writer.writerow([
                    self.saved_pairs,
                    capture_index,
                    f"{host_time:.9f}",
                    f"{color_device_time:.9f}",
                    f"{depth_device_time:.9f}",
                ])

                self.host_times.append(
                    float(host_time)
                )

                self.color_device_times.append(
                    float(color_device_time)
                )

                self.depth_device_times.append(
                    float(depth_device_time)
                )

                self.saved_pairs += 1

        except Exception as exc:
            self.writer_error = (
                f"{type(exc).__name__}: {exc}"
            )

        finally:
            if video_writer is not None:
                video_writer.release()

            if depth_file is not None:
                depth_file.flush()
                depth_file.close()

            if csv_file is not None:
                csv_file.flush()
                csv_file.close()

    def start_writer(self):
        self.writer_thread = threading.Thread(
            target=self.writer_loop,
            name=f"{self.role}-writer",
            daemon=True,
        )

        self.writer_thread.start()

    def finish_writer(self):
        if self.writer_thread is None:
            return

        if not self.writer_thread.is_alive():
            return

        try:
            self.frame_queue.put(
                None,
                timeout=10.0,
            )
        except queue.Full:
            self.writer_error = (
                "unable to enqueue writer stop sentinel"
            )
            return

        self.writer_thread.join(
            timeout=30.0
        )

        if self.writer_thread.is_alive():
            self.writer_error = (
                "writer thread did not terminate"
            )

    @staticmethod
    def timing_stats(values):
        if len(values) < 2:
            return {}

        values = np.asarray(
            values,
            dtype=np.float64,
        )

        dt = np.diff(values)

        return {
            "median_s": float(np.median(dt)),
            "min_s": float(np.min(dt)),
            "max_s": float(np.max(dt)),
            "p95_s": float(
                np.percentile(dt, 95)
            ),
            "duplicates": int(
                np.sum(dt == 0)
            ),
            "backward": int(
                np.sum(dt < 0)
            ),
        }

    def summary(self):
        return {
            "role": self.role,
            "capture_pairs": self.capture_pairs,
            "saved_pairs": self.saved_pairs,
            "queue_drops": self.queue_drops,
            "queue_max_observed":
                self.queue_max_observed,
            "capture_error":
                self.capture_error,
            "writer_error":
                self.writer_error,
            "stop_error":
                self.stop_error,
            "host_timing":
                self.timing_stats(
                    self.host_times
                ),
            "color_device_timing":
                self.timing_stats(
                    self.color_device_times
                ),
            "depth_device_timing":
                self.timing_stats(
                    self.depth_device_times
                ),
        }


class RealSenseRecorder(
    AsyncCameraRecorder
):
    def __init__(
        self,
        role,
        cfg,
        output_dir,
        duration_s,
        queue_size,
        gate,
    ):
        super().__init__(
            role=role,
            output_dir=output_dir,
            duration_s=duration_s,
            queue_size=queue_size,
            gate=gate,
            width=cfg["color"]["width"],
            height=cfg["color"]["height"],
            fps=cfg["color"]["fps"],
        )

        self.serial = str(
            cfg["serial_number"]
        )

        self.pipeline = None

    def decode_color(self, payload):
        return payload

    def run(self):
        try:
            print(
                f"[{self.role}] starting RealSense "
                f"SN={self.serial}",
                flush=True,
            )

            self.pipeline = rs.pipeline()
            config = rs.config()

            config.enable_device(
                self.serial
            )

            config.enable_stream(
                rs.stream.color,
                self.width,
                self.height,
                rs.format.bgr8,
                self.fps,
            )

            config.enable_stream(
                rs.stream.depth,
                self.width,
                self.height,
                rs.format.z16,
                self.fps,
            )

            self.pipeline.start(config)

            self.start_writer()

            print(
                f"[{self.role}] ready",
                flush=True,
            )

            self.ready.set()

            while not self.gate.event.is_set():
                self.pipeline.wait_for_frames(
                    1000
                )

            deadline = (
                self.gate.start_monotonic
                + self.duration_s
            )

            while time.monotonic() < deadline:
                frames = (
                    self.pipeline.wait_for_frames(
                        1000
                    )
                )

                host_time = time.monotonic()

                color = frames.get_color_frame()
                depth = frames.get_depth_frame()

                if not color or not depth:
                    continue

                color_bgr = (
                    np.asanyarray(
                        color.get_data()
                    )
                    .reshape(
                        self.height,
                        self.width,
                        3,
                    )
                    .copy()
                )

                depth_u16 = (
                    np.asanyarray(
                        depth.get_data()
                    )
                    .reshape(
                        self.height,
                        self.width,
                    )
                    .astype(
                        np.uint16,
                        copy=True,
                    )
                )

                color_time = (
                    float(
                        color.get_timestamp()
                    )
                    / 1000.0
                )

                depth_time = (
                    float(
                        depth.get_timestamp()
                    )
                    / 1000.0
                )

                capture_index = (
                    self.capture_pairs
                )

                self.capture_pairs += 1

                self.enqueue(
                    capture_index,
                    host_time,
                    color_time,
                    depth_time,
                    color_bgr,
                    depth_u16,
                )

        except Exception as exc:
            self.capture_error = (
                f"{type(exc).__name__}: {exc}"
            )

            self.ready.set()

        finally:
            if self.pipeline is not None:
                try:
                    self.pipeline.stop()
                except Exception as exc:
                    self.stop_error = (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )

            self.finish_writer()

            print(
                f"[{self.role}] stopped",
                flush=True,
            )


class OrbbecRecorder(
    AsyncCameraRecorder
):
    def __init__(
        self,
        role,
        cfg,
        device,
        output_dir,
        duration_s,
        queue_size,
        gate,
    ):
        super().__init__(
            role=role,
            output_dir=output_dir,
            duration_s=duration_s,
            queue_size=queue_size,
            gate=gate,
            width=cfg["color"]["width"],
            height=cfg["color"]["height"],
            fps=cfg["color"]["fps"],
        )

        self.serial = str(
            cfg["serial_number"]
        )

        self.device = device
        self.pipeline = None

    def decode_color(self, payload):
        return cv2.imdecode(
            payload,
            cv2.IMREAD_COLOR,
        )

    def run(self):
        try:
            print(
                f"[{self.role}] starting Orbbec "
                f"SN={self.serial}",
                flush=True,
            )

            self.pipeline = Pipeline(
                self.device
            )

            config = Config()

            color_profile = (
                self.pipeline
                .get_stream_profile_list(
                    OBSensorType.COLOR_SENSOR
                )
                .get_video_stream_profile(
                    self.width,
                    self.height,
                    OBFormat.MJPG,
                    self.fps,
                )
            )

            depth_profile = (
                self.pipeline
                .get_stream_profile_list(
                    OBSensorType.DEPTH_SENSOR
                )
                .get_video_stream_profile(
                    self.width,
                    self.height,
                    OBFormat.Y16,
                    self.fps,
                )
            )

            config.enable_stream(
                color_profile
            )

            config.enable_stream(
                depth_profile
            )

            self.pipeline.start(config)

            self.start_writer()

            print(
                f"[{self.role}] ready",
                flush=True,
            )

            self.ready.set()

            while not self.gate.event.is_set():
                self.pipeline.wait_for_frames(
                    1000
                )

            deadline = (
                self.gate.start_monotonic
                + self.duration_s
            )

            while time.monotonic() < deadline:
                frames = (
                    self.pipeline.wait_for_frames(
                        1000
                    )
                )

                if frames is None:
                    continue

                host_time = time.monotonic()

                color = (
                    frames.get_color_frame()
                )

                depth = (
                    frames.get_depth_frame()
                )

                if (
                    color is None
                    or depth is None
                ):
                    continue

                color_payload = (
                    np.asanyarray(
                        color.get_data()
                    )
                    .copy()
                )

                raw_depth = np.asanyarray(
                    depth.get_data()
                )

                if (
                    raw_depth.dtype
                    == np.uint16
                    and raw_depth.size
                    == self.width * self.height
                ):
                    depth_u16 = (
                        raw_depth
                        .reshape(
                            self.height,
                            self.width,
                        )
                        .copy()
                    )

                elif (
                    raw_depth.size
                    == self.width
                    * self.height
                    * 2
                ):
                    depth_u16 = (
                        raw_depth
                        .view(np.uint16)
                        .reshape(
                            self.height,
                            self.width,
                        )
                        .copy()
                    )

                else:
                    raise RuntimeError(
                        f"{self.role}: "
                        f"unexpected depth "
                        f"shape={raw_depth.shape}, "
                        f"dtype={raw_depth.dtype}, "
                        f"size={raw_depth.size}"
                    )

                color_time = (
                    float(
                        color.get_timestamp_us()
                    )
                    / 1_000_000.0
                )

                depth_time = (
                    float(
                        depth.get_timestamp_us()
                    )
                    / 1_000_000.0
                )

                capture_index = (
                    self.capture_pairs
                )

                self.capture_pairs += 1

                self.enqueue(
                    capture_index,
                    host_time,
                    color_time,
                    depth_time,
                    color_payload,
                    depth_u16,
                )

        except Exception as exc:
            self.capture_error = (
                f"{type(exc).__name__}: {exc}"
            )

            self.ready.set()

        finally:
            if self.pipeline is not None:
                try:
                    self.pipeline.stop()
                except Exception as exc:
                    self.stop_error = (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )

            self.finish_writer()

            print(
                f"[{self.role}] stopped",
                flush=True,
            )


def main():
    args = parse_args()

    if args.duration_s <= 0:
        raise SystemExit(
            "--duration-s must be positive"
        )

    if args.queue_size <= 0:
        raise SystemExit(
            "--queue-size must be positive"
        )

    cfg = load_yaml(
        args.config
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    camera_cfg = cfg["cameras"]

    context = Context()
    device_list = (
        context.query_devices()
    )

    orbbec_devices = {}

    for i in range(
        device_list.get_count()
    ):
        device = (
            device_list
            .get_device_by_index(i)
        )

        info = (
            device.get_device_info()
        )

        serial = (
            info.get_serial_number()
        )

        print(
            f"Found Orbbec #{i}: "
            f"{info.get_name()} "
            f"SN={serial}",
            flush=True,
        )

        orbbec_devices[
            serial
        ] = device

    left_sn = str(
        camera_cfg[
            "left_wrist"
        ]["serial_number"]
    )

    right_sn = str(
        camera_cfg[
            "right_wrist"
        ]["serial_number"]
    )

    if left_sn not in orbbec_devices:
        raise RuntimeError(
            f"left Orbbec not found: {left_sn}"
        )

    if right_sn not in orbbec_devices:
        raise RuntimeError(
            f"right Orbbec not found: {right_sn}"
        )

    gate = StartGate()

    top = RealSenseRecorder(
        "top",
        camera_cfg["top"],
        args.output_dir,
        args.duration_s,
        args.queue_size,
        gate,
    )

    left = OrbbecRecorder(
        "left_wrist",
        camera_cfg["left_wrist"],
        orbbec_devices[left_sn],
        args.output_dir,
        args.duration_s,
        args.queue_size,
        gate,
    )

    right = OrbbecRecorder(
        "right_wrist",
        camera_cfg["right_wrist"],
        orbbec_devices[right_sn],
        args.output_dir,
        args.duration_s,
        args.queue_size,
        gate,
    )

    workers = [
        top,
        left,
        right,
    ]

    for worker in workers:
        worker.start()

        if not worker.ready.wait(
            timeout=10.0
        ):
            raise RuntimeError(
                f"{worker.role}: startup timeout"
            )

        if worker.capture_error:
            raise RuntimeError(
                f"{worker.role}: "
                f"{worker.capture_error}"
            )

        time.sleep(0.5)

    warmup_s = float(
        cfg.get(
            "recording",
            {},
        ).get(
            "warmup_s",
            2.0,
        )
    )

    print()
    print(
        f"All cameras ready. "
        f"Warm-up {warmup_s:.1f}s...",
        flush=True,
    )

    time.sleep(warmup_s)

    gate.start()

    print(
        f"RECORD START host_monotonic="
        f"{gate.start_monotonic:.9f}",
        flush=True,
    )

    for worker in workers:
        worker.join(
            timeout=
                args.duration_s
                + 40.0
        )

    summaries = [
        worker.summary()
        for worker in workers
    ]

    manifest = {
        "schema_version":
            "r1a7_act_camera_async_v1",
        "duration_requested_s":
            float(args.duration_s),
        "record_start_monotonic":
            float(gate.start_monotonic),
        "queue_size":
            int(args.queue_size),
        "depth": {
            "dtype": "uint16",
            "width": 640,
            "height": 480,
            "one_frame_bytes":
                640 * 480 * 2,
        },
        "cameras": summaries,
    }

    manifest_path = (
        args.output_dir
        / "camera_recording_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    expected_min = int(
        args.duration_s
        * 30.0
        * 0.90
    )

    passed = True

    print()
    print("=" * 78)
    print("RESULT")
    print("=" * 78)

    for summary in summaries:
        print()
        print(summary["role"])

        print(
            "  capture_pairs      :",
            summary["capture_pairs"],
        )

        print(
            "  saved_pairs        :",
            summary["saved_pairs"],
        )

        print(
            "  queue_drops        :",
            summary["queue_drops"],
        )

        print(
            "  queue_max_observed :",
            summary["queue_max_observed"],
        )

        print(
            "  capture_error      :",
            summary["capture_error"],
        )

        print(
            "  writer_error       :",
            summary["writer_error"],
        )

        print(
            "  stop_error         :",
            summary["stop_error"],
        )

        print(
            "  host median dt     :",
            summary[
                "host_timing"
            ].get("median_s"),
        )

        print(
            "  color dup/back     :",
            summary[
                "color_device_timing"
            ].get("duplicates"),
            "/",
            summary[
                "color_device_timing"
            ].get("backward"),
        )

        print(
            "  depth dup/back     :",
            summary[
                "depth_device_timing"
            ].get("duplicates"),
            "/",
            summary[
                "depth_device_timing"
            ].get("backward"),
        )

        if (
            summary["capture_pairs"]
            < expected_min
        ):
            passed = False

        if (
            summary["saved_pairs"]
            < expected_min
        ):
            passed = False

        if (
            summary["capture_pairs"]
            != summary["saved_pairs"]
        ):
            passed = False

        if summary["queue_drops"] != 0:
            passed = False

        if summary["capture_error"] is not None:
            passed = False

        if summary["writer_error"] is not None:
            passed = False

        if (
            summary[
                "color_device_timing"
            ].get("duplicates", 0)
            != 0
        ):
            passed = False

        if (
            summary[
                "color_device_timing"
            ].get("backward", 0)
            != 0
        ):
            passed = False

        if (
            summary[
                "depth_device_timing"
            ].get("duplicates", 0)
            != 0
        ):
            passed = False

        if (
            summary[
                "depth_device_timing"
            ].get("backward", 0)
            != 0
        ):
            passed = False

    print()
    print("Files:")

    for path in sorted(
        args.output_dir.iterdir()
    ):
        if path.is_file():
            size_mib = (
                path.stat().st_size
                / 1024
                / 1024
            )

            print(
                f"  {path.name:38s} "
                f"{size_mib:8.2f} MiB"
            )

    print()

    if passed:
        print(
            "ACT_THREE_CAMERA_ASYNC_TEST=PASS"
        )
        return 0

    print(
        "ACT_THREE_CAMERA_ASYNC_TEST=FAIL"
    )

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
