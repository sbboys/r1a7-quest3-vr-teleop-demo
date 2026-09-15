#!/usr/bin/env python3

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


def load_config(path):
    path = Path(path).expanduser().resolve()

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class BaseCameraWorker(threading.Thread):
    def __init__(self, role):
        super().__init__(
            name=f"act-camera-{role}",
            daemon=True,
        )

        self.role = role

        self.ready = threading.Event()
        self.stop_event = threading.Event()

        self.framesets = 0
        self.color_frames = 0
        self.depth_frames = 0

        self.last_host_time = None
        self.last_color_device_time = None
        self.last_depth_device_time = None

        self.error = None
        self.stop_error = None

        self.episode_gate = None

        self.episode_framesets = 0
        self.episode_color_frames = 0
        self.episode_depth_frames = 0

        self.episode_capture_pairs = 0
        self.episode_saved_pairs = 0
        self.episode_queue_drops = 0
        self.episode_queue_max_observed = 0

        self.episode_dir = None
        self.episode_queue = None
        self.episode_writer_thread = None
        self.episode_writer_ready = None
        self.episode_writer_error = None

        self.episode_io_lock = threading.Lock()

        # Latest RGB frame for real-time ACT inference.
        # Independent of episode recording.
        self.latest_color_lock = threading.Lock()
        self.latest_color_bgr = None
        self.latest_color_host_time = None

    def set_latest_color_bgr(self, image_bgr, host_time):
        if image_bgr is None:
            return

        image_bgr = np.asarray(image_bgr)

        if (
            image_bgr.ndim != 3
            or image_bgr.shape[2] != 3
            or image_bgr.dtype != np.uint8
        ):
            raise RuntimeError(
                f"{self.role}: invalid latest RGB frame "
                f"shape={image_bgr.shape} dtype={image_bgr.dtype}"
            )

        with self.latest_color_lock:
            self.latest_color_bgr = image_bgr.copy()
            self.latest_color_host_time = float(host_time)

    def get_latest_color_bgr(self):
        with self.latest_color_lock:
            if (
                self.latest_color_bgr is None
                or self.latest_color_host_time is None
            ):
                return None

            return {
                "host_time": float(self.latest_color_host_time),
                "image_bgr": self.latest_color_bgr.copy(),
            }

    def decode_episode_color(self, payload):
        return payload

    def start_episode_writer(
        self,
        episode_dir,
        queue_size=180,
    ):
        if (
            self.episode_writer_thread is not None
            and self.episode_writer_thread.is_alive()
        ):
            raise RuntimeError(
                f"{self.role}: episode writer already running"
            )

        self.episode_dir = (
            Path(episode_dir)
            .expanduser()
            .resolve()
        )

        self.episode_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        targets = [
            self.episode_dir
            / f"{self.role}_color.avi",
            self.episode_dir
            / f"{self.role}_depth.bin",
            self.episode_dir
            / f"{self.role}_frames.csv",
        ]

        existing = [
            str(p)
            for p in targets
            if p.exists()
        ]

        if existing:
            raise RuntimeError(
                f"{self.role}: output files already exist: "
                + ", ".join(existing)
            )

        self.episode_queue = queue.Queue(
            maxsize=int(queue_size)
        )

        self.episode_writer_ready = (
            threading.Event()
        )

        self.episode_writer_error = None

        self.episode_writer_thread = (
            threading.Thread(
                target=self._episode_writer_loop,
                name=f"{self.role}-episode-writer",
                daemon=True,
            )
        )

        self.episode_writer_thread.start()

        if not self.episode_writer_ready.wait(
            timeout=5.0
        ):
            raise RuntimeError(
                f"{self.role}: writer startup timeout"
            )

        if self.episode_writer_error is not None:
            raise RuntimeError(
                f"{self.role}: writer startup failed: "
                f"{self.episode_writer_error}"
            )

    def enqueue_episode_pair(
        self,
        host_time,
        color_device_time,
        depth_device_time,
        color_payload,
        depth_u16,
    ):
        with self.episode_io_lock:
            if (
                self.episode_gate is None
                or not self.episode_gate.is_set()
                or self.episode_queue is None
            ):
                return

            capture_index = (
                self.episode_capture_pairs
            )

            self.episode_capture_pairs += 1

            item = (
                capture_index,
                float(host_time),
                float(color_device_time),
                float(depth_device_time),
                color_payload,
                depth_u16,
            )

            try:
                self.episode_queue.put_nowait(
                    item
                )

                self.episode_queue_max_observed = max(
                    self.episode_queue_max_observed,
                    self.episode_queue.qsize(),
                )

            except queue.Full:
                self.episode_queue_drops += 1

    def _episode_writer_loop(self):
        video_writer = None
        depth_file = None
        csv_file = None

        try:
            color_path = (
                self.episode_dir
                / f"{self.role}_color.avi"
            )

            depth_path = (
                self.episode_dir
                / f"{self.role}_depth.bin"
            )

            csv_path = (
                self.episode_dir
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
                    f"{self.role}: "
                    "VideoWriter open failed"
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

            self.episode_writer_ready.set()

            while True:
                item = self.episode_queue.get()

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

                color_bgr = (
                    self.decode_episode_color(
                        color_payload
                    )
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
                    self.depth_height,
                    self.depth_width,
                ):
                    raise RuntimeError(
                        f"{self.role}: "
                        f"bad depth shape "
                        f"{depth_u16.shape}"
                    )

                video_writer.write(
                    color_bgr
                )

                depth_file.write(
                    np.ascontiguousarray(
                        depth_u16
                    ).tobytes(order="C")
                )

                writer.writerow([
                    self.episode_saved_pairs,
                    capture_index,
                    f"{host_time:.9f}",
                    f"{color_device_time:.9f}",
                    f"{depth_device_time:.9f}",
                ])

                self.episode_saved_pairs += 1

        except Exception as exc:
            self.episode_writer_error = (
                f"{type(exc).__name__}: {exc}"
            )

        finally:
            if self.episode_writer_ready is not None:
                self.episode_writer_ready.set()

            if video_writer is not None:
                video_writer.release()

            if depth_file is not None:
                depth_file.flush()
                depth_file.close()

            if csv_file is not None:
                csv_file.flush()
                csv_file.close()

    def stop_episode_writer(self):
        with self.episode_io_lock:
            q = self.episode_queue
            thread = self.episode_writer_thread

            if q is None or thread is None:
                return

            if thread.is_alive():
                try:
                    q.put(
                        None,
                        timeout=10.0,
                    )
                except queue.Full:
                    self.episode_writer_error = (
                        "writer queue did not accept "
                        "stop sentinel"
                    )

        if thread.ident is not None:
            thread.join(timeout=30.0)

        if thread.is_alive():
            self.episode_writer_error = (
                "writer thread did not terminate"
            )

        self.episode_queue = None
        self.episode_writer_thread = None
        self.episode_writer_ready = None

    def request_stop(self):
        self.stop_event.set()

    def reset_episode_counters(self):
        self.episode_framesets = 0
        self.episode_color_frames = 0
        self.episode_depth_frames = 0

        self.episode_capture_pairs = 0
        self.episode_saved_pairs = 0
        self.episode_queue_drops = 0
        self.episode_queue_max_observed = 0
        self.episode_writer_error = None

    def snapshot(self):
        return {
            "framesets": self.framesets,
            "color_frames": self.color_frames,
            "depth_frames": self.depth_frames,
            "episode_framesets": self.episode_framesets,
            "episode_color_frames": self.episode_color_frames,
            "episode_depth_frames": self.episode_depth_frames,
            "episode_capture_pairs": self.episode_capture_pairs,
            "episode_saved_pairs": self.episode_saved_pairs,
            "episode_queue_drops": self.episode_queue_drops,
            "episode_queue_max_observed":
                self.episode_queue_max_observed,
            "episode_writer_error":
                self.episode_writer_error,
            "episode_active": bool(
                self.episode_gate is not None
                and self.episode_gate.is_set()
            ),
            "error": self.error,
            "stop_error": self.stop_error,
            "alive": self.is_alive(),
        }


class RealSenseWorker(BaseCameraWorker):
    def __init__(self, role, cfg):
        super().__init__(role)

        self.serial = str(cfg["serial_number"])

        self.width = int(cfg["color"]["width"])
        self.height = int(cfg["color"]["height"])
        self.fps = int(cfg["color"]["fps"])

        self.depth_width = int(cfg["depth"]["width"])
        self.depth_height = int(cfg["depth"]["height"])
        self.depth_fps = int(cfg["depth"]["fps"])

        self.pipeline = None

    def run(self):
        try:
            print(
                f"[{self.role}] starting persistent RealSense "
                f"SN={self.serial}",
                flush=True,
            )

            self.pipeline = rs.pipeline()

            config = rs.config()

            config.enable_device(self.serial)

            config.enable_stream(
                rs.stream.color,
                self.width,
                self.height,
                rs.format.bgr8,
                self.fps,
            )

            config.enable_stream(
                rs.stream.depth,
                self.depth_width,
                self.depth_height,
                rs.format.z16,
                self.depth_fps,
            )

            self.pipeline.start(config)

            self.ready.set()

            print(
                f"[{self.role}] persistent pipeline ready",
                flush=True,
            )

            while not self.stop_event.is_set():
                frames = self.pipeline.wait_for_frames(1000)

                if not frames:
                    continue

                host_time = time.monotonic()

                color = frames.get_color_frame()
                depth = frames.get_depth_frame()

                self.framesets += 1
                self.last_host_time = host_time

                if color:
                    self.color_frames += 1
                    self.last_color_device_time = (
                        float(color.get_timestamp())
                        / 1000.0
                    )

                    latest_color_bgr = (
                        np.asanyarray(
                            color.get_data()
                        )
                        .reshape(
                            self.height,
                            self.width,
                            3,
                        )
                    )

                    self.set_latest_color_bgr(
                        latest_color_bgr,
                        host_time,
                    )

                if depth:
                    self.depth_frames += 1
                    self.last_depth_device_time = (
                        float(depth.get_timestamp())
                        / 1000.0
                    )

                if (
                    self.episode_gate is not None
                    and self.episode_gate.is_set()
                ):
                    self.episode_framesets += 1

                    if color:
                        self.episode_color_frames += 1

                    if depth:
                        self.episode_depth_frames += 1

                    if color and depth:
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
                                self.depth_height,
                                self.depth_width,
                            )
                            .astype(
                                np.uint16,
                                copy=True,
                            )
                        )

                        self.enqueue_episode_pair(
                            host_time,
                            self.last_color_device_time,
                            self.last_depth_device_time,
                            color_bgr,
                            depth_u16,
                        )

        except Exception as exc:
            if not self.stop_event.is_set():
                self.error = (
                    f"{type(exc).__name__}: {exc}"
                )

            self.ready.set()

        finally:
            if self.pipeline is not None:
                try:
                    self.pipeline.stop()
                except Exception as exc:
                    self.stop_error = (
                        f"{type(exc).__name__}: {exc}"
                    )

            print(
                f"[{self.role}] persistent pipeline stopped",
                flush=True,
            )


class OrbbecWorker(BaseCameraWorker):
    def decode_episode_color(self, payload):
        return cv2.imdecode(
            payload,
            cv2.IMREAD_COLOR,
        )

    def __init__(self, role, cfg, device):
        super().__init__(role)

        self.serial = str(cfg["serial_number"])

        self.width = int(cfg["color"]["width"])
        self.height = int(cfg["color"]["height"])
        self.fps = int(cfg["color"]["fps"])

        self.depth_width = int(cfg["depth"]["width"])
        self.depth_height = int(cfg["depth"]["height"])
        self.depth_fps = int(cfg["depth"]["fps"])

        self.device = device
        self.pipeline = None
        self.config = None
        self.prepared = False

    def prepare(self):
        """Prepare Orbbec without starting USB streaming.

        Both Gemini cameras must be fully configured before
        either Gemini starts streaming. Tests on this system
        showed that creating/configuring the second Gemini
        while the first Gemini is already streaming can cause
        the already-running device to disconnect from USB.
        """
        if self.prepared:
            return

        print(
            f"[{self.role}] preparing persistent Orbbec "
            f"SN={self.serial}",
            flush=True,
        )

        self.pipeline = Pipeline(self.device)

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
                self.depth_width,
                self.depth_height,
                OBFormat.Y16,
                self.depth_fps,
            )
        )

        config.enable_stream(color_profile)
        config.enable_stream(depth_profile)

        self.config = config
        self.prepared = True

        print(
            f"[{self.role}] Orbbec prepared "
            f"(pipeline configured, not streaming)",
            flush=True,
        )

    def start(self):
        """Start a previously prepared Orbbec pipeline."""
        if (
            not self.prepared
            or self.pipeline is None
            or self.config is None
        ):
            raise RuntimeError(
                f"{self.role}: Orbbec worker must be "
                f"prepared before start()"
            )

        try:
            print(
                f"[{self.role}] starting prepared Orbbec "
                f"SN={self.serial} in manager thread",
                flush=True,
            )

            self.pipeline.start(self.config)

            print(
                f"[{self.role}] pipeline started "
                f"in manager thread",
                flush=True,
            )

        except Exception as exc:
            self.error = (
                f"{type(exc).__name__}: {exc}"
            )
            self.ready.set()

            if self.pipeline is not None:
                try:
                    self.pipeline.stop()
                except Exception:
                    pass

            raise

        try:
            super().start()

        except Exception:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            raise

    def run(self):
        try:
            self.ready.set()

            print(
                f"[{self.role}] persistent capture thread ready",
                flush=True,
            )

            while not self.stop_event.is_set():
                frames = self.pipeline.wait_for_frames(1000)

                if frames is None:
                    continue

                host_time = time.monotonic()

                color = frames.get_color_frame()
                depth = frames.get_depth_frame()

                self.framesets += 1
                self.last_host_time = host_time

                if color is not None:
                    self.color_frames += 1
                    self.last_color_device_time = (
                        float(color.get_timestamp_us())
                        / 1_000_000.0
                    )

                    latest_color_payload = (
                        np.asanyarray(
                            color.get_data()
                        )
                        .copy()
                    )

                    latest_color_bgr = cv2.imdecode(
                        latest_color_payload,
                        cv2.IMREAD_COLOR,
                    )

                    if latest_color_bgr is None:
                        raise RuntimeError(
                            f"{self.role}: failed to decode "
                            "latest MJPG color frame"
                        )

                    self.set_latest_color_bgr(
                        latest_color_bgr,
                        host_time,
                    )

                if depth is not None:
                    self.depth_frames += 1
                    self.last_depth_device_time = (
                        float(depth.get_timestamp_us())
                        / 1_000_000.0
                    )

                if (
                    self.episode_gate is not None
                    and self.episode_gate.is_set()
                ):
                    self.episode_framesets += 1

                    if color is not None:
                        self.episode_color_frames += 1

                    if depth is not None:
                        self.episode_depth_frames += 1

                    if (
                        color is not None
                        and depth is not None
                    ):
                        color_payload = (
                            np.asanyarray(
                                color.get_data()
                            )
                            .copy()
                        )

                        raw_depth = np.asanyarray(
                            depth.get_data()
                        )

                        expected_pixels = (
                            self.depth_width
                            * self.depth_height
                        )

                        if (
                            raw_depth.dtype
                            == np.uint16
                            and raw_depth.size
                            == expected_pixels
                        ):
                            depth_u16 = (
                                raw_depth
                                .reshape(
                                    self.depth_height,
                                    self.depth_width,
                                )
                                .copy()
                            )

                        elif (
                            raw_depth.size
                            == expected_pixels * 2
                        ):
                            depth_u16 = (
                                raw_depth
                                .view(np.uint16)
                                .reshape(
                                    self.depth_height,
                                    self.depth_width,
                                )
                                .copy()
                            )

                        else:
                            raise RuntimeError(
                                f"{self.role}: "
                                f"unexpected depth "
                                f"shape={raw_depth.shape} "
                                f"dtype={raw_depth.dtype} "
                                f"size={raw_depth.size}"
                            )

                        self.enqueue_episode_pair(
                            host_time,
                            self.last_color_device_time,
                            self.last_depth_device_time,
                            color_payload,
                            depth_u16,
                        )

        except Exception as exc:
            if not self.stop_event.is_set():
                self.error = (
                    f"{type(exc).__name__}: {exc}"
                )

            self.ready.set()

        finally:
            if self.pipeline is not None:
                try:
                    self.pipeline.stop()
                except Exception as exc:
                    self.stop_error = (
                        f"{type(exc).__name__}: {exc}"
                    )

            print(
                f"[{self.role}] persistent pipeline stopped",
                flush=True,
            )


class ActThreeCameraManager:
    def __init__(self, config_path):
        self.config_path = (
            Path(config_path)
            .expanduser()
            .resolve()
        )

        self.config = load_config(
            self.config_path
        )

        self.context = None
        self.devices = {}
        self.workers = {}

        self.started = False

        self.episode_gate = threading.Event()
        self.episode_active = False
        self.episode_id = None
        self.episode_started_at = None
        self.episode_dir = None

        self.episode_queue_size = int(
            self.config
            .get("recording", {})
            .get("queue_size", 180)
        )

        # Camera health is based on actual RGB+Depth progress,
        # not only capture-thread liveness.
        self.health_stall_timeout_s = float(
            self.config
            .get("recording", {})
            .get("health_stall_timeout_s", 2.0)
        )

        self._health_last_pairs = {}
        self._health_last_progress_at = {}

    @staticmethod
    def _capture_pair_count(worker):
        snapshot = worker.snapshot()

        return min(
            int(snapshot.get("color_frames", 0)),
            int(snapshot.get("depth_frames", 0)),
        )

    def _wait_for_capture_pairs(
        self,
        role,
        minimum_pairs=5,
        timeout_s=8.0,
    ):
        worker = self.workers[role]

        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            if worker.error is not None:
                raise RuntimeError(
                    f"{role}: {worker.error}"
                )

            if not worker.is_alive():
                raise RuntimeError(
                    f"{role}: capture thread stopped "
                    f"during startup"
                )

            pair_count = self._capture_pair_count(
                worker
            )

            if pair_count >= minimum_pairs:
                return pair_count

            time.sleep(0.05)

        raise RuntimeError(
            f"{role}: no RGB+Depth frame progress "
            f"during startup "
            f"(pairs={self._capture_pair_count(worker)})"
        )

    def _verify_started_workers_progress(
        self,
        roles,
        observe_s=1.0,
        minimum_delta=10,
    ):
        before = {
            role: self._capture_pair_count(
                self.workers[role]
            )
            for role in roles
        }

        deadline = time.monotonic() + observe_s

        while time.monotonic() < deadline:
            for role in roles:
                worker = self.workers[role]

                if worker.error is not None:
                    raise RuntimeError(
                        f"{role}: {worker.error}"
                    )

                if not worker.is_alive():
                    raise RuntimeError(
                        f"{role}: capture thread stopped "
                        f"during startup verification"
                    )

            time.sleep(0.05)

        after = {
            role: self._capture_pair_count(
                self.workers[role]
            )
            for role in roles
        }

        for role in roles:
            delta = after[role] - before[role]

            if delta < minimum_delta:
                raise RuntimeError(
                    f"{role}: camera stalled during "
                    f"startup verification "
                    f"(pair_delta={delta}, "
                    f"observe_s={observe_s:.1f})"
                )

        print(
            "ACT camera startup progress: "
            + ", ".join(
                f"{role}=+"
                f"{after[role] - before[role]}"
                for role in roles
            ),
            flush=True,
        )

    def start(self):
        if self.started:
            return

        cameras = self.config["cameras"]

        self.context = Context()

        device_list = self.context.query_devices()

        for i in range(device_list.get_count()):
            device = device_list.get_device_by_index(i)
            info = device.get_device_info()

            serial = info.get_serial_number()

            print(
                f"Manager found Orbbec #{i}: "
                f"{info.get_name()} "
                f"SN={serial}",
                flush=True,
            )

            self.devices[serial] = device

        left_sn = str(
            cameras["left_wrist"]["serial_number"]
        )

        right_sn = str(
            cameras["right_wrist"]["serial_number"]
        )

        if left_sn not in self.devices:
            raise RuntimeError(
                f"left wrist camera not found: {left_sn}"
            )

        if right_sn not in self.devices:
            raise RuntimeError(
                f"right wrist camera not found: {right_sn}"
            )

        self.workers = {
            "top": RealSenseWorker(
                "top",
                cameras["top"],
            ),

            "left_wrist": OrbbecWorker(
                "left_wrist",
                cameras["left_wrist"],
                self.devices[left_sn],
            ),

            "right_wrist": OrbbecWorker(
                "right_wrist",
                cameras["right_wrist"],
                self.devices[right_sn],
            ),
        }

        for worker in self.workers.values():
            worker.episode_gate = self.episode_gate

        try:
            # IMPORTANT:
            # Prepare both Gemini devices before ANY camera
            # pipeline starts. This mirrors the independently
            # validated staged three-camera test.
            for role in (
                "right_wrist",
                "left_wrist",
            ):
                self.workers[role].prepare()

            print(
                "ACT both Orbbec workers prepared "
                "before camera streaming",
                flush=True,
            )

            # Validated USB-safe startup order:
            #
            #   top -> right_wrist -> left_wrist
            #
            # Each stage must produce real RGB+Depth frames.
            # After adding each camera, every camera already
            # started must continue making frame progress.
            startup_order = (
                "top",
                "right_wrist",
                "left_wrist",
            )

            started_roles = []

            for role in startup_order:
                worker = self.workers[role]

                worker.start()

                if not worker.ready.wait(10.0):
                    raise RuntimeError(
                        f"{role}: startup timeout"
                    )

                if worker.error is not None:
                    raise RuntimeError(
                        f"{role}: {worker.error}"
                    )

                pair_count = (
                    self._wait_for_capture_pairs(
                        role,
                        minimum_pairs=5,
                        timeout_s=8.0,
                    )
                )

                print(
                    f"[{role}] startup frames ready "
                    f"(RGB+Depth pairs={pair_count})",
                    flush=True,
                )

                started_roles.append(role)

                self._verify_started_workers_progress(
                    tuple(started_roles),
                    observe_s=2.0,
                    minimum_delta=20,
                )

            warmup_s = float(
                self.config
                .get("recording", {})
                .get("warmup_s", 2.0)
            )

            print(
                f"ACT camera manager warm-up "
                f"{warmup_s:.1f}s...",
                flush=True,
            )

            time.sleep(warmup_s)

            now = time.monotonic()

            self._health_last_pairs = {
                role: self._capture_pair_count(
                    worker
                )
                for role, worker
                in self.workers.items()
            }

            self._health_last_progress_at = {
                role: now
                for role in self.workers
            }

            self.started = True

            self.check_health()

            print(
                "ACT THREE CAMERA MANAGER READY",
                flush=True,
            )

        except Exception:
            self.close()
            raise

    def get_latest_bgr_frames(self):
        if not self.started:
            raise RuntimeError(
                "camera manager is not started"
            )

        result = {}

        for role in (
            "top",
            "left_wrist",
            "right_wrist",
        ):
            sample = (
                self.workers[role]
                .get_latest_color_bgr()
            )

            if sample is None:
                raise RuntimeError(
                    f"{role}: latest RGB frame unavailable"
                )

            result[role] = sample

        return result

    def start_episode(self, episode_dir):
        if not self.started:
            raise RuntimeError(
                "camera manager is not started"
            )

        if self.episode_active:
            raise RuntimeError(
                f"episode already active: {self.episode_id}"
            )

        self.check_health()

        episode_dir = (
            Path(episode_dir)
            .expanduser()
            .resolve()
        )

        episode_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for worker in self.workers.values():
            worker.reset_episode_counters()

        started_writers = []

        try:
            for worker in self.workers.values():
                worker.start_episode_writer(
                    episode_dir,
                    queue_size=
                        self.episode_queue_size,
                )

                started_writers.append(
                    worker
                )

        except Exception:
            for worker in started_writers:
                worker.stop_episode_writer()

            raise

        self.episode_dir = episode_dir
        self.episode_id = episode_dir.name
        self.episode_started_at = time.monotonic()
        self.episode_active = True

        # Writers are already ready before opening the gate.
        self.episode_gate.set()

        print(
            f"ACT CAMERA EPISODE START: "
            f"{self.episode_id} "
            f"host_monotonic="
            f"{self.episode_started_at:.9f}",
            flush=True,
        )

    def stop_episode(self):
        if not self.episode_active:
            return None

        # Close gate first. Pipelines keep running, but no new
        # frames may enter the episode queues.
        self.episode_gate.clear()

        stopped_at = time.monotonic()

        for worker in self.workers.values():
            worker.stop_episode_writer()

        result = {
            "schema_version":
                "r1a7_act_camera_episode_v1",
            "episode_id":
                self.episode_id,
            "episode_dir":
                str(self.episode_dir),
            "started_at":
                self.episode_started_at,
            "stopped_at":
                stopped_at,
            "duration_s": (
                stopped_at
                - self.episode_started_at
            ),
            "queue_size":
                self.episode_queue_size,
            "cameras": {
                role: worker.snapshot()
                for role, worker
                in self.workers.items()
            },
        }

        manifest_path = (
            self.episode_dir
            / "camera_recording_manifest.json"
        )

        manifest_path.write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            f"ACT CAMERA EPISODE STOP: "
            f"{self.episode_id} "
            f"elapsed={result['duration_s']:.3f}s",
            flush=True,
        )

        self.episode_active = False
        self.episode_id = None
        self.episode_dir = None
        self.episode_started_at = None

        return result

    def check_health(self):
        now = time.monotonic()

        for role, worker in self.workers.items():

            if worker.error is not None:
                raise RuntimeError(
                    f"{role}: {worker.error}"
                )

            if not worker.is_alive():
                raise RuntimeError(
                    f"{role}: capture thread stopped"
                )

            pair_count = self._capture_pair_count(
                worker
            )

            previous = self._health_last_pairs.get(
                role
            )

            if previous is None:
                self._health_last_pairs[role] = (
                    pair_count
                )
                self._health_last_progress_at[role] = (
                    now
                )
                continue

            if pair_count > previous:
                self._health_last_pairs[role] = (
                    pair_count
                )
                self._health_last_progress_at[role] = (
                    now
                )
                continue

            last_progress_at = (
                self._health_last_progress_at.get(
                    role,
                    now,
                )
            )

            stalled_for = (
                now - last_progress_at
            )

            if (
                stalled_for
                > self.health_stall_timeout_s
            ):
                raise RuntimeError(
                    f"{role}: RGB+Depth capture stalled "
                    f"for {stalled_for:.2f}s "
                    f"(pairs={pair_count})"
                )

        return True

    def counters(self):
        return {
            role: worker.snapshot()
            for role, worker
            in self.workers.items()
        }

    def close(self):
        if self.episode_active:
            self.stop_episode()
        else:
            self.episode_gate.clear()

        if not self.workers:
            self.started = False
            return

        for worker in self.workers.values():
            worker.request_stop()

        for worker in self.workers.values():
            # Thread.ident is None until Thread.start()
            # has actually succeeded. Joining an unstarted
            # thread raises:
            # RuntimeError: cannot join thread before it is started
            if worker.ident is not None:
                worker.join(timeout=5.0)

        self.started = False

        self._health_last_pairs.clear()
        self._health_last_progress_at.clear()

        print(
            "ACT THREE CAMERA MANAGER CLOSED",
            flush=True,
        )
