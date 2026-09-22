#!/usr/bin/env python3
"""R1-A7 teleoperation using Unitree's default debug-mode execution path.

The input and IK path matches the verified MuJoCo program:
TeleVuer controller poses -> R1A7_ArmIK -> R1-A7 arm joint targets.
Only the real-robot executor differs: this program publishes CRC-protected
LowCmd messages on rt/lowcmd, as Unitree's default (non-motion) XR path does.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from typing import Optional

import numpy as np

# ============================================================
# VISUAL_ABSOLUTE_XY_V1
# Camera target selection + saved Homography.
#
# Vision decides WHERE to go.
# The verified continuous Cartesian controller decides HOW.
# ============================================================

import json as _visual_json
import sys as _visual_sys
import threading as _visual_threading
import zlib as _visual_zlib
import time as _visual_time
from pathlib import Path as _VisualPath

import cv2 as _visual_cv2

_VISUAL_DIR = _VisualPath(
    "/home/robot/unitree_sim_isaaclab_threading/"
    "r1a7_threading_project/vision"
)

if str(_VISUAL_DIR) not in _visual_sys.path:
    _visual_sys.path.insert(
        0,
        str(_VISUAL_DIR),
    )

import r1a7_head_visual_servo_lowcmd as _visual_vs

# RIGID_TOOL_TRACKER_V1
from r1a7_rigid_tool_tracker import (
    RigidRoiTracker as _RigidRoiTracker,
)

_VISUAL_CALIB_PATH = (
    _VISUAL_DIR
    / "r1a7_head_absolute_xy_homography.json"
)


class _HeadCameraBuffer:
    """Continuously acquire the latest head RGB frame."""

    def __init__(self):
        self._lock = _visual_threading.Lock()
        self._frame = None

        # UNIQUE_CAMERA_FRAME_V1
        self._frame_seq = 0
        self._last_crc = None
        self._running = True
        self._last_error_print = 0.0

        self._video = _visual_vs.VideoClient()
        self._video.SetTimeout(3.0)
        self._video.Init()

        self._thread = _visual_threading.Thread(
            target=self._worker,
            daemon=True,
        )
        self._thread.start()

    def _worker(self):
        while self._running:
            try:
                result = self._video.GetImageSample()

                if (
                    not isinstance(result, tuple)
                    or len(result) < 2
                ):
                    _visual_time.sleep(0.01)
                    continue

                code = result[0]
                payload = result[1]

                if code != 0 or payload is None:
                    _visual_time.sleep(0.01)
                    continue

                # UNIQUE_CAMERA_FRAME_V1
                try:
                    raw = bytes(payload)
                except Exception:
                    raw = np.asarray(
                        payload,
                        dtype=np.uint8,
                    ).reshape(-1).tobytes()

                crc = _visual_zlib.crc32(
                    raw
                )

                if crc == self._last_crc:
                    continue

                encoded = np.frombuffer(
                    raw,
                    dtype=np.uint8,
                )

                frame = _visual_cv2.imdecode(
                    encoded,
                    _visual_cv2.IMREAD_COLOR,
                )

                if frame is None:
                    continue

                with self._lock:
                    self._frame = frame
                    self._last_crc = crc
                    self._frame_seq += 1

            except Exception as exc:
                now = _visual_time.monotonic()

                if (
                    now - self._last_error_print
                    >= 2.0
                ):
                    print(
                        "[VISION CAMERA] transient error:",
                        repr(exc),
                        flush=True,
                    )
                    self._last_error_print = now

                _visual_time.sleep(0.02)

    def snapshot(self):
        with self._lock:
            if self._frame is None:
                return None

            return self._frame.copy()

    def snapshot_with_seq(self):
        with self._lock:
            if self._frame is None:
                return None, -1

            return (
                self._frame.copy(),
                int(self._frame_seq),
            )

    def stop(self):
        self._running = False


def _load_visual_homography():
    if not _VISUAL_CALIB_PATH.exists():
        raise RuntimeError(
            "Visual Homography file not found: "
            + str(_VISUAL_CALIB_PATH)
        )

    data = _visual_json.loads(
        _VISUAL_CALIB_PATH.read_text()
    )

    if not bool(data.get("success", False)):
        raise RuntimeError(
            "Visual Homography is not marked success."
        )

    H = np.asarray(
        data["H_image_to_robot_xy_mm"],
        dtype=np.float64,
    )

    samples = list(
        data.get("samples", [])
    )

    if len(samples) < 4:
        raise RuntimeError(
            "Visual calibration has fewer than 4 samples."
        )

    image_points = np.asarray(
        [
            [
                float(sample["u"]),
                float(sample["v"]),
            ]
            for sample in samples
        ],
        dtype=np.float32,
    )

    hull = _visual_cv2.convexHull(
        image_points
    )

    return data, H, hull



# ============================================================
# VISUAL_OVERLAY_V1
# Draw calibration footprint, visual target and tool ROI.
# ============================================================

def _draw_visual_overlay(
    frame,
    calib_data,
    calib_hull,
    target_pixel=None,
    tool_roi=None,
):
    out = frame.copy()

    # 1. Draw the exact convex hull of the calibration samples.
    if calib_hull is not None:
        hull_i = np.asarray(
            calib_hull,
            dtype=np.int32,
        ).reshape(-1, 1, 2)

        _visual_cv2.polylines(
            out,
            [hull_i],
            True,
            (0, 255, 255),
            2,
        )

        if len(hull_i) > 0:
            hx, hy = hull_i[0, 0]
            _visual_cv2.putText(
                out,
                "CALIBRATED REGION",
                (int(hx), max(20, int(hy) - 8)),
                _visual_cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                _visual_cv2.LINE_AA,
            )

    # 2. Draw all nine original calibration samples.
    for sample in calib_data.get("samples", []):
        u = int(round(float(sample["u"])))
        v = int(round(float(sample["v"])))

        _visual_cv2.circle(
            out,
            (u, v),
            3,
            (255, 255, 0),
            -1,
        )

    # 3. Persistently display selected target.
    if target_pixel is not None:
        u = int(round(float(target_pixel[0])))
        v = int(round(float(target_pixel[1])))

        _visual_cv2.drawMarker(
            out,
            (u, v),
            (0, 0, 255),
            markerType=_visual_cv2.MARKER_CROSS,
            markerSize=24,
            thickness=2,
        )

        _visual_cv2.circle(
            out,
            (u, v),
            8,
            (0, 0, 255),
            2,
        )

        _visual_cv2.putText(
            out,
            "TARGET",
            (u + 12, v - 12),
            _visual_cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            _visual_cv2.LINE_AA,
        )

    # 4. Display manually selected tweezer/tool ROI.
    if tool_roi is not None:
        x, y, w, h = [
            int(round(float(v)))
            for v in tool_roi
        ]

        _visual_cv2.rectangle(
            out,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2,
        )

        cx = int(round(x + w / 2.0))
        cy = int(round(y + h / 2.0))

        _visual_cv2.circle(
            out,
            (cx, cy),
            5,
            (0, 255, 0),
            -1,
        )

        _visual_cv2.putText(
            out,
            "TOOL ROI",
            (x, max(20, y - 8)),
            _visual_cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            _visual_cv2.LINE_AA,
        )

    return out


def _visual_pixel_to_xy(H, pixel):
    uv1 = np.asarray(
        [
            float(pixel[0]),
            float(pixel[1]),
            1.0,
        ],
        dtype=np.float64,
    )

    result = H @ uv1

    if abs(float(result[2])) < 1.0e-12:
        raise RuntimeError(
            "Invalid Homography denominator."
        )

    return (
        result[:2]
        / result[2]
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XR_ROOT = Path("/home/robot/R1A7_VR_dual_arm_transfer_20260831_001/robot_dev/xr_teleoperate")
DEFAULT_SDK_PYTHON = Path("/home/robot/R1A7_VR_dual_arm_transfer_20260831_001/robot_dev/unitree_sdk2_python")

# Confirmed from the R1-A7 low-level examples and live lowstate diagnostics.
ARM_INDICES = tuple(range(15, 29))
UPPER_BODY_INDICES = (13, *range(15, 31))
GRIPPER_INDICES = (31, 33)
GRIPPER_OPEN_Q = np.asarray([4.86, 4.80], dtype=float)
GRIPPER_CLOSE_Q = np.asarray([-0.08, -0.20], dtype=float)

# Gains from the repository's R1-A7 low-level examples. Waist and head remain
# at the measured startup posture; only the 14 arm targets are changed by IK.
UPPER_BODY_KP = np.asarray(
    [60.0, 100.0, 100.0, 100.0, 100.0, 50.0, 35.0, 35.0,
     100.0, 100.0, 100.0, 100.0, 50.0, 35.0, 35.0, 50.0, 10.0],
    dtype=float,
)
UPPER_BODY_KD = np.asarray(
    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.2, 1.2,
     2.0, 2.0, 2.0, 2.0, 2.0, 1.2, 1.2, 2.0, 0.1],
    dtype=float,
)


def validate_robot_interface(interface: str) -> str:
    path = Path("/sys/class/net") / interface
    if not path.exists():
        raise RuntimeError(f"robot DDS interface {interface!r} does not exist")
    operstate = (path / "operstate").read_text(encoding="ascii").strip()
    carrier_path = path / "carrier"
    carrier = (
        carrier_path.read_text(encoding="ascii").strip()
        if carrier_path.exists()
        else "unknown"
    )
    if operstate != "up" or carrier == "0":
        raise RuntimeError(
            f"robot DDS interface {interface!r} is disconnected "
            f"(operstate={operstate}, carrier={carrier}); check the robot Ethernet cable"
        )
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            request = struct.pack("256s", interface[:15].encode("ascii"))
            address = socket.inet_ntoa(
                fcntl.ioctl(sock.fileno(), 0x8915, request)[20:24]
            )
        except OSError as exc:
            raise RuntimeError(
                f"robot DDS interface {interface!r} has no IPv4 address"
            ) from exc
    return address


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official TeleVuer/R1A7 IK with Unitree debug-mode rt/lowcmd output."
    )
    parser.add_argument("--interface", default="eno1")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--host-ip", required=True)
    parser.add_argument("--state-topic", default="rt/lowstate")
    parser.add_argument("--command-topic", default="rt/lowcmd")
    parser.add_argument("--ik-frequency", type=float, default=30.0)
    parser.add_argument("--publish-frequency", type=float, default=250.0)
    parser.add_argument("--max-joint-speed", type=float, default=2.0)
    parser.add_argument("--lowstate-timeout", type=float, default=0.25)
    parser.add_argument("--xr-stale-timeout", type=float, default=0.50)
    parser.add_argument("--session-close-timeout", type=float, default=1.0)
    parser.add_argument("--max-start-error-rad", type=float, default=0.35)
    parser.add_argument(
        "--controller-rotation-mode",
        choices=("spatial", "local"),
        default="spatial",
    )
    parser.add_argument("--print-period", type=float, default=0.5)
    parser.add_argument(
        "--record-root",
        type=Path,
        default=Path("/home/robot/unitree_sim_isaaclab/r1a7_wrench_project/data/episodes"),
    )
    parser.add_argument("--record-episode-id", default="r1a7_vr_001")
    parser.add_argument("--record-duration-s", type=float, default=60.0)
    parser.add_argument("--record-d435i", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--d435i-recorder-mode",
        choices=("opencv", "rs-record"),
        default="opencv",
    )
    parser.add_argument(
        "--d435i-opencv-script",
        type=Path,
        default=Path("/home/robot/unitree_sim_isaaclab/r1a7_wrench_project/scripts/record_d435i_color_depth.py"),
    )
    parser.add_argument("--d435i-python-bin", default="/usr/bin/python3")
    parser.add_argument("--rs-record-bin", default="rs-record")
    parser.add_argument("--d435i-filename", default="d435i.db3")
    parser.add_argument("--enable-gripper", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gripper-initial-mode",
        choices=("current", "open"),
        default="current",
        help="Initialize gripper target from measured current q or from the original fully-open command.",
    )
    parser.add_argument(
        "--right-gripper-open-cap-current",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When using current gripper initialization, prevent the right gripper from opening beyond its startup q.",
    )
    parser.add_argument("--gripper-kp", type=float, default=8.0)
    parser.add_argument("--gripper-kd", type=float, default=1.5)
    parser.add_argument("--gripper-speed", type=float, default=1.5)
    parser.add_argument("--gripper-contact-hold", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gripper-contact-error", type=float, default=0.08)
    parser.add_argument("--gripper-contact-stall-eps", type=float, default=0.004)
    parser.add_argument("--gripper-contact-stall-time", type=float, default=0.25)
    parser.add_argument("--gripper-contact-hold-bias", type=float, default=0.035)
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("/tmp/r1a7_official_real_lowcmd.log"),
    )
    parser.add_argument(
        "--xr-root",
        type=Path,
        default=Path(os.environ.get("XR_TELEOP_ROOT", DEFAULT_XR_ROOT)),
    )
    parser.add_argument("--sdk-python-root", type=Path, default=DEFAULT_SDK_PYTHON)
    args = parser.parse_args()
    for name in ("ik_frequency", "publish_frequency", "max_joint_speed"):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    if args.command_topic != "rt/lowcmd":
        parser.error("this debug-mode executor only permits --command-topic rt/lowcmd")
    return args


def tele_session_active(tele) -> bool:
    """Accept current TeleVuer wrappers that do not expose vuer_session_active.

    In this transfer package the Quest wrapper may report controller buttons and
    motion_data_ready while vuer_session_active stays absent/false.  Treat fresh
    motion data as an active WebXR session so the real robot is not blocked by a
    wrapper field mismatch.
    """
    if bool(getattr(tele, "vuer_session_active", False)):
        return True
    return bool(getattr(tele, "motion_data_ready", False))


def tele_xr_age(tele, now: float) -> float:
    event_time = float(getattr(tele, "motion_event_time", 0.0))
    if event_time > 0.0:
        return now - event_time
    if bool(getattr(tele, "motion_data_ready", False)):
        return 0.0
    return float("inf")


class ButtonEdge:
    def __init__(self):
        self.previous: dict[str, bool] = {}

    def rising(self, tele, name: str) -> bool:
        pressed = bool(getattr(tele, name, False))
        was_pressed = self.previous.get(name, False)
        self.previous[name] = pressed
        return pressed and not was_pressed


class SimpleCsvRecorder:
    def __init__(
        self,
        root: Path,
        episode_id: str,
        record_d435i: bool = True,
        d435i_recorder_mode: str = "opencv",
        d435i_opencv_script: Path = Path("/home/robot/unitree_sim_isaaclab/r1a7_wrench_project/scripts/record_d435i_color_depth.py"),
        d435i_python_bin: str = "/usr/bin/python3",
        rs_record_bin: str = "rs-record",
        d435i_filename: str = "d435i.db3",
        record_duration_s: float = 60.0,
    ):
        self.root = root.expanduser().resolve()
        self.episode_id = episode_id
        self.record_d435i = bool(record_d435i)
        self.d435i_recorder_mode = str(d435i_recorder_mode)
        self.d435i_opencv_script = d435i_opencv_script.expanduser().resolve()
        self.d435i_python_bin = str(d435i_python_bin)
        self.rs_record_bin = str(rs_record_bin)
        self.d435i_filename = str(d435i_filename)
        self.record_duration_s = max(1.0, float(record_duration_s))
        self.active = False
        self.episode_dir: Optional[Path] = None
        self.file = None
        self.writer: Optional[csv.writer] = None
        self.d435i_record_process: Optional[subprocess.Popen] = None
        self.d435i_record_file: Optional[Path] = None
        self.d435i_record_log_file = None
        self.samples = 0
        self.started_at = 0.0

    def _next_dir(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        text = self.episode_id.strip() or "r1a7_vr_001"
        prefix = text.rstrip("_")
        start = 1
        width = 3
        import re
        match = re.match(r"^(.*?)(\d+)$", text)
        if match:
            prefix = match.group(1).rstrip("_")
            start = int(match.group(2))
            width = len(match.group(2))
        for number in range(start, 1000000):
            path = self.root / f"{prefix}_{number:0{width}d}"
            if not path.exists():
                return path
        raise RuntimeError(f"cannot allocate recording dir under {self.root}")

    def toggle(self) -> None:
        if self.active:
            self.stop()
        else:
            self.start()

    def start(self) -> None:
        if self.active:
            return
        self.episode_dir = self._next_dir()
        self.episode_dir.mkdir(parents=True, exist_ok=False)
        metadata = {
            "robot": "Unitree R1-A7",
            "mode": "R1A7_ArmIK TeleVuer lowcmd simple csv",
            "episode_id": self.episode_dir.name,
            "created_at_system": time.time(),
            "recording_mode": "x_toggle_robot_csv_plus_d435i_db3",
            "record_duration_s": self.record_duration_s,
            "d435i_record_enabled": self.record_d435i,
            "d435i_recorder_mode": self.d435i_recorder_mode,
            "d435i_filename": self.d435i_filename,
            "d435i_color_video": "d435i_color.mp4",
            "d435i_depth_preview_video": "d435i_depth_preview.mp4",
            "note": "X toggles robot CSV and D435i recording together in software; this is not hardware synchronization.",
        }
        (self.episode_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.file = (self.episode_dir / "states.csv").open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            "time_monotonic", "tracking_enabled", "input_valid", "xr_age",
            "arm_q", "arm_dq", "goal_arm_q", "sent_arm_q",
            "measured_gripper_q", "measured_gripper_dq", "goal_gripper_q",
            "sent_gripper_q", "gripper_contact_hold", "gripper_contact_q",
            "left_wrist_pose", "right_wrist_pose",
            "left_trigger", "right_trigger", "right_A", "left_X",
        ])
        self.active = True
        self.samples = 0
        self.started_at = time.monotonic()
        print(f"X RECORD START: {self.episode_dir}", flush=True)
        self._start_d435i()

    def _start_d435i(self) -> None:
        if not self.record_d435i or self.episode_dir is None:
            return
        self._stop_camera_conflicts()
        self.d435i_record_file = self.episode_dir / self.d435i_filename
        log_path = self.episode_dir / "d435i_record.log"
        try:
            self.d435i_record_log_file = log_path.open(
                "w",
                encoding="utf-8",
                buffering=1,
            )
            if self.d435i_recorder_mode == "opencv":
                cmd = [
                    self.d435i_python_bin,
                    str(self.d435i_opencv_script),
                    str(self.episode_dir),
                    "--duration-s",
                    str(max(1, int(round(self.record_duration_s)))),
                ]
            else:
                cmd = [
                    self.rs_record_bin,
                    "-f",
                    str(self.d435i_record_file),
                    "-t",
                    str(max(1, int(round(self.record_duration_s)))),
                ]
            camera_env = dict(os.environ)
            camera_env.pop("PYTHONNOUSERSITE", None)
            self.d435i_record_process = subprocess.Popen(
                cmd,
                stdout=self.d435i_record_log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=camera_env,
            )
            print(
                "D435i RECORD START: "
                f"pid={self.d435i_record_process.pid} file={self.d435i_record_file}",
                flush=True,
            )
        except Exception as exc:
            self.d435i_record_process = None
            if self.d435i_record_log_file is not None:
                self.d435i_record_log_file.close()
                self.d435i_record_log_file = None
            print(
                "D435i RECORD START FAILED: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    def _stop_camera_conflicts(self) -> None:
        patterns = (
            "realsense-viewer",
            "test_three_cameras.py",
            "rs-record",
        )
        for pattern in patterns:
            try:
                subprocess.run(
                    ["pkill", "-TERM", "-f", pattern],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass
        time.sleep(0.5)

    def _stop_d435i(self) -> None:
        proc = self.d435i_record_process
        if proc is None:
            if self.d435i_record_log_file is not None:
                self.d435i_record_log_file.close()
                self.d435i_record_log_file = None
            return
        if proc.poll() is None:
            try:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)
        print(
            "D435i RECORD STOP: "
            f"returncode={proc.returncode} file={self.d435i_record_file}",
            flush=True,
        )
        self.d435i_record_process = None
        if self.d435i_record_log_file is not None:
            self.d435i_record_log_file.flush()
            self.d435i_record_log_file.close()
            self.d435i_record_log_file = None

    def write(
        self,
        tele,
        tracking_enabled: bool,
        input_valid: bool,
        xr_age: float,
        arm_q: np.ndarray,
        arm_dq: np.ndarray,
        goal_arm_q: np.ndarray,
        sent_arm_q: np.ndarray,
        measured_gripper_q: np.ndarray,
        measured_gripper_dq: np.ndarray,
        goal_gripper_q: np.ndarray,
        sent_gripper_q: np.ndarray,
        gripper_contact_hold: np.ndarray,
        gripper_contact_q: np.ndarray,
    ) -> None:
        if not self.active or self.writer is None:
            return
        self.writer.writerow([
            f"{time.monotonic():.6f}",
            int(tracking_enabled),
            int(input_valid),
            f"{xr_age:.6f}",
            json.dumps(np.asarray(arm_q, dtype=float).tolist()),
            json.dumps(np.asarray(arm_dq, dtype=float).tolist()),
            json.dumps(np.asarray(goal_arm_q, dtype=float).tolist()),
            json.dumps(np.asarray(sent_arm_q, dtype=float).tolist()),
            json.dumps(np.asarray(measured_gripper_q, dtype=float).tolist()),
            json.dumps(np.asarray(measured_gripper_dq, dtype=float).tolist()),
            json.dumps(np.asarray(goal_gripper_q, dtype=float).tolist()),
            json.dumps(np.asarray(sent_gripper_q, dtype=float).tolist()),
            json.dumps(np.asarray(gripper_contact_hold, dtype=bool).astype(int).tolist()),
            json.dumps(np.asarray(gripper_contact_q, dtype=float).tolist()),
            json.dumps(np.asarray(getattr(tele, "left_wrist_pose", np.eye(4)), dtype=float).reshape(4, 4).tolist()),
            json.dumps(np.asarray(getattr(tele, "right_wrist_pose", np.eye(4)), dtype=float).reshape(4, 4).tolist()),
            float(getattr(tele, "left_ctrl_triggerValue", 10.0)),
            float(getattr(tele, "right_ctrl_triggerValue", 10.0)),
            int(bool(getattr(tele, "right_ctrl_aButton", False))),
            int(bool(getattr(tele, "left_ctrl_aButton", False))),
        ])
        self.samples += 1
        if self.file is not None:
            self.file.flush()

    def stop(self) -> None:
        if not self.active:
            self._stop_d435i()
            return
        elapsed = time.monotonic() - self.started_at
        print(
            f"X RECORD STOP: {self.episode_dir} samples={self.samples} elapsed={elapsed:.3f}s",
            flush=True,
        )
        self.active = False
        if self.file is not None:
            self.file.flush()
            self.file.close()
        self.file = None
        self._stop_d435i()
        self.writer = None


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


class StateBuffer:
    def __init__(self, crc):
        self._crc = crc
        self._lock = threading.Lock()
        self._message = None
        self._received_at = 0.0
        self.received_count = 0
        self.count = 0
        self.crc_errors = 0

    def callback(self, message) -> None:
        self.received_count += 1
        try:
            if int(message.crc) != int(self._crc.Crc(message)):
                self.crc_errors += 1
                return
        except Exception:
            self.crc_errors += 1
            return
        with self._lock:
            self._message = message
            self._received_at = time.monotonic()
            self.count += 1

    def snapshot(self) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float]]:
        with self._lock:
            if self._message is None:
                return None
            message = self._message
            upper_q = np.asarray(
                [message.motor_state[index].q for index in UPPER_BODY_INDICES],
                dtype=float,
            )
            arm_q = np.asarray(
                [message.motor_state[index].q for index in ARM_INDICES],
                dtype=float,
            )
            arm_dq = np.asarray(
                [message.motor_state[index].dq for index in ARM_INDICES],
                dtype=float,
            )
            gripper_q = np.asarray(
                [message.motor_state[index].q for index in GRIPPER_INDICES],
                dtype=float,
            )
            gripper_dq = np.asarray(
                [message.motor_state[index].dq for index in GRIPPER_INDICES],
                dtype=float,
            )
            return (
                upper_q,
                arm_q,
                arm_dq,
                gripper_q,
                gripper_dq,
                int(message.mode_machine),
                self._received_at,
            )


class R1A7LowCmdOutput:
    def __init__(
        self,
        publisher,
        low_cmd_factory,
        crc,
        publish_frequency: float,
        max_joint_speed: float,
        enable_gripper: bool = False,
        gripper_kp: float = 8.0,
        gripper_kd: float = 1.5,
        gripper_speed: float = 1.5,
        gripper_contact_hold: bool = True,
        gripper_contact_error: float = 0.08,
        gripper_contact_stall_eps: float = 0.004,
        gripper_contact_stall_time: float = 0.25,
        gripper_contact_hold_bias: float = 0.035,
    ):
        self.publisher = publisher
        self.low_cmd_factory = low_cmd_factory
        self.crc = crc
        self.period = 1.0 / publish_frequency
        self.max_step = max_joint_speed * self.period
        self.lock = threading.Lock()
        self.upper_target = np.zeros(len(UPPER_BODY_INDICES), dtype=float)
        self.arm_goal = np.zeros(len(ARM_INDICES), dtype=float)
        self.enable_gripper = bool(enable_gripper)
        self.gripper_target = GRIPPER_OPEN_Q.copy()
        self.gripper_goal = GRIPPER_OPEN_Q.copy()
        self.gripper_max_step = max(0.0, float(gripper_speed)) * self.period
        self.gripper_kp = float(gripper_kp)
        self.gripper_kd = float(gripper_kd)
        self.gripper_contact_hold = bool(gripper_contact_hold)
        self.gripper_contact_error = float(gripper_contact_error)
        self.gripper_contact_stall_eps = float(gripper_contact_stall_eps)
        self.gripper_contact_stall_time = float(gripper_contact_stall_time)
        self.gripper_contact_hold_bias = float(gripper_contact_hold_bias)
        self.gripper_stall_since = np.full(2, np.nan, dtype=float)
        self.gripper_hold = np.zeros(2, dtype=bool)
        self.gripper_contact_q = np.full(2, np.nan, dtype=float)
        self.gripper_prev_state_q: Optional[np.ndarray] = None
        self.gripper_open_cap_q: Optional[np.ndarray] = None
        self.mode_machine = 0
        self.enabled = False
        self.stop = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.publish_count = 0
        self.error = ""

    def enable(
        self,
        upper_q: np.ndarray,
        mode_machine: int,
        gripper_q: Optional[np.ndarray] = None,
        gripper_initial_mode: str = "open",
        right_gripper_open_cap_current: bool = False,
    ) -> None:
        upper = np.asarray(upper_q, dtype=float).reshape(len(UPPER_BODY_INDICES))
        with self.lock:
            self.upper_target = upper.copy()
            self.arm_goal = upper[1:15].copy()
            if self.enable_gripper and gripper_initial_mode == "current" and gripper_q is not None:
                initial_gripper = np.asarray(gripper_q, dtype=float).reshape(2)
                self.gripper_target = initial_gripper.copy()
                self.gripper_goal = initial_gripper.copy()
                self.gripper_open_cap_q = None
                if right_gripper_open_cap_current:
                    self.gripper_open_cap_q = np.asarray(
                        [GRIPPER_OPEN_Q[0], initial_gripper[1]],
                        dtype=float,
                    )
            else:
                self.gripper_target = GRIPPER_OPEN_Q.copy()
                self.gripper_goal = GRIPPER_OPEN_Q.copy()
                self.gripper_open_cap_q = None
            self.gripper_stall_since[:] = np.nan
            self.gripper_hold[:] = False
            self.gripper_contact_q[:] = np.nan
            self.gripper_prev_state_q = None
            self.mode_machine = int(mode_machine)
            self.enabled = True
        self.stop.clear()
        self.thread = threading.Thread(
            target=self._loop,
            name="r1a7-official-lowcmd-250hz",
            daemon=True,
        )
        self.thread.start()

    def set_arm_goal(self, arm_q: np.ndarray, mode_machine: int) -> None:
        goal = np.asarray(arm_q, dtype=float).reshape(len(ARM_INDICES))
        if not np.all(np.isfinite(goal)):
            raise ValueError("refusing non-finite arm target")
        with self.lock:
            self.arm_goal = goal.copy()
            self.mode_machine = int(mode_machine)

    def set_waist_goal(self, waist_yaw: float, mode_machine: int) -> None:
        if not np.isfinite(waist_yaw):
            raise ValueError("refusing non-finite waist target")
        with self.lock:
            self.upper_target[0] = float(waist_yaw)
            self.mode_machine = int(mode_machine)

    def hold_measured(self, arm_q: np.ndarray, mode_machine: int) -> None:
        self.set_arm_goal(arm_q, mode_machine)

    def set_gripper_from_triggers(
        self,
        left_trigger: float,
        right_trigger: float,
    ) -> None:
        if not self.enable_gripper:
            return

        # TeleVuer:
        #   10.0 = released
        #    0.0 = fully pulled
        right_alpha = float(
            np.clip(
                (10.0 - float(right_trigger)) / 10.0,
                0.0,
                1.0,
            )
        )

        # Current tweezer calibration.
        # Larger q = open, smaller q = close.
        right_open_q = 0.1585
        right_close_q = -0.0500

        with self.lock:
            goal = self.gripper_goal.copy()

            # RIGHT trigger directly controls the tweezer.
            #
            # Trigger released:
            #   goal = 0.1585
            # Therefore immediately after VR control starts, if the trigger
            # is released, the gripper automatically moves from its current
            # position to the tweezer-open/contact position.
            #
            # Trigger pulled:
            #   goal decreases toward -0.0500
            # Left gripper: trigger released=OPEN, fully pulled=CLOSE.
            left_open_q = 5.34765
            left_close_q = 0.00000
            left_alpha = np.clip(
                (10.0 - left_trigger) / 10.0,
                0.0,
                1.0,
            )
            goal[0] = left_open_q + left_alpha * (
                left_close_q - left_open_q
            )

            goal[1] = right_open_q + right_alpha * (
                right_close_q - right_open_q
            )

            goal[1] = float(
                np.clip(
                    goal[1],
                    right_close_q,
                    right_open_q,
                )
            )

            # Clear previous contact hold when opening.
            if goal[1] > self.gripper_goal[1] + 0.0003:
                self.gripper_hold[1] = False
                self.gripper_stall_since[1] = np.nan
                self.gripper_contact_q[1] = np.nan

            self.gripper_goal = goal.copy()

    def update_gripper_contact(self, gripper_q: np.ndarray, gripper_dq: np.ndarray, now: float) -> None:
        if not self.enable_gripper or not self.gripper_contact_hold:
            return
        q = np.asarray(gripper_q, dtype=float).reshape(2)
        with self.lock:
            prev_q = self.gripper_prev_state_q
            close_dir = np.sign(GRIPPER_CLOSE_Q - GRIPPER_OPEN_Q)
            for index in range(2):
                pulled = self.gripper_goal[index] <= (
                    GRIPPER_OPEN_Q[index]
                    + 0.15 * (GRIPPER_CLOSE_Q[index] - GRIPPER_OPEN_Q[index])
                )
                if not pulled:
                    self.gripper_stall_since[index] = np.nan
                    self.gripper_hold[index] = False
                    self.gripper_contact_q[index] = np.nan
                    continue

                target_error = abs(self.gripper_goal[index] - q[index])
                state_delta = (
                    abs(q[index] - prev_q[index])
                    if prev_q is not None
                    else float("inf")
                )
                still_blocked = target_error >= self.gripper_contact_error
                barely_moving = state_delta <= self.gripper_contact_stall_eps

                if self.gripper_hold[index]:
                    contact_q = self.gripper_contact_q[index]
                    if not np.isfinite(contact_q):
                        contact_q = q[index]
                        self.gripper_contact_q[index] = contact_q
                elif still_blocked and barely_moving:
                    if np.isnan(self.gripper_stall_since[index]):
                        self.gripper_stall_since[index] = now
                    contact_q = q[index]
                    if now - self.gripper_stall_since[index] >= self.gripper_contact_stall_time:
                        self.gripper_hold[index] = True
                        self.gripper_contact_q[index] = q[index]
                        contact_q = q[index]
                else:
                    self.gripper_stall_since[index] = np.nan
                    continue

                if self.gripper_hold[index]:
                    hold_target = contact_q + close_dir[index] * abs(self.gripper_contact_hold_bias)
                    low = min(GRIPPER_OPEN_Q[index], GRIPPER_CLOSE_Q[index])
                    high = max(GRIPPER_OPEN_Q[index], GRIPPER_CLOSE_Q[index])
                    hold_target = float(np.clip(hold_target, low, high))
                    self.gripper_target[index] = hold_target
                    self.gripper_goal[index] = hold_target
            self.gripper_prev_state_q = q.copy()

    def _build_message(self):
        message = self.low_cmd_factory()
        with self.lock:
            delta = np.clip(
                self.arm_goal - self.upper_target[1:15],
                -self.max_step,
                self.max_step,
            )
            self.upper_target[1:15] += delta
            if self.enable_gripper:
                gripper_delta = np.clip(
                    self.gripper_goal - self.gripper_target,
                    -self.gripper_max_step,
                    self.gripper_max_step,
                )
                self.gripper_target += gripper_delta
                gripper_target = self.gripper_target.copy()
            else:
                gripper_target = None
            upper_target = self.upper_target.copy()
            mode_machine = self.mode_machine

        message.mode_pr = 0
        message.mode_machine = int(mode_machine)
        for local_index, motor_index in enumerate(UPPER_BODY_INDICES):
            motor = message.motor_cmd[motor_index]
            motor.mode = 1
            motor.q = float(upper_target[local_index])
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = float(UPPER_BODY_KP[local_index])
            motor.kd = float(UPPER_BODY_KD[local_index])
        if gripper_target is not None:
            for local_index, motor_index in enumerate(GRIPPER_INDICES):
                motor = message.motor_cmd[motor_index]
                motor.mode = 1
                motor.q = float(gripper_target[local_index])
                motor.dq = 0.0
                motor.tau = 0.0
                # Left motor 31 uses normal gripper gain.
                # Right motor 33 carries the tweezer and needs higher stiffness.
                if motor_index == 33:
                    motor.kp = 25.0
                    motor.kd = 0.4
                else:
                    motor.kp = self.gripper_kp
                    motor.kd = self.gripper_kd
        message.crc = self.crc.Crc(message)
        return message

    def _loop(self) -> None:
        next_tick = time.monotonic()
        while not self.stop.is_set():
            try:
                self.publisher.Write(self._build_message())
                self.publish_count += 1
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                self.stop.set()
                break
            next_tick += self.period
            wait_time = next_tick - time.monotonic()
            if wait_time > 0.0:
                self.stop.wait(wait_time)
            else:
                next_tick = time.monotonic()

    def close(self) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
        self.enabled = False


def check_debug_mode(MotionSwitcherClient, attempts: int = 3) -> bool:
    switcher = MotionSwitcherClient()
    switcher.SetTimeout(1.0)
    switcher.Init()
    last_status = None
    last_result = None
    for attempt in range(1, max(1, attempts) + 1):
        status, result = switcher.CheckMode()
        last_status, last_result = status, result
        if status == 0:
            mode_name = str((result or {}).get("name", ""))
            if mode_name:
                raise RuntimeError(
                    f"robot motion service {mode_name!r} is active; switch the robot "
                    "to debug mode before using rt/lowcmd"
                )
            print(
                "MotionSwitcher: debug mode confirmed (no active motion service).",
                flush=True,
            )
            return True
        print(
            f"MotionSwitcher check {attempt}/{attempts} did not respond "
            f"(status={status}); retrying...",
            flush=True,
        )
        time.sleep(0.5)

    # Unitree's official default teleoperation path logs a failed mode query and
    # continues. Some R1 firmware does not expose the generic switcher RPC even
    # though debug-mode lowstate/lowcmd DDS is available.
    print(
        "WARNING: MotionSwitcher mode could not be verified "
        f"(status={last_status}, result={last_result}).",
        flush=True,
    )
    print(
        "No mode switch was attempted. Confirm the robot is already in its "
        "manufacturer debug/low-level state before typing ENABLE.",
        flush=True,
    )
    return False


def main() -> int:
    args = parse_args()
    log_file = args.log_file.expanduser().open("w", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    print(f"Logging to {args.log_file.expanduser()}", flush=True)

    xr_root = args.xr_root.expanduser().resolve()
    sdk_python_root = args.sdk_python_root.expanduser().resolve()
    if not (xr_root / "teleop").is_dir():
        raise RuntimeError(f"official xr_teleoperate checkout not found: {xr_root}")
    if not (sdk_python_root / "unitree_sdk2py").is_dir():
        raise RuntimeError(f"official unitree_sdk2_python checkout not found: {sdk_python_root}")
    robot_ip = validate_robot_interface(args.interface)
    print(f"Robot DDS interface: {args.interface} ({robot_ip})", flush=True)

    sys.path.insert(0, str(sdk_python_root))
    sys.path.insert(0, str(xr_root))
    os.chdir(xr_root / "teleop")

    from televuer import TeleVuerWrapper
    from teleop.robot_control.robot_arm_ik import R1A7_ArmIK
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
        MotionSwitcherClient,
    )
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC
    from r1a7_lowcmd_guard import acquire_lowcmd_guard, ensure_tcp_port_available

    ensure_tcp_port_available(8012)
    guard = acquire_lowcmd_guard(Path(__file__).name, topic=args.command_topic)
    tv = None
    output = None
    recorder = SimpleCsvRecorder(
        args.record_root,
        args.record_episode_id,
        record_d435i=args.record_d435i,
        d435i_recorder_mode=args.d435i_recorder_mode,
        d435i_opencv_script=args.d435i_opencv_script,
        d435i_python_bin=args.d435i_python_bin,
        rs_record_bin=args.rs_record_bin,
        d435i_filename=args.d435i_filename,
        record_duration_s=args.record_duration_s,
    )
    stop = threading.Event()

    def request_stop(_signum=None, _frame=None) -> None:
        stop.set()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        ChannelFactoryInitialize(args.domain_id, args.interface)
        crc = CRC()
        state_buffer = StateBuffer(crc)
        subscriber = ChannelSubscriber(args.state_topic, LowState_)
        subscriber.Init(state_buffer.callback, 10)

        deadline = time.monotonic() + 10.0
        state = state_buffer.snapshot()
        while state is None and time.monotonic() < deadline and not stop.is_set():
            time.sleep(0.05)
            state = state_buffer.snapshot()
        if state is None:
            raise RuntimeError(
                f"no valid {args.state_topic} received "
                f"(raw={state_buffer.received_count}, crc_errors={state_buffer.crc_errors}); "
                "verify robot power, Ethernet connection, DDS domain and topic; "
                "no command was sent"
            )
        upper_q, arm_q, _arm_dq, _gripper_q, _gripper_dq, _mode_machine, _received_at = state
        print("Measured R1-A7 arm q:", np.round(arm_q, 4).tolist(), flush=True)

        check_debug_mode(MotionSwitcherClient)
        publisher = ChannelPublisher(args.command_topic, LowCmd_)
        publisher.Init()
        # SCRIPTED_CARTESIAN_STREAM_V1
        # No VR/Quest input is used in this mode.
        tv = None
        ik = R1A7_ArmIK(Unit_Test=False, Visualization=False)
        output = R1A7LowCmdOutput(
            publisher,
            unitree_hg_msg_dds__LowCmd_,
            crc,
            args.publish_frequency,
            args.max_joint_speed,
            enable_gripper=args.enable_gripper,
            gripper_kp=args.gripper_kp,
            gripper_kd=args.gripper_kd,
            gripper_speed=args.gripper_speed,
            gripper_contact_hold=args.gripper_contact_hold,
            gripper_contact_error=args.gripper_contact_error,
            gripper_contact_stall_eps=args.gripper_contact_stall_eps,
            gripper_contact_stall_time=args.gripper_contact_stall_time,
            gripper_contact_hold_bias=args.gripper_contact_hold_bias,
        )

        print("WARNING: real R1-A7 debug-mode control on rt/lowcmd.", flush=True)
        if args.enable_gripper:
            print(
                "Internal DEX1 gripper enabled on LowCmd motors 31/33: "
                "trigger released=open, trigger pulled=close; "
                f"contact_hold={args.gripper_contact_hold}.",
                flush=True,
            )
        print("MuJoCo and rt/arm_sdk are not used.", flush=True)
        print("Keep the emergency stop ready and clear both arm workspaces.", flush=True)
        print(
            "Scripted Cartesian stream mode: "
            "VR / Quest input is NOT used.",
            flush=True,
        )

        answer = input("Type ENABLE to arm this program (still no command is sent): ").strip()
        if answer.casefold() != "enable":
            print("Aborted before publishing.", flush=True)
            return 2

        # ====================================================
        # SCRIPTED_CARTESIAN_STREAM_V1
        #
        # Continuous Cartesian streaming controller.
        #
        # W = +X = forward
        # S = -X = backward
        # A = +Y = left
        # D = -Y = right
        # U = +Z = up
        # J = -Z = down
        # H = smooth return to startup arm q
        # Q = quit
        #
        # No NAV_PLAN / NAV_WAIT / PATH_STEP_MM.
        # No Cartesian stop-and-go stepping.
        # No artificial joint-settle gate.
        #
        # IK call intentionally matches the successful VR path:
        #
        #   ik.solve_ik(left_target, right_target, arm_q, arm_dq)
        #
        # LowCmdOutput keeps the original 250 Hz joint velocity
        # limiting used by the VR controller.
        # ====================================================

        import pinocchio as pin

        XY_MOVE_MM = 30.0
        Z_MOVE_MM = 10.0

        CARTESIAN_DURATION_S = 5.0
        HOME_DURATION_S = 3.0

        direction_delta_mm = {
            "w": np.asarray([+XY_MOVE_MM, 0.0, 0.0], dtype=float),
            "s": np.asarray([-XY_MOVE_MM, 0.0, 0.0], dtype=float),
            "a": np.asarray([0.0, +XY_MOVE_MM, 0.0], dtype=float),
            "d": np.asarray([0.0, -XY_MOVE_MM, 0.0], dtype=float),
            "u": np.asarray([0.0, 0.0, +Z_MOVE_MM], dtype=float),
            "j": np.asarray([0.0, 0.0, -Z_MOVE_MM], dtype=float),

            # VR-like coupled forward-and-up motion.
            # X and Z change simultaneously in one IK trajectory.
            "i": np.asarray([+XY_MOVE_MM, 0.0, +Z_MOVE_MM], dtype=float),
        }

        direction_name = {
            "w": "FORWARD +X",
            "s": "BACKWARD -X",
            "a": "LEFT +Y",
            "d": "RIGHT -Y",
            "u": "UP +Z",
            "j": "DOWN -Z",
            "i": "FORWARD-UP +X+Z",
        }

        def quintic_time_scaling(value):
            value = float(
                np.clip(
                    value,
                    0.0,
                    1.0,
                )
            )

            return (
                10.0 * value ** 3
                - 15.0 * value ** 4
                + 6.0 * value ** 5
            )

        def get_ee_poses_from_q(q):
            q = np.asarray(
                q,
                dtype=float,
            ).reshape(14)

            pin.framesForwardKinematics(
                ik.reduced_robot.model,
                ik.reduced_robot.data,
                q,
            )

            left_pose = np.asarray(
                ik.reduced_robot.data.oMf[
                    ik.L_hand_id
                ].homogeneous,
                dtype=float,
            ).copy()

            right_pose = np.asarray(
                ik.reduced_robot.data.oMf[
                    ik.R_hand_id
                ].homogeneous,
                dtype=float,
            ).copy()

            return left_pose, right_pose

        # ----------------------------------------------------
        # Re-read fresh robot state immediately before enabling
        # LowCmd.
        # ----------------------------------------------------

        state = state_buffer.snapshot()

        if state is None:
            raise RuntimeError(
                "Robot lowstate unavailable before enable."
            )

        (
            upper_q,
            arm_q,
            arm_dq,
            gripper_q,
            gripper_dq,
            mode_machine,
            received_at,
        ) = state

        if (
            time.monotonic() - received_at
            > args.lowstate_timeout
        ):
            raise RuntimeError(
                "Robot lowstate stale before enable."
            )

        startup_arm_q = np.asarray(
            arm_q,
            dtype=float,
        ).copy()

        startup_left_pose, startup_right_pose = (
            get_ee_poses_from_q(
                startup_arm_q
            )
        )

        # VR_CONTINUOUS_IK_HOLD_V1
        #
        # Persistent Cartesian targets, analogous to the
        # stationary VR controller poses.
        hold_left_target = startup_left_pose.copy()
        hold_right_target = startup_right_pose.copy()

        # VIRTUAL_VR_TARGET_STREAM_V1
        #
        # desired_* = where the virtual controller wants to go.
        # hold_*    = continuously moving Cartesian command.
        #
        # Keyboard input changes desired_* only.
        # It NEVER starts/restarts a new Cartesian trajectory.
        desired_left_target = hold_left_target.copy()
        desired_right_target = hold_right_target.copy()

        # Cartesian velocity state for acceleration-limited,
        # continuous motion.
        cartesian_velocity_m_s = np.zeros(3, dtype=float)

        # First conservative streaming parameters.
        # VIRTUAL_VR_SOFT_LAG_V2
        #
        # Slower virtual-controller motion gives the real arm
        # more time to follow while preserving continuous motion.
        VIRTUAL_VR_MAX_SPEED_M_S = 0.006
        VIRTUAL_VR_MAX_ACCEL_M_S2 = 0.020
        VIRTUAL_VR_POSITION_EPS_M = 0.00015

        VIRTUAL_VR_LAG_WARN_MM = 30.0
        VIRTUAL_VR_LAG_SLOW_MM = 15.0
        VIRTUAL_VR_LAG_HIGH_MM = 35.0

        # IMPORTANT:
        # Never reduce Cartesian command speed to zero merely
        # because the physical arm has tracking error.
        VIRTUAL_VR_MIN_SPEED_SCALE = 0.25

        # Keyboard commands are allowed to accumulate, but the
        # virtual destination cannot get arbitrarily far ahead
        # of the continuously moving command reference.
        VIRTUAL_VR_MAX_BACKLOG_M = 0.040

        last_virtual_vr_tick = time.monotonic()

        # ====================================================
        # VISUAL_TARGET_STATE_V1
        #
        # This replaces the legacy NAV / bias / settle motion
        # backend.  Vision only supplies an absolute XY target.
        # ====================================================

        (
            visual_calib_data,
            visual_H_img_to_xy,
            visual_calib_hull,
        ) = _load_visual_homography()

        visual_target_xy_mm = None
        visual_target_pixel = None

        # VISUAL_TOOL_ROI_STATE_V1
        # Initial tweezer/tool ROI selected by R.
        # For now this is a visual reference only.
        # It is NOT yet used to drive robot motion.
        visual_tool_roi = None

        # RIGID_TOOL_TRACKER_V1
        visual_tool_tracker = (
            _RigidRoiTracker()
        )

        visual_tool_center_px = None
        visual_tool_error_px = None

        # TRACK_ONLY_UNIQUE_FRAME_V1
        # Last unique camera frame consumed by LK tracker.
        visual_last_tracker_seq = -1

        # ====================================================
        # VISUAL_RT_SERVO_V1
        #
        # True visual servo:
        #
        #   R tracked center -> current tool pixel
        #   T selected point -> target pixel
        #
        # Both pixels are mapped through the saved Homography.
        #
        # The resulting XY error drives the existing continuous
        # Cartesian velocity controller.
        #
        # actualXYZ is NOT used as the visual alignment error.
        # ====================================================

        visual_rt_servo_active = False

        # Alignment criterion only.
        # This is not a motion/step-size limit.
        VISUAL_RT_ALIGN_PX = 1.0
        VISUAL_RT_ALIGN_FRAMES = 3

        visual_rt_aligned_frames = 0
        visual_rt_last_eval_seq = -1
        visual_rt_last_print_at = 0.0

        # Persistent desired position at the moment GO starts.
        # A manual Cartesian command changing this reference
        # cancels visual servo automatically.
        visual_rt_reference_desired_xyz = None

        visual_camera = _HeadCameraBuffer()

        print()
        print(
            "[VISION READY] Saved Homography loaded.",
            flush=True,
        )

        print(
            "[VISION READY] RMSE mm =",
            round(
                float(
                    visual_calib_data.get(
                        "rmse_mm",
                        float("nan"),
                    )
                ),
                3,
            ),
            flush=True,
        )

        print(
            "[VISION READY] T = select target, "
            "G = send target to Cartesian stream.",
            flush=True,
        )

        print()
        print(
            "Startup right EE XYZ mm =",
            np.round(
                startup_right_pose[:3, 3]
                * 1000.0,
                3,
            ).tolist(),
            flush=True,
        )

        # Take over while holding exactly the measured posture.
        output.enable(
            upper_q,
            mode_machine,
            gripper_q=gripper_q,
            gripper_initial_mode=
                args.gripper_initial_mode,
            right_gripper_open_cap_current=
                args.right_gripper_open_cap_current,
        )

        time.sleep(0.10)

        output.set_arm_goal(
            startup_arm_q,
            mode_machine,
        )

        print()
        print(
            "========================================",
            flush=True,
        )
        print(
            "R1-A7 CONTINUOUS CARTESIAN STREAM ACTIVE",
            flush=True,
        )
        print(
            "W + Enter : forward  +X 30 mm",
            flush=True,
        )
        print(
            "S + Enter : backward -X 30 mm",
            flush=True,
        )
        print(
            "A + Enter : left     +Y 30 mm",
            flush=True,
        )
        print(
            "D + Enter : right    -Y 30 mm",
            flush=True,
        )
        print(
            "U + Enter : up       +Z 10 mm",
            flush=True,
        )
        print(
            "J + Enter : down     -Z 10 mm",
            flush=True,
        )
        print(
            "I + Enter : forward-up (+X and +Z together)",
            flush=True,
        )
        print(
            "H + Enter : smooth return to startup q",
            flush=True,
        )
        print(
            "Q + Enter : quit",
            flush=True,
        )
        print(
            f"Cartesian duration = "
            f"{CARTESIAN_DURATION_S:.1f} s",
            flush=True,
        )
        print(
            f"IK frequency = "
            f"{args.ik_frequency:.1f} Hz",
            flush=True,
        )
        print(
            f"LowCmd publisher = "
            f"{args.publish_frequency:.1f} Hz",
            flush=True,
        )
        print(
            f"LowCmd max joint speed = "
            f"{args.max_joint_speed:.3f} rad/s",
            flush=True,
        )
        print(
            "========================================",
            flush=True,
        )
        print()

        motion = None

        period = 1.0 / args.ik_frequency
        next_tick = time.monotonic()

        last_print = 0.0
        solve_count = 0

        while not stop.is_set():
            now = time.monotonic()

            # -----------------------------------------------
            # Read fresh LowState every IK period.
            # -----------------------------------------------

            state = state_buffer.snapshot()

            if state is None:
                raise RuntimeError(
                    "robot lowstate disappeared"
                )

            (
                _upper_q,
                arm_q,
                arm_dq,
                gripper_q,
                gripper_dq,
                mode_machine,
                received_at,
            ) = state

            if (
                now - received_at
                > args.lowstate_timeout
            ):
                raise RuntimeError(
                    "robot lowstate stale for "
                    f"{now - received_at:.3f}s"
                )

            if output.error:
                raise RuntimeError(
                    "lowcmd publisher failed: "
                    f"{output.error}"
                )

            arm_q = np.asarray(
                arm_q,
                dtype=float,
            ).reshape(14)

            arm_dq = np.asarray(
                arm_dq,
                dtype=float,
            ).reshape(14)

            # -----------------------------------------------
            # Terminal command input.
            #
            # One command launches one CONTINUOUS trajectory.
            # No intermediate Cartesian stopping.
            # -----------------------------------------------

            readable, _writable, _errors = (
                select.select(
                    [sys.stdin],
                    [],
                    [],
                    0.0,
                )
            )

            if readable:
                command = (
                    sys.stdin.readline()
                    .strip()
                    .casefold()
                )

                if command == "q":
                    print(
                        "[QUIT] User requested stop.",
                        flush=True,
                    )
                    break

                # ===========================================
                # VISUAL_COMMAND_TG_V1
                #
                # T:
                #   select an image-space target and convert it
                #   to absolute robot XY through Homography.
                #
                # G:
                #   copy that absolute XY directly into the
                #   persistent Cartesian desired target.
                #
                # There is deliberately NO legacy NAV state,
                # virtual Cartesian bias, settle gate, or
                # max-navigation-step counter here.
                # ===========================================

                # VISUAL_COMMAND_R_V1
                if command == "r":
                    frame = visual_camera.snapshot()

                    if frame is None:
                        print(
                            "[TOOL ROI] No camera frame available.",
                            flush=True,
                        )
                        continue

                    display_frame = _draw_visual_overlay(
                        frame,
                        visual_calib_data,
                        visual_calib_hull,
                        visual_target_pixel,
                        None,
                    )

                    selected = _visual_cv2.selectROI(
                        "Select TWEEZER / TOOL ROI",
                        display_frame,
                        fromCenter=False,
                        showCrosshair=True,
                    )

                    _visual_cv2.destroyWindow(
                        "Select TWEEZER / TOOL ROI"
                    )

                    rx, ry, rw, rh = [
                        int(v)
                        for v in selected
                    ]

                    if rw <= 0 or rh <= 0:
                        print(
                            "[TOOL ROI] selection cancelled.",
                            flush=True,
                        )
                        continue

                    visual_tool_roi = np.asarray(
                        [rx, ry, rw, rh],
                        dtype=float,
                    )

                    tracker_ok = (
                        visual_tool_tracker.initialize(
                            frame,
                            visual_tool_roi,
                        )
                    )

                    if not tracker_ok:
                        print(
                            "[TOOL TRACK] initialization "
                            "failed:",
                            visual_tool_tracker.status,
                            flush=True,
                        )

                        visual_tool_roi = None
                        visual_tool_center_px = None
                        continue

                    visual_tool_roi = (
                        visual_tool_tracker.roi.copy()
                    )

                    visual_tool_center_px = (
                        visual_tool_tracker.center.copy()
                    )

                    print(
                        "[TOOL TRACK] initialized "
                        f"features="
                        f"{visual_tool_tracker.good_count}",
                        flush=True,
                    )

                    print(
                        "[TOOL ROI]",
                        visual_tool_roi.astype(int).tolist(),
                        "center=",
                        [
                            round(rx + rw / 2.0, 2),
                            round(ry + rh / 2.0, 2),
                        ],
                        flush=True,
                    )

                    continue

                if command == "t":
                    # VISUAL_RT_SERVO_V1
                    # G requires a live R tracker because R is
                    # now part of the actual control loop.
                    if (
                        not visual_tool_tracker.active
                        or visual_tool_center_px is None
                    ):
                        print(
                            "[VISION GO] No active R tool tracker. "
                            "Press R and select the tool first.",
                            flush=True,
                        )
                        continue

                    if (
                        motion is not None
                        and motion.get("type")
                        == "joint_home"
                    ):
                        print(
                            "[VISION TARGET] ignored while "
                            "HOME is active.",
                            flush=True,
                        )
                        continue

                    frame = visual_camera.snapshot()

                    if frame is None:
                        print(
                            "[VISION TARGET] No camera frame "
                            "available yet.",
                            flush=True,
                        )
                        continue

                    # VISUAL_T_OVERLAY_V1
                    target_select_frame = _draw_visual_overlay(
                        frame,
                        visual_calib_data,
                        visual_calib_hull,
                        visual_target_pixel,
                        visual_tool_roi,
                    )

                    selected = _visual_cv2.selectROI(
                        "Select VISUAL TARGET region",
                        target_select_frame,
                        fromCenter=False,
                        showCrosshair=True,
                    )

                    _visual_cv2.destroyWindow(
                        "Select VISUAL TARGET region"
                    )

                    tx, ty, tw, th = [
                        int(value)
                        for value in selected
                    ]

                    if tw <= 0 or th <= 0:
                        print(
                            "[VISION TARGET] selection cancelled.",
                            flush=True,
                        )
                        continue

                    candidate = np.asarray(
                        [
                            tx + tw / 2.0,
                            ty + th / 2.0,
                        ],
                        dtype=float,
                    )

                    inside = _visual_cv2.pointPolygonTest(
                        visual_calib_hull,
                        (
                            float(candidate[0]),
                            float(candidate[1]),
                        ),
                        False,
                    )

                    if inside < 0:
                        print(
                            "[VISION TARGET] REJECTED: "
                            "outside calibrated image region.",
                            flush=True,
                        )
                        continue

                    candidate_xy_mm = (
                        _visual_pixel_to_xy(
                            visual_H_img_to_xy,
                            candidate,
                        )
                    )

                    if not np.all(
                        np.isfinite(
                            candidate_xy_mm
                        )
                    ):
                        print(
                            "[VISION TARGET] REJECTED: "
                            "non-finite XY.",
                            flush=True,
                        )
                        continue

                    (
                        _left_visual_actual,
                        right_visual_actual,
                    ) = get_ee_poses_from_q(
                        arm_q
                    )

                    current_visual_xy_mm = (
                        right_visual_actual[
                            :2, 3
                        ]
                        * 1000.0
                    )

                    visual_target_pixel = (
                        candidate.copy()
                    )

                    visual_target_xy_mm = (
                        np.asarray(
                            candidate_xy_mm,
                            dtype=float,
                        ).copy()
                    )

                    print()
                    print(
                        "[VISION TARGET] pixel =",
                        np.round(
                            visual_target_pixel,
                            3,
                        ).tolist(),
                        flush=True,
                    )

                    print(
                        "[VISION TARGET] absolute XY mm =",
                        np.round(
                            visual_target_xy_mm,
                            3,
                        ).tolist(),
                        flush=True,
                    )

                    print(
                        "[VISION TARGET] current XY mm =",
                        np.round(
                            current_visual_xy_mm,
                            3,
                        ).tolist(),
                        flush=True,
                    )

                    print(
                        "[VISION TARGET] required dXY mm =",
                        np.round(
                            visual_target_xy_mm
                            - current_visual_xy_mm,
                            3,
                        ).tolist(),
                        flush=True,
                    )

                    print(
                        "[VISION TARGET] Press G then type GO "
                        "to execute.",
                        flush=True,
                    )

                    continue

                elif command == "g":
                    if visual_target_xy_mm is None:
                        print(
                            "[VISION GO] No visual target. "
                            "Press T first.",
                            flush=True,
                        )
                        continue

                    if (
                        motion is not None
                        and motion.get("type")
                        == "joint_home"
                    ):
                        print(
                            "[VISION GO] ignored while "
                            "HOME is active.",
                            flush=True,
                        )
                        continue

                    (
                        _left_go_actual,
                        right_go_actual,
                    ) = get_ee_poses_from_q(
                        arm_q
                    )

                    current_go_xy_mm = (
                        right_go_actual[
                            :2, 3
                        ]
                        * 1000.0
                    )

                    visual_distance_mm = float(
                        np.linalg.norm(
                            visual_target_xy_mm
                            - current_go_xy_mm
                        )
                    )

                    print()
                    print(
                        "[VISION GO] target XY mm =",
                        np.round(
                            visual_target_xy_mm,
                            3,
                        ).tolist(),
                        flush=True,
                    )

                    print(
                        "[VISION GO] current XY mm =",
                        np.round(
                            current_go_xy_mm,
                            3,
                        ).tolist(),
                        flush=True,
                    )

                    print(
                        "[VISION GO] planar distance mm =",
                        round(
                            visual_distance_mm,
                            3,
                        ),
                        flush=True,
                    )

                    answer = input(
                        "Type GO to send this XY target "
                        "to the real robot: "
                    ).strip()

                    if answer.casefold() != "go":
                        print(
                            "[VISION GO] cancelled.",
                            flush=True,
                        )
                        continue

                    # ---------------------------------------
                    # THIS is the entire vision -> motion
                    # interface.
                    #
                    # Only X/Y are changed.
                    # Z remains at the current persistent
                    # Cartesian desired height.
                    # Tool orientation remains fixed to the
                    # startup orientation.
                    #
                    # No NAV step.
                    # No virtual bias.
                    # No settle gate.
                    # No artificial STOP threshold.
                    # ---------------------------------------

                    # =======================================
                    # VISUAL_RT_SERVO_V1
                    #
                    # GO no longer sends T's absolute XY as a
                    # one-shot Cartesian destination.
                    #
                    # Instead, GO arms continuous R -> T visual
                    # servo.  Start the Cartesian reference from
                    # the robot's CURRENT measured XY.
                    # =======================================

                    _visual_rt_keep_z = float(
                        desired_right_target[
                            2, 3
                        ]
                    )

                    desired_right_target[
                        0, 3
                    ] = float(
                        right_go_actual[
                            0, 3
                        ]
                    )

                    desired_right_target[
                        1, 3
                    ] = float(
                        right_go_actual[
                            1, 3
                        ]
                    )

                    desired_right_target[
                        2, 3
                    ] = _visual_rt_keep_z

                    hold_right_target[
                        :3, 3
                    ] = desired_right_target[
                        :3, 3
                    ]

                    desired_right_target[
                        :3, :3
                    ] = startup_right_pose[
                        :3, :3
                    ]

                    hold_right_target[
                        :3, :3
                    ] = startup_right_pose[
                        :3, :3
                    ]

                    visual_rt_servo_active = True
                    visual_rt_aligned_frames = 0
                    visual_rt_last_eval_seq = -1
                    visual_rt_last_print_at = 0.0

                    visual_rt_reference_desired_xyz = (
                        desired_right_target[
                            :3, 3
                        ].copy()
                    )

                    # Start the visual servo from rest.
                    # Existing Cartesian acceleration limiting
                    # handles the ramp-up afterwards.
                    cartesian_velocity_m_s[:] = 0.0

                    print(
                        "[VISUAL RT] START "
                        f"R_px="
                        f"{np.round(visual_tool_center_px, 2).tolist()} "
                        f"T_px="
                        f"{np.round(visual_target_pixel, 2).tolist()}",
                        flush=True,
                    )

                    print()
                    print(
                        "[VISION GO ACCEPTED]",
                        flush=True,
                    )

                    print(
                        "  desired XYZ mm =",
                        np.round(
                            desired_right_target[
                                :3, 3
                            ]
                            * 1000.0,
                            3,
                        ).tolist(),
                        flush=True,
                    )

                    print(
                        "  Z unchanged; orientation fixed.",
                        flush=True,
                    )

                    print(
                        "  Motion backend = continuous "
                        "VR-style full 7-DOF IK.",
                        flush=True,
                    )

                    continue

                if command in direction_delta_mm:
                    if (
                        motion is not None
                        and motion.get("type")
                        == "joint_home"
                    ):
                        print(
                            "[TARGET UPDATE] ignored while HOME is active.",
                            flush=True,
                        )
                    else:
                        delta_m = (
                            direction_delta_mm[
                                command
                            ]
                            / 1000.0
                        )

                        desired_right_target[
                            :3, 3
                        ] += delta_m

                        # Keep only a bounded amount of pending
                        # virtual-controller motion.
                        pending_delta = (
                            desired_right_target[:3, 3]
                            - hold_right_target[:3, 3]
                        )

                        pending_distance = float(
                            np.linalg.norm(
                                pending_delta
                            )
                        )

                        if (
                            pending_distance
                            > VIRTUAL_VR_MAX_BACKLOG_M
                        ):
                            desired_right_target[:3, 3] = (
                                hold_right_target[:3, 3]
                                + pending_delta
                                / pending_distance
                                * VIRTUAL_VR_MAX_BACKLOG_M
                            )

                            print(
                                "[TARGET LIMIT] pending Cartesian "
                                "target limited to "
                                f"{VIRTUAL_VR_MAX_BACKLOG_M * 1000.0:.1f} mm",
                                flush=True,
                            )

                        # Keep desired tool orientation fixed.
                        desired_right_target[
                            :3, :3
                        ] = startup_right_pose[
                            :3, :3
                        ]

                        print()
                        print(
                            "[TARGET UPDATE]",
                            direction_name[
                                command
                            ],
                            flush=True,
                        )

                        print(
                            "  delta XYZ mm   =",
                            np.round(
                                delta_m * 1000.0,
                                3,
                            ).tolist(),
                            flush=True,
                        )

                        print(
                            "  desired XYZ mm =",
                            np.round(
                                desired_right_target[
                                    :3, 3
                                ]
                                * 1000.0,
                                3,
                            ).tolist(),
                            flush=True,
                        )

                elif command == "h":
                    desired_left_target = (
                        startup_left_pose.copy()
                    )

                    desired_right_target = (
                        startup_right_pose.copy()
                    )

                    cartesian_velocity_m_s[:] = 0.0

                    motion = {
                        "type": "joint_home",
                        "name": "HOME",
                        "started_at": now,
                        "duration":
                            HOME_DURATION_S,
                        "start_q":
                            arm_q.copy(),
                        "goal_q":
                            startup_arm_q.copy(),
                    }

                    print()
                    print(
                        "[STREAM START] HOME",
                        flush=True,
                    )

                elif command:
                    print(
                        "Unknown command. "
                        "Use W/S/A/D/U/J/I/R/T/G/H/Q.",
                        flush=True,
                    )

            # ===============================================
            # LIVE_CAMERA_PREVIEW_V1
            # Non-blocking live head-camera preview.
            # This does NOT affect LowCmd / IK frequency.
            # ===============================================
            try:
                (
                    _preview_frame,
                    _preview_seq,
                ) = visual_camera.snapshot_with_seq()

                if _preview_frame is not None:

                    # =======================================
                    # TRACK_ONLY_UNIQUE_FRAME_V1
                    #
                    # The Unitree head camera can return the
                    # same JPEG multiple times.  LK optical
                    # flow must run only once for each unique
                    # decoded camera frame.
                    # =======================================
                    if (
                        visual_tool_tracker.active
                        and _preview_seq
                        != visual_last_tracker_seq
                    ):
                        visual_tool_tracker.update(
                            _preview_frame
                        )

                        visual_last_tracker_seq = (
                            _preview_seq
                        )

                        if (
                            visual_tool_tracker.roi
                            is not None
                        ):
                            visual_tool_roi = (
                                visual_tool_tracker.roi.copy()
                            )

                        if (
                            visual_tool_tracker.center
                            is not None
                        ):
                            visual_tool_center_px = (
                                visual_tool_tracker.center.copy()
                            )

                    # LIVE_CAMERA_OVERLAY_V2
                    _preview_display = _draw_visual_overlay(
                        _preview_frame,
                        visual_calib_data,
                        visual_calib_hull,
                        visual_target_pixel,
                        visual_tool_roi,
                    )

                    # Draw tracker status.
                    if visual_tool_center_px is not None:
                        _tc = tuple(
                            int(round(v))
                            for v in visual_tool_center_px
                        )

                        if visual_tool_tracker.active:
                            _tool_color = (
                                0,
                                255,
                                0,
                            )
                        else:
                            _tool_color = (
                                0,
                                0,
                                255,
                            )

                        _visual_cv2.drawMarker(
                            _preview_display,
                            _tc,
                            _tool_color,
                            markerType=(
                                _visual_cv2.MARKER_TILTED_CROSS
                            ),
                            markerSize=18,
                            thickness=2,
                        )

                        _visual_cv2.putText(
                            _preview_display,
                            (
                                "TRACK "
                                f"{visual_tool_tracker.status} "
                                f"N={visual_tool_tracker.good_count}"
                            ),
                            (
                                max(5, _tc[0] + 10),
                                max(20, _tc[1] + 24),
                            ),
                            _visual_cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            _tool_color,
                            1,
                            _visual_cv2.LINE_AA,
                        )

                    # Display current image-space error from
                    # tracked tool center to selected target.
                    if (
                        visual_target_pixel is not None
                        and visual_tool_center_px is not None
                        and visual_tool_tracker.active
                    ):
                        visual_tool_error_px = (
                            np.asarray(
                                visual_target_pixel,
                                dtype=float,
                            )
                            - np.asarray(
                                visual_tool_center_px,
                                dtype=float,
                            )
                        )

                        _target_xy = tuple(
                            int(round(v))
                            for v in visual_target_pixel
                        )

                        _tool_xy = tuple(
                            int(round(v))
                            for v in visual_tool_center_px
                        )

                        _visual_cv2.line(
                            _preview_display,
                            _tool_xy,
                            _target_xy,
                            (255, 0, 255),
                            1,
                        )

                        _err_norm = float(
                            np.linalg.norm(
                                visual_tool_error_px
                            )
                        )

                        _visual_cv2.putText(
                            _preview_display,
                            (
                                "image error "
                                f"du={visual_tool_error_px[0]:+.1f} "
                                f"dv={visual_tool_error_px[1]:+.1f} "
                                f"|e|={_err_norm:.1f}px"
                            ),
                            (20, 35),
                            _visual_cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 0, 255),
                            2,
                            _visual_cv2.LINE_AA,
                        )

                    elif visual_tool_tracker.lost:
                        _visual_cv2.putText(
                            _preview_display,
                            "TOOL TRACK LOST - press R",
                            (20, 35),
                            _visual_cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 0, 255),
                            2,
                            _visual_cv2.LINE_AA,
                        )

                    _visual_cv2.imshow(
                        "R1-A7 Head RGB - R: tool  T: target",
                        _preview_display,
                    )

                    # Required by OpenCV to refresh GUI events.
                    _visual_cv2.waitKey(1)

            except Exception as _preview_exc:
                pass

            # -----------------------------------------------
            # CONTINUOUS Cartesian trajectory.
            # -----------------------------------------------

            if (
                motion is not None
                and motion["type"]
                == "cartesian"
            ):
                elapsed = (
                    now
                    - motion["started_at"]
                )

                s_time = (
                    elapsed
                    / motion["duration"]
                )

                alpha = (
                    quintic_time_scaling(
                        s_time
                    )
                )

                left_target = (
                    motion[
                        "left_start"
                    ].copy()
                )

                right_target = (
                    motion[
                        "right_start"
                    ].copy()
                )

                right_target[:3, 3] = (
                    motion["start_xyz"]
                    + alpha
                    * motion["delta_m"]
                )

                # Keep the tool orientation exactly equal
                # to the orientation at motion start.
                right_target[:3, :3] = (
                    motion[
                        "right_start"
                    ][:3, :3]
                )

                # IMPORTANT:
                # This intentionally matches the successful
                # original VR solve_ik call.
                #
                # NO max_joint_step.
                # NO position_only=True.
                # NO wrist-pitch penalty.
                # NO settle gate.
                solution, _ = ik.solve_ik(
                    left_target,
                    right_target,
                    arm_q,
                    arm_dq,
                )

                solution = np.asarray(
                    solution,
                    dtype=float,
                ).reshape(14)

                if not np.all(
                    np.isfinite(
                        solution
                    )
                ):
                    raise RuntimeError(
                        "IK produced non-finite q"
                    )

                output.set_arm_goal(
                    solution,
                    mode_machine,
                )

                # Persistent virtual-controller target.
                hold_left_target = left_target.copy()
                hold_right_target = right_target.copy()

                solve_count += 1

                if (
                    now - last_print
                    >= args.print_period
                ):
                    _left_actual, right_actual = (
                        get_ee_poses_from_q(
                            arm_q
                        )
                    )

                    target_xyz_mm = (
                        right_target[:3, 3]
                        * 1000.0
                    )

                    actual_xyz_mm = (
                        right_actual[:3, 3]
                        * 1000.0
                    )

                    print(
                        "[STREAM]",
                        motion["name"],
                        f"t={elapsed:.2f}/"
                        f"{motion['duration']:.2f}s",
                        f"alpha={alpha:.3f}",
                        "targetXYZ=",
                        np.round(
                            target_xyz_mm,
                            2,
                        ).tolist(),
                        "actualXYZ=",
                        np.round(
                            actual_xyz_mm,
                            2,
                        ).tolist(),
                        flush=True,
                    )

                    last_print = now

                if (
                    elapsed
                    >= motion["duration"]
                ):
                    _left_actual, right_actual = (
                        get_ee_poses_from_q(
                            arm_q
                        )
                    )

                    actual_xyz_mm = (
                        right_actual[:3, 3]
                        * 1000.0
                    )

                    requested_xyz_mm = (
                        motion["goal_xyz"]
                        * 1000.0
                    )

                    print()
                    print(
                        "[STREAM COMPLETE]",
                        motion["name"],
                        flush=True,
                    )
                    print(
                        "  requested XYZ mm =",
                        np.round(
                            requested_xyz_mm,
                            3,
                        ).tolist(),
                        flush=True,
                    )
                    print(
                        "  actual XYZ mm    =",
                        np.round(
                            actual_xyz_mm,
                            3,
                        ).tolist(),
                        flush=True,
                    )
                    print(
                        "  residual mm      =",
                        np.round(
                            requested_xyz_mm
                            - actual_xyz_mm,
                            3,
                        ).tolist(),
                        flush=True,
                    )

                    hold_left_target = (
                        motion["left_start"].copy()
                    )

                    hold_right_target = (
                        motion["right_start"].copy()
                    )

                    hold_right_target[:3, 3] = (
                        motion["goal_xyz"].copy()
                    )

                    motion = None

            # -----------------------------------------------
            # Smooth joint-space return to startup posture.
            # -----------------------------------------------

            elif (
                motion is not None
                and motion["type"]
                == "joint_home"
            ):
                elapsed = (
                    now
                    - motion["started_at"]
                )

                s_time = (
                    elapsed
                    / motion["duration"]
                )

                alpha = (
                    quintic_time_scaling(
                        s_time
                    )
                )

                q_reference = (
                    motion["start_q"]
                    + alpha
                    * (
                        motion["goal_q"]
                        - motion["start_q"]
                    )
                )

                output.set_arm_goal(
                    q_reference,
                    mode_machine,
                )

                if (
                    now - last_print
                    >= args.print_period
                ):
                    print(
                        "[STREAM] HOME",
                        f"t={elapsed:.2f}/"
                        f"{motion['duration']:.2f}s",
                        f"alpha={alpha:.3f}",
                        flush=True,
                    )

                    last_print = now

                if (
                    elapsed
                    >= motion["duration"]
                ):
                    output.set_arm_goal(
                        startup_arm_q,
                        mode_machine,
                    )

                    print(
                        "[STREAM COMPLETE] HOME",
                        flush=True,
                    )

                    hold_left_target = (
                        startup_left_pose.copy()
                    )

                    hold_right_target = (
                        startup_right_pose.copy()
                    )

                    motion = None

            # -----------------------------------------------
            # VR-style continuous IK hold.
            #
            # The VR controller keeps solving IK even when
            # the human stops moving the controller.
            #
            # Every cycle uses the latest measured q/dq and
            # solves the complete 7-DOF arm again.
            # -----------------------------------------------
            elif motion is None:
                # ===========================================
                # Virtual VR continuous Cartesian streaming.
                #
                # 1. Keyboard updates desired_right_target.
                # 2. hold_right_target moves toward desired
                #    target continuously.
                # 3. Motion velocity is acceleration-limited.
                # 4. Complete arm IK is re-solved every cycle
                #    using latest measured arm_q / arm_dq.
                # ===========================================

                dt = (
                    now
                    - last_virtual_vr_tick
                )

                last_virtual_vr_tick = now

                dt = float(
                    np.clip(
                        dt,
                        1.0e-4,
                        0.10,
                    )
                )

                current_command_xyz = (
                    hold_right_target[
                        :3, 3
                    ].copy()
                )

                desired_xyz = (
                    desired_right_target[
                        :3, 3
                    ].copy()
                )

                # Current measured EE position.
                _left_actual_now, right_actual_now = (
                    get_ee_poses_from_q(
                        arm_q
                    )
                )

                actual_xyz_now = (
                    right_actual_now[
                        :3, 3
                    ].copy()
                )

                command_actual_lag_mm = float(
                    np.linalg.norm(
                        current_command_xyz
                        - actual_xyz_now
                    )
                    * 1000.0
                )

                # =================================================
                # VISUAL_RT_SERVO_V1
                #
                # IMPORTANT:
                #
                # Normal/manual mode:
                #
                #   cart_error = desired - command
                #
                # Visual RT mode:
                #
                #   R center pixel ----+
                #                      +--> Homography --> XY error
                #   T target pixel ----+
                #
                # and that visual XY error becomes the Cartesian
                # error seen by the existing velocity generator.
                #
                # Therefore the command does NOT stop merely
                # because a one-shot robot XY target was reached.
                #
                # It stops only after R and T visually align.
                # =================================================

                if visual_rt_servo_active:

                    # ---------------------------------------
                    # Manual Cartesian target modification
                    # automatically hands control back.
                    # ---------------------------------------
                    if (
                        visual_rt_reference_desired_xyz
                        is not None
                        and float(
                            np.linalg.norm(
                                desired_right_target[
                                    :3, 3
                                ]
                                - visual_rt_reference_desired_xyz
                            )
                        )
                        > 1.0e-6
                    ):
                        visual_rt_servo_active = False
                        visual_rt_aligned_frames = 0

                        print(
                            "[VISUAL RT] MANUAL_CANCEL",
                            flush=True,
                        )

                    # ---------------------------------------
                    # Tracker is a required sensor.
                    # Losing it is a real fault, not an
                    # artificial motion restriction.
                    # ---------------------------------------
                    elif (
                        not visual_tool_tracker.active
                        or visual_tool_center_px is None
                        or visual_target_pixel is None
                    ):
                        visual_rt_servo_active = False
                        visual_rt_aligned_frames = 0

                        desired_right_target[
                            :3, 3
                        ] = current_command_xyz

                        desired_xyz = (
                            current_command_xyz.copy()
                        )

                        cartesian_velocity_m_s[:] = 0.0

                        print(
                            "[VISUAL RT] STOP TRACKER_INVALID",
                            flush=True,
                        )

                    # ---------------------------------------
                    # Existing backlog safety boundary.
                    # It is deliberately large and does not
                    # affect normal R->T motion.
                    # ---------------------------------------
                    elif (
                        command_actual_lag_mm
                        > (
                            VIRTUAL_VR_MAX_BACKLOG_M
                            * 1000.0
                        )
                    ):
                        visual_rt_servo_active = False
                        visual_rt_aligned_frames = 0

                        desired_right_target[
                            :3, 3
                        ] = current_command_xyz

                        desired_xyz = (
                            current_command_xyz.copy()
                        )

                        cartesian_velocity_m_s[:] = 0.0

                        print(
                            "[VISUAL RT] STOP BACKLOG "
                            f"lag_mm={command_actual_lag_mm:.1f}",
                            flush=True,
                        )

                    else:
                        _rt_tool_px = np.asarray(
                            visual_tool_center_px,
                            dtype=float,
                        ).reshape(2)

                        _rt_target_px = np.asarray(
                            visual_target_pixel,
                            dtype=float,
                        ).reshape(2)

                        _rt_error_px = (
                            _rt_target_px
                            - _rt_tool_px
                        )

                        _rt_error_px_norm = float(
                            np.linalg.norm(
                                _rt_error_px
                            )
                        )

                        try:
                            _rt_tool_xy_mm = np.asarray(
                                _visual_pixel_to_xy(
                                    visual_H_img_to_xy,
                                    _rt_tool_px,
                                ),
                                dtype=float,
                            ).reshape(2)

                            _rt_target_xy_mm = np.asarray(
                                _visual_pixel_to_xy(
                                    visual_H_img_to_xy,
                                    _rt_target_px,
                                ),
                                dtype=float,
                            ).reshape(2)

                        except Exception as _rt_exc:
                            visual_rt_servo_active = False
                            visual_rt_aligned_frames = 0

                            desired_right_target[
                                :3, 3
                            ] = current_command_xyz

                            desired_xyz = (
                                current_command_xyz.copy()
                            )

                            cartesian_velocity_m_s[:] = 0.0

                            print(
                                "[VISUAL RT] STOP HOMOGRAPHY_ERROR "
                                f"{_rt_exc}",
                                flush=True,
                            )

                        else:
                            # -----------------------------------
                            # This is the actual visual control
                            # error used to move the robot.
                            # -----------------------------------
                            _rt_error_xy_mm = (
                                _rt_target_xy_mm
                                - _rt_tool_xy_mm
                            )

                            _rt_error_xy_m = (
                                _rt_error_xy_mm
                                / 1000.0
                            )

                            # -----------------------------------
                            # Alignment count is updated only
                            # once per UNIQUE tracked camera frame.
                            # -----------------------------------
                            if (
                                visual_last_tracker_seq
                                != visual_rt_last_eval_seq
                            ):
                                if (
                                    _rt_error_px_norm
                                    <= VISUAL_RT_ALIGN_PX
                                ):
                                    visual_rt_aligned_frames += 1
                                else:
                                    visual_rt_aligned_frames = 0

                                visual_rt_last_eval_seq = (
                                    visual_last_tracker_seq
                                )

                            if (
                                visual_rt_aligned_frames
                                >= VISUAL_RT_ALIGN_FRAMES
                            ):
                                visual_rt_servo_active = False

                                # Freeze exactly where the visual
                                # servo has aligned R with T.
                                desired_right_target[
                                    :3, 3
                                ] = current_command_xyz

                                desired_xyz = (
                                    current_command_xyz.copy()
                                )

                                cartesian_velocity_m_s[:] = 0.0

                                print(
                                    "[VISUAL RT] ALIGNED "
                                    f"R_px="
                                    f"{np.round(_rt_tool_px, 2).tolist()} "
                                    f"T_px="
                                    f"{np.round(_rt_target_px, 2).tolist()} "
                                    f"error_px="
                                    f"{_rt_error_px_norm:.3f}",
                                    flush=True,
                                )

                            else:
                                # --------------------------------
                                # CRITICAL CONTROL EQUATION
                                #
                                # We do NOT make T a one-shot
                                # absolute robot target.
                                #
                                # Instead:
                                #
                                # desired = current command
                                #           + current R->T visual
                                #             XY error
                                #
                                # Therefore the existing Cartesian
                                # controller continuously moves in
                                # the direction demanded by R->T.
                                #
                                # No artificial 0.5 mm / 1 mm
                                # incremental step limit is added.
                                # Existing 6 mm/s speed and
                                # acceleration limits remain.
                                # --------------------------------

                                desired_xyz = (
                                    current_command_xyz.copy()
                                )

                                desired_xyz[
                                    :2
                                ] = (
                                    current_command_xyz[
                                        :2
                                    ]
                                    + _rt_error_xy_m
                                )

                                # Visual servo is XY-only.
                                # Do not command Z from the image.
                                desired_xyz[
                                    2
                                ] = current_command_xyz[
                                    2
                                ]

                                if (
                                    now
                                    - visual_rt_last_print_at
                                    >= 0.5
                                ):
                                    print(
                                        "[VISUAL RT] "
                                        f"R_px="
                                        f"{np.round(_rt_tool_px, 2).tolist()} "
                                        f"T_px="
                                        f"{np.round(_rt_target_px, 2).tolist()} "
                                        f"e_px="
                                        f"{np.round(_rt_error_px, 2).tolist()} "
                                        f"|e_px|="
                                        f"{_rt_error_px_norm:.2f} "
                                        f"eXY_mm="
                                        f"{np.round(_rt_error_xy_mm, 2).tolist()} "
                                        f"lag_mm="
                                        f"{command_actual_lag_mm:.1f}",
                                        flush=True,
                                    )

                                    visual_rt_last_print_at = now

                # =================================================
                # SAFE COMMAND-REFERENCE CARTESIAN BASELINE
                #
                # Keyboard / visual target updates desired_xyz.
                # The internal Cartesian command approaches that
                # requested target smoothly.
                #
                # IMPORTANT:
                # Do NOT integrate measured FK error directly into
                # current_command_xyz here.
                # =================================================

                cart_error = (
                    desired_xyz
                    - current_command_xyz
                )

                cart_distance = float(
                    np.linalg.norm(
                        cart_error
                    )
                )

                # -------------------------------------------
                # Generate a smooth Cartesian velocity.
                #
                # sqrt(2*a*d) creates automatic deceleration
                # as the virtual controller approaches its
                # destination.
                # -------------------------------------------

                if (
                    cart_distance
                    > VIRTUAL_VR_POSITION_EPS_M
                ):
                    cart_direction = (
                        cart_error
                        / cart_distance
                    )

                    stopping_speed = np.sqrt(
                        2.0
                        * VIRTUAL_VR_MAX_ACCEL_M_S2
                        * cart_distance
                    )

                    desired_speed = min(
                        VIRTUAL_VR_MAX_SPEED_M_S,
                        float(stopping_speed),
                    )

                    # Continuous speed scaling according to
                    # how far the commanded Cartesian pose is
                    # ahead of the actual robot.
                    if (
                        command_actual_lag_mm
                        <= VIRTUAL_VR_LAG_SLOW_MM
                    ):
                        lag_speed_scale = 1.0

                    elif (
                        command_actual_lag_mm
                        >= VIRTUAL_VR_LAG_HIGH_MM
                    ):
                        lag_speed_scale = (
                            VIRTUAL_VR_MIN_SPEED_SCALE
                        )

                    else:
                        ratio = (
                            command_actual_lag_mm
                            - VIRTUAL_VR_LAG_SLOW_MM
                        ) / (
                            VIRTUAL_VR_LAG_HIGH_MM
                            - VIRTUAL_VR_LAG_SLOW_MM
                        )

                        lag_speed_scale = (
                            1.0
                            - ratio
                            * (
                                1.0
                                - VIRTUAL_VR_MIN_SPEED_SCALE
                            )
                        )

                    desired_speed *= float(
                        np.clip(
                            lag_speed_scale,
                            VIRTUAL_VR_MIN_SPEED_SCALE,
                            1.0,
                        )
                    )

                    desired_velocity = (
                        cart_direction
                        * desired_speed
                    )

                else:
                    lag_speed_scale = 1.0

                    desired_velocity = (
                        np.zeros(
                            3,
                            dtype=float,
                        )
                    )

                # -------------------------------------------
                # Acceleration limiter.
                # This prevents a new W/U/A/D key from
                # causing an instantaneous direction jump.
                # -------------------------------------------

                velocity_delta = (
                    desired_velocity
                    - cartesian_velocity_m_s
                )

                velocity_delta_norm = float(
                    np.linalg.norm(
                        velocity_delta
                    )
                )

                max_velocity_delta = (
                    VIRTUAL_VR_MAX_ACCEL_M_S2
                    * dt
                )

                if (
                    velocity_delta_norm
                    > max_velocity_delta
                    and velocity_delta_norm
                    > 1.0e-12
                ):
                    velocity_delta *= (
                        max_velocity_delta
                        / velocity_delta_norm
                    )

                cartesian_velocity_m_s += (
                    velocity_delta
                )

                next_command_xyz = (
                    current_command_xyz
                    + cartesian_velocity_m_s
                    * dt
                )

                # If the target is extremely close and the
                # stream is nearly stopped, snap only the
                # numerical reference to the exact target.
                if (
                    cart_distance
                    <= VIRTUAL_VR_POSITION_EPS_M
                    and float(
                        np.linalg.norm(
                            cartesian_velocity_m_s
                        )
                    )
                    < 5.0e-4
                ):
                    next_command_xyz = (
                        desired_xyz.copy()
                    )

                    cartesian_velocity_m_s[:] = (
                        0.0
                    )

                hold_left_target = (
                    desired_left_target.copy()
                )

                hold_right_target[
                    :3, 3
                ] = next_command_xyz

                # Keep tool orientation fixed just like the
                # successful forward-up I experiment.
                hold_right_target[
                    :3, :3
                ] = startup_right_pose[
                    :3, :3
                ]

                # -------------------------------------------
                # This is intentionally the SAME IK call as
                # the verified successful VR controller.
                #
                # No max_joint_step.
                # No per-joint manual direction.
                # No shoulder-roll override.
                # No settle gate.
                # -------------------------------------------

                stream_solution, _ = ik.solve_ik(
                    hold_left_target,
                    hold_right_target,
                    arm_q,
                    arm_dq,
                )

                stream_solution = np.asarray(
                    stream_solution,
                    dtype=float,
                ).reshape(14)

                if not np.all(
                    np.isfinite(
                        stream_solution
                    )
                ):
                    raise RuntimeError(
                        "Virtual VR IK produced "
                        "non-finite q"
                    )

                output.set_arm_goal(
                    stream_solution,
                    mode_machine,
                )

                solve_count += 1

                # -------------------------------------------
                # Diagnostics.
                # -------------------------------------------

                if (
                    now - last_print
                    >= args.print_period
                ):
                    _left_actual, right_actual = (
                        get_ee_poses_from_q(
                            arm_q
                        )
                    )

                    actual_xyz = (
                        right_actual[
                            :3, 3
                        ].copy()
                    )

                    desired_xyz_mm = (
                        desired_xyz
                        * 1000.0
                    )

                    command_xyz_mm = (
                        hold_right_target[
                            :3, 3
                        ]
                        * 1000.0
                    )

                    actual_xyz_mm = (
                        actual_xyz
                        * 1000.0
                    )

                    command_actual_lag_mm = float(
                        np.linalg.norm(
                            command_xyz_mm
                            - actual_xyz_mm
                        )
                    )

                    # VIRTUAL_VR_RIGHT_JOINT_DIAG_V2
                    # goal   = latest IK solution
                    # sent   = q actually written into LowCmd
                    # actual = lowstate measured joint position
                    with output.lock:
                        diagnostic_goal_q = (
                            output.arm_goal.copy()
                        )
                        diagnostic_sent_q = (
                            output.upper_target[1:15].copy()
                        )

                    print(
                        "[VIRTUAL VR]",
                        "desiredXYZ=",
                        np.round(
                            desired_xyz_mm,
                            2,
                        ).tolist(),
                        "commandXYZ=",
                        np.round(
                            command_xyz_mm,
                            2,
                        ).tolist(),
                        "actualXYZ=",
                        np.round(
                            actual_xyz_mm,
                            2,
                        ).tolist(),
                        "vel_mm_s=",
                        np.round(
                            cartesian_velocity_m_s
                            * 1000.0,
                            2,
                        ).tolist(),
                        "lag_mm=",
                        f"{command_actual_lag_mm:.1f}",
                        "speed_scale=",
                        f"{lag_speed_scale:.2f}",
                        flush=True,
                    )

                    # Most relevant joints for the current
                    # forward/up diagnostic.
                    for joint_name, idx in (
                        ("R_shoulder_pitch", 7),
                        ("R_shoulder_roll", 8),
                        ("R_shoulder_yaw", 9),
                        ("R_elbow", 10),
                        ("R_wrist_pitch", 12),
                    ):
                        q_goal_diag = float(
                            diagnostic_goal_q[idx]
                        )

                        q_sent_diag = float(
                            diagnostic_sent_q[idx]
                        )

                        q_actual_diag = float(
                            arm_q[idx]
                        )

                        print(
                            f"  {joint_name:18s} "
                            f"goal={q_goal_diag:+.5f} "
                            f"sent={q_sent_diag:+.5f} "
                            f"actual={q_actual_diag:+.5f} "
                            f"g-s={q_goal_diag-q_sent_diag:+.5f} "
                            f"s-a={q_sent_diag-q_actual_diag:+.5f}",
                            flush=True,
                        )

                    if (
                        command_actual_lag_mm
                        > VIRTUAL_VR_LAG_WARN_MM
                    ):
                        print(
                            "[WARN] command/actual "
                            f"Cartesian lag = "
                            f"{command_actual_lag_mm:.1f} mm",
                            flush=True,
                        )

                    last_print = now

            # -----------------------------------------------
            # 30 Hz IK scheduling.
            # LowCmd continues independently at 250 Hz.
            # -----------------------------------------------

            next_tick += period

            remaining = (
                next_tick
                - time.monotonic()
            )

            if remaining > 0.0:
                stop.wait(
                    remaining
                )
            else:
                next_tick = (
                    time.monotonic()
                )

        print(
            "Scripted Cartesian stream stopped.",
            flush=True,
        )

        return 0

        first_solution = None
        start_edges = ButtonEdge()
        last_wait_print = 0.0
        print(
            "Enter VR, align controllers with the real arms, release then press "
            "right A. Terminal R also starts; Q quits. Left X is reserved for recording.",
            flush=True,
        )
        while not stop.is_set():
            now = time.monotonic()
            tele = tv.get_tele_data()
            right_a = bool(getattr(tele, "right_ctrl_aButton", False))
            left_x = bool(getattr(tele, "left_ctrl_aButton", False))
            a_rising = start_edges.rising(tele, "right_ctrl_aButton")
            terminal_start = False
            readable, _writable, _errors = select.select([sys.stdin], [], [], 0.0)
            if readable:
                key = sys.stdin.readline().strip().casefold()
                if key == "q":
                    return 0
                terminal_start = key == "r"
            if not a_rising and not terminal_start:
                if now - last_wait_print >= args.print_period:
                    xr_age = tele_xr_age(tele, now)
                    print(
                        "waiting_start "
                        f"session={int(tele_session_active(tele))} "
                        f"motion_ready={int(bool(getattr(tele, 'motion_data_ready', False)))} "
                        f"right_A={int(right_a)} left_X={int(left_x)} "
                        f"RT_bool={int(bool(getattr(tele, 'right_ctrl_trigger', False)))} "
                        f"RT={float(getattr(tele, 'right_ctrl_triggerValue', 10.0)):.3f} "
                        f"RS={float(getattr(tele, 'right_ctrl_squeezeValue', 0.0)):.3f} "
                        f"xr_age={xr_age:.3f}s",
                        flush=True,
                    )
                    last_wait_print = now
                stop.wait(0.02)
                continue

            xr_age = tele_xr_age(tele, time.monotonic())
            if not tele_session_active(tele):
                print("Start ignored: Quest Vuer session is not active.", flush=True)
                continue
            if not bool(getattr(tele, "motion_data_ready", False)) or xr_age > args.xr_stale_timeout:
                print(f"Start ignored: controller data is stale (age={xr_age:.3f}s).", flush=True)
                continue
            state = state_buffer.snapshot()
            if state is None:
                print("Robot lowstate is unavailable.", flush=True)
                continue
            upper_q, arm_q, arm_dq, gripper_q, gripper_dq, mode_machine, received_at = state
            if time.monotonic() - received_at > args.lowstate_timeout:
                print("Robot lowstate is stale.", flush=True)
                continue

            ik.reset_target_calibration(arm_q)
            left_target, right_target = ik.map_wrist_targets(
                tele.left_wrist_pose,
                tele.right_wrist_pose,
                arm_q,
                rotation_mode=args.controller_rotation_mode,
            )
            first_solution, _ = ik.solve_ik(
                left_target,
                right_target,
                arm_q,
                arm_dq,
            )
            first_solution = np.asarray(first_solution, dtype=float).reshape(14)
            start_error = float(np.max(np.abs(first_solution - arm_q)))
            print(f"Startup IK difference: {start_error:.3f} rad", flush=True)
            if start_error > args.max_start_error_rad:
                print("Controller alignment rejected; align again and press A or R.", flush=True)
                continue
            break

        if stop.is_set() or first_solution is None:
            return 0

        output.enable(
            upper_q,
            mode_machine,
            gripper_q=gripper_q,
            gripper_initial_mode=args.gripper_initial_mode,
            right_gripper_open_cap_current=args.right_gripper_open_cap_current,
        )
        time.sleep(0.10)
        output.set_arm_goal(first_solution, mode_machine)
        print("R1-A7 lowcmd enabled; TeleVuer tracking started.", flush=True)
        print("Type Q or press Ctrl+C to stop.", flush=True)

        period = 1.0 / args.ik_frequency
        next_tick = time.monotonic()
        session_lost_since: Optional[float] = None
        last_print = 0.0
        solve_count = 0
        tracking_enabled = True
        run_edges = ButtonEdge()
        tele = tv.get_tele_data()
        run_edges.previous["right_ctrl_aButton"] = bool(getattr(tele, "right_ctrl_aButton", False))
        run_edges.previous["left_ctrl_aButton"] = bool(getattr(tele, "left_ctrl_aButton", False))
        while not stop.is_set():
            now = time.monotonic()
            state = state_buffer.snapshot()
            if state is None:
                raise RuntimeError("robot lowstate disappeared")
            _upper_q, arm_q, arm_dq, gripper_q, gripper_dq, mode_machine, received_at = state
            if now - received_at > args.lowstate_timeout:
                raise RuntimeError(f"robot lowstate stale for {now - received_at:.3f}s")
            if output.error:
                raise RuntimeError(f"lowcmd publisher failed: {output.error}")

            tele = tv.get_tele_data()
            if run_edges.rising(tele, "right_ctrl_aButton"):
                tracking_enabled = not tracking_enabled
                if tracking_enabled:
                    ik.reset_target_calibration(arm_q)
                    print("Right A: teleoperation resumed from current arm posture.", flush=True)
                else:
                    with output.lock:
                        hold_q = output.upper_target[1:15].copy()
                    output.set_arm_goal(hold_q, mode_machine)
                    print("Right A: teleoperation paused; holding current commanded posture.", flush=True)
            if run_edges.rising(tele, "left_ctrl_aButton"):
                recorder.toggle()
            output.set_gripper_from_triggers(
                float(getattr(tele, "left_ctrl_triggerValue", 10.0)),
                float(getattr(tele, "right_ctrl_triggerValue", 10.0)),
            )
            output.update_gripper_contact(gripper_q, gripper_dq, now)

            session_active = tele_session_active(tele)
            if not session_active:
                output.hold_measured(arm_q, mode_machine)
                if session_lost_since is None:
                    session_lost_since = now
                elif now - session_lost_since >= args.session_close_timeout:
                    print("Quest page closed; stopping lowcmd publisher.", flush=True)
                    break
            else:
                session_lost_since = None

            xr_age = tele_xr_age(tele, now)
            fresh = bool(getattr(tele, "motion_data_ready", False)) and xr_age <= args.xr_stale_timeout
            input_valid = session_active and fresh
            if input_valid and tracking_enabled:
                left_target, right_target = ik.map_wrist_targets(
                    tele.left_wrist_pose,
                    tele.right_wrist_pose,
                    arm_q,
                    rotation_mode=args.controller_rotation_mode,
                )
                solution, _ = ik.solve_ik(left_target, right_target, arm_q, arm_dq)
                output.set_arm_goal(solution, mode_machine)
                solve_count += 1

            readable, _writable, _errors = select.select([sys.stdin], [], [], 0.0)
            if readable and sys.stdin.readline().strip().casefold() == "q":
                break

            with output.lock:
                sent_arm_q_for_record = output.upper_target[1:15].copy()
                goal_arm_q_for_record = output.arm_goal.copy()
                sent_gripper_q_for_record = output.gripper_target.copy()
                goal_gripper_q_for_record = output.gripper_goal.copy()
                gripper_contact_hold_for_record = output.gripper_hold.copy()
                gripper_contact_q_for_record = output.gripper_contact_q.copy()
            recorder.write(
                tele,
                tracking_enabled,
                input_valid,
                xr_age,
                arm_q,
                arm_dq,
                goal_arm_q_for_record,
                sent_arm_q_for_record,
                gripper_q,
                gripper_dq,
                goal_gripper_q_for_record,
                sent_gripper_q_for_record,
                gripper_contact_hold_for_record,
                gripper_contact_q_for_record,
            )

            if now - last_print >= args.print_period:
                with output.lock:
                    sent_arm_q = output.upper_target[1:15].copy()
                    goal_arm_q = output.arm_goal.copy()
                    sent_gripper_q = output.gripper_target.copy()
                    goal_gripper_q = output.gripper_goal.copy()
                    gripper_hold = output.gripper_hold.copy()

                # Read controller input directly from the current TeleVuer
                # object so these debug variables are always defined.
                debug_right_trigger = float(
                    getattr(tele, "right_ctrl_triggerValue", 10.0)
                )
                debug_right_alpha = float(
                    np.clip(
                        (10.0 - debug_right_trigger) / 10.0,
                        0.0,
                        1.0,
                    )
                )

                print(
                    f"tracking={int(session_active and fresh and tracking_enabled)} "
                    f"enabled={int(tracking_enabled)} "
                    f"session={int(session_active)} "
                    f"fresh={int(fresh)} "
                    f"solves={solve_count} "
                    f"xr_age={xr_age:.3f}s "
                    f"published={output.publish_count} "
                    f"right_trigger={debug_right_trigger:.3f} "
                    f"right_alpha={debug_right_alpha:.3f} "
                    f"right_goal={goal_gripper_q[1]:.5f} "
                    f"right_sent={sent_gripper_q[1]:.5f} "
                    f"right_measured={gripper_q[1]:.5f} "
                    f"right_dq={gripper_dq[1]:.5f} "
                    f"right_hold={int(gripper_hold[1])}",
                    flush=True,
                )
                last_print = now

            next_tick += period
            wait_time = next_tick - time.monotonic()
            if wait_time > 0.0:
                stop.wait(wait_time)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        print("Stop requested.", flush=True)
    finally:
        stop.set()
        if output is not None:
            output.close()
            print("R1-A7 lowcmd publisher stopped.", flush=True)
        recorder.stop()
        if tv is not None:
            tv.close()
        guard.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
