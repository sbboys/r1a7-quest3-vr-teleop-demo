#!/usr/bin/env python3
"""R1-A7 teleoperation using Unitree's default debug-mode execution path.

The input and IK path matches the verified MuJoCo program:
TeleVuer controller poses -> R1A7_ArmIK -> R1-A7 arm joint targets.
Only the real-robot executor differs: this program publishes CRC-protected
LowCmd messages on rt/lowcmd, as Unitree's default (non-motion) XR path does.
"""

from __future__ import annotations

from visual_servo_receiver import VisualServoReceiver

import argparse
import atexit
from collections import deque
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
import termios
import threading
import time
import tty
from typing import Optional

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XR_ROOT = Path("/home/robot/R1A7_VR_dual_arm_transfer_20260831_001/robot_dev/xr_teleoperate")
DEFAULT_SDK_PYTHON = Path("/home/robot/R1A7_VR_dual_arm_transfer_20260831_001/robot_dev/unitree_sdk2_python")

# Confirmed from the R1-A7 low-level examples and live lowstate diagnostics.
ARM_INDICES = tuple(range(15, 29))
UPPER_BODY_INDICES = (13, *range(15, 31))
GRIPPER_INDICES = (31, 33)
GRIPPER_NAMES = ("LEFT", "RIGHT_TWEEZER")
GRIPPER_OPEN_Q = np.asarray([4.86, 0.1585], dtype=float)
GRIPPER_CLOSE_Q = np.asarray([-0.08, -0.0500], dtype=float)

ARM_JOINT_NAMES = (
    "L_SHOULDER_PITCH", "L_SHOULDER_ROLL", "L_SHOULDER_YAW",
    "L_ELBOW", "L_WRIST_ROLL", "L_WRIST_PITCH", "L_WRIST_YAW",
    "R_SHOULDER_PITCH", "R_SHOULDER_ROLL", "R_SHOULDER_YAW",
    "R_ELBOW", "R_WRIST_ROLL", "R_WRIST_PITCH", "R_WRIST_YAW",
)

# Official A7 URDF limits. The margin prevents deliberate jog commands from
# driving directly onto a mechanical endpoint.
ARM_JOINT_LOWER_Q = np.asarray(
    [-3.1416, -0.22689, -1.9199, -0.97564, -1.9199, -1.6144, -1.6144,
     -3.1416, -2.47849, -1.9199, -0.97564, -1.9199, -1.6144, -1.6144],
    dtype=float,
)
ARM_JOINT_UPPER_Q = np.asarray(
    [2.0944, 2.4784, 1.9199, 2.1852, 1.9199, 1.6144, 1.6144,
     2.0944, 0.2268, 1.9199, 2.1852, 1.9199, 1.6144, 1.6144],
    dtype=float,
)
ARM_JOINT_LIMIT_MARGIN_RAD = np.deg2rad(2.0)

# RAW_KEYBOARD_CARTESIAN_JOG_V1
TERMINAL_KEY_INITIAL_REPEAT_GRACE_S = 0.0


def parse_joint_jog_command(command: str) -> Optional[tuple[int, int]]:
    """Parse joint:<0..13>:+/- commands used by the local console."""
    parts = command.strip().casefold().split(":")
    if len(parts) != 3 or parts[0] != "joint" or parts[2] not in ("+", "-"):
        return None
    try:
        joint_index = int(parts[1])
    except ValueError:
        return None
    if not 0 <= joint_index < len(ARM_INDICES):
        return None
    return joint_index, 1 if parts[2] == "+" else -1


def parse_gripper_command(command: str) -> Optional[tuple[int, str]]:
    """Parse gripper:<left|right>:<open|hold|close> commands."""
    parts = command.strip().casefold().split(":")
    if len(parts) != 3 or parts[0] != "gripper":
        return None
    side_to_index = {"left": 0, "right": 1}
    if parts[1] not in side_to_index or parts[2] not in ("open", "hold", "close"):
        return None
    return side_to_index[parts[1]], parts[2]


def parse_cartesian_jog_command(command: str) -> Optional[tuple[str, Optional[str]]]:
    """Parse deadman-style Cartesian jog commands from the web console."""
    parts = command.strip().casefold().split(":")
    if parts == ["jog", "stop"]:
        return "stop", None
    if (
        len(parts) == 3
        and parts[0] == "jog"
        and parts[1] in ("start", "keepalive")
        and parts[2] in ("w", "s", "a", "d", "u", "j", "i")
    ):
        return parts[1], parts[2]
    return None


def limit_joint_jog_target(
    current_q: float,
    direction: int,
    step_rad: float,
    lower_q: float,
    upper_q: float,
) -> tuple[float, bool]:
    """Apply a soft limit without pulling an already-outside joint abruptly."""
    requested_q = float(current_q + direction * step_rad)
    if current_q < lower_q:
        target_q = current_q if direction < 0 else min(requested_q, upper_q)
    elif current_q > upper_q:
        target_q = current_q if direction > 0 else max(requested_q, lower_q)
    else:
        target_q = float(np.clip(requested_q, lower_q, upper_q))
    return float(target_q), not np.isclose(requested_q, target_q)


def limit_cartesian_backlog(
    command_xyz: np.ndarray,
    actual_xyz: np.ndarray,
    max_distance_m: float,
    active_axes: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    """Limit look-ahead only inside the axes commanded by the operator.

    Scaling the complete XYZ error vector makes a nominal single-axis jog drift
    on its locked axes whenever the real arm has tracking error. Restricting
    the limiter to the active subspace preserves a horizontal/vertical path.
    """
    command = np.asarray(command_xyz, dtype=float).reshape(3)
    actual = np.asarray(actual_xyz, dtype=float).reshape(3)
    if active_axes is None:
        active = np.ones(3, dtype=bool)
    else:
        active = np.asarray(active_axes, dtype=bool).reshape(3)
    if not np.any(active):
        return command.copy(), False

    offset = command - actual
    distance = float(np.linalg.norm(offset[active]))
    if distance <= max_distance_m or distance <= 1.0e-12:
        return command.copy(), False

    limited = command.copy()
    limited[active] = (
        actual[active] + offset[active] / distance * max_distance_m
    )
    return limited, True

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
    parser.add_argument("--jog-step-mm", type=float, default=30.0,
                        help="Cartesian X/Y increment for each jog command")
    parser.add_argument("--jog-vertical-step-mm", type=float, default=10.0,
                        help="Cartesian Z increment for each jog command")
    parser.add_argument("--joint-jog-step-deg", type=float, default=1.0,
                        help="Joint angle increment for each joint jog command")
    parser.add_argument("--cartesian-jog-speed-mm-s", type=float, default=6.0,
                        help="Maximum Cartesian speed while a console jog button is held")
    parser.add_argument("--cartesian-jog-accel-mm-s2", type=float, default=20.0,
                        help="Cartesian acceleration/deceleration limit for console jog")
    parser.add_argument(
        "--ik-joint-accel-rad-s2",
        type=float,
        default=2.0,
        help="Acceleration limit applied to the 30 Hz IK joint target stream",
    )
    parser.add_argument(
        "--right-arm-gravity-feedforward-scale",
        type=float,
        default=0.0,
        help="Scale [0, 1] applied to model gravity torque on the right arm",
    )
    parser.add_argument(
        "--gravity-feedforward-max-torque",
        type=float,
        default=2.0,
        help="Absolute per-joint gravity feedforward limit in Nm",
    )
    parser.add_argument(
        "--gravity-feedforward-slew",
        type=float,
        default=2.0,
        help="Maximum gravity feedforward change in Nm/s",
    )
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
    for name in (
        "ik_frequency",
        "publish_frequency",
        "max_joint_speed",
        "cartesian_jog_speed_mm_s",
        "cartesian_jog_accel_mm_s2",
        "ik_joint_accel_rad_s2",
        "gravity_feedforward_max_torque",
        "gravity_feedforward_slew",
    ):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    if args.command_topic != "rt/lowcmd":
        parser.error("this debug-mode executor only permits --command-topic rt/lowcmd")
    if args.jog_step_mm <= 0.0 or args.jog_vertical_step_mm <= 0.0:
        parser.error("jog step sizes must be greater than zero")
    if not 0.1 <= args.joint_jog_step_deg <= 5.0:
        parser.error("--joint-jog-step-deg must be within [0.1, 5.0]")
    if not 0.0 <= args.right_arm_gravity_feedforward_scale <= 1.0:
        parser.error("--right-arm-gravity-feedforward-scale must be within [0, 1]")
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
        right_arm_gravity_feedforward_scale: float = 0.0,
        gravity_feedforward_max_torque: float = 2.0,
        gravity_feedforward_slew: float = 2.0,
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
        self.right_arm_gravity_feedforward_scale = float(
            right_arm_gravity_feedforward_scale
        )
        self.gravity_feedforward_max_torque = float(
            gravity_feedforward_max_torque
        )
        self.gravity_feedforward_max_step = float(
            gravity_feedforward_slew
        ) * self.period
        self.arm_tau_goal = np.zeros(len(ARM_INDICES), dtype=float)
        self.arm_tau_target = np.zeros(len(ARM_INDICES), dtype=float)
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
            self.arm_tau_goal[:] = 0.0
            self.arm_tau_target[:] = 0.0
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

    def set_arm_goal(
        self,
        arm_q: np.ndarray,
        mode_machine: int,
        arm_tau_ff: Optional[np.ndarray] = None,
    ) -> None:
        goal = np.asarray(arm_q, dtype=float).reshape(len(ARM_INDICES))
        if not np.all(np.isfinite(goal)):
            raise ValueError("refusing non-finite arm target")
        if arm_tau_ff is None:
            tau_goal = np.zeros(len(ARM_INDICES), dtype=float)
        else:
            tau_goal = np.asarray(arm_tau_ff, dtype=float).reshape(len(ARM_INDICES))
            if not np.all(np.isfinite(tau_goal)):
                raise ValueError("refusing non-finite arm torque target")
            tau_goal = tau_goal.copy()
            tau_goal[:7] = 0.0
            tau_goal[7:] *= self.right_arm_gravity_feedforward_scale
            tau_goal = np.clip(
                tau_goal,
                -self.gravity_feedforward_max_torque,
                self.gravity_feedforward_max_torque,
            )
        with self.lock:
            self.arm_goal = goal.copy()
            self.arm_tau_goal = tau_goal
            self.mode_machine = int(mode_machine)

    def set_waist_goal(self, waist_yaw: float, mode_machine: int) -> None:
        if not np.isfinite(waist_yaw):
            raise ValueError("refusing non-finite waist target")
        with self.lock:
            self.upper_target[0] = float(waist_yaw)
            self.mode_machine = int(mode_machine)

    def hold_measured(self, arm_q: np.ndarray, mode_machine: int) -> None:
        measured = np.asarray(arm_q, dtype=float).reshape(len(ARM_INDICES))
        if not np.all(np.isfinite(measured)):
            raise ValueError("refusing non-finite measured arm posture")
        with self.lock:
            # Synchronize both layers. Updating arm_goal alone leaves the 250 Hz
            # slew target executing the previous Cartesian segment for several
            # cycles after a direction change.
            self.upper_target[1:15] = measured
            self.arm_goal = measured.copy()
            self.mode_machine = int(mode_machine)

    def set_gripper_goal(self, index: int, q: float) -> float:
        if not self.enable_gripper:
            raise RuntimeError("gripper output is disabled")
        if not 0 <= index < len(GRIPPER_INDICES):
            raise ValueError("invalid gripper index")
        low = float(min(GRIPPER_OPEN_Q[index], GRIPPER_CLOSE_Q[index]))
        high = float(max(GRIPPER_OPEN_Q[index], GRIPPER_CLOSE_Q[index]))
        goal_q = float(np.clip(q, low, high))
        with self.lock:
            self.gripper_goal[index] = goal_q
            self.gripper_hold[index] = False
            self.gripper_stall_since[index] = np.nan
            self.gripper_contact_q[index] = np.nan
        return goal_q

    def hold_gripper_measured(self, index: int, measured_q: float) -> float:
        if not self.enable_gripper:
            raise RuntimeError("gripper output is disabled")
        if not 0 <= index < len(GRIPPER_INDICES):
            raise ValueError("invalid gripper index")
        hold_q = float(measured_q)
        if not np.isfinite(hold_q):
            raise ValueError("refusing non-finite gripper hold target")
        with self.lock:
            self.gripper_target[index] = hold_q
            self.gripper_goal[index] = hold_q
            self.gripper_hold[index] = False
            self.gripper_stall_since[index] = np.nan
            self.gripper_contact_q[index] = np.nan
        return hold_q

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
            left_open_q = GRIPPER_OPEN_Q[0]
            left_close_q = GRIPPER_CLOSE_Q[0]
            right_open_q = GRIPPER_OPEN_Q[1]
            right_close_q = GRIPPER_CLOSE_Q[1]
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
            arm_tau_delta = np.clip(
                self.arm_tau_goal - self.arm_tau_target,
                -self.gravity_feedforward_max_step,
                self.gravity_feedforward_max_step,
            )
            self.arm_tau_target += arm_tau_delta
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
            arm_tau_target = self.arm_tau_target.copy()
            mode_machine = self.mode_machine

        message.mode_pr = 0
        message.mode_machine = int(mode_machine)
        for local_index, motor_index in enumerate(UPPER_BODY_INDICES):
            motor = message.motor_cmd[motor_index]
            motor.mode = 1
            motor.q = float(upper_target[local_index])
            motor.dq = 0.0
            if 1 <= local_index <= len(ARM_INDICES):
                motor.tau = float(arm_tau_target[local_index - 1])
            else:
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
    diagnostic_trace_path = Path("/tmp") / (
        "r1a7_cartesian_trace_" + time.strftime("%Y%m%d_%H%M%S") + ".csv"
    )
    diagnostic_trace_rows = deque(
        maxlen=max(300, int(round(args.ik_frequency * 180.0)))
    )

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
            right_arm_gravity_feedforward_scale=(
                args.right_arm_gravity_feedforward_scale
            ),
            gravity_feedforward_max_torque=(
                args.gravity_feedforward_max_torque
            ),
            gravity_feedforward_slew=args.gravity_feedforward_slew,
        )

        print("WARNING: real R1-A7 debug-mode control on rt/lowcmd.", flush=True)
        if args.enable_gripper:
            print(
                "Internal DEX1 gripper enabled on LowCmd motors 31/33: "
                f"initial_mode={args.gripper_initial_mode}; "
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

        print(
            "Type ENABLE to arm this program (still no command is sent):",
            flush=True,
        )
        answer = input().strip()
        if answer.casefold() != "enable":
            print("Aborted before publishing.", flush=True)
            return 2

        # RAW_KEYBOARD_CARTESIAN_JOG_V1
        # Switch to single-character terminal input only AFTER ENABLE.
        terminal_fd = sys.stdin.fileno()
        terminal_original_attrs = termios.tcgetattr(terminal_fd)
        terminal_original_flags = fcntl.fcntl(
            terminal_fd,
            fcntl.F_GETFL,
        )

        def restore_terminal():
            try:
                termios.tcsetattr(
                    terminal_fd,
                    termios.TCSADRAIN,
                    terminal_original_attrs,
                )
                fcntl.fcntl(
                    terminal_fd,
                    fcntl.F_SETFL,
                    terminal_original_flags,
                )
            except Exception:
                pass

        tty.setcbreak(terminal_fd)
        fcntl.fcntl(
            terminal_fd,
            fcntl.F_SETFL,
            terminal_original_flags | os.O_NONBLOCK,
        )
        atexit.register(restore_terminal)

        print(
            "[KEYBOARD] Raw single-key mode active: "
            "press W/S/A/D/U/J/I to start continuous jog; "
            "SPACE or X = stop jog; T = marked hole approach; H = HOME; Q = quit. ""Key release alone does not stop terminal jog.",
            flush=True,
        )

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

        XY_MOVE_MM = args.jog_step_mm
        Z_MOVE_MM = args.jog_vertical_step_mm
        JOINT_JOG_STEP_RAD = np.deg2rad(args.joint_jog_step_deg)
        joint_soft_lower_q = ARM_JOINT_LOWER_Q + ARM_JOINT_LIMIT_MARGIN_RAD
        joint_soft_upper_q = ARM_JOINT_UPPER_Q - ARM_JOINT_LIMIT_MARGIN_RAD

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

        # ========================================================
        # KEYBOARD_6DOF_VR_V1
        #
        # Industrial-teach-pendant style orientation jog.
        #
        # Rotations are applied in the CURRENT TOOL/LOCAL frame.
        #
        #   8 / 2 : Pitch + / -
        #   4 / 6 : Yaw   + / -
        #   7 / 9 : Roll  + / -
        #
        # Text aliases are also accepted:
        #   pitch+ pitch-
        #   yaw+   yaw-
        #   roll+  roll-
        # ========================================================
        keyboard_6dof_rotation_commands = {
            "8": ("PITCH+", np.array([0.0, 1.0, 0.0]), +1.0),
            "2": ("PITCH-", np.array([0.0, 1.0, 0.0]), -1.0),
            "4": ("YAW+",   np.array([0.0, 0.0, 1.0]), +1.0),
            "6": ("YAW-",   np.array([0.0, 0.0, 1.0]), -1.0),
            "7": ("ROLL+",  np.array([1.0, 0.0, 0.0]), +1.0),
            "9": ("ROLL-",  np.array([1.0, 0.0, 0.0]), -1.0),

            "pitch+": ("PITCH+", np.array([0.0, 1.0, 0.0]), +1.0),
            "pitch-": ("PITCH-", np.array([0.0, 1.0, 0.0]), -1.0),
            "yaw+":   ("YAW+",   np.array([0.0, 0.0, 1.0]), +1.0),
            "yaw-":   ("YAW-",   np.array([0.0, 0.0, 1.0]), -1.0),
            "roll+":  ("ROLL+",  np.array([1.0, 0.0, 0.0]), +1.0),
            "roll-":  ("ROLL-",  np.array([1.0, 0.0, 0.0]), -1.0),
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

        def right_arm_jacobian_metrics(q):
            q = np.asarray(q, dtype=float).reshape(14)
            jacobian = pin.computeFrameJacobian(
                ik.reduced_robot.model,
                ik.reduced_robot.data,
                q,
                ik.R_hand_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:, 7:14]
            singular_values = np.linalg.svd(jacobian, compute_uv=False)
            minimum = float(singular_values[-1])
            condition = float(
                singular_values[0] / max(minimum, 1.0e-12)
            )
            return minimum, condition

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

        # Visual servo XY input
        visual_servo_receiver = VisualServoReceiver()
        visual_servo_enable = True

        cartesian_right_orientation = startup_right_pose[:3, :3].copy()

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
        VIRTUAL_VR_MAX_SPEED_M_S = args.cartesian_jog_speed_mm_s / 1000.0
        VIRTUAL_VR_MAX_ACCEL_M_S2 = args.cartesian_jog_accel_mm_s2 / 1000.0
        VIRTUAL_VR_POSITION_EPS_M = 0.00015

        # ====================================================
        # VR_POSTURE_MODE_V1
        #
        # Manual Cartesian motion should behave more like VR:
        #
        # - XYZ remains strongly controlled.
        # - World-frame end-effector orientation is only a
        #   SOFT reference instead of a hard practical lock.
        # - Right wrist pitch is softly encouraged to remain
        #   near its current joint angle, so it does not
        #   counter-rotate aggressively while shoulder/elbow
        #   lift the whole arm.
        #
        # TOOL_LOCK / visual precision mode will remain
        # unchanged in r1a7_visual_cartesian_control.py.
        # ====================================================

        VR_POSTURE_ROTATION_WEIGHT = 0.15

        VR_POSTURE_RIGHT_WRIST_PITCH_WEIGHT = 0.50

        # Approximately 4.6 deg per IK solution.
        # At ~30 Hz this is above normal commanded motion and
        # acts mainly as protection against sudden IK branch
        # jumps rather than a normal-motion limiter.
        VR_POSTURE_MAX_JOINT_STEP_RAD = 0.080

        # ====================================================
        # KEYBOARD_6DOF_VR_V1
        #
        # Keyboard Cartesian controller now commands a complete
        # end-effector SE(3) target:
        #
        #   translation: X / Y / Z
        #   rotation:    Roll / Pitch / Yaw
        #
        # Translation keys never overwrite orientation.
        # Rotation keys update a persistent orientation target.
        # ====================================================

        KEYBOARD_6DOF_ROT_STEP_DEG = 3.0

        # Deliberate orientation target should be tracked like
        # the full wrist pose supplied by the VR controller.
        KEYBOARD_6DOF_ROTATION_WEIGHT = 1.0

        # Disable the old per-frame wrist-pitch hold cost.
        # It fights deliberate Pitch commands and only penalizes
        # q(k)-q(k-1), rather than a meaningful tool pose.

        KEYBOARD_6DOF_RIGHT_WRIST_PITCH_WEIGHT = 0.0

        # ====================================================
        # ARM_POSTURE_MODE_V1
        #
        # Large Cartesian arm motions:
        # W/S/A/D/U/J/I
        #
        # Do not lock world tool orientation.
        # Keep wrist roll/pitch/yaw near the stored posture.
        # ====================================================

        ARM_POSTURE_ROTATION_WEIGHT = 0.05

        ARM_POSTURE_WRIST_POSTURE_WEIGHT = 0.2

        # ARM_POSTURE_V2
        #
        # Keep right elbow configuration while moving.

        # ARM_POSTURE_V2
        #
        # Weak elbow preference only.
        # The elbow must remain free to flex/extend during
        # natural Cartesian arm motion.
        # ELBOW_DIRECTION_BIAS_V1
        #
        # Soft elbow preference, NOT an elbow lock.
        #
        # Real-robot test confirmed:
        # +Z arm lift must bias right_elbow_joint toward
        # decreasing q[10].
        ARM_POSTURE_ELBOW_POSTURE_WEIGHT = 0.5

        # Desired elbow change for every +10 mm of Z motion.
        # Negative sign means: arm up -> elbow q decreases.
        ARM_POSTURE_ELBOW_UP_PER_10MM_RAD = np.deg2rad(2.0)

        # Do not let the heuristic posture bias grow without bound.
        # CONTINUOUS_DIRECTION_AND_ELBOW_FIX_V2
        # Remove the artificial fixed 20-degree elbow bias cap.
        ARM_POSTURE_ELBOW_BIAS_MAX_RAD = float("inf")
        # ====================================================
        # VR_CARTESIAN_BASELINE_V2
        #
        # Parameters restored from the verified continuous
        # Cartesian / VR-style controller.
        #
        # Normal tracking lag only SLOWS the virtual Cartesian
        # controller. It must not reduce normal motion speed to
        # zero.
        # ====================================================
        VIRTUAL_VR_LAG_WARN_MM = 30.0
        VIRTUAL_VR_LAG_SLOW_MM = 15.0
        VIRTUAL_VR_LAG_HIGH_MM = 35.0

        VIRTUAL_VR_MIN_SPEED_SCALE = 0.25

        # Maximum amount by which persistent desired position
        # may run ahead of the internal Cartesian command.
        #
        # This is NOT a per-key motion limit:
        # W=+30 mm is therefore allowed in full.
        VIRTUAL_VR_MAX_BACKLOG_M = 0.040

        # A held browser button uses velocity mode instead of repeatedly adding
        # position steps. The keepalive makes a lost pointer-up event fail safe.
        # Terminal jog stays active until explicit stop/direction change.
        CARTESIAN_JOG_WATCHDOG_S = float('inf')
        # DIRECT_CARTESIAN_CONTROL_V1
        # No artificial Cartesian command/actual backlog clamp
        # for manual Cartesian jog.
        CARTESIAN_JOG_MAX_COMMAND_LAG_M = float('inf')

        # IBVS must not integrate a visual target far ahead of
        # the measured real TCP.  Keep only a small look-ahead
        # during visual servoing.
        IBVS_MAX_COMMAND_LAG_M = float('inf')

        # Minimum resultant speed for active visual servoing.
        #
        # This avoids the Cartesian low-speed snap removing
        # small but meaningful IBVS corrections near the hole.
        #
        # Applied only when IBVS already reports a non-zero
        # valid velocity.
        IBVS_MIN_SPEED_MM_S = 1.0

        # The nominal IK target is generated from the commanded Cartesian
        # path.  On the real arm, gravity, gearbox compliance, and small model
        # errors can leave the measured TCP below that path while the joint
        # target is already correct.  Apply a bounded Cartesian feedback term
        # only to the internal IK target; the operator command and locked-axis
        # telemetry remain unchanged.
        # Disabled by default after the 2026-09-18 real-robot startup incident.
        # The isolated implementation remains for offline validation, but it
        # must not affect ENABLE until explicitly re-qualified.
        CARTESIAN_FEEDBACK_GAIN = 0.0
        CARTESIAN_FEEDBACK_MAX_M = 0.012

        # Detect a real reach/IK boundary from lack of measured progress instead
        # of imposing an arbitrary travel distance on every button hold.
        CARTESIAN_STALL_WINDOW_S = 0.75
        CARTESIAN_STALL_MIN_PROGRESS_M = 0.001
        CARTESIAN_STALL_WINDOWS_TO_STOP = 2

        # Stop fixed-orientation motion before the complete 6x7 right-arm
        # Jacobian becomes rank deficient. This follows the full configuration
        # instead of imposing an arbitrary elbow-angle limit.
        CARTESIAN_SINGULAR_CONDITION_STOP = 500.0

        # Sustained high error means the target is unreachable or the arm is
        # close to a singular configuration. Re-anchor instead of pushing it.
        # VR_CARTESIAN_BASELINE_V2
        # Abnormal tracking protection only.
        TRACKING_REANCHOR_LAG_MM = 60.0
        TRACKING_REANCHOR_RELEASE_MM = 45.0
        TRACKING_REANCHOR_DELAY_S = 1.00

        last_virtual_vr_tick = time.monotonic()

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
        if args.enable_gripper:
            print(
                "Startup gripper q [left, right] =",
                np.round(gripper_q, 5).tolist(),
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
            f"W : start continuous forward  +X at {args.cartesian_jog_speed_mm_s:g} mm/s",
            flush=True,
        )
        print(
            f"S : start continuous backward -X at {args.cartesian_jog_speed_mm_s:g} mm/s",
            flush=True,
        )
        print(
            f"A : start continuous left     +Y at {args.cartesian_jog_speed_mm_s:g} mm/s",
            flush=True,
        )
        print(
            f"D : start continuous right    -Y at {args.cartesian_jog_speed_mm_s:g} mm/s",
            flush=True,
        )
        print(
            f"U : start continuous up       +Z at {args.cartesian_jog_speed_mm_s:g} mm/s",
            flush=True,
        )
        print(
            f"J : start continuous down     -Z at {args.cartesian_jog_speed_mm_s:g} mm/s",
            flush=True,
        )
        print(
            "I : start continuous forward-up (+X and +Z together)",
            flush=True,
        )
        print(
            "H : smooth return to startup q",
            flush=True,
        )
        print(
            "joint:<0..13>:+/- : single-joint jog "
            f"{args.joint_jog_step_deg:g} deg per command",
            flush=True,
        )
        if args.enable_gripper:
            print(
                "gripper:<left|right>:<open|hold|close> : gripper command",
                flush=True,
            )
        print(
            "Q : quit",
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
        control_mode = "cartesian"

        # ARM_POSTURE_MODE_V1
        #
        # Stored right wrist configuration.
        # Large Cartesian motions keep this posture
        # instead of forcing world-frame tool orientation.
        #
        # q[11] right_wrist_roll
        # q[12] right_wrist_pitch
        # q[13] right_wrist_yaw

        right_wrist_posture_ref = arm_q[11:14].copy()

        # ARM_POSTURE_V2
        #
        # Keep right elbow configuration.
        #
        # R1-A7:
        # q[10] = right_elbow_joint

        right_elbow_posture_ref = arm_q[10]

        joint_target_q = startup_arm_q.copy()
        active_joint_index = None
        active_cartesian_jog = None
        active_cartesian_jog_seen_at = 0.0
        active_cartesian_jog_start_xyz = None

        # IBVS_FILE_VELOCITY_INPUT_V1
        #
        # External vision controller writes:
        #   /tmp/r1a7_dual_ibvs_velocity.json
        #
        # Velocity convention:
        #   vx_mm_s -> Base +X
        #   vy_mm_s -> Base +Y
        #   vz_mm_s -> Base +Z
        #
        # This is only another Cartesian velocity source.
        # The existing acceleration limiter, IK and LowCmd path
        # remain unchanged.
        ibvs_command_path = Path(
            "/tmp/r1a7_dual_ibvs_velocity.json"
        )
        ibvs_velocity_active = False
        ibvs_velocity_m_s = np.zeros(
            3,
            dtype=float
        )
        cartesian_jog_releasing = False
        cartesian_progress_started_at = None
        cartesian_progress_start_xyz = None
        cartesian_stall_windows = 0
        stream_joint_goal = startup_arm_q.copy()
        stream_joint_velocity_rad_s = np.zeros(14, dtype=float)
        tracking_lag_started_at = None
        tracking_guard_count = 0
        # Match the verified VR controller: initialize the IK filter once when
        # Cartesian control is armed, not once per direction-button press.
        ik.reset_target_calibration(startup_arm_q)

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

            gripper_q = np.asarray(gripper_q, dtype=float).reshape(2)
            gripper_dq = np.asarray(gripper_dq, dtype=float).reshape(2)
            output.update_gripper_contact(gripper_q, gripper_dq, now)

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
                try:
                    raw_keys = os.read(
                        sys.stdin.fileno(),
                        64,
                    )
                except BlockingIOError:
                    raw_keys = b""

                decoded_keys = raw_keys.decode(
                    errors="ignore"
                )

                # Ignore terminal escape sequences such as arrow keys.
                # Otherwise ESC-[A could accidentally be interpreted as A.
                if "\x1b" in decoded_keys:
                    command = ""
                else:
                    key_candidates = [
                        ch.casefold()
                        for ch in decoded_keys
                        if ch.casefold()
                        in "wsadujithq824679x "
                    ]

                    if "q" in key_candidates:
                        command = "q"
                    elif (
                        " " in key_candidates
                        or "x" in key_candidates
                    ):
                        command = "jog:stop"
                    elif key_candidates:
                        command = key_candidates[-1]
                    else:
                        command = ""

                if command == "q":
                    print(
                        "[QUIT] User requested stop.",
                        flush=True,
                    )
                    break

                joint_jog = parse_joint_jog_command(command)
                gripper_command = parse_gripper_command(command)

                if command in direction_delta_mm:
                    if active_cartesian_jog is None:
                        # stopped -> moving
                        cartesian_jog = (
                            "start",
                            command,
                        )

                    elif active_cartesian_jog == command:
                        # Same direction; keep existing Cartesian state.
                        cartesian_jog = (
                            "keepalive",
                            command,
                        )

                    else:
                        # CONTINUOUS_DIRECTION_SWITCH_V2
                        #
                        # Moving -> different direction:
                        # only change requested direction.
                        #
                        # Do NOT re-anchor/reinitialize:
                        #   hold_right_target
                        #   desired_right_target
                        #   stream_joint_goal
                        #   cartesian_right_orientation
                        #   cartesian_velocity_m_s
                        previous_jog = active_cartesian_jog
                        active_cartesian_jog = command
                        active_cartesian_jog_seen_at = time.monotonic()
                        cartesian_jog = None

                        print(
                            "[CARTESIAN JOG] SWITCH "
                            f"{previous_jog.upper()} -> "
                            f"{command.upper()} "
                            "(continuous; no re-anchor)",
                            flush=True,
                        )
                else:
                    cartesian_jog = parse_cartesian_jog_command(
                        command
                    )

                if cartesian_jog is not None:
                    jog_action, jog_direction = cartesian_jog
                    if jog_action == "stop":
                        if active_cartesian_jog is not None:
                            print(
                                f"[CARTESIAN JOG] STOP {direction_name[active_cartesian_jog]}",
                                flush=True,
                            )
                        cartesian_jog_releasing = active_cartesian_jog is not None
                        active_cartesian_jog = None
                        active_cartesian_jog_seen_at = 0.0
                        active_cartesian_jog_start_xyz = None
                        desired_right_target[:3, 3] = hold_right_target[:3, 3]
                        cartesian_progress_started_at = None
                        cartesian_progress_start_xyz = None
                        cartesian_stall_windows = 0
                    elif motion is not None:
                        if jog_action == "start":
                            print(
                                "[CARTESIAN JOG] ignored while HOME is active.",
                                flush=True,
                            )
                    elif jog_action == "keepalive":
                        if active_cartesian_jog == jog_direction:
                            active_cartesian_jog_seen_at = now
                    else:
                        if control_mode != "cartesian":
                            synced_left_pose, synced_right_pose = get_ee_poses_from_q(arm_q)
                            hold_left_target = synced_left_pose.copy()
                            hold_right_target = synced_right_pose.copy()
                            desired_left_target = synced_left_pose.copy()
                            desired_right_target = synced_right_pose.copy()
                            cartesian_right_orientation = (
                                synced_right_pose[:3, :3].copy()
                            )
                            cartesian_velocity_m_s[:] = 0.0
                            stream_joint_goal = arm_q.copy()
                            stream_joint_velocity_rad_s[:] = 0.0
                            ik.reset_target_calibration(arm_q)
                            tracking_lag_started_at = None
                            control_mode = "cartesian"
                            active_joint_index = None
                            print(
                                "[MODE] CARTESIAN; TCP position and orientation "
                                "synchronized from measured q.",
                                flush=True,
                            )
                        # Start every new direction from the measured TCP. This
                        # discards residual XYZ and joint look-ahead from the
                        # preceding direction while retaining tool orientation.
                        _synced_left_pose, synced_right_pose = get_ee_poses_from_q(
                            arm_q
                        )
                        hold_right_target[:3, 3] = synced_right_pose[:3, 3]
                        desired_right_target[:3, 3] = synced_right_pose[:3, 3]
                        cartesian_velocity_m_s[:] = 0.0
                        stream_joint_goal = arm_q.copy()
                        stream_joint_velocity_rad_s[:] = 0.0
                        output.hold_measured(arm_q, mode_machine)
                        tracking_lag_started_at = None
                        active_cartesian_jog = jog_direction
                        active_cartesian_jog_seen_at = (
                            now
                            + TERMINAL_KEY_INITIAL_REPEAT_GRACE_S
                        )
                        active_cartesian_jog_start_xyz = hold_right_target[:3, 3].copy()
                        cartesian_jog_releasing = False
                        cartesian_progress_started_at = now
                        cartesian_progress_start_xyz = synced_right_pose[:3, 3].copy()
                        cartesian_stall_windows = 0
                        print(
                            f"[CARTESIAN JOG] START {direction_name[jog_direction]}",
                            flush=True,
                        )

                elif command in keyboard_6dof_rotation_commands:
                    if (
                        motion is not None
                        and motion.get("type") == "joint_home"
                    ):
                        print(
                            "[ROT TARGET] ignored while HOME is active.",
                            flush=True,
                        )
                    else:
                        active_cartesian_jog = None
                        active_cartesian_jog_seen_at = 0.0
                        active_cartesian_jog_start_xyz = None
                        cartesian_jog_releasing = False
                        cartesian_progress_started_at = None
                        cartesian_progress_start_xyz = None
                        cartesian_stall_windows = 0

                        # Enter Cartesian mode from the CURRENT measured pose
                        # exactly once. Do not re-anchor on later rotation keys.
                        if control_mode != "cartesian":
                            synced_left_pose, synced_right_pose = (
                                get_ee_poses_from_q(arm_q)
                            )
                            hold_left_target = synced_left_pose.copy()
                            hold_right_target = synced_right_pose.copy()
                            desired_left_target = synced_left_pose.copy()
                            desired_right_target = synced_right_pose.copy()
                            cartesian_right_orientation = (
                                synced_right_pose[:3, :3].copy()
                            )
                            cartesian_velocity_m_s[:] = 0.0
                            stream_joint_goal = arm_q.copy()
                            stream_joint_velocity_rad_s[:] = 0.0
                            ik.reset_target_calibration(arm_q)
                            tracking_lag_started_at = None
                            control_mode = "cartesian"
                            active_joint_index = None
                            print(
                                "[MODE] CARTESIAN 6DOF; pose synchronized "
                                "from measured q.",
                                flush=True,
                            )

                        (
                            rotation_name,
                            rotation_axis_local,
                            rotation_sign,
                        ) = keyboard_6dof_rotation_commands[command]

                        theta = np.deg2rad(
                            KEYBOARD_6DOF_ROT_STEP_DEG * rotation_sign
                        )

                        axis = np.asarray(
                            rotation_axis_local,
                            dtype=float,
                        )
                        axis /= max(float(np.linalg.norm(axis)), 1.0e-12)

                        kx, ky, kz = axis.tolist()

                        K = np.array(
                            [
                                [0.0, -kz, ky],
                                [kz, 0.0, -kx],
                                [-ky, kx, 0.0],
                            ],
                            dtype=float,
                        )

                        rotation_increment = (
                            np.eye(3)
                            + np.sin(theta) * K
                            + (1.0 - np.cos(theta)) * (K @ K)
                        )

                        # IMPORTANT:
                        # Post-multiply => rotation in current TOOL/LOCAL frame.
                        new_orientation = (
                            cartesian_right_orientation
                            @ rotation_increment
                        )

                        # Numerical re-orthonormalization.
                        U_rot, _, Vt_rot = np.linalg.svd(
                            new_orientation
                        )
                        new_orientation = U_rot @ Vt_rot

                        if np.linalg.det(new_orientation) < 0.0:
                            U_rot[:, -1] *= -1.0
                            new_orientation = U_rot @ Vt_rot

                        cartesian_right_orientation = (
                            new_orientation.copy()
                        )

                        # Persistent orientation target.
                        desired_right_target[:3, :3] = (
                            cartesian_right_orientation
                        )
                        hold_right_target[:3, :3] = (
                            cartesian_right_orientation
                        )

                        print(
                            "[ROT TARGET] "
                            f"{rotation_name} "
                            f"{KEYBOARD_6DOF_ROT_STEP_DEG:.1f} deg "
                            "(tool/local frame)",
                            flush=True,
                        )
                        print(
                            "[ROT TARGET] R =\n"
                            + np.array2string(
                                cartesian_right_orientation,
                                precision=4,
                                suppress_small=True,
                            ),
                            flush=True,
                        )

                # REMOVE_LEGACY_CARTESIAN_STEP_V1
                # Legacy W/S/A/D/U/J fixed-step target branch removed.
                #
                # Translation keys are now handled only by the
                # continuous Cartesian jog state machine above.
                #
                # W/S/A/D/U/J no longer:
                #   - clear active_cartesian_jog
                #   - add fixed 30/10 mm targets
                #   - stop after reaching a finite desired target

                elif joint_jog is not None:
                    if motion is not None:
                        print(
                            "[JOINT JOG] ignored while another motion is active.",
                            flush=True,
                        )
                    else:
                        active_cartesian_jog = None
                        active_cartesian_jog_seen_at = 0.0
                        active_cartesian_jog_start_xyz = None
                        cartesian_jog_releasing = False
                        joint_index, joint_direction = joint_jog
                        if control_mode != "joint":
                            with output.lock:
                                joint_target_q = output.arm_goal.copy()
                            cartesian_velocity_m_s[:] = 0.0
                            stream_joint_velocity_rad_s[:] = 0.0
                            tracking_lag_started_at = None
                            control_mode = "joint"
                            print(
                                "[MODE] JOINT; holding the current commanded arm posture.",
                                flush=True,
                            )

                        current_target_q = float(joint_target_q[joint_index])
                        lower_q = float(joint_soft_lower_q[joint_index])
                        upper_q = float(joint_soft_upper_q[joint_index])
                        limited_q, limited = limit_joint_jog_target(
                            current_target_q,
                            joint_direction,
                            JOINT_JOG_STEP_RAD,
                            lower_q,
                            upper_q,
                        )
                        joint_target_q[joint_index] = limited_q
                        active_joint_index = joint_index
                        output.set_arm_goal(joint_target_q, mode_machine)

                        print(
                            "[JOINT JOG]"
                            f" {ARM_JOINT_NAMES[joint_index]}"
                            f" {'+' if joint_direction > 0 else '-'}"
                            f"{args.joint_jog_step_deg:.2f} deg"
                            f" -> target={limited_q:+.5f} rad"
                            + (" [SOFT LIMIT]" if limited else ""),
                            flush=True,
                        )

                elif gripper_command is not None:
                    gripper_index, gripper_action = gripper_command
                    if not args.enable_gripper:
                        print(
                            "[GRIPPER] command ignored because output is disabled.",
                            flush=True,
                        )
                    else:
                        if gripper_action == "hold":
                            gripper_goal_q = output.hold_gripper_measured(
                                gripper_index,
                                gripper_q[gripper_index],
                            )
                        else:
                            requested_gripper_q = (
                                GRIPPER_OPEN_Q[gripper_index]
                                if gripper_action == "open"
                                else GRIPPER_CLOSE_Q[gripper_index]
                            )
                            gripper_goal_q = output.set_gripper_goal(
                                gripper_index,
                                requested_gripper_q,
                            )
                        print(
                            f"[GRIPPER] {GRIPPER_NAMES[gripper_index]} "
                            f"{gripper_action.upper()} -> "
                            f"target={gripper_goal_q:+.5f}",
                            flush=True,
                        )

                # =====================================================
                # TASK6_MARKED_HOLE_TARGET_V1
                #
                # T = move the calibrated tool TCP to the manually
                # marked hole approach point.
                #
                # Vision provides TCP target in Base coordinates.
                # Existing IK controls R_ee, therefore:
                #
                #   p_tcp_base =
                #       p_ee_base + R_base_ee @ p_tip_ee
                #
                # so:
                #
                #   p_ee_goal =
                #       p_tcp_goal - R_base_ee @ p_tip_ee
                #
                # No new controller is introduced here.
                # The resulting EE goal is handed to the existing
                # Cartesian motion -> solve_ik -> LowCmd path.
                # =====================================================
                elif command == "t":

                    project_root = os.path.abspath(
                        os.path.join(
                            os.path.dirname(__file__),
                            "..",
                            "..",
                        )
                    )

                    hole_json_path = (
                        "/tmp/r1a7_task6_hole_base.json"
                    )

                    tcp_json_path = os.path.join(
                        project_root,
                        "tool_calibration",
                        "tcp_pivot",
                        "r1a7_right_tool_tcp.json",
                    )

                    with open(
                        hole_json_path,
                        "r",
                        encoding="utf-8",
                    ) as f:
                        hole_data = json.load(f)

                    with open(
                        tcp_json_path,
                        "r",
                        encoding="utf-8",
                    ) as f:
                        tcp_data = json.load(f)

                    tcp_goal_base = np.asarray(
                        hole_data["approach_base_m"],
                        dtype=float,
                    ).reshape(3)

                    p_tip_ee = np.asarray(
                        tcp_data["P_tip_ee_m"],
                        dtype=float,
                    ).reshape(3)

                    (
                        synced_left_pose,
                        synced_right_pose,
                    ) = get_ee_poses_from_q(
                        arm_q
                    )

                    r_base_ee = (
                        synced_right_pose[
                            :3,
                            :3
                        ].copy()
                    )

                    start_xyz = (
                        synced_right_pose[
                            :3,
                            3
                        ].copy()
                    )

                    ee_goal_xyz = (
                        tcp_goal_base
                        - r_base_ee @ p_tip_ee
                    )

                    delta_m = (
                        ee_goal_xyz
                        - start_xyz
                    )

                    distance_m = float(
                        np.linalg.norm(
                            delta_m
                        )
                    )

                    # Use the EXISTING Cartesian speed and
                    # acceleration settings to determine the
                    # quintic trajectory duration.
                    speed_m_s = (
                        float(
                            args.cartesian_jog_speed_mm_s
                        )
                        / 1000.0
                    )

                    accel_m_s2 = (
                        float(
                            args.cartesian_jog_accel_mm_s2
                        )
                        / 1000.0
                    )

                    if speed_m_s <= 0.0:
                        raise RuntimeError(
                            "cartesian jog speed must be > 0"
                        )

                    if accel_m_s2 <= 0.0:
                        raise RuntimeError(
                            "cartesian jog acceleration must be > 0"
                        )

                    # For the existing quintic time-scaling:
                    #
                    # max ds/dtau  = 1.875
                    # max d2s/dtau2 ~= 5.7735
                    #
                    # Choose duration from the same configured
                    # Cartesian speed/acceleration.
                    if distance_m > 1.0e-9:
                        duration_speed = (
                            1.875
                            * distance_m
                            / speed_m_s
                        )

                        duration_accel = np.sqrt(
                            5.7735
                            * distance_m
                            / accel_m_s2
                        )

                        duration = float(
                            max(
                                duration_speed,
                                duration_accel,
                            )
                        )
                    else:
                        duration = 0.0

                    # Stop any terminal jog and synchronize the
                    # existing Cartesian controller to measured q.
                    active_cartesian_jog = None
                    active_cartesian_jog_seen_at = 0.0
                    active_cartesian_jog_start_xyz = None
                    cartesian_jog_releasing = False
                    cartesian_progress_started_at = None
                    cartesian_progress_start_xyz = None
                    cartesian_stall_windows = 0

                    hold_left_target = (
                        synced_left_pose.copy()
                    )

                    hold_right_target = (
                        synced_right_pose.copy()
                    )

                    desired_left_target = (
                        synced_left_pose.copy()
                    )

                    desired_right_target = (
                        synced_right_pose.copy()
                    )

                    cartesian_right_orientation = (
                        r_base_ee.copy()
                    )

                    cartesian_velocity_m_s[:] = 0.0

                    stream_joint_goal = arm_q.copy()
                    stream_joint_velocity_rad_s[:] = 0.0

                    ik.reset_target_calibration(
                        arm_q
                    )

                    tracking_lag_started_at = None
                    control_mode = "cartesian"
                    active_joint_index = None

                    output.hold_measured(
                        arm_q,
                        mode_machine,
                    )

                    print()
                    print(
                        "[TASK6 VISUAL TARGET]",
                        flush=True,
                    )

                    print(
                        "  TCP target Base mm =",
                        np.round(
                            tcp_goal_base
                            * 1000.0,
                            3,
                        ).tolist(),
                        flush=True,
                    )

                    print(
                        "  current EE Base mm =",
                        np.round(
                            start_xyz
                            * 1000.0,
                            3,
                        ).tolist(),
                        flush=True,
                    )

                    print(
                        "  EE target Base mm  =",
                        np.round(
                            ee_goal_xyz
                            * 1000.0,
                            3,
                        ).tolist(),
                        flush=True,
                    )

                    print(
                        "  EE delta mm        =",
                        np.round(
                            delta_m
                            * 1000.0,
                            3,
                        ).tolist(),
                        flush=True,
                    )

                    print(
                        f"  distance mm        = "
                        f"{distance_m * 1000.0:.3f}",
                        flush=True,
                    )

                    if distance_m <= 1.0e-9:
                        print(
                            "[TASK6 VISUAL TARGET] "
                            "already at target.",
                            flush=True,
                        )
                    else:
                        motion = {
                            "type": "cartesian",
                            "name":
                                "TASK6_MARKED_HOLE_APPROACH",
                            "started_at":
                                now,
                            "duration":
                                duration,
                            "left_start":
                                synced_left_pose.copy(),
                            "right_start":
                                synced_right_pose.copy(),
                            "start_xyz":
                                start_xyz.copy(),
                            "delta_m":
                                delta_m.copy(),
                            "goal_xyz":
                                ee_goal_xyz.copy(),
                        }

                        print(
                            "[STREAM START] "
                            "TASK6_MARKED_HOLE_APPROACH "
                            f"duration={duration:.2f}s",
                            flush=True,
                        )

                elif command == "h":
                    active_cartesian_jog = None
                    active_cartesian_jog_seen_at = 0.0
                    active_cartesian_jog_start_xyz = None
                    cartesian_jog_releasing = False
                    desired_left_target = (
                        startup_left_pose.copy()
                    )

                    desired_right_target = (
                        startup_right_pose.copy()
                    )

                    cartesian_velocity_m_s[:] = 0.0
                    control_mode = "home"
                    active_joint_index = None

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
                        "Use W/S/A/D/U/J/I/T/H/Q, jog:<start|keepalive>:<direction>, "
                        "jog:stop, joint:<0..13>:+/-, or "
                        "gripper:<left|right>:<open|hold|close>.",
                        flush=True,
                    )

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

                    desired_left_target = startup_left_pose.copy()
                    desired_right_target = startup_right_pose.copy()
                    cartesian_right_orientation = (
                        startup_right_pose[:3, :3].copy()
                    )
                    stream_joint_goal = startup_arm_q.copy()
                    stream_joint_velocity_rad_s[:] = 0.0
                    tracking_lag_started_at = None
                    joint_target_q = startup_arm_q.copy()
                    control_mode = "cartesian"
                    active_joint_index = None
                    motion = None

            # -----------------------------------------------
            # Direct joint-space jog hold. The 250 Hz output layer applies the
            # configured max joint speed while this loop owns the target.
            # -----------------------------------------------
            elif motion is None and control_mode == "joint":
                output.set_arm_goal(joint_target_q, mode_machine)

                if now - last_print >= args.print_period:
                    _left_actual, right_actual = get_ee_poses_from_q(arm_q)
                    _left_goal, right_goal = get_ee_poses_from_q(joint_target_q)
                    actual_xyz_mm = right_actual[:3, 3] * 1000.0
                    command_xyz_mm = right_goal[:3, 3] * 1000.0
                    command_actual_lag_mm = float(
                        np.linalg.norm(command_xyz_mm - actual_xyz_mm)
                    )
                    with output.lock:
                        diagnostic_gripper_goal = output.gripper_goal.copy()
                        diagnostic_gripper_target = output.gripper_target.copy()
                        diagnostic_gripper_hold = output.gripper_hold.copy()

                    print(
                        "[JOINT HOLD]",
                        ARM_JOINT_NAMES[active_joint_index]
                        if active_joint_index is not None else "ARM",
                        "target=",
                        np.round(joint_target_q, 4).tolist(),
                        flush=True,
                    )
                    print(
                        "[TELEMETRY_JSON] "
                        + json.dumps(
                            {
                                "timestamp": time.time(),
                                "control_mode": "joint",
                                "active_joint_index": active_joint_index,
                                "arm_q_actual": np.round(arm_q, 7).tolist(),
                                "arm_q_goal": np.round(joint_target_q, 7).tolist(),
                                "gripper_q_actual": np.round(gripper_q, 7).tolist(),
                                "gripper_q_goal": np.round(diagnostic_gripper_goal, 7).tolist(),
                                "gripper_q_command": np.round(diagnostic_gripper_target, 7).tolist(),
                                "gripper_contact_hold": diagnostic_gripper_hold.tolist(),
                                "desired_xyz_mm": np.round(command_xyz_mm, 3).tolist(),
                                "command_xyz_mm": np.round(command_xyz_mm, 3).tolist(),
                                "actual_xyz_mm": np.round(actual_xyz_mm, 3).tolist(),
                                "lag_mm": round(command_actual_lag_mm, 3),
                                "speed_scale": 1.0,
                            },
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )
                    last_print = now

            # -----------------------------------------------
            # VR-style continuous IK hold.
            #
            # The VR controller keeps solving IK even when
            # the human stops moving the controller.
            #
            # Every cycle uses the latest measured q/dq and
            # solves the complete 7-DOF arm again.
            # -----------------------------------------------
            elif motion is None and control_mode == "cartesian":
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
                left_actual_now, right_actual_now = (
                    get_ee_poses_from_q(
                        arm_q
                    )
                )

                actual_xyz_now = (
                    right_actual_now[
                        :3, 3
                    ].copy()
                )

                (
                    right_jacobian_min_singular,
                    right_jacobian_condition,
                ) = right_arm_jacobian_metrics(arm_q)

                command_actual_lag_mm = float(
                    np.linalg.norm(
                        current_command_xyz
                        - actual_xyz_now
                    )
                    * 1000.0
                )

                if command_actual_lag_mm > TRACKING_REANCHOR_LAG_MM:
                    if tracking_lag_started_at is None:
                        tracking_lag_started_at = now
                elif command_actual_lag_mm < TRACKING_REANCHOR_RELEASE_MM:
                    tracking_lag_started_at = None

                if (
                    tracking_lag_started_at is not None
                    and now - tracking_lag_started_at >= TRACKING_REANCHOR_DELAY_S
                ):
                    tracking_guard_count += 1
                    print(
                        "[TRACKING GUARD] sustained Cartesian lag "
                        f"{command_actual_lag_mm:.1f} mm; stopping and "
                        "re-anchoring to measured pose.",
                        flush=True,
                    )
                    active_cartesian_jog = None
                    active_cartesian_jog_seen_at = 0.0
                    active_cartesian_jog_start_xyz = None
                    cartesian_jog_releasing = False
                    cartesian_velocity_m_s[:] = 0.0
                    hold_left_target = left_actual_now.copy()
                    desired_left_target = left_actual_now.copy()
                    hold_right_target = right_actual_now.copy()
                    desired_right_target = right_actual_now.copy()
                    cartesian_right_orientation = (
                        right_actual_now[:3, :3].copy()
                    )
                    stream_joint_goal = arm_q.copy()
                    stream_joint_velocity_rad_s[:] = 0.0
                    output.hold_measured(arm_q, mode_machine)
                    current_command_xyz = actual_xyz_now.copy()
                    desired_xyz = actual_xyz_now.copy()
                    command_actual_lag_mm = 0.0
                    tracking_lag_started_at = None
                    cartesian_progress_started_at = None
                    cartesian_progress_start_xyz = None
                    cartesian_stall_windows = 0

                if (
                    active_cartesian_jog is not None
                    and now - active_cartesian_jog_seen_at > CARTESIAN_JOG_WATCHDOG_S
                ):
                    print(
                        "[CARTESIAN JOG] watchdog timeout; stopping.",
                        flush=True,
                    )
                    active_cartesian_jog = None
                    active_cartesian_jog_seen_at = 0.0
                    active_cartesian_jog_start_xyz = None
                    cartesian_jog_releasing = True
                    desired_right_target[:3, 3] = current_command_xyz
                    cartesian_progress_started_at = None
                    cartesian_progress_start_xyz = None
                    cartesian_stall_windows = 0

                # DIRECTIONAL_SINGULAR_GUARD_FIX_V1
                # Do not assume W/U/I always move deeper into a singular
                # configuration. The real R1-A7 test showed that S can
                # increase the Jacobian condition while W may move back
                # toward a better configuration.
                #
                # Keep the metric visible for diagnostics, but do not
                # forcibly cancel the operator's Cartesian jog here.
                if (
                    active_cartesian_jog is not None
                    and right_jacobian_condition
                    >= CARTESIAN_SINGULAR_CONDITION_STOP
                ):
                    print(
                        "[KINEMATIC WARNING] right-arm Jacobian condition="
                        f"{right_jacobian_condition:.1f}; "
                        f"continuing {direction_name[active_cartesian_jog]} "
                        "without the old hard-coded directional stop.",
                        flush=True,
                    )

                if active_cartesian_jog is not None:
                    progress_delta = direction_delta_mm[active_cartesian_jog]
                    progress_direction = progress_delta / float(
                        np.linalg.norm(progress_delta)
                    )
                    if (
                        cartesian_progress_started_at is None
                        or cartesian_progress_start_xyz is None
                    ):
                        cartesian_progress_started_at = now
                        cartesian_progress_start_xyz = actual_xyz_now.copy()
                    elif (
                        now - cartesian_progress_started_at
                        >= CARTESIAN_STALL_WINDOW_S
                    ):
                        measured_progress_m = float(
                            np.dot(
                                actual_xyz_now - cartesian_progress_start_xyz,
                                progress_direction,
                            )
                        )
                        if measured_progress_m < CARTESIAN_STALL_MIN_PROGRESS_M:
                            cartesian_stall_windows += 1
                        else:
                            cartesian_stall_windows = 0
                        cartesian_progress_started_at = now
                        cartesian_progress_start_xyz = actual_xyz_now.copy()

                        if (
                            cartesian_stall_windows
                            >= CARTESIAN_STALL_WINDOWS_TO_STOP
                        ):
                            print(
                                "[PROGRESS WARNING] TCP made no progress in the "
                                "requested direction; VR-equivalent execution "
                                "keeps the command active while lag and Jacobian "
                                "safety guards remain armed.",
                                flush=True,
                            )
                            cartesian_stall_windows = 0

                cart_error = (
                    desired_xyz
                    - current_command_xyz
                )

                cart_distance = float(
                    np.linalg.norm(
                        cart_error
                    )
                )

                # ===========================================
                # IBVS_FILE_VELOCITY_INPUT_V1
                #
                # Read PID velocity generated by:
                # task6_external_aruco_pid.py
                #
                # No separate LowCmd publisher is introduced.
                # ===========================================

                previous_ibvs_velocity_active = (
                    ibvs_velocity_active
                )

                ibvs_velocity_active = False
                ibvs_velocity_m_s[:] = 0.0

                try:
                    if ibvs_command_path.exists():

                        ibvs_data = json.loads(
                            ibvs_command_path.read_text()
                        )

                        if bool(
                            ibvs_data.get(
                                "enabled",
                                False
                            )
                        ):

                            candidate_velocity_m_s = (
                                np.array(
                                    [
                                        float(
                                            ibvs_data.get(
                                                "vx_mm_s",
                                                0.0
                                            )
                                        ),
                                        float(
                                            ibvs_data.get(
                                                "vy_mm_s",
                                                0.0
                                            )
                                        ),
                                        float(
                                            ibvs_data.get(
                                                "vz_mm_s",
                                                0.0
                                            )
                                        ),
                                    ],
                                    dtype=float,
                                )
                                / 1000.0
                            )

                            if np.all(
                                np.isfinite(
                                    candidate_velocity_m_s
                                )
                            ):

                                # Reuse the EXISTING Cartesian
                                # maximum speed setting.
                                candidate_speed_m_s = float(
                                    np.linalg.norm(
                                        candidate_velocity_m_s
                                    )
                                )

                                if (
                                    candidate_speed_m_s
                                    >
                                    VIRTUAL_VR_MAX_SPEED_M_S
                                    and
                                    candidate_speed_m_s
                                    >
                                    0.0
                                ):
                                    candidate_velocity_m_s *= (
                                        VIRTUAL_VR_MAX_SPEED_M_S
                                        /
                                        candidate_speed_m_s
                                    )

                                # ---------------------------------------
                                # IBVS minimum resultant speed
                                #
                                # Preserve direction. Do not force each axis
                                # independently.
                                #
                                # Example:
                                # [0.1,0.3,0] mm/s
                                # becomes approximately
                                # [0.32,0.95,0] mm/s
                                # instead of [1,1,1].
                                # ---------------------------------------

                                candidate_speed_mm_s = (
                                    candidate_speed_m_s
                                    * 1000.0
                                )

                                if (
                                    candidate_speed_mm_s > 0.0
                                    and
                                    candidate_speed_mm_s
                                    <
                                    IBVS_MIN_SPEED_MM_S
                                ):
                                    candidate_velocity_m_s *= (
                                        IBVS_MIN_SPEED_MM_S
                                        /
                                        candidate_speed_mm_s
                                    )

                                ibvs_velocity_m_s[:] = (
                                    candidate_velocity_m_s
                                )

                                ibvs_velocity_active = True

                except Exception:
                    ibvs_velocity_active = False
                    ibvs_velocity_m_s[:] = 0.0


                if (
                    ibvs_velocity_active
                    and
                    not previous_ibvs_velocity_active
                ):

                    print(
                        "[IBVS INPUT] ENABLED "
                        f"vx={ibvs_velocity_m_s[0] * 1000.0:+.3f} "
                        f"vy={ibvs_velocity_m_s[1] * 1000.0:+.3f} "
                        f"vz={ibvs_velocity_m_s[2] * 1000.0:+.3f} "
                        "mm/s",
                        flush=True,
                    )


                if (
                    previous_ibvs_velocity_active
                    and
                    not ibvs_velocity_active
                ):

                    print(
                        "[IBVS INPUT] DISABLED; "
                        "decelerating with existing Cartesian limiter.",
                        flush=True,
                    )

                    cartesian_jog_releasing = True


                if ibvs_velocity_active:

                    # IBVS takes the Cartesian velocity input.
                    # Stop terminal jog state, but do not
                    # re-anchor or reset IK.
                    active_cartesian_jog = None
                    active_cartesian_jog_seen_at = 0.0
                    active_cartesian_jog_start_xyz = None
                    cartesian_jog_releasing = False


                # -------------------------------------------
                # Generate a smooth Cartesian velocity.
                #
                # sqrt(2*a*d) creates automatic deceleration
                # as the virtual controller approaches its
                # destination.
                # -------------------------------------------

                if ibvs_velocity_active:

                    # PID output is already a Cartesian
                    # velocity vector in Base coordinates.
                    #
                    # It now enters exactly the same downstream
                    # acceleration limiter / target integration /
                    # IK / LowCmd chain as keyboard Cartesian jog.
                    lag_speed_scale = 1.0

                    desired_velocity = (
                        ibvs_velocity_m_s.copy()
                    )

                    desired_right_target[
                        :3, 3
                    ] = current_command_xyz

                    desired_xyz = (
                        current_command_xyz
                    )

                elif active_cartesian_jog is not None:
                    jog_delta = direction_delta_mm[active_cartesian_jog]
                    cart_direction = jog_delta / float(np.linalg.norm(jog_delta))
                    # The target is clamped near measured TCP below. Do not
                    # accumulate a distant target and then throttle to zero.
                    lag_speed_scale = 1.0
                    desired_velocity = (
                        cart_direction * VIRTUAL_VR_MAX_SPEED_M_S
                    )
                    desired_right_target[:3, 3] = current_command_xyz
                    desired_xyz = current_command_xyz
                elif cartesian_jog_releasing:
                    lag_speed_scale = 1.0
                    desired_velocity = np.zeros(3, dtype=float)
                    desired_right_target[:3, 3] = current_command_xyz
                    desired_xyz = current_command_xyz
                elif (
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

                if (
                    cartesian_jog_releasing
                    and float(np.linalg.norm(cartesian_velocity_m_s)) < 5.0e-5
                ):
                    cartesian_velocity_m_s[:] = 0.0
                    cartesian_jog_releasing = False

                next_command_xyz = (
                    current_command_xyz
                    + cartesian_velocity_m_s
                    * dt
                )

                if ibvs_velocity_active:

                    active_axes = (
                        np.abs(
                            ibvs_velocity_m_s
                        )
                        >
                        1.0e-12
                    )

                    next_command_xyz, _backlog_limited = (
                        limit_cartesian_backlog(
                            next_command_xyz,
                            actual_xyz_now,
                            IBVS_MAX_COMMAND_LAG_M,
                            active_axes,
                        )
                    )

                elif active_cartesian_jog is not None:
                    active_axes = np.abs(
                        direction_delta_mm[active_cartesian_jog]
                    ) > 1.0e-12
                    next_command_xyz, _backlog_limited = limit_cartesian_backlog(
                        next_command_xyz,
                        actual_xyz_now,
                        CARTESIAN_JOG_MAX_COMMAND_LAG_M,
                        active_axes,
                    )

                # If the target is extremely close and the
                # stream is nearly stopped, snap only the
                # numerical reference to the exact target.
                if (
                    not ibvs_velocity_active
                    and
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


                # ===========================================
                # VISUAL SERVO XY CORRECTION
                #
                # Unit:
                # receiver: mm
                # Cartesian target: m
                # ===========================================

                visual_delta = (
                    visual_servo_receiver.update()
                )


                if visual_servo_enable:

                    hold_right_target[:3,3] += (
                        np.asarray(
                            visual_delta,
                            dtype=float
                        )
                        /
                        1000.0
                    )


                # Keep the nominal command transform immutable.  Feedback is
                # applied only to a temporary IK target while a deadman jog is
                # active.  Mutating hold_right_target here would feed the
                # correction back into the next cycle and make ENABLE move the
                # arm without operator input.
                cartesian_feedback = np.zeros(3, dtype=float)
                if (
                    active_cartesian_jog is not None
                    or
                    ibvs_velocity_active
                ):
                    cartesian_feedback_error = (
                        next_command_xyz - actual_xyz_now
                    )
                    cartesian_feedback = np.clip(
                        CARTESIAN_FEEDBACK_GAIN * cartesian_feedback_error,
                        -CARTESIAN_FEEDBACK_MAX_M,
                        CARTESIAN_FEEDBACK_MAX_M,
                    )
                ik_target_xyz = (
                    next_command_xyz + cartesian_feedback
                )

                # Hold the current Cartesian orientation reference. Entering
                # Cartesian mode after a joint jog refreshes this reference
                # from measured q, so a manual wrist adjustment is preserved.
                hold_right_target[
                    :3, :3
                ] = cartesian_right_orientation

                ik_right_target = hold_right_target.copy()
                ik_right_target[:3, 3] = ik_target_xyz

                # Match the verified VR controller exactly at this boundary:
                # solve from the latest measured arm state, then hand the IK
                # solution directly to the existing 250 Hz speed-limited
                # LowCmd output.  A second joint-space trajectory generator
                # here caused the console to chase its delayed internal state.
                ik_solve_started = time.perf_counter()
                # =================================================
                # VR_POSTURE_MODE_V1
                #
                # Keep Cartesian position control strong, but do not
                # force the wrist to counter-rotate just to preserve
                # the startup world-frame tool orientation.
                #
                # right_wrist_pitch_weight is a SOFT joint continuity
                # term already implemented inside R1A7_ArmIK.
                # =================================================
                # OFFICIAL_VR_TAUFF_V1
                # Keep the gravity/feed-forward torque returned by R1A7_ArmIK.
                # =============================================
                # ELBOW_DIRECTION_BIAS_V1
                #
                # The real robot showed the wrong redundancy
                # branch during +Z lift: right elbow q[10]
                # increased strongly.
                #
                # Bias the elbow in the opposite direction while
                # still leaving it free to move.
                #
                # Use the smoothed IK Cartesian target so the
                # elbow reference moves continuously.
                # =============================================
                elbow_z_delta_m = float(
                    ik_right_target[2, 3]
                    - startup_right_pose[2, 3]
                )

                elbow_bias_rad = (
                    -ARM_POSTURE_ELBOW_UP_PER_10MM_RAD
                    * (elbow_z_delta_m / 0.010)
                )

                elbow_bias_rad = float(
                    np.clip(
                        elbow_bias_rad,
                        -ARM_POSTURE_ELBOW_BIAS_MAX_RAD,
                        +ARM_POSTURE_ELBOW_BIAS_MAX_RAD,
                    )
                )

                dynamic_right_elbow_posture_ref = float(
                    np.clip(
                        startup_arm_q[10] + elbow_bias_rad,
                        joint_soft_lower_q[10],
                        joint_soft_upper_q[10],
                    )
                )

                raw_stream_solution, raw_stream_tau_ff = ik.solve_ik(
                    hold_left_target,
                    ik_right_target,
                    arm_q,
                    arm_dq,
                    position_only=False,
                    max_joint_step=VR_POSTURE_MAX_JOINT_STEP_RAD,
                    # KEYBOARD_6DOF_VR_V1
                    # Explicit user-defined tool orientation.
                    # ARM_POSTURE_MODE_V1
                    #
                    # Large Cartesian motions should not force
                    # world-frame tool orientation.
                    #
                    # Keep wrist roll/pitch/yaw near the stored
                    # operator posture while shoulders/elbow
                    # perform the main motion.

                    rotation_weight=ARM_POSTURE_ROTATION_WEIGHT,

                    right_wrist_pitch_weight=(
                        KEYBOARD_6DOF_RIGHT_WRIST_PITCH_WEIGHT
                    ),

                    right_wrist_posture_weight=(
                        ARM_POSTURE_WRIST_POSTURE_WEIGHT
                    ),

                    right_wrist_posture_ref=(
                        right_wrist_posture_ref
                    ),

                    # ARM_POSTURE_V2
                    right_elbow_posture_weight=(
                        ARM_POSTURE_ELBOW_POSTURE_WEIGHT
                    ),

                    right_elbow_posture_ref=(
                        dynamic_right_elbow_posture_ref
                    ),
                )
                ik_solve_ms = (
                    time.perf_counter() - ik_solve_started
                ) * 1000.0

                raw_stream_solution = np.asarray(
                    raw_stream_solution,
                    dtype=float,
                ).reshape(14)

                if not np.all(
                    np.isfinite(
                        raw_stream_solution
                    )
                ):
                    raise RuntimeError(
                        "Virtual VR IK produced "
                        "non-finite q"
                    )

                stream_solution = raw_stream_solution.copy()
                stream_joint_velocity_rad_s = (
                    stream_solution - stream_joint_goal
                ) / max(dt, 1.0e-4)
                stream_joint_goal = stream_solution.copy()

                # OFFICIAL_VR_TAUFF_V1
                # The output layer applies:
                #   - right-arm-only scaling
                #   - finite-value checks
                #   - torque clipping
                #   - torque slew limiting
                output.set_arm_goal(
                    stream_solution,
                    mode_machine,
                    arm_tau_ff=raw_stream_tau_ff,
                )

                with output.lock:
                    sent_arm_target = output.upper_target[1:15].copy()
                    sent_arm_tau = output.arm_tau_target.copy()
                diagnostic_trace_rows.append(
                    (
                        now,
                        dt,
                        active_cartesian_jog or "",
                        ik_solve_ms,
                        float(np.linalg.norm(cartesian_velocity_m_s) * 1000.0),
                        command_actual_lag_mm,
                        arm_q.copy(),
                        arm_dq.copy(),
                        raw_stream_solution.copy(),
                        stream_solution.copy(),
                        sent_arm_target,
                        sent_arm_tau,
                        right_jacobian_min_singular,
                        right_jacobian_condition,
                        current_command_xyz.copy() * 1000.0,
                        actual_xyz_now.copy() * 1000.0,
                    )
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
                    orientation_error_deg = float(
                        np.rad2deg(
                            np.linalg.norm(
                                pin.log3(
                                    right_actual[:3, :3].T
                                    @ hold_right_target[:3, :3]
                                )
                            )
                        )
                    )
                    max_joint_command_speed_deg_s = float(
                        np.rad2deg(
                            np.max(np.abs(stream_joint_velocity_rad_s))
                        )
                    )
                    # VIRTUAL_VR_RIGHT_JOINT_DIAG_V1
                    with output.lock:
                        diagnostic_goal_q = (
                            output.arm_goal.copy()
                        )
                        diagnostic_gripper_goal = output.gripper_goal.copy()
                        diagnostic_gripper_target = output.gripper_target.copy()
                        diagnostic_gripper_hold = output.gripper_hold.copy()
                        diagnostic_arm_tau = output.arm_tau_target.copy()


                    # ===========================================
                    # TRACKING_SPLIT_DIAGNOSTIC_V1
                    #
                    # Cartesian error decomposition:
                    #
                    # command -> effective IK target
                    # IK target -> FK of joint goal
                    # joint goal FK -> measured FK
                    # ===========================================
                    _left_goal_diag, right_goal_diag = (
                        get_ee_poses_from_q(
                            diagnostic_goal_q
                        )
                    )

                    goal_xyz_now = (
                        right_goal_diag[:3, 3].copy()
                    )

                    effective_ik_xyz = (
                        ik_right_target[:3, 3].copy()
                    )

                    command_to_ik_mm = float(
                        np.linalg.norm(
                            current_command_xyz
                            - effective_ik_xyz
                        )
                        * 1000.0
                    )

                    ik_to_goal_mm = float(
                        np.linalg.norm(
                            effective_ik_xyz
                            - goal_xyz_now
                        )
                        * 1000.0
                    )

                    goal_to_actual_mm = float(
                        np.linalg.norm(
                            goal_xyz_now
                            - actual_xyz_now
                        )
                        * 1000.0
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
                        "rot_err_deg=",
                        f"{orientation_error_deg:.2f}",
                        "joint_cmd_deg_s=",
                        f"{max_joint_command_speed_deg_s:.1f}",
                        "speed_scale=",
                        f"{lag_speed_scale:.2f}",
                        flush=True,
                    )

                    print(
                        "[TRACKING SPLIT]",
                        "commandXYZ=",
                        np.round(
                            current_command_xyz * 1000.0,
                            2,
                        ).tolist(),
                        "ikTargetXYZ=",
                        np.round(
                            effective_ik_xyz * 1000.0,
                            2,
                        ).tolist(),
                        "goalXYZ=",
                        np.round(
                            goal_xyz_now * 1000.0,
                            2,
                        ).tolist(),
                        "actualXYZ=",
                        np.round(
                            actual_xyz_now * 1000.0,
                            2,
                        ).tolist(),
                        "cmd->ik=",
                        f"{command_to_ik_mm:.1f}",
                        "ik->goal=",
                        f"{ik_to_goal_mm:.1f}",
                        "goal->actual=",
                        f"{goal_to_actual_mm:.1f}",

                        # ELBOW_REFERENCE_DIAGNOSTIC_V1
                        "elbow_dz_mm=",
                        f"{elbow_z_delta_m * 1000.0:.1f}",
                        "elbow_bias_deg=",
                        f"{np.rad2deg(elbow_bias_rad):.1f}",
                        "elbow_ref_deg=",
                        f"{np.rad2deg(dynamic_right_elbow_posture_ref):.1f}",
                        "elbow_goal_deg=",
                        f"{np.rad2deg(diagnostic_goal_q[10]):.1f}",
                        "elbow_actual_deg=",
                        f"{np.rad2deg(arm_q[10]):.1f}",
                        "wrist_goal_deg=",
                        f"{np.rad2deg(diagnostic_goal_q[12]):.1f}",
                        "wrist_actual_deg=",
                        f"{np.rad2deg(arm_q[12]):.1f}",

                        flush=True,
                    )

                    print(
                        "[TELEMETRY_JSON] "
                        + json.dumps(
                            {
                                "timestamp": time.time(),
                                "control_mode": "cartesian",
                                "motion_state": (
                                    "continuous_jog"
                                    if active_cartesian_jog is not None
                                    else (
                                        "decelerating"
                                        if float(np.linalg.norm(cartesian_velocity_m_s)) >= 5.0e-4
                                        else "holding"
                                    )
                                ),
                                "jog_direction": active_cartesian_jog,
                                "command_velocity_mm_s": round(
                                    float(np.linalg.norm(cartesian_velocity_m_s)) * 1000.0,
                                    3,
                                ),
                                "active_joint_index": None,
                                "arm_q_actual": np.round(arm_q, 7).tolist(),
                                "arm_q_goal": np.round(diagnostic_goal_q, 7).tolist(),
                                "arm_tau_ff": np.round(diagnostic_arm_tau, 7).tolist(),
                                "gripper_q_actual": np.round(gripper_q, 7).tolist(),
                                "gripper_q_goal": np.round(diagnostic_gripper_goal, 7).tolist(),
                                "gripper_q_command": np.round(diagnostic_gripper_target, 7).tolist(),
                                "gripper_contact_hold": diagnostic_gripper_hold.tolist(),
                                "desired_xyz_mm": np.round(desired_xyz_mm, 3).tolist(),
                                "command_xyz_mm": np.round(command_xyz_mm, 3).tolist(),
                                "actual_xyz_mm": np.round(actual_xyz_mm, 3).tolist(),
                                "ik_target_xyz_mm": np.round(
                                    effective_ik_xyz * 1000.0,
                                    3,
                                ).tolist(),
                                "goal_xyz_mm": np.round(
                                    goal_xyz_now * 1000.0,
                                    3,
                                ).tolist(),
                                "command_to_ik_mm": round(
                                    command_to_ik_mm,
                                    3,
                                ),
                                "ik_to_goal_mm": round(
                                    ik_to_goal_mm,
                                    3,
                                ),
                                "goal_to_actual_mm": round(
                                    goal_to_actual_mm,
                                    3,
                                ),
                                "lag_mm": round(command_actual_lag_mm, 3),
                                "orientation_error_deg": round(
                                    orientation_error_deg,
                                    3,
                                ),
                                "orientation_locked": True,
                                "max_joint_command_speed_deg_s": round(
                                    max_joint_command_speed_deg_s,
                                    3,
                                ),
                                "tracking_guard_count": tracking_guard_count,
                                "right_jacobian_min_singular": round(
                                    right_jacobian_min_singular,
                                    7,
                                ),
                                "right_jacobian_condition": round(
                                    right_jacobian_condition,
                                    2,
                                ),
                                "speed_scale": round(float(lag_speed_scale), 4),
                            },
                            separators=(",", ":"),
                        ),
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

                        q_actual_diag = float(
                            arm_q[idx]
                        )

                        print(
                            f"  {joint_name:18s} "
                            f"goal={q_goal_diag:+.5f} "
                            f"actual={q_actual_diag:+.5f} "
                            f"error="
                            f"{q_goal_diag-q_actual_diag:+.5f}",
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
        if diagnostic_trace_rows:
            try:
                with diagnostic_trace_path.open(
                    "w", newline="", encoding="utf-8"
                ) as trace_file:
                    trace_writer = csv.writer(trace_file)
                    trace_writer.writerow(
                        [
                            "time_monotonic",
                            "loop_dt_s",
                            "jog_direction",
                            "ik_solve_ms",
                            "cartesian_speed_mm_s",
                            "cartesian_lag_mm",
                            "arm_q_actual",
                            "arm_dq_actual",
                            "arm_q_ik_raw",
                            "arm_q_stream_goal",
                            "arm_q_lowcmd_sent",
                            "arm_tau_ff_sent",
                            "right_jacobian_min_singular",
                            "right_jacobian_condition",
                            "command_xyz_mm",
                            "actual_xyz_mm",
                        ]
                    )
                    for row in diagnostic_trace_rows:
                        trace_writer.writerow(
                            [
                                f"{row[0]:.9f}",
                                f"{row[1]:.9f}",
                                row[2],
                                f"{row[3]:.6f}",
                                f"{row[4]:.6f}",
                                f"{row[5]:.6f}",
                                *(
                                    json.dumps(
                                        np.asarray(value, dtype=float).tolist(),
                                        separators=(",", ":"),
                                    )
                                    for value in row[6:]
                                ),
                            ]
                        )
                print(
                    "[DIAGNOSTIC TRACE] saved "
                    f"{len(diagnostic_trace_rows)} samples to "
                    f"{diagnostic_trace_path}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    "[DIAGNOSTIC TRACE] save failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
        recorder.stop()
        if tv is not None:
            tv.close()
        guard.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# KEYBOARD_6DOF_VR_V1 INSTALLED
