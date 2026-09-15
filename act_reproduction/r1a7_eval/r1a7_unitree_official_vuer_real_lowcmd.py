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


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ACT_CAMERA_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "act_cameras.yaml"
)
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
        default=Path("/data/R1A7/episodes"),
    )
    parser.add_argument(
        "--record-episode-id",
        default="ACT_wrench_001",
    )
    parser.add_argument(
        "--act-cameras",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable persistent ACT three-camera recording. "
            "Use --no-act-cameras for robot-only debugging."
        ),
    )
    parser.add_argument(
        "--act-camera-config",
        type=Path,
        default=DEFAULT_ACT_CAMERA_CONFIG,
    )
    parser.add_argument(
        "--startup-abort-before-publisher",
        action="store_true",
        help=(
            "Diagnostic safety mode: receive lowstate, "
            "check debug mode, start/validate ACT cameras, "
            "then exit before creating the lowcmd publisher."
        ),
    )
    parser.add_argument("--record-duration-s", type=float, default=60.0)
    parser.add_argument("--record-d435i", action=argparse.BooleanOptionalAction, default=False)
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
    parser.add_argument(
        "--act-real",
        action="store_true",
        help=(
            "Use realtime ACT joint-space policy instead of "
            "TeleVuer IK for arm/gripper goals."
        ),
    )
    parser.add_argument(
        "--act-socket",
        default="/tmp/r1a7_act.sock",
        help="Unix socket of the external ACT inference server.",
    )
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

    def next_episode_dir(self) -> Path:
        """Return the next available episode directory path.

        This does not create the directory. It allows robot-state
        recording and the ACT camera manager to share exactly the
        same episode directory.
        """
        return self._next_dir()

    def toggle(self) -> None:
        if self.active:
            self.stop()
        else:
            self.start()

    def start(
        self,
        episode_dir: Optional[Path] = None,
    ) -> Path:
        if self.active:
            if self.episode_dir is None:
                raise RuntimeError(
                    "recorder active but episode_dir is None"
                )
            return self.episode_dir

        if episode_dir is None:
            self.episode_dir = self._next_dir()
            self.episode_dir.mkdir(
                parents=True,
                exist_ok=False,
            )
        else:
            self.episode_dir = (
                Path(episode_dir)
                .expanduser()
                .resolve()
            )

            self.episode_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            for filename in (
                "metadata.json",
                "states.csv",
            ):
                target = (
                    self.episode_dir
                    / filename
                )

                if target.exists():
                    raise RuntimeError(
                        f"robot recording output already exists: "
                        f"{target}"
                    )

        metadata = {
            "robot": "Unitree R1-A7",
            "mode": "R1A7_ArmIK TeleVuer lowcmd simple csv",
            "episode_id": self.episode_dir.name,
            "created_at_system": time.time(),
            "states_schema_version": "r1a7_act_v1",
            "arm_dim": 14,
            "gripper_dim": 2,
            "act_qpos_dim": 16,
            "act_action_dim": 16,
            "recording_mode": "x_toggle_shared_act_episode",
            "legacy_d435i_subprocess_enabled": False,
            "note": "X toggles a shared ACT episode: robot states.csv plus three-camera RGB/depth recording when ACT cameras are enabled. Legacy per-episode D435i subprocess recording is disabled by default.",
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
            "gripper_q", "gripper_dq",
            "goal_gripper_q", "sent_gripper_q",
            "left_wrist_pose", "right_wrist_pose",
            "left_trigger", "right_trigger", "right_A", "left_X",
        ])
        self.active = True
        self.samples = 0
        self.started_at = time.monotonic()
        print(f"X RECORD START: {self.episode_dir}", flush=True)

        return self.episode_dir

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
        gripper_q: np.ndarray,
        gripper_dq: np.ndarray,
        goal_gripper_q: np.ndarray,
        sent_gripper_q: np.ndarray,
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
            json.dumps(np.asarray(gripper_q, dtype=float).tolist()),
            json.dumps(np.asarray(gripper_dq, dtype=float).tolist()),
            json.dumps(np.asarray(goal_gripper_q, dtype=float).tolist()),
            json.dumps(np.asarray(sent_gripper_q, dtype=float).tolist()),
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
        self.mode_machine = 0
        self.enabled = False
        self.stop = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.publish_count = 0
        self.error = ""

    def enable(self, upper_q: np.ndarray, mode_machine: int) -> None:
        upper = np.asarray(upper_q, dtype=float).reshape(len(UPPER_BODY_INDICES))
        with self.lock:
            self.upper_target = upper.copy()
            self.arm_goal = upper[1:15].copy()
            self.gripper_target = GRIPPER_OPEN_Q.copy()
            self.gripper_goal = GRIPPER_OPEN_Q.copy()
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

    def set_gripper_from_triggers(self, left_trigger: float, right_trigger: float) -> None:
        if not self.enable_gripper:
            return
        # TeleVuer convention: 10.0=released/open, 0.0=fully pulled/closed.
        alpha = np.clip((10.0 - np.asarray([left_trigger, right_trigger], dtype=float)) / 10.0, 0.0, 1.0)
        goal = GRIPPER_OPEN_Q + alpha * (GRIPPER_CLOSE_Q - GRIPPER_OPEN_Q)
        with self.lock:
            opening = goal > self.gripper_goal + 0.02
            self.gripper_hold[opening] = False
            self.gripper_stall_since[opening] = np.nan
            self.gripper_contact_q[opening] = np.nan
            self.gripper_goal = goal.copy()

    def set_gripper_openness_goal(self, openness: np.ndarray) -> None:
        if not self.enable_gripper:
            return

        openness = np.asarray(
            openness,
            dtype=float,
        ).reshape(2)

        if not np.all(np.isfinite(openness)):
            raise ValueError(
                "refusing non-finite ACT gripper target"
            )

        openness = np.clip(
            openness,
            0.0,
            1.0,
        )

        # ACT V1 convention:
        # 1.0 = calibrated open
        # 0.0 = calibrated close
        goal = (
            GRIPPER_CLOSE_Q
            + openness
            * (GRIPPER_OPEN_Q - GRIPPER_CLOSE_Q)
        )

        with self.lock:
            opening = (
                goal
                > self.gripper_goal
                + 0.02
            )

            self.gripper_hold[opening] = False
            self.gripper_stall_since[opening] = np.nan
            self.gripper_contact_q[opening] = np.nan
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



def build_act_qpos(
    arm_q: np.ndarray,
    gripper_q: np.ndarray,
) -> np.ndarray:
    """Build ACT V1 canonical 16D qpos.

    Order:
      left_arm_7,
      left_gripper_openness,
      right_arm_7,
      right_gripper_openness
    """

    arm_q = np.asarray(
        arm_q,
        dtype=np.float32,
    ).reshape(14)

    gripper_q = np.asarray(
        gripper_q,
        dtype=np.float32,
    ).reshape(2)

    denominator = (
        GRIPPER_OPEN_Q
        - GRIPPER_CLOSE_Q
    )

    openness = (
        gripper_q
        - GRIPPER_CLOSE_Q
    ) / denominator

    openness = np.clip(
        openness,
        0.0,
        1.0,
    )

    qpos = np.empty(
        16,
        dtype=np.float32,
    )

    qpos[0:7] = arm_q[0:7]
    qpos[7] = openness[0]

    qpos[8:15] = arm_q[7:14]
    qpos[15] = openness[1]

    return qpos


class ActRealtimeClient:
    def __init__(
        self,
        socket_path: str,
        timeout_s: float = 1.0,
    ):
        import socket

        self.socket_path = str(
            socket_path
        )

        self.sock = socket.socket(
            socket.AF_UNIX,
            socket.SOCK_STREAM,
        )

        self.sock.settimeout(
            float(timeout_s)
        )

        self.sock.connect(
            self.socket_path
        )

    @staticmethod
    def _recv_exact(sock, n):
        parts = []
        remaining = int(n)

        while remaining:
            chunk = sock.recv(
                remaining
            )

            if not chunk:
                raise ConnectionError(
                    "ACT inference server closed socket"
                )

            parts.append(chunk)
            remaining -= len(chunk)

        return b"".join(parts)

    def _send_object(self, obj):
        import pickle
        import struct

        payload = pickle.dumps(
            obj,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

        self.sock.sendall(
            struct.pack(
                "!Q",
                len(payload),
            )
            + payload
        )

    def _recv_object(self):
        import pickle
        import struct

        header = self._recv_exact(
            self.sock,
            8,
        )

        size = struct.unpack(
            "!Q",
            header,
        )[0]

        payload = self._recv_exact(
            self.sock,
            size,
        )

        return pickle.loads(
            payload
        )

    def ping(self):
        self._send_object(
            {
                "command": "ping",
            }
        )

        response = (
            self._recv_object()
        )

        if (
            not response.get("ok")
            or response.get("command")
            != "pong"
        ):
            raise RuntimeError(
                f"unexpected ACT ping response: "
                f"{response}"
            )

        return response

    def infer(
        self,
        qpos: np.ndarray,
        frames,
    ):
        qpos = np.asarray(
            qpos,
            dtype=np.float32,
        ).reshape(16)

        images = {}

        for role in (
            "top",
            "left_wrist",
            "right_wrist",
        ):
            image = np.asarray(
                frames[role][
                    "image_bgr"
                ]
            )

            if (
                image.shape
                != (480, 640, 3)
                or image.dtype
                != np.uint8
            ):
                raise RuntimeError(
                    f"{role}: invalid ACT image "
                    f"shape={image.shape} "
                    f"dtype={image.dtype}"
                )

            images[role] = image

        self._send_object(
            {
                "qpos": qpos,
                "images": images,
            }
        )

        response = (
            self._recv_object()
        )

        if not response.get(
            "ok",
            False,
        ):
            raise RuntimeError(
                "ACT inference failed: "
                + str(
                    response.get(
                        "error",
                        "unknown server error",
                    )
                )
            )

        action = np.asarray(
            response["action"],
            dtype=np.float32,
        )

        if action.shape != (
            100,
            16,
        ):
            raise RuntimeError(
                f"unexpected ACT action shape: "
                f"{action.shape}"
            )

        if not np.all(
            np.isfinite(action)
        ):
            raise RuntimeError(
                "ACT returned non-finite action"
            )

        return (
            action,
            float(
                response[
                    "latency_ms"
                ]
            ),
        )

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


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

    teleop_root = xr_root / "teleop"
    televuer_src_root = (
        teleop_root
        / "televuer"
        / "src"
    )

    # r1a7_lowcmd_guard.py belongs to the companion
    # unitree_sdk2-main checkout in the same robot bundle.
    bundle_root = xr_root.parent.parent
    lowcmd_guard_tools_root = (
        bundle_root
        / "unitree_sdk2-main"
        / "tools"
    )

    if not (
        televuer_src_root
        / "televuer"
    ).is_dir():
        raise RuntimeError(
            "local TeleVuer src package not found: "
            f"{televuer_src_root}"
        )

    if not (
        lowcmd_guard_tools_root
        / "r1a7_lowcmd_guard.py"
    ).is_file():
        raise RuntimeError(
            "r1a7_lowcmd_guard.py not found: "
            f"{lowcmd_guard_tools_root}"
        )

    # ``teleop`` is imported from xr_root. TeleVuer uses a
    # src-layout package at teleop/televuer/src/televuer.
    for import_root in (
        sdk_python_root,
        xr_root,
        teleop_root,
        televuer_src_root,
        lowcmd_guard_tools_root,
    ):
        import_root_str = str(import_root)

        if import_root_str not in sys.path:
            sys.path.insert(
                0,
                import_root_str,
            )

    os.chdir(teleop_root)

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
    camera_manager = None
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

        if args.act_cameras:
            act_scripts_dir = (
                PROJECT_ROOT
                / "scripts"
            )

            act_scripts_str = str(
                act_scripts_dir
            )

            if act_scripts_str not in sys.path:
                sys.path.insert(
                    0,
                    act_scripts_str,
                )

            from act_three_camera_manager import (
                ActThreeCameraManager,
            )

            camera_manager = (
                ActThreeCameraManager(
                    args.act_camera_config
                )
            )

            camera_manager.start()

            print(
                "ACT three-camera manager started; "
                "pipelines will remain active between episodes.",
                flush=True,
            )

        if args.startup_abort_before_publisher:
            if camera_manager is not None:
                camera_manager.check_health()

            print(
                "STARTUP ABORT BEFORE LOWCMD PUBLISHER: PASS",
                flush=True,
            )
            print(
                "No lowcmd publisher was created and no command "
                "was sent.",
                flush=True,
            )
            return 0

        publisher = ChannelPublisher(args.command_topic, LowCmd_)
        publisher.Init()
        tv = TeleVuerWrapper(
            use_hand_tracking=False,
            binocular=False,
            img_shape=(480, 640),
            display_fps=args.ik_frequency,
            display_mode="pass-through",
            zmq=False,
            webrtc=False,
            arm_reference_mode="head_yaw",
        )
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
            f"Quest URL: https://{args.host_ip}:8012/?ws=wss://{args.host_ip}:8012",
            flush=True,
        )

        answer = input("Type ENABLE to arm this program (still no command is sent): ").strip()
        if answer.casefold() != "enable":
            print("Aborted before publishing.", flush=True)
            return 2


        first_solution = None

        if args.act_real:
            # =====================================================
            # ACT REAL MODE
            #
            # No Quest / TeleVuer / IK startup alignment.
            # Use current robot measured posture.
            # =====================================================

            state = state_buffer.snapshot()

            if state is None:
                raise RuntimeError(
                    "Robot lowstate unavailable during ACT startup"
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

            lowstate_age = (
                time.monotonic()
                - received_at
            )

            if lowstate_age > args.lowstate_timeout:
                raise RuntimeError(
                    "Robot lowstate stale during ACT startup: "
                    f"{lowstate_age:.3f}s"
                )

            first_solution = np.asarray(
                arm_q,
                dtype=float,
            ).reshape(14)

            print(
                "ACT REAL startup: "
                "Quest/VR bypassed; "
                "using current measured arm posture.",
                flush=True,
            )


        else:

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

        output.enable(upper_q, mode_machine)
        time.sleep(0.10)
        output.set_arm_goal(first_solution, mode_machine)
        if args.act_real:
            print(
                "R1-A7 lowcmd enabled; "
                "ACT real control started.",
                flush=True,
            )
        else:
            print(
                "R1-A7 lowcmd enabled; "
                "TeleVuer tracking started.",
                flush=True,
            )
        print("Type Q or press Ctrl+C to stop.", flush=True)

        period = 1.0 / args.ik_frequency
        next_tick = time.monotonic()
        session_lost_since: Optional[float] = None
        act_client = None
        act_latency_ms = float("nan")

        # ACT-real diagnostic recording only.
        # This does not change any ACT / LowCmd control behavior.
        act_debug_file = None
        act_debug_dir = None

        if args.act_real:
            if camera_manager is None:
                raise RuntimeError(
                    "--act-real requires --act-cameras"
                )

            act_client = ActRealtimeClient(
                args.act_socket,
                timeout_s=1.0,
            )

            act_client.ping()

            print(
                "ACT REALTIME CLIENT CONNECTED: "
                f"{args.act_socket}",
                flush=True,
            )

            debug_root = Path(
                "/data/R1A7/act_real_debug_v2"
            )

            debug_root.mkdir(
                parents=True,
                exist_ok=True,
            )

            act_debug_dir = (
                debug_root
                / (
                    "run_"
                    + time.strftime(
                        "%Y%m%d_%H%M%S"
                    )
                )
            )

            act_debug_dir.mkdir(
                parents=True,
                exist_ok=False,
            )

            # Reuse the existing asynchronous camera episode writer.
            # No image encoding or disk I/O is added to the ACT loop.
            camera_manager.check_health()

            camera_manager.start_episode(
                act_debug_dir
            )

            act_debug_file = open(
                act_debug_dir / "act_debug.csv",
                "w",
                encoding="utf-8",
                buffering=1,
            )

            act_debug_file.write(
                "monotonic_time,"
                "solve_count,"
                "right_wrist_host_time,"
                "pred_right_gripper_s0,"
                "pred_right_gripper_min,"
                "pred_right_gripper_max,"
                "measured_right_gripper_q,"
                "sent_right_gripper_q,"
                "goal_right_gripper_openness,"
                "right_contact_hold,"
                "act_latency_ms\n"
            )

            print(
                "ACT REAL DEBUG RECORDING START: "
                f"{act_debug_dir}",
                flush=True,
            )

        last_print = 0.0
        solve_count = 0

        # ACT temporal aggregation.
        #
        # This follows the official ACT evaluation logic:
        #   query_frequency = 1
        #   k = 0.01
        #   aggregate all historical chunk predictions
        #   aligned to the current timestep.
        #
        # The realtime inference server already returns
        # de-normalized actions. Since the ACT post-process
        # is affine and the aggregation weights sum to 1,
        # aggregating after de-normalization is equivalent.
        act_temporal_k = 0.01
        act_temporal_history = []

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
                if recorder.active:
                    # Stop robot-state recording first. Camera
                    # stop_episode() may wait for writer queues to
                    # flush, so states.csv must not remain open
                    # during that wait.
                    recorder.stop()

                    if camera_manager is not None:
                        camera_manager.stop_episode()

                    print(
                        "ACT SHARED EPISODE STOP COMPLETE",
                        flush=True,
                    )

                else:
                    episode_dir = (
                        recorder.next_episode_dir()
                    )

                    camera_episode_started = False

                    try:
                        if camera_manager is not None:
                            camera_manager.check_health()

                            camera_manager.start_episode(
                                episode_dir
                            )

                            camera_episode_started = True

                        recorder.start(
                            episode_dir
                        )

                    except Exception:
                        # If robot recorder startup fails after
                        # camera recording has opened, close that
                        # partial camera episode before propagating
                        # the error.
                        if (
                            camera_manager is not None
                            and camera_episode_started
                            and camera_manager.episode_active
                        ):
                            camera_manager.stop_episode()

                        raise

                    print(
                        "ACT SHARED EPISODE START: "
                        f"{episode_dir}",
                        flush=True,
                    )

            if camera_manager is not None:
                # A failed camera must never silently produce an
                # incomplete ACT demonstration.
                camera_manager.check_health()

            if not args.act_real:
                output.set_gripper_from_triggers(
                    float(
                        getattr(
                            tele,
                            "left_ctrl_triggerValue",
                            10.0,
                        )
                    ),
                    float(
                        getattr(
                            tele,
                            "right_ctrl_triggerValue",
                            10.0,
                        )
                    ),
                )

            output.update_gripper_contact(
                gripper_q,
                gripper_dq,
                now,
            )

            session_active = tele_session_active(tele)
            xr_age = tele_xr_age(tele, now)
            fresh = (
                bool(
                    getattr(
                        tele,
                        "motion_data_ready",
                        False,
                    )
                )
                and xr_age <= args.xr_stale_timeout
            )

            if args.act_real:
                # ACT observations come from robot lowstate and
                # the three physical cameras, not from XR poses.
                # Losing/stalling Quest must therefore not stop
                # ACT inference once ACT mode is running.
                session_lost_since = None
                input_valid = True

            else:
                if not session_active:
                    output.hold_measured(
                        arm_q,
                        mode_machine,
                    )

                    if session_lost_since is None:
                        session_lost_since = now

                    elif (
                        now - session_lost_since
                        >= args.session_close_timeout
                    ):
                        print(
                            "Quest page closed; "
                            "stopping lowcmd publisher.",
                            flush=True,
                        )
                        break
                else:
                    session_lost_since = None

                input_valid = (
                    session_active
                    and fresh
                )

            if input_valid and tracking_enabled:

                if args.act_real:
                    frames = (
                        camera_manager
                        .get_latest_bgr_frames()
                    )

                    act_qpos = build_act_qpos(
                        arm_q,
                        gripper_q,
                    )

                    action_chunk, act_latency_ms = (
                        act_client.infer(
                            act_qpos,
                            frames,
                        )
                    )

                    # ACT V1 canonical action order:
                    #
                    # left_arm_7,
                    # left_gripper,
                    # right_arm_7,
                    # right_gripper
                    #
                    # Replan every control iteration and use
                    # the first action from the current chunk.
                        # Diagnostic only: inspect the predicted right-gripper
                    # trajectory across the full ACT action chunk.
                    # Canonical action index 15 = right gripper openness:
                    # 1.0=open, 0.0=closed.
                    if solve_count % 15 == 0:
                        right_grip_chunk = np.asarray(
                            action_chunk[:, 15],
                            dtype=float,
                        )

                        probe_idx = (
                            0,
                            5,
                            10,
                            20,
                            40,
                            60,
                            80,
                            99,
                        )

                        probe_text = " ".join(
                            f"s{i}={right_grip_chunk[i]:.3f}"
                            for i in probe_idx
                            if i < right_grip_chunk.shape[0]
                        )

                        print(
                            "ACT_CHUNK_RIGHT_GRIP "
                            f"{probe_text} "
                            f"min={np.min(right_grip_chunk):.3f} "
                            f"max={np.max(right_grip_chunk):.3f}",
                            flush=True,
                        )

                    # -------------------------------------------------
                    # Official ACT-style temporal aggregation.
                    #
                    # A chunk queried at query_t predicts:
                    #
                    #   query_t,
                    #   query_t + 1,
                    #   ...
                    #
                    # For current time act_t, use:
                    #
                    #   chunk[act_t - query_t]
                    #
                    # from every still-valid historical chunk.
                    # -------------------------------------------------

                    act_t = solve_count

                    chunk_for_agg = np.asarray(
                        action_chunk,
                        dtype=float,
                    )

                    if (
                        chunk_for_agg.ndim != 2
                        or chunk_for_agg.shape[1] != 16
                    ):
                        raise RuntimeError(
                            "unexpected ACT chunk shape for "
                            f"temporal aggregation: "
                            f"{chunk_for_agg.shape}"
                        )

                    num_queries = int(
                        chunk_for_agg.shape[0]
                    )

                    act_temporal_history.append(
                        (
                            act_t,
                            chunk_for_agg.copy(),
                        )
                    )

                    # Older chunks cannot contribute after
                    # num_queries timesteps, so keep only the
                    # bounded valid history instead of allocating
                    # the official max_timesteps x
                    # (max_timesteps + num_queries) tensor.
                    if (
                        len(act_temporal_history)
                        > num_queries
                    ):
                        del act_temporal_history[
                            :(
                                len(act_temporal_history)
                                - num_queries
                            )
                        ]

                    actions_for_curr_step = []

                    for (
                        query_t,
                        query_chunk,
                    ) in act_temporal_history:

                        offset = (
                            act_t
                            - query_t
                        )

                        if (
                            0
                            <= offset
                            < query_chunk.shape[0]
                        ):
                            actions_for_curr_step.append(
                                query_chunk[offset]
                            )

                    if not actions_for_curr_step:
                        raise RuntimeError(
                            "ACT temporal aggregation "
                            "has no current-step actions"
                        )

                    actions_for_curr_step = np.stack(
                        actions_for_curr_step,
                        axis=0,
                    )

                    # Match official ACT:
                    #
                    # k = 0.01
                    # exp_weights =
                    # exp(-k * arange(N))
                    temporal_weights = np.exp(
                        -act_temporal_k
                        * np.arange(
                            actions_for_curr_step.shape[0],
                            dtype=float,
                        )
                    )

                    temporal_weights = (
                        temporal_weights
                        / temporal_weights.sum()
                    )

                    act_action = np.sum(
                        actions_for_curr_step
                        * temporal_weights[:, None],
                        axis=0,
                    )

                    if solve_count % 15 == 0:
                        print(
                            "ACT_TEMPORAL_AGG "
                            f"t={act_t} "
                            f"n={actions_for_curr_step.shape[0]} "
                            f"raw_s0_right="
                            f"{chunk_for_agg[0,15]:.3f} "
                            f"agg_right="
                            f"{act_action[15]:.3f} "
                            f"k={act_temporal_k:.3f}",
                            flush=True,
                        )

                    act_arm_goal = np.concatenate(
                        [
                            act_action[0:7],
                            act_action[8:15],
                        ],
                        axis=0,
                    )

                    act_gripper_goal = np.asarray(
                        [
                            act_action[7],
                            act_action[15],
                        ],
                        dtype=float,
                    )

                    output.set_arm_goal(
                        act_arm_goal,
                        mode_machine,
                    )

                    output.set_gripper_openness_goal(
                        act_gripper_goal
                    )

                    if act_debug_file is not None:
                        right_grip_chunk_debug = np.asarray(
                            action_chunk[:, 15],
                            dtype=float,
                        )

                        right_wrist_host_time = float(
                            frames[
                                "right_wrist"
                            ][
                                "host_time"
                            ]
                        )

                        with output.lock:
                            sent_right_gripper_q_debug = float(
                                output.gripper_target[1]
                            )

                            right_contact_hold_debug = int(
                                output.gripper_hold[1]
                            )

                        act_debug_file.write(
                            f"{time.monotonic():.9f},"
                            f"{solve_count},"
                            f"{right_wrist_host_time:.9f},"
                            f"{right_grip_chunk_debug[0]:.6f},"
                            f"{np.min(right_grip_chunk_debug):.6f},"
                            f"{np.max(right_grip_chunk_debug):.6f},"
                            f"{float(gripper_q[1]):.6f},"
                            f"{sent_right_gripper_q_debug:.6f},"
                            f"{float(act_gripper_goal[1]):.6f},"
                            f"{right_contact_hold_debug},"
                            f"{act_latency_ms:.3f}\n"
                        )

                    solve_count += 1

                else:
                    left_target, right_target = (
                        ik.map_wrist_targets(
                            tele.left_wrist_pose,
                            tele.right_wrist_pose,
                            arm_q,
                            rotation_mode=
                                args.controller_rotation_mode,
                        )
                    )

                    solution, _ = ik.solve_ik(
                        left_target,
                        right_target,
                        arm_q,
                        arm_dq,
                    )

                    output.set_arm_goal(
                        solution,
                        mode_machine,
                    )

                    solve_count += 1

            readable, _writable, _errors = select.select([sys.stdin], [], [], 0.0)
            if readable and sys.stdin.readline().strip().casefold() == "q":
                break

            with output.lock:
                sent_arm_q_for_record = output.upper_target[1:15].copy()
                goal_arm_q_for_record = output.arm_goal.copy()
                sent_gripper_q_for_record = output.gripper_target.copy()
                goal_gripper_q_for_record = output.gripper_goal.copy()
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
            )

            if now - last_print >= args.print_period:
                with output.lock:
                    sent_arm_q = output.upper_target[1:15].copy()
                    goal_arm_q = output.arm_goal.copy()
                    gripper_q = output.gripper_target.copy()
                    gripper_hold = output.gripper_hold.copy()
                print(
                    f"mode={'ACT' if args.act_real else 'VR'} "
                    f"tracking={int(tracking_enabled if args.act_real else (session_active and fresh and tracking_enabled))} "
                    f"enabled={int(tracking_enabled)} solves={solve_count} "
                    f"act_latency={act_latency_ms:.1f}ms "
                    f"xr_age={xr_age:.3f}s goal_state_error="
                    f"{np.max(np.abs(goal_arm_q - arm_q)):.3f}rad "
                    f"sent_state_error={np.max(np.abs(sent_arm_q - arm_q)):.3f}rad "
                    f"published={output.publish_count} "
                    f"gripper=[{gripper_q[0]:.3f},{gripper_q[1]:.3f}] "
                    f"contact_hold=[{int(gripper_hold[0])},{int(gripper_hold[1])}]",
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

        # Cleanup must continue even if one subsystem fails.
        # In particular, a camera/USB error must never prevent
        # TeleVuer or the lowcmd guard from being released.
        cleanup_errors = []

        if output is not None:
            try:
                output.close()
                print(
                    "R1-A7 lowcmd publisher stopped.",
                    flush=True,
                )
            except Exception as exc:
                cleanup_errors.append(
                    f"output.close: {type(exc).__name__}: {exc}"
                )

        debug_file_for_cleanup = locals().get(
            "act_debug_file"
        )

        if debug_file_for_cleanup is not None:
            try:
                debug_file_for_cleanup.close()

                print(
                    "ACT REAL DEBUG CSV CLOSED",
                    flush=True,
                )

            except Exception as exc:
                cleanup_errors.append(
                    "act_debug_file.close: "
                    f"{type(exc).__name__}: {exc}"
                )

        try:
            recorder.stop()
        except Exception as exc:
            cleanup_errors.append(
                f"recorder.stop: {type(exc).__name__}: {exc}"
            )

        if camera_manager is not None:
            try:
                if camera_manager.episode_active:
                    camera_manager.stop_episode()
            except Exception as exc:
                cleanup_errors.append(
                    "camera_manager.stop_episode: "
                    f"{type(exc).__name__}: {exc}"
                )

            try:
                camera_manager.close()
            except Exception as exc:
                cleanup_errors.append(
                    f"camera_manager.close: "
                    f"{type(exc).__name__}: {exc}"
                )

        if tv is not None:
            try:
                tv.close()
            except Exception as exc:
                cleanup_errors.append(
                    f"tv.close: {type(exc).__name__}: {exc}"
                )

        try:
            guard.close()
        except Exception as exc:
            cleanup_errors.append(
                f"guard.close: {type(exc).__name__}: {exc}"
            )

        for cleanup_error in cleanup_errors:
            print(
                f"CLEANUP WARNING: {cleanup_error}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
