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
import sys
import threading
import time
from typing import Optional

import numpy as np


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
        default=ROOT / "datasets" / "r1a7_vr_button_records",
    )
    parser.add_argument("--record-episode-id", default="r1a7_vr_001")
    parser.add_argument("--enable-gripper", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gripper-kp", type=float, default=8.0)
    parser.add_argument("--gripper-kd", type=float, default=1.5)
    parser.add_argument("--gripper-speed", type=float, default=1.5)
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
    def __init__(self, root: Path, episode_id: str):
        self.root = root.expanduser().resolve()
        self.episode_id = episode_id
        self.active = False
        self.episode_dir: Optional[Path] = None
        self.file = None
        self.writer: Optional[csv.writer] = None
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
            "left_wrist_pose", "right_wrist_pose",
            "left_trigger", "right_trigger", "right_A", "left_X",
        ])
        self.active = True
        self.samples = 0
        self.started_at = time.monotonic()
        print(f"X RECORD START: {self.episode_dir}", flush=True)

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
            json.dumps(np.asarray(getattr(tele, "left_wrist_pose", np.eye(4)), dtype=float).reshape(4, 4).tolist()),
            json.dumps(np.asarray(getattr(tele, "right_wrist_pose", np.eye(4)), dtype=float).reshape(4, 4).tolist()),
            float(getattr(tele, "left_ctrl_triggerValue", 10.0)),
            float(getattr(tele, "right_ctrl_triggerValue", 10.0)),
            int(bool(getattr(tele, "right_ctrl_aButton", False))),
            int(bool(getattr(tele, "left_ctrl_aButton", False))),
        ])
        self.samples += 1

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

    def snapshot(self) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, int, float]]:
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
            return (
                upper_q,
                arm_q,
                arm_dq,
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
            self.gripper_goal = goal.copy()

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
    recorder = SimpleCsvRecorder(args.record_root, args.record_episode_id)
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
        upper_q, arm_q, _arm_dq, _mode_machine, _received_at = state
        print("Measured R1-A7 arm q:", np.round(arm_q, 4).tolist(), flush=True)

        check_debug_mode(MotionSwitcherClient)
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
        )

        print("WARNING: real R1-A7 debug-mode control on rt/lowcmd.", flush=True)
        if args.enable_gripper:
            print(
                "Internal DEX1 gripper enabled on LowCmd motors 31/33: "
                "trigger released=open, trigger pulled=close.",
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
            upper_q, arm_q, arm_dq, mode_machine, received_at = state
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
            _upper_q, arm_q, arm_dq, mode_machine, received_at = state
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
            recorder.write(
                tele,
                tracking_enabled,
                input_valid,
                xr_age,
                arm_q,
                arm_dq,
                goal_arm_q_for_record,
                sent_arm_q_for_record,
            )

            if now - last_print >= args.print_period:
                with output.lock:
                    sent_arm_q = output.upper_target[1:15].copy()
                    goal_arm_q = output.arm_goal.copy()
                    gripper_q = output.gripper_target.copy()
                print(
                    f"tracking={int(session_active and fresh and tracking_enabled)} "
                    f"enabled={int(tracking_enabled)} solves={solve_count} "
                    f"xr_age={xr_age:.3f}s goal_state_error="
                    f"{np.max(np.abs(goal_arm_q - arm_q)):.3f}rad "
                    f"sent_state_error={np.max(np.abs(sent_arm_q - arm_q)):.3f}rad "
                    f"published={output.publish_count} "
                    f"gripper=[{gripper_q[0]:.3f},{gripper_q[1]:.3f}]",
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
