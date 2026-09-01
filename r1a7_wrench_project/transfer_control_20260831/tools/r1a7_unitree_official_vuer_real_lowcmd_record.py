#!/usr/bin/env python3
"""Recording copy of the verified R1-A7 Unitree debug-mode teleoperation path.

The input and IK path matches the verified MuJoCo program:
TeleVuer controller poses -> R1A7_ArmIK -> R1-A7 arm joint targets.
Only the real-robot executor differs: this program publishes CRC-protected
LowCmd messages on rt/lowcmd, as Unitree's default (non-motion) XR path does.
"""

from __future__ import annotations

import argparse
import fcntl
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
import pinocchio as pin

from r1a7_unitree_episode_recorder import R1A7UnitreeEpisodeRecorder


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XR_ROOT = Path("/home/version/robot_dev/xr_teleoperate")
DEFAULT_SDK_PYTHON = Path("/home/version/robot_dev/unitree_sdk2_python")

# Confirmed from the R1-A7 low-level examples and live lowstate diagnostics.
ARM_INDICES = tuple(range(15, 29))
UPPER_BODY_INDICES = (13, *range(15, 31))

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
    parser.add_argument(
        "--motion-scale",
        type=float,
        default=1.0,
        help=(
            "Scale target end-effector translation around the current real arm "
            "pose. 1.0 preserves the verified mapping; smaller values reduce "
            "teleoperation range."
        ),
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
        "--log-file",
        type=Path,
        default=Path("/tmp/r1a7_official_real_lowcmd_record.log"),
    )
    parser.add_argument(
        "--record-task-dir",
        type=Path,
        default=(
            ROOT / "datasets/dataset_setup_001/unitree_xr_data/"
            "r1a7_charge_connector_coarse_approach"
        ),
    )
    parser.add_argument(
        "--camera-binary",
        type=Path,
        default=ROOT / "tools/bin/dataset_setup_001_record_camera_episode_sync",
    )
    parser.add_argument(
        "--record-duration", type=float, default=3600.0,
        help="safety maximum episode duration; close the Quest page to stop earlier",
    )
    parser.add_argument(
        "--record-stale-stop-timeout",
        type=float,
        default=0.0,
        help=(
            "optional: while recording, stop and save if Quest controller data "
            "stays inactive/stale for this many seconds; <=0 disables it"
        ),
    )
    parser.add_argument("--camera-warmup", type=int, default=3)
    parser.add_argument("--record-frequency", type=float, default=15.0)
    parser.add_argument("--enable-dex1", action="store_true")
    parser.add_argument("--dex1-state-timeout", type=float, default=0.25)
    parser.add_argument("--dex1-ready-timeout", type=float, default=5.0)
    parser.add_argument("--dex1-publish-frequency", type=float, default=100.0)
    parser.add_argument("--dex1-close-q", type=float, default=0.5)
    parser.add_argument("--dex1-open-q", type=float, default=5.4)
    parser.add_argument("--dex1-close-speed", type=float, default=0.30)
    parser.add_argument("--dex1-open-speed", type=float, default=0.45)
    parser.add_argument("--dex1-kp", type=float, default=3.0)
    parser.add_argument("--dex1-kd", type=float, default=0.08)
    parser.add_argument("--dex1-stall-error", type=float, default=0.08)
    parser.add_argument("--dex1-stall-time", type=float, default=0.25)
    parser.add_argument("--dex1-torque-limit", type=float, default=0.45)
    parser.add_argument("--dex1-torque-time", type=float, default=0.12)
    parser.add_argument("--dex1-contact-preload", type=float, default=0.02)
    parser.add_argument(
        "--auto-enable",
        action="store_true",
        help=(
            "Skip the terminal ENABLE prompt. Intended only for the local GUI "
            "launcher; real motion still waits for Quest A/X."
        ),
    )
    parser.add_argument(
        "--xr-root",
        type=Path,
        default=Path(os.environ.get("XR_TELEOP_ROOT", DEFAULT_XR_ROOT)),
    )
    parser.add_argument("--sdk-python-root", type=Path, default=DEFAULT_SDK_PYTHON)
    args = parser.parse_args()
    for name in (
        "ik_frequency", "publish_frequency", "max_joint_speed", "motion_scale",
        "record_duration", "record_frequency", "dex1_state_timeout",
        "dex1_ready_timeout", "dex1_publish_frequency", "dex1_close_speed",
        "dex1_open_speed", "dex1_kp", "dex1_stall_error",
        "dex1_stall_time", "dex1_torque_limit", "dex1_torque_time",
    ):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    if args.camera_warmup < 0:
        parser.error("--camera-warmup must be non-negative")
    if args.dex1_kd < 0.0 or args.dex1_contact_preload < 0.0:
        parser.error("DEX1 kd and contact preload must be non-negative")
    if not args.dex1_close_q < args.dex1_open_q:
        parser.error("--dex1-close-q must be less than --dex1-open-q")
    if args.command_topic != "rt/lowcmd":
        parser.error("this debug-mode executor only permits --command-topic rt/lowcmd")
    return args


def apply_motion_scale(ik, left_target, right_target, arm_q: np.ndarray, scale: float):
    scale = float(scale)
    if abs(scale - 1.0) < 1e-9:
        return left_target, right_target
    if scale <= 0.0:
        raise ValueError("--motion-scale must be greater than zero")
    pin.framesForwardKinematics(ik.reduced_robot.model, ik.reduced_robot.data, arm_q)
    left_current = ik.reduced_robot.data.oMf[ik.L_hand_id].translation.copy()
    right_current = ik.reduced_robot.data.oMf[ik.R_hand_id].translation.copy()
    scaled_left = np.asarray(left_target, dtype=float).copy()
    scaled_right = np.asarray(right_target, dtype=float).copy()
    scaled_left[:3, 3] = left_current + scale * (scaled_left[:3, 3] - left_current)
    scaled_right[:3, 3] = right_current + scale * (scaled_right[:3, 3] - right_current)
    return scaled_left, scaled_right


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
    def __init__(self, crc, sample_observer=None):
        self._crc = crc
        self._sample_observer = sample_observer
        self._lock = threading.Lock()
        self._message = None
        self._received_at = 0.0
        self.received_count = 0
        self.count = 0
        self.crc_errors = 0

    def callback(self, message) -> None:
        host_ns = time.monotonic_ns()
        self.received_count += 1
        try:
            if int(message.crc) != int(self._crc.Crc(message)):
                self.crc_errors += 1
                return
        except Exception:
            self.crc_errors += 1
            return
        arm_q = np.asarray(
            [message.motor_state[index].q for index in ARM_INDICES],
            dtype=float,
        )
        arm_dq = np.asarray(
            [message.motor_state[index].dq for index in ARM_INDICES],
            dtype=float,
        )
        arm_tau = np.asarray(
            [message.motor_state[index].tau_est for index in ARM_INDICES],
            dtype=float,
        )
        mode_machine = int(message.mode_machine)
        with self._lock:
            self._message = message
            self._received_at = time.monotonic()
            self.count += 1
        if self._sample_observer is not None:
            try:
                self._sample_observer(
                    host_ns, mode_machine, arm_q, arm_dq, arm_tau
                )
            except Exception:
                pass

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
        command_observer=None,
    ):
        self.publisher = publisher
        self.low_cmd_factory = low_cmd_factory
        self.crc = crc
        self.period = 1.0 / publish_frequency
        self.max_step = max_joint_speed * self.period
        self.command_observer = command_observer
        self.lock = threading.Lock()
        self.upper_target = np.zeros(len(UPPER_BODY_INDICES), dtype=float)
        self.arm_goal = np.zeros(len(ARM_INDICES), dtype=float)
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

    def hold_measured(self, arm_q: np.ndarray, mode_machine: int) -> None:
        self.set_arm_goal(arm_q, mode_machine)

    def _build_message(self):
        message = self.low_cmd_factory()
        with self.lock:
            delta = np.clip(
                self.arm_goal - self.upper_target[1:15],
                -self.max_step,
                self.max_step,
            )
            self.upper_target[1:15] += delta
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
        message.crc = self.crc.Crc(message)
        return message, upper_target[1:15], mode_machine

    def _loop(self) -> None:
        next_tick = time.monotonic()
        while not self.stop.is_set():
            try:
                message, sent_arm_q, mode_machine = self._build_message()
                self.publisher.Write(message)
                self.publish_count += 1
                if self.command_observer is not None:
                    try:
                        self.command_observer(
                            time.monotonic_ns(), self.publish_count,
                            mode_machine, sent_arm_q,
                        )
                    except Exception:
                        pass
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
    record_task_dir = args.record_task_dir.expanduser().resolve()
    camera_binary = args.camera_binary.expanduser().resolve()
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
    from teleop.utils.episode_writer import EpisodeWriter
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
    from r1a7_dex1_trigger_controller import R1A7Dex1TriggerController
    from r1a7_lowcmd_guard import acquire_lowcmd_guard, ensure_tcp_port_available

    ensure_tcp_port_available(8012)
    guard = acquire_lowcmd_guard(Path(__file__).name, topic=args.command_topic)
    tv = None
    output = None
    gripper = None
    recorder = None
    stop = threading.Event()

    def request_stop(_signum=None, _frame=None) -> None:
        stop.set()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        recorder = R1A7UnitreeEpisodeRecorder(
            EpisodeWriter,
            task_dir=record_task_dir,
            camera_binary=camera_binary,
            duration_s=args.record_duration,
            camera_warmup_s=args.camera_warmup,
            official_frequency_hz=args.record_frequency,
            include_gripper=args.enable_dex1,
        )
        ChannelFactoryInitialize(args.domain_id, args.interface)
        crc = CRC()
        state_buffer = StateBuffer(crc, sample_observer=recorder.record_lowstate)
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

        if args.enable_dex1:
            gripper = R1A7Dex1TriggerController(
                publish_frequency=args.dex1_publish_frequency,
                close_q=args.dex1_close_q,
                open_q=args.dex1_open_q,
                close_speed=args.dex1_close_speed,
                open_speed=args.dex1_open_speed,
                kp=args.dex1_kp,
                kd=args.dex1_kd,
                stall_error=args.dex1_stall_error,
                stall_time=args.dex1_stall_time,
                torque_limit=args.dex1_torque_limit,
                torque_time=args.dex1_torque_time,
                contact_preload=args.dex1_contact_preload,
                state_timeout=args.dex1_state_timeout,
            )
            print(
                "Waiting for rt/dex1/left/state and rt/dex1/right/state...",
                flush=True,
            )
            if not gripper.wait_ready(args.dex1_ready_timeout):
                print(
                    "WARNING: DEX1 state topics are unavailable; gripper trigger "
                    "control will be disabled for this run.",
                    flush=True,
                )
                print(
                    "VR teleoperation and recording will continue waiting for "
                    "the Quest page. Start dex1_1_gripper_server before launching "
                    "again if gripper triggers are needed.",
                    flush=True,
                )
                try:
                    gripper.close()
                except Exception:
                    pass
                gripper = None
                recorder.include_gripper = False
            else:
                gripper_state = gripper.snapshot()
                print(
                    "DEX1 feedback ready: "
                    f"left_q={gripper_state['left']['q']:.3f} "
                    f"right_q={gripper_state['right']['q']:.3f}",
                    flush=True,
                )

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
            command_observer=recorder.record_lowcmd,
        )

        print("WARNING: real R1-A7 debug-mode control on rt/lowcmd.", flush=True)
        print("MuJoCo and rt/arm_sdk are not used.", flush=True)
        if args.enable_dex1:
            print(
                "DEX1 trigger control enabled: release opens slowly; pull closes "
                "slowly; contact feedback freezes further closing.",
                flush=True,
            )
        print("Keep the emergency stop ready and clear both arm workspaces.", flush=True)
        print(
            f"Quest URL: https://{args.host_ip}:8012/?ws=wss://{args.host_ip}:8012",
            flush=True,
        )

        if args.auto_enable:
            print(
                "AUTO_ENABLE is set: terminal ENABLE prompt skipped. "
                "No command is sent until Quest A/X starts tracking.",
                flush=True,
            )
        else:
            answer = input("Type ENABLE to arm this program (still no command is sent): ").strip()
            if answer.casefold() != "enable":
                print("Aborted before publishing.", flush=True)
                return 2

        first_solution = None
        last_start_button_pressed = False
        last_wait_print = 0.0
        print(
            "Enter VR, align controllers with the real arms, release then press "
            "right A or left X. Terminal R also starts; Q quits.",
            flush=True,
        )
        while not stop.is_set():
            now = time.monotonic()
            tele = tv.get_tele_data()
            right_a = bool(getattr(tele, "right_ctrl_aButton", False))
            left_x = bool(getattr(tele, "left_ctrl_aButton", False))
            start_button_pressed = right_a or left_x
            a_rising = start_button_pressed and not last_start_button_pressed
            last_start_button_pressed = start_button_pressed
            terminal_start = False
            readable, _writable, _errors = select.select([sys.stdin], [], [], 0.0)
            if readable:
                key = sys.stdin.readline().strip().casefold()
                if key == "q":
                    return 0
                terminal_start = key == "r"
            if not a_rising and not terminal_start:
                if now - last_wait_print >= args.print_period:
                    event_time = float(getattr(tele, "motion_event_time", 0.0))
                    xr_age = now - event_time if event_time > 0.0 else float("inf")
                    print(
                        "waiting_start "
                        f"session={int(bool(getattr(tele, 'vuer_session_active', False)))} "
                        f"motion_ready={int(bool(getattr(tele, 'motion_data_ready', False)))} "
                        f"right_A={int(right_a)} left_X={int(left_x)} "
                        f"xr_age={xr_age:.3f}s",
                        flush=True,
                    )
                    last_wait_print = now
                stop.wait(0.02)
                continue

            event_time = float(getattr(tele, "motion_event_time", 0.0))
            xr_age = time.monotonic() - event_time if event_time > 0.0 else float("inf")
            if not bool(getattr(tele, "vuer_session_active", False)):
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
            left_target, right_target = apply_motion_scale(
                ik, left_target, right_target, arm_q, args.motion_scale
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

        if gripper is not None:
            gripper.enable()
        output.enable(upper_q, mode_machine)
        time.sleep(0.10)
        output.set_arm_goal(first_solution, mode_machine)
        print("R1-A7 lowcmd enabled; TeleVuer tracking started.", flush=True)
        if recorder.status in ("idle", "complete"):
            print("A/X start accepted: starting synchronized episode recording.", flush=True)
            recorder.request_start()
        else:
            print(
                f"A/X start accepted, but recorder was not idle "
                f"(status={recorder.status}).",
                flush=True,
            )
        print(
            "Recording starts with the same A/X press that starts teleoperation. "
            "Close the Quest page to stop and save. "
            f"Safety maximum duration is {args.record_duration:.0f} seconds. "
            "Type Q or press Ctrl+C to stop.",
            flush=True,
        )

        period = 1.0 / args.ik_frequency
        next_tick = time.monotonic()
        session_lost_since: Optional[float] = None
        record_stale_since: Optional[float] = None
        last_print = 0.0
        solve_count = 0
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
            session_active = bool(getattr(tele, "vuer_session_active", False))
            if not session_active:
                output.hold_measured(arm_q, mode_machine)
                if session_lost_since is None:
                    session_lost_since = now
                elif now - session_lost_since >= args.session_close_timeout:
                    if recorder.status == "recording":
                        print("Quest page closed: stopping and saving recording.", flush=True)
                        recorder.request_stop()
                    print("Quest page closed; stopping lowcmd publisher.", flush=True)
                    break
            else:
                session_lost_since = None

            event_time = float(getattr(tele, "motion_event_time", 0.0))
            xr_age = now - event_time if event_time > 0.0 else float("inf")
            fresh = bool(getattr(tele, "motion_data_ready", False)) and xr_age <= args.xr_stale_timeout
            input_valid = session_active and fresh
            if recorder.status == "recording" and args.record_stale_stop_timeout > 0.0:
                if input_valid:
                    record_stale_since = None
                else:
                    if record_stale_since is None:
                        record_stale_since = now
                    elif now - record_stale_since >= args.record_stale_stop_timeout:
                        print(
                            "Quest tracking became stale while recording: "
                            "stopping and saving recording.",
                            flush=True,
                        )
                        recorder.request_stop()
                        print("Stopping lowcmd publisher after recording stop.", flush=True)
                        break
            if gripper is not None:
                gripper.update_inputs(
                    float(getattr(tele, "left_ctrl_triggerValue", 10.0)),
                    float(getattr(tele, "right_ctrl_triggerValue", 10.0)),
                    session_active,
                )
                if gripper.error:
                    raise RuntimeError(f"DEX1 controller stopped: {gripper.error}")
            left_target = None
            right_target = None
            solution = None
            ik_solve_ms = 0.0
            if input_valid:
                left_target, right_target = ik.map_wrist_targets(
                    tele.left_wrist_pose,
                    tele.right_wrist_pose,
                    arm_q,
                    rotation_mode=args.controller_rotation_mode,
                )
                left_target, right_target = apply_motion_scale(
                    ik, left_target, right_target, arm_q, args.motion_scale
                )
                solve_started_ns = time.perf_counter_ns()
                solution, _ = ik.solve_ik(left_target, right_target, arm_q, arm_dq)
                ik_solve_ms = (time.perf_counter_ns() - solve_started_ns) / 1e6
                output.set_arm_goal(solution, mode_machine)
                solve_count += 1

            readable, _writable, _errors = select.select([sys.stdin], [], [], 0.0)
            if readable:
                key = sys.stdin.readline().strip().casefold()
                if key == "q":
                    if recorder.status == "recording":
                        print("Terminal Q: stopping and saving recording.", flush=True)
                        recorder.request_stop()
                    break

            with output.lock:
                sent_arm_q = output.upper_target[1:15].copy()
                goal_arm_q = output.arm_goal.copy()
            gripper_snapshot = gripper.snapshot() if gripper is not None else None
            try:
                recorder.record_control(
                    time.monotonic_ns(), tele, input_valid, xr_age,
                    left_target, right_target, arm_q, arm_dq, solution,
                    goal_arm_q, sent_arm_q, ik_solve_ms, gripper_snapshot,
                )
                recorder.poll()
            except Exception as exc:
                print(
                    f"WARNING: recorder hook failed without changing control: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

            if now - last_print >= args.print_period:
                gripper_text = ""
                if gripper_snapshot is not None:
                    gripper_text = (
                        f" dex1=L({gripper_snapshot['left']['q']:.2f}->"
                        f"{gripper_snapshot['left']['target_q']:.2f},"
                        f"contact={int(gripper_snapshot['left']['contact'])}) "
                        f"R({gripper_snapshot['right']['q']:.2f}->"
                        f"{gripper_snapshot['right']['target_q']:.2f},"
                        f"contact={int(gripper_snapshot['right']['contact'])})"
                    )
                print(
                    f"tracking={int(session_active and fresh)} solves={solve_count} "
                    f"xr_age={xr_age:.3f}s goal_state_error="
                    f"{np.max(np.abs(goal_arm_q - arm_q)):.3f}rad "
                    f"sent_state_error={np.max(np.abs(sent_arm_q - arm_q)):.3f}rad "
                    f"published={output.publish_count} record={recorder.status}"
                    f"{gripper_text}",
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
        if gripper is not None:
            gripper.close()
            print("DEX1 command publishers stopped.", flush=True)
        if output is not None:
            output.close()
            print("R1-A7 lowcmd publisher stopped.", flush=True)
        if recorder is not None:
            recorder.close()
        if tv is not None:
            tv.close()
        guard.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
