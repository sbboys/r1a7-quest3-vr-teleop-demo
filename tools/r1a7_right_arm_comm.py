#!/usr/bin/env python3
"""R1-A 7-DoF right-arm DDS communication smoke test.

Default mode is read-only: subscribe to ``rt/lowstate`` and print the right arm.
Use ``--mode hold --enable_control`` only after confirming the printed joint
indices match the real robot.
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as GoLowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as GoLowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as HGLowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as HGLowState_
from unitree_sdk2py.utils.crc import CRC


RIGHT_ARM_7_NAMES = [
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]

R1_ARM_SDK_INDICES = [15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 13, 29, 30]
R1_A7_UPPER_BODY_INDICES = [13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]


@dataclass
class ArmState:
    q: List[float]
    dq: List[float]
    stamp: float


class R1A7RightArmDDS:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.crc = CRC()
        self.low_cmd = unitree_go_msg_dds__LowCmd_() if args.idl == "go" else unitree_hg_msg_dds__LowCmd_()
        self.low_state: Optional[LowState_] = None
        self.first_state_time: Optional[float] = None
        self.last_state_time: Optional[float] = None
        self.done = False

        self.right_arm_indices = self._parse_indices(args.right_arm_indices)
        if len(self.right_arm_indices) != 7:
            raise ValueError("--right_arm_indices must contain exactly 7 comma-separated motor indices")
        self.r1_arm_sdk_indices = self._parse_indices(args.r1_arm_sdk_indices)
        self.lowcmd_hold_indices = self._parse_indices(args.lowcmd_hold_indices)
        self.weight_index = int(args.weight_index)
        self.current_command: Optional[List[float]] = None
        self.hold_start: Optional[List[float]] = None
        self.start_time = time.monotonic()
        self.no_motion_start: Optional[float] = None
        self.no_motion_baseline_q: Optional[List[float]] = None
        self.last_no_motion_warning = 0.0

        self.publisher = None
        self.subscriber = None

    @staticmethod
    def _parse_indices(text: str) -> List[int]:
        return [int(part.strip()) for part in text.split(",") if part.strip()]

    def init_dds(self) -> None:
        if self.args.interface:
            ChannelFactoryInitialize(self.args.domain_id, self.args.interface)
        else:
            ChannelFactoryInitialize(self.args.domain_id)

        if self.args.enter_debug_mode:
            self._enter_debug_mode()

        low_state_type = GoLowState_ if self.args.idl == "go" else HGLowState_
        low_cmd_type = GoLowCmd_ if self.args.idl == "go" else HGLowCmd_

        self.subscriber = ChannelSubscriber(self.args.state_topic, low_state_type)
        self.subscriber.Init(self._lowstate_handler, 10)

        if self.args.enable_control:
            self.publisher = ChannelPublisher(self.args.command_topic, low_cmd_type)
            self.publisher.Init()

        print("[R1-A7 RIGHT ARM] DDS initialized")
        print("[R1-A7 RIGHT ARM] DDS domain id:", self.args.domain_id)
        print("[R1-A7 RIGHT ARM] IDL:", self.args.idl)
        print("[R1-A7 RIGHT ARM] lowstate subscriber:", self.args.state_topic)
        print(
            "[R1-A7 RIGHT ARM] arm_sdk publisher:",
            self.args.command_topic if self.args.enable_control else "disabled (read-only)",
        )
        print("[R1-A7 RIGHT ARM] right arm indices:", self.right_arm_indices)
        if self.args.r1_arm_sdk:
            print("[R1-A7 RIGHT ARM] R1 arm_sdk mode: mode_pr weight, joints:", self.r1_arm_sdk_indices)
        if self.args.debug_lowcmd:
            print("[R1-A7 RIGHT ARM] lowcmd hold indices:", self.lowcmd_hold_indices)
        print("[R1-A7 RIGHT ARM] weight index:", self.weight_index)
        print("[R1-A7 RIGHT ARM] debug lowcmd:", self.args.debug_lowcmd)
        print("[R1-A7 RIGHT ARM] mode:", self.args.mode)

    def _enter_debug_mode(self) -> None:
        msc = MotionSwitcherClient()
        msc.SetTimeout(2.0)
        msc.Init()
        status, result = msc.CheckMode()
        print(f"[R1-A7 RIGHT ARM] motion_switcher CheckMode: status={status} result={result}")
        try:
            while result and result.get("name"):
                print("[R1-A7 RIGHT ARM] releasing active mode:", result)
                msc.ReleaseMode()
                time.sleep(0.5)
                status, result = msc.CheckMode()
                print(f"[R1-A7 RIGHT ARM] motion_switcher CheckMode: status={status} result={result}")
        except Exception as exc:
            print(f"[R1-A7 RIGHT ARM] failed to enter debug mode: {exc}")

    def _lowstate_handler(self, msg: LowState_) -> None:
        self.low_state = msg
        now = time.monotonic()
        if self.first_state_time is None:
            self.first_state_time = now
        self.last_state_time = now

    def _read_right_arm(self) -> Optional[ArmState]:
        if self.low_state is None:
            return None
        motor_state = self.low_state.motor_state
        max_idx = max(max(self.right_arm_indices), self.weight_index)
        if len(motor_state) <= max_idx:
            raise RuntimeError(
                f"lowstate has {len(motor_state)} motors, but requested index {max_idx}. "
                "Check --right_arm_indices and --weight_index."
            )
        q = [float(motor_state[i].q) for i in self.right_arm_indices]
        dq = [float(motor_state[i].dq) for i in self.right_arm_indices]
        return ArmState(q=q, dq=dq, stamp=time.monotonic())

    def _init_low_cmd_stop(self) -> None:
        # For the high-level arm_sdk topic, only the selected arm joints and the
        # weight slot are actively used. Leave all other entries at zero.
        for motor in self.low_cmd.motor_cmd:
            motor.tau = 0.0
            motor.q = 0.0
            motor.dq = 0.0
            motor.kp = 0.0
            motor.kd = 0.0

    def _target_for_mode(self, state: ArmState, elapsed: float) -> List[float]:
        if self.current_command is None:
            self.current_command = list(state.q)
        if self.hold_start is None:
            self.hold_start = list(state.q)

        if self.args.mode == "hold":
            return list(self.hold_start)

        if self.args.mode == "test":
            target = list(self.hold_start)
            amp = math.radians(self.args.test_amplitude_deg)
            freq = max(0.01, self.args.test_frequency_hz)
            joint_idx = max(0, min(6, self.args.test_joint_index - 1))
            # Small single-joint movement only. Other joints hold.
            target[joint_idx] = self.hold_start[joint_idx] + amp * math.sin(2.0 * math.pi * freq * elapsed)
            return target

        if self.args.mode == "lift":
            target = list(self.hold_start)
            offset = math.radians(self.args.lift_offset_deg)
            ramp = min(1.0, max(0.0, elapsed / max(0.1, self.args.lift_ramp_s)))
            target[0] = self.hold_start[0] + offset * ramp
            return target

        return list(state.q)

    def _rate_limit(self, target: List[float], dt: float) -> List[float]:
        assert self.current_command is not None
        max_delta = max(0.0, self.args.max_speed_rad_s) * max(dt, 1e-3)
        out = []
        for cur, des in zip(self.current_command, target):
            delta = max(-max_delta, min(max_delta, des - cur))
            out.append(cur + delta)
        self.current_command = out
        return out

    def _compute_command(self, state: ArmState, elapsed: float, dt: float) -> tuple[List[float], List[float]]:
        target = self._target_for_mode(state, elapsed)
        command = self._rate_limit(target, dt)
        return target, command

    def _publish_arm_command(self, state: ArmState, command: List[float]) -> None:
        if self.publisher is None:
            return
        self._init_low_cmd_stop()

        if self.args.debug_lowcmd and self.low_state is not None:
            if hasattr(self.low_cmd, "mode_pr"):
                self.low_cmd.mode_pr = 0
            if hasattr(self.low_cmd, "mode_machine") and hasattr(self.low_state, "mode_machine"):
                self.low_cmd.mode_machine = self.low_state.mode_machine
            count = min(len(self.low_cmd.motor_cmd), len(self.low_state.motor_state))
            for i in self.lowcmd_hold_indices:
                if i >= count:
                    continue
                motor = self.low_cmd.motor_cmd[i]
                motor.mode = 1
                motor.tau = 0.0
                motor.q = float(self.low_state.motor_state[i].q)
                motor.dq = 0.0
                motor.kp = self.args.hold_kp
                motor.kd = self.args.hold_kd
        elif self.args.r1_arm_sdk and self.low_state is not None:
            if hasattr(self.low_cmd, "mode_pr"):
                self.low_cmd.mode_pr = int(max(0.0, min(1.0, self.args.weight)) * 100.0)
            count = min(len(self.low_cmd.motor_cmd), len(self.low_state.motor_state))
            for sdk_i, idx in enumerate(self.r1_arm_sdk_indices):
                if idx >= count:
                    continue
                motor = self.low_cmd.motor_cmd[idx]
                motor.mode = 1
                motor.tau = 0.0
                motor.q = float(self.low_state.motor_state[idx].q)
                motor.dq = 0.0
                motor.kp = self._r1_arm_sdk_kp(sdk_i)
                motor.kd = self._r1_arm_sdk_kd(sdk_i)
        else:
            weight = max(0.0, min(1.0, self.args.weight))
            self.low_cmd.motor_cmd[self.weight_index].q = weight

        for idx, q in zip(self.right_arm_indices, command):
            if self.args.r1_arm_sdk and not self.args.debug_lowcmd and idx not in self.r1_arm_sdk_indices:
                continue
            motor = self.low_cmd.motor_cmd[idx]
            motor.mode = 1
            motor.tau = 0.0
            motor.q = q
            motor.dq = 0.0
            motor.kp = self.args.kp
            motor.kd = self.args.kd

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)

    @staticmethod
    def _r1_arm_sdk_kp(index_in_sdk: int) -> float:
        gains = [50.0, 50.0, 40.0, 40.0, 30.0, 50.0, 50.0, 40.0, 40.0, 30.0, 50.0, 15.0, 15.0]
        return gains[index_in_sdk] if index_in_sdk < len(gains) else 30.0

    @staticmethod
    def _r1_arm_sdk_kd(index_in_sdk: int) -> float:
        gains = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 1.0, 1.0]
        return gains[index_in_sdk] if index_in_sdk < len(gains) else 2.0

    def run(self) -> int:
        self.init_dds()
        deadline = time.monotonic() + max(0.0, self.args.duration)
        next_print = 0.0
        last_loop = time.monotonic()

        print(f"[R1-A7 RIGHT ARM] waiting for {self.args.state_topic} ...")
        while not self.done:
            now = time.monotonic()
            if self.args.duration > 0 and now >= deadline:
                break

            state = self._read_right_arm()
            if state is None:
                time.sleep(0.02)
                continue

            elapsed = now - self.start_time
            dt = now - last_loop
            last_loop = now

            if self.args.enable_control:
                target, command = self._compute_command(state, elapsed, dt)
                self._publish_arm_command(state, command)
                cmd_error = max(abs(c - q) for c, q in zip(command[:5], state.q[:5]))
                if cmd_error >= self.args.no_motion_command_error:
                    if self.no_motion_start is None:
                        self.no_motion_start = now
                        self.no_motion_baseline_q = list(state.q)
                    baseline = self.no_motion_baseline_q or state.q
                    moved = max(abs(q - b) for q, b in zip(state.q[:5], baseline[:5]))
                    if (
                        now - self.no_motion_start >= self.args.no_motion_warn_s
                        and moved <= self.args.no_motion_joint_delta
                        and now - self.last_no_motion_warning >= self.args.no_motion_warn_period
                    ):
                        self.last_no_motion_warning = now
                        print(
                            "[R1-A7 RIGHT ARM] WARNING: command is changing but lowstate is not moving. "
                            "The robot is ignoring this command topic or arm motors/control authority are not enabled. "
                            f"topic={self.args.command_topic} r1_arm_sdk={self.args.r1_arm_sdk} "
                            f"debug_lowcmd={self.args.debug_lowcmd} cmd_error={cmd_error:.3f} moved={moved:.3f}"
                        )
                else:
                    self.no_motion_start = None
                    self.no_motion_baseline_q = None
            else:
                target = self._target_for_mode(state, elapsed)
                command = list(state.q)

            if now >= next_print:
                next_print = now + max(0.1, self.args.print_period)
                q_text = " ".join(f"{name}={value:+.3f}" for name, value in zip(RIGHT_ARM_7_NAMES, state.q))
                cmd_text = " ".join(f"{name}={value:+.3f}" for name, value in zip(RIGHT_ARM_7_NAMES, command))
                tgt_text = " ".join(f"{name}={value:+.3f}" for name, value in zip(RIGHT_ARM_7_NAMES, target))
                print(f"[R1-A7 RIGHT ARM] q: {q_text}")
                print(f"[R1-A7 RIGHT ARM] cmd: {cmd_text}")
                print(f"[R1-A7 RIGHT ARM] tgt: {tgt_text}")

            time.sleep(max(0.0, 1.0 / max(1.0, self.args.hz)))

        if self.args.enable_control and self.publisher is not None:
            # Release arm_sdk control weight on exit.
            self._init_low_cmd_stop()
            if self.args.r1_arm_sdk and not self.args.debug_lowcmd and hasattr(self.low_cmd, "mode_pr"):
                self.low_cmd.mode_pr = 0
            elif not self.args.debug_lowcmd:
                self.low_cmd.motor_cmd[self.weight_index].q = 0.0
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.publisher.Write(self.low_cmd)
            print("[R1-A7 RIGHT ARM] released control")

        if self.first_state_time is None:
            print(f"[R1-A7 RIGHT ARM] no {self.args.state_topic} received")
            return 2
        print("[R1-A7 RIGHT ARM] done")
        return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="R1-A 7-DoF right-arm DDS communication")
    parser.add_argument("--interface", default="", help="network interface, e.g. enp3s0; empty uses SDK default")
    parser.add_argument("--domain_id", type=int, default=0, help="DDS domain id / Unitree channel id")
    parser.add_argument("--idl", choices=["hg", "go"], default="hg", help="Unitree DDS IDL family")
    parser.add_argument("--state_topic", default="rt/lowstate", help="robot low-state DDS topic")
    parser.add_argument("--command_topic", default="rt/arm_sdk", help="high-level arm command DDS topic")
    parser.add_argument("--mode", choices=["monitor", "hold", "test", "lift"], default="monitor")
    parser.add_argument("--enable_control", action="store_true", help="publish rt/arm_sdk commands")
    parser.add_argument("--debug_lowcmd", action="store_true", help="publish direct debug commands on rt/lowcmd")
    parser.add_argument("--r1_arm_sdk", action="store_true", help="use R1 official rt/arm_sdk format: mode_pr is weight")
    parser.add_argument("--enter_debug_mode", action="store_true", help="release active motion mode with motion_switcher")
    parser.add_argument("--duration", type=float, default=10.0, help="run duration in seconds; 0 means forever")
    parser.add_argument("--hz", type=float, default=50.0, help="publish/read loop frequency")
    parser.add_argument("--print_period", type=float, default=0.5)
    parser.add_argument(
        "--right_arm_indices",
        default="22,23,24,25,26,27,28",
        help="right arm motor indices: shoulder_pitch,roll,yaw,elbow,wrist_roll,pitch,yaw",
    )
    parser.add_argument(
        "--weight_index",
        type=int,
        default=31,
        help="arm_sdk weight motor index. R1-A/H2-style is 31; G1-style is 29.",
    )
    parser.add_argument("--r1_arm_sdk_indices", default="15,16,17,18,19,22,23,24,25,26,13,29,30")
    parser.add_argument("--lowcmd_hold_indices", default="22,23,24,25,26,27,28")
    parser.add_argument("--weight", type=float, default=1.0, help="arm_sdk control weight when control is enabled")
    parser.add_argument("--kp", type=float, default=20.0, help="low initial position gain for real robot test")
    parser.add_argument("--kd", type=float, default=1.0, help="velocity damping gain")
    parser.add_argument("--hold_kp", type=float, default=20.0, help="debug_lowcmd hold gain for non-commanded joints")
    parser.add_argument("--hold_kd", type=float, default=1.0, help="debug_lowcmd hold damping for non-commanded joints")
    parser.add_argument("--max_speed_rad_s", type=float, default=0.20, help="right arm command rate limit")
    parser.add_argument("--test_amplitude_deg", type=float, default=3.0, help="test shoulder-pitch amplitude")
    parser.add_argument("--test_frequency_hz", type=float, default=0.10, help="test shoulder-pitch frequency")
    parser.add_argument(
        "--test_joint_index",
        type=int,
        default=1,
        help="1-based right-arm joint index used in test mode; 7 is right_wrist_yaw",
    )
    parser.add_argument("--lift_offset_deg", type=float, default=35.0, help="positive shoulder-pitch lift target")
    parser.add_argument("--lift_ramp_s", type=float, default=6.0, help="seconds to ramp into lift target")
    parser.add_argument("--no_motion_command_error", type=float, default=0.03)
    parser.add_argument("--no_motion_joint_delta", type=float, default=0.005)
    parser.add_argument("--no_motion_warn_s", type=float, default=2.0)
    parser.add_argument("--no_motion_warn_period", type=float, default=3.0)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.mode != "monitor" and not args.enable_control:
        print("[R1-A7 RIGHT ARM] refusing to run control mode without --enable_control")
        return 2
    if args.enable_control:
        print("WARNING: This will publish rt/arm_sdk commands to the real robot right arm.")
        print("Keep one hand on the emergency stop. Ensure the arm workspace is clear.")
        if not sys.stdin.isatty():
            print("[R1-A7 RIGHT ARM] control requires an interactive terminal")
            return 2
        answer = input("Type ENABLE to continue: ").strip()
        if answer != "ENABLE":
            print("[R1-A7 RIGHT ARM] aborted")
            return 2

    node = R1A7RightArmDDS(args)

    def _handle_signal(_signum, _frame):
        node.done = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    return node.run()


if __name__ == "__main__":
    raise SystemExit(main())
