#!/usr/bin/env python3
"""Real R1-A7 right-arm VR IK control through rt/lowcmd.

This is intentionally right-arm only for the first hardware test. It reads the
current real right-arm joints as the IK seed, calibrates the current controller
pose as zero, maps small VR deltas to the right wrist target, solves IK, and
publishes rate-limited right-arm joint commands.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pinocchio as pin

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.r1a7_dual_arm_ik import RIGHT_JOINTS, R1A7DualArmIK
from tools.r1a7_vr_ik_preview import (
    _apply_axis_options,
    _controller_position,
    _limit_warnings,
    _vr_delta_to_robot,
)
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


STEAMVR_ROOT = Path.home() / ".local/share/Steam/steamapps/common/SteamVR"
RIGHT_ARM_NAMES = [
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]


@dataclass
class ArmState:
    q: np.ndarray
    dq: np.ndarray
    stamp: float


class R1A7VRRightArmReal:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.done = False
        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state: Optional[LowState_] = None
        self.first_state_time: Optional[float] = None
        self.arm_indices = self._parse_indices(args.right_arm_indices)
        if len(self.arm_indices) != 7:
            raise ValueError("--right_arm_indices must contain exactly 7 indices")

        self.subscriber = None
        self.publisher = None
        self.ik = R1A7DualArmIK(tcp_x=args.tcp_x)
        self.q_model = pin.neutral(self.ik.model)
        self.command_q: Optional[np.ndarray] = None
        self.vr_zero: Optional[np.ndarray] = None
        self.home_target = None
        self.last_valid_pose_time: Optional[float] = None
        self.last_print = 0.0

    @staticmethod
    def _parse_indices(text: str) -> list[int]:
        return [int(part.strip()) for part in text.split(",") if part.strip()]

    def _enter_debug_mode(self) -> None:
        msc = MotionSwitcherClient()
        msc.SetTimeout(2.0)
        msc.Init()
        status, result = msc.CheckMode()
        print(f"[R1-A7 VR REAL] motion_switcher CheckMode: status={status} result={result}")
        try:
            while result and result.get("name"):
                print("[R1-A7 VR REAL] releasing active mode:", result)
                msc.ReleaseMode()
                time.sleep(0.5)
                status, result = msc.CheckMode()
                print(f"[R1-A7 VR REAL] motion_switcher CheckMode: status={status} result={result}")
        except Exception as exc:
            print(f"[R1-A7 VR REAL] failed to enter debug mode: {exc}")

    def init(self) -> None:
        ChannelFactoryInitialize(self.args.domain_id, self.args.interface)
        if self.args.enter_debug_mode:
            self._enter_debug_mode()
        self.subscriber = ChannelSubscriber(self.args.state_topic, LowState_)
        self.subscriber.Init(self._lowstate_handler, 10)
        self.publisher = ChannelPublisher(self.args.command_topic, LowCmd_)
        self.publisher.Init()
        print("[R1-A7 VR REAL] DDS initialized")
        print("[R1-A7 VR REAL] interface:", self.args.interface)
        print("[R1-A7 VR REAL] state topic:", self.args.state_topic)
        print("[R1-A7 VR REAL] command topic:", self.args.command_topic)
        print("[R1-A7 VR REAL] right arm indices:", self.arm_indices)
        print(
            "[R1-A7 VR REAL] control limits:"
            f" scale={self.args.scale:.3f}"
            f" max_delta_m={self.args.max_delta_m:.3f}"
            f" max_speed_rad_s={self.args.max_speed_rad_s:.3f}"
            f" max_command_lead={self.args.max_command_lead:.3f}"
        )

    def _lowstate_handler(self, msg: LowState_) -> None:
        self.low_state = msg
        if self.first_state_time is None:
            self.first_state_time = time.monotonic()

    def _read_arm(self) -> Optional[ArmState]:
        if self.low_state is None:
            return None
        motor_state = self.low_state.motor_state
        max_idx = max(self.arm_indices)
        if len(motor_state) <= max_idx:
            raise RuntimeError(f"lowstate has {len(motor_state)} motors, requested index {max_idx}")
        q = np.array([motor_state[i].q for i in self.arm_indices], dtype=float)
        dq = np.array([motor_state[i].dq for i in self.arm_indices], dtype=float)
        return ArmState(q=q, dq=dq, stamp=time.monotonic())

    def _write_right_arm_to_model(self, arm_q: np.ndarray) -> None:
        for name, value in zip(RIGHT_JOINTS, arm_q):
            jid = self.ik.model.getJointId(name)
            self.q_model[self.ik.model.joints[jid].idx_q] = float(value)

    def _right_q_from_model(self, q_model: np.ndarray) -> np.ndarray:
        values = []
        for name in RIGHT_JOINTS:
            jid = self.ik.model.getJointId(name)
            values.append(float(q_model[self.ik.model.joints[jid].idx_q]))
        return np.array(values, dtype=float)

    def _solve_right_position(
        self,
        target: pin.SE3,
        q0: np.ndarray,
    ) -> tuple[np.ndarray, float, int]:
        q = np.clip(q0.copy(), self.ik.lower, self.ik.upper)
        active = np.array(self.ik.right_v, dtype=int)
        for it in range(self.args.ik_max_iters):
            current = self.ik._frame_pose(q, self.ik.right.frame, self.ik.right.offset)
            err = target.translation - current.translation
            err_norm = float(np.linalg.norm(err))
            if err_norm <= self.args.max_ik_err:
                return q, err_norm, it
            j = self.ik._frame_jacobian(q, self.ik.right.frame, self.ik.right.offset)[:3, :][:, active]
            lhs = j @ j.T + (self.args.ik_damping**2) * np.eye(3)
            dq_active = j.T @ np.linalg.solve(lhs, err)
            dq = np.zeros(self.ik.model.nv)
            dq[active] = np.clip(self.args.ik_step_scale * dq_active, -0.035, 0.035)
            q = pin.integrate(self.ik.model, q, dq)
            q = np.clip(q, self.ik.lower, self.ik.upper)
        current = self.ik._frame_pose(q, self.ik.right.frame, self.ik.right.offset)
        return q, float(np.linalg.norm(target.translation - current.translation)), self.args.ik_max_iters

    def _init_low_cmd_stop(self) -> None:
        for motor in self.low_cmd.motor_cmd:
            motor.tau = 0.0
            motor.q = 0.0
            motor.dq = 0.0
            motor.kp = 0.0
            motor.kd = 0.0

    def _publish(self, state: ArmState, command_q: np.ndarray) -> None:
        self._init_low_cmd_stop()
        if self.low_state is not None:
            if hasattr(self.low_cmd, "mode_pr"):
                self.low_cmd.mode_pr = 0
            if hasattr(self.low_cmd, "mode_machine") and hasattr(self.low_state, "mode_machine"):
                self.low_cmd.mode_machine = self.low_state.mode_machine
            count = min(len(self.low_cmd.motor_cmd), len(self.low_state.motor_state))
            for i in range(count):
                motor = self.low_cmd.motor_cmd[i]
                motor.mode = 1
                motor.tau = 0.0
                motor.q = float(self.low_state.motor_state[i].q)
                motor.dq = 0.0
                motor.kp = self.args.hold_kp
                motor.kd = self.args.hold_kd
        for idx, q in zip(self.arm_indices, command_q):
            motor = self.low_cmd.motor_cmd[idx]
            motor.mode = 1
            motor.tau = 0.0
            motor.q = float(q)
            motor.dq = 0.0
            motor.kp = self.args.kp
            motor.kd = self.args.kd
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)

    def _release(self) -> None:
        self._init_low_cmd_stop()
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)
        print("[R1-A7 VR REAL] released lowcmd gains")

    def _step_command(self, state: ArmState, target_q: Optional[np.ndarray], dt: float) -> np.ndarray:
        if self.command_q is None:
            self.command_q = state.q.copy()
        if target_q is None:
            target_q = self.command_q.copy()
        max_delta = max(0.0, self.args.max_speed_rad_s) * max(dt, 1e-3)
        next_q = self.command_q + np.clip(target_q - self.command_q, -max_delta, max_delta)
        lead = np.clip(next_q - state.q, -self.args.max_command_lead, self.args.max_command_lead)
        self.command_q = state.q + lead
        return self.command_q.copy()

    def run(self) -> int:
        os.environ.setdefault("VR_OVERRIDE", str(STEAMVR_ROOT))
        try:
            import openvr  # type: ignore
        except Exception as exc:
            print("[R1-A7 VR REAL] missing openvr module:", exc)
            return 4

        self.init()
        vr = openvr.init(openvr.VRApplication_Other)
        deadline = time.monotonic() + max(0.0, self.args.duration)
        last_loop = time.monotonic()
        print("[R1-A7 VR REAL] waiting for robot lowstate and right controller pose ...")
        try:
            while not self.done:
                now = time.monotonic()
                if self.args.duration > 0 and now >= deadline:
                    break
                state = self._read_arm()
                if state is None:
                    time.sleep(0.02)
                    continue
                if self.home_target is None:
                    self._write_right_arm_to_model(state.q)
                    self.home_target = self.ik._frame_pose(
                        self.q_model,
                        self.ik.right.frame,
                        self.ik.right.offset,
                    )
                    self.command_q = state.q.copy()
                    print("[R1-A7 VR REAL] calibrated robot right q:", state.q.tolist())

                left_pos = _controller_position(vr, openvr, "left")
                right_pos = _controller_position(vr, openvr, "right")
                controller_pos = left_pos if self.args.swap_hands else right_pos
                target_q = None
                info = "no_pose"
                err = 0.0
                if controller_pos is not None:
                    if self.vr_zero is None:
                        self.vr_zero = controller_pos.copy()
                        print("[R1-A7 VR REAL] calibrated VR right zero:", self.vr_zero.tolist())
                    delta = _apply_axis_options(
                        _vr_delta_to_robot(controller_pos - self.vr_zero, self.args.scale),
                        self.args,
                    )
                    delta = np.clip(delta, -self.args.max_delta_m, self.args.max_delta_m)
                    target = self.home_target.copy()
                    target.translation += delta
                    self._write_right_arm_to_model(self.command_q if self.command_q is not None else state.q)
                    q_seed = self.q_model.copy()
                    q_model, err, _iters = self._solve_right_position(target, q_seed)
                    warnings = _limit_warnings(self.ik, RIGHT_JOINTS, q_model)
                    if err <= self.args.max_ik_err and not warnings:
                        target_q = self._right_q_from_model(q_model)
                        self.q_model = q_model.copy()
                        self.last_valid_pose_time = now
                    else:
                        target_q = None
                    info = (
                        f"delta=({delta[0]:+.3f},{delta[1]:+.3f},{delta[2]:+.3f})"
                        f" err={err:.4f}"
                        f" iters={_iters}"
                        f" limit={'yes' if warnings else 'no'}"
                    )

                dt = now - last_loop
                last_loop = now
                command_q = self._step_command(state, target_q, dt)
                self._publish(state, command_q)

                if now - self.last_print >= self.args.print_period:
                    self.last_print = now
                    current = " ".join(f"{n}={v:+.3f}" for n, v in zip(RIGHT_ARM_NAMES[:4], state.q[:4]))
                    command = " ".join(f"{n}={v:+.3f}" for n, v in zip(RIGHT_ARM_NAMES[:4], command_q[:4]))
                    print(
                        f"[R1-A7 VR REAL] target={'yes' if target_q is not None else 'no '} "
                        f"{info} current: {current} cmd: {command}"
                    )
                time.sleep(max(0.0, 1.0 / max(1.0, self.args.hz)))
        finally:
            self._release()
            openvr.shutdown()
        return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real R1-A7 right-arm VR IK control")
    parser.add_argument("--interface", default="enx9c69d37d0967")
    parser.add_argument("--domain_id", type=int, default=0)
    parser.add_argument("--state_topic", default="rt/lowstate")
    parser.add_argument("--command_topic", default="rt/lowcmd")
    parser.add_argument("--right_arm_indices", default="22,23,24,25,26,27,28")
    parser.add_argument("--enter_debug_mode", action="store_true")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--print_period", type=float, default=0.25)
    parser.add_argument("--swap_hands", action="store_true")
    parser.add_argument("--scale", type=float, default=0.12)
    parser.add_argument("--max_delta_m", type=float, default=0.04)
    parser.add_argument(
        "--axis_mode",
        choices=["full", "vertical", "lateral", "depth"],
        default="full",
    )
    parser.add_argument("--x_sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--y_sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--z_sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--tcp_x", type=float, default=0.0)
    parser.add_argument("--ik_max_iters", type=int, default=30)
    parser.add_argument("--ik_damping", type=float, default=0.05)
    parser.add_argument("--ik_step_scale", type=float, default=0.55)
    parser.add_argument("--max_ik_err", type=float, default=0.008)
    parser.add_argument("--kp", type=float, default=16.0)
    parser.add_argument("--kd", type=float, default=1.0)
    parser.add_argument("--hold_kp", type=float, default=10.0)
    parser.add_argument("--hold_kd", type=float, default=0.8)
    parser.add_argument("--max_speed_rad_s", type=float, default=0.08)
    parser.add_argument("--max_command_lead", type=float, default=0.08)
    parser.add_argument("--assume_yes", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    print("WARNING: This will publish rt/lowcmd commands to the real R1-A7 right arm.")
    print("Keep the emergency stop ready and keep the arm workspace clear.")
    if not args.assume_yes:
        if not sys.stdin.isatty():
            print("[R1-A7 VR REAL] interactive confirmation required")
            return 2
        answer = input("Type ENABLE to continue: ").strip()
        if answer != "ENABLE":
            print("[R1-A7 VR REAL] aborted")
            return 2
    node = R1A7VRRightArmReal(args)

    def _handle_signal(_signum, _frame):
        node.done = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    return node.run()


if __name__ == "__main__":
    raise SystemExit(main())
