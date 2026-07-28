#!/usr/bin/env python3
"""Real R1-A7 dual-arm Quest/Vuer Cartesian IK control through rt/lowcmd."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pinocchio as pin

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

XR_TELEOP = Path(os.getenv("XR_TELEOP_ROOT", "/home/robot/xr_teleoperate"))
XR_TELEOP_TELEOP = XR_TELEOP / "teleop"
XR_TELEOP_TV_SRC = XR_TELEOP_TELEOP / "televuer" / "src"
for path in (XR_TELEOP_TV_SRC, XR_TELEOP_TELEOP, XR_TELEOP):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from televuer import TeleVuerWrapper  # noqa: E402
from tools.r1a7_dual_arm_ik import LEFT_JOINTS, RIGHT_JOINTS, R1A7DualArmIK  # noqa: E402
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient  # noqa: E402
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber  # noqa: E402
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_  # noqa: E402
from unitree_sdk2py.utils.crc import CRC  # noqa: E402


ARM_NAMES = [
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]


class R1A7VRDualArmCartesianReal:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.done = False
        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state: Optional[LowState_] = None
        self.left_indices = self._parse_indices(args.left_arm_indices)
        self.right_indices = self._parse_indices(args.right_arm_indices)
        self.arm_indices = self.left_indices + self.right_indices
        self.hold_indices = self._parse_indices(args.lowcmd_hold_indices)
        if len(self.left_indices) != 7 or len(self.right_indices) != 7:
            raise ValueError("left/right arm indices must each contain exactly 7 indices")

        self.ik = R1A7DualArmIK(args.urdf, args.left_frame, args.right_frame, args.tcp_x)
        self.q_model = pin.neutral(self.ik.model)
        self.command_q: Optional[np.ndarray] = None
        self.left_home: Optional[pin.SE3] = None
        self.right_home: Optional[pin.SE3] = None
        self.left_zero: Optional[np.ndarray] = None
        self.right_zero: Optional[np.ndarray] = None
        self.publisher = None
        self.tv = None
        self.last_print = 0.0

    @staticmethod
    def _parse_indices(text: str) -> list[int]:
        return [int(part.strip()) for part in text.split(",") if part.strip()]

    def _enter_debug_mode(self) -> None:
        msc = MotionSwitcherClient()
        msc.SetTimeout(2.0)
        msc.Init()
        status, result = msc.CheckMode()
        print(f"[R1-A7 VR CART REAL] motion_switcher CheckMode: status={status} result={result}")
        try:
            while result and result.get("name"):
                print("[R1-A7 VR CART REAL] releasing active mode:", result)
                msc.ReleaseMode()
                time.sleep(0.5)
                status, result = msc.CheckMode()
                print(f"[R1-A7 VR CART REAL] motion_switcher CheckMode: status={status} result={result}")
        except Exception as exc:
            print(f"[R1-A7 VR CART REAL] failed to enter debug mode: {exc}")

    def init(self) -> None:
        ChannelFactoryInitialize(self.args.domain_id, self.args.interface)
        if self.args.enter_debug_mode:
            self._enter_debug_mode()
        ChannelSubscriber(self.args.state_topic, LowState_).Init(self._lowstate_handler, 10)
        self.publisher = ChannelPublisher(self.args.command_topic, LowCmd_)
        self.publisher.Init()
        self.tv = TeleVuerWrapper(
            use_hand_tracking=False,
            binocular=False,
            img_shape=(480, 640),
            display_mode="pass-through",
            zmq=False,
            webrtc=False,
            arm_reference_mode="head_yaw",
        )
        print("[R1-A7 VR CART REAL] DDS initialized")
        print("[R1-A7 VR CART REAL] IK: R1-A7 URDF Cartesian position IK")
        print("[R1-A7 VR CART REAL] interface:", self.args.interface)
        print("[R1-A7 VR CART REAL] state topic:", self.args.state_topic)
        print("[R1-A7 VR CART REAL] command topic:", self.args.command_topic)
        print("[R1-A7 VR CART REAL] left arm indices:", self.left_indices)
        print("[R1-A7 VR CART REAL] right arm indices:", self.right_indices)
        print("[R1-A7 VR CART REAL] open Quest URL:")
        print(f"https://{self.args.host_ip}:8012/?ws=wss://{self.args.host_ip}:8012")

    def _lowstate_handler(self, msg: LowState_) -> None:
        self.low_state = msg

    def _read_arm_q(self) -> Optional[np.ndarray]:
        if self.low_state is None:
            return None
        motor_state = self.low_state.motor_state
        if len(motor_state) <= max(self.arm_indices):
            raise RuntimeError(f"lowstate has {len(motor_state)} motors, requested index {max(self.arm_indices)}")
        return np.array([motor_state[i].q for i in self.arm_indices], dtype=float)

    def _write_arms_to_model(self, q14: np.ndarray) -> None:
        for name, value in zip(LEFT_JOINTS + RIGHT_JOINTS, q14):
            jid = self.ik.model.getJointId(name)
            self.q_model[self.ik.model.joints[jid].idx_q] = float(value)

    def _arms_from_model(self, q_model: np.ndarray) -> np.ndarray:
        values = []
        for name in LEFT_JOINTS + RIGHT_JOINTS:
            jid = self.ik.model.getJointId(name)
            values.append(float(q_model[self.ik.model.joints[jid].idx_q]))
        return np.array(values, dtype=float)

    def _joint_limit_warnings(self, q_model: np.ndarray) -> list[str]:
        warnings = []
        for name in LEFT_JOINTS + RIGHT_JOINTS:
            jid = self.ik.model.getJointId(name)
            idx = self.ik.model.joints[jid].idx_q
            value = float(q_model[idx])
            lower = float(self.ik.lower[idx])
            upper = float(self.ik.upper[idx])
            if value <= lower + self.args.limit_margin:
                warnings.append(f"{name} lower {value:+.3f}/{lower:+.3f}")
            elif value >= upper - self.args.limit_margin:
                warnings.append(f"{name} upper {value:+.3f}/{upper:+.3f}")
        return warnings

    def _map_vuer_delta(self, delta: np.ndarray, side_sign: float) -> np.ndarray:
        if self.args.vuer_mapping == "direct":
            out = delta.copy()
        else:
            out = np.array([delta[2], side_sign * delta[0], delta[1]], dtype=float)
        out *= self.args.scale
        out[0] *= self.args.x_sign
        out[1] *= self.args.y_sign
        out[2] *= self.args.z_sign
        if self.args.axis_mode == "vertical":
            out[0] = 0.0
            out[1] = 0.0
        elif self.args.axis_mode == "lateral":
            out[0] = 0.0
            out[2] = 0.0
        elif self.args.axis_mode == "depth":
            out[1] = 0.0
            out[2] = 0.0
        return np.clip(out, -self.args.max_delta_m, self.args.max_delta_m)

    def _solve_position(
        self,
        left_target: pin.SE3,
        right_target: pin.SE3,
        q0: np.ndarray,
    ) -> tuple[np.ndarray, float, int]:
        q = np.clip(q0.copy(), self.ik.lower, self.ik.upper)
        active = np.array(self.ik.active_v, dtype=int)
        for it in range(self.args.ik_max_iters):
            left_pose = self.ik._frame_pose(q, self.ik.left.frame, self.ik.left.offset)
            right_pose = self.ik._frame_pose(q, self.ik.right.frame, self.ik.right.offset)
            err = np.concatenate(
                [
                    left_target.translation - left_pose.translation,
                    right_target.translation - right_pose.translation,
                ]
            )
            err_norm = float(np.linalg.norm(err))
            if err_norm <= self.args.max_ik_err:
                return q, err_norm, it
            j_full = np.vstack(
                [
                    self.ik._frame_jacobian(q, self.ik.left.frame, self.ik.left.offset)[:3, :],
                    self.ik._frame_jacobian(q, self.ik.right.frame, self.ik.right.offset)[:3, :],
                ]
            )
            j = j_full[:, active]
            lhs = j @ j.T + (self.args.ik_damping**2) * np.eye(j.shape[0])
            dq_active = j.T @ np.linalg.solve(lhs, err)
            dq = np.zeros(self.ik.model.nv)
            dq[active] = np.clip(self.args.ik_step_scale * dq_active, -self.args.ik_joint_step, self.args.ik_joint_step)
            q = pin.integrate(self.ik.model, q, dq)
            q = np.clip(q, self.ik.lower, self.ik.upper)
        left_pose = self.ik._frame_pose(q, self.ik.left.frame, self.ik.left.offset)
        right_pose = self.ik._frame_pose(q, self.ik.right.frame, self.ik.right.offset)
        err = np.concatenate([left_target.translation - left_pose.translation, right_target.translation - right_pose.translation])
        return q, float(np.linalg.norm(err)), self.args.ik_max_iters

    def _solve_with_limit_backoff(
        self,
        left_home: pin.SE3,
        right_home: pin.SE3,
        left_delta: np.ndarray,
        right_delta: np.ndarray,
        q_seed: np.ndarray,
    ) -> tuple[Optional[np.ndarray], float, float, int, list[str]]:
        last_err = 0.0
        last_iters = 0
        last_warnings: list[str] = []
        for scale in self.args.backoff_scales:
            left_target = left_home.copy()
            right_target = right_home.copy()
            left_target.translation += left_delta * scale
            right_target.translation += right_delta * scale
            q_model, err, iters = self._solve_position(left_target, right_target, q_seed)
            warnings = self._joint_limit_warnings(q_model)
            last_err, last_iters, last_warnings = err, iters, warnings
            if err <= self.args.max_ik_err and not warnings:
                return self._arms_from_model(q_model), err, scale, iters, warnings
        return None, last_err, 0.0, last_iters, last_warnings

    def _init_low_cmd_stop(self) -> None:
        for motor in self.low_cmd.motor_cmd:
            motor.tau = 0.0
            motor.q = 0.0
            motor.dq = 0.0
            motor.kp = 0.0
            motor.kd = 0.0

    def _arm_gain(self, joint_i: int) -> tuple[float, float]:
        local_i = joint_i % 7
        if local_i >= 4:
            return self.args.kp_wrist, self.args.kd_wrist
        return self.args.kp_low, self.args.kd_low

    def _publish(self, command_q: np.ndarray) -> None:
        assert self.publisher is not None
        self._init_low_cmd_stop()
        if self.low_state is not None:
            if hasattr(self.low_cmd, "mode_pr"):
                self.low_cmd.mode_pr = 0
            if hasattr(self.low_cmd, "mode_machine") and hasattr(self.low_state, "mode_machine"):
                self.low_cmd.mode_machine = self.low_state.mode_machine
            count = min(len(self.low_cmd.motor_cmd), len(self.low_state.motor_state))
            for i in self.hold_indices:
                if i >= count:
                    continue
                motor = self.low_cmd.motor_cmd[i]
                motor.mode = 1
                motor.tau = 0.0
                motor.q = float(self.low_state.motor_state[i].q)
                motor.dq = 0.0
                motor.kp = self.args.hold_kp
                motor.kd = self.args.hold_kd
        for joint_i, (idx, q) in enumerate(zip(self.arm_indices, command_q)):
            motor = self.low_cmd.motor_cmd[idx]
            motor.mode = 1
            motor.tau = 0.0
            motor.q = float(q)
            motor.dq = 0.0
            motor.kp, motor.kd = self._arm_gain(joint_i)
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)

    def _release(self) -> None:
        if self.publisher is None:
            return
        self._init_low_cmd_stop()
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)
        print("[R1-A7 VR CART REAL] released lowcmd gains")

    def _step_command(self, state_q: np.ndarray, target_q: Optional[np.ndarray], dt: float) -> np.ndarray:
        if self.command_q is None:
            self.command_q = state_q.copy()
        if target_q is None:
            target_q = self.command_q.copy()
        delta = target_q - state_q
        max_delta = max(0.0, self.args.arm_velocity_limit) * max(dt, 1e-3)
        motion_scale = float(np.max(np.abs(delta)) / max(max_delta, 1e-6))
        self.command_q = state_q + delta / max(motion_scale, 1.0)
        return self.command_q.copy()

    def run(self) -> int:
        self.init()
        assert self.tv is not None
        deadline = time.monotonic() + max(0.0, self.args.duration)
        last_loop = time.monotonic()
        print("[R1-A7 VR CART REAL] waiting for robot lowstate and Quest controller poses ...")
        try:
            while not self.done:
                now = time.monotonic()
                if self.args.duration > 0 and now >= deadline:
                    break
                state_q = self._read_arm_q()
                if state_q is None:
                    time.sleep(0.02)
                    continue
                if self.left_home is None or self.right_home is None:
                    self._write_arms_to_model(state_q)
                    self.left_home = self.ik._frame_pose(self.q_model, self.ik.left.frame, self.ik.left.offset)
                    self.right_home = self.ik._frame_pose(self.q_model, self.ik.right.frame, self.ik.right.offset)
                    self.command_q = state_q.copy()
                    print("[R1-A7 VR CART REAL] calibrated robot arm q:", np.round(state_q, 3).tolist())

                tele = self.tv.get_tele_data()
                target_q = None
                info = "waiting"
                if tele.motion_data_ready:
                    left_vr = np.asarray(tele.left_wrist_pose[:3, 3], dtype=float)
                    right_vr = np.asarray(tele.right_wrist_pose[:3, 3], dtype=float)
                    if self.left_zero is None or self.right_zero is None:
                        self.left_zero = left_vr.copy()
                        self.right_zero = right_vr.copy()
                        print("[R1-A7 VR CART REAL] calibrated Quest controller zero")
                    left_delta = self._map_vuer_delta(left_vr - self.left_zero, 1.0)
                    right_delta = self._map_vuer_delta(right_vr - self.right_zero, -1.0)
                    self._write_arms_to_model(self.command_q if self.command_q is not None else state_q)
                    target_q, err, scale, iters, warnings = self._solve_with_limit_backoff(
                        self.left_home,
                        self.right_home,
                        left_delta,
                        right_delta,
                        self.q_model.copy(),
                    )
                    info = (
                        f"cart L=({left_delta[0]:+.3f},{left_delta[1]:+.3f},{left_delta[2]:+.3f})"
                        f" R=({right_delta[0]:+.3f},{right_delta[1]:+.3f},{right_delta[2]:+.3f})"
                        f" err={err:.4f} backoff={scale:.2f} iters={iters}"
                        f" limit={'yes' if warnings else 'no'}"
                    )
                    if warnings and self.args.print_limits:
                        info += " " + "; ".join(warnings[:2])

                dt = now - last_loop
                last_loop = now
                command_q = self._step_command(state_q, target_q, dt)
                self._publish(command_q)

                if now - self.last_print >= self.args.print_period:
                    self.last_print = now
                    cmd_err = float(np.max(np.abs(command_q - state_q)))
                    left_q = " ".join(f"{n}={v:+.3f}" for n, v in zip(ARM_NAMES[:4], state_q[:4]))
                    left_cmd = " ".join(f"{n}={v:+.3f}" for n, v in zip(ARM_NAMES[:4], command_q[:4]))
                    right_q = " ".join(f"{n}={v:+.3f}" for n, v in zip(ARM_NAMES[7:11], state_q[7:11]))
                    right_cmd = " ".join(f"{n}={v:+.3f}" for n, v in zip(ARM_NAMES[7:11], command_q[7:11]))
                    print(f"[R1-A7 VR CART REAL] {info} cmd_err={cmd_err:.3f}")
                    print(f"  left_q  : {left_q}")
                    print(f"  left_cmd: {left_cmd}")
                    print(f"  right_q  : {right_q}")
                    print(f"  right_cmd: {right_cmd}")
                time.sleep(max(0.0, 1.0 / max(1.0, self.args.hz)))
        finally:
            self._release()
            try:
                self.tv.close()
            except Exception:
                pass
        return 0


def _parse_backoff(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated backoff scales")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real R1-A7 dual-arm Quest/Vuer Cartesian IK control")
    parser.add_argument("--interface", default="enx9c69d37d0967")
    parser.add_argument("--domain_id", type=int, default=0)
    parser.add_argument("--state_topic", default="rt/lowstate")
    parser.add_argument("--command_topic", default="rt/lowcmd")
    parser.add_argument("--host_ip", default=os.getenv("HOST_IP", "192.168.1.127"))
    parser.add_argument("--urdf", default="/home/robot/IsaacLab/bolt_nut_assembly/R1_A7_official/A7.urdf")
    parser.add_argument("--left_frame", default="left_wrist_yaw_link")
    parser.add_argument("--right_frame", default="right_wrist_yaw_link")
    parser.add_argument("--left_arm_indices", default="15,16,17,18,19,20,21")
    parser.add_argument("--right_arm_indices", default="22,23,24,25,26,27,28")
    parser.add_argument("--lowcmd_hold_indices", default="13,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30")
    parser.add_argument("--enter_debug_mode", action="store_true")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--print_period", type=float, default=0.25)
    parser.add_argument("--scale", type=float, default=0.35)
    parser.add_argument("--max_delta_m", type=float, default=0.12)
    parser.add_argument("--vuer_mapping", choices=["direct", "head_yaw"], default="direct")
    parser.add_argument("--axis_mode", choices=["full", "vertical", "lateral", "depth"], default="full")
    parser.add_argument("--x_sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--y_sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--z_sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--tcp_x", type=float, default=0.0)
    parser.add_argument("--ik_max_iters", type=int, default=40)
    parser.add_argument("--ik_damping", type=float, default=0.06)
    parser.add_argument("--ik_step_scale", type=float, default=0.55)
    parser.add_argument("--ik_joint_step", type=float, default=0.035)
    parser.add_argument("--max_ik_err", type=float, default=0.015)
    parser.add_argument("--limit_margin", type=float, default=0.08)
    parser.add_argument("--backoff_scales", type=_parse_backoff, default=[1.0, 0.75, 0.5, 0.25, 0.0])
    parser.add_argument("--kp_low", type=float, default=80.0)
    parser.add_argument("--kd_low", type=float, default=3.0)
    parser.add_argument("--kp_wrist", type=float, default=40.0)
    parser.add_argument("--kd_wrist", type=float, default=1.5)
    parser.add_argument("--hold_kp", type=float, default=10.0)
    parser.add_argument("--hold_kd", type=float, default=0.8)
    parser.add_argument("--arm_velocity_limit", type=float, default=2.0)
    parser.add_argument("--print_limits", action="store_true")
    parser.add_argument("--assume_yes", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    print("WARNING: This will publish rt/lowcmd commands to the real R1-A7 dual arms.")
    print("Keep the emergency stop ready and keep both arm workspaces clear.")
    print("This controller uses R1-A7 Cartesian IK with workspace and joint-limit backoff.")
    if not args.assume_yes:
        if not sys.stdin.isatty():
            print("[R1-A7 VR CART REAL] interactive confirmation required")
            return 2
        answer = input("Type ENABLE to continue: ").strip()
        if answer != "ENABLE":
            print("[R1-A7 VR CART REAL] aborted")
            return 2
    node = R1A7VRDualArmCartesianReal(args)

    def _handle_signal(_signum, _frame):
        node.done = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    return node.run()


if __name__ == "__main__":
    raise SystemExit(main())
