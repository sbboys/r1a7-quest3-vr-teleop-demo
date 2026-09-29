#!/usr/bin/env python3
"""Camera-pose teleoperation for the real R1-A 7-DoF right arm.

Default mode is preview-only. Use --enable_control only after preview prints
stable camera targets and the right-arm joint indices match the robot state.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from action_provider.gemini_pose_source import GeminiPoseSource
from tools.dex1_1_gripper_dds import Dex1GripperDDS
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


RIGHT_ARM_NAMES = [
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
    q: np.ndarray
    dq: np.ndarray
    stamp: float


class R1A7CameraRealTeleop:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.done = False
        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state: Optional[LowState_] = None
        self.first_state_time: Optional[float] = None

        self.arm_indices = self._parse_indices(args.right_arm_indices)
        if len(self.arm_indices) != 7:
            raise ValueError("--right_arm_indices must contain exactly 7 motor indices")
        self.r1_arm_sdk_indices = self._parse_indices(args.r1_arm_sdk_indices)
        self.lowcmd_hold_indices = self._parse_indices(args.lowcmd_hold_indices)
        self.weight_index = int(args.weight_index)

        self.source = GeminiPoseSource(
            hand=args.human_hand,
            show=args.show,
            mirror_view=args.mirror_view,
            filter_alpha=args.camera_filter_alpha,
            debug=args.debug_pose,
            min_visibility=args.min_visibility,
            min_wrist_shoulder_m=args.min_wrist_shoulder_m,
        )

        self.subscriber = None
        self.publisher = None
        self.home_q: Optional[np.ndarray] = None
        self.command_q: Optional[np.ndarray] = None
        self.human_zero: Optional[np.ndarray] = None
        self.view_zero: Optional[np.ndarray] = None
        self.skeleton_zero: Optional[np.ndarray] = None
        self.locked_wrist: Optional[np.ndarray] = None
        self.last_valid_target_time: Optional[float] = None
        self.last_print = 0.0
        self.reference_q = self._parse_optional_reference_q(args.robot_reference_q)
        self.latest_grip = 0.0
        self.grip_command = 0.0
        self.grip_open_count = 0
        self.grip_close_count = 0
        self.gripper = None
        self.gripper_ready = False
        self.teach_profile = self._load_teach_profile(args.teach_profile)
        self.teach_limit_text = ""
        self._apply_teach_joint_limits()
        self.palm_zero: Optional[np.ndarray] = None
        self.no_motion_start: Optional[float] = None
        self.no_motion_baseline_q: Optional[np.ndarray] = None
        self.last_no_motion_warning = 0.0

    @staticmethod
    def _parse_indices(text: str) -> List[int]:
        return [int(part.strip()) for part in text.split(",") if part.strip()]

    @staticmethod
    def _parse_optional_reference_q(text: str) -> Optional[np.ndarray]:
        text = text.strip()
        if not text or text.lower() == "current":
            return None
        values = [float(part.strip()) for part in text.split(",") if part.strip()]
        if len(values) != 7:
            raise ValueError("--robot_reference_q must be 'current' or 7 comma-separated joint values")
        return np.array(values, dtype=np.float32)

    @staticmethod
    def _load_teach_profile(path_text: str) -> Optional[dict]:
        path_text = path_text.strip()
        if not path_text:
            return None
        path = Path(path_text).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"--teach_profile not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            profile = json.load(handle)
        if "motions" not in profile:
            raise ValueError(f"invalid teach profile: {path}")
        profile["_path"] = str(path)
        return profile

    def _apply_teach_joint_limits(self) -> None:
        if not self.args.teach_elbow_limits or self.teach_profile is None:
            return
        motions = self.teach_profile.get("motions", {})
        motion = motions.get(self.args.teach_elbow_limit_motion)
        if not motion:
            self.teach_limit_text = f"missing motion {self.args.teach_elbow_limit_motion}"
            return

        elbow_index = RIGHT_ARM_NAMES.index("right_elbow")
        if motion.get("selected_joint") == "right_elbow":
            elbow_min = float(motion["selected_min"])
            elbow_max = float(motion["selected_max"])
        else:
            elbow_min = float(motion["min_q"][elbow_index])
            elbow_max = float(motion["max_q"][elbow_index])

        if elbow_min > elbow_max:
            elbow_min, elbow_max = elbow_max, elbow_min
        margin = max(0.0, float(self.args.teach_elbow_limit_margin))
        old_min, old_max = float(self.args.elbow_min), float(self.args.elbow_max)
        self.args.elbow_min = elbow_min - margin
        self.args.elbow_max = elbow_max + margin
        self.teach_limit_text = (
            f"{self.args.teach_elbow_limit_motion}:"
            f" {old_min:+.2f}..{old_max:+.2f}"
            f" -> {self.args.elbow_min:+.2f}..{self.args.elbow_max:+.2f}"
        )

    def init(self) -> None:
        ChannelFactoryInitialize(self.args.domain_id, self.args.interface)
        if self.args.enter_debug_mode:
            self._enter_debug_mode()
        self.subscriber = ChannelSubscriber(self.args.state_topic, LowState_)
        self.subscriber.Init(self._lowstate_handler, 10)
        if self.args.enable_control:
            self.publisher = ChannelPublisher(self.args.command_topic, LowCmd_)
            self.publisher.Init()
        if self.args.enable_dex1:
            self.gripper = Dex1GripperDDS(
                side=self.args.dex1_side,
                open_q=self.args.dex1_open_q,
                close_q=self.args.dex1_close_q,
                kp=self.args.dex1_kp,
                kd=self.args.dex1_kd,
                max_step=self.args.dex1_max_step,
            )
            self.gripper_ready = self.gripper.wait_state(self.args.dex1_state_timeout)
            if not self.gripper_ready:
                message = (
                    "[R1-A7 CAMERA REAL] Dex1 state not received. "
                    "Start dex1_1_gripper_server and verify rt/dex1/right/state before gripper control."
                )
                if self.args.require_dex1:
                    raise RuntimeError(message)
                print(message)
            elif self.args.enable_control and self.args.dex1_start_open_s > 0.0:
                deadline = time.monotonic() + self.args.dex1_start_open_s
                while time.monotonic() < deadline:
                    self.latest_grip = 0.0
                    self.grip_command = 0.0
                    self.gripper.publish_grip(0.0)
                    time.sleep(0.02)
        self.source.start()

        print("[R1-A7 CAMERA REAL] DDS initialized")
        print("[R1-A7 CAMERA REAL] interface:", self.args.interface)
        print("[R1-A7 CAMERA REAL] state topic:", self.args.state_topic)
        print("[R1-A7 CAMERA REAL] command topic:", self.args.command_topic if self.args.enable_control else "disabled")
        print("[R1-A7 CAMERA REAL] debug lowcmd:", self.args.debug_lowcmd)
        print("[R1-A7 CAMERA REAL] right arm indices:", self.arm_indices)
        if self.args.debug_lowcmd:
            print("[R1-A7 CAMERA REAL] lowcmd hold indices:", self.lowcmd_hold_indices)
        if self.args.r1_arm_sdk:
            print("[R1-A7 CAMERA REAL] R1 arm_sdk mode: mode_pr weight, joints:", self.r1_arm_sdk_indices)
        print("[R1-A7 CAMERA REAL] weight index:", self.weight_index)
        print(
            "[R1-A7 CAMERA REAL] robot reference q:",
            "current" if self.reference_q is None else self.reference_q.tolist(),
        )
        print("[R1-A7 CAMERA REAL] Dex1:", self.args.enable_dex1, self.args.dex1_side)
        print("[R1-A7 CAMERA REAL] human hand:", self.args.human_hand)
        if self.teach_profile is not None:
            print("[R1-A7 CAMERA REAL] teach profile:", self.teach_profile.get("_path"))
            print("[R1-A7 CAMERA REAL] teach profile blend:", self.args.teach_profile_blend)
            if self.args.teach_elbow_limits:
                print("[R1-A7 CAMERA REAL] teach elbow limits:", self.teach_limit_text or "not applied")
        print(
            "[R1-A7 CAMERA REAL] elbow config:"
            f" min={self.args.elbow_min:+.2f}"
            f" max={self.args.elbow_max:+.2f}"
            f" skeleton_sign={self.args.skeleton_elbow_sign:+.1f}"
            f" teach_sign={self.args.teach_elbow_sign:+.1f}"
            f" speed={self.args.elbow_max_speed_rad_s:.2f}"
            f" step={self.args.elbow_direct_max_step:.3f}"
            f" lead={self.args.elbow_max_command_lead:.2f}"
        )
        print("[R1-A7 CAMERA REAL] control:", "ENABLED" if self.args.enable_control else "preview-only")

    def _enter_debug_mode(self) -> None:
        msc = MotionSwitcherClient()
        msc.SetTimeout(2.0)
        msc.Init()
        status, result = msc.CheckMode()
        print(f"[R1-A7 CAMERA REAL] motion_switcher CheckMode: status={status} result={result}")
        try:
            while result and result.get("name"):
                print("[R1-A7 CAMERA REAL] releasing active mode:", result)
                msc.ReleaseMode()
                time.sleep(0.5)
                status, result = msc.CheckMode()
                print(f"[R1-A7 CAMERA REAL] motion_switcher CheckMode: status={status} result={result}")
        except Exception as exc:
            print(f"[R1-A7 CAMERA REAL] failed to enter debug mode: {exc}")

    def _lowstate_handler(self, msg: LowState_) -> None:
        self.low_state = msg
        if self.first_state_time is None:
            self.first_state_time = time.monotonic()

    def _read_arm(self) -> Optional[ArmState]:
        if self.low_state is None:
            return None
        motor_state = self.low_state.motor_state
        max_idx = max(max(self.arm_indices), self.weight_index)
        if len(motor_state) <= max_idx:
            raise RuntimeError(f"lowstate has {len(motor_state)} motors, requested index {max_idx}")
        q = np.array([motor_state[i].q for i in self.arm_indices], dtype=np.float32)
        dq = np.array([motor_state[i].dq for i in self.arm_indices], dtype=np.float32)
        return ArmState(q=q, dq=dq, stamp=time.monotonic())

    def _init_low_cmd_stop(self) -> None:
        for motor in self.low_cmd.motor_cmd:
            motor.tau = 0.0
            motor.q = 0.0
            motor.dq = 0.0
            motor.kp = 0.0
            motor.kd = 0.0

    @staticmethod
    def _wrap_angles(angles: np.ndarray) -> np.ndarray:
        return np.remainder(angles + np.pi, 2.0 * np.pi) - np.pi

    @staticmethod
    def _apply_deadband(value: float, deadband: float) -> float:
        deadband = max(0.0, float(deadband))
        value = float(value)
        if abs(value) <= deadband:
            return 0.0
        return math.copysign(abs(value) - deadband, value)

    @staticmethod
    def _human_joint_features(target) -> Optional[np.ndarray]:
        if target.shoulder_m is None or target.elbow_m is None or target.wrist_m is None:
            return None
        shoulder = np.asarray(target.shoulder_m, dtype=np.float32)
        elbow = np.asarray(target.elbow_m, dtype=np.float32)
        wrist = np.asarray(target.wrist_m, dtype=np.float32)
        upper_raw = elbow - shoulder
        forearm_raw = wrist - elbow
        upper = np.array([upper_raw[0], upper_raw[1], -upper_raw[2]], dtype=np.float32)
        forearm = np.array([forearm_raw[0], forearm_raw[1], -forearm_raw[2]], dtype=np.float32)
        upper_norm = max(float(np.linalg.norm(upper)), 1e-6)
        forearm_norm = max(float(np.linalg.norm(forearm)), 1e-6)
        upper_dir = upper / upper_norm
        elbow_cos = float(np.clip(np.dot(upper, forearm) / (upper_norm * forearm_norm), -1.0, 1.0))
        elbow_bend = math.pi - math.acos(elbow_cos)
        lift = -upper_dir[1]
        side = upper_dir[0]
        reach = -upper_dir[2]
        return np.array([lift, side, reach, elbow_bend], dtype=np.float32)

    def _teach_delta_for_motion(self, motion_name: str, scalar: float, positive_branch: str = "max") -> np.ndarray:
        if self.teach_profile is None or self.home_q is None:
            return np.zeros(7, dtype=np.float32)
        motion = self.teach_profile.get("motions", {}).get(motion_name)
        if not motion:
            return np.zeros(7, dtype=np.float32)
        scalar = float(np.clip(scalar, -1.0, 1.0))
        if abs(scalar) < 1e-6:
            return np.zeros(7, dtype=np.float32)
        min_delta = np.asarray(motion["min_delta"], dtype=np.float32)
        max_delta = np.asarray(motion["max_delta"], dtype=np.float32)
        if positive_branch == "min":
            pos_delta, neg_delta = min_delta, max_delta
        else:
            pos_delta, neg_delta = max_delta, min_delta
        return (pos_delta * scalar) if scalar >= 0.0 else (neg_delta * -scalar)

    def _apply_teach_profile(
        self,
        q: np.ndarray,
        horizontal: float,
        reach: float,
        skeleton_features: Optional[np.ndarray],
        target,
    ) -> tuple[np.ndarray, str]:
        if self.teach_profile is None or self.home_q is None or self.args.teach_profile_blend <= 0.0:
            return q, ""

        teach_delta = np.zeros(7, dtype=np.float32)
        parts = []
        mapping = self.teach_profile.get("camera_mapping", {})

        reach_cfg = mapping.get("reach", {})
        reach_scale = max(1e-3, float(reach_cfg.get("input_scale", self.args.teach_reach_scale)))
        reach_scalar = float(np.clip(reach / reach_scale, -1.0, 1.0))
        motion_name = reach_cfg.get("motion", "shoulder_pitch_forward_back")
        teach_delta += self._teach_delta_for_motion(
            motion_name,
            reach_scalar,
            reach_cfg.get("positive_branch", "min"),
        )
        parts.append(f"reach={reach_scalar:+.2f}")

        roll_cfg = mapping.get("horizontal_roll", {})
        horizontal_scale = max(1e-3, float(self.args.teach_horizontal_scale))
        horizontal_scalar = float(
            np.clip(self.args.teach_horizontal_roll_sign * horizontal / horizontal_scale, -1.0, 1.0)
        )
        motion_name = roll_cfg.get("motion", "shoulder_roll_out_in")
        roll_weight = float(self.args.teach_horizontal_roll_weight)
        if roll_weight > 0.0:
            teach_delta += roll_weight * self._teach_delta_for_motion(
                motion_name,
                horizontal_scalar,
                roll_cfg.get("positive_branch", "min"),
            )
        parts.append(f"roll={horizontal_scalar:+.2f}")

        yaw_cfg = mapping.get("horizontal_yaw", {})
        yaw_scale = max(1e-3, float(self.args.teach_horizontal_yaw_scale))
        yaw_scalar = float(np.clip(self.args.teach_horizontal_yaw_sign * horizontal / yaw_scale, -1.0, 1.0))
        yaw_weight = float(self.args.teach_horizontal_yaw_weight)
        if yaw_weight > 0.0:
            teach_delta += yaw_weight * self._teach_delta_for_motion(
                yaw_cfg.get("motion", "shoulder_yaw_twist"),
                yaw_scalar,
                yaw_cfg.get("positive_branch", "max"),
            )
        parts.append(f"yaw={yaw_scalar:+.2f}")

        if skeleton_features is not None:
            elbow_cfg = mapping.get("elbow_bend", {})
            elbow_scale = max(1e-3, float(elbow_cfg.get("input_scale", self.args.teach_elbow_scale)))
            elbow_scalar = float(np.clip(self.args.teach_elbow_sign * float(skeleton_features[3]) / elbow_scale, -1.0, 1.0))
            motion_name = elbow_cfg.get("motion", "elbow_bend_extend")
            teach_delta += self._teach_delta_for_motion(
                motion_name,
                elbow_scalar,
                elbow_cfg.get("positive_branch", "max"),
            )
            parts.append(f"elbow={elbow_scalar:+.2f}")

        if self.args.teach_wrist_profile and target.palm_angles_rad is not None:
            palm = np.asarray(target.palm_angles_rad, dtype=np.float32)
            if self.palm_zero is None:
                self.palm_zero = palm.copy()
                print("[R1-A7 CAMERA REAL] calibrated palm zero:", self.palm_zero.tolist())
            palm_delta = self._wrap_angles(palm - self.palm_zero)
            wrist_keys = [
                ("wrist_roll", 0, self.args.teach_palm_roll_scale, "wrist_roll_rotate"),
                ("wrist_pitch", 1, self.args.teach_palm_pitch_scale, "wrist_pitch_up_down"),
                ("wrist_yaw", 2, self.args.teach_palm_yaw_scale, "wrist_yaw_left_right"),
            ]
            for key, idx, default_scale, default_motion in wrist_keys:
                cfg = mapping.get(key, {})
                scale = max(1e-3, float(cfg.get("input_scale", default_scale)))
                scalar = float(np.clip(float(palm_delta[idx]) / scale, -1.0, 1.0))
                teach_delta += self._teach_delta_for_motion(
                    cfg.get("motion", default_motion),
                    scalar,
                    cfg.get("positive_branch", "max"),
                )
                parts.append(f"{key}={scalar:+.2f}")

        teach_q = self.home_q + teach_delta
        blend = float(np.clip(self.args.teach_profile_blend, 0.0, 1.0))
        out = (1.0 - blend) * q + blend * teach_q
        return out, " teach=(" + ",".join(parts) + ")"

    def _target_from_camera(self, state: ArmState) -> tuple[Optional[np.ndarray], str]:
        target = self.source.get_latest(self.args.max_age_s)
        if target is None or not target.valid:
            return None, "no_target"

        if self.home_q is None:
            self.home_q = state.q.copy() if self.reference_q is None else self.reference_q.copy()
            self.command_q = state.q.copy()
            self.locked_wrist = state.q[4:7].copy()
            print("[R1-A7 CAMERA REAL] current robot q:", state.q.tolist())
            print("[R1-A7 CAMERA REAL] calibrated robot reference:", self.home_q.tolist())
            if self.args.lock_wrist:
                print("[R1-A7 CAMERA REAL] locked wrist:", self.locked_wrist.tolist())

        if self.human_zero is None:
            self.human_zero = target.wrist_rel_m.copy()
            print("[R1-A7 CAMERA REAL] calibrated human zero:", self.human_zero.tolist())
        if target.wrist_rel_view is not None and self.view_zero is None:
            self.view_zero = target.wrist_rel_view.copy()
            print("[R1-A7 CAMERA REAL] calibrated view zero:", self.view_zero.tolist())

        human_delta = target.wrist_rel_m - self.human_zero
        if float(np.linalg.norm(human_delta)) > self.args.max_human_delta_m:
            return None, "human_delta_too_large"

        view_delta = None
        if target.wrist_rel_view is not None and self.view_zero is not None:
            view_delta = target.wrist_rel_view - self.view_zero

        horizontal = float(human_delta[0])
        vertical = float(human_delta[1])
        if view_delta is not None:
            if self.args.direct_view_horizontal:
                horizontal = float(view_delta[0])
            if self.args.direct_view_vertical:
                vertical = float(view_delta[1])
        if self.args.mirror_input:
            horizontal = -horizontal
        vertical *= self.args.vertical_sign

        depth = float(human_delta[2])
        reach = self.args.depth_sign * depth
        horizontal = self._apply_deadband(horizontal, self.args.input_deadband)
        vertical = self._apply_deadband(vertical, self.args.input_deadband)
        reach = self._apply_deadband(reach, self.args.depth_deadband)
        horizontal *= max(0.0, float(self.args.horizontal_amplitude_scale))
        q = self.home_q.copy()

        pitch_sign = self.args.shoulder_pitch_sign
        direct_q = self.home_q.copy()
        direct_q[0] += pitch_sign * self.args.pitch_gain * vertical
        direct_q[1] += 0.45 * self.args.roll_gain * horizontal
        direct_q[2] -= self.args.yaw_gain * horizontal
        direct_q[3] += self.args.elbow_vertical_ratio * self.args.elbow_gain * vertical
        direct_q[0] += (
            pitch_sign
            * self.args.depth_pitch_sign
            * self.args.pitch_depth_ratio
            * self.args.depth_gain
            * reach
        )
        direct_q[3] += self.args.elbow_depth_ratio * self.args.depth_gain * reach
        q = direct_q.copy()

        features = self._human_joint_features(target)
        skeleton_features = None
        feature_text = ""
        if features is not None:
            if self.skeleton_zero is None:
                self.skeleton_zero = features.copy()
                print("[R1-A7 CAMERA REAL] calibrated skeleton zero:", self.skeleton_zero.tolist())
            sk = np.clip(features - self.skeleton_zero, -1.4, 1.4)
            sk = np.array(
                [self._apply_deadband(value, self.args.skeleton_deadband) for value in sk],
                dtype=np.float32,
            )
            lift, side_feature, skeleton_reach, elbow_bend = sk
            skeleton_features = sk
            skeleton_q = self.home_q.copy()
            skeleton_q[0] += (
                pitch_sign
                * self.args.skeleton_lift_sign
                * self.args.skeleton_lift_gain
                * self.args.vertical_sign
                * lift
            )
            skeleton_q[0] += pitch_sign * self.args.skeleton_reach_pitch_gain * skeleton_reach
            skeleton_q[1] += self.args.skeleton_side_roll_gain * side_feature
            skeleton_q[2] -= (
                self.args.skeleton_reach_yaw_gain * skeleton_reach
                + self.args.skeleton_side_yaw_gain * side_feature
            )
            skeleton_q[3] += self.args.skeleton_elbow_sign * self.args.skeleton_elbow_gain * elbow_bend
            skeleton_q[3] += self.args.skeleton_lift_elbow_gain * self.args.vertical_sign * lift
            if self.args.coordination_mode == "anatomic":
                blend = float(np.clip(self.args.skeleton_blend, 0.0, 1.0))
                q = (1.0 - blend) * direct_q + blend * skeleton_q
                q[3] = (1.0 - self.args.elbow_skeleton_blend) * direct_q[3] + self.args.elbow_skeleton_blend * skeleton_q[3]
            else:
                q = direct_q + (skeleton_q - self.home_q)
            feature_text = (
                f" skel=({lift:+.3f},{side_feature:+.3f},{skeleton_reach:+.3f},{elbow_bend:+.3f})"
                f" mode={self.args.coordination_mode}"
            )

        q, teach_text = self._apply_teach_profile(q, horizontal, reach, skeleton_features, target)
        feature_text += teach_text

        q = self.home_q + self.args.amplitude_scale * (q - self.home_q)
        raw_q = q.copy()
        q[0] = float(np.clip(q[0], self.args.pitch_min, self.args.pitch_max))
        q[1] = float(np.clip(q[1], self.args.roll_min, self.args.roll_max))
        q[2] = float(np.clip(q[2], self.args.yaw_min, self.args.yaw_max))
        q[3] = float(np.clip(q[3], self.args.elbow_min, self.args.elbow_max))
        saturated = []
        for name, raw, clipped in zip(RIGHT_ARM_NAMES[:4], raw_q[:4], q[:4]):
            if abs(float(raw) - float(clipped)) > 1e-4:
                saturated.append(f"{name}:{raw:+.2f}->{clipped:+.2f}")
        if self.args.lock_wrist and self.locked_wrist is not None:
            q[4:7] = self.locked_wrist
        else:
            q[4:7] = state.q[4:7]

        info = (
            f"in=({horizontal:+.3f},{vertical:+.3f},depth={depth:+.3f},reach={reach:+.3f})"
            f"{feature_text}"
            f" SAT={';'.join(saturated) if saturated else 'none'}"
        )
        grip = float(np.clip(target.grip, 0.0, 1.0))
        if self.args.dex1_binary:
            if grip >= self.args.dex1_grip_close_threshold:
                self.grip_close_count += 1
                self.grip_open_count = 0
            elif grip <= self.args.dex1_grip_open_threshold:
                self.grip_open_count += 1
                self.grip_close_count = 0
            else:
                self.grip_open_count = 0
                self.grip_close_count = 0
            if self.grip_close_count >= self.args.dex1_grip_close_frames:
                self.grip_command = 1.0
            elif self.grip_open_count >= self.args.dex1_grip_open_frames:
                self.grip_command = 0.0
            grip = self.grip_command
        else:
            grip = float(np.clip(grip ** self.args.dex1_grip_gamma, 0.0, 1.0))
        self.latest_grip = grip
        self.last_valid_target_time = time.monotonic()
        return q, info

    def _step_command(self, state: ArmState, target_q: Optional[np.ndarray], dt: float) -> np.ndarray:
        if self.command_q is None:
            self.command_q = state.q.copy()
        if target_q is None:
            if self.last_valid_target_time is None:
                return self.command_q.copy()
            lost_s = time.monotonic() - self.last_valid_target_time
            if lost_s <= self.args.lost_hold_s:
                return self.command_q.copy()
            if self.home_q is None:
                return self.command_q.copy()
            target_q = self.home_q

        max_delta = max(0.0, min(self.args.direct_max_step, self.args.max_speed_rad_s * max(dt, 1e-3)))
        max_delta_vec = np.full_like(self.command_q, max_delta)
        shoulder_delta = max(
            max_delta,
            min(self.args.shoulder_direct_max_step, self.args.shoulder_max_speed_rad_s * max(dt, 1e-3)),
        )
        max_delta_vec[0:3] = shoulder_delta
        max_delta_vec[3] = max(
            max_delta,
            min(self.args.elbow_direct_max_step, self.args.elbow_max_speed_rad_s * max(dt, 1e-3)),
        )
        step = np.clip(target_q - self.command_q, -max_delta_vec, max_delta_vec)
        next_q = self.command_q + step
        lead_limit = np.full_like(self.command_q, self.args.max_command_lead)
        lead_limit[0:3] = self.args.shoulder_max_command_lead
        lead_limit[3] = self.args.elbow_max_command_lead
        lead = np.clip(next_q - state.q, -lead_limit, lead_limit)
        self.command_q = state.q + lead
        return self.command_q.copy()

    def _publish(self, command_q: np.ndarray) -> None:
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
        if self.args.debug_lowcmd:
            # rt/lowcmd does not use the arm_sdk transition-weight slot.
            pass
        elif self.args.r1_arm_sdk and self.low_state is not None:
            if hasattr(self.low_cmd, "mode_pr"):
                self.low_cmd.mode_pr = int(np.clip(self.args.weight, 0.0, 1.0) * 100.0)
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
            self.low_cmd.motor_cmd[self.weight_index].q = float(np.clip(self.args.weight, 0.0, 1.0))
        for idx, q in zip(self.arm_indices, command_q):
            if self.args.r1_arm_sdk and not self.args.debug_lowcmd and idx not in self.r1_arm_sdk_indices:
                continue
            motor = self.low_cmd.motor_cmd[idx]
            motor.mode = 1
            motor.tau = 0.0
            motor.q = float(q)
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

    def _publish_gripper(self) -> Optional[float]:
        if self.gripper is None:
            return None
        if not self.gripper_ready:
            self.gripper_ready = self.gripper.wait_state(0.0)
            if not self.gripper_ready:
                return None
        return self.gripper.publish_grip(self.latest_grip)

    def _gripper_state_q(self) -> Optional[float]:
        if self.gripper is None or self.gripper.state is None:
            return None
        return float(self.gripper.state.q)

    def _release(self) -> None:
        if self.gripper is not None and self.gripper_ready and self.args.dex1_exit_open_s > 0.0:
            deadline = time.monotonic() + self.args.dex1_exit_open_s
            while time.monotonic() < deadline:
                self.latest_grip = 0.0
                self.grip_command = 0.0
                self.gripper.publish_grip(0.0)
                time.sleep(0.02)
            print("[R1-A7 CAMERA REAL] released Dex1 to open")
        if self.publisher is None:
            return
        self._init_low_cmd_stop()
        if self.args.r1_arm_sdk and not self.args.debug_lowcmd and hasattr(self.low_cmd, "mode_pr"):
            self.low_cmd.mode_pr = 0
        else:
            self.low_cmd.motor_cmd[self.weight_index].q = 0.0
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)
        print("[R1-A7 CAMERA REAL] released arm_sdk weight")

    def run(self) -> int:
        self.init()
        deadline = time.monotonic() + max(0.0, self.args.duration)
        last_loop = time.monotonic()
        print("[R1-A7 CAMERA REAL] waiting for robot lowstate and camera target ...")
        try:
            while not self.done:
                now = time.monotonic()
                if self.args.duration > 0 and now >= deadline:
                    break
                state = self._read_arm()
                if state is None:
                    time.sleep(0.02)
                    continue

                dt = now - last_loop
                last_loop = now
                target_q, info = self._target_from_camera(state)
                if target_q is None and not getattr(self.source, "_running", True):
                    print("[R1-A7 CAMERA REAL] camera source stopped; aborting")
                    return 3
                if (
                    self.args.dex1_open_on_lost_s >= 0.0
                    and target_q is None
                    and self.last_valid_target_time is not None
                    and now - self.last_valid_target_time >= self.args.dex1_open_on_lost_s
                ):
                    self.latest_grip = 0.0
                    self.grip_command = 0.0
                command_q = self._step_command(state, target_q, dt)
                if self.args.enable_control:
                    self._publish(command_q)
                    gripper_cmd = self._publish_gripper()
                    if target_q is not None:
                        cmd_error = float(np.max(np.abs(command_q[:4] - state.q[:4])))
                        if cmd_error >= self.args.no_motion_command_error:
                            if self.no_motion_start is None:
                                self.no_motion_start = now
                                self.no_motion_baseline_q = state.q.copy()
                            baseline = self.no_motion_baseline_q if self.no_motion_baseline_q is not None else state.q
                            moved = float(np.max(np.abs(state.q[:4] - baseline[:4])))
                            if (
                                now - self.no_motion_start >= self.args.no_motion_warn_s
                                and moved <= self.args.no_motion_joint_delta
                                and now - self.last_no_motion_warning >= self.args.no_motion_warn_period
                            ):
                                self.last_no_motion_warning = now
                                print(
                                    "[R1-A7 CAMERA REAL] WARNING: command is changing but lowstate is not moving. "
                                    "Robot is probably ignoring this command topic or arm motors are not enabled. "
                                    f"topic={self.args.command_topic} debug_lowcmd={self.args.debug_lowcmd} "
                                    f"cmd_error={cmd_error:.3f} moved={moved:.3f}"
                                )
                        else:
                            self.no_motion_start = None
                            self.no_motion_baseline_q = None
                else:
                    gripper_cmd = None

                if now - self.last_print >= self.args.print_period:
                    self.last_print = now
                    current = " ".join(f"{name}={value:+.3f}" for name, value in zip(RIGHT_ARM_NAMES[:4], state.q[:4]))
                    command = " ".join(f"{name}={value:+.3f}" for name, value in zip(RIGHT_ARM_NAMES[:4], command_q[:4]))
                    if target_q is not None:
                        target_text = " ".join(
                            f"{name}={value:+.3f}" for name, value in zip(RIGHT_ARM_NAMES[:4], target_q[:4])
                        )
                    else:
                        target_text = "none"
                    print(
                        f"[R1-A7 CAMERA REAL] target={'yes' if target_q is not None else 'no '} "
                        f"{info} current: {current} cmd: {command}"
                        f" tgt: {target_text}"
                        f" grip={self.latest_grip:.2f}"
                        f" dex1_cmd={gripper_cmd if gripper_cmd is not None else 'none'}"
                        f" dex1_state={self._gripper_state_q() if self._gripper_state_q() is not None else 'none'}"
                    )
                    if self.args.debug_pose and target_q is None:
                        reason, counts = self.source.get_debug_snapshot()
                        print(f"[R1-A7 CAMERA REAL] no-target reason={reason} counts={counts}")

                time.sleep(max(0.0, 1.0 / max(1.0, self.args.hz)))
        finally:
            self._release()
            self.source.stop()

        if self.first_state_time is None:
            print("[R1-A7 CAMERA REAL] no lowstate received")
            return 2
        return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real R1-A7 right-arm camera teleoperation")
    parser.add_argument("--interface", default="enx9c69d37d0967")
    parser.add_argument("--domain_id", type=int, default=0)
    parser.add_argument("--state_topic", default="rt/lowstate")
    parser.add_argument("--command_topic", default="rt/arm_sdk")
    parser.add_argument("--enable_control", action="store_true")
    parser.add_argument("--enable_dex1", action="store_true", help="publish camera grip to Dex1_1 DDS gripper")
    parser.add_argument("--require_dex1", action="store_true", help="abort if Dex1_1 state is not received")
    parser.add_argument("--debug_lowcmd", action="store_true", help="publish direct debug commands on rt/lowcmd")
    parser.add_argument("--r1_arm_sdk", action="store_true", help="use R1 official rt/arm_sdk format: mode_pr is weight")
    parser.add_argument("--enter_debug_mode", action="store_true", help="release active motion mode with motion_switcher")
    parser.add_argument("--duration", type=float, default=0.0, help="seconds; 0 means forever")
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--print_period", type=float, default=0.5)
    parser.add_argument("--right_arm_indices", default="22,23,24,25,26,27,28")
    parser.add_argument("--r1_arm_sdk_indices", default="15,16,17,18,19,22,23,24,25,26,13,29,30")
    parser.add_argument("--lowcmd_hold_indices", default="22,23,24,25,26,27,28")
    parser.add_argument(
        "--robot_reference_q",
        default="current",
        help=(
            "right-arm reference posture used as teleop zero. Use 'current' or "
            "pitch,roll,yaw,elbow,wrist_roll,wrist_pitch,wrist_yaw"
        ),
    )
    parser.add_argument(
        "--weight_index",
        type=int,
        default=31,
        help="arm_sdk weight motor index. R1-A/H2-style 7-DoF arms use 31; G1-style uses 29.",
    )
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--kp", type=float, default=20.0)
    parser.add_argument("--kd", type=float, default=1.0)
    parser.add_argument("--hold_kp", type=float, default=20.0, help="debug_lowcmd hold gain for non-commanded joints")
    parser.add_argument("--hold_kd", type=float, default=1.0, help="debug_lowcmd hold damping for non-commanded joints")
    parser.add_argument("--max_speed_rad_s", type=float, default=0.25)
    parser.add_argument("--max_command_lead", type=float, default=0.25)
    parser.add_argument("--shoulder_max_speed_rad_s", type=float, default=0.45)
    parser.add_argument("--shoulder_direct_max_step", type=float, default=0.080)
    parser.add_argument("--shoulder_max_command_lead", type=float, default=0.55)
    parser.add_argument("--elbow_max_speed_rad_s", type=float, default=0.45)
    parser.add_argument("--elbow_direct_max_step", type=float, default=0.080)
    parser.add_argument("--elbow_max_command_lead", type=float, default=0.55)
    parser.add_argument("--teach_profile", default="", help="JSON profile generated from R1-A7 teach-in data")
    parser.add_argument("--teach_profile_blend", type=float, default=0.0, help="0 disables teach profile; 1 uses it fully")
    parser.add_argument(
        "--teach_elbow_limits",
        action="store_true",
        help="derive elbow min/max from the recorded teach-profile elbow motion",
    )
    parser.add_argument("--no_teach_elbow_limits", action="store_false", dest="teach_elbow_limits")
    parser.set_defaults(teach_elbow_limits=False)
    parser.add_argument("--teach_elbow_limit_motion", default="elbow_bend_extend")
    parser.add_argument("--teach_elbow_limit_margin", type=float, default=0.05)
    parser.add_argument("--teach_reach_scale", type=float, default=0.45)
    parser.add_argument("--teach_horizontal_scale", type=float, default=0.35)
    parser.add_argument("--teach_horizontal_roll_weight", type=float, default=0.25)
    parser.add_argument("--teach_horizontal_yaw_scale", type=float, default=0.30)
    parser.add_argument("--teach_horizontal_yaw_weight", type=float, default=0.45)
    parser.add_argument(
        "--teach_horizontal_roll_sign",
        type=float,
        choices=[-1.0, 1.0],
        default=1.0,
        help="flip camera horizontal input before applying teach-profile shoulder-roll motion",
    )
    parser.add_argument(
        "--teach_horizontal_yaw_sign",
        type=float,
        choices=[-1.0, 1.0],
        default=1.0,
        help="flip camera horizontal input before applying teach-profile shoulder-yaw twist motion",
    )
    parser.add_argument("--teach_elbow_scale", type=float, default=1.20)
    parser.add_argument(
        "--teach_elbow_sign",
        type=float,
        choices=[-1.0, 1.0],
        default=-1.0,
        help="flip camera elbow feature before applying teach-profile elbow motion",
    )
    parser.add_argument("--teach_wrist_profile", action="store_true", help="map palm orientation to wrist joints from teach profile")
    parser.add_argument("--teach_palm_roll_scale", type=float, default=1.00)
    parser.add_argument("--teach_palm_pitch_scale", type=float, default=0.80)
    parser.add_argument("--teach_palm_yaw_scale", type=float, default=0.80)

    parser.add_argument("--human_hand", choices=["left", "right"], default="right")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--mirror_view", action="store_true")
    parser.add_argument("--mirror_input", action="store_true")
    parser.add_argument("--debug_pose", action="store_true")
    parser.add_argument("--camera_filter_alpha", type=float, default=0.25)
    parser.add_argument("--max_age_s", type=float, default=0.35)
    parser.add_argument("--min_visibility", type=float, default=0.05)
    parser.add_argument("--min_wrist_shoulder_m", type=float, default=0.025)
    parser.add_argument("--max_human_delta_m", type=float, default=1.20)
    parser.add_argument("--direct_view_vertical", action="store_true")
    parser.add_argument("--direct_view_horizontal", action="store_true")
    parser.add_argument(
        "--coordination_mode",
        choices=["additive", "anatomic"],
        default="additive",
        help="additive keeps legacy wrist+skeleton sum; anatomic blends wrist displacement with upper-arm geometry",
    )
    parser.add_argument("--skeleton_blend", type=float, default=0.65, help="anatomic mode blend for shoulder joints")
    parser.add_argument("--elbow_skeleton_blend", type=float, default=0.85, help="anatomic mode blend for elbow joint")
    parser.add_argument("--input_deadband", type=float, default=0.010, help="ignore small camera X/Y motion")
    parser.add_argument("--horizontal_amplitude_scale", type=float, default=1.0, help="multiply only camera left/right motion")
    parser.add_argument("--depth_deadband", type=float, default=0.020, help="ignore small camera depth motion")
    parser.add_argument("--skeleton_deadband", type=float, default=0.020, help="ignore small skeleton feature changes")
    parser.add_argument("--vertical_sign", type=float, default=1.0)
    parser.add_argument("--depth_sign", type=float, default=-1.0)
    parser.add_argument("--lock_wrist", action="store_true", default=True)
    parser.add_argument("--unlock_wrist", action="store_false", dest="lock_wrist")
    parser.add_argument("--lost_hold_s", type=float, default=0.55)
    parser.add_argument("--dex1_side", choices=["left", "right"], default="right")
    parser.add_argument("--dex1_open_q", type=float, default=5.40)
    parser.add_argument("--dex1_close_q", type=float, default=0.0)
    parser.add_argument("--dex1_kp", type=float, default=5.0)
    parser.add_argument("--dex1_kd", type=float, default=0.05)
    parser.add_argument("--dex1_max_step", type=float, default=0.18)
    parser.add_argument("--dex1_start_open_s", type=float, default=0.0)
    parser.add_argument("--dex1_exit_open_s", type=float, default=0.0)
    parser.add_argument(
        "--dex1_open_on_lost_s",
        type=float,
        default=1.2,
        help="open Dex1_1 after this many seconds without a valid camera target; negative disables it",
    )
    parser.add_argument("--dex1_state_timeout", type=float, default=1.0)
    parser.add_argument("--dex1_binary", action="store_true", help="convert camera grip to open/close")
    parser.add_argument("--dex1_grip_threshold", type=float, default=0.55, help="legacy single threshold")
    parser.add_argument("--dex1_grip_open_threshold", type=float, default=0.38)
    parser.add_argument("--dex1_grip_close_threshold", type=float, default=0.68)
    parser.add_argument("--dex1_grip_open_frames", type=int, default=2)
    parser.add_argument("--dex1_grip_close_frames", type=int, default=3)
    parser.add_argument("--dex1_grip_gamma", type=float, default=1.0, help="continuous grip response curve")

    parser.add_argument("--roll_gain", type=float, default=1.8)
    parser.add_argument("--yaw_gain", type=float, default=1.4)
    parser.add_argument("--pitch_gain", type=float, default=4.0)
    parser.add_argument(
        "--shoulder_pitch_sign",
        type=float,
        choices=[-1.0, 1.0],
        default=1.0,
        help="flip shoulder-pitch response if upper-arm lift moves in the wrong direction",
    )
    parser.add_argument("--elbow_gain", type=float, default=1.0)
    parser.add_argument("--amplitude_scale", type=float, default=1.0, help="multiply all camera-induced joint deltas")
    parser.add_argument("--depth_gain", type=float, default=1.8)
    parser.add_argument(
        "--depth_pitch_sign",
        type=float,
        choices=[-1.0, 1.0],
        default=-1.0,
        help="shoulder-pitch sign for forward/back reach; R1-A7 forward reach needs negative pitch",
    )
    parser.add_argument("--pitch_depth_ratio", type=float, default=0.35)
    parser.add_argument("--elbow_depth_ratio", type=float, default=-0.85)
    parser.add_argument("--elbow_vertical_ratio", type=float, default=0.15)
    parser.add_argument("--skeleton_lift_gain", type=float, default=1.10)
    parser.add_argument(
        "--skeleton_lift_sign",
        type=float,
        choices=[-1.0, 1.0],
        default=-1.0,
        help="upper-arm lift sign; this R1-A7 needs -1 because positive shoulder pitch moves backward",
    )
    parser.add_argument("--skeleton_side_roll_gain", type=float, default=0.95)
    parser.add_argument("--skeleton_side_yaw_gain", type=float, default=0.45)
    parser.add_argument("--skeleton_reach_pitch_gain", type=float, default=0.25)
    parser.add_argument("--skeleton_reach_yaw_gain", type=float, default=0.35)
    parser.add_argument("--skeleton_elbow_gain", type=float, default=0.85)
    parser.add_argument(
        "--skeleton_elbow_sign",
        type=float,
        choices=[-1.0, 1.0],
        default=-1.0,
        help="flip skeleton elbow-bend feature if human bending drives robot elbow straight",
    )
    parser.add_argument("--skeleton_lift_elbow_gain", type=float, default=0.20)
    parser.add_argument("--direct_max_step", type=float, default=0.040)
    parser.add_argument("--no_motion_warn_s", type=float, default=2.0)
    parser.add_argument("--no_motion_warn_period", type=float, default=3.0)
    parser.add_argument("--no_motion_command_error", type=float, default=0.25)
    parser.add_argument("--no_motion_joint_delta", type=float, default=0.03)

    parser.add_argument("--pitch_min", type=float, default=-0.8)
    parser.add_argument("--pitch_max", type=float, default=1.2)
    parser.add_argument("--roll_min", type=float, default=-0.85)
    parser.add_argument("--roll_max", type=float, default=0.35)
    parser.add_argument("--yaw_min", type=float, default=-1.15)
    parser.add_argument("--yaw_max", type=float, default=1.15)
    parser.add_argument("--elbow_min", type=float, default=0.05)
    parser.add_argument("--elbow_max", type=float, default=1.55)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.enable_control:
        print("WARNING: This will publish rt/arm_sdk commands to the real R1-A right arm.")
        print("Keep the emergency stop ready and clear the arm workspace.")
        if not sys.stdin.isatty():
            print("[R1-A7 CAMERA REAL] control requires an interactive terminal")
            return 2
        answer = input("Type ENABLE to continue: ").strip()
        if answer != "ENABLE":
            print("[R1-A7 CAMERA REAL] aborted")
            return 2

    node = R1A7CameraRealTeleop(args)

    def _handle_signal(_signum, _frame):
        node.done = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    return node.run()


if __name__ == "__main__":
    raise SystemExit(main())
