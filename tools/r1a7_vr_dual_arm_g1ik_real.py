#!/usr/bin/env python3
"""Real R1-A7 dual-arm VR control using Unitree G1_29 official IK.

Quest/Vuer supplies controller wrist poses. Unitree's G1_29_ArmIK solves the
14 arm joints, then the result is mapped to the R1-A7 7-DoF arm motor indices
and published through rt/lowcmd with conservative rate and lead limits.

When --enable_gripper is supplied, the Quest left/right index triggers control
the matching grippers. Newer R1-A7 internal-wiring grippers can be driven
directly as lowcmd motors 31 and 33. The older external Dex1 service path can
still be selected with --gripper_mode dex1_dds.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import sys
import time
import subprocess
import threading
from multiprocessing import Array, Lock, Value
from pathlib import Path
from typing import Optional

import numpy as np

XR_TELEOP = Path(os.getenv("XR_TELEOP_ROOT", "/home/robot/unitree_sdk2-main_code_only_20260824/xr_teleoperate"))
XR_TELEOP_TELEOP = XR_TELEOP / "teleop"
XR_TELEOP_TV_SRC = XR_TELEOP_TELEOP / "televuer" / "src"
for path in (XR_TELEOP_TV_SRC, XR_TELEOP_TELEOP, XR_TELEOP):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from robot_control.robot_arm_ik import G1_29_ArmIK  # noqa: E402
from televuer import TeleVuerWrapper  # noqa: E402
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

GRIPPER_NAMES = ["left_gripper", "right_gripper"]

R1A7_ARM_LIMITS = np.array(
    [
        [-3.1416, 2.0944],  # left_shoulder_pitch
        [-0.2269, 2.4784],  # left_shoulder_roll
        [-1.9199, 1.9199],  # left_shoulder_yaw
        [-0.9757, 2.1850],  # left_elbow
        [-1.9199, 1.9199],  # left_wrist_roll
        [-1.61429558, 1.61429558],  # left_wrist_pitch
        [-1.61429558, 1.61429558],  # left_wrist_yaw
        [-3.1416, 2.0944],  # right_shoulder_pitch
        [-2.4784, 0.2269],  # right_shoulder_roll
        [-1.9199, 1.9199],  # right_shoulder_yaw
        [-0.9757, 2.1850],  # right_elbow
        [-1.9199, 1.9199],  # right_wrist_roll
        [-1.61429558, 1.61429558],  # right_wrist_pitch
        [-1.61429558, 1.61429558],  # right_wrist_yaw
    ],
    dtype=float,
)

R1A7_A7_URDF = "/home/robot/unitree_ros_official_latest/robots/r1_a7_description/r1_a7.urdf"

# R1-A7 internal two-finger grippers on LowCmd/LowState motors 31 and 33.
# The real robot reports the fully/open end around q ~= 4.8, so use absolute
# motor-q coordinates here instead of startup-relative offsets. Quest trigger
# convention remains: released=10 -> OPEN, fully pulled=0 -> CLOSE.
R1A7_GRIPPER_INDICES = "31,33"
R1A7_GRIPPER_RELATIVE = False
R1A7_GRIPPER_LEFT_OPEN_Q = 4.86
R1A7_GRIPPER_LEFT_CLOSE_Q = -0.08
R1A7_GRIPPER_RIGHT_OPEN_Q = 4.80
R1A7_GRIPPER_RIGHT_CLOSE_Q = -0.20
R1A7_GRIPPER_EXTRA_MARGIN = 0.03
R1A7_GRIPPER_VELOCITY_LIMIT = 1.5
R1A7_GRIPPER_KP = 8.0
R1A7_GRIPPER_KD = 1.5
R1A7_GRIPPER_CONTACT_HOLD = True
R1A7_GRIPPER_CONTACT_TRIGGER_ALPHA = 0.85
R1A7_GRIPPER_CONTACT_ERROR = 0.08
R1A7_GRIPPER_CONTACT_STALL_EPS = 0.004
R1A7_GRIPPER_CONTACT_STALL_TIME = 0.25
R1A7_GRIPPER_CONTACT_HOLD_BIAS = 0.035

R1A7_A7_MODEL_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


class R1A7GravityCompensator:
    def __init__(self, args: argparse.Namespace, arm_indices: list[int]):
        self.args = args
        self.arm_indices = arm_indices
        self.enabled = bool(args.enable_gravity_comp)
        self.available = False
        self.last_error: Optional[str] = None
        self.last_tau = np.zeros(14, dtype=float)
        self.pin = None
        self.model = None
        self.data = None
        self.arm_v_indices: list[int] = []
        self.waist_index = args.gravity_model_waist_index
        self.head_pitch_index = args.gravity_model_head_pitch_index
        self.head_yaw_index = args.gravity_model_head_yaw_index
        if self.enabled:
            self._init_model()

    def _init_model(self) -> None:
        try:
            import pinocchio as pin

            self.pin = pin
            self.model = pin.buildModelFromUrdf(self.args.gravity_model_urdf)
            self.data = self.model.createData()
            name_to_joint = {name: i for i, name in enumerate(self.model.names)}
            self.arm_v_indices = [
                self.model.joints[name_to_joint[name]].idx_v
                for name in R1A7_A7_MODEL_ARM_JOINT_NAMES
            ]
            self.available = True
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.available = False

    def compute(self, low_state: Optional[LowState_], arm_q: np.ndarray) -> np.ndarray:
        self.last_tau = np.zeros(14, dtype=float)
        if not self.enabled or not self.available or self.model is None or self.data is None:
            return self.last_tau.copy()
        try:
            q = self.pin.neutral(self.model)
            motor_state = low_state.motor_state if low_state is not None else []
            if self.waist_index >= 0 and self.waist_index < len(motor_state):
                q[0] = float(motor_state[self.waist_index].q)
            if self.head_pitch_index >= 0 and self.head_pitch_index < len(motor_state):
                q[1] = float(motor_state[self.head_pitch_index].q)
            if self.head_yaw_index >= 0 and self.head_yaw_index < len(motor_state):
                q[2] = float(motor_state[self.head_yaw_index].q)
            q[3:17] = np.asarray(arm_q, dtype=float).reshape(14)
            gravity = self.pin.computeGeneralizedGravity(self.model, self.data, q)
            tau = np.asarray([gravity[i] for i in self.arm_v_indices], dtype=float)
            tau *= float(self.args.gravity_comp_scale)
            tau_limit = max(0.0, float(self.args.gravity_comp_tau_limit))
            if tau_limit > 0.0:
                tau = np.clip(tau, -tau_limit, tau_limit)
            self.last_tau = tau
            self.last_error = None
            return tau.copy()
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.last_tau = np.zeros(14, dtype=float)
            return self.last_tau.copy()


class R1A7G1StyleArmController:
    """R1-A7 arm target adapter modeled after Unitree's G1_29_ArmController.

    The official G1 controller clips every publish target from the measured arm
    q instead of integrating commands blindly. This adapter keeps that behavior
    but leaves motor-index mapping, hold joints, grippers, and LowCmd publishing
    in the R1-A7-specific outer controller.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.q_target: Optional[np.ndarray] = None
        self.q_command: Optional[np.ndarray] = None
        self.tauff_target = np.zeros(14, dtype=float)

    def reset(self, current_q: np.ndarray) -> None:
        self.q_target = current_q.copy()
        self.q_command = current_q.copy()
        self.tauff_target[:] = 0.0

    def set_target(self, target_q: Optional[np.ndarray], current_q: np.ndarray) -> None:
        if self.q_command is None:
            self.reset(current_q)
        self.q_target = current_q.copy() if target_q is None else target_q.copy()

    def clip_arm_q_target(self, current_q: np.ndarray, target_q: np.ndarray, dt: float) -> np.ndarray:
        del dt
        control_dt = max(1e-3, float(self.args.g1_control_dt))
        velocity_limit = max(0.0, float(self.args.arm_velocity_limit))
        if velocity_limit <= 0.0:
            return current_q.copy()
        delta = target_q - current_q
        motion_scale = float(np.max(np.abs(delta)) / max(velocity_limit * control_dt, 1e-6))
        return current_q + delta / max(motion_scale, 1.0)

    def _limit_elbow_command_step(self, next_q: np.ndarray) -> np.ndarray:
        if self.q_command is None:
            return next_q
        limit = max(0.0, float(self.args.elbow_command_step_limit))
        if limit <= 0.0:
            return next_q
        limited = next_q.copy()
        for joint_i in (3, 10):
            limited[joint_i] = self.q_command[joint_i] + float(
                np.clip(next_q[joint_i] - self.q_command[joint_i], -limit, limit)
            )
        return limited

    def step(self, current_q: np.ndarray, target_q: Optional[np.ndarray], dt: float) -> np.ndarray:
        self.set_target(target_q, current_q)
        assert self.q_target is not None
        next_q = self.clip_arm_q_target(current_q, self.q_target, dt)
        if self.args.r1a7_extra_elbow_limit:
            next_q = self._limit_elbow_command_step(next_q)
        self.q_command = next_q
        return self.q_command.copy()


class R1A7VRDualArmG1IKReal:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.done = False
        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state: Optional[LowState_] = None
        self.first_state_time: Optional[float] = None
        self.last_lowstate_rx_time: Optional[float] = None
        self.lowstate_count = 0
        self.last_print = 0.0
        self.last_pose_delta = 0.0
        self.last_tele_error: Optional[str] = None
        self.last_ik_error: Optional[str] = None
        self.last_limit_diag = ""
        self.last_vr_debug_print = 0.0
        self.last_arm_tau_cmd = np.zeros(14, dtype=float)
        self.pose_frozen_reported = False
        self.left_indices = self._parse_indices(args.left_arm_indices)
        self.right_indices = self._parse_indices(args.right_arm_indices)
        if len(self.left_indices) != 7 or len(self.right_indices) != 7:
            raise ValueError("left/right arm indices must each contain exactly 7 indices")
        self.arm_indices = self.left_indices + self.right_indices
        self.lowcmd_gripper_indices = self._parse_indices(args.lowcmd_gripper_indices)
        if len(self.lowcmd_gripper_indices) != 2:
            raise ValueError("lowcmd_gripper_indices must contain exactly 2 indices: left,right")
        self.hold_indices = self._parse_indices(args.lowcmd_hold_indices)
        self.fixed_hold_indices = self._parse_indices(args.fixed_hold_indices)
        self.fixed_hold_q: dict[int, float] = {}
        self.command_q: Optional[np.ndarray] = None
        self.arm_controller = R1A7G1StyleArmController(args)
        self.gravity_comp = R1A7GravityCompensator(args, self.arm_indices)
        self.home_q: Optional[np.ndarray] = None
        self.ik_zero_q: Optional[np.ndarray] = None
        self.last_pose_signature: Optional[np.ndarray] = None
        self.last_pose_update_time: Optional[float] = None
        self.prev_motion_ready = False
        self.teleop_active = False
        self.last_arm_enable = False
        self.arm_control_enabled = False
        self.prev_button_state: dict[str, bool] = {}
        self.arm_hold_q: Optional[np.ndarray] = None
        self.arm_hold_strong = False
        self.arm_disabled_hold_active = False
        self.vr_disconnected_uncontrolled = False
        self.lowcmd_released_for_vr_exit = False
        self.rearm_until: Optional[float] = None
        self.recording = False
        self.record_episode_dir: Optional[Path] = None
        self.record_states_file = None
        self.record_writer: Optional[csv.writer] = None
        self.record_samples = 0
        self.record_start_monotonic: Optional[float] = None
        self.record_start_system: Optional[float] = None
        self.d435i_record_process: Optional[subprocess.Popen] = None
        self.d435i_record_file: Optional[Path] = None
        self.d435i_record_log_file = None
        self.d435i_early_exit_reported = False
        # Right-B does not execute a second HOME controller inside VR.
        # It requests a clean VR shutdown; the outer launcher owns verified AUTO_HOME V2.
        self.home_return_requested = False
        self.publisher = None
        self.subscriber = None
        self.tv = None
        self.arm_ik = None

        # Optional Dex1-1 gripper controller. These multiprocessing containers
        # match the official xr_teleoperate Dex1_1_Gripper_Controller API.
        self.gripper_ctrl = None
        self.gripper_init_thread = None
        self.gripper_connected = False
        self.xr_motion_data_ready = None
        self.left_gripper_value = None
        self.right_gripper_value = None
        self.dual_gripper_data_lock = None
        self.dual_gripper_state_array = None
        self.dual_gripper_action_array = None
        self.last_left_trigger = 10.0
        self.last_right_trigger = 10.0
        self.last_gripper_error: Optional[str] = None
        self.lowcmd_gripper_home_q: Optional[np.ndarray] = None
        self.lowcmd_gripper_cmd_q: Optional[np.ndarray] = None
        self.lowcmd_gripper_target_q: Optional[np.ndarray] = None
        self.lowcmd_gripper_ready = False
        self.lowcmd_gripper_contact_q = np.full(2, np.nan, dtype=float)
        self.lowcmd_gripper_contact_hold = np.zeros(2, dtype=bool)
        self.lowcmd_gripper_stall_since: list[Optional[float]] = [None, None]
        self.lowcmd_gripper_prev_state_q: Optional[np.ndarray] = None
        self.lowcmd_gripper_prev_state_time: Optional[float] = None

    @staticmethod
    def _parse_indices(text: str) -> list[int]:
        return [int(part.strip()) for part in text.split(",") if part.strip()]

    @staticmethod
    def _parse_float_list(text: str, expected_len: int, name: str) -> np.ndarray:
        values = [float(part.strip()) for part in text.split(",") if part.strip()]
        if len(values) != expected_len:
            raise ValueError(f"{name} must contain exactly {expected_len} comma-separated values")
        return np.asarray(values, dtype=float)

    def _enter_debug_mode(self) -> None:
        msc = MotionSwitcherClient()
        msc.SetTimeout(2.0)
        msc.Init()
        status, result = msc.CheckMode()
        print(f"[R1-A7 VR G1IK REAL] motion_switcher CheckMode: status={status} result={result}")
        try:
            while result and result.get("name"):
                print("[R1-A7 VR G1IK REAL] releasing active mode:", result)
                msc.ReleaseMode()
                time.sleep(0.5)
                status, result = msc.CheckMode()
                print(f"[R1-A7 VR G1IK REAL] motion_switcher CheckMode: status={status} result={result}")
        except Exception as exc:
            print(f"[R1-A7 VR G1IK REAL] failed to enter debug mode: {exc}")

    def init(self) -> None:
        ChannelFactoryInitialize(self.args.domain_id, self.args.interface)
        if self.args.enter_debug_mode:
            self._enter_debug_mode()
        self.subscriber = ChannelSubscriber(self.args.state_topic, LowState_)
        self.subscriber.Init(self._lowstate_handler, 10)
        self.publisher = ChannelPublisher(self.args.command_topic, LowCmd_)
        self.publisher.Init()

        old_cwd = Path.cwd()
        os.chdir(XR_TELEOP_TELEOP)
        try:
            self.arm_ik = G1_29_ArmIK(Unit_Test=False, Visualization=False)
        finally:
            os.chdir(old_cwd)

        self.tv = TeleVuerWrapper(
            use_hand_tracking=False,
            binocular=False,
            img_shape=(480, 640),
            display_mode="pass-through",
            zmq=False,
            webrtc=False,
            arm_reference_mode="head_yaw",
        )

        self._init_gripper_controller()

        print("[R1-A7 VR G1IK REAL] DDS initialized")
        print("[R1-A7 VR G1IK REAL] IK: Unitree G1_29_ArmIK.solve_ik")
        print("[R1-A7 VR G1IK REAL] interface:", self.args.interface)
        print("[R1-A7 VR G1IK REAL] state topic:", self.args.state_topic)
        print("[R1-A7 VR G1IK REAL] command topic:", self.args.command_topic)
        print("[R1-A7 VR G1IK REAL] left arm indices:", self.left_indices)
        print("[R1-A7 VR G1IK REAL] right arm indices:", self.right_indices)
        if self.args.g1_official_adapter:
            print(
                "[R1-A7 VR G1IK REAL] controller: R1-A7 G1-style adapter "
                f"dt={self.args.g1_control_dt:.4f}s velocity_limit={self.args.arm_velocity_limit:.3f}rad/s"
            )
        if self.args.enable_gravity_comp:
            print(
                "[R1-A7 VR G1IK REAL] gravity compensation: "
                f"enabled={self.gravity_comp.available} scale={self.args.gravity_comp_scale:.3f} "
                f"limit={self.args.gravity_comp_tau_limit:.3f} urdf={self.args.gravity_model_urdf}"
            )
            if self.gravity_comp.last_error:
                print(f"  gravity_error: {self.gravity_comp.last_error}")
        if self.args.arm_enable_button != "none":
            print(
                "[R1-A7 VR G1IK REAL] arm/gripper enable gate: "
                f"{self.args.arm_enable_mode} {self.args.arm_enable_button}"
            )
        if self.args.record_button != "none":
            print(
                "[R1-A7 VR G1IK REAL] one-shot synchronized recording: "
                f"press {self.args.record_button} -> {self.args.record_duration_s:.1f}s "
                f"robot CSV + D435i={self.args.record_d435i}"
            )
        if self.args.home_button != "none":
            print(
                "[R1-A7 VR G1IK REAL] HOME request button: "
                f"{self.args.home_button} -> stop recording, release LowCmd, exit code 42"
            )
        if self.args.enable_gripper:
            if self.args.gripper_mode == "lowcmd":
                print("[R1-A7 VR G1IK REAL] lowcmd gripper trigger control: ENABLED")
                print("  left trigger  -> motor 31 L_HAND; right trigger -> motor 33 R_HAND")
                mode_text = "relative-to-startup" if self.args.lowcmd_gripper_relative else "absolute motor q"
                print(f"  coordinate mode: {mode_text}")
                print(
                    "  q range       : "
                    f"left {self.args.lowcmd_gripper_left_open_q:.3f}->{self.args.lowcmd_gripper_left_close_q:.3f}, "
                    f"right {self.args.lowcmd_gripper_right_open_q:.3f}->{self.args.lowcmd_gripper_right_close_q:.3f}"
                )
            else:
                print("[R1-A7 VR G1IK REAL] Dex1 DDS trigger control: ENABLED")
                print("  left trigger  -> left Dex1; right trigger -> right Dex1")
            print("  release trigger -> OPEN; pull trigger -> CLOSE")
        else:
            print("[R1-A7 VR G1IK REAL] gripper trigger control: disabled")
        print("[R1-A7 VR G1IK REAL] open Quest URL:")
        print(f"https://{self.args.host_ip}:8012/?ws=wss://{self.args.host_ip}:8012")

    def _init_gripper_controller(self) -> None:
        if not self.args.enable_gripper or self.args.gripper_mode != "dex1_dds":
            return

        try:
            from robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller
        except Exception as exc:
            raise RuntimeError(
                "Cannot import Dex1_1_Gripper_Controller from "
                "/home/robot/xr_teleoperate/teleop/robot_control/robot_hand_unitree.py"
            ) from exc

        self.left_gripper_value = Value("d", 10.0, lock=True)
        self.right_gripper_value = Value("d", 10.0, lock=True)
        self.xr_motion_data_ready = Value("b", False, lock=True)
        self.dual_gripper_data_lock = Lock()
        self.dual_gripper_state_array = Array("d", 2, lock=False)
        self.dual_gripper_action_array = Array("d", 2, lock=False)

        print("[R1-A7 VR G1IK REAL] initializing Dex1-1 gripper DDS in background...")
        print("  waiting for rt/dex1/left/state and rt/dex1/right/state")

        def _worker() -> None:
            try:
                controller = Dex1_1_Gripper_Controller(
                    self.left_gripper_value,
                    self.right_gripper_value,
                    self.dual_gripper_data_lock,
                    self.dual_gripper_state_array,
                    self.dual_gripper_action_array,
                    filter=not self.args.disable_gripper_filter,
                    fps=self.args.gripper_fps,
                    simulation_mode=False,
                    xr_motion_data_ready_in=self.xr_motion_data_ready,
                )
                self.gripper_ctrl = controller
                self.gripper_connected = True
                self.last_gripper_error = None
                print("[R1-A7 VR G1IK REAL] Dex1-1 gripper DDS ready")
            except Exception as exc:
                self.gripper_connected = False
                self.last_gripper_error = f"{type(exc).__name__}: {exc}"
                print(f"[R1-A7 VR G1IK REAL] Dex1 init error: {self.last_gripper_error}")

        self.gripper_init_thread = threading.Thread(target=_worker, daemon=True)
        self.gripper_init_thread.start()

    def _update_gripper_inputs(self, tele, active: bool) -> None:
        if not self.args.enable_gripper:
            return

        if self.args.gripper_mode == "lowcmd":
            self._update_lowcmd_gripper_inputs(tele, active)
            return

        if self.xr_motion_data_ready is None:
            return

        ready = False
        if active and tele is not None:
            if not hasattr(tele, "left_ctrl_triggerValue") or not hasattr(tele, "right_ctrl_triggerValue"):
                self.last_gripper_error = (
                    "TeleVuer data has no controller trigger fields; update xr_teleoperate/televuer"
                )
            else:
                left_value = float(np.clip(float(tele.left_ctrl_triggerValue), 0.0, 10.0))
                right_value = float(np.clip(float(tele.right_ctrl_triggerValue), 0.0, 10.0))
                self.last_left_trigger = left_value
                self.last_right_trigger = right_value
                assert self.left_gripper_value is not None
                assert self.right_gripper_value is not None
                with self.left_gripper_value.get_lock():
                    self.left_gripper_value.value = left_value
                with self.right_gripper_value.get_lock():
                    self.right_gripper_value.value = right_value
                if self.gripper_connected:
                    self.last_gripper_error = None
                ready = True

        with self.xr_motion_data_ready.get_lock():
            self.xr_motion_data_ready.value = bool(ready)

    def _gripper_status(self) -> tuple[bool, bool, float, float, float, float]:
        if not self.args.enable_gripper:
            return False, False, float("nan"), float("nan"), float("nan"), float("nan")
        if self.args.gripper_mode == "lowcmd":
            state = self._read_lowcmd_gripper_q()
            if state is None:
                return False, False, float("nan"), float("nan"), float("nan"), float("nan")
            action = (
                self.lowcmd_gripper_cmd_q
                if self.lowcmd_gripper_cmd_q is not None
                else np.asarray(state, dtype=float)
            )
            return (
                self.lowcmd_gripper_ready,
                bool(self.lowcmd_gripper_target_q is not None),
                float(state[0]),
                float(state[1]),
                float(action[0]),
                float(action[1]),
            )
        if self.xr_motion_data_ready is None:
            return False, False, float("nan"), float("nan"), float("nan"), float("nan")
        with self.xr_motion_data_ready.get_lock():
            ready = bool(self.xr_motion_data_ready.value)
        if self.dual_gripper_data_lock is None:
            return self.gripper_connected, ready, float("nan"), float("nan"), float("nan"), float("nan")
        with self.dual_gripper_data_lock:
            state = np.asarray(self.dual_gripper_state_array[:], dtype=float)
            action = np.asarray(self.dual_gripper_action_array[:], dtype=float)
        return self.gripper_connected, ready, float(state[0]), float(state[1]), float(action[0]), float(action[1])

    def _stop_gripper_controller(self) -> None:
        if self.xr_motion_data_ready is not None:
            with self.xr_motion_data_ready.get_lock():
                self.xr_motion_data_ready.value = False
        if self.gripper_ctrl is not None and hasattr(self.gripper_ctrl, "running"):
            self.gripper_ctrl.running = False

    def _read_lowcmd_gripper_q(self) -> Optional[np.ndarray]:
        if self.low_state is None:
            return None
        motor_state = self.low_state.motor_state
        max_idx = max(self.lowcmd_gripper_indices)
        if len(motor_state) <= max_idx:
            raise RuntimeError(f"lowstate has {len(motor_state)} motors, requested gripper index {max_idx}")
        return np.array([motor_state[i].q for i in self.lowcmd_gripper_indices], dtype=float)

    @staticmethod
    def _trigger_to_open_close_alpha(trigger_value: float) -> float:
        # TeleVuer currently reports 10.0 when released and 0.0 when fully pulled.
        # Convert to 0=open, 1=closed.
        return float(np.clip((10.0 - trigger_value) / 10.0, 0.0, 1.0))

    def _lowcmd_gripper_open_close_q(self) -> tuple[np.ndarray, np.ndarray]:
        open_q = np.array(
            [self.args.lowcmd_gripper_left_open_q, self.args.lowcmd_gripper_right_open_q],
            dtype=float,
        )
        close_q = np.array(
            [self.args.lowcmd_gripper_left_close_q, self.args.lowcmd_gripper_right_close_q],
            dtype=float,
        )
        if self.args.lowcmd_gripper_relative:
            if self.lowcmd_gripper_home_q is None:
                current = self._read_lowcmd_gripper_q()
                if current is None:
                    current = np.zeros(2, dtype=float)
                self.lowcmd_gripper_home_q = current.copy()
            open_q = self.lowcmd_gripper_home_q + open_q
            close_q = self.lowcmd_gripper_home_q + close_q
        return open_q, close_q

    def _update_lowcmd_gripper_inputs(self, tele, active: bool) -> None:
        current_q = self._read_lowcmd_gripper_q()
        if current_q is None:
            self.lowcmd_gripper_ready = False
            self.last_gripper_error = "lowstate unavailable for lowcmd gripper motors"
            return
        now = time.monotonic()

        if self.lowcmd_gripper_home_q is None:
            self.lowcmd_gripper_home_q = current_q.copy()
            self.lowcmd_gripper_cmd_q = current_q.copy()
            self.lowcmd_gripper_target_q = current_q.copy()
            self.lowcmd_gripper_prev_state_q = current_q.copy()
            self.lowcmd_gripper_prev_state_time = now
            print(
                "[R1-A7 VR G1IK REAL] calibrated lowcmd gripper q:",
                np.round(current_q, 3).tolist(),
            )

        self.lowcmd_gripper_ready = True
        if not active or tele is None:
            self.lowcmd_gripper_target_q = current_q.copy()
            self.lowcmd_gripper_cmd_q = current_q.copy()
            self._reset_lowcmd_gripper_contact_hold()
            return

        if not hasattr(tele, "left_ctrl_triggerValue") or not hasattr(tele, "right_ctrl_triggerValue"):
            self.last_gripper_error = "TeleVuer data has no controller trigger fields"
            self.lowcmd_gripper_target_q = current_q.copy()
            self._reset_lowcmd_gripper_contact_hold()
            return

        left_value = float(np.clip(float(tele.left_ctrl_triggerValue), 0.0, 10.0))
        right_value = float(np.clip(float(tele.right_ctrl_triggerValue), 0.0, 10.0))
        self.last_left_trigger = left_value
        self.last_right_trigger = right_value
        left_alpha = self._trigger_to_open_close_alpha(left_value)
        right_alpha = self._trigger_to_open_close_alpha(right_value)
        open_q, close_q = self._lowcmd_gripper_open_close_q()
        target = np.array(
            [
                open_q[0] + left_alpha * (close_q[0] - open_q[0]),
                open_q[1] + right_alpha * (close_q[1] - open_q[1]),
            ],
            dtype=float,
        )
        low = np.minimum(open_q, close_q) - abs(self.args.lowcmd_gripper_extra_margin)
        high = np.maximum(open_q, close_q) + abs(self.args.lowcmd_gripper_extra_margin)
        trigger_alpha = np.array([left_alpha, right_alpha], dtype=float)
        self.lowcmd_gripper_target_q = self._apply_lowcmd_gripper_contact_hold(
            current_q=current_q,
            requested_target_q=np.clip(target, low, high),
            open_q=open_q,
            close_q=close_q,
            low=low,
            high=high,
            trigger_alpha=trigger_alpha,
            now=now,
        )
        self.lowcmd_gripper_prev_state_q = current_q.copy()
        self.lowcmd_gripper_prev_state_time = now
        self.last_gripper_error = None

    def _reset_lowcmd_gripper_contact_hold(self) -> None:
        self.lowcmd_gripper_contact_q[:] = np.nan
        self.lowcmd_gripper_contact_hold[:] = False
        self.lowcmd_gripper_stall_since = [None, None]

    def _apply_lowcmd_gripper_contact_hold(
        self,
        current_q: np.ndarray,
        requested_target_q: np.ndarray,
        open_q: np.ndarray,
        close_q: np.ndarray,
        low: np.ndarray,
        high: np.ndarray,
        trigger_alpha: np.ndarray,
        now: float,
    ) -> np.ndarray:
        if not self.args.lowcmd_gripper_contact_hold:
            return requested_target_q

        target_q = requested_target_q.copy()
        close_dir = np.sign(close_q - open_q)
        prev_q = self.lowcmd_gripper_prev_state_q

        for i in range(2):
            pulled = trigger_alpha[i] >= self.args.lowcmd_gripper_contact_trigger_alpha
            if not pulled:
                self.lowcmd_gripper_contact_hold[i] = False
                self.lowcmd_gripper_contact_q[i] = np.nan
                self.lowcmd_gripper_stall_since[i] = None
                continue

            target_error = abs(requested_target_q[i] - current_q[i])
            state_delta = abs(current_q[i] - prev_q[i]) if prev_q is not None else float("inf")
            still_blocked = target_error >= self.args.lowcmd_gripper_contact_error
            barely_moving = state_delta <= self.args.lowcmd_gripper_contact_stall_eps

            if self.lowcmd_gripper_contact_hold[i]:
                contact_q = self.lowcmd_gripper_contact_q[i]
            elif still_blocked and barely_moving:
                if self.lowcmd_gripper_stall_since[i] is None:
                    self.lowcmd_gripper_stall_since[i] = now
                contact_q = current_q[i]
                if now - self.lowcmd_gripper_stall_since[i] >= self.args.lowcmd_gripper_contact_stall_time:
                    self.lowcmd_gripper_contact_hold[i] = True
                    self.lowcmd_gripper_contact_q[i] = current_q[i]
                    contact_q = current_q[i]
            else:
                self.lowcmd_gripper_stall_since[i] = None
                contact_q = current_q[i]

            if self.lowcmd_gripper_contact_hold[i]:
                bias = abs(self.args.lowcmd_gripper_contact_hold_bias)
                target_q[i] = np.clip(contact_q + close_dir[i] * bias, low[i], high[i])

        return target_q

    def _step_lowcmd_gripper(self, state_q: np.ndarray, dt: float) -> Optional[np.ndarray]:
        if not self.args.enable_gripper or self.args.gripper_mode != "lowcmd":
            return None
        if self.lowcmd_gripper_target_q is None:
            self.lowcmd_gripper_cmd_q = state_q.copy()
            return self.lowcmd_gripper_cmd_q.copy()
        if self.lowcmd_gripper_cmd_q is None:
            self.lowcmd_gripper_cmd_q = state_q.copy()
        max_delta = max(0.0, self.args.lowcmd_gripper_velocity_limit) * max(dt, 1e-3)
        self.lowcmd_gripper_cmd_q = self.lowcmd_gripper_cmd_q + np.clip(
            self.lowcmd_gripper_target_q - self.lowcmd_gripper_cmd_q,
            -max_delta,
            max_delta,
        )
        return self.lowcmd_gripper_cmd_q.copy()

    def _lowstate_handler(self, msg: LowState_) -> None:
        now = time.monotonic()
        self.low_state = msg
        self.last_lowstate_rx_time = now
        self.lowstate_count += 1
        if self.first_state_time is None:
            self.first_state_time = now

    def _read_arm_qdq(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if self.low_state is None:
            return None
        motor_state = self.low_state.motor_state
        max_idx = max(self.arm_indices)
        if len(motor_state) <= max_idx:
            raise RuntimeError(f"lowstate has {len(motor_state)} motors, requested index {max_idx}")
        q = np.array([motor_state[i].q for i in self.arm_indices], dtype=float)
        dq = np.array([motor_state[i].dq for i in self.arm_indices], dtype=float)
        return q, dq

    def _button_pressed(self, tele, button: str) -> bool:
        if button == "none":
            return False
        return bool(getattr(tele, button, False))

    def _button_rising_edge(self, tele, button: str) -> bool:
        if button == "none":
            return False
        pressed = self._button_pressed(tele, button)
        prev = self.prev_button_state.get(button, False)
        self.prev_button_state[button] = pressed
        return pressed and not prev

    def _init_low_cmd_stop(self) -> None:
        for motor in self.low_cmd.motor_cmd:
            motor.tau = 0.0
            motor.q = 0.0
            motor.dq = 0.0
            motor.kp = 0.0
            motor.kd = 0.0

    def _publish(self, command_q: np.ndarray, gripper_q: Optional[np.ndarray] = None) -> None:
        assert self.publisher is not None
        self._init_low_cmd_stop()
        arm_tau = self.gravity_comp.compute(self.low_state, command_q)
        shoulder_pitch_ff = float(self.args.shoulder_pitch_extra_tau)
        if shoulder_pitch_ff != 0.0:
            arm_tau[0] += shoulder_pitch_ff
            arm_tau[7] += shoulder_pitch_ff
            tau_limit = max(0.0, float(self.args.total_tau_limit))
            if tau_limit > 0.0:
                arm_tau = np.clip(arm_tau, -tau_limit, tau_limit)
        if self.low_state is not None:
            if hasattr(self.low_cmd, "mode_pr"):
                self.low_cmd.mode_pr = 0
            if hasattr(self.low_cmd, "mode_machine") and hasattr(self.low_state, "mode_machine"):
                self.low_cmd.mode_machine = self.low_state.mode_machine
            count = min(len(self.low_cmd.motor_cmd), len(self.low_state.motor_state))
            if not self.fixed_hold_q:
                for i in self.fixed_hold_indices:
                    if i < count:
                        self.fixed_hold_q[i] = float(self.low_state.motor_state[i].q)
            for i in self.hold_indices:
                if i >= count:
                    continue
                motor = self.low_cmd.motor_cmd[i]
                motor.mode = 1
                motor.tau = 0.0
                motor.q = self.fixed_hold_q.get(i, float(self.low_state.motor_state[i].q))
                motor.dq = 0.0
                motor.kp = self.args.hold_kp
                motor.kd = self.args.hold_kd

        for joint_i, (idx, q) in enumerate(zip(self.arm_indices, command_q)):
            motor = self.low_cmd.motor_cmd[idx]
            motor.mode = 1
            motor.tau = float(arm_tau[joint_i])
            motor.q = float(q)
            motor.dq = 0.0
            motor.kp, motor.kd = self._arm_gain(
                joint_i,
                disabled_hold=self.arm_disabled_hold_active,
            )
        self.last_arm_tau_cmd = arm_tau.copy()

        if gripper_q is not None and self.args.enable_gripper and self.args.gripper_mode == "lowcmd":
            for idx, q in zip(self.lowcmd_gripper_indices, gripper_q):
                motor = self.low_cmd.motor_cmd[idx]
                motor.mode = 1
                motor.tau = 0.0
                motor.q = float(q)
                motor.dq = 0.0
                motor.kp = self.args.lowcmd_gripper_kp
                motor.kd = self.args.lowcmd_gripper_kd

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)

    def _release(self) -> None:
        if self.publisher is None:
            return
        self._init_low_cmd_stop()
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)
        print("[R1-A7 VR G1IK REAL] released lowcmd gains")

    def _release_for_vr_exit_once(self) -> None:
        if self.lowcmd_released_for_vr_exit:
            return
        self._release()
        self.lowcmd_released_for_vr_exit = True
        print("[R1-A7 VR G1IK REAL] VR disconnected; LowCmd output released, robot is no longer controlled")

    def _mark_lowcmd_active(self) -> None:
        self.vr_disconnected_uncontrolled = False
        self.lowcmd_released_for_vr_exit = False

    def _step_command(self, state_q: np.ndarray, target_q: Optional[np.ndarray], dt: float) -> np.ndarray:
        if self.args.g1_official_adapter:
            self.command_q = self.arm_controller.step(state_q, target_q, dt)
            return self.command_q.copy()

        if target_q is None:
            if self.command_q is None:
                self.command_q = state_q.copy()
            target_q = self.command_q.copy()

        if self.args.g1_style_velocity_clip:
            delta = target_q - state_q
            max_delta = max(0.0, self.args.arm_velocity_limit) * max(dt, 1e-3)
            motion_scale = float(np.max(np.abs(delta)) / max(max_delta, 1e-6))
            next_q = state_q + delta / max(motion_scale, 1.0)
            self.command_q = self._limit_elbow_command_step(next_q)
            return self.command_q.copy()

        if self.command_q is None:
            self.command_q = state_q.copy()
        max_delta = max(0.0, self.args.max_speed_rad_s) * max(dt, 1e-3)
        next_q = self.command_q + np.clip(target_q - self.command_q, -max_delta, max_delta)
        lead = np.clip(next_q - state_q, -self.args.max_command_lead, self.args.max_command_lead)
        self.command_q = self._limit_elbow_command_step(state_q + lead)
        return self.command_q.copy()

    def _limit_elbow_command_step(self, next_q: np.ndarray) -> np.ndarray:
        if self.command_q is None:
            return next_q
        limit = max(0.0, self.args.elbow_command_step_limit)
        if limit <= 0.0:
            return next_q
        limited = next_q.copy()
        for joint_i in (3, 10):
            limited[joint_i] = self.command_q[joint_i] + float(
                np.clip(next_q[joint_i] - self.command_q[joint_i], -limit, limit)
            )
        return limited

    def _arm_gain(self, joint_i: int, disabled_hold: bool = False) -> tuple[float, float]:
        if disabled_hold and not self.args.strong_hold_after_arm_disable:
            return self.args.hold_kp, self.args.hold_kd
        # Match Unitree G1_29_ArmController: shoulder/elbow use kp_low/kd_low,
        # wrist roll/pitch/yaw use kp_wrist/kd_wrist.
        local_i = joint_i % 7
        if self.args.g1_style_gains:
            if local_i >= 4:
                return self.args.kp_wrist, self.args.kd_wrist
            return self.args.kp_low, self.args.kd_low
        return self.args.kp, self.args.kd

    def _limited_target(self, sol_q: np.ndarray) -> np.ndarray:
        assert self.home_q is not None
        hard_low, hard_high = self._active_arm_limits()
        if self.args.official_ik_target:
            clipped_q = np.clip(sol_q, hard_low, hard_high)
            self._update_limit_diag(sol_q, clipped_q, hard_low, hard_high)
            return clipped_q
        low = np.maximum(self.home_q - self.args.max_joint_offset_rad, hard_low)
        high = np.minimum(self.home_q + self.args.max_joint_offset_rad, hard_high)
        clipped_q = np.clip(sol_q, low, high)
        self._update_limit_diag(sol_q, clipped_q, low, high)
        return clipped_q

    def _active_arm_limits(self) -> tuple[np.ndarray, np.ndarray]:
        low = R1A7_ARM_LIMITS[:, 0].copy()
        high = R1A7_ARM_LIMITS[:, 1].copy()
        margin = max(0.0, self.args.joint_limit_margin_rad)
        if margin > 0.0:
            low += margin
            high -= margin
        pitch_margin = max(0.0, self.args.shoulder_pitch_low_margin_rad)
        if pitch_margin > 0.0:
            low[0] = max(low[0], R1A7_ARM_LIMITS[0, 0] + pitch_margin)
            low[7] = max(low[7], R1A7_ARM_LIMITS[7, 0] + pitch_margin)
        return low, high

    def _update_limit_diag(self, raw_q: np.ndarray, clipped_q: np.ndarray, low: np.ndarray, high: np.ndarray) -> None:
        changed = np.abs(raw_q - clipped_q) > self.args.limit_diag_eps
        if not np.any(changed):
            self.last_limit_diag = ""
            return
        entries = []
        for i in np.where(changed)[0]:
            side = "low" if raw_q[i] < low[i] else "high"
            entries.append(
                f"{ARM_NAMES[i]} {side} raw={raw_q[i]:+.3f} clip={clipped_q[i]:+.3f} "
                f"range=[{low[i]:+.3f},{high[i]:+.3f}]"
            )
        self.last_limit_diag = "; ".join(entries[:6])

    def _retarget_solution(self, sol_q: np.ndarray) -> tuple[Optional[np.ndarray], str]:
        assert self.home_q is not None
        if self.args.official_ik_target:
            return self._limited_target(sol_q), "official_ik_target"
        if self.args.absolute_ik:
            return self._limited_target(sol_q), "absolute_tracking"

        if self.ik_zero_q is None:
            self.ik_zero_q = sol_q.copy()
            return self.home_q.copy(), "calibrated_ik_zero"

        joint_scales = self._parse_float_list(self.args.ik_joint_scales, 14, "ik_joint_scales")
        delta_q = self.args.ik_delta_scale * joint_scales * (sol_q - self.ik_zero_q)
        raw_target_q = self.home_q + delta_q
        clipped_delta_q = np.clip(delta_q, -self.args.max_joint_offset_rad, self.args.max_joint_offset_rad)
        target_q = self.home_q + clipped_delta_q
        hard_low, hard_high = self._active_arm_limits()
        target_q = np.clip(target_q, hard_low, hard_high)
        low = np.maximum(self.home_q - self.args.max_joint_offset_rad, hard_low)
        high = np.minimum(self.home_q + self.args.max_joint_offset_rad, hard_high)
        self._update_limit_diag(raw_target_q, target_q, low, high)
        max_abs_delta = float(np.max(np.abs(delta_q))) if delta_q.size else 0.0
        return target_q, f"relative_tracking dq={max_abs_delta:.3f}"

    @staticmethod
    def _pose_signature(tele) -> np.ndarray:
        # Include trigger values so a trigger pull counts as fresh XR input even
        # when both wrists remain still. TeleVuer uses 10=release and 0=full pull.
        controller_values = np.array(
            [
                float(getattr(tele, "left_ctrl_triggerValue", 10.0)),
                float(getattr(tele, "right_ctrl_triggerValue", 10.0)),
                float(bool(getattr(tele, "left_ctrl_aButton", False))),
                float(bool(getattr(tele, "right_ctrl_aButton", False))),
            ],
            dtype=float,
        )
        return np.concatenate(
            [
                np.asarray(tele.left_wrist_pose, dtype=float).reshape(-1),
                np.asarray(tele.right_wrist_pose, dtype=float).reshape(-1),
                controller_values,
            ]
        )

    def _teleop_data_fresh(self, tele, now: float) -> bool:
        if not bool(tele.motion_data_ready):
            self.last_pose_delta = 0.0
            self.last_pose_signature = None
            self.last_pose_update_time = None
            self.pose_frozen_reported = False
            return False
        if self.args.treat_motion_ready_as_fresh:
            signature = self._pose_signature(tele)
            if self.last_pose_signature is not None:
                self.last_pose_delta = float(np.max(np.abs(signature - self.last_pose_signature)))
            self.last_pose_signature = signature
            self.last_pose_update_time = now
            return True

        signature = self._pose_signature(tele)
        if self.last_pose_signature is None:
            self.last_pose_signature = signature
            self.last_pose_update_time = now
            self.last_pose_delta = 0.0
            return True

        self.last_pose_delta = float(np.max(np.abs(signature - self.last_pose_signature)))
        if self.last_pose_delta >= self.args.pose_change_eps:
            self.last_pose_signature = signature
            self.last_pose_update_time = now
            self.pose_frozen_reported = False
            return True

        if self.last_pose_update_time is None:
            self.last_pose_update_time = now
            return True
        unchanged_age = now - self.last_pose_update_time
        if (
            self.teleop_active
            and self.args.frozen_pose_hold_timeout > 0.0
            and unchanged_age > self.args.frozen_pose_hold_timeout
        ):
            if not self.pose_frozen_reported:
                print(
                    "[R1-A7 VR G1IK REAL] Quest pose frozen before stale timeout; "
                    "holding current arm q"
                )
                self.pose_frozen_reported = True
            return False
        return unchanged_age <= self.args.stale_pose_timeout

    def _home_return_button_rising(self, tele) -> bool:
        """Return True once when the configured HOME-request button rises.

        The VR process never generates a HOME trajectory here.  A B press only
        asks this process to stop recording, release LowCmd, and exit with code
        42.  The outer launcher then starts the separately verified AUTO_HOME V2.
        """
        button = self.args.home_button
        if button == "none" or tele is None:
            return False
        if not self._button_rising_edge(tele, button):
            return False
        self.home_return_requested = True
        print(
            "[R1-A7 VR G1IK REAL] HOME RETURN REQUESTED: "
            f"button={button}"
        )
        return True

    def _arm_enable_pressed(self, tele) -> bool:
        button = self.args.arm_enable_button
        if button == "none":
            return True
        if self.args.arm_enable_mode == "hold":
            return self._button_pressed(tele, button)
        if self._button_rising_edge(tele, button):
            self.arm_control_enabled = not self.arm_control_enabled
            if self.arm_control_enabled:
                print("[R1-A7 VR G1IK REAL] arm control toggled ON")
            else:
                print("[R1-A7 VR G1IK REAL] arm control toggled OFF; holding current arm q")
        return self.arm_control_enabled

    def _maybe_print_vr_debug(self, tele, now: float) -> None:
        if not self.args.debug_vr_data:
            return
        if now - self.last_vr_debug_print < max(0.2, self.args.debug_vr_period):
            return
        self.last_vr_debug_print = now
        fields = [
            "motion_data_ready",
            "left_ctrl_aButton",
            "left_ctrl_bButton",
            "left_ctrl_squeeze",
            "left_ctrl_thumbstick",
            "left_ctrl_triggerValue",
            "right_ctrl_aButton",
            "right_ctrl_bButton",
            "right_ctrl_squeeze",
            "right_ctrl_thumbstick",
            "right_ctrl_triggerValue",
        ]
        values = {name: getattr(tele, name, None) for name in fields}
        left_pose = getattr(tele, "left_wrist_pose", None)
        right_pose = getattr(tele, "right_wrist_pose", None)
        try:
            left_t = np.asarray(left_pose, dtype=float).reshape(4, 4)[:3, 3].round(4).tolist()
        except Exception:
            left_t = None
        try:
            right_t = np.asarray(right_pose, dtype=float).reshape(4, 4)[:3, 3].round(4).tolist()
        except Exception:
            right_t = None
        available = [name for name in dir(tele) if "Button" in name or "ctrl" in name or "motion" in name]
        print(
            "[R1-A7 VR G1IK REAL] vr_debug "
            f"values={values} left_t={left_t} right_t={right_t} fields={available}"
        )

    def _pose_age_text(self, now: float) -> str:
        if self.last_pose_update_time is None:
            return "n/a"
        return f"{max(0.0, now - self.last_pose_update_time):.3f}s"

    def _hold_current_after_enable_release(self, state_q: np.ndarray) -> tuple[np.ndarray, str]:
        if self.teleop_active or self.prev_motion_ready or self.last_arm_enable:
            print("[R1-A7 VR G1IK REAL] arm control disabled; holding last commanded arm q")
        if self.arm_hold_q is None:
            if self.command_q is not None:
                low, high = self._active_arm_limits()
                self.arm_hold_q = np.clip(self.command_q.copy(), low, high)
            else:
                self.arm_hold_q = state_q.copy()
        self.teleop_active = False
        self.prev_motion_ready = False
        self.last_arm_enable = False
        self.ik_zero_q = None
        self.home_q = self.arm_hold_q.copy()
        self.command_q = self.arm_hold_q.copy()
        self.arm_controller.reset(self.arm_hold_q)
        self._mark_lowcmd_active()
        self.arm_hold_strong = self.args.strong_hold_after_arm_disable
        self.arm_disabled_hold_active = True
        self.rearm_until = None
        self.pose_frozen_reported = False
        return self.arm_hold_q.copy(), "enable_released_hold"

    def _get_next_record_episode_dir(self) -> Path:
        """Return the next unused episode directory without overwriting data.

        Examples:
            --record_episode_id right_safe_001
              -> right_safe_001, right_safe_002, right_safe_003, ...

            --record_episode_id right_safe
              -> right_safe_001, right_safe_002, right_safe_003, ...

        Existing directories are skipped automatically.
        """
        root = Path(self.args.record_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)

        requested_id = str(self.args.record_episode_id).strip()
        if not requested_id:
            requested_id = "vr_x60_001"

        match = re.match(r"^(.*?)(\d+)$", requested_id)
        if match:
            prefix = match.group(1)
            start_number = int(match.group(2))
            width = len(match.group(2))
        else:
            prefix = requested_id.rstrip("_") + "_"
            start_number = 1
            width = 3

        number = start_number
        while number <= 999999:
            episode_id = f"{prefix}{number:0{width}d}"
            episode_dir = root / episode_id
            if not episode_dir.exists():
                return episode_dir
            number += 1

        raise RuntimeError(
            f"unable to allocate a new recording episode under {root}"
        )

    def _record_start(self) -> None:
        if self.recording:
            return

        episode_dir = self._get_next_record_episode_dir()
        episode_dir.mkdir(parents=True, exist_ok=False)
        episode_id = episode_dir.name

        duration_s = max(0.1, float(self.args.record_duration_s))
        self.record_start_monotonic = time.monotonic()
        self.record_start_system = time.time()
        self.d435i_early_exit_reported = False

        metadata = {
            "robot": "Unitree R1-A7",
            "task": "vr_dual_arm_x_button_synchronized_record",
            "episode_id": episode_id,
            "date": time.strftime("%Y-%m-%d"),
            "recording_mode": "x_one_shot_robot_csv_plus_d435i_db3",
            "record_duration_s": duration_s,
            "record_start_monotonic": self.record_start_monotonic,
            "record_start_system": self.record_start_system,
            "interface": self.args.interface,
            "state_topic": self.args.state_topic,
            "command_topic": self.args.command_topic,
            "arm_enable_mode": self.args.arm_enable_mode,
            "arm_enable_button": self.args.arm_enable_button,
            "record_button": self.args.record_button,
            "d435i_record_enabled": bool(self.args.record_d435i),
            "d435i_filename": self.args.d435i_filename,
            "note": "X starts both recorders in software; this is near-simultaneous software triggering, not hardware synchronization.",
        }
        (episode_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self.record_states_file = (episode_dir / "states.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self.record_writer = csv.writer(self.record_states_file)
        self.record_writer.writerow(
            [
                "timestamp_monotonic",
                "timestamp_system",
                "info",
                "motion_ready",
                "tele_fresh",
                "arm_enable",
                "arm_control_enabled",
                "joint_position",
                "joint_velocity",
                "command_position",
                "gravity_tau",
                "tau_command",
                "left_wrist_pose",
                "right_wrist_pose",
                "left_trigger",
                "right_trigger",
                "gripper_command",
                "record_button",
                "enable_button",
            ]
        )
        self.record_states_file.flush()

        self.recording = True
        self.record_episode_dir = episode_dir
        self.record_samples = 0

        print(f"[R1-A7 VR G1IK REAL] RECORDING START: {episode_dir}")
        print(f"[R1-A7 VR G1IK REAL] episode_id={episode_id}")
        print(
            "[R1-A7 VR G1IK REAL] one-shot recording window: "
            f"{duration_s:.1f}s"
        )

        if self.args.record_d435i:
            self.d435i_record_file = episode_dir / self.args.d435i_filename
            d435i_log_path = episode_dir / "d435i_record.log"
            try:
                self.d435i_record_log_file = d435i_log_path.open(
                    "w", encoding="utf-8", buffering=1
                )
                cmd = [
                    self.args.rs_record_bin,
                    "-f",
                    str(self.d435i_record_file),
                    "-t",
                    str(max(1, int(round(duration_s)))),
                ]
                self.d435i_record_process = subprocess.Popen(
                    cmd,
                    stdout=self.d435i_record_log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                print(
                    "[R1-A7 VR G1IK REAL] D435i RECORDING START: "
                    f"pid={self.d435i_record_process.pid} file={self.d435i_record_file}"
                )
            except Exception as exc:
                self.d435i_record_process = None
                if self.d435i_record_log_file is not None:
                    self.d435i_record_log_file.close()
                    self.d435i_record_log_file = None
                print(
                    "[R1-A7 VR G1IK REAL] D435i RECORDING START FAILED: "
                    f"{type(exc).__name__}: {exc}"
                )

    def _stop_d435i_recorder(self) -> None:
        proc = self.d435i_record_process
        if proc is None:
            if self.d435i_record_log_file is not None:
                self.d435i_record_log_file.close()
                self.d435i_record_log_file = None
            return

        if proc.poll() is None:
            # Match the user's normal Ctrl+C shutdown path so rs-record gets a
            # chance to finalize the SQLite DB3 cleanly.
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

        rc = proc.returncode
        print(
            "[R1-A7 VR G1IK REAL] D435i RECORDING STOP: "
            f"returncode={rc} file={self.d435i_record_file}"
        )
        self.d435i_record_process = None

        if self.d435i_record_log_file is not None:
            self.d435i_record_log_file.flush()
            self.d435i_record_log_file.close()
            self.d435i_record_log_file = None

    def _record_stop(self) -> None:
        if not self.recording:
            self._stop_d435i_recorder()
            return

        elapsed = 0.0
        if self.record_start_monotonic is not None:
            elapsed = max(0.0, time.monotonic() - self.record_start_monotonic)

        print(
            "[R1-A7 VR G1IK REAL] RECORDING STOP: "
            f"{self.record_episode_dir} samples={self.record_samples} "
            f"elapsed={elapsed:.3f}s"
        )

        self.recording = False
        if self.record_states_file is not None:
            self.record_states_file.flush()
            self.record_states_file.close()
        self.record_states_file = None
        self.record_writer = None

        self._stop_d435i_recorder()
        self.record_start_monotonic = None
        self.record_start_system = None

        print(
            "[R1-A7 VR G1IK REAL] Recording finished. "
            "VR remains active; press X again to start the next episode."
        )

    def _maybe_auto_stop_recording(self, now: float) -> None:
        if not self.recording or self.record_start_monotonic is None:
            return

        # Report an unexpected camera-recorder exit, but keep robot CSV active
        # for the full requested window so the episode remains diagnosable.
        if (
            self.d435i_record_process is not None
            and self.d435i_record_process.poll() is not None
            and not self.d435i_early_exit_reported
        ):
            elapsed = max(0.0, now - self.record_start_monotonic)
            if elapsed + 1.0 < float(self.args.record_duration_s):
                self.d435i_early_exit_reported = True
                print(
                    "[R1-A7 VR G1IK REAL] WARNING: D435i recorder exited early "
                    f"at {elapsed:.2f}s returncode={self.d435i_record_process.returncode}; "
                    f"check {self.record_episode_dir / 'd435i_record.log'}"
                )

        if now - self.record_start_monotonic >= max(
            0.1, float(self.args.record_duration_s)
        ):
            print(
                "[R1-A7 VR G1IK REAL] recording duration reached; "
                "stopping robot data and D435i automatically"
            )
            self._record_stop()

    def _maybe_toggle_recording(self, tele) -> None:
        if tele is None or self.args.record_button == "none":
            return
        if not self._button_rising_edge(tele, self.args.record_button):
            return

        if self.recording:
            elapsed = 0.0
            if self.record_start_monotonic is not None:
                elapsed = max(0.0, time.monotonic() - self.record_start_monotonic)
            remaining = max(0.0, float(self.args.record_duration_s) - elapsed)
            print(
                "[R1-A7 VR G1IK REAL] recording already active; "
                f"X ignored, remaining={remaining:.1f}s"
            )
            return

        self._record_start()

    @staticmethod
    def _json_array(value) -> str:
        if value is None:
            return ""
        return json.dumps(np.asarray(value, dtype=float).tolist())

    def _record_sample(
        self,
        tele,
        state_q: np.ndarray,
        state_dq: np.ndarray,
        command_q: np.ndarray,
        gripper_command_q: Optional[np.ndarray],
        info: str,
        motion_ready: bool,
        tele_fresh: bool,
        arm_enable: bool,
    ) -> None:
        if not self.recording or self.record_writer is None:
            return
        self.record_writer.writerow(
            [
                f"{time.monotonic():.6f}",
                f"{time.time():.6f}",
                info,
                bool(motion_ready),
                bool(tele_fresh),
                bool(arm_enable),
                bool(self.arm_control_enabled),
                self._json_array(state_q),
                self._json_array(state_dq),
                self._json_array(command_q),
                self._json_array(self.gravity_comp.last_tau),
                self._json_array(self.last_arm_tau_cmd),
                self._json_array(getattr(tele, "left_wrist_pose", None)),
                self._json_array(getattr(tele, "right_wrist_pose", None)),
                float(getattr(tele, "left_ctrl_triggerValue", 10.0)) if tele is not None else 10.0,
                float(getattr(tele, "right_ctrl_triggerValue", 10.0)) if tele is not None else 10.0,
                self._json_array(gripper_command_q),
                self._button_pressed(tele, self.args.record_button) if tele is not None else False,
                self._button_pressed(tele, self.args.arm_enable_button) if tele is not None else False,
            ]
        )
        self.record_samples += 1

    def _hold_current_after_stale(self, state_q: np.ndarray) -> tuple[np.ndarray, str]:
        if self.teleop_active:
            print("[R1-A7 VR G1IK REAL] Quest pose stream stale; holding current arm q and waiting for fresh poses")
        if self.args.arm_enable_mode == "toggle" and self.arm_control_enabled:
            print("[R1-A7 VR G1IK REAL] Quest stale; arm control toggled OFF for safe re-entry")
        self.arm_control_enabled = False
        self.prev_button_state[self.args.arm_enable_button] = False
        self.teleop_active = False
        self.prev_motion_ready = False
        self.last_arm_enable = False
        self.ik_zero_q = None
        self.home_q = state_q.copy()
        self.command_q = state_q.copy()
        self.arm_controller.reset(state_q)
        self.arm_hold_q = state_q.copy()
        self.arm_hold_strong = False
        self.arm_disabled_hold_active = False
        self.rearm_until = None
        self.last_pose_signature = None
        self.last_pose_update_time = None
        self.pose_frozen_reported = False
        return state_q.copy(), "stale_hold"

    def _hold_current_after_vr_exit(self, state_q: np.ndarray, reason: str) -> tuple[np.ndarray, str]:
        """Stop commanding the arms whenever VR tracking drops.

        Pressing A off is the intentional hold-current-position path. A VR
        browser exit or connection break is treated differently: clear teleop
        state and release LowCmd so the robot is no longer controlled by stale
        VR data.
        """
        if self.teleop_active or self.prev_motion_ready:
            print(f"[R1-A7 VR G1IK REAL] VR tracking unavailable ({reason}); releasing arm control")
        if self.args.arm_enable_mode == "toggle" and self.arm_control_enabled:
            print("[R1-A7 VR G1IK REAL] VR tracking lost; arm control toggled OFF for safe re-entry")
        self.arm_control_enabled = False
        self.prev_button_state[self.args.arm_enable_button] = False
        self.teleop_active = False
        self.prev_motion_ready = False
        self.last_arm_enable = False
        self.ik_zero_q = None
        self.home_q = state_q.copy()
        self.command_q = state_q.copy()
        self.arm_controller.reset(state_q)
        self.arm_hold_q = state_q.copy()
        self.arm_hold_strong = False
        self.arm_disabled_hold_active = False
        self.rearm_until = None
        self.last_pose_signature = None
        self.last_pose_update_time = None
        self.pose_frozen_reported = False
        self.vr_disconnected_uncontrolled = True
        return state_q.copy(), f"{reason}_uncontrolled"

    def _begin_rearm(self, state_q: np.ndarray, now: float) -> None:
        self.home_q = state_q.copy()
        self.command_q = state_q.copy()
        self.arm_controller.reset(state_q)
        self._mark_lowcmd_active()
        self.arm_hold_strong = False
        self.arm_disabled_hold_active = False
        self.ik_zero_q = None
        if self.arm_ik is not None and hasattr(self.arm_ik, "reset_target_calibration"):
            self.arm_ik.reset_target_calibration(state_q)
        self.teleop_active = True
        self.rearm_until = now + max(0.0, self.args.rearm_hold_time)
        print("[R1-A7 VR G1IK REAL] fresh Quest poses; rearm holding current q:", np.round(state_q, 3).tolist())

    def run(self) -> int:
        self.init()
        assert self.tv is not None
        assert self.arm_ik is not None
        deadline = time.monotonic() + max(0.0, self.args.duration)
        last_loop = time.monotonic()
        print("[R1-A7 VR G1IK REAL] waiting for robot lowstate and Quest controller poses ...")
        try:
            while not self.done:
                now = time.monotonic()
                if self.args.duration > 0 and now >= deadline:
                    break

                self._maybe_auto_stop_recording(now)

                state = self._read_arm_qdq()
                if state is None:
                    if now - self.last_print >= max(0.5, self.args.print_period):
                        self.last_print = now
                        print(
                            "[R1-A7 VR G1IK REAL] waiting_lowstate "
                            f"count={self.lowstate_count} interface={self.args.interface} "
                            f"topic={self.args.state_topic}"
                        )
                        print(
                            "  DIAG: no rt/lowstate received yet; verify robot power, "
                            "control Ethernet, interface name, domain_id, and topic."
                        )
                    time.sleep(0.02)
                    continue

                state_q, state_dq = state
                if self.home_q is None:
                    self.home_q = state_q.copy()
                    self.command_q = state_q.copy()
                    self.arm_controller.reset(state_q)
                    print("[R1-A7 VR G1IK REAL] calibrated robot arm q:", np.round(state_q, 3).tolist())

                tele = None
                motion_ready = False
                tele_fresh = False
                target_q: Optional[np.ndarray] = None
                info = "waiting_quest_pose"
                arm_enable = False

                try:
                    tele = self.tv.get_tele_data()
                    motion_ready = bool(tele.motion_data_ready)
                    self._maybe_print_vr_debug(tele, now)

                    # Highest-priority controller command: right B requests HOME.
                    # It is intentionally handled before X recording and A gating.
                    if self._home_return_button_rising(tele):
                        if self.recording:
                            print(
                                "[R1-A7 VR G1IK REAL] stopping active recording before HOME return"
                            )
                            self._record_stop()
                        self.arm_control_enabled = False
                        self.teleop_active = False
                        self.last_arm_enable = False
                        self._stop_gripper_controller()
                        self._release()
                        print(
                            "[R1-A7 VR G1IK REAL] LowCmd released; "
                            "leaving VR for external AUTO_HOME V2"
                        )
                        break

                    self._maybe_toggle_recording(tele)
                    tele_fresh = self._teleop_data_fresh(tele, now)
                    arm_enable = self._arm_enable_pressed(tele)
                    self.last_tele_error = None
                except Exception as exc:
                    self.last_tele_error = f"{type(exc).__name__}: {exc}"
                    target_q, info = self._hold_current_after_vr_exit(state_q, "tele_data_error")

                if tele is not None and self.last_tele_error is None:
                    if not motion_ready:
                        target_q, info = self._hold_current_after_vr_exit(state_q, "waiting_quest_pose")
                    elif motion_ready and not tele_fresh:
                        target_q, info = self._hold_current_after_vr_exit(state_q, "stale")
                    elif not arm_enable and self.vr_disconnected_uncontrolled:
                        target_q = state_q.copy()
                        info = "waiting_arm_enable_uncontrolled"
                    elif not arm_enable:
                        target_q, info = self._hold_current_after_enable_release(state_q)
                    elif tele_fresh:
                        self.arm_hold_q = None
                        self.last_arm_enable = True
                        if not self.teleop_active:
                            self._begin_rearm(state_q, now)
                        self.prev_motion_ready = True
                        try:
                            sol_q, _sol_tau = self.arm_ik.solve_ik(
                                tele.left_wrist_pose,
                                tele.right_wrist_pose,
                                state_q,
                                state_dq,
                            )
                            sol_q = np.asarray(sol_q, dtype=float).reshape(14)
                            if self.rearm_until is not None and now < self.rearm_until:
                                self.home_q = state_q.copy()
                                self.command_q = state_q.copy()
                                self.arm_controller.reset(state_q)
                                self.ik_zero_q = sol_q.copy()
                                target_q = state_q.copy()
                                info = "rearm_hold"
                            elif self.rearm_until is not None:
                                self.home_q = state_q.copy()
                                self.command_q = state_q.copy()
                                self.arm_controller.reset(state_q)
                                self.ik_zero_q = sol_q.copy()
                                self.rearm_until = None
                                target_q = state_q.copy()
                                info = "calibrated_ik_zero"
                            else:
                                target_q, info = self._retarget_solution(sol_q)
                            self.last_ik_error = None
                        except Exception as exc:
                            self.last_ik_error = f"{type(exc).__name__}: {exc}"
                            target_q = state_q.copy()
                            info = "ik_error_hold"

                # Arm and gripper use the same freshness gate. When XR data is
                # stale or unavailable, the official Dex1 controller holds the
                # measured gripper positions instead of applying a new target.
                self._update_gripper_inputs(
                    tele,
                    active=bool(
                        tele is not None
                        and motion_ready
                        and tele_fresh
                        and arm_enable
                        and self.last_tele_error is None
                    ),
                )

                dt = now - last_loop
                last_loop = now
                if self.vr_disconnected_uncontrolled and not arm_enable:
                    command_q = state_q.copy()
                    self._release_for_vr_exit_once()
                    self._record_sample(
                        tele,
                        state_q,
                        state_dq,
                        command_q,
                        None,
                        info,
                        motion_ready,
                        tele_fresh,
                        arm_enable,
                    )
                    if now - self.last_print >= self.args.print_period:
                        self.last_print = now
                        print(
                            f"[R1-A7 VR G1IK REAL] {info} "
                            f"lowstate_count={self.lowstate_count} motion_ready={motion_ready} "
                            f"tele_fresh={tele_fresh} arm_enable={arm_enable} "
                            "lowcmd=RELEASED"
                        )
                    time.sleep(max(0.0, 1.0 / max(1.0, self.args.hz)))
                    continue

                command_q = self._step_command(state_q, target_q, dt)
                gripper_state_q = self._read_lowcmd_gripper_q() if self.args.enable_gripper and self.args.gripper_mode == "lowcmd" else None
                gripper_command_q = (
                    self._step_lowcmd_gripper(gripper_state_q, dt)
                    if gripper_state_q is not None
                    else None
                )
                self._publish(command_q, gripper_command_q)
                self._record_sample(
                    tele,
                    state_q,
                    state_dq,
                    command_q,
                    gripper_command_q,
                    info,
                    motion_ready,
                    tele_fresh,
                    arm_enable,
                )

                if now - self.last_print >= self.args.print_period:
                    self.last_print = now
                    cmd_err = float(np.max(np.abs(command_q - state_q)))
                    tgt_err = float(np.max(np.abs(target_q - state_q))) if target_q is not None else 0.0
                    lowstate_age = (
                        max(0.0, now - self.last_lowstate_rx_time)
                        if self.last_lowstate_rx_time is not None
                        else float("inf")
                    )
                    left_q = " ".join(f"{n}={v:+.3f}" for n, v in zip(ARM_NAMES[:4], state_q[:4]))
                    left_cmd = " ".join(f"{n}={v:+.3f}" for n, v in zip(ARM_NAMES[:4], command_q[:4]))
                    right_q = " ".join(f"{n}={v:+.3f}" for n, v in zip(ARM_NAMES[7:11], state_q[7:11]))
                    right_cmd = " ".join(f"{n}={v:+.3f}" for n, v in zip(ARM_NAMES[7:11], command_q[7:11]))
                    gripper_connected, gripper_ready, left_grip_q, right_grip_q, left_grip_cmd, right_grip_cmd = self._gripper_status()
                    print(
                        f"[R1-A7 VR G1IK REAL] {info} "
                        f"lowstate_count={self.lowstate_count} lowstate_age={lowstate_age:.3f}s "
                        f"motion_ready={motion_ready} tele_fresh={tele_fresh} arm_enable={arm_enable} "
                        f"arm_toggle={self.arm_control_enabled} recording={self.recording} "
                        f"pose_delta={self.last_pose_delta:.6f} pose_age={self._pose_age_text(now)} "
                        f"cmd_err={cmd_err:.3f} tgt_err={tgt_err:.3f}"
                    )
                    if self.last_tele_error:
                        print(f"  tele_error: {self.last_tele_error}")
                    if self.last_ik_error:
                        print(f"  ik_error  : {self.last_ik_error}")
                    if self.last_limit_diag:
                        print(f"  arm_limit : {self.last_limit_diag}")
                    if self.args.enable_gravity_comp:
                        print(
                            "  gravity   : "
                            f"enabled={self.gravity_comp.available} "
                            f"tau={np.round(self.gravity_comp.last_tau, 3).tolist()}"
                        )
                        if self.gravity_comp.last_error:
                            print(f"  gravity_error: {self.gravity_comp.last_error}")
                    print(f"  left_q    : {left_q}")
                    print(f"  left_cmd  : {left_cmd}")
                    print(f"  right_q   : {right_q}")
                    print(f"  right_cmd : {right_cmd}")
                    if self.args.enable_gripper:
                        print(
                            "  gripper   : "
                            f"connected={gripper_connected} ready={gripper_ready} "
                            f"trigger_L={self.last_left_trigger:.3f} trigger_R={self.last_right_trigger:.3f} "
                            f"state_L={left_grip_q:.3f} state_R={right_grip_q:.3f} "
                            f"cmd_L={left_grip_cmd:.3f} cmd_R={right_grip_cmd:.3f} "
                            f"contact_hold_L={bool(self.lowcmd_gripper_contact_hold[0])} "
                            f"contact_hold_R={bool(self.lowcmd_gripper_contact_hold[1])}"
                        )
                        if self.last_gripper_error:
                            print(f"  grip_error: {self.last_gripper_error}")
                time.sleep(max(0.0, 1.0 / max(1.0, self.args.hz)))
        finally:
            self._record_stop()
            self._stop_gripper_controller()
            self._release()
            try:
                self.tv.close()
            except Exception:
                pass
        if self.home_return_requested:
            print(
                "[R1-A7 VR G1IK REAL] VR cleanup complete; "
                "requesting verified external AUTO_HOME V2 (exit code 42)"
            )
            return 42
        return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real R1-A7 dual-arm Quest/Vuer control through G1_29 IK")
    parser.add_argument("--interface", default="enp6s0")
    parser.add_argument("--domain_id", type=int, default=0)
    parser.add_argument("--state_topic", default="rt/lowstate")
    parser.add_argument("--command_topic", default="rt/lowcmd")
    parser.add_argument("--host_ip", default=os.getenv("HOST_IP", "192.168.1.103"))
    parser.add_argument("--left_arm_indices", default="15,16,17,18,19,20,21")
    parser.add_argument("--right_arm_indices", default="22,23,24,25,26,27,28")
    parser.add_argument("--lowcmd_hold_indices", default="13,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30")
    parser.add_argument(
        "--fixed_hold_indices",
        default="13",
        help="motor indices held at their startup lowstate q instead of following current q; 13 is waist on R1-A7",
    )
    parser.add_argument(
        "--enable_gripper",
        action="store_true",
        help="control left/right grippers with the matching Quest index triggers",
    )
    parser.add_argument(
        "--gripper_mode",
        choices=("lowcmd", "dex1_dds"),
        default="lowcmd",
        help="lowcmd controls internal R1-A7 hand motors 31/33; dex1_dds uses the external Dex1 DDS service",
    )
    parser.add_argument(
        "--lowcmd_gripper_indices",
        default=R1A7_GRIPPER_INDICES,
        help="left,right internal gripper motor indices in LowCmd/LowState",
    )
    parser.add_argument(
        "--lowcmd_gripper_relative",
        action=argparse.BooleanOptionalAction,
        default=R1A7_GRIPPER_RELATIVE,
        help=(
            "interpret lowcmd gripper open/close q values as offsets from startup gripper q; "
            "R1-A7 internal grippers default to absolute motor-q coordinates"
        ),
    )
    parser.add_argument(
        "--lowcmd_gripper_left_open_q",
        type=float,
        default=R1A7_GRIPPER_LEFT_OPEN_Q,
    )
    parser.add_argument(
        "--lowcmd_gripper_left_close_q",
        type=float,
        default=R1A7_GRIPPER_LEFT_CLOSE_Q,
    )
    parser.add_argument(
        "--lowcmd_gripper_right_open_q",
        type=float,
        default=R1A7_GRIPPER_RIGHT_OPEN_Q,
    )
    parser.add_argument(
        "--lowcmd_gripper_right_close_q",
        type=float,
        default=R1A7_GRIPPER_RIGHT_CLOSE_Q,
    )
    parser.add_argument(
        "--lowcmd_gripper_extra_margin",
        type=float,
        default=R1A7_GRIPPER_EXTRA_MARGIN,
    )
    parser.add_argument(
        "--lowcmd_gripper_velocity_limit",
        type=float,
        default=R1A7_GRIPPER_VELOCITY_LIMIT,
    )
    parser.add_argument(
        "--lowcmd_gripper_kp",
        type=float,
        default=R1A7_GRIPPER_KP,
    )
    parser.add_argument(
        "--lowcmd_gripper_kd",
        type=float,
        default=R1A7_GRIPPER_KD,
    )
    parser.add_argument(
        "--lowcmd_gripper_contact_hold",
        action=argparse.BooleanOptionalAction,
        default=R1A7_GRIPPER_CONTACT_HOLD,
        help="lock each lowcmd gripper near the contact position after a stalled full trigger pull",
    )
    parser.add_argument(
        "--lowcmd_gripper_contact_trigger_alpha",
        type=float,
        default=R1A7_GRIPPER_CONTACT_TRIGGER_ALPHA,
        help="trigger close ratio above which contact hold can engage",
    )
    parser.add_argument(
        "--lowcmd_gripper_contact_error",
        type=float,
        default=R1A7_GRIPPER_CONTACT_ERROR,
        help="minimum target-state q error treated as object blockage",
    )
    parser.add_argument(
        "--lowcmd_gripper_contact_stall_eps",
        type=float,
        default=R1A7_GRIPPER_CONTACT_STALL_EPS,
        help="maximum q change per control frame treated as stalled gripper motion",
    )
    parser.add_argument(
        "--lowcmd_gripper_contact_stall_time",
        type=float,
        default=R1A7_GRIPPER_CONTACT_STALL_TIME,
        help="seconds of stalled motion before contact hold engages",
    )
    parser.add_argument(
        "--lowcmd_gripper_contact_hold_bias",
        type=float,
        default=R1A7_GRIPPER_CONTACT_HOLD_BIAS,
        help="small extra closing q offset from the detected contact position to keep a soft grip",
    )
    parser.add_argument(
        "--lowcmd_gripper_test_only",
        action="store_true",
        help="reserved flag for command wrappers; this script always keeps arms safe while controlling grippers",
    )
    parser.add_argument(
        "--gripper_fps",
        type=float,
        default=200.0,
        help="Dex1 command loop frequency used by the official gripper controller",
    )
    parser.add_argument(
        "--disable_gripper_filter",
        action="store_true",
        help="disable the official Dex1 weighted moving filter",
    )
    parser.add_argument("--enter_debug_mode", action="store_true")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--hz", type=float, default=250.0)
    parser.add_argument("--print_period", type=float, default=0.25)
    parser.add_argument(
        "--arm_enable_button",
        choices=(
            "none",
            "left_ctrl_aButton",
            "left_ctrl_bButton",
            "left_ctrl_squeeze",
            "left_ctrl_thumbstick",
            "right_ctrl_aButton",
            "right_ctrl_bButton",
            "right_ctrl_squeeze",
            "right_ctrl_thumbstick",
        ),
        default="right_ctrl_aButton",
        help="controller button used for arm and gripper command gating; use none to disable the gate",
    )
    parser.add_argument(
        "--arm_enable_mode",
        choices=("toggle", "hold"),
        default="toggle",
        help="toggle means one press enables control and the next press holds; hold keeps the old hold-to-run behavior",
    )
    parser.add_argument(
        "--home_button",
        choices=(
            "none",
            "left_ctrl_aButton",
            "left_ctrl_bButton",
            "left_ctrl_squeeze",
            "left_ctrl_thumbstick",
            "right_ctrl_aButton",
            "right_ctrl_bButton",
            "right_ctrl_squeeze",
            "right_ctrl_thumbstick",
        ),
        default="right_ctrl_bButton",
        help=(
            "Quest button that requests verified external AUTO_HOME V2; "
            "default right_ctrl_bButton is right B"
        ),
    )
    parser.add_argument(
        "--record_button",
        choices=(
            "none",
            "left_ctrl_aButton",
            "left_ctrl_bButton",
            "left_ctrl_squeeze",
            "left_ctrl_thumbstick",
            "right_ctrl_aButton",
            "right_ctrl_bButton",
            "right_ctrl_squeeze",
            "right_ctrl_thumbstick",
        ),
        default="left_ctrl_aButton",
        help="controller button used to toggle CSV recording; left_ctrl_aButton is Quest left X",
    )
    parser.add_argument(
        "--record_root",
        default="data/episodes",
        help="directory where VR button-toggle recordings are written",
    )
    parser.add_argument(
        "--record_episode_id",
        default="",
        help="episode base/start id; trailing number auto-increments on each X recording (e.g. right_safe_001 -> _002 -> _003)",
    )
    parser.add_argument(
        "--record_duration_s",
        type=float,
        default=60.0,
        help="one-shot recording duration after X; default 60 seconds",
    )
    parser.add_argument(
        "--record_d435i",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="start rs-record together with robot CSV when X is pressed",
    )
    parser.add_argument(
        "--rs_record_bin",
        default="rs-record",
        help="rs-record executable used for D435i DB3 capture",
    )
    parser.add_argument(
        "--d435i_filename",
        default="d435i.db3",
        help="D435i DB3 filename inside each episode directory",
    )
    parser.add_argument("--kp", type=float, default=12.0)
    parser.add_argument("--kd", type=float, default=0.8)
    parser.add_argument("--kp_low", type=float, default=80.0)
    parser.add_argument("--kd_low", type=float, default=3.0)
    parser.add_argument("--kp_wrist", type=float, default=40.0)
    parser.add_argument("--kd_wrist", type=float, default=1.5)
    parser.add_argument("--hold_kp", type=float, default=8.0)
    parser.add_argument("--hold_kd", type=float, default=0.6)
    parser.add_argument(
        "--strong_hold_after_arm_disable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="after A toggles arm control off, hold arm motors with movement gains instead of hold_kp/hold_kd",
    )
    parser.add_argument(
        "--enable_gravity_comp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="add limited Pinocchio gravity feedforward from the R1-A7 A7 URDF inertial model into arm motor tau",
    )
    parser.add_argument(
        "--gravity_model_urdf",
        default=R1A7_A7_URDF,
        help="R1-A7/A7 URDF used for Pinocchio gravity compensation",
    )
    parser.add_argument(
        "--gravity_comp_scale",
        type=float,
        default=0.20,
        help="scale applied to model gravity torque; start small and increase only after safe validation",
    )
    parser.add_argument(
        "--gravity_comp_tau_limit",
        type=float,
        default=3.0,
        help="absolute per-joint tau feedforward limit in LowCmd units/Nm-equivalent",
    )
    parser.add_argument("--gravity_model_waist_index", type=int, default=13)
    parser.add_argument("--gravity_model_head_pitch_index", type=int, default=29)
    parser.add_argument("--gravity_model_head_yaw_index", type=int, default=30)
    parser.add_argument(
        "--shoulder_pitch_extra_tau",
        type=float,
        default=0.0,
        help=(
            "extra feedforward tau added to both shoulder_pitch joints after gravity compensation; "
            "use small values such as -0.3 first, 0 disables it"
        ),
    )
    parser.add_argument(
        "--total_tau_limit",
        type=float,
        default=4.0,
        help="absolute final arm tau limit after gravity compensation and shoulder_pitch_extra_tau; 0 disables clipping",
    )
    parser.add_argument("--max_speed_rad_s", type=float, default=0.06)
    parser.add_argument("--max_command_lead", type=float, default=0.06)
    parser.add_argument(
        "--elbow_command_step_limit",
        type=float,
        default=0.008,
        help="maximum command-position change per control frame for left/right elbow; 0 disables this extra clamp",
    )
    parser.add_argument(
        "--r1a7_extra_elbow_limit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="optional non-official R1-A7 elbow command step limiter; disabled by default for official-style control",
    )
    parser.add_argument(
        "--arm_velocity_limit",
        type=float,
        default=20.0,
        help="G1-style velocity clip in rad/s; official G1 uses much higher values",
    )
    parser.add_argument("--max_joint_offset_rad", type=float, default=1.40)
    parser.add_argument(
        "--joint_limit_margin_rad",
        type=float,
        default=0.0,
        help="global soft margin kept away from every R1-A7 arm joint hard limit",
    )
    parser.add_argument(
        "--shoulder_pitch_low_margin_rad",
        type=float,
        default=0.0,
        help="extra soft margin above both shoulder_pitch lower limits to avoid high-lift saturation",
    )
    parser.add_argument("--ik_delta_scale", type=float, default=1.0)
    parser.add_argument(
        "--ik_joint_scales",
        default="0.95,0.80,0.60,0.40,0.70,0.95,0.75,0.95,0.80,0.60,0.40,0.70,0.95,0.65",
        help=(
            "14 comma-separated per-joint multipliers for relative G1 IK deltas. "
            "Order matches left 7 arm joints then right 7 arm joints."
        ),
    )
    parser.add_argument(
        "--stale_pose_timeout",
        type=float,
        default=0.25,
        help="seconds before unchanged Quest poses are treated as stale and the real arms hold current q",
    )
    parser.add_argument(
        "--frozen_pose_hold_timeout",
        type=float,
        default=0.15,
        help="seconds of unchanged Quest pose while active before holding current q to avoid VR-exit jumps",
    )
    parser.add_argument(
        "--rearm_hold_time",
        type=float,
        default=1.0,
        help="seconds to hold current arm q and refresh IK zero after Quest poses become fresh again",
    )
    parser.add_argument(
        "--pose_change_eps",
        type=float,
        default=1e-4,
        help="minimum wrist-pose matrix change used to detect fresh Quest controller samples",
    )
    parser.add_argument(
        "--treat_motion_ready_as_fresh",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="keep solving IK whenever TeleVuer reports motion_data_ready, even if pose matrices do not change",
    )
    parser.add_argument(
        "--debug_vr_data",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="print TeleVuer motion/button fields for diagnosing Quest WebXR input",
    )
    parser.add_argument("--debug_vr_period", type=float, default=1.0)
    parser.add_argument("--limit_diag_eps", type=float, default=1e-3)
    parser.add_argument(
        "--g1_official_adapter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use the R1-A7 adapter modeled after Unitree G1_29_ArmController target clipping",
    )
    parser.add_argument(
        "--official_ik_target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="send official G1_29 IK q target directly through the R1-A7 joint mapping, only clipping to R1-A7 limits",
    )
    parser.add_argument(
        "--g1_control_dt",
        type=float,
        default=1.0 / 250.0,
        help="minimum control dt used by the G1-style velocity clip",
    )
    parser.add_argument(
        "--g1_style_gains",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use Unitree G1-style low/wrist gains for arm joints",
    )
    parser.add_argument("--g1_style_velocity_clip", action="store_true")
    parser.add_argument(
        "--absolute_ik",
        action="store_true",
        help="use raw G1_29 IK joint targets instead of relative retargeting",
    )
    parser.add_argument("--assume_yes", action="store_true")
    return parser


def cleanup_vuer_port() -> None:
    """Release old Vuer websocket server on port 8012 before starting.

    Prevents: OSError: [Errno 98] address already in use
    when restarting Quest VR teleoperation after exiting VR.
    """
    try:
        pid_text = subprocess.check_output(
            "lsof -ti:8012",
            shell=True,
            stderr=subprocess.DEVNULL,
        ).decode().strip()

        if pid_text:
            print(f"[cleanup] old Vuer process on 8012: {pid_text}")
            for pid in pid_text.split():
                subprocess.run(
                    ["kill", "-9", pid],
                    check=False,
                )
            time.sleep(1.0)
            print("[cleanup] Vuer port 8012 released")

    except subprocess.CalledProcessError:
        pass


def main() -> int:
    cleanup_vuer_port()
    args = build_arg_parser().parse_args()
    print("WARNING: This will publish rt/lowcmd commands to the real R1-A7 dual arms.")
    if args.enable_gripper:
        print("WARNING: Quest index triggers will also command both real R1-A7 grippers.")
        print("At the first fresh controller frame, released triggers command the grippers OPEN.")
    print("Keep the emergency stop ready and keep both arm workspaces clear.")
    print("Initial limits are conservative: low gains, low speed, and joint offset clamp.")
    if not args.assume_yes:
        if not sys.stdin.isatty():
            print("[R1-A7 VR G1IK REAL] interactive confirmation required")
            return 2
        answer = input("Type ENABLE to continue: ").strip()
        if answer != "ENABLE":
            print("[R1-A7 VR G1IK REAL] aborted")
            return 2

    node = R1A7VRDualArmG1IKReal(args)

    def _handle_signal(_signum, _frame):
        node.done = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    rc = node.run()

    # B-HOME safety cleanup has already completed inside node.run():
    #   - active recording stopped
    #   - D435i recorder stopped
    #   - gripper controller stopped
    #   - LowCmd gains released
    #   - TeleVuer close requested
    #
    # TeleVuer/Vuer background threads can keep the Python interpreter alive
    # even after run() returns.  For the dedicated HOME-request path only,
    # terminate the VR process immediately so the outer launcher receives
    # exit code 42 and can hand control to the verified AUTO_HOME V2 process.
    if rc == 42:
        print(
            "[R1-A7 VR G1IK REAL] HOME cleanup finished; "
            "forcing VR process exit with code 42 for launcher handoff"
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(42)

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
