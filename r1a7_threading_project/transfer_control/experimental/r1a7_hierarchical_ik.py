#!/usr/bin/env python3
"""Strict-priority differential IK for the R1-A7 experimental console.

Cartesian position is the primary task. Tool orientation is applied only in
the position-task null space, and right elbow/wrist posture is applied only in
the remaining null space. This module intentionally lives outside the official
XR teleoperation checkout and the validated Cartesian controller.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin

from teleop.robot_control.robot_arm_ik import R1A7_ArmIK


class R1A7HierarchicalArmIK(R1A7_ArmIK):
    """R1-A7 IK with explicit position, orientation, posture priorities."""

    POSITION_DAMPING = 2.0e-4
    ORIENTATION_DAMPING = 2.0e-3
    SOLVE_ITERATIONS = 8
    FINAL_POSITION_ITERATIONS = 3
    POSITION_TOLERANCE_M = 2.0e-4
    MAX_ITERATION_STEP_RAD = 0.035

    @staticmethod
    def _damped_pinv(matrix: np.ndarray, damping: float) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        rows = matrix.shape[0]
        regularized = matrix @ matrix.T + (damping**2) * np.eye(rows)
        return matrix.T @ np.linalg.solve(regularized, np.eye(rows))

    @staticmethod
    def _rotation_error_world(current: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.asarray(pin.log3(target @ current.T), dtype=float).reshape(3)

    def _frame_pose_and_jacobian(
        self,
        q: np.ndarray,
        frame_id: int,
        arm_slice: slice,
    ) -> tuple[np.ndarray, np.ndarray]:
        pin.framesForwardKinematics(
            self.reduced_robot.model,
            self.reduced_robot.data,
            q,
        )
        pose = np.asarray(
            self.reduced_robot.data.oMf[frame_id].homogeneous,
            dtype=float,
        ).copy()
        jacobian = pin.computeFrameJacobian(
            self.reduced_robot.model,
            self.reduced_robot.data,
            q,
            frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )[:, arm_slice]
        return pose, np.asarray(jacobian, dtype=float)

    def _arm_step(
        self,
        q: np.ndarray,
        target: np.ndarray,
        frame_id: int,
        arm_slice: slice,
        rotation_weight: float,
        posture_delta: np.ndarray,
        final_position_only: bool,
    ) -> tuple[np.ndarray, float, float]:
        pose, jacobian = self._frame_pose_and_jacobian(
            q,
            frame_id,
            arm_slice,
        )
        position_error = target[:3, 3] - pose[:3, 3]
        position_jacobian = jacobian[:3, :]
        position_pinv = self._damped_pinv(
            position_jacobian,
            self.POSITION_DAMPING,
        )
        position_step = position_pinv @ position_error
        position_null = np.eye(7) - position_pinv @ position_jacobian

        step = position_step
        rotation_error = self._rotation_error_world(
            pose[:3, :3],
            target[:3, :3],
        )
        if not final_position_only and rotation_weight > 0.0:
            rotation_jacobian = jacobian[3:, :]
            projected_rotation = rotation_jacobian @ position_null
            rotation_pinv = self._damped_pinv(
                projected_rotation,
                self.ORIENTATION_DAMPING,
            )
            rotation_residual = (
                rotation_weight * rotation_error
                - rotation_jacobian @ position_step
            )
            rotation_step = position_null @ rotation_pinv @ rotation_residual
            step = step + rotation_step
            posture_null = position_null @ (
                np.eye(7) - rotation_pinv @ projected_rotation
            )
            step = step + posture_null @ posture_delta

        max_abs = float(np.max(np.abs(step)))
        if max_abs > self.MAX_ITERATION_STEP_RAD:
            step *= self.MAX_ITERATION_STEP_RAD / max_abs
        return (
            step,
            float(np.linalg.norm(position_error)),
            float(np.linalg.norm(rotation_error)),
        )

    def solve_ik(
        self,
        left_wrist,
        right_wrist,
        current_lr_arm_motor_q=None,
        current_lr_arm_motor_dq=None,
        position_only=False,
        max_joint_step=None,
        rotation_weight=1.0,
        right_wrist_pitch_weight=0.0,
        right_wrist_posture_weight=0.0,
        right_wrist_posture_ref=None,
        right_elbow_posture_weight=0.0,
        right_elbow_posture_ref=None,
    ):
        del current_lr_arm_motor_dq, right_wrist_pitch_weight
        left_target = self._validated_pose(left_wrist, "left_wrist")
        right_target = self._validated_pose(right_wrist, "right_wrist")

        if current_lr_arm_motor_q is None:
            seed_q = np.asarray(self.init_data, dtype=float).reshape(14).copy()
        else:
            seed_q = np.asarray(current_lr_arm_motor_q, dtype=float).reshape(14).copy()
        if not np.all(np.isfinite(seed_q)):
            raise ValueError("current arm q contains non-finite values")

        lower = np.asarray(self.reduced_robot.model.lowerPositionLimit, dtype=float)
        upper = np.asarray(self.reduced_robot.model.upperPositionLimit, dtype=float)
        seed_q = np.clip(seed_q, lower, upper)
        q = seed_q.copy()

        if max_joint_step is None:
            max_joint_step = 1.0e6
        max_joint_step = float(max_joint_step)
        if not np.isfinite(max_joint_step) or max_joint_step <= 0.0:
            raise ValueError("max_joint_step must be finite and greater than zero")
        cycle_lower = np.maximum(lower, seed_q - max_joint_step)
        cycle_upper = np.minimum(upper, seed_q + max_joint_step)

        rotation_weight = float(rotation_weight)
        if not np.isfinite(rotation_weight) or rotation_weight < 0.0:
            raise ValueError("rotation_weight must be finite and non-negative")
        if position_only:
            rotation_weight = 0.0

        if right_wrist_posture_ref is None:
            right_wrist_posture_ref = seed_q[11:14]
        wrist_ref = np.asarray(right_wrist_posture_ref, dtype=float).reshape(3)
        if right_elbow_posture_ref is None:
            right_elbow_posture_ref = seed_q[10]
        elbow_ref = float(right_elbow_posture_ref)

        wrist_gain = min(max(float(right_wrist_posture_weight), 0.0), 1.0) * 0.08
        elbow_gain = min(max(float(right_elbow_posture_weight), 0.0), 1.0) * 0.08

        last_position_errors = (float("inf"), float("inf"))
        last_rotation_errors = (float("inf"), float("inf"))
        for iteration in range(self.SOLVE_ITERATIONS + self.FINAL_POSITION_ITERATIONS):
            final_position_only = iteration >= self.SOLVE_ITERATIONS

            left_posture = np.zeros(7, dtype=float)
            right_posture = np.zeros(7, dtype=float)
            right_posture[3] = elbow_gain * (elbow_ref - q[10])
            right_posture[4:7] = wrist_gain * (wrist_ref - q[11:14])

            left_step, left_pos_error, left_rot_error = self._arm_step(
                q,
                left_target,
                self.L_hand_id,
                slice(0, 7),
                rotation_weight,
                left_posture,
                final_position_only,
            )
            right_step, right_pos_error, right_rot_error = self._arm_step(
                q,
                right_target,
                self.R_hand_id,
                slice(7, 14),
                rotation_weight,
                right_posture,
                final_position_only,
            )
            q[:7] += left_step
            q[7:14] += right_step
            q = np.clip(q, cycle_lower, cycle_upper)
            last_position_errors = (left_pos_error, right_pos_error)
            last_rotation_errors = (left_rot_error, right_rot_error)

            if (
                final_position_only
                and max(last_position_errors) <= self.POSITION_TOLERANCE_M
            ):
                break

        if not np.all(np.isfinite(q)):
            raise RuntimeError("hierarchical IK returned non-finite joint values")

        self.init_data = q.copy()
        zero_velocity = np.zeros(self.reduced_robot.model.nv)
        tau_ff = pin.rnea(
            self.reduced_robot.model,
            self.reduced_robot.data,
            q,
            zero_velocity,
            zero_velocity,
        )
        self.last_hierarchy_metrics = {
            "left_position_error_mm": 1000.0 * last_position_errors[0],
            "right_position_error_mm": 1000.0 * last_position_errors[1],
            "left_rotation_error_deg": np.degrees(last_rotation_errors[0]),
            "right_rotation_error_deg": np.degrees(last_rotation_errors[1]),
        }
        return q, tau_ff
