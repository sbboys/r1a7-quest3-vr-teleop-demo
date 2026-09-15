#!/usr/bin/env python3

import time
import numpy as np

import r1a7_head_visual_servo_lowcmd as vs


STEP = 0.002


def print_pose(name, T):
    print(
        f"{name} xyz m =",
        np.round(T[:3, 3], 6).tolist(),
        flush=True,
    )


def run_case(
    name,
    axis,
    position_only,
    measured_q,
    measured_dq,
):
    print()
    print("=" * 72)
    print(
        f"{name}: axis={axis}, "
        f"position_only={position_only}"
    )
    print("=" * 72)

    # Fresh IK object so X/Y tests do not share smoothing history.
    ik = vs.R1A7_ArmIK(
        Unit_Test=False,
        Visualization=False,
    )

    ik.reset_target_calibration(
        measured_q
    )

    left0, right0 = vs.get_ee_poses(
        ik,
        measured_q,
    )

    target = right0.copy()
    target[axis, 3] += STEP

    print_pose("Current right EE", right0)
    print_pose("Target right EE ", target)

    axis_name = ["X", "Y", "Z"][axis]

    print(
        f"Requested +{axis_name} = "
        f"{STEP * 1000:.3f} mm"
    )

    q_seed = measured_q.copy()
    dq_seed = measured_dq.copy()

    for i in range(1, 11):

        solution, _ = ik.solve_ik(
            left0,
            target,
            q_seed,
            dq_seed,
            position_only=position_only,
            max_joint_step=vs.IK_MAX_JOINT_STEP,
            rotation_weight=vs.IK_ROTATION_WEIGHT,
        )

        solution = np.asarray(
            solution,
            dtype=float,
        ).reshape(14)

        _, predicted = vs.get_ee_poses(
            ik,
            solution,
        )

        cart_delta = (
            predicted[:3, 3]
            - right0[:3, 3]
        )

        cart_error = (
            target[:3, 3]
            - predicted[:3, 3]
        )

        joint_step = (
            solution - q_seed
        )

        total_joint_delta = (
            solution - measured_q
        )

        print(
            f"[{i:02d}] "
            f"pred delta mm = "
            f"{np.round(cart_delta * 1000.0, 4).tolist()}  "
            f"target error mm = "
            f"{np.round(cart_error * 1000.0, 4).tolist()}  "
            f"max joint step = "
            f"{np.max(np.abs(joint_step)):.6f} rad  "
            f"max total dq = "
            f"{np.max(np.abs(total_joint_delta)):.6f} rad",
            flush=True,
        )

        # Offline iterative IK only.
        # NO command is sent to the robot.
        q_seed = solution.copy()
        dq_seed = np.zeros_like(dq_seed)

    print()


def main():

    print(
        "READ-ONLY IK DIAGNOSTIC",
        flush=True,
    )

    print(
        "No rt/lowcmd publisher will be created.",
        flush=True,
    )

    vs.ChannelFactoryInitialize(
        vs.DOMAIN_ID,
        vs.INTERFACE,
    )

    crc = vs.CRC()

    state_buffer = vs.base.StateBuffer(
        crc
    )

    subscriber = vs.ChannelSubscriber(
        vs.STATE_TOPIC,
        vs.LowState_,
    )

    subscriber.Init(
        state_buffer.callback,
        10,
    )

    deadline = time.monotonic() + 10.0

    state = state_buffer.snapshot()

    while (
        state is None
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)
        state = state_buffer.snapshot()

    if state is None:
        raise RuntimeError(
            "No valid rt/lowstate."
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

    arm_q = np.asarray(
        arm_q,
        dtype=float,
    ).copy()

    arm_dq = np.asarray(
        arm_dq,
        dtype=float,
    ).copy()

    print(
        "Measured arm q:",
        np.round(arm_q, 4).tolist(),
        flush=True,
    )

    # Current production settings:
    # fixed full EE orientation.
    run_case(
        "X / current orientation constraint",
        0,
        False,
        arm_q,
        arm_dq,
    )

    run_case(
        "Y / current orientation constraint",
        1,
        False,
        arm_q,
        arm_dq,
    )

    # Diagnostic comparison only:
    # position-only IK.
    run_case(
        "X / position-only diagnostic",
        0,
        True,
        arm_q,
        arm_dq,
    )

    run_case(
        "Y / position-only diagnostic",
        1,
        True,
        arm_q,
        arm_dq,
    )


if __name__ == "__main__":
    main()
