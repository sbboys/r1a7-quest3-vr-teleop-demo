#!/usr/bin/env python3

import signal
import time
from pathlib import Path

import numpy as np

import r1a7_head_visual_servo_lowcmd as vs


STEP_M = +0.010
TEST_TIME_S = 4.0
RETURN_TIME_S = 4.0


def main():
    print("R1-A7 LOWCMD 5 mm EXECUTION TEST")
    print("NO camera / NO visual servo.")
    print("Test direction: model +X 5 mm")
    print()

    vs.base.validate_robot_interface(
        vs.INTERFACE
    )

    guard = vs.acquire_lowcmd_guard(
        Path(__file__).name,
        topic=vs.COMMAND_TOPIC,
    )

    stopped = False

    def stop_handler(_sig=None, _frame=None):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    output = None

    try:
        # --------------------------------------------------
        # DDS state
        # --------------------------------------------------
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

        q_start = arm_q.copy()

        print(
            "Measured arm q:",
            np.round(q_start, 5).tolist(),
            flush=True,
        )

        # --------------------------------------------------
        # MotionSwitcher
        # --------------------------------------------------
        vs.base.check_debug_mode(
            vs.MotionSwitcherClient
        )

        # --------------------------------------------------
        # Fresh IK
        # --------------------------------------------------
        ik = vs.R1A7_ArmIK(
            Unit_Test=False,
            Visualization=False,
        )

        ik.reset_target_calibration(
            q_start
        )

        left_start, right_start = (
            vs.get_ee_poses(
                ik,
                q_start,
            )
        )

        right_target = right_start.copy()
        right_target[0, 3] += STEP_M

        print(
            "Start right EE xyz:",
            np.round(
                right_start[:3, 3],
                6,
            ).tolist(),
            flush=True,
        )

        print(
            "Target right EE xyz:",
            np.round(
                right_target[:3, 3],
                6,
            ).tolist(),
            flush=True,
        )

        # One IK solve only.
        solution, _ = ik.solve_ik(
            left_start,
            right_target,
            q_start,
            arm_dq,
            position_only=False,
            max_joint_step=vs.IK_MAX_JOINT_STEP,
            rotation_weight=vs.IK_ROTATION_WEIGHT,
        )

        solution = np.asarray(
            solution,
            dtype=float,
        ).reshape(14)

        _, predicted_right = (
            vs.get_ee_poses(
                ik,
                solution,
            )
        )

        predicted_delta = (
            predicted_right[:3, 3]
            - right_start[:3, 3]
        )

        q_delta = solution - q_start

        print()
        print(
            "IK goal joint delta rad:",
            np.round(q_delta, 6).tolist(),
            flush=True,
        )

        print(
            "Max |joint delta| = "
            f"{np.max(np.abs(q_delta)):.6f} rad",
            flush=True,
        )

        print(
            "Predicted EE delta mm =",
            np.round(
                predicted_delta * 1000.0,
                4,
            ).tolist(),
            flush=True,
        )

        if (
            np.max(np.abs(q_delta))
            > 0.02
        ):
            raise RuntimeError(
                "Unexpectedly large joint goal."
            )

        # --------------------------------------------------
        # Publisher
        # --------------------------------------------------
        publisher = vs.ChannelPublisher(
            vs.COMMAND_TOPIC,
            vs.LowCmd_,
        )

        publisher.Init()

        output = vs.base.R1A7LowCmdOutput(
            publisher,
            vs.unitree_hg_msg_dds__LowCmd_,
            crc,
            vs.PUBLISH_FREQUENCY,
            vs.MAX_JOINT_SPEED,
            enable_gripper=True,
            gripper_kp=8.0,
            gripper_kd=0.4,
            gripper_speed=0.6,
            gripper_contact_hold=False,
        )

        print()
        print(
            "WARNING: REAL ROBOT rt/lowcmd TEST."
        )
        print(
            "The right arm will make a very small "
            "model +Y motion."
        )
        print(
            "Keep emergency stop ready."
        )

        answer = input(
            "Type ENABLE to take over at current pose: "
        ).strip()

        if answer.casefold() != "enable":
            print("Aborted.")
            return

        output.enable(
            upper_q,
            mode_machine,
            gripper_q=gripper_q,
            gripper_initial_mode="current",
            right_gripper_open_cap_current=True,
        )

        time.sleep(0.2)

        # Explicitly hold start first.
        output.set_arm_goal(
            q_start,
            mode_machine,
        )

        time.sleep(0.5)

        answer = input(
            "Type TESTX to send the +X 5 mm IK joint goal: "
        ).strip()

        if answer.casefold() != "testy":
            print("Test cancelled.")
            return

        # --------------------------------------------------
        # Send one fixed joint goal
        # --------------------------------------------------
        print()
        print("[TEST] Sending fixed IK joint goal.")

        output.set_arm_goal(
            solution,
            mode_machine,
        )

        t0 = time.monotonic()
        last_print = 0.0

        while (
            not stopped
            and time.monotonic() - t0
            < TEST_TIME_S
        ):
            now = time.monotonic()

            state = state_buffer.snapshot()

            if state is None:
                raise RuntimeError(
                    "LowState lost."
                )

            (
                _upper_q,
                q_now,
                dq_now,
                _gripper_q,
                _gripper_dq,
                _mode_machine,
                received_at,
            ) = state

            q_now = np.asarray(
                q_now,
                dtype=float,
            )

            if (
                now - received_at
                > vs.LOWSTATE_TIMEOUT
            ):
                raise RuntimeError(
                    "LowState stale."
                )

            if now - last_print >= 0.25:
                _, right_now = (
                    vs.get_ee_poses(
                        ik,
                        q_now,
                    )
                )

                ee_delta = (
                    right_now[:3, 3]
                    - right_start[:3, 3]
                )

                actual_joint_delta = (
                    q_now - q_start
                )

                goal_error = (
                    solution - q_now
                )

                print(
                    f"[MOVE {now-t0:4.2f}s] "
                    f"EE mm="
                    f"{np.round(ee_delta*1000, 4).tolist()} "
                    f"max_q_move="
                    f"{np.max(np.abs(actual_joint_delta)):.6f} "
                    f"max_goal_err="
                    f"{np.max(np.abs(goal_error)):.6f}",
                    flush=True,
                )

                last_print = now

            time.sleep(0.01)

        # --------------------------------------------------
        # Detailed right-arm steady-state diagnosis
        # --------------------------------------------------
        state = state_buffer.snapshot()

        if state is not None:
            (
                _upper_q,
                q_final,
                _dq_final,
                _gripper_q,
                _gripper_dq,
                _mode_machine,
                _received_at,
            ) = state

            q_final = np.asarray(
                q_final,
                dtype=float,
            )

            names = [
                "R_shoulder_pitch",
                "R_shoulder_roll",
                "R_shoulder_yaw",
                "R_elbow",
                "R_wrist_roll",
                "R_wrist_pitch",
                "R_wrist_yaw",
            ]

            kp = np.asarray(
                [
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    50.0,
                    35.0,
                    35.0,
                ],
                dtype=float,
            )

            goal_delta = (
                solution[7:14]
                - q_start[7:14]
            )

            actual_delta = (
                q_final[7:14]
                - q_start[7:14]
            )

            residual = (
                solution[7:14]
                - q_final[7:14]
            )

            pd_static = (
                kp * residual
            )

            print()
            print(
                "[RIGHT ARM STEADY-STATE]",
                flush=True,
            )

            print(
                "joint"
                "                    goal_mrad"
                "   actual_mrad"
                "   error_mrad"
                "   Kp*error_Nm",
                flush=True,
            )

            for i, name in enumerate(names):
                print(
                    f"{name:22s}"
                    f"{goal_delta[i]*1000:+11.3f}"
                    f"{actual_delta[i]*1000:+14.3f}"
                    f"{residual[i]*1000:+13.3f}"
                    f"{pd_static[i]:+14.4f}",
                    flush=True,
                )

        # --------------------------------------------------
        # Return to exact measured start joint pose
        # --------------------------------------------------
        print()
        print(
            "[RETURN] Sending original measured arm q.",
            flush=True,
        )

        output.set_arm_goal(
            q_start,
            mode_machine,
        )

        t0 = time.monotonic()
        last_print = 0.0

        while (
            not stopped
            and time.monotonic() - t0
            < RETURN_TIME_S
        ):
            now = time.monotonic()

            state = state_buffer.snapshot()

            if state is None:
                break

            (
                _upper_q,
                q_now,
                dq_now,
                _gripper_q,
                _gripper_dq,
                _mode_machine,
                received_at,
            ) = state

            q_now = np.asarray(
                q_now,
                dtype=float,
            )

            if now - last_print >= 0.5:
                _, right_now = (
                    vs.get_ee_poses(
                        ik,
                        q_now,
                    )
                )

                ee_delta = (
                    right_now[:3, 3]
                    - right_start[:3, 3]
                )

                print(
                    f"[RETURN {now-t0:4.2f}s] "
                    f"EE mm="
                    f"{np.round(ee_delta*1000, 4).tolist()} "
                    f"max_start_q_err="
                    f"{np.max(np.abs(q_now-q_start)):.6f}",
                    flush=True,
                )

                last_print = now

            time.sleep(0.01)

        print()
        print("[DONE] Test finished.")

    finally:
        if output is not None:
            output.close()

            print(
                "R1-A7 LowCmd publisher stopped.",
                flush=True,
            )


if __name__ == "__main__":
    main()
