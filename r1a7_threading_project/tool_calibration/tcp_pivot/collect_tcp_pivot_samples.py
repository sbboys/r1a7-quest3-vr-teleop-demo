#!/usr/bin/env python3

import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Paths
# ============================================================

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]

DATA_DIR = THIS_FILE.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ROBOT_MODULE_PATH = (
    REPO_ROOT
    / "r1a7_threading_project"
    / "vision"
    / "r1a7_head_visual_servo_lowcmd.py"
)


# ============================================================
# Runtime configuration
# ============================================================

STATE_TOPIC = "rt/lowstate"

DEFAULT_DOMAIN_ID = 0
DEFAULT_INTERFACE = "enp6s0"

MAX_LOWSTATE_AGE_MS = 50.0

# Only allow saving when arm is approximately stationary.
# This is deliberately conservative but not excessively strict.
MAX_ARM_DQ_RAD_S = 0.05

WINDOW_NAME = "R1-A7 TCP Pivot Collector"


# ============================================================
# Helpers
# ============================================================

def load_robot_module():
    if not ROBOT_MODULE_PATH.exists():
        raise FileNotFoundError(
            f"Robot module not found: {ROBOT_MODULE_PATH}"
        )

    sys.path.insert(0, str(REPO_ROOT))

    spec = importlib.util.spec_from_file_location(
        "r1a7_existing_visual_servo",
        str(ROBOT_MODULE_PATH),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import: {ROBOT_MODULE_PATH}"
        )

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def next_sample_id():
    paths = sorted(
        DATA_DIR.glob("sample_*.json")
    )

    max_id = 0

    for path in paths:
        try:
            idx = int(
                path.stem.split("_")[-1]
            )
            max_id = max(max_id, idx)
        except Exception:
            pass

    return max_id + 1


def matrix_valid(T):
    if T is None:
        return False

    T = np.asarray(
        T,
        dtype=np.float64,
    )

    return (
        T.shape == (4, 4)
        and np.all(np.isfinite(T))
    )


def save_sample(
    sample_id,
    arm_q,
    arm_dq,
    lowstate_age_ms,
    T_base_ee,
):
    path = (
        DATA_DIR
        / f"sample_{sample_id:04d}.json"
    )

    q = np.asarray(
        arm_q,
        dtype=np.float64,
    ).reshape(14)

    dq = np.asarray(
        arm_dq,
        dtype=np.float64,
    ).reshape(14)

    T = np.asarray(
        T_base_ee,
        dtype=np.float64,
    ).reshape(4, 4)

    payload = {
        "sample_id": int(sample_id),

        "host_time_ns": int(
            time.time_ns()
        ),

        "calibration_type":
            "tcp_pivot",

        "robot":
            "Unitree R1-A7",

        "ee_frame":
            "R_ee",

        "ee_definition": {
            "parent_joint":
                "right_wrist_yaw_joint",

            "local_translation_m": [
                0.05,
                0.0,
                0.0
            ],
        },

        "measurement": {
            "q_arm_rad":
                q.tolist(),

            "dq_arm_rad_s":
                dq.tolist(),

            "max_abs_dq_rad_s":
                float(
                    np.max(
                        np.abs(dq)
                    )
                ),

            "lowstate_age_ms":
                float(
                    lowstate_age_ms
                ),

            "T_base_ee":
                T.tolist(),
        },

        "geometry": {
            "constraint":
                (
                    "R_base_ee @ P_tip_ee "
                    "+ t_base_ee "
                    "= P_pivot_base"
                ),

            "note":
                (
                    "The same physical tool point "
                    "must coincide with the same fixed "
                    "pivot point for every sample."
                ),
        },
    }

    tmp = path.with_suffix(
        ".json.tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
        )

    tmp.replace(path)

    return path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "R1-A7 TCP/P_tip pivot calibration collector"
        )
    )

    parser.add_argument(
        "--interface",
        default=DEFAULT_INTERFACE,
    )

    parser.add_argument(
        "--domain-id",
        type=int,
        default=DEFAULT_DOMAIN_ID,
    )

    args = parser.parse_args()

    print()
    print(
        "=============================================="
    )
    print(
        " R1-A7 TCP / P_tip Pivot Collector"
    )
    print(
        " READ ONLY: NO LowCmd publisher"
    )
    print(
        "=============================================="
    )
    print()

    # --------------------------------------------------------
    # Load existing verified robot state / FK implementation
    # --------------------------------------------------------

    vs = load_robot_module()

    # --------------------------------------------------------
    # DDS
    # --------------------------------------------------------

    print(
        "[ROBOT] Initializing DDS..."
    )

    vs.ChannelFactoryInitialize(
        args.domain_id,
        args.interface,
    )

    crc = vs.CRC()

    state_buffer = (
        vs.base.StateBuffer(
            crc
        )
    )

    subscriber = (
        vs.ChannelSubscriber(
            STATE_TOPIC,
            vs.LowState_,
        )
    )

    subscriber.Init(
        state_buffer.callback,
        10,
    )

    print(
        "[ROBOT] Waiting for LowState..."
    )

    deadline = (
        time.monotonic()
        + 10.0
    )

    state = state_buffer.snapshot()

    while (
        state is None
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)
        state = state_buffer.snapshot()

    if state is None:
        raise RuntimeError(
            "No valid rt/lowstate received."
        )

    print(
        "[ROBOT] LowState OK."
    )

    # --------------------------------------------------------
    # FK model
    # --------------------------------------------------------

    print(
        "[ROBOT] Loading R1A7_ArmIK model "
        "for FK only..."
    )

    ik = vs.R1A7_ArmIK()

    print(
        "[ROBOT] FK model ready."
    )

    print()
    print(
        "[IMPORTANT]"
    )
    print(
        "Keep the SAME tool point on the SAME "
        "fixed physical pivot for every sample."
    )
    print()
    print(
        "[CONTROL]"
    )
    print(
        "S = save current TCP pivot sample"
    )
    print(
        "Q / ESC = quit"
    )
    print()

    sample_id = next_sample_id()

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        900,
        560,
    )

    last_save_time = 0.0

    try:
        while True:

            canvas = np.zeros(
                (560, 900, 3),
                dtype=np.uint8,
            )

            state = (
                state_buffer.snapshot()
            )

            arm_q = None
            arm_dq = None
            T_base_ee = None

            lowstate_age_ms = float(
                "inf"
            )

            max_abs_dq = float(
                "inf"
            )

            if state is not None:
                (
                    upper_q,
                    arm_q,
                    arm_dq,
                    gripper_q,
                    gripper_dq,
                    mode_machine,
                    received_at,
                ) = state

                lowstate_age_ms = (
                    1000.0
                    * (
                        time.monotonic()
                        - received_at
                    )
                )

                try:
                    arm_q_np = np.asarray(
                        arm_q,
                        dtype=np.float64,
                    ).reshape(14)

                    arm_dq_np = np.asarray(
                        arm_dq,
                        dtype=np.float64,
                    ).reshape(14)

                    max_abs_dq = float(
                        np.max(
                            np.abs(
                                arm_dq_np
                            )
                        )
                    )

                    _, T_base_ee = (
                        vs.get_ee_poses(
                            ik,
                            arm_q_np,
                        )
                    )

                except Exception:
                    T_base_ee = None

            valid = (
                arm_q is not None
                and arm_dq is not None
                and matrix_valid(
                    T_base_ee
                )
                and lowstate_age_ms
                <= MAX_LOWSTATE_AGE_MS
                and max_abs_dq
                <= MAX_ARM_DQ_RAD_S
            )

            lines = [
                "R1-A7 TCP / P_tip Pivot Calibration",
                "",
                "Same tool point -> same fixed pivot point",
                "",
                f"LowState age: {lowstate_age_ms:.3f} ms",
                (
                    "max |dq|: "
                    f"{max_abs_dq:.5f} rad/s"
                ),
                "",
            ]

            if T_base_ee is not None:
                xyz_mm = (
                    1000.0
                    * np.asarray(
                        T_base_ee
                    )[:3, 3]
                )

                lines.extend(
                    [
                        (
                            "R_ee Base XYZ [mm]: "
                            f"{xyz_mm[0]:+.2f}, "
                            f"{xyz_mm[1]:+.2f}, "
                            f"{xyz_mm[2]:+.2f}"
                        ),
                        "",
                    ]
                )

            lines.extend(
                [
                    (
                        "STATUS: READY TO SAVE"
                        if valid
                        else
                        "STATUS: NOT READY"
                    ),
                    "",
                    (
                        f"Next sample: "
                        f"sample_{sample_id:04d}"
                    ),
                    "",
                    "S = save",
                    "Q / ESC = quit",
                ]
            )

            y = 38

            for line in lines:
                cv2.putText(
                    canvas,
                    line,
                    (25, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70,
                    (
                        255,
                        255,
                        255,
                    ),
                    2,
                    cv2.LINE_AA,
                )

                y += 34

            cv2.imshow(
                WINDOW_NAME,
                canvas,
            )

            key = (
                cv2.waitKey(20)
                & 0xFF
            )

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):
                break

            if key in (
                ord("s"),
                ord("S"),
            ):
                now = time.monotonic()

                # Avoid accidental double-save.
                if (
                    now - last_save_time
                    < 0.5
                ):
                    continue

                state_save = (
                    state_buffer.snapshot()
                )

                if state_save is None:
                    print(
                        "[REJECT] No LowState."
                    )
                    continue

                (
                    upper_q,
                    arm_q_save,
                    arm_dq_save,
                    gripper_q,
                    gripper_dq,
                    mode_machine,
                    received_at_save,
                ) = state_save

                q_save = np.asarray(
                    arm_q_save,
                    dtype=np.float64,
                ).reshape(14)

                dq_save = np.asarray(
                    arm_dq_save,
                    dtype=np.float64,
                ).reshape(14)

                age_save_ms = (
                    1000.0
                    * (
                        time.monotonic()
                        - received_at_save
                    )
                )

                max_dq_save = float(
                    np.max(
                        np.abs(
                            dq_save
                        )
                    )
                )

                try:
                    _, T_save = (
                        vs.get_ee_poses(
                            ik,
                            q_save,
                        )
                    )
                except Exception as exc:
                    print(
                        "[REJECT] FK failed:",
                        repr(exc),
                    )
                    continue

                save_valid = (
                    matrix_valid(
                        T_save
                    )
                    and np.all(
                        np.isfinite(
                            q_save
                        )
                    )
                    and np.all(
                        np.isfinite(
                            dq_save
                        )
                    )
                    and age_save_ms
                    <= MAX_LOWSTATE_AGE_MS
                    and max_dq_save
                    <= MAX_ARM_DQ_RAD_S
                )

                if not save_valid:
                    print(
                        "[REJECT] Robot is not "
                        "sufficiently stationary."
                    )

                    print(
                        "  LowState age:",
                        f"{age_save_ms:.3f} ms"
                    )

                    print(
                        "  max |dq|:",
                        f"{max_dq_save:.5f} rad/s"
                    )

                    continue

                path = save_sample(
                    sample_id,
                    q_save,
                    dq_save,
                    age_save_ms,
                    T_save,
                )

                xyz_mm = (
                    1000.0
                    * np.asarray(
                        T_save
                    )[:3, 3]
                )

                print()
                print(
                    "======================================"
                )

                print(
                    f"[SAVED] sample_{sample_id:04d}"
                )

                print(
                    "======================================"
                )

                print(
                    "lowstate_age_ms =",
                    f"{age_save_ms:.3f}",
                )

                print(
                    "max_abs_dq_rad_s =",
                    f"{max_dq_save:.6f}",
                )

                print(
                    "R_ee Base XYZ [mm] =",
                    np.round(
                        xyz_mm,
                        3,
                    ).tolist(),
                )

                print()
                print(
                    "T_base_ee ="
                )

                print(
                    np.array2string(
                        np.asarray(
                            T_save
                        ),
                        precision=8,
                        suppress_small=True,
                    )
                )

                print()
                print(
                    "JSON:",
                    path,
                )
                print()

                sample_id += 1
                last_save_time = now

    finally:
        cv2.destroyAllWindows()

        print()
        print(
            "[DONE] TCP pivot collector stopped."
        )

        print(
            "[SAFETY] No LowCmd publisher "
            "was created."
        )


if __name__ == "__main__":
    main()
