#!/usr/bin/env python3

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent
REPO_ROOT = THIS_FILE.parents[3]

TCP_FILE = ROOT / "r1a7_right_tool_tcp.json"
OUTPUT_DIR = ROOT / "validation_live"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ROBOT_MODULE_PATH = (
    REPO_ROOT
    / "r1a7_threading_project"
    / "vision"
    / "r1a7_head_visual_servo_lowcmd.py"
)

STATE_TOPIC = "rt/lowstate"

MAX_LOWSTATE_AGE_MS = 50.0
MAX_ARM_DQ_RAD_S = 0.05

WINDOW = "R1-A7 TCP Live Validation"


def load_robot_module():
    sys.path.insert(0, str(REPO_ROOT))

    spec = importlib.util.spec_from_file_location(
        "r1a7_robot_module",
        str(ROBOT_MODULE_PATH),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot load robot module: {ROBOT_MODULE_PATH}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def load_tcp():
    with TCP_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    p_tip_ee = np.asarray(
        data["P_tip_ee_m"],
        dtype=np.float64,
    ).reshape(3)

    p_calibration = np.asarray(
        data["calibration_pivot_base_m"],
        dtype=np.float64,
    ).reshape(3)

    return data, p_tip_ee, p_calibration


def compute_summary(samples):
    if len(samples) < 2:
        return None

    points = np.asarray(
        [s["P_tip_base_m"] for s in samples],
        dtype=np.float64,
    )

    mean_point = np.mean(
        points,
        axis=0,
    )

    errors = np.linalg.norm(
        points - mean_point,
        axis=1,
    )

    xyz_std = np.std(
        points,
        axis=0,
    )

    return {
        "mean_tip_base_mm":
            (1000.0 * mean_point).tolist(),

        "xyz_std_mm":
            (1000.0 * xyz_std).tolist(),

        "position_rms_mm":
            float(
                1000.0
                * np.sqrt(
                    np.mean(errors ** 2)
                )
            ),

        "position_max_mm":
            float(
                1000.0
                * np.max(errors)
            ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--interface",
        default="enp6s0",
    )

    parser.add_argument(
        "--domain-id",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    print()
    print("============================================")
    print(" R1-A7 TCP / P_tip Live Validation")
    print(" READ ONLY: NO LowCmd publisher")
    print("============================================")
    print()

    tcp_data, p_tip_ee, p_calibration = load_tcp()

    print(
        "P_tip_ee [mm]:",
        np.round(
            1000.0 * p_tip_ee,
            3,
        ).tolist(),
    )

    print(
        "Calibration pivot Base [mm]:",
        np.round(
            1000.0 * p_calibration,
            3,
        ).tolist(),
    )

    vs = load_robot_module()

    print()
    print("[ROBOT] Initializing DDS...")

    vs.ChannelFactoryInitialize(
        args.domain_id,
        args.interface,
    )

    crc = vs.CRC()

    state_buffer = vs.base.StateBuffer(
        crc
    )

    subscriber = vs.ChannelSubscriber(
        STATE_TOPIC,
        vs.LowState_,
    )

    subscriber.Init(
        state_buffer.callback,
        10,
    )

    print("[ROBOT] Waiting for LowState...")

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
            "No valid rt/lowstate received."
        )

    print("[ROBOT] LowState OK.")

    print("[ROBOT] Loading FK model...")

    ik = vs.R1A7_ArmIK()

    print("[ROBOT] FK model ready.")

    print()
    print("[CONTROL]")
    print("R = set current P_tip Base as local reference")
    print("S = save validation pose")
    print("Q / ESC = quit")
    print()

    reference = None
    saved = []

    cv2.namedWindow(
        WINDOW,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        WINDOW,
        950,
        650,
    )

    try:
        while True:

            canvas = np.zeros(
                (650, 950, 3),
                dtype=np.uint8,
            )

            state = state_buffer.snapshot()

            T_base_ee = None
            P_tip_base = None

            lowstate_age_ms = float("inf")
            max_dq = float("inf")

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

                q = np.asarray(
                    arm_q,
                    dtype=np.float64,
                ).reshape(14)

                dq = np.asarray(
                    arm_dq,
                    dtype=np.float64,
                ).reshape(14)

                max_dq = float(
                    np.max(
                        np.abs(dq)
                    )
                )

                try:
                    _, T_base_ee = (
                        vs.get_ee_poses(
                            ik,
                            q,
                        )
                    )

                    R = T_base_ee[:3, :3]
                    t = T_base_ee[:3, 3]

                    P_tip_base = (
                        R @ p_tip_ee
                        + t
                    )

                except Exception:
                    T_base_ee = None
                    P_tip_base = None

            valid = (
                P_tip_base is not None
                and lowstate_age_ms
                <= MAX_LOWSTATE_AGE_MS
                and max_dq
                <= MAX_ARM_DQ_RAD_S
            )

            lines = [
                "R1-A7 TCP / P_tip independent validation",
                "",
                (
                    "P_tip_ee [mm]: "
                    f"{1000*p_tip_ee[0]:+.2f}, "
                    f"{1000*p_tip_ee[1]:+.2f}, "
                    f"{1000*p_tip_ee[2]:+.2f}"
                ),
                "",
                (
                    "LowState age: "
                    f"{lowstate_age_ms:.3f} ms"
                ),
                (
                    "max |dq|: "
                    f"{max_dq:.5f} rad/s"
                ),
                "",
            ]

            if P_tip_base is not None:

                p_mm = (
                    1000.0
                    * P_tip_base
                )

                lines.append(
                    "P_tip Base XYZ [mm]: "
                    f"{p_mm[0]:+.3f}, "
                    f"{p_mm[1]:+.3f}, "
                    f"{p_mm[2]:+.3f}"
                )

                calibration_error_mm = (
                    1000.0
                    * np.linalg.norm(
                        P_tip_base
                        - p_calibration
                    )
                )

                lines.append(
                    "Delta calibration pivot: "
                    f"{calibration_error_mm:.3f} mm"
                )

                if reference is not None:

                    ref_error_mm = (
                        1000.0
                        * np.linalg.norm(
                            P_tip_base
                            - reference
                        )
                    )

                    lines.append(
                        "Delta local REF: "
                        f"{ref_error_mm:.3f} mm"
                    )

                else:
                    lines.append(
                        "Local REF: NOT SET (press R)"
                    )

            lines.extend(
                [
                    "",
                    (
                        "STATUS: READY"
                        if valid
                        else "STATUS: NOT READY"
                    ),
                    "",
                    f"Saved poses: {len(saved)}",
                    "",
                    "R = set local reference",
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
                    0.68,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                y += 35

            cv2.imshow(
                WINDOW,
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
                ord("r"),
                ord("R"),
            ):

                if not valid:
                    print(
                        "[REF REJECT] Robot not stationary."
                    )
                    continue

                reference = (
                    P_tip_base.copy()
                )

                print()
                print("[REFERENCE SET]")

                print(
                    "P_tip Base XYZ [mm]:",
                    np.round(
                        1000.0 * reference,
                        3,
                    ).tolist(),
                )

            if key in (
                ord("s"),
                ord("S"),
            ):

                if not valid:
                    print(
                        "[SAVE REJECT] Robot not stationary."
                    )
                    continue

                p = P_tip_base.copy()

                calibration_error_mm = float(
                    1000.0
                    * np.linalg.norm(
                        p - p_calibration
                    )
                )

                ref_error_mm = None

                if reference is not None:
                    ref_error_mm = float(
                        1000.0
                        * np.linalg.norm(
                            p - reference
                        )
                    )

                sample = {
                    "index":
                        len(saved) + 1,

                    "host_time_ns":
                        time.time_ns(),

                    "lowstate_age_ms":
                        float(
                            lowstate_age_ms
                        ),

                    "max_abs_dq_rad_s":
                        float(max_dq),

                    "T_base_ee":
                        np.asarray(
                            T_base_ee
                        ).tolist(),

                    "P_tip_base_m":
                        p.tolist(),

                    "delta_calibration_pivot_mm":
                        calibration_error_mm,

                    "delta_local_reference_mm":
                        ref_error_mm,
                }

                saved.append(sample)

                print()
                print(
                    "======================================"
                )

                print(
                    "[SAVED VALIDATION POSE] "
                    f"{len(saved):02d}"
                )

                print(
                    "P_tip Base XYZ [mm]: "
                    f"{1000*p[0]:+.3f}, "
                    f"{1000*p[1]:+.3f}, "
                    f"{1000*p[2]:+.3f}"
                )

                print(
                    "Delta calibration pivot: "
                    f"{calibration_error_mm:.3f} mm"
                )

                if ref_error_mm is not None:
                    print(
                        "Delta local REF: "
                        f"{ref_error_mm:.3f} mm"
                    )

                print(
                    "lowstate_age_ms:",
                    f"{lowstate_age_ms:.3f}",
                )

                print(
                    "max_abs_dq_rad_s:",
                    f"{max_dq:.6f}",
                )

    finally:
        cv2.destroyAllWindows()

    print()
    print("============================================")
    print(" Validation Summary")
    print("============================================")

    summary = compute_summary(
        saved
    )

    if summary is None:

        print(
            "Need at least 2 saved poses."
        )

    else:

        print(
            "Saved poses:",
            len(saved),
        )

        print(
            "Mean P_tip Base XYZ [mm]: "
            f"{summary['mean_tip_base_mm'][0]:+.3f}, "
            f"{summary['mean_tip_base_mm'][1]:+.3f}, "
            f"{summary['mean_tip_base_mm'][2]:+.3f}"
        )

        print(
            "XYZ std [mm]: "
            f"{summary['xyz_std_mm'][0]:.3f}, "
            f"{summary['xyz_std_mm'][1]:.3f}, "
            f"{summary['xyz_std_mm'][2]:.3f}"
        )

        print(
            "Position RMS [mm]: "
            f"{summary['position_rms_mm']:.3f}"
        )

        print(
            "Position MAX [mm]: "
            f"{summary['position_max_mm']:.3f}"
        )

    timestamp = (
        time.strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    output_path = (
        OUTPUT_DIR
        / f"tcp_validation_{timestamp}.json"
    )

    output = {
        "tcp_file":
            str(TCP_FILE),

        "P_tip_ee_m":
            p_tip_ee.tolist(),

        "calibration_pivot_base_m":
            p_calibration.tolist(),

        "reference_tip_base_m":
            (
                reference.tolist()
                if reference is not None
                else None
            ),

        "samples":
            saved,

        "summary":
            summary,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "Saved validation:",
        output_path,
    )

    print(
        "[SAFETY] No LowCmd publisher was created."
    )


if __name__ == "__main__":
    main()
