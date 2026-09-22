#!/usr/bin/env python3

import json
import math
from pathlib import Path

import cv2
import numpy as np


THIS_FILE = Path(__file__).resolve()
HAND_EYE_DIR = THIS_FILE.parent
DATA_DIR = HAND_EYE_DIR / "data"
OUTPUT_FILE = HAND_EYE_DIR / "handeye_result.json"


METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def load_samples():
    paths = sorted(DATA_DIR.glob("sample_*.json"))

    if len(paths) < 3:
        raise RuntimeError(
            f"Need >= 3 samples, found {len(paths)}"
        )

    samples = []

    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        T_base_ee = np.asarray(
            data["robot"]["T_base_ee"],
            dtype=np.float64,
        )

        T_camera_board = np.asarray(
            data["vision"]["T_camera_board"],
            dtype=np.float64,
        )

        if T_base_ee.shape != (4, 4):
            raise RuntimeError(
                f"{path.name}: invalid T_base_ee shape"
            )

        if T_camera_board.shape != (4, 4):
            raise RuntimeError(
                f"{path.name}: invalid T_camera_board shape"
            )

        if not np.all(np.isfinite(T_base_ee)):
            raise RuntimeError(
                f"{path.name}: T_base_ee contains NaN/Inf"
            )

        if not np.all(np.isfinite(T_camera_board)):
            raise RuntimeError(
                f"{path.name}: T_camera_board contains NaN/Inf"
            )

        samples.append(
            {
                "path": path,
                "data": data,
                "T_base_ee": T_base_ee,
                "T_camera_board": T_camera_board,
            }
        )

    return samples


def rotation_angle_deg(R):
    value = (np.trace(R) - 1.0) / 2.0
    value = np.clip(value, -1.0, 1.0)
    return math.degrees(math.acos(value))


def rotation_mean(rotations):
    """
    Chordal SO(3) mean:
        average matrices -> project back to SO(3) with SVD.
    """
    M = np.zeros((3, 3), dtype=np.float64)

    for R in rotations:
        M += R

    M /= float(len(rotations))

    U, _, Vt = np.linalg.svd(M)

    R_mean = U @ Vt

    if np.linalg.det(R_mean) < 0:
        U[:, -1] *= -1.0
        R_mean = U @ Vt

    return R_mean


def make_transform(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def validate_solution(samples, T_ee_camera):
    """
    Fixed board validation:

        T_base_board_i
          = T_base_ee_i
          @ T_ee_camera
          @ T_camera_board_i

    Because the board never moved, all T_base_board_i
    should be nearly identical.
    """

    T_base_board_all = []

    for sample in samples:
        T = (
            sample["T_base_ee"]
            @ T_ee_camera
            @ sample["T_camera_board"]
        )

        T_base_board_all.append(T)

    positions = np.array(
        [T[:3, 3] for T in T_base_board_all],
        dtype=np.float64,
    )

    rotations = [
        T[:3, :3]
        for T in T_base_board_all
    ]

    mean_position = np.mean(
        positions,
        axis=0,
    )

    mean_rotation = rotation_mean(
        rotations
    )

    position_errors_m = np.linalg.norm(
        positions - mean_position,
        axis=1,
    )

    rotation_errors_deg = np.array(
        [
            rotation_angle_deg(
                mean_rotation.T @ R
            )
            for R in rotations
        ],
        dtype=np.float64,
    )

    xyz_std_m = np.std(
        positions,
        axis=0,
    )

    position_rms_m = float(
        np.sqrt(
            np.mean(
                position_errors_m ** 2
            )
        )
    )

    rotation_rms_deg = float(
        np.sqrt(
            np.mean(
                rotation_errors_deg ** 2
            )
        )
    )

    max_position_error_m = float(
        np.max(position_errors_m)
    )

    max_rotation_error_deg = float(
        np.max(rotation_errors_deg)
    )

    worst_position_index = int(
        np.argmax(position_errors_m)
    )

    worst_rotation_index = int(
        np.argmax(rotation_errors_deg)
    )

    return {
        "T_base_board_all": T_base_board_all,
        "mean_position_m": mean_position,
        "mean_rotation": mean_rotation,
        "xyz_std_m": xyz_std_m,
        "position_errors_m": position_errors_m,
        "rotation_errors_deg": rotation_errors_deg,
        "position_rms_m": position_rms_m,
        "rotation_rms_deg": rotation_rms_deg,
        "max_position_error_m": max_position_error_m,
        "max_rotation_error_deg": max_rotation_error_deg,
        "worst_position_index": worst_position_index,
        "worst_rotation_index": worst_rotation_index,
    }


def print_matrix(name, T):
    print()
    print(name)
    print(
        np.array2string(
            T,
            precision=8,
            suppress_small=True,
        )
    )


def main():
    print()
    print("==============================================")
    print(" R1-A7 Eye-in-Hand Solver")
    print("==============================================")
    print()

    samples = load_samples()

    print(f"Samples loaded: {len(samples)}")

    print()
    print("Sample quality:")
    print(
        "ID   corners   RMSE(px)   LowStateAge(ms)"
    )

    for sample in samples:
        d = sample["data"]

        sample_id = d["sample_id"]

        corners = d["vision"].get(
            "charuco_corner_count",
            -1,
        )

        rmse = d["vision"].get(
            "reprojection_rmse_px",
            float("nan"),
        )

        age = d["robot"].get(
            "lowstate_age_ms",
            float("nan"),
        )

        print(
            f"{sample_id:02d}   "
            f"{corners:7d}   "
            f"{rmse:8.4f}   "
            f"{age:15.3f}"
        )

    # --------------------------------------------------------
    # OpenCV input convention
    #
    # R_gripper2base:
    #       ^base R_ee
    #
    # t_gripper2base:
    #       ^base t_ee
    #
    # R_target2cam:
    #       ^camera R_board
    #
    # t_target2cam:
    #       ^camera t_board
    #
    # Output:
    #       ^ee T_camera
    # --------------------------------------------------------

    R_gripper2base = []
    t_gripper2base = []

    R_target2cam = []
    t_target2cam = []

    for sample in samples:
        T_b_e = sample["T_base_ee"]
        T_c_t = sample["T_camera_board"]

        R_gripper2base.append(
            T_b_e[:3, :3].copy()
        )

        t_gripper2base.append(
            T_b_e[:3, 3].reshape(3, 1).copy()
        )

        R_target2cam.append(
            T_c_t[:3, :3].copy()
        )

        t_target2cam.append(
            T_c_t[:3, 3].reshape(3, 1).copy()
        )

    results = {}

    print()
    print("==============================================")
    print(" Solving")
    print("==============================================")

    for name, method in METHODS.items():
        print()
        print(f"[{name}]")

        try:
            R_cam2gripper, t_cam2gripper = (
                cv2.calibrateHandEye(
                    R_gripper2base,
                    t_gripper2base,
                    R_target2cam,
                    t_target2cam,
                    method=method,
                )
            )

            R_cam2gripper = np.asarray(
                R_cam2gripper,
                dtype=np.float64,
            ).reshape(3, 3)

            t_cam2gripper = np.asarray(
                t_cam2gripper,
                dtype=np.float64,
            ).reshape(3)

            if not np.all(
                np.isfinite(
                    R_cam2gripper
                )
            ):
                raise RuntimeError(
                    "rotation contains NaN/Inf"
                )

            if not np.all(
                np.isfinite(
                    t_cam2gripper
                )
            ):
                raise RuntimeError(
                    "translation contains NaN/Inf"
                )

            T_ee_camera = make_transform(
                R_cam2gripper,
                t_cam2gripper,
            )

            validation = validate_solution(
                samples,
                T_ee_camera,
            )

            print_matrix(
                "T_ee_camera =",
                T_ee_camera,
            )

            t_mm = (
                1000.0
                * T_ee_camera[:3, 3]
            )

            print()
            print(
                "camera origin in R_ee [mm]: "
                f"X={t_mm[0]:+.3f}, "
                f"Y={t_mm[1]:+.3f}, "
                f"Z={t_mm[2]:+.3f}"
            )

            print(
                "translation norm [mm]: "
                f"{np.linalg.norm(t_mm):.3f}"
            )

            print()
            print(
                "Fixed-board validation:"
            )

            print(
                "  XYZ std [mm]: "
                f"{1000.0 * validation['xyz_std_m'][0]:.3f}, "
                f"{1000.0 * validation['xyz_std_m'][1]:.3f}, "
                f"{1000.0 * validation['xyz_std_m'][2]:.3f}"
            )

            print(
                "  Position RMS [mm]: "
                f"{1000.0 * validation['position_rms_m']:.3f}"
            )

            print(
                "  Position MAX [mm]: "
                f"{1000.0 * validation['max_position_error_m']:.3f}"
            )

            print(
                "  Rotation RMS [deg]: "
                f"{validation['rotation_rms_deg']:.4f}"
            )

            print(
                "  Rotation MAX [deg]: "
                f"{validation['max_rotation_error_deg']:.4f}"
            )

            wp = (
                validation[
                    "worst_position_index"
                ]
            )

            wr = (
                validation[
                    "worst_rotation_index"
                ]
            )

            print(
                "  Worst position sample: "
                f"{samples[wp]['data']['sample_id']:04d}"
            )

            print(
                "  Worst rotation sample: "
                f"{samples[wr]['data']['sample_id']:04d}"
            )

            results[name] = {
                "success": True,

                "T_ee_camera":
                    T_ee_camera.tolist(),

                "translation_mm":
                    t_mm.tolist(),

                "translation_norm_mm":
                    float(
                        np.linalg.norm(t_mm)
                    ),

                "validation": {
                    "mean_board_position_base_m":
                        validation[
                            "mean_position_m"
                        ].tolist(),

                    "mean_board_rotation_base":
                        validation[
                            "mean_rotation"
                        ].tolist(),

                    "xyz_std_mm":
                        (
                            1000.0
                            * validation[
                                "xyz_std_m"
                            ]
                        ).tolist(),

                    "position_rms_mm":
                        1000.0
                        * validation[
                            "position_rms_m"
                        ],

                    "position_max_mm":
                        1000.0
                        * validation[
                            "max_position_error_m"
                        ],

                    "rotation_rms_deg":
                        validation[
                            "rotation_rms_deg"
                        ],

                    "rotation_max_deg":
                        validation[
                            "max_rotation_error_deg"
                        ],

                    "worst_position_sample":
                        samples[
                            validation[
                                "worst_position_index"
                            ]
                        ]["data"]["sample_id"],

                    "worst_rotation_sample":
                        samples[
                            validation[
                                "worst_rotation_index"
                            ]
                        ]["data"]["sample_id"],
                },
            }

        except Exception as exc:
            print(
                f"FAILED: {repr(exc)}"
            )

            results[name] = {
                "success": False,
                "error": repr(exc),
            }

    successful = {
        name: result
        for name, result in results.items()
        if result.get("success")
    }

    if not successful:
        raise RuntimeError(
            "All hand-eye methods failed."
        )

    # --------------------------------------------------------
    # Rank ONLY by fixed-board consistency.
    #
    # This is not proof of absolute calibration accuracy;
    # it is an internal consistency metric.
    # --------------------------------------------------------

    ranking = sorted(
        successful.keys(),
        key=lambda name: (
            successful[name]["validation"][
                "position_rms_mm"
            ],
            successful[name]["validation"][
                "rotation_rms_deg"
            ],
        ),
    )

    print()
    print("==============================================")
    print(" Consistency ranking")
    print("==============================================")
    print()

    for index, name in enumerate(
        ranking,
        start=1,
    ):
        val = (
            successful[name]["validation"]
        )

        print(
            f"{index}. {name:11s} "
            f"pos_RMS={val['position_rms_mm']:.3f} mm   "
            f"rot_RMS={val['rotation_rms_deg']:.4f} deg"
        )

    best_name = ranking[0]

    output = {
        "opencv_version":
            cv2.__version__,

        "sample_count":
            len(samples),

        "frame_definition": {
            "robot_input":
                "T_base_ee",

            "vision_input":
                "T_camera_board",

            "solution":
                "T_ee_camera",

            "validation_equation":
                (
                    "T_base_board = "
                    "T_base_ee @ "
                    "T_ee_camera @ "
                    "T_camera_board"
                ),
        },

        "methods":
            results,

        "consistency_ranking":
            ranking,

        "lowest_internal_scatter_method":
            best_name,

        "T_ee_camera_candidate":
            successful[
                best_name
            ]["T_ee_camera"],
    }

    with OUTPUT_FILE.open(
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
        "Saved:",
        OUTPUT_FILE,
    )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "Do not use the result for robot motion yet."
    )
    print(
        "First inspect multi-method agreement and "
        "fixed-board validation."
    )


if __name__ == "__main__":
    main()
