#!/usr/bin/env python3
# Calibrate the 3D tweezer-tip position in the ArUco ID20 marker frame
# using the LEFT wrist Orbbec CPCBC530007Z images.
# This program only captures images and calibration samples.

import sys
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "vision_dual_camera" / "handeye"))
sys.path.insert(
    0,
    str(ROOT / "vision_dual_camera" / "task6_threading"
        / "perception" / "tweezer_tip"),
)

import collect_handeye_samples as hc
import test_external_aruco_local_tip_refine_v2 as v2
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat

SERIAL = "CPCBC530007Z"
WIDTH, HEIGHT, FPS = 1280, 800, 30
MIN_SAMPLES = 8

OUTPUT = (
    ROOT / "calibration"
    / "left_wrist_007z_marker_tip_calibration_candidate.json"
)

# Initial guesses only. The final value will be fitted from left-wrist samples.
SEEDS = [
    [-0.0855, 0.0070, -0.0245],
    [ 0.0855, 0.0070, -0.0245],
    [-0.0855,-0.0070, -0.0245],
    [ 0.0855,-0.0070, -0.0245],
    [-0.0855, 0.0070,  0.0245],
    [ 0.0855, 0.0070,  0.0245],
    [-0.0855,-0.0070,  0.0245],
    [ 0.0855,-0.0070,  0.0245],
    [0.0, 0.0, 0.0],
]

K = np.asarray(hc.CAMERA_MATRIX, dtype=np.float64)
DIST = np.asarray(hc.DIST_COEFFS, dtype=np.float64)

samples = []
state = {
    "frame": None,
    "poses": [],
    "clicked": False,
    "armed": True,
}


def project_point(p_marker, pose):
    uv, _ = cv2.projectPoints(
        np.asarray(p_marker, dtype=np.float64).reshape(1, 1, 3),
        np.asarray(pose["rvec"], dtype=np.float64).reshape(3, 1),
        np.asarray(pose["tvec"], dtype=np.float64).reshape(3, 1),
        K,
        DIST,
    )
    return uv.reshape(2)


def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if not state["armed"] or state["clicked"]:
        return
    if not state["poses"]:
        print("[SKIP] ID20 pose unavailable. Wait for a visible marker.")
        return

    # The clicked pixel and pose candidates belong to the same displayed frame.
    pose_list = state["poses"]
    pose = min(
        pose_list,
        key=lambda p: float(p.get("marker_reproj_px", 1e9)),
    )

    sample = {
        "tip_uv": [float(x), float(y)],
        "rvec": np.asarray(pose["rvec"]).reshape(-1).tolist(),
        "tvec": np.asarray(pose["tvec"]).reshape(-1).tolist(),
        "branch_id": int(pose.get("branch_id", -1)),
        "marker_reproj_px": float(
            pose.get("marker_reproj_px", -1.0)
        ),
        "timestamp": time.time(),
    }
    samples.append(sample)
    state["clicked"] = True
    state["armed"] = False

    print(
        f"[SAMPLE {len(samples)}] tip_uv={sample['tip_uv']} "
        f"branch={sample['branch_id']} "
        f"marker_reproj={sample['marker_reproj_px']:.3f}px"
    )
    print("[NEXT] Change the viewing angle/pose, then press N for next sample.")
    print("[UNDO] Press U to remove the latest sample.")
    print("[FIT] Press Q to fit after collecting enough samples.")


def residuals(p_marker, data):
    residual = []
    for s in data:
        pose = {
            "rvec": np.asarray(s["rvec"], dtype=np.float64).reshape(3, 1),
            "tvec": np.asarray(s["tvec"], dtype=np.float64).reshape(3, 1),
        }
        predicted = project_point(p_marker, pose)
        observed = np.asarray(s["tip_uv"], dtype=np.float64)
        residual.extend((predicted - observed).tolist())
    return np.asarray(residual, dtype=np.float64)


def fit_offset(data):
    best = None

    # Multiple starts reduce dependence on the External-camera seed.
    for seed in SEEDS:
        result = least_squares(
            residuals,
            np.asarray(seed, dtype=np.float64),
            args=(data,),
            bounds=(
                np.array([-0.25, -0.25, -0.25]),
                np.array([ 0.25,  0.25,  0.25]),
            ),
            loss="soft_l1",
            f_scale=3.0,
            max_nfev=5000,
        )
        r = residuals(result.x, data)
        rms = float(np.sqrt(np.mean(r ** 2)))
        if best is None or rms < best["rms"]:
            best = {
                "p": result.x,
                "rms": rms,
                "median": float(np.median(
                    np.linalg.norm(r.reshape(-1, 2), axis=1)
                )),
                "max": float(np.max(
                    np.linalg.norm(r.reshape(-1, 2), axis=1)
                )),
                "result": result,
            }

    return best


def main():
    context = None
    pipeline = None
    started = False

    print("Camera serial:", SERIAL)
    print("Image:", WIDTH, "x", HEIGHT, "@", FPS)
    print("Intrinsics source: existing wrist calibration")
    print("ArUco ID:", getattr(v2, "ARUCO_ID", "unknown"))
    print()
    print("Instructions:")
    print("  1. Keep the physical tweezer tip and ID20 visible.")
    print("  2. Left-click the ACTUAL physical tweezer tip.")
    print("  3. Change wrist/tool viewing angle before each new sample.")
    print("  N = arm next sample; U = undo; Q = fit and save candidate.")
    print("  Collect at least 8 samples; 12-20 diverse poses are preferable.")
    print()

    try:
        context, device = hc.find_orbbec_device(SERIAL)
        actual = hc.normalize_serial(
            device.get_device_info().get_serial_number()
        ).strip()
        if actual != SERIAL:
            raise RuntimeError(
                f"Serial mismatch: expected {SERIAL}, got {actual}"
            )

        pipeline = Pipeline(device)
        config = Config()
        profiles = pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        )
        profile = profiles.get_video_stream_profile(
            WIDTH, HEIGHT, OBFormat.RGB, FPS
        )
        config.enable_stream(profile)
        pipeline.start(config)
        started = True

        window = "LEFT 007Z - CLICK PHYSICAL TWEEZER TIP"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, on_mouse)

        while True:
            frames = pipeline.wait_for_frames(500)
            if frames is None:
                continue

            color = frames.get_color_frame()
            if color is None:
                continue

            frame = hc.color_frame_to_bgr(color)
            if frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = v2.detect_marker(gray)

            poses = []
            if corners is not None:
                try:
                    poses = v2.solve_pose_candidates(corners, K, DIST) or []
                except Exception as exc:
                    print("[POSE ERROR]", repr(exc))

            state["frame"] = frame.copy()
            state["poses"] = [
                {
                    **p,
                    "rvec": np.asarray(p["rvec"]).copy(),
                    "tvec": np.asarray(p["tvec"]).copy(),
                }
                for p in poses
            ]

            vis = frame.copy()

            if corners is not None:
                # detect_marker() returns one refined 4x2 corner array.
                # Draw it directly instead of passing it to drawDetectedMarkers(),
                # which expects OpenCV's list-of-marker-corners representation.
                pts = np.asarray(
                    corners, dtype=np.float32
                ).reshape(4, 2)

                polygon = np.rint(pts).astype(
                    np.int32
                ).reshape(-1, 1, 2)

                cv2.polylines(
                    vis, [polygon], True, (0, 255, 0), 2
                )

                center = np.rint(
                    pts.mean(axis=0)
                ).astype(int)

                cv2.putText(
                    vis,
                    f"ID{int(getattr(v2, 'ARUCO_ID', 20))}",
                    (int(center[0]), max(20, int(center[1]) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    vis, "ID20 FOUND",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2,
                )
            else:
                cv2.putText(
                    vis, "ID20 NOT FOUND",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2,
                )

            cv2.putText(
                vis,
                f"samples={len(samples)} poses={len(poses)}",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2,
            )
            cv2.putText(
                vis,
                "Click actual tweezer tip; N next; U undo; Q fit",
                (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 2,
            )

            if samples:
                uv = tuple(np.rint(samples[-1]["tip_uv"]).astype(int))
                cv2.circle(vis, uv, 6, (0, 0, 255), -1)

            cv2.imshow(window, vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("n"), ord("N")):
                state["clicked"] = False
                state["armed"] = True
                print("[READY] Click the actual tip in the next pose.")

            elif key in (ord("u"), ord("U")):
                if samples:
                    removed = samples.pop()
                    print("[UNDO] Removed", removed["tip_uv"])
                state["clicked"] = False
                state["armed"] = True

            elif key in (ord("q"), ord("Q"), 27):
                break

    finally:
        if pipeline is not None and started:
            pipeline.stop()
        cv2.destroyAllWindows()
        print("Camera pipeline stopped.")

    if len(samples) < MIN_SAMPLES:
        print(
            f"[ABORT] Only {len(samples)} samples; "
            f"at least {MIN_SAMPLES} are required. No candidate saved."
        )
        return

    fit = fit_offset(samples)
    p = fit["p"]

    result = {
        "camera_serial": SERIAL,
        "image_size": [WIDTH, HEIGHT],
        "marker_dictionary": "DICT_5X5_50",
        "marker_id": int(getattr(v2, "ARUCO_ID", 20)),
        "marker_size_m": 0.020,
        "camera_intrinsics_source": "collect_handeye_samples.py",
        "camera_intrinsics_note": (
            "Same-model right-wrist intrinsics reused as initial values; "
            "left-camera intrinsic equivalence has not been independently verified."
        ),
        "tip_marker_calibration_status": "CANDIDATE_NOT_INDEPENDENTLY_VALIDATED",
        "tip_marker_m": p.tolist(),
        "tip_marker_mm": (p * 1000.0).tolist(),
        "sample_count": len(samples),
        "reprojection_error_px": {
            "rms": fit["rms"],
            "median": fit["median"],
            "max": fit["max"],
        },
        "samples": samples,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("========== LEFT WRIST TIP-MARKER FIT ==========")
    print("P_tip_marker [m] :", p.tolist())
    print("P_tip_marker [mm]:", (p * 1000.0).tolist())
    print("RMS reprojection error [px]:", fit["rms"])
    print("Median error [px]:", fit["median"])
    print("Max error [px]:", fit["max"])
    print("Saved candidate:", OUTPUT)

    if fit["rms"] > 6.0:
        print(
            "[WARNING] RMS > 6 px. Do not use this candidate in the live "
            "detector yet. Check pose diversity, click accuracy, branch "
            "consistency and left-camera intrinsics."
        )
    else:
        print(
            "[INFO] Candidate fit is promising but still requires "
            "independent validation before use."
        )


if __name__ == "__main__":
    main()
