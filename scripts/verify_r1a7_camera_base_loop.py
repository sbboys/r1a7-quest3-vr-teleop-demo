#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORBBEC_EXAMPLES = ROOT / "doc" / "orbbec_gemini_336l" / "pyorbbecsdk_src" / "examples"
sys.path.insert(0, str(ORBBEC_EXAMPLES))

from pyorbbecsdk import OBFormat, OBSensorType, Pipeline  # noqa: E402
from utils import frame_to_bgr_image  # noqa: E402


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    r = a[:3, :3].T @ b[:3, :3]
    cos_theta = (np.trace(r) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return math.degrees(math.acos(cos_theta))


def detect_tag_pose(image: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, tag_size_m: float, tag_id: int):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    ids_flat = ids.flatten().tolist()
    if tag_id not in ids_flat:
        return None
    idx = ids_flat.index(tag_id)
    image_points = corners[idx].reshape(4, 2).astype(np.float64)
    half = tag_size_m / 2.0
    object_points = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    t_tag_to_camera = np.eye(4)
    t_tag_to_camera[:3, :3] = rmat
    t_tag_to_camera[:3, 3] = tvec.reshape(3)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    reproj = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    return t_tag_to_camera, float(reproj.mean()), image


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify R1-A7 camera-to-base calibration with current AprilTag observation")
    parser.add_argument("--calibration", default="calibration/r1a7_base_camera_from_apriltag.json")
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--timeout-s", type=float, default=15.0)
    parser.add_argument("--tag-id", type=int, default=0)
    parser.add_argument("--tag-size-m", type=float, default=0.092)
    parser.add_argument("--output", default="calibration/r1a7_camera_base_loop_verify.json")
    args = parser.parse_args()

    calib = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    t_base_camera = np.array(calib["T_base_camera"], dtype=float)
    t_base_tag_ref = np.array(calib["T_base_tag"], dtype=float)

    pipeline = Pipeline()
    profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = profile_list.get_video_stream_profile(0, 0, OBFormat.RGB, 0)
    config = __import__("pyorbbecsdk").Config()
    config.enable_stream(color_profile)
    pipeline.start(config)

    try:
        cam = pipeline.get_camera_param()
        intr = cam.rgb_intrinsic
        dist = cam.rgb_distortion
        camera_matrix = np.array(
            [[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coeffs = np.array([dist.k1, dist.k2, dist.p1, dist.p2, dist.k3], dtype=np.float64)

        observations = []
        reproj_errors = []
        deadline = time.monotonic() + args.timeout_s
        while len(observations) < args.samples and time.monotonic() < deadline:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
            image = frame_to_bgr_image(color_frame)
            if image is None:
                continue
            detected = detect_tag_pose(image, camera_matrix, dist_coeffs, args.tag_size_m, args.tag_id)
            if detected is None:
                continue
            t_tag_to_camera, reproj, _ = detected
            observations.append(t_base_camera @ t_tag_to_camera)
            reproj_errors.append(reproj)

        if not observations:
            raise RuntimeError("no AprilTag observations collected")

        translations = np.array([obs[:3, 3] for obs in observations])
        mean_translation = translations.mean(axis=0)
        # Keep rotation from the last sample for orientation error; translation is the main closure metric here.
        t_base_tag_obs = observations[-1].copy()
        t_base_tag_obs[:3, 3] = mean_translation

        translation_error = mean_translation - t_base_tag_ref[:3, 3]
        translation_error_norm = float(np.linalg.norm(translation_error))
        rot_error = rotation_error_deg(t_base_tag_ref, t_base_tag_obs)

        result = {
            "calibration": str(Path(args.calibration).resolve()),
            "samples_used": len(observations),
            "tag_size_m": args.tag_size_m,
            "tag_id": args.tag_id,
            "mean_reprojection_error_px": float(np.mean(reproj_errors)),
            "T_base_tag_reference": t_base_tag_ref.tolist(),
            "T_base_tag_observed": t_base_tag_obs.tolist(),
            "translation_error_m": translation_error.tolist(),
            "translation_error_norm_m": translation_error_norm,
            "rotation_error_deg": rot_error,
            "observed_translation_std_m": translations.std(axis=0).tolist(),
        }
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        print("Camera/base loop verification")
        print(f"  samples_used: {len(observations)}")
        print(f"  mean_reprojection_error_px: {np.mean(reproj_errors):.3f}")
        print("  reference T_base_tag translation:", np.array2string(t_base_tag_ref[:3, 3], precision=6))
        print("  observed  T_base_tag translation:", np.array2string(mean_translation, precision=6))
        print("  translation_error_m:", np.array2string(translation_error, precision=6))
        print(f"  translation_error_norm_m: {translation_error_norm:.6f}")
        print(f"  rotation_error_deg: {rot_error:.3f}")
        print("  observed_translation_std_m:", np.array2string(translations.std(axis=0), precision=6))
        print(f"  saved: {Path(args.output).resolve()}")
    finally:
        pipeline.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
