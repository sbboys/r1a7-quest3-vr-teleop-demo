#!/usr/bin/env python3
"""Estimate Orbbec RGB camera pose from a fixed AprilTag.

This script reads Orbbec Gemini color frames through pyorbbecsdk, detects one
AprilTag marker with OpenCV, estimates tag->camera pose from SDK RGB intrinsics,
and writes a calibration result JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ORBEC_EXAMPLES = Path(__file__).resolve().parents[1] / "doc/orbbec_gemini_336l/pyorbbecsdk_src/examples"
sys.path.insert(0, str(ORBEC_EXAMPLES))

from utils import frame_to_bgr_image  # noqa: E402
from pyorbbecsdk import Config, Context, OBFormat, OBSensorType, Pipeline  # noqa: E402


def _dist_coeffs(dist) -> np.ndarray:
    # OpenCV uses [k1, k2, p1, p2, k3].
    return np.array([dist.k1, dist.k2, dist.p1, dist.p2, dist.k3], dtype=np.float64)


def _camera_matrix(intr) -> np.ndarray:
    return np.array(
        [
            [intr.fx, 0.0, intr.cx],
            [0.0, intr.fy, intr.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _make_tag_object_points(tag_size_m: float) -> np.ndarray:
    half = tag_size_m / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def _matrix_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rot, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3] = tvec.reshape(3)
    return T


def _invert_transform(T: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = T[:3, :3].T
    inv[:3, 3] = -inv[:3, :3] @ T[:3, 3]
    return inv


def _rotation_angle_deg(R: np.ndarray) -> float:
    value = (np.trace(R) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(value, -1.0, 1.0))))


def _average_transforms(transforms: list[np.ndarray]) -> np.ndarray:
    translations = np.array([T[:3, 3] for T in transforms], dtype=np.float64)
    # For the fixed-tag calibration use case, frame-to-frame orientation noise is
    # small, so an SVD projection of the element-wise mean is adequate.
    R_mean = np.mean([T[:3, :3] for T in transforms], axis=0)
    u, _, vt = np.linalg.svd(R_mean)
    R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1.0
        R = u @ vt
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = translations.mean(axis=0)
    return T


def _jsonify_matrix(T: np.ndarray) -> list[list[float]]:
    return [[float(v) for v in row] for row in T]


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate Orbbec RGB camera extrinsic from AprilTag")
    parser.add_argument("--tag_size_m", type=float, default=0.10, help="black AprilTag square size in meters")
    parser.add_argument("--tag_id", type=int, default=0)
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--timeout_s", type=float, default=20.0)
    parser.add_argument("--output", default="calibration/orbbec_gemini336l_apriltag_extrinsic.json")
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    ctx = Context()
    device_list = ctx.query_devices()
    if device_list.get_count() == 0:
        print("ERROR: Orbbec camera not found")
        return 1

    pipeline = Pipeline()
    config = Config()
    color_profile = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_video_stream_profile(
        0, 0, OBFormat.RGB, 0
    )
    config.enable_stream(color_profile)
    pipeline.start(config)

    # Wait for stream resolution to settle before reading camera param.
    cam_param = None
    for _ in range(30):
        frames = pipeline.wait_for_frames(1000)
        if frames and frames.get_color_frame():
            cam_param = pipeline.get_camera_param()
            break
    if cam_param is None:
        pipeline.stop()
        print("ERROR: no color frames from camera")
        return 1

    K = _camera_matrix(cam_param.rgb_intrinsic)
    dist = _dist_coeffs(cam_param.rgb_distortion)
    obj_pts = _make_tag_object_points(args.tag_size_m)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

    transforms: list[np.ndarray] = []
    reproj_errors: list[float] = []
    last_image = None
    start = time.monotonic()

    print("Collecting AprilTag samples...")
    print(f"  expected family=tag36h11 id={args.tag_id} size={args.tag_size_m:.3f} m")
    print(f"  rgb intrinsics fx={K[0,0]:.3f} fy={K[1,1]:.3f} cx={K[0,2]:.3f} cy={K[1,2]:.3f}")

    while len(transforms) < args.samples and time.monotonic() - start < args.timeout_s:
        frames = pipeline.wait_for_frames(1000)
        if not frames:
            continue
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        image = frame_to_bgr_image(color_frame)
        if image is None:
            continue

        corners, ids, _ = detector.detectMarkers(image)
        if ids is None:
            if args.preview:
                cv2.imshow("Orbbec AprilTag calibration", image)
                cv2.waitKey(1)
            continue

        ids_flat = ids.flatten().tolist()
        if args.tag_id not in ids_flat:
            continue

        idx = ids_flat.index(args.tag_id)
        img_pts = corners[idx].reshape(4, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue

        projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
        err = np.linalg.norm(projected.reshape(4, 2) - img_pts, axis=1).mean()
        T_tag_cam = _matrix_from_rvec_tvec(rvec, tvec)
        transforms.append(T_tag_cam)
        reproj_errors.append(float(err))

        annotated = image.copy()
        cv2.aruco.drawDetectedMarkers(annotated, [corners[idx]], np.array([[args.tag_id]], dtype=np.int32))
        cv2.drawFrameAxes(annotated, K, dist, rvec, tvec, args.tag_size_m * 0.5)
        cv2.putText(
            annotated,
            f"samples {len(transforms)}/{args.samples} err {err:.2f}px",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        last_image = annotated
        if args.preview:
            cv2.imshow("Orbbec AprilTag calibration", annotated)
            cv2.waitKey(1)

    pipeline.stop()
    if args.preview:
        cv2.destroyAllWindows()

    if not transforms:
        print("ERROR: AprilTag was not detected. Check tag36h11 ID 0 visibility and lighting.")
        return 2

    T_tag_cam = _average_transforms(transforms)
    T_cam_tag = _invert_transform(T_tag_cam)
    translations = np.array([T[:3, 3] for T in transforms], dtype=np.float64)
    rot_errors = [_rotation_angle_deg(T_tag_cam[:3, :3].T @ T[:3, :3]) for T in transforms]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path = out_path.with_suffix(".png")
    if last_image is not None:
        cv2.imwrite(str(snapshot_path), last_image)

    result = {
        "timestamp_unix": time.time(),
        "camera": "Orbbec Gemini 336L",
        "tag_family": "tag36h11",
        "tag_id": args.tag_id,
        "tag_size_m": args.tag_size_m,
        "samples_used": len(transforms),
        "rgb_intrinsic": {
            "width": int(cam_param.rgb_intrinsic.width),
            "height": int(cam_param.rgb_intrinsic.height),
            "fx": float(cam_param.rgb_intrinsic.fx),
            "fy": float(cam_param.rgb_intrinsic.fy),
            "cx": float(cam_param.rgb_intrinsic.cx),
            "cy": float(cam_param.rgb_intrinsic.cy),
            "K": _jsonify_matrix(K),
            "distortion_opencv_k1_k2_p1_p2_k3": [float(v) for v in dist],
        },
        "T_tag_to_camera": _jsonify_matrix(T_tag_cam),
        "T_camera_to_tag": _jsonify_matrix(T_cam_tag),
        "quality": {
            "mean_reprojection_error_px": float(np.mean(reproj_errors)),
            "max_reprojection_error_px": float(np.max(reproj_errors)),
            "translation_std_m": [float(v) for v in translations.std(axis=0)],
            "rotation_std_deg_approx": float(np.std(rot_errors)),
        },
        "snapshot": str(snapshot_path),
        "notes": "T_tag_to_camera maps AprilTag-frame points into RGB camera frame: P_camera = R * P_tag + t.",
    }
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nCalibration complete")
    print(f"  samples_used: {len(transforms)}")
    print(f"  mean reprojection error: {np.mean(reproj_errors):.3f} px")
    print(f"  translation std: {translations.std(axis=0).tolist()} m")
    print("  T_tag_to_camera:")
    print(np.array2string(T_tag_cam, precision=6, suppress_small=False))
    print("  T_camera_to_tag:")
    print(np.array2string(T_cam_tag, precision=6, suppress_small=False))
    print(f"  saved: {out_path.resolve()}")
    if last_image is not None:
        print(f"  snapshot: {snapshot_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
