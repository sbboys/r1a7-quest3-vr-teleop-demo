#!/usr/bin/env python3
# Left wrist 007Z: External wire and tweezer-tip perception migration.
# RGB only. Shadow/perception only. Never sends robot commands.

import sys
import select
import json
import time
import logging
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]

HAND_EYE_DIR = ROOT / "vision_dual_camera" / "handeye"
TASK6_DIR = ROOT / "vision_dual_camera" / "task6_threading"
TIP_DIR = TASK6_DIR / "perception" / "tweezer_tip"
WIRE_DIR = (
    TASK6_DIR / "perception" / "wire" / "proposed" / "r1a7_wire"
)

for directory in (HAND_EYE_DIR, TIP_DIR, TASK6_DIR, WIRE_DIR):
    sys.path.insert(0, str(directory))

import collect_handeye_samples as hc
import external_wire_grasp_guidance_v1 as ext

from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat
LEFT_WRIST_SN = "CPCBC530007Z"
COLOR_WIDTH = 1280
COLOR_HEIGHT = 800
COLOR_FPS = 30

OUTPUT_DIR = Path(
    "/tmp/r1a7_left_wrist_007z_external_migration"
)
LOG_DIR = OUTPUT_DIR / "logs"
FRAME_DIR = OUTPUT_DIR / "frames"
STATUS_FILE = OUTPUT_DIR / "left_wrist_007z_perception.json"
LOG_FILE = LOG_DIR / "left_wrist_007z_perception.log"

# Use the same candidate calibration selected by the current External
# program. This remains a candidate and must be independently validated.
TIP_CALIB = (
    ROOT / "calibration"
    / "left_wrist_007z_marker_tip_calibration_candidate_no_sample5.json"
)



def click_tip(frame):
    """Manually select the wire free tip in the current left-wrist image."""
    selected = {"uv": None}
    window = "TASK7 LEFT 007Z - Select wire free tip"

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected["uv"] = (int(x), int(y))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, mouse_callback)
    try:
        while True:
            vis = frame.copy()
            cv2.putText(
                vis,
                "Click actual wire free tip; Q/Esc cancels",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )
            cv2.imshow(window, vis)
            key = cv2.waitKey(20) & 0xFF
            if selected["uv"] is not None:
                return selected["uv"]
            if key in (ord("q"), ord("Q"), 27):
                return None
    finally:
        cv2.destroyWindow(window)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def point_xy(value):
    if value is None:
        return None
    try:
        p = np.asarray(value, dtype=np.float64).reshape(-1)
        if p.size < 2 or not np.isfinite(p[:2]).all():
            return None
        return [int(round(float(p[0]))), int(round(float(p[1])))]
    except (TypeError, ValueError):
        return None


def line_xy(value):
    if value is None:
        return []
    try:
        a = np.asarray(value, dtype=np.float64)
        if a.size == 0 or a.ndim != 2 or a.shape[1] < 2:
            return []
        a = a[:, :2]
        a = a[np.isfinite(a).all(axis=1)]
        return a.tolist()
    except (TypeError, ValueError):
        return []


def write_status(data):
    tmp = STATUS_FILE.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(jsonable(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(STATUS_FILE)



def detect_aruco_debug(frame):
    """Independent diagnostic for DICT_5X5_50 marker visibility."""
    aruco = getattr(cv2, "aruco", None)
    if aruco is None:
        return [], [], "CV2_ARUCO_UNAVAILABLE"

    try:
        dictionary = aruco.getPredefinedDictionary(
            aruco.DICT_5X5_50
        )

        if hasattr(aruco, "DetectorParameters"):
            params = aruco.DetectorParameters()
        else:
            params = aruco.DetectorParameters_create()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if hasattr(aruco, "ArucoDetector"):
            detector = aruco.ArucoDetector(dictionary, params)
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = aruco.detectMarkers(
                gray,
                dictionary,
                parameters=params,
            )

        ids_list = (
            []
            if ids is None
            else [int(x) for x in np.asarray(ids).reshape(-1)]
        )

        return list(corners or []), ids_list, "OK"

    except Exception as exc:
        return [], [], f"ERROR:{type(exc).__name__}:{exc}"




# The ID20 marker is 20 mm wide. Estimate image scale from its detected corners.
ARUCO_MARKER_SIZE_MM = 20.0
WIRE_ROI_SIZE_MM = 10.0
WIRE_ROI_MIN_PX = 24


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=str(LOG_FILE),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    context = None
    pipeline = None
    pipeline_started = False

    write_status({
        "task": "task7_left_wrist_external_migration",
        "camera_serial": LEFT_WRIST_SN,
        "state": "STARTING",
        "timestamp": time.time(),
        "valid": False,
        "motion_enabled": False,
        "motion_command_sent": False,
    })

    try:
        print("=" * 72)
        print("TASK7 LEFT WRIST 007Z")
        print("External wire detector + External V4 tweezer tip")
        print("RGB ONLY / NO FUSION / NO ROBOT MOTION")
        print("=" * 72)

        if not TIP_CALIB.is_file():
            raise FileNotFoundError(
                "External candidate tip calibration not found: "
                + str(TIP_CALIB)
            )

        # ----------------------------------------------------
        # Camera calibration:
        # reuse the existing 1280x800 wrist-camera intrinsics.
        # ----------------------------------------------------
        K = np.asarray(hc.CAMERA_MATRIX, dtype=np.float64)
        dist = np.asarray(hc.DIST_COEFFS, dtype=np.float64)

        if K.shape != (3, 3):
            raise RuntimeError(
                f"Invalid wrist camera matrix shape: {K.shape}"
            )

        # Load the left-wrist fit directly. Its schema uses tip_marker_m,
        # unlike External's robust_tip_marker_m field. Do not silently
        # fall back to External's old coarse offset.
        calib_data = json.loads(TIP_CALIB.read_text(encoding="utf-8"))

        if calib_data.get("camera_serial") != LEFT_WRIST_SN:
            raise RuntimeError(
                "Tip calibration camera_serial mismatch: "
                f"{calib_data.get('camera_serial')!r}"
            )

        p_tip_marker = np.asarray(
            calib_data["tip_marker_m"],
            dtype=np.float64,
        ).reshape(-1)

        if p_tip_marker.shape != (3,) or not np.isfinite(p_tip_marker).all():
            raise RuntimeError("Invalid left-wrist tip_marker_m.")

        if int(calib_data.get("sample_count", 0)) < 8:
            raise RuntimeError("Insufficient left-wrist calibration samples.")

        fit_error = calib_data.get("reprojection_error_px", {})
        print("[CALIB] Left-wrist fitted tip_marker_m =", p_tip_marker.tolist())
        print("[CALIB] Left-wrist fit errors =", fit_error)
        print("[CALIB] Status =", calib_data.get(
            "tip_marker_calibration_status", "UNKNOWN"
        ))

        print("[CALIB] Intrinsics source: collect_handeye_samples.py")
        print("[CALIB] K =", K.tolist())
        print("[CALIB] dist =", dist.reshape(-1).tolist())
        print("[CALIB] Tip-marker candidate =", TIP_CALIB)
        print("[CALIB] P_tip_marker =", np.asarray(p_tip_marker).tolist())
        print("[CALIB] Tip-marker calibration is not independently validated.")

        # ----------------------------------------------------
        # Select the Orbbec camera by serial number.
        # ----------------------------------------------------
        context, device = hc.find_orbbec_device(LEFT_WRIST_SN)
        actual_sn = hc.normalize_serial(
            device.get_device_info().get_serial_number()
        ).strip()

        if actual_sn != LEFT_WRIST_SN:
            raise RuntimeError(
                f"Serial mismatch: expected {LEFT_WRIST_SN}, got {actual_sn}"
            )

        print("[CAMERA] Locked to", actual_sn)

        pipeline = Pipeline(device)
        config = Config()

        profiles = pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        )
        profile = profiles.get_video_stream_profile(
            COLOR_WIDTH,
            COLOR_HEIGHT,
            OBFormat.RGB,
            COLOR_FPS,
        )

        config.enable_stream(profile)
        pipeline.start(config)
        pipeline_started = True

        print(
            f"[CAMERA] RGB {COLOR_WIDTH}x{COLOR_HEIGHT}"
            f" @ {COLOR_FPS} FPS; depth disabled"
        )

        # Wait for the first color frame.
        frame = None
        while frame is None:
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                continue
            color = frames.get_color_frame()
            if color is not None:
                frame = hc.color_frame_to_bgr(color)

        print("[CAMERA] First frame:", frame.shape)
        initial_marker_corners, initial_marker_ids, _ = (
            detect_aruco_debug(frame)
        )
        print("[WIRE ROI] External dynamic ROI enabled", flush=True)

        # Wire initialization: select the real free tip.
        # No manual tweezer exclusion mask is used.
        # ----------------------------------------------------
        print("[WIRE] External detector pipeline; SAM2 mask gate disabled.", flush=True)

        seed = click_tip(frame)
        if seed is None:
            print("[WIRE] Initialization cancelled.")
            return

        wire = ext.WireLocalDetectorV65(
            local_length_px=200.0,
            grasp_offset_px=60.0,
            grasp_band_start_px=40.0,
            grasp_band_end_px=110.0,
        )
        wire.set_seed_tip(seed)

        active_roi = ext.roi_from_tip(seed, frame.shape)
        anchor_roi = active_roi
        miss_count = 0
        wire_ready = True

        # Reuse External temporal gate and External V4 marker-tip method.
        tip_gate = ext.ExternalTipGate()


        print("[WIRE] Seed:", seed)
        print("[WIRE] Initial ROI:", active_roi)
        print("[RUN] R/W=reselect wire tip, S=save, Q=quit; hole detection disabled")
        window = "TASK7 LEFT 007Z - EXTERNAL WIRE + TIP"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        frame_index = 0
        last_log = 0.0


        while True:
            frames = pipeline.wait_for_frames(200)
            if frames is None:
                continue

            color = frames.get_color_frame()
            if color is None:
                continue

            frame = hc.color_frame_to_bgr(color)
            if frame is None:
                continue


            # Update independent hole-mouth detector every frame.

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Independent diagnostic: detect and display visible ArUco IDs.
            debug_corners, debug_ids, debug_aruco_status = (
                detect_aruco_debug(frame)
            )
            debug_id20_found = 20 in debug_ids
            # External V4: ArUco ID 20 -> pose candidates ->
            # local tip refinement -> ExternalTipGate.
            # Use the left wrist camera K/dist, not External K/dist.
            # ------------------------------------------------
            v4_result = None
            marker_corners = None

            try:
                v4_result, marker_corners = ext.ext_v4.get_v4_tip(
                    frame,
                    gray,
                    K,
                    dist,
                    p_tip_marker,
                    tip_gate.anchor,
                    tip_gate.branch,
                    tip_gate.prior_age,
                )
                # Preserve cold-start relock samples locally.
                # ExternalTipGate clears its relock buffer when the Kalman
                # filter has no track yet; without this workaround, the
                # 3-frame initialization can repeatedly restart at frame 1.
                gate_relock_before = [
                    np.asarray(p, dtype=np.float64).copy()
                    for p in tip_gate.relock
                ]
                gate_branch_before = list(tip_gate.relock_branch)

                tip_state = tip_gate.update(v4_result)

                if (
                    isinstance(v4_result, dict)
                    and tip_gate.anchor is None
                    and isinstance(tip_state, dict)
                    and tip_state.get("mode") == "REJECT"
                    and "NO_TRACK" in str(tip_state.get("status", ""))
                ):
                    raw_candidate = v4_result.get("tip_uv")

                    if raw_candidate is not None:
                        candidate = np.asarray(
                            raw_candidate, dtype=np.float64
                        ).reshape(-1)

                        if candidate.size >= 2 and np.isfinite(candidate[:2]).all():
                            branch_id = int(
                                v4_result.get("branch_id", -1)
                            )
                            relock_frames = int(
                                ext.ext_v4.EXT_TIP_RELOCK_FRAMES
                            )

                            tip_gate.relock = (
                                gate_relock_before
                                + [candidate[:2].copy()]
                            )[-relock_frames:]

                            tip_gate.relock_branch = (
                                gate_branch_before
                                + [branch_id]
                            )[-relock_frames:]

                            tip_state["status"] = (
                                str(tip_state.get("status", "REJECT"))
                                + " / INIT_BUFFER_PRESERVED"
                            )

                if not isinstance(tip_state, dict):
                    tip_state = {
                        "valid": False,
                        "measurement_valid": False,
                        "predicted_valid": False,
                        "tip": None,
                        "status": "TIP_GATE_EMPTY",
                    }

            except Exception as exc:
                v4_result = None
                marker_corners = None
                if frame_index % 30 == 0:
                    logging.exception("External V4 tip detection failed")
                tip_state = {
                    "valid": False,
                    "measurement_valid": False,
                    "predicted_valid": False,
                    "tip": None,
                    "status": "TIP_ERROR:" + type(exc).__name__,
                }

            tip_pixel = None
            if tip_state.get("valid", False):
                tip_pixel = point_xy(tip_state.get("tip"))

            # Keep raw V4 candidate separate from the gated output.
            v4_tip_candidate = (
                point_xy(v4_result.get("tip_uv"))
                if isinstance(v4_result, dict)
                else None
            )
            v4_branch_id = (
                v4_result.get("branch_id")
                if isinstance(v4_result, dict)
                else None
            )

            # ------------------------------------------------
            # Match the actual External wire detection call.
            # Use the left-wrist V4 tip, with no SAM2 mask gate.
            try:
                if wire_ready and active_roi is not None:
                    wire_result = wire.detect(
                        frame,
                        active_roi,
                        tweezer_exclusion_mask=None,
                        tweezer_tip=tip_pixel,
                    )
                    if not isinstance(wire_result, dict):
                        wire_result = {}
                else:
                    wire_result = {
                        "valid": False,
                        "tip_valid": False,
                        "grasp_guidance_valid": False,
                        "state": "WAIT_WIRE_CLICK",
                        "candidate_count": 0,
                        "centerline": [],
                        "debug_centerline": [],
                        "wire_tip_uv": None,
                        "grasp_target_uv": None,
                        "rank_score": 0.0,
                        "score": 0.0,
                    }
            except Exception as exc:
                if frame_index % 30 == 0:
                    logging.exception("External V65 wire detection failed")
                wire_result = {
                    "valid": False,
                    "tip_valid": False,
                    "grasp_guidance_valid": False,
                    "state": "WIRE_ERROR:" + type(exc).__name__,
                    "candidate_count": 0,
                    "centerline": [],
                    "debug_centerline": [],
                    "wire_tip_uv": None,
                    "grasp_target_uv": None,
                    "rank_score": 0.0,
                    "score": 0.0,
                }

            centerline = line_xy(wire_result.get("centerline"))
            raw_wire_tip = point_xy(wire_result.get("wire_tip_uv"))
            wire_tip_observed = bool(
                wire_result.get("tip_valid", False)
                and raw_wire_tip is not None
            )
            wire_tip = raw_wire_tip if wire_tip_observed else None
            wire_tip_display = wire_tip
            tip_tracking_source = (
                "DETECTOR" if wire_tip is not None else "LOST"
            )
            target = point_xy(wire_result.get("grasp_target_uv"))

            wire_geometry_valid = bool(
                wire_result.get("valid", False)
                or wire_result.get("grasp_guidance_valid", False)
            )

            if (
                wire_ready
                and wire_geometry_valid
                and len(wire_result.get("centerline", [])) >= 2
            ):
                miss_count = 0
                active_roi = ext.roi_from_path(
                    wire_result["centerline"],
                    frame.shape,
                )
                if wire_result.get("tip_valid", False):
                    anchor_roi = active_roi

            elif wire_ready and anchor_roi is not None:
                miss_count = min(miss_count + 1, 20)
                active_roi = ext.expand_roi(
                    anchor_roi,
                    frame.shape,
                    min(10 * miss_count, 60),
                )

            # Visualization
            # ------------------------------------------------
            vis = frame.copy()


            # Orange = raw V4 tip candidate; cyan/yellow = gated tip.
            if v4_tip_candidate is not None:
                cv2.drawMarker(
                    vis, tuple(v4_tip_candidate), (0, 165, 255),
                    cv2.MARKER_CROSS, 24, 2,
                )
                cv2.putText(
                    vis, f"V4 RAW b={v4_branch_id}",
                    (v4_tip_candidate[0] + 8, v4_tip_candidate[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 2,
                )

            # Draw every marker found by the independent diagnostic detector.
            if debug_corners and debug_ids:
                cv2.aruco.drawDetectedMarkers(
                    vis,
                    debug_corners,
                    np.asarray(debug_ids, dtype=np.int32).reshape(-1, 1),
                )

            id20_color = (0, 255, 0) if debug_id20_found else (0, 0, 255)
            ids_text = ",".join(map(str, debug_ids)) if debug_ids else "NONE"

            cv2.putText(
                vis,
                f"DEBUG ARUCO IDs={ids_text} ID20={debug_id20_found}",
                (15, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                id20_color,
                2,
            )
            cv2.putText(
                vis,
                f"ARUCO API={debug_aruco_status}",
                (15, 124),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
            )

            if active_roi is not None:
                x, y, w, h = map(int, active_roi)
                cv2.rectangle(
                    vis, (x, y), (x + w - 1, y + h - 1),
                    (255, 200, 0), 2,
                )

            if len(centerline) >= 2:
                pts = np.asarray(centerline, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [pts], False, (0, 255, 0), 2)

            if wire_tip_display is not None:
                display_tip = tuple(wire_tip_display)
                cv2.circle(vis, display_tip, 7, (0, 0, 255), -1)
                cv2.circle(vis, display_tip, 10, (255, 255, 255), 1)

                tip_label = (
                    "WIRE TIP"
                    if tip_tracking_source == "DETECTOR"
                    else "WIRE TIP (FLOW)"
                )
                cv2.putText(
                    vis,
                    tip_label,
                    (display_tip[0] + 9, display_tip[1] - 9),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    2,
                )
            if target is not None:
                cv2.drawMarker(
                    vis, tuple(target), (0, 255, 255),
                    cv2.MARKER_CROSS, 18, 2,
                )

            if tip_pixel is not None:
                cv2.drawMarker(
                    vis, tuple(tip_pixel), (255, 255, 0),
                    cv2.MARKER_CROSS, 26, 2,
                )
                cv2.putText(
                    vis, "TWEEZER TIP",
                    (tip_pixel[0] + 10, tip_pixel[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 2,
                )

            wire_state = str(wire_result.get("state", "UNKNOWN"))
            tip_status = str(tip_state.get("status", "UNKNOWN"))
            guidance_valid = bool(
                tip_pixel is not None
                and target is not None
                and wire_result.get("grasp_guidance_valid", False)
            )

            cv2.putText(
                vis,
                f"WIRE={wire_state}",
                (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                vis,
                f"TIP={tip_status}",
                (15, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                vis,
                f"miss={miss_count} guidance_valid={guidance_valid} SHADOW ONLY",
                (15, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                2,
            )

            v4_summary = (
                {
                    k: jsonable(v4_result[k])
                    for k in (
                        "valid", "status", "state", "tip", "tip_uv",
                        "branch", "branch_id", "mode", "reason",
                        "score", "quality", "coarse_uv",
                        "candidate_count", "marker_reproj_px",
                        "refine_dist",
                    )
                    if k in v4_result
                }
                if isinstance(v4_result, dict)
                else None
            )

            status = {
                "task": "task7_left_wrist_external_migration",
                "camera_serial": LEFT_WRIST_SN,
                "timestamp": time.time(),
                "debug_aruco_status": debug_aruco_status,
                "debug_aruco_ids": debug_ids,
                "debug_aruco_id20_found": debug_id20_found,
                "v4_raw_result": v4_summary,
                "v4_tip_candidate_uv": v4_tip_candidate,
                "v4_branch_id": v4_branch_id,
                "tip_gate_anchor_uv": point_xy(tip_gate.anchor),
                "tip_gate_branch": tip_gate.branch,
                "tip_gate_relock_count": len(tip_gate.relock),
                "tip_gate_relock_branches": list(tip_gate.relock_branch),
                "tip_gate_prior_age": int(tip_gate.prior_age),
                "frame_index": frame_index,
                "image_width": int(frame.shape[1]),
                "image_height": int(frame.shape[0]),

                "camera_intrinsics_source": "collect_handeye_samples.py right-wrist calibration",
                "tip_marker_calibration_source": str(TIP_CALIB),
                "active_roi": list(active_roi) if active_roi is not None else None,
                "wire_roi_size_mm": None,
                "wire_roi_size_px": int(active_roi[2]) if active_roi is not None else None,
                "anchor_roi": list(anchor_roi) if anchor_roi is not None else None,
                "miss_count": miss_count,

                "wire_state": wire_state,
                "wire_valid": bool(wire_result.get("valid", False)),
                "wire_tip_valid": bool(wire_result.get("tip_valid", False)),
                "wire_tip_uv": wire_tip,
                "wire_tip_detector_uv": raw_wire_tip,
                "wire_tip_display_uv": wire_tip_display,
                "wire_tip_observed_this_frame": bool(wire_tip_observed),
                "wire_tip_tracking_valid": wire_tip_display is not None,
                "wire_tip_display_source": tip_tracking_source,
                "raw_wire_tip_uv": jsonable(wire_result.get("raw_wire_tip_uv")),
                "grasp_target_uv": target,
                "raw_grasp_target_uv": None,
                "grasp_guidance_valid": False,
                "wire_centerline_uv": centerline[:300],
                "wire_candidate_count": wire_result.get("candidate_count"),
                "wire_score": wire_result.get("score"),
                "wire_rank_score": wire_result.get("rank_score"),
                "wire_tip_filter_state": wire_result.get(
                    "wire_tip_filter_state"
                ),
                "grasp_filter_state": wire_result.get(
                    "grasp_filter_state"
                ),
                "tweezer_tip_uv": tip_pixel,
                "tweezer_tip_valid": tip_pixel is not None,
                "tweezer_tip_measurement_valid": bool(
                    tip_state.get("measurement_valid", False)
                ),
                "tweezer_tip_predicted_valid": bool(
                    tip_state.get("predicted_valid", False)
                ),
                "tweezer_tip_state": tip_status,
                "guidance_valid": guidance_valid,

                "motion_enabled": False,
                "motion_command_sent": False,
            }
            write_status(status)

            now = time.time()
            if now - last_log >= 1.0:
                msg = (
                    f"[TASK7] wire={wire_state} "
                    f"wire_tip={wire_tip} target={target} "
                    f"tip={tip_pixel} tip_state={tip_status} "
                    f"guidance={guidance_valid} miss={miss_count}"
                )
                print(msg, flush=True)
                logging.info(msg)
                last_log = now

            cv2.imshow(window, vis)
            key = cv2.waitKey(1) & 0xFF

            # Also accept hotkeys typed into the launching terminal.
            # Type one command letter and press Enter.
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0)
                if ready:
                    command = sys.stdin.readline().strip().lower()

                    if command and command[0] in ("h", "r", "w", "t", "s", "q"):
                        command_key = command[0]
                        print(
                            f"[KEYBOARD] terminal command: {command_key}",
                            flush=True,
                        )

                        if command_key == "h":
                            print(
                                "[KEYBOARD] Hole detection is disabled.",
                                flush=True,
                            )
                            key = 255
                        elif command_key == "r":
                            key = ord("r")
                        else:
                            key = ord(command_key)

            except (OSError, ValueError) as exc:
                print(
                    f"[KEYBOARD] Terminal input error: {exc}",
                    flush=True,
                )

            if key in (ord("q"), ord("Q"), 27):
                break

            if key in (ord("r"), ord("R"), ord("w"), ord("W")):
                new_seed = click_tip(frame)
                if new_seed is not None:
                    seed = new_seed
                    wire = ext.WireLocalDetectorV65(
                        local_length_px=200.0,
                        grasp_offset_px=60.0,
                        grasp_band_start_px=40.0,
                        grasp_band_end_px=110.0,
                    )
                    wire.set_seed_tip(seed)
                    active_roi = ext.roi_from_tip(seed, frame.shape)
                    anchor_roi = active_roi
                    miss_count = 0
                    wire_ready = True
                    print("[WIRE] Reinitialized:", seed)

            if key in (ord("t"), ord("T")):
                tip_gate = ext.ExternalTipGate()
                print("[TIP] External tip tracker reset.")

            if key in (ord("s"), ord("S")):
                image_path = FRAME_DIR / f"left_wrist_{time.time_ns()}.png"
                cv2.imwrite(str(image_path), vis)
                print("[TASK7] Saved:", image_path)

            frame_index += 1

    except KeyboardInterrupt:
        print("\n[TASK7] Stopped by Ctrl+C.")

    except Exception as exc:
        logging.exception("Task7 startup/runtime failure")
        write_status({
            "task": "task7_left_wrist_external_migration",
            "camera_serial": LEFT_WRIST_SN,
            "timestamp": time.time(),
            "state": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "motion_enabled": False,
            "motion_command_sent": False,
        })
        raise

    finally:
        if pipeline is not None and pipeline_started:
            pipeline.stop()
        cv2.destroyAllWindows()
        print("[TASK7] Pipeline stopped. No robot motion commands sent.")


if __name__ == "__main__":
    main()
