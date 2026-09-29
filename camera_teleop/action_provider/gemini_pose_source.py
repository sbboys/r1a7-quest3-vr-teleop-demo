# -*- coding: utf-8 -*-
"""Gemini 336L RGB-D human pose reader for camera teleoperation.

The source reads aligned color/depth frames from Orbbec SDK v2, runs MediaPipe
Pose/Hands, and exposes a filtered 3D wrist target relative to the shoulder.
"""

from __future__ import annotations

import math
import ctypes
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class HumanPoseTarget:
    wrist_rel_m: np.ndarray
    grip: float
    timestamp: float
    valid: bool = True
    wrist_rel_view: Optional[np.ndarray] = None
    shoulder_m: Optional[np.ndarray] = None
    elbow_m: Optional[np.ndarray] = None
    wrist_m: Optional[np.ndarray] = None
    palm_angles_rad: Optional[np.ndarray] = None


class GeminiPoseSource:
    """Threaded Gemini 336L pose reader.

    Camera frame convention: +X image right, +Y image down, +Z forward.
    """

    MIN_DEPTH_M = 0.25
    MAX_DEPTH_M = 6.0
    MIN_WRIST_SHOULDER_M = 0.12
    MAX_WRIST_SHOULDER_M = 1.15
    MAX_WRIST_SHOULDER_DEPTH_DELTA_M = 0.80
    MIN_ARM_SEGMENT_M = 0.10
    MAX_ARM_SEGMENT_M = 0.75

    def __init__(
        self,
        hand: str = "right",
        show: bool = False,
        mirror_view: bool = False,
        filter_alpha: float = 0.25,
        debug: bool = False,
        min_visibility: float = 0.35,
        min_wrist_shoulder_m: float = 0.035,
        allow_view_fallback: bool = True,
        fallback_hfov_deg: float = 94.0,
        fallback_vfov_deg: float = 68.0,
    ):
        self.hand = hand.lower()
        if self.hand not in ("left", "right"):
            raise ValueError(f"hand must be 'left' or 'right', got {hand!r}")
        self.show = show
        self.mirror_view = mirror_view
        self.debug = debug
        self.min_visibility = float(np.clip(min_visibility, 0.05, 0.95))
        self.min_wrist_shoulder_m = float(np.clip(min_wrist_shoulder_m, 0.01, self.MAX_WRIST_SHOULDER_M))
        self.allow_view_fallback = allow_view_fallback
        self.filter_alpha = float(np.clip(filter_alpha, 0.01, 1.0))
        self.fallback_hfov_deg = fallback_hfov_deg
        self.fallback_vfov_deg = fallback_vfov_deg

        self._lock = threading.Lock()
        self._latest: Optional[HumanPoseTarget] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_rel: Optional[np.ndarray] = None
        self._last_rel_view: Optional[np.ndarray] = None
        self._last_shoulder: Optional[np.ndarray] = None
        self._last_elbow: Optional[np.ndarray] = None
        self._last_wrist: Optional[np.ndarray] = None
        self._last_palm_angles: Optional[np.ndarray] = None
        self._last_reason = "not_started"
        self._debug_counts = {}

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def get_latest(self, max_age_s: float = 0.25) -> Optional[HumanPoseTarget]:
        with self._lock:
            latest = self._latest
        if latest is None:
            return None
        if time.monotonic() - latest.timestamp > max_age_s:
            return None
        return latest

    def _run(self) -> None:
        try:
            self._run_orbbec()
        except Exception as exc:
            with self._lock:
                self._latest = HumanPoseTarget(
                    wrist_rel_m=np.zeros(3, dtype=np.float32),
                    grip=0.0,
                    timestamp=time.monotonic(),
                    valid=False,
                )
            print(f"[GeminiPoseSource] stopped: {exc}")
            self._running = False

    def get_debug_snapshot(self):
        with self._lock:
            return self._last_reason, dict(self._debug_counts)

    def _set_debug_reason(self, reason: str) -> None:
        self._last_reason = reason
        self._debug_counts[reason] = self._debug_counts.get(reason, 0) + 1

    def _run_orbbec(self) -> None:
        import cv2
        import mediapipe as mp
        self._preload_orbbec_libraries()
        from pyorbbecsdk import (
            AlignFilter,
            Config,
            OBFormat,
            OBFrameAggregateOutputMode,
            FrameSet,
            OBSensorType,
            OBStreamType,
            Pipeline,
        )

        pipeline = Pipeline()
        config = Config()

        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        try:
            color_profile = color_profiles.get_video_stream_profile(0, 0, OBFormat.RGB, 0)
        except Exception:
            color_profile = color_profiles.get_default_video_stream_profile()
        config.enable_stream(color_profile)
        config.enable_stream(depth_profiles.get_default_video_stream_profile())
        try:
            config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        except Exception:
            pass
        try:
            pipeline.enable_frame_sync()
        except Exception as exc:
            print(f"[GeminiPoseSource] frame sync unavailable: {exc}")

        pipeline.start(config)
        align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

        mp_pose = mp.solutions.pose
        mp_hands = mp.solutions.hands
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        try:
            while self._running:
                frames: FrameSet = pipeline.wait_for_frames(100)
                if frames is None:
                    continue
                aligned_frames = align_filter.process(frames)
                if aligned_frames is not None:
                    frames = aligned_frames
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if color_frame is None or depth_frame is None:
                    continue

                color = self._frame_to_bgr(color_frame)
                if color is None:
                    self._set_debug_reason("unsupported_color_frame")
                    continue
                depth = self._depth_to_meters(depth_frame)
                if depth is None:
                    self._set_debug_reason("invalid_depth_frame")
                    continue

                target = self._detect_target(color, depth, pose, hands)
                if target is not None:
                    self._set_debug_reason("target_ok")
                    with self._lock:
                        self._latest = target

                if self.show:
                    display = cv2.flip(color, 1) if self.mirror_view else color
                    self._draw_debug(display, target)
                    cv2.imshow("Gemini 336L pose", display)
                    if cv2.waitKey(1) in (ord("q"), 27):
                        self._running = False
                        break
        finally:
            pose.close()
            hands.close()
            pipeline.stop()
            if self.show:
                cv2.destroyWindow("Gemini 336L pose")

    def _detect_target(self, color, depth, pose, hands) -> Optional[HumanPoseTarget]:
        import cv2
        import mediapipe as mp

        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        pose_result = pose.process(rgb)
        hands_result = hands.process(rgb)
        if self.show:
            self._draw_landmarks(color, pose_result, hands_result)
        if not pose_result.pose_landmarks:
            self._set_debug_reason("pose_not_detected")
            return None

        landmarks = pose_result.pose_landmarks.landmark
        height, width = depth.shape[:2]
        if self.hand == "right":
            wrist_lm = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
            elbow_lm = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW]
            shoulder_lm = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        else:
            wrist_lm = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
            elbow_lm = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
            shoulder_lm = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]

        if wrist_lm.visibility < self.min_visibility or shoulder_lm.visibility < self.min_visibility:
            self._set_debug_reason(
                f"low_visibility:wrist={wrist_lm.visibility:.2f},shoulder={shoulder_lm.visibility:.2f}"
            )
            return None

        rel_view = np.array(
            [
                float(wrist_lm.x - shoulder_lm.x),
                float(wrist_lm.y - shoulder_lm.y),
                float(wrist_lm.z - shoulder_lm.z),
            ],
            dtype=np.float32,
        )

        wrist = self._landmark_to_point3d(wrist_lm, depth)
        shoulder = self._landmark_to_point3d(shoulder_lm, depth)
        if wrist is None or shoulder is None:
            if self.allow_view_fallback:
                return self._make_view_fallback_target(rel_view, hands_result, shoulder_lm, elbow_lm, wrist_lm)
            else:
                self._set_debug_reason("missing_wrist_or_shoulder_depth")
                return None
        elbow = self._landmark_to_point3d(elbow_lm, depth) if elbow_lm.visibility >= 0.35 else None

        rel = wrist - shoulder
        rel_norm = float(np.linalg.norm(rel))
        if not (
            self.min_wrist_shoulder_m
            <= rel_norm
            <= self.MAX_WRIST_SHOULDER_M
        ):
            if self.allow_view_fallback:
                return self._make_view_fallback_target(
                    rel_view,
                    hands_result,
                    shoulder_lm,
                    elbow_lm,
                    wrist_lm,
                    reason=f"view_fallback_arm_length:{rel_norm:.3f}",
                )
            else:
                self._set_debug_reason(f"arm_length_out_of_range:{rel_norm:.3f}")
                return None
        if abs(float(rel[2])) > self.MAX_WRIST_SHOULDER_DEPTH_DELTA_M:
            if self.allow_view_fallback:
                return self._make_view_fallback_target(
                    rel_view,
                    hands_result,
                    shoulder_lm,
                    elbow_lm,
                    wrist_lm,
                    reason=f"view_fallback_depth_delta:{float(rel[2]):.3f}",
                )
            else:
                self._set_debug_reason(f"depth_delta_out_of_range:{float(rel[2]):.3f}")
                return None

        elbow_valid = False
        if elbow is not None:
            upper_norm = float(np.linalg.norm(elbow - shoulder))
            forearm_norm = float(np.linalg.norm(wrist - elbow))
            elbow_valid = (
                self.MIN_ARM_SEGMENT_M
                <= upper_norm
                <= self.MAX_ARM_SEGMENT_M
                and self.MIN_ARM_SEGMENT_M
                <= forearm_norm
                <= self.MAX_ARM_SEGMENT_M
            )

        if self._last_shoulder is None or self._last_wrist is None:
            shoulder_f = shoulder
            wrist_f = wrist
        else:
            shoulder_f = self.filter_alpha * shoulder + (1.0 - self.filter_alpha) * self._last_shoulder
            wrist_f = self.filter_alpha * wrist + (1.0 - self.filter_alpha) * self._last_wrist
        if elbow_valid:
            if self._last_elbow is None:
                elbow_f = elbow
            else:
                elbow_f = self.filter_alpha * elbow + (1.0 - self.filter_alpha) * self._last_elbow
            self._last_elbow = elbow_f
        else:
            elbow_f = None
        self._last_shoulder = shoulder_f
        self._last_wrist = wrist_f
        filtered = wrist_f - shoulder_f
        self._last_rel = filtered

        if self._last_rel_view is None:
            rel_view_f = rel_view
        else:
            rel_view_f = self.filter_alpha * rel_view + (1.0 - self.filter_alpha) * self._last_rel_view
        self._last_rel_view = rel_view_f

        grip = self._estimate_grip(hands_result)
        palm_angles = self._estimate_palm_angles(hands_result, depth)
        if palm_angles is not None:
            if self._last_palm_angles is None:
                palm_angles_f = palm_angles
            else:
                diff = self._wrap_angles(palm_angles - self._last_palm_angles)
                palm_angles_f = self._last_palm_angles + self.filter_alpha * diff
            self._last_palm_angles = palm_angles_f
        else:
            palm_angles_f = None

        return HumanPoseTarget(
            wrist_rel_m=filtered.astype(np.float32),
            grip=float(grip),
            timestamp=time.monotonic(),
            valid=True,
            wrist_rel_view=rel_view_f.astype(np.float32),
            shoulder_m=shoulder_f.astype(np.float32),
            elbow_m=elbow_f.astype(np.float32) if elbow_f is not None else None,
            wrist_m=wrist_f.astype(np.float32),
            palm_angles_rad=palm_angles_f.astype(np.float32) if palm_angles_f is not None else None,
        )

    @staticmethod
    def _landmark_to_view_point(landmark) -> np.ndarray:
        return np.array(
            [float(landmark.x), float(landmark.y), float(landmark.z)],
            dtype=np.float32,
        )

    def _make_view_fallback_target(
        self,
        rel_view,
        hands_result,
        shoulder_lm=None,
        elbow_lm=None,
        wrist_lm=None,
        reason="view_fallback_missing_depth",
    ):
        self._set_debug_reason(reason)
        if self._last_rel_view is None:
            rel_view_f = rel_view
        else:
            rel_view_f = self.filter_alpha * rel_view + (1.0 - self.filter_alpha) * self._last_rel_view
        self._last_rel_view = rel_view_f

        # Convert normalized image deltas into a pseudo-metric signal. Direct
        # planar control uses X/Y deltas, so this preserves arm swing even when
        # depth is invalid at the wrist or shoulder pixel.
        rel_pseudo_m = np.array(
            [0.60 * rel_view_f[0], 0.60 * rel_view_f[1], 0.20 * rel_view_f[2]],
            dtype=np.float32,
        )
        self._last_rel = rel_pseudo_m

        shoulder = self._landmark_to_view_point(shoulder_lm) if shoulder_lm is not None else None
        elbow = self._landmark_to_view_point(elbow_lm) if elbow_lm is not None else None
        wrist = self._landmark_to_view_point(wrist_lm) if wrist_lm is not None else None

        return HumanPoseTarget(
            wrist_rel_m=rel_pseudo_m,
            grip=float(self._estimate_grip(hands_result)),
            timestamp=time.monotonic(),
            valid=True,
            wrist_rel_view=rel_view_f.astype(np.float32),
            shoulder_m=shoulder,
            elbow_m=elbow,
            wrist_m=wrist,
        )

    def _landmark_to_point3d(self, landmark, depth) -> Optional[np.ndarray]:
        height, width = depth.shape[:2]
        u = int(np.clip(landmark.x * width, 0, width - 1))
        v = int(np.clip(landmark.y * height, 0, height - 1))
        z = self._median_depth(depth, u, v)
        if z is None:
            return None
        fx, fy, cx, cy = self._fallback_intrinsics(width, height)
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([x, y, z], dtype=np.float32)

    def _median_depth(self, depth, u: int, v: int) -> Optional[float]:
        height, width = depth.shape[:2]
        radius = 3
        patch = depth[
            max(0, v - radius) : min(height, v + radius + 1),
            max(0, u - radius) : min(width, u + radius + 1),
        ]
        valid = patch[(patch > self.MIN_DEPTH_M) & (patch < self.MAX_DEPTH_M)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def _fallback_intrinsics(self, width: int, height: int):
        hfov = math.radians(self.fallback_hfov_deg)
        vfov = math.radians(self.fallback_vfov_deg)
        fx = width / (2.0 * math.tan(hfov * 0.5))
        fy = height / (2.0 * math.tan(vfov * 0.5))
        return fx, fy, width * 0.5, height * 0.5

    def _estimate_grip(self, hands_result) -> float:
        selected = self._selected_hand_landmarks(hands_result)
        if selected is None:
            return 0.0

        lm = selected.landmark
        wrist = np.array([lm[0].x, lm[0].y], dtype=np.float32)
        middle_mcp = np.array([lm[9].x, lm[9].y], dtype=np.float32)
        palm = np.linalg.norm(middle_mcp - wrist) + 1e-6
        tips = [4, 8, 12, 16, 20]
        extension = np.mean(
            [np.linalg.norm(np.array([lm[i].x, lm[i].y], dtype=np.float32) - wrist) for i in tips]
        )
        open_score = np.clip((extension / palm - 1.35) / 0.65, 0.0, 1.0)
        return 1.0 - float(open_score)

    def _selected_hand_landmarks(self, hands_result):
        if not hands_result.multi_hand_landmarks:
            return None

        selected_idx = 0
        if hands_result.multi_handedness:
            for idx, handedness in enumerate(hands_result.multi_handedness):
                label = handedness.classification[0].label.lower()
                if label == self.hand:
                    selected_idx = idx
                    break
        return hands_result.multi_hand_landmarks[selected_idx]

    def _estimate_palm_angles(self, hands_result, depth) -> Optional[np.ndarray]:
        selected = self._selected_hand_landmarks(hands_result)
        if selected is None:
            return None

        lm = selected.landmark
        wrist = self._landmark_to_point3d(lm[0], depth)
        index_mcp = self._landmark_to_point3d(lm[5], depth)
        middle_mcp = self._landmark_to_point3d(lm[9], depth)
        pinky_mcp = self._landmark_to_point3d(lm[17], depth)
        if wrist is None or index_mcp is None or middle_mcp is None or pinky_mcp is None:
            return None

        wrist = self._to_display_camera_frame(wrist)
        index_mcp = self._to_display_camera_frame(index_mcp)
        middle_mcp = self._to_display_camera_frame(middle_mcp)
        pinky_mcp = self._to_display_camera_frame(pinky_mcp)

        across = index_mcp - pinky_mcp
        fingers = middle_mcp - wrist
        across_norm = float(np.linalg.norm(across))
        fingers_norm = float(np.linalg.norm(fingers))
        if across_norm < 0.025 or fingers_norm < 0.035:
            return None

        across = across / across_norm
        fingers = fingers / fingers_norm
        normal = np.cross(across, fingers)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm < 1e-5:
            return None
        normal = normal / normal_norm

        # Display camera frame: +X image right, +Y image down, -Z forward.
        roll = math.atan2(float(across[1]), float(across[0]))
        pitch = math.atan2(float(fingers[2]), float(np.linalg.norm(fingers[:2]) + 1e-6))
        yaw = math.atan2(float(normal[0]), float(normal[2]))
        return np.array([roll, pitch, yaw], dtype=np.float32)

    @staticmethod
    def _wrap_angles(angles: np.ndarray) -> np.ndarray:
        return (angles + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _to_display_camera_frame(point: np.ndarray) -> np.ndarray:
        return np.array([point[0], point[1], -point[2]], dtype=np.float32)

    @staticmethod
    def _preload_orbbec_libraries() -> None:
        candidates = []
        for path in list(dict.fromkeys(os.sys.path)):
            if not path:
                continue
            root = Path(path)
            candidates.extend(
                [
                    root / "libdepthengine.so",
                    root / "libob_usb.so",
                    root / "liblive555.so",
                    root / "libOrbbecSDK.so.1.10",
                    root / "libOrbbecSDK.so",
                ]
            )
        for lib in candidates:
            if lib.exists():
                try:
                    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass

    @staticmethod
    def _depth_to_meters(depth_frame):
        data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        height = depth_frame.get_height()
        width = depth_frame.get_width()
        scale = depth_frame.get_depth_scale()
        if data.size != height * width:
            return None
        depth = data.reshape((height, width)).astype(np.float32) * float(scale)
        valid = depth[depth > 0]
        if valid.size and float(np.median(valid)) > 10.0:
            depth *= 0.001
        return depth

    @staticmethod
    def _frame_to_bgr(frame):
        import cv2
        from pyorbbecsdk import OBFormat

        width = frame.get_width()
        height = frame.get_height()
        color_format = frame.get_format()
        data = np.asanyarray(frame.get_data())
        if color_format == OBFormat.RGB:
            image = np.resize(data, (height, width, 3))
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if color_format == OBFormat.BGR:
            return np.resize(data, (height, width, 3))
        if color_format == OBFormat.MJPG:
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        if color_format == OBFormat.YUYV:
            image = np.resize(data, (height, width, 2))
            return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
        if color_format == OBFormat.UYVY:
            image = np.resize(data, (height, width, 2))
            return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
        if color_format == OBFormat.NV12:
            image = np.resize(data, (height * 3 // 2, width))
            return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_NV12)
        if color_format == OBFormat.NV21:
            image = np.resize(data, (height * 3 // 2, width))
            return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_NV21)
        print(f"[GeminiPoseSource] unsupported color format: {color_format}")
        return None

    @staticmethod
    def _normalized_xy(landmark, width: int, height: int):
        return int(np.clip(landmark.x * width, 0, width - 1)), int(np.clip(landmark.y * height, 0, height - 1))

    def _draw_landmarks(self, color, pose_result, hands_result) -> None:
        import cv2
        import mediapipe as mp

        height, width = color.shape[:2]
        if pose_result.pose_landmarks:
            pose_lm = pose_result.pose_landmarks.landmark
            if self.hand == "right":
                arm_ids = [
                    mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
                    mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
                    mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
                ]
                arm_color = (0, 210, 255)
            else:
                arm_ids = [
                    mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
                    mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
                    mp.solutions.pose.PoseLandmark.LEFT_WRIST,
                ]
                arm_color = (255, 180, 0)

            points = []
            for landmark_id in arm_ids:
                lm = pose_lm[landmark_id]
                if lm.visibility < 0.25:
                    points.append(None)
                else:
                    points.append(self._normalized_xy(lm, width, height))

            for start, end in ((points[0], points[1]), (points[1], points[2])):
                if start is not None and end is not None:
                    cv2.line(color, start, end, arm_color, 4, cv2.LINE_AA)
            for point in points:
                if point is not None:
                    cv2.circle(color, point, 7, arm_color, -1, cv2.LINE_AA)
                    cv2.circle(color, point, 10, (20, 20, 20), 2, cv2.LINE_AA)

        if not hands_result.multi_hand_landmarks:
            return

        drawing_utils = mp.solutions.drawing_utils
        hand_connections = mp.solutions.hands.HAND_CONNECTIONS
        landmark_style = drawing_utils.DrawingSpec(color=(80, 255, 80), thickness=2, circle_radius=3)
        connection_style = drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
        selected_idx = None
        if hands_result.multi_handedness:
            for idx, handedness in enumerate(hands_result.multi_handedness):
                label = handedness.classification[0].label.lower()
                if label == self.hand:
                    selected_idx = idx
                    break

        for idx, hand_landmarks in enumerate(hands_result.multi_hand_landmarks):
            if selected_idx is not None and idx != selected_idx:
                continue
            drawing_utils.draw_landmarks(
                color,
                hand_landmarks,
                hand_connections,
                landmark_style,
                connection_style,
            )

    @staticmethod
    def _draw_debug(color, target: Optional[HumanPoseTarget]) -> None:
        import cv2

        text = "no target"
        if target is not None and target.valid:
            rel = target.wrist_rel_m
            text = f"rel=({rel[0]:+.2f},{rel[1]:+.2f},{rel[2]:+.2f}) grip={target.grip:.2f}"
        cv2.putText(color, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
