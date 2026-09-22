#!/usr/bin/env python3

import importlib.util
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Paths
# ============================================================

THIS_FILE = Path(__file__).resolve()

COLLECTOR_PATH = (
    THIS_FILE.parents[1]
    / "handeye"
    / "collect_handeye_samples.py"
)


# ============================================================
# Fixed camera
# ============================================================

FIXED_DEVICE = "/dev/video16"
FIXED_WIDTH = 1280
FIXED_HEIGHT = 720
FIXED_FPS = 30


# ============================================================
# Tracking parameters
# ============================================================

ROI_RADIUS = 35

MAX_CORNERS = 50
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 4
BLOCK_SIZE = 7

LK_WIN = (21, 21)
LK_LEVEL = 3

MIN_TRACK_POINTS = 5

MAX_FB_ERROR_PX = 1.5
MAX_FRAME_JUMP_PX = 35.0

RESEED_INTERVAL = 10


# ============================================================
# Load existing wrist-camera implementation
# ============================================================

def load_collector():
    spec = importlib.util.spec_from_file_location(
        "r1a7_handeye_collector",
        str(COLLECTOR_PATH),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot load collector: {COLLECTOR_PATH}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


# ============================================================
# Local LK target tracker
#
# Important:
# - only tracks locally from manually selected target
# - NEVER searches the full image
# - if lost -> LOST
# - does NOT jump automatically to another hole
# ============================================================

class LKTargetTracker:

    def __init__(self, label):
        self.label = label

        self.center = None
        self.points = None

        self.initialized = False
        self.valid = False

        self.frame_counter = 0
        self.num_points = 0

    def reset(self):
        self.center = None
        self.points = None

        self.initialized = False
        self.valid = False

        self.frame_counter = 0
        self.num_points = 0

    def _inside_image(self, gray, center):
        h, w = gray.shape[:2]

        x = float(center[0])
        y = float(center[1])

        return (
            0 <= x < w
            and 0 <= y < h
        )

    def _seed_features(self, gray):
        if self.center is None:
            return False

        h, w = gray.shape[:2]

        cx = int(round(self.center[0]))
        cy = int(round(self.center[1]))

        x0 = max(0, cx - ROI_RADIUS)
        x1 = min(w, cx + ROI_RADIUS + 1)

        y0 = max(0, cy - ROI_RADIUS)
        y1 = min(h, cy + ROI_RADIUS + 1)

        if (
            x1 - x0 < 10
            or y1 - y0 < 10
        ):
            return False

        mask = np.zeros_like(
            gray,
            dtype=np.uint8,
        )

        mask[y0:y1, x0:x1] = 255

        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=MAX_CORNERS,
            qualityLevel=QUALITY_LEVEL,
            minDistance=MIN_DISTANCE,
            mask=mask,
            blockSize=BLOCK_SIZE,
        )

        if pts is None:
            self.points = None
            self.num_points = 0
            return False

        if len(pts) < MIN_TRACK_POINTS:
            self.points = None
            self.num_points = int(len(pts))
            return False

        self.points = np.asarray(
            pts,
            dtype=np.float32,
        ).reshape(-1, 1, 2)

        self.num_points = len(self.points)

        return True

    def initialize(self, gray, xy):
        self.center = np.array(
            [float(xy[0]), float(xy[1])],
            dtype=np.float64,
        )

        self.initialized = True
        self.frame_counter = 0

        self.valid = self._seed_features(
            gray
        )

        if not self.valid:
            print(
                f"[WARN] {self.label}: "
                "not enough features around selected point"
            )

        return self.valid

    def update(self, prev_gray, gray):

        if not self.initialized:
            return

        if self.points is None:
            self.valid = False
            return

        if len(self.points) < MIN_TRACK_POINTS:
            self.valid = False
            return

        next_pts, st_fwd, _ = (
            cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                self.points,
                None,
                winSize=LK_WIN,
                maxLevel=LK_LEVEL,
                criteria=(
                    cv2.TERM_CRITERIA_EPS
                    | cv2.TERM_CRITERIA_COUNT,
                    30,
                    0.01,
                ),
            )
        )

        if (
            next_pts is None
            or st_fwd is None
        ):
            self.valid = False
            return

        back_pts, st_back, _ = (
            cv2.calcOpticalFlowPyrLK(
                gray,
                prev_gray,
                next_pts,
                None,
                winSize=LK_WIN,
                maxLevel=LK_LEVEL,
                criteria=(
                    cv2.TERM_CRITERIA_EPS
                    | cv2.TERM_CRITERIA_COUNT,
                    30,
                    0.01,
                ),
            )
        )

        if (
            back_pts is None
            or st_back is None
        ):
            self.valid = False
            return

        p0 = self.points.reshape(-1, 2)
        p1 = next_pts.reshape(-1, 2)
        pb = back_pts.reshape(-1, 2)

        st_fwd = st_fwd.reshape(-1).astype(bool)
        st_back = st_back.reshape(-1).astype(bool)

        fb_error = np.linalg.norm(
            p0 - pb,
            axis=1,
        )

        good = (
            st_fwd
            & st_back
            & np.isfinite(fb_error)
            & (fb_error < MAX_FB_ERROR_PX)
        )

        p0 = p0[good]
        p1 = p1[good]

        if len(p0) < MIN_TRACK_POINTS:
            self.valid = False
            self.num_points = len(p0)
            return

        displacement = p1 - p0

        median_disp = np.median(
            displacement,
            axis=0,
        )

        residual = np.linalg.norm(
            displacement - median_disp,
            axis=1,
        )

        median_residual = float(
            np.median(residual)
        )

        threshold = max(
            2.0,
            2.5 * median_residual + 0.5,
        )

        inliers = residual <= threshold

        p1 = p1[inliers]
        displacement = displacement[inliers]

        if len(p1) < MIN_TRACK_POINTS:
            self.valid = False
            self.num_points = len(p1)
            return

        median_disp = np.median(
            displacement,
            axis=0,
        )

        jump = float(
            np.linalg.norm(median_disp)
        )

        if jump > MAX_FRAME_JUMP_PX:
            print(
                f"[LOST] {self.label}: "
                f"frame jump {jump:.1f}px"
            )

            self.valid = False
            return

        proposed_center = (
            self.center
            + median_disp
        )

        if not self._inside_image(
            gray,
            proposed_center,
        ):
            self.valid = False
            return

        self.center = proposed_center

        self.points = p1.astype(
            np.float32
        ).reshape(-1, 1, 2)

        self.num_points = len(self.points)
        self.valid = True

        self.frame_counter += 1

        if (
            self.frame_counter
            % RESEED_INTERVAL
            == 0
            or self.num_points < 12
        ):
            self._seed_features(gray)


# ============================================================
# Camera view state
# ============================================================

class ViewState:

    def __init__(self, name):
        self.name = name

        self.hole = LKTargetTracker(
            f"{name}/HOLE"
        )

        self.tip = LKTargetTracker(
            f"{name}/TIP"
        )

        self.prev_gray = None
        self.latest_gray = None

    def reset(self):
        self.hole.reset()
        self.tip.reset()

        print(
            f"[RESET] {self.name}"
        )

    def mouse_callback(
        self,
        event,
        x,
        y,
        flags,
        param,
    ):
        if self.latest_gray is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:

            if not self.hole.initialized:

                self.hole.initialize(
                    self.latest_gray,
                    (x, y),
                )

                print(
                    f"[SELECT] {self.name} "
                    f"HOLE = ({x}, {y})"
                )

            elif not self.tip.initialized:

                self.tip.initialize(
                    self.latest_gray,
                    (x, y),
                )

                print(
                    f"[SELECT] {self.name} "
                    f"TIP = ({x}, {y})"
                )

            else:
                print(
                    f"[INFO] {self.name}: "
                    "already selected. "
                    "Right-click to reset."
                )

        elif event == cv2.EVENT_RBUTTONDOWN:

            self.reset()

    def update(self, frame):
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        if self.prev_gray is not None:

            self.hole.update(
                self.prev_gray,
                gray,
            )

            self.tip.update(
                self.prev_gray,
                gray,
            )

        self.latest_gray = gray.copy()
        self.prev_gray = gray.copy()

    def target_locked(self):
        return (
            self.hole.initialized
            and self.tip.initialized
        )

    def valid(self):
        return (
            self.target_locked()
            and self.hole.valid
            and self.tip.valid
        )

    def error(self):
        if not self.valid():
            return None

        e = (
            self.hole.center
            - self.tip.center
        )

        du = float(e[0])
        dv = float(e[1])

        distance = math.hypot(
            du,
            dv,
        )

        return du, dv, distance


# ============================================================
# Drawing
# ============================================================

def draw_cross(
    image,
    point,
    size=10,
    thickness=2,
):
    x = int(round(point[0]))
    y = int(round(point[1]))

    cv2.line(
        image,
        (x - size, y),
        (x + size, y),
        (0, 255, 0),
        thickness,
    )

    cv2.line(
        image,
        (x, y - size),
        (x, y + size),
        (0, 255, 0),
        thickness,
    )


def draw_view(frame, state):
    display = frame.copy()

    # --------------------------------------------------------
    # Hole
    # --------------------------------------------------------

    if state.hole.initialized:

        center = state.hole.center

        color = (
            (0, 0, 255)
            if state.hole.valid
            else (0, 128, 255)
        )

        cv2.circle(
            display,
            (
                int(round(center[0])),
                int(round(center[1])),
            ),
            12,
            color,
            2,
        )

        cv2.putText(
            display,
            "HOLE",
            (
                int(center[0]) + 15,
                int(center[1]) - 10,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )

    # --------------------------------------------------------
    # Tip
    # --------------------------------------------------------

    if state.tip.initialized:

        center = state.tip.center

        draw_cross(
            display,
            center,
        )

        cv2.putText(
            display,
            "TIP",
            (
                int(center[0]) + 15,
                int(center[1]) + 20,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )

    # --------------------------------------------------------
    # Tip -> Hole
    # --------------------------------------------------------

    if state.valid():

        h = state.hole.center
        t = state.tip.center

        cv2.arrowedLine(
            display,
            (
                int(round(t[0])),
                int(round(t[1])),
            ),
            (
                int(round(h[0])),
                int(round(h[1])),
            ),
            (0, 255, 255),
            2,
            tipLength=0.12,
        )

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    y0 = 30

    cv2.putText(
        display,
        f"{state.name}  READ ONLY",
        (20, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    if not state.target_locked():

        cv2.putText(
            display,
            "LEFT CLICK: HOLE then TIP",
            (20, y0 + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

    elif not state.valid():

        cv2.putText(
            display,
            "TARGET LOST - RIGHT CLICK AND RESELECT",
            (20, y0 + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
        )

    else:

        du, dv, dist = state.error()

        text = (
            f"du={du:+.1f}px "
            f"dv={dv:+.1f}px "
            f"err={dist:.1f}px"
        )

        cv2.putText(
            display,
            text,
            (20, y0 + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            display,
            (
                f"H_pts={state.hole.num_points} "
                f"T_pts={state.tip.num_points}"
            ),
            (20, y0 + 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

    cv2.putText(
        display,
        "Right-click: reset this camera | Q: quit",
        (20, display.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )

    return display


# ============================================================
# Fixed camera
# ============================================================

def open_fixed_camera():

    cap = cv2.VideoCapture(
        FIXED_DEVICE,
        cv2.CAP_V4L2,
    )

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(
            *"MJPG"
        ),
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FIXED_WIDTH,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FIXED_HEIGHT,
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        FIXED_FPS,
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open fixed camera "
            f"{FIXED_DEVICE}"
        )

    return cap


# ============================================================
# Wrist camera
# ============================================================

def open_wrist_camera(hc):

    from pyorbbecsdk import (
        Config,
        OBFormat,
        OBSensorType,
        Pipeline,
    )

    context, device = (
        hc.find_orbbec_device(
            hc.RIGHT_WRIST_SN
        )
    )

    pipeline = Pipeline(
        device
    )

    config = Config()

    profiles = (
        pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        )
    )

    profile = (
        profiles.get_video_stream_profile(
            hc.COLOR_WIDTH,
            hc.COLOR_HEIGHT,
            OBFormat.RGB,
            hc.COLOR_FPS,
        )
    )

    config.enable_stream(
        profile
    )

    pipeline.start(
        config
    )

    print(
        "[CAMERA] Wrist RGB started: "
        f"{hc.COLOR_WIDTH}x"
        f"{hc.COLOR_HEIGHT}@"
        f"{hc.COLOR_FPS}"
    )

    # Keep context alive.
    return context, pipeline


# ============================================================
# Main
# ============================================================

def main():

    print()
    print(
        "============================================"
    )
    print(
        " R1-A7 Dual Camera Manual Guidance"
    )
    print(
        " READ ONLY - NO LowCmd - NO ROBOT MOTION"
    )
    print(
        "============================================"
    )
    print()

    print(
        "[CONTROL] Left click:"
        " HOLE first, TIP second"
    )

    print(
        "[CONTROL] Right click:"
        " reset selected camera"
    )

    print(
        "[CONTROL] Q: quit"
    )

    print()

    hc = load_collector()

    fixed_cap = None
    wrist_pipeline = None
    wrist_context = None

    fixed_state = ViewState(
        "FIXED"
    )

    wrist_state = ViewState(
        "WRIST"
    )

    fixed_window = (
        "R1-A7 FIXED CAMERA"
    )

    wrist_window = (
        "R1-A7 WRIST CAMERA"
    )

    last_print = 0.0

    try:

        fixed_cap = (
            open_fixed_camera()
        )

        (
            wrist_context,
            wrist_pipeline,
        ) = open_wrist_camera(hc)

        cv2.namedWindow(
            fixed_window,
            cv2.WINDOW_NORMAL,
        )

        cv2.namedWindow(
            wrist_window,
            cv2.WINDOW_NORMAL,
        )

        cv2.resizeWindow(
            fixed_window,
            960,
            540,
        )

        cv2.resizeWindow(
            wrist_window,
            960,
            600,
        )

        cv2.setMouseCallback(
            fixed_window,
            fixed_state.mouse_callback,
        )

        cv2.setMouseCallback(
            wrist_window,
            wrist_state.mouse_callback,
        )

        while True:

            # ----------------------------------------------
            # Fixed camera
            # ----------------------------------------------

            ok_fixed, fixed_frame = (
                fixed_cap.read()
            )

            if not ok_fixed:
                print(
                    "[ERROR] Fixed camera "
                    "frame read failed"
                )
                break

            # ----------------------------------------------
            # Wrist camera
            # ----------------------------------------------

            frames = (
                wrist_pipeline.wait_for_frames(
                    100
                )
            )

            if frames is None:
                continue

            color_frame = (
                frames.get_color_frame()
            )

            if color_frame is None:
                continue

            wrist_frame = (
                hc.color_frame_to_bgr(
                    color_frame
                )
            )

            # ----------------------------------------------
            # Track
            # ----------------------------------------------

            fixed_state.update(
                fixed_frame
            )

            wrist_state.update(
                wrist_frame
            )

            # ----------------------------------------------
            # Display
            # ----------------------------------------------

            fixed_display = draw_view(
                fixed_frame,
                fixed_state,
            )

            wrist_display = draw_view(
                wrist_frame,
                wrist_state,
            )

            # Overall state
            both_locked = (
                fixed_state.target_locked()
                and wrist_state.target_locked()
            )

            both_valid = (
                fixed_state.valid()
                and wrist_state.valid()
            )

            if both_locked:

                status = (
                    "DUAL TARGET VALID"
                    if both_valid
                    else "TARGET LOST"
                )

                color = (
                    (0, 255, 0)
                    if both_valid
                    else (0, 0, 255)
                )

                cv2.putText(
                    fixed_display,
                    status,
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    color,
                    2,
                )

                cv2.putText(
                    wrist_display,
                    status,
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    color,
                    2,
                )

            cv2.imshow(
                fixed_window,
                fixed_display,
            )

            cv2.imshow(
                wrist_window,
                wrist_display,
            )

            # ----------------------------------------------
            # Terminal report
            # ----------------------------------------------

            now = time.monotonic()

            if now - last_print >= 0.5:

                last_print = now

                if (
                    fixed_state.valid()
                    and wrist_state.valid()
                ):

                    fdu, fdv, ferr = (
                        fixed_state.error()
                    )

                    wdu, wdv, werr = (
                        wrist_state.error()
                    )

                    print(
                        "[GUIDANCE] "
                        f"FIXED "
                        f"du={fdu:+7.1f} "
                        f"dv={fdv:+7.1f} "
                        f"err={ferr:7.1f}px | "
                        f"WRIST "
                        f"du={wdu:+7.1f} "
                        f"dv={wdv:+7.1f} "
                        f"err={werr:7.1f}px"
                    )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):
                break

    finally:

        if fixed_cap is not None:
            fixed_cap.release()

        if wrist_pipeline is not None:
            try:
                wrist_pipeline.stop()
            except Exception:
                pass

        cv2.destroyAllWindows()

        print()
        print(
            "[DONE] Cameras closed."
        )
        print(
            "[DONE] No robot command "
            "was published."
        )


if __name__ == "__main__":
    main()
