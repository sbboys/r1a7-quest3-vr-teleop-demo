#!/usr/bin/env python3

import sys
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HANDEYE_DIR = PROJECT_ROOT / "vision_dual_camera" / "handeye"

sys.path.insert(0, str(HANDEYE_DIR))

import collect_handeye_samples as hc

from pyorbbecsdk import (
    AlignFilter,
    Config,
    OBFormat,
    OBSensorType,
    OBStreamType,
    Pipeline,
)


WINDOW = "TASK6 Right Wrist Target Tracking"

# ------------------------------------------------------------
# Tracking state
# ------------------------------------------------------------

latest_image = None

target_template = None
target_center = None

# Half size of the template captured around the clicked hole.
TEMPLATE_HALF = 24

# Search only near the previous target position.
SEARCH_HALF = 110

# Template correlation required to update target.
MATCH_THRESHOLD = 0.60

last_print_time = 0.0
last_state = None



# ============================================================
# TASK6_RGBD_DEPTH_V1
# ============================================================

DEPTH_WIDTH = 848
DEPTH_HEIGHT = 480
DEPTH_FPS = 30

DEPTH_RING_INNER_PX = 12
DEPTH_RING_OUTER_PX = 30


def get_depth_scale_mm(depth_frame):
    """
    Return scale converting raw uint16 depth to millimetres.
    """

    for name in (
        "get_depth_scale",
        "get_value_scale",
    ):
        if hasattr(depth_frame, name):
            value = getattr(depth_frame, name)()

            if value is not None:
                return float(value)

    raise RuntimeError(
        "Depth frame has no supported depth scale method"
    )


def depth_frame_to_mm(depth_frame):
    """
    Convert aligned Y16 depth frame to float32 millimetres.
    """

    h = int(depth_frame.get_height())
    w = int(depth_frame.get_width())

    raw = np.frombuffer(
        depth_frame.get_data(),
        dtype=np.uint16,
    )

    expected = h * w

    if raw.size != expected:
        raise RuntimeError(
            f"Unexpected depth buffer: "
            f"{raw.size} != {expected}"
        )

    raw = raw.reshape(
        h,
        w,
    )

    scale = get_depth_scale_mm(
        depth_frame
    )

    return (
        raw.astype(np.float32)
        * scale
    )


def estimate_hole_plane_depth_mm(
    depth_mm,
    u,
    v,
):
    """
    Estimate the board surface depth around the hole.

    The centre of the hole itself is deliberately excluded,
    because depth through the hole may correspond to the
    background rather than the hole plane.
    """

    h, w = depth_mm.shape[:2]

    r = DEPTH_RING_OUTER_PX

    x0 = max(
        0,
        int(u - r),
    )

    x1 = min(
        w,
        int(u + r + 1),
    )

    y0 = max(
        0,
        int(v - r),
    )

    y1 = min(
        h,
        int(v + r + 1),
    )

    roi = depth_mm[
        y0:y1,
        x0:x1
    ]

    if roi.size == 0:
        return None, 0

    yy, xx = np.ogrid[
        y0:y1,
        x0:x1
    ]

    d2 = (
        (xx - u) ** 2
        + (yy - v) ** 2
    )

    mask = (
        (d2 >= DEPTH_RING_INNER_PX ** 2)
        &
        (d2 <= DEPTH_RING_OUTER_PX ** 2)
    )

    values = roi[
        mask
    ]

    valid = (
        np.isfinite(values)
        &
        (values > 100.0)
        &
        (values < 3000.0)
    )

    values = values[
        valid
    ]

    if values.size == 0:
        return None, 0

    return (
        float(np.median(values)),
        int(values.size),
    )


def color_pixel_depth_to_camera_mm(
    u,
    v,
    z_mm,
):
    """
    Back-project aligned Color pixel into the wrist Color
    camera coordinate system using the verified 1280x800
    Gemini color intrinsics.
    """

    fx = float(
        hc.CAMERA_MATRIX[0, 0]
    )

    fy = float(
        hc.CAMERA_MATRIX[1, 1]
    )

    cx = float(
        hc.CAMERA_MATRIX[0, 2]
    )

    cy = float(
        hc.CAMERA_MATRIX[1, 2]
    )

    x_mm = (
        (float(u) - cx)
        * z_mm
        / fx
    )

    y_mm = (
        (float(v) - cy)
        * z_mm
        / fy
    )

    return np.array(
        [
            x_mm,
            y_mm,
            z_mm,
        ],
        dtype=np.float64,
    )


def preprocess(gray):
    """
    Improve local contrast while retaining target appearance.
    """
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    return clahe.apply(gray)


def clear_target():
    global target_template
    global target_center

    target_template = None
    target_center = None

    print(
        "[TARGET] cleared",
        flush=True,
    )


def mouse_callback(
    event,
    x,
    y,
    flags,
    userdata,
):
    global latest_image
    global target_template
    global target_center

    if event == cv2.EVENT_RBUTTONDOWN:
        clear_target()
        return

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if latest_image is None:
        return

    h, w = latest_image.shape[:2]

    x = int(x)
    y = int(y)

    x0 = x - TEMPLATE_HALF
    x1 = x + TEMPLATE_HALF + 1
    y0 = y - TEMPLATE_HALF
    y1 = y + TEMPLATE_HALF + 1

    if (
        x0 < 0
        or y0 < 0
        or x1 > w
        or y1 > h
    ):
        print(
            "[TARGET] click too close to image border",
            flush=True,
        )
        return

    gray = cv2.cvtColor(
        latest_image,
        cv2.COLOR_BGR2GRAY,
    )

    gray = preprocess(gray)

    target_template = (
        gray[y0:y1, x0:x1].copy()
    )

    target_center = (
        x,
        y,
    )

    print()
    print(
        f"[TARGET] selected hole pixel = ({x}, {y})",
        flush=True,
    )

    print(
        f"[TARGET] template size = "
        f"{target_template.shape[1]}x"
        f"{target_template.shape[0]}",
        flush=True,
    )


def track_target(image):
    """
    Track only the manually selected target.

    No Hough circle search.
    No full-frame automatic target selection.
    """

    global target_center
    global target_template

    if (
        target_template is None
        or target_center is None
    ):
        return None

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gray = preprocess(gray)

    h, w = gray.shape[:2]

    cx, cy = target_center

    x0 = max(
        0,
        int(cx - SEARCH_HALF),
    )

    x1 = min(
        w,
        int(cx + SEARCH_HALF + 1),
    )

    y0 = max(
        0,
        int(cy - SEARCH_HALF),
    )

    y1 = min(
        h,
        int(cy + SEARCH_HALF + 1),
    )

    search = gray[
        y0:y1,
        x0:x1
    ]

    th, tw = target_template.shape[:2]

    if (
        search.shape[0] < th
        or search.shape[1] < tw
    ):
        return {
            "ok": False,
            "score": 0.0,
            "roi": (
                x0,
                y0,
                x1,
                y1,
            ),
        }

    response = cv2.matchTemplate(
        search,
        target_template,
        cv2.TM_CCOEFF_NORMED,
    )

    (
        _min_val,
        max_val,
        _min_loc,
        max_loc,
    ) = cv2.minMaxLoc(
        response
    )

    score = float(max_val)

    match_x = (
        x0
        + max_loc[0]
    )

    match_y = (
        y0
        + max_loc[1]
    )

    candidate_center = (
        int(
            match_x
            + tw // 2
        ),
        int(
            match_y
            + th // 2
        ),
    )

    result = {
        "ok":
            score >= MATCH_THRESHOLD,

        "score":
            score,

        "candidate":
            candidate_center,

        "box": (
            int(match_x),
            int(match_y),
            int(match_x + tw),
            int(match_y + th),
        ),

        "roi": (
            x0,
            y0,
            x1,
            y1,
        ),
    }

    # Only update the tracked target when the match is valid.
    if result["ok"]:
        target_center = (
            candidate_center[0],
            candidate_center[1],
        )

    return result


def main():
    global latest_image
    global last_print_time
    global last_state

    print(
        "============================================="
    )

    print(
        " TASK6 Right Wrist Target Tracking"
    )

    print(
        " MANUAL TARGET SEED + LOCAL IMAGE TRACKING"
    )

    print(
        " READ ONLY: NO ROBOT COMMAND"
    )

    print(
        "============================================="
    )

    context, device = (
        hc.find_orbbec_device(
            hc.RIGHT_WRIST_SN
        )
    )

    print(
        "[CAMERA] Right wrist camera:",
        hc.RIGHT_WRIST_SN,
    )

    pipeline = Pipeline(
        device
    )

    config = Config()

    profiles = (
        pipeline
        .get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        )
    )

    color_profile = (
        profiles
        .get_video_stream_profile(
            hc.COLOR_WIDTH,
            hc.COLOR_HEIGHT,
            OBFormat.RGB,
            hc.COLOR_FPS,
        )
    )

    config.enable_stream(
        color_profile
    )

    # --------------------------------------------------------
    # TASK6_RGBD_DEPTH_V1
    # Depth 848x480 @ 30 Hz, Y16.
    # --------------------------------------------------------

    depth_profiles = (
        pipeline
        .get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR
        )
    )

    depth_profile = (
        depth_profiles
        .get_video_stream_profile(
            DEPTH_WIDTH,
            DEPTH_HEIGHT,
            OBFormat.Y16,
            DEPTH_FPS,
        )
    )

    config.enable_stream(
        depth_profile
    )

    align_filter = AlignFilter(
        align_to_stream=
            OBStreamType.COLOR_STREAM
    )

    pipeline.start(
        config
    )

    print(
        "[CAMERA] RGB started:",
        f"{hc.COLOR_WIDTH}x"
        f"{hc.COLOR_HEIGHT}@"
        f"{hc.COLOR_FPS}",
    )

    print(
        "[CAMERA] Depth started:",
        f"{DEPTH_WIDTH}x"
        f"{DEPTH_HEIGHT}@"
        f"{DEPTH_FPS}",
        "Y16",
    )

    print(
        "[CAMERA] Software D2C alignment enabled.",
    )

    print()
    print("[CONTROL]")
    print(
        "Left click  = select the REAL target hole"
    )
    print(
        "Right click = clear target"
    )
    print(
        "C           = clear target"
    )
    print(
        "Q / ESC     = quit"
    )
    print()

    cv2.namedWindow(
        WINDOW,
        cv2.WINDOW_NORMAL,
    )

    cv2.setMouseCallback(
        WINDOW,
        mouse_callback,
    )

    try:
        while True:

            frames = (
                pipeline.wait_for_frames(
                    1000
                )
            )

            if frames is None:
                continue

            # --------------------------------------------------
            # TASK6_RGBD_DEPTH_V1
            # Project depth into the Color camera view.
            # --------------------------------------------------

            frames = align_filter.process(
                frames
            )

            if frames is None:
                continue

            color_frame = (
                frames.get_color_frame()
            )

            depth_frame = (
                frames.get_depth_frame()
            )

            if (
                color_frame is None
                or depth_frame is None
            ):
                continue

            image = (
                hc.color_frame_to_bgr(
                    color_frame
                )
            )

            try:
                depth_mm = (
                    depth_frame_to_mm(
                        depth_frame
                    )
                )
            except Exception as e:
                print(
                    "[DEPTH ERROR]",
                    e,
                    flush=True,
                )
                continue

            latest_image = image.copy()

            display = image.copy()

            h, w = display.shape[:2]

            image_center = (
                w // 2,
                h // 2,
            )

            # ----------------------------------------------
            # Camera optical center
            # ----------------------------------------------

            cv2.drawMarker(
                display,
                image_center,
                (255, 255, 0),
                cv2.MARKER_CROSS,
                30,
                2,
            )

            result = track_target(
                image
            )

            state = "UNSELECTED"

            if target_center is None:

                cv2.putText(
                    display,
                    "LEFT CLICK THE TARGET HOLE",
                    (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                )

            elif result is not None:

                # Search ROI
                (
                    rx0,
                    ry0,
                    rx1,
                    ry1,
                ) = result["roi"]

                cv2.rectangle(
                    display,
                    (rx0, ry0),
                    (rx1, ry1),
                    (120, 120, 120),
                    1,
                )

                if result["ok"]:

                    state = "LOCKED"

                    u, v = target_center

                    du = (
                        u
                        - image_center[0]
                    )

                    dv = (
                        v
                        - image_center[1]
                    )

                    # ------------------------------------------
                    # Aligned depth at the HOLE PLANE.
                    # Use the surrounding board, not the hole
                    # centre itself.
                    # ------------------------------------------

                    z_mm = None
                    camera_xyz_mm = None
                    depth_samples = 0

                    if (
                        depth_mm.shape[1]
                        == w
                        and
                        depth_mm.shape[0]
                        == h
                    ):
                        (
                            z_mm,
                            depth_samples,
                        ) = (
                            estimate_hole_plane_depth_mm(
                                depth_mm,
                                u,
                                v,
                            )
                        )

                        if z_mm is not None:
                            camera_xyz_mm = (
                                color_pixel_depth_to_camera_mm(
                                    u,
                                    v,
                                    z_mm,
                                )
                            )

                    (
                        bx0,
                        by0,
                        bx1,
                        by1,
                    ) = result["box"]

                    cv2.rectangle(
                        display,
                        (bx0, by0),
                        (bx1, by1),
                        (0, 255, 0),
                        2,
                    )

                    cv2.drawMarker(
                        display,
                        (u, v),
                        (0, 255, 0),
                        cv2.MARKER_CROSS,
                        28,
                        2,
                    )

                    cv2.putText(
                        display,
                        "TARGET HOLE: LOCKED",
                        (30, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                    )

                    text = (
                        f"u={u} v={v} "
                        f"du={du:+d} "
                        f"dv={dv:+d} "
                        f"score={result['score']:.3f}"
                    )

                    cv2.putText(
                        display,
                        text,
                        (30, 82),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                    )

                    if camera_xyz_mm is not None:

                        depth_text = (
                            f"Depth plane="
                            f"{z_mm:.1f} mm "
                            f"N={depth_samples}"
                        )

                        xyz_text = (
                            "Camera XYZ mm="
                            f"[{camera_xyz_mm[0]:.1f}, "
                            f"{camera_xyz_mm[1]:.1f}, "
                            f"{camera_xyz_mm[2]:.1f}]"
                        )

                        cv2.putText(
                            display,
                            depth_text,
                            (30, 117),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 255, 255),
                            2,
                        )

                        cv2.putText(
                            display,
                            xyz_text,
                            (30, 152),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 255, 255),
                            2,
                        )

                        cv2.circle(
                            display,
                            (u, v),
                            DEPTH_RING_INNER_PX,
                            (255, 0, 255),
                            1,
                        )

                        cv2.circle(
                            display,
                            (u, v),
                            DEPTH_RING_OUTER_PX,
                            (255, 0, 255),
                            1,
                        )

                    else:

                        cv2.putText(
                            display,
                            "Depth plane: INVALID",
                            (30, 117),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 0, 255),
                            2,
                        )

                    now = time.monotonic()

                    if (
                        now
                        - last_print_time
                        >= 0.25
                    ):
                        print(
                            "[TARGET] LOCKED "
                            f"u={u} "
                            f"v={v} "
                            f"du={du:+d} "
                            f"dv={dv:+d} "
                            f"score="
                            f"{result['score']:.3f}"
                            + (
                                (
                                    f" depth_mm={z_mm:.1f}"
                                    f" cameraXYZ_mm="
                                    f"[{camera_xyz_mm[0]:.1f},"
                                    f"{camera_xyz_mm[1]:.1f},"
                                    f"{camera_xyz_mm[2]:.1f}]"
                                )
                                if camera_xyz_mm is not None
                                else
                                " depth=INVALID"
                            ),
                            flush=True,
                        )

                        last_print_time = now

                else:

                    state = "LOST"

                    u, v = target_center

                    cv2.drawMarker(
                        display,
                        (u, v),
                        (0, 0, 255),
                        cv2.MARKER_CROSS,
                        28,
                        2,
                    )

                    cv2.putText(
                        display,
                        "TARGET HOLE: LOST",
                        (30, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2,
                    )

                    cv2.putText(
                        display,
                        (
                            "score="
                            f"{result['score']:.3f}"
                        ),
                        (30, 82),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

            if state != last_state:

                print(
                    f"[TRACK STATE] {state}",
                    flush=True,
                )

                last_state = state

            cv2.imshow(
                WINDOW,
                display,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key in (
                ord("q"),
                27,
            ):
                break

            if key == ord("c"):
                clear_target()

    finally:

        try:
            pipeline.stop()
        except Exception:
            pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
