#!/usr/bin/env python3

import sys
import cv2
import time
import numpy as np
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(
        ROOT /
        "vision_dual_camera" /
        "handeye"
    )
)

import collect_handeye_samples as hc


from pyorbbecsdk import (
    Config,
    OBFormat,
    OBSensorType,
    Pipeline,
)


hole_roi = None
tip_roi = None

select_state = 0

frame_now = None


# ============================================================
# Hole tracking memory
# ============================================================

hole_ref_radius = None

hole_last_center = None

hole_filtered_center = None


HOLE_MAX_MOVE_PIXEL = 15

HOLE_RADIUS_RATIO_MIN = 0.7

HOLE_RADIUS_RATIO_MAX = 1.3

HOLE_FILTER_ALPHA = 0.2


# ============================================================
# Tweezer tip tracking memory
# ============================================================

tip_axis_unit = None

tip_filtered_center = None

TIP_FILTER_ALPHA = 0.2


def mouse_callback(event,x,y,flags,param):

    global hole_roi
    global tip_roi
    global select_state


    if event != cv2.EVENT_LBUTTONDOWN:
        return


    size = 60


    if select_state == 0:

        hole_roi = (
            x-size,
            y-size,
            x+size,
            y+size
        )

        select_state = 1

        print(
            "[SELECT] HOLE ROI",
            hole_roi,
            flush=True
        )


    elif select_state == 1:

        tip_roi = (
            x-size,
            y-size,
            x+size,
            y+size
        )

        select_state = 2

        print(
            "[SELECT] TIP ROI",
            tip_roi,
            flush=True
        )





def detect_hole_circle(gray, roi):

    global hole_ref_radius
    global hole_last_center
    global hole_filtered_center

    if roi is None:
        return None

    image_h, image_w = gray.shape[:2]

    # ========================================================
    # Search region
    #
    # First frame:
    #   use the manually selected ROI.
    #
    # Later frames:
    #   move the same-size ROI together with the last
    #   detected hole center.
    #
    # This prevents the hole from being lost simply because
    # the wrist camera moves.
    # ========================================================

    roi_w = max(
        20,
        int(roi[2] - roi[0])
    )

    roi_h = max(
        20,
        int(roi[3] - roi[1])
    )

    half_w = roi_w // 2
    half_h = roi_h // 2

    if hole_last_center is None:

        x0 = int(roi[0])
        y0 = int(roi[1])
        x1 = int(roi[2])
        y1 = int(roi[3])

    else:

        cx_prev = int(
            hole_last_center[0]
        )

        cy_prev = int(
            hole_last_center[1]
        )

        x0 = cx_prev - half_w
        y0 = cy_prev - half_h

        x1 = cx_prev + half_w
        y1 = cy_prev + half_h

    # Clip ROI to image boundaries.
    x0 = max(
        0,
        min(
            x0,
            image_w - 1
        )
    )

    y0 = max(
        0,
        min(
            y0,
            image_h - 1
        )
    )

    x1 = max(
        x0 + 1,
        min(
            x1,
            image_w
        )
    )

    y1 = max(
        y0 + 1,
        min(
            y1,
            image_h
        )
    )

    crop = gray[
        y0:y1,
        x0:x1
    ]

    if (
        crop.shape[0] < 20
        or
        crop.shape[1] < 20
    ):
        return None

    blur = cv2.GaussianBlur(
        crop,
        (5, 5),
        0
    )

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=15,
        param1=80,
        param2=12,
        minRadius=5,
        maxRadius=40
    )

    if circles is None:
        return None

    circles = np.round(
        circles[0]
    ).astype(int)

    candidates = []

    for c in circles:

        local_x = int(c[0])
        local_y = int(c[1])
        radius = int(c[2])

        cx = int(
            local_x + x0
        )

        cy = int(
            local_y + y0
        )

        candidates.append(
            (
                cx,
                cy,
                radius
            )
        )

    if not candidates:
        return None

    # ========================================================
    # Initialization
    # ========================================================

    if hole_ref_radius is None:

        # At initialization choose the circle closest to
        # the CENTER of the manually selected ROI.

        manual_cx = (
            roi[0] + roi[2]
        ) / 2.0

        manual_cy = (
            roi[1] + roi[3]
        ) / 2.0

        best = min(
            candidates,
            key=lambda c:
            (
                (c[0] - manual_cx) ** 2
                +
                (c[1] - manual_cy) ** 2
            )
        )

        cx, cy, radius = best

        hole_ref_radius = float(
            radius
        )

        hole_last_center = (
            int(cx),
            int(cy)
        )

        hole_filtered_center = np.array(
            [
                float(cx),
                float(cy)
            ],
            dtype=float
        )

        print(
            "[HOLE INIT] "
            f"center=({cx},{cy}) "
            f"radius={radius}",
            flush=True
        )

        return (
            int(cx),
            int(cy),
            int(radius)
        )

    # ========================================================
    # Subsequent frames:
    #
    # 1. Radius must remain close to initial radius.
    # 2. Center must remain close to previous center.
    # 3. Among valid circles choose the one most consistent
    #    with BOTH position and radius.
    # ========================================================

    valid = []

    for cx, cy, radius in candidates:

        radius_min = (
            hole_ref_radius
            *
            HOLE_RADIUS_RATIO_MIN
        )

        radius_max = (
            hole_ref_radius
            *
            HOLE_RADIUS_RATIO_MAX
        )

        if not (
            radius_min
            <=
            radius
            <=
            radius_max
        ):
            continue

        dx = (
            cx
            -
            hole_last_center[0]
        )

        dy = (
            cy
            -
            hole_last_center[1]
        )

        move_px = float(
            np.hypot(
                dx,
                dy
            )
        )

        if (
            move_px
            >
            HOLE_MAX_MOVE_PIXEL
        ):
            continue

        radius_error = abs(
            float(radius)
            -
            hole_ref_radius
        )

        score = (
            move_px
            +
            radius_error
        )

        valid.append(
            (
                score,
                cx,
                cy,
                radius
            )
        )

    if not valid:

        # Do NOT invent a new circle and do NOT jump
        # to another object.
        return None

    valid.sort(
        key=lambda item:
        item[0]
    )

    _, cx_raw, cy_raw, radius = (
        valid[0]
    )

    measurement = np.array(
        [
            float(cx_raw),
            float(cy_raw)
        ],
        dtype=float
    )

    # ========================================================
    # EMA smoothing
    # ========================================================

    hole_filtered_center = (
        (1.0 - HOLE_FILTER_ALPHA)
        *
        hole_filtered_center
        +
        HOLE_FILTER_ALPHA
        *
        measurement
    )

    cx = int(
        round(
            hole_filtered_center[0]
        )
    )

    cy = int(
        round(
            hole_filtered_center[1]
        )
    )

    hole_last_center = (
        cx,
        cy
    )

    return (
        cx,
        cy,
        int(radius)
    )



def detect_tip_feature(gray, roi):

    global tip_axis_unit
    global tip_filtered_center

    if roi is None:
        return None

    image_h, image_w = gray.shape[:2]

    x0, y0, x1, y1 = roi

    x0 = max(
        0,
        min(
            int(x0),
            image_w - 1
        )
    )

    y0 = max(
        0,
        min(
            int(y0),
            image_h - 1
        )
    )

    x1 = max(
        x0 + 1,
        min(
            int(x1),
            image_w
        )
    )

    y1 = max(
        y0 + 1,
        min(
            int(y1),
            image_h
        )
    )

    crop = gray[
        y0:y1,
        x0:x1
    ]

    if (
        crop.shape[0] < 10
        or
        crop.shape[1] < 10
    ):
        return None

    # --------------------------------------------------------
    # Edge extraction
    # --------------------------------------------------------

    blur = cv2.GaussianBlur(
        crop,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blur,
        50,
        150
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return None

    # --------------------------------------------------------
    # Tweezer edge should normally be one of the longest
    # contours inside a tightly selected tip ROI.
    # --------------------------------------------------------

    contours = sorted(
        contours,
        key=lambda c:
        cv2.arcLength(
            c,
            False
        ),
        reverse=True
    )

    contour = None

    for c in contours:

        if len(c) < 8:
            continue

        length = cv2.arcLength(
            c,
            False
        )

        if length < 15:
            continue

        contour = c
        break

    if contour is None:
        return None

    pts = contour.reshape(
        -1,
        2
    ).astype(float)

    if len(pts) < 8:
        return None

    roi_cx = (
        crop.shape[1] - 1
    ) / 2.0

    roi_cy = (
        crop.shape[0] - 1
    ) / 2.0

    roi_center = np.array(
        [
            roi_cx,
            roi_cy
        ],
        dtype=float
    )

    # ========================================================
    # First frame:
    #
    # Determine WHICH end of the contour is the tweezer tip.
    # The direction is then locked for later frames.
    # ========================================================

    if tip_axis_unit is None:

        delta = (
            pts
            -
            roi_center
        )

        dist2 = np.sum(
            delta * delta,
            axis=1
        )

        idx = int(
            np.argmax(
                dist2
            )
        )

        tip_local = pts[idx]

        axis = (
            tip_local
            -
            roi_center
        )

        norm = float(
            np.linalg.norm(
                axis
            )
        )

        if norm < 1.0:
            return None

        tip_axis_unit = (
            axis
            /
            norm
        )

        tx_raw = float(
            tip_local[0]
            +
            x0
        )

        ty_raw = float(
            tip_local[1]
            +
            y0
        )

        tip_filtered_center = np.array(
            [
                tx_raw,
                ty_raw
            ],
            dtype=float
        )

        print(
            "[TIP INIT] "
            f"center=({int(round(tx_raw))},"
            f"{int(round(ty_raw))}) "
            f"axis=({tip_axis_unit[0]:+.3f},"
            f"{tip_axis_unit[1]:+.3f})",
            flush=True
        )

    else:

        # ----------------------------------------------------
        # Later frames:
        #
        # Project every contour point onto the INITIAL tip
        # direction and always choose the most forward point.
        #
        # This prevents switching between opposite contour
        # endpoints.
        # ----------------------------------------------------

        delta = (
            pts
            -
            roi_center
        )

        projection = (
            delta
            @
            tip_axis_unit
        )

        idx = int(
            np.argmax(
                projection
            )
        )

        tip_local = pts[idx]

        tx_raw = float(
            tip_local[0]
            +
            x0
        )

        ty_raw = float(
            tip_local[1]
            +
            y0
        )

        measurement = np.array(
            [
                tx_raw,
                ty_raw
            ],
            dtype=float
        )

        # ----------------------------------------------------
        # EMA only smooths image measurement.
        # It does not affect robot control.
        # ----------------------------------------------------

        tip_filtered_center = (
            (1.0 - TIP_FILTER_ALPHA)
            *
            tip_filtered_center
            +
            TIP_FILTER_ALPHA
            *
            measurement
        )

    tx = int(
        round(
            tip_filtered_center[0]
        )
    )

    ty = int(
        round(
            tip_filtered_center[1]
        )
    )

    return (
        tx,
        ty,
        len(pts)
    )


def main():

    global frame_now


    print(
        "================================="
    )

    print(
        "TASK6 WRIST GEOMETRY TRACKER"
    )

    print(
        "NO ROBOT CONTROL"
    )

    print(
        "================================="
    )


    print(
        "[CAMERA]",
        hc.RIGHT_WRIST_SN
    )


    context,device = (
        hc.find_orbbec_device(
            hc.RIGHT_WRIST_SN
        )
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


    profile = (
        profiles
        .get_video_stream_profile(
            hc.COLOR_WIDTH,
            hc.COLOR_HEIGHT,
            OBFormat.RGB,
            hc.COLOR_FPS
        )
    )


    config.enable_stream(
        profile
    )


    pipeline.start(
        config
    )


    print(
        "[CAMERA] RGB started:",
        f"{hc.COLOR_WIDTH}x"
        f"{hc.COLOR_HEIGHT}@"
        f"{hc.COLOR_FPS}",
        flush=True
    )


    print("")
    print("First click : hole ROI")
    print("Second click: tip ROI")
    print("Q quit")
    print("")


    window = "WRIST GEOMETRY"


    cv2.namedWindow(
        window
    )


    cv2.setMouseCallback(
        window,
        mouse_callback
    )


    last_print = 0.0


    while True:


        frames = (
            pipeline.wait_for_frames(
                1000
            )
        )


        if frames is None:
            continue


        color = (
            frames.get_color_frame()
        )


        if color is None:
            continue


        frame = (
            hc.color_frame_to_bgr(
                color
            )
        )


        frame_now = frame.copy()


        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # ============================================
        # Geometry detection
        # ============================================

        hole_result = detect_hole_circle(
            gray,
            hole_roi
        )

        tip_result = detect_tip_feature(
            gray,
            tip_roi
        )


        img = frame.copy()


        if hole_roi:

            x0,y0,x1,y1 = hole_roi

            cv2.rectangle(
                img,
                (x0,y0),
                (x1,y1),
                (0,255,0),
                2
            )


        if tip_roi:

            x0,y0,x1,y1 = tip_roi

            cv2.rectangle(
                img,
                (x0,y0),
                (x1,y1),
                (0,0,255),
                2
            )


        # ============================================
        # Display geometry result
        # ============================================

        if hole_result is not None:

            hx,hy,hr = hole_result

            cv2.drawMarker(
                img,
                (hx,hy),
                (0,255,0),
                cv2.MARKER_CROSS,
                30,
                2
            )

            cv2.circle(
                img,
                (hx,hy),
                hr,
                (0,255,0),
                2
            )


        if tip_result is not None:

            tx,ty,fc = tip_result

            cv2.drawMarker(
                img,
                (tx,ty),
                (0,0,255),
                cv2.MARKER_CROSS,
                30,
                2
            )


        if (
            hole_result is not None
            and
            tip_result is not None
        ):

            hx,hy,hr = hole_result

            tx,ty,fc = tip_result


            du = hx-tx
            dv = hy-ty


            err = (
                du*du
                +
                dv*dv
            )**0.5


            cv2.line(
                img,
                (tx,ty),
                (hx,hy),
                (255,255,0),
                2
            )


            cv2.putText(
                img,
                (
                    f"du={du:+.1f} "
                    f"dv={dv:+.1f} "
                    f"err={err:.1f}"
                ),
                (30,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,0),
                2
            )


            now=time.time()

            if now-last_print > 0.2:

                print(
                    "[WRIST GEO] "
                    f"H=({hx},{hy}) "
                    f"r={hr} "
                    f"T=({tx},{ty}) "
                    f"du={du:+.1f} "
                    f"dv={dv:+.1f} "
                    f"err={err:.1f}",
                    flush=True
                )

                last_print=now



        cv2.imshow(
            window,
            img
        )


        key = (
            cv2.waitKey(1)
            &
            0xff
        )


        if key == ord("q"):
            break


    pipeline.stop()

    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
