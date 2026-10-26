#!/usr/bin/env python3

import json
import time
from pathlib import Path

import cv2
import numpy as np

from pyorbbecsdk import (
    Context,
    Pipeline,
    Config,
    OBSensorType,
    OBFormat,
)

from wire_local_detector_v61 import (
    WireLocalDetectorV61,
)


SERIAL = "CPCBC53000C5"

OUTPUT = Path(
    "/tmp/r1a7_right_wrist_wire_v61.json"
)


# ============================================================
# Camera
# ============================================================

def find_device():

    ctx = Context()

    devices = ctx.query_devices()

    for i in range(
        devices.get_count()
    ):

        dev = (
            devices.get_device_by_index(
                i
            )
        )

        sn = str(
            dev
            .get_device_info()
            .get_serial_number()
        )

        print(
            f"[CAMERA] #{i} SN={sn}"
        )

        if sn == SERIAL:

            print(
                "[CAMERA] RIGHT WRIST LOCKED:",
                sn
            )

            return ctx, dev

    raise RuntimeError(
        f"Cannot find {SERIAL}"
    )


def frame_to_bgr(
    frame
):

    w = int(
        frame.get_width()
    )

    h = int(
        frame.get_height()
    )

    raw = np.frombuffer(
        frame.get_data(),
        dtype=np.uint8
    )

    if raw.size != w * h * 3:
        return None

    rgb = raw.reshape(
        h,
        w,
        3
    )

    return cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )


# ============================================================
# Click initial free tip
# ============================================================

def click_tip(
    frame
):

    point = {
        "uv": None
    }

    win = (
        "CLICK REAL WIRE FREE TIP"
    )

    cv2.namedWindow(
        win,
        cv2.WINDOW_NORMAL
    )

    def callback(
        event,
        x,
        y,
        flags,
        param
    ):

        if (
            event
            ==
            cv2.EVENT_LBUTTONDOWN
        ):

            point[
                "uv"
            ] = (
                int(x),
                int(y)
            )

    cv2.setMouseCallback(
        win,
        callback
    )

    while True:

        vis = frame.copy()

        if (
            point["uv"]
            is not None
        ):

            cv2.circle(
                vis,
                point["uv"],
                9,
                (0, 0, 255),
                -1
            )

        cv2.putText(
            vis,
            "LEFT: REAL wire free tip",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            vis,
            "ENTER: confirm",
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            win,
            vis
        )

        key = (
            cv2.waitKey(20)
            &
            0xFF
        )

        if (
            key in (10, 13)
            and
            point["uv"]
            is not None
        ):

            cv2.destroyWindow(
                win
            )

            return point[
                "uv"
            ]

        if key == 27:

            cv2.destroyWindow(
                win
            )

            return None


# ============================================================
# Dynamic ROI
# ============================================================

def clip_roi(
    roi,
    shape
):

    H, W = shape[:2]

    x, y, w, h = [
        int(round(v))
        for v in roi
    ]

    x = max(
        0,
        min(
            x,
            W - 1
        )
    )

    y = max(
        0,
        min(
            y,
            H - 1
        )
    )

    x2 = max(
        x + 1,
        min(
            x + w,
            W
        )
    )

    y2 = max(
        y + 1,
        min(
            y + h,
            H
        )
    )

    return (
        x,
        y,
        x2 - x,
        y2 - y
    )


def roi_from_tip(
    tip,
    shape,
    half_w=130,
    half_h=130,
):

    u, v = tip

    return clip_roi(
        (
            u - half_w,
            v - half_h,
            2 * half_w,
            2 * half_h
        ),
        shape
    )


def roi_from_path(
    path,
    shape,
    margin=85,
    min_w=280,
    min_h=280,
):

    pts = np.asarray(
        path,
        dtype=np.float64
    )

    xmin = float(
        np.min(
            pts[:, 0]
        )
    )

    xmax = float(
        np.max(
            pts[:, 0]
        )
    )

    ymin = float(
        np.min(
            pts[:, 1]
        )
    )

    ymax = float(
        np.max(
            pts[:, 1]
        )
    )

    cx = 0.5 * (
        xmin + xmax
    )

    cy = 0.5 * (
        ymin + ymax
    )

    w = max(
        min_w,
        (
            xmax - xmin
        )
        +
        2 * margin
    )

    h = max(
        min_h,
        (
            ymax - ymin
        )
        +
        2 * margin
    )

    return clip_roi(
        (
            cx - w / 2,
            cy - h / 2,
            w,
            h
        ),
        shape
    )


def expand_roi(
    roi,
    shape,
    extra
):

    x, y, w, h = roi

    cx = x + w / 2
    cy = y + h / 2

    return clip_roi(
        (
            cx - w / 2 - extra,
            cy - h / 2 - extra,
            w + 2 * extra,
            h + 2 * extra
        ),
        shape
    )


# ============================================================
# Main
# ============================================================

def main():

    ctx, device = (
        find_device()
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
            1280,
            800,
            OBFormat.RGB,
            30
        )
    )

    config.enable_stream(
        profile
    )

    pipeline.start(
        config
    )

    detector = (
        WireLocalDetectorV61()
    )

    try:

        frame = None

        while frame is None:

            frames = (
                pipeline
                .wait_for_frames(
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
                frame_to_bgr(
                    color
                )
            )

        seed = click_tip(
            frame
        )

        if seed is None:
            return

        detector.set_seed_tip(
            seed
        )

        active_roi = (
            roi_from_tip(
                seed,
                frame.shape
            )
        )

        miss_count = 0
        anchor_roi = active_roi

        print()
        print(
            "[V6] IMAGE-DRIVEN DYNAMIC ROI"
        )

        print(
            "[V6] Initial tip:",
            seed
        )

        print(
            "R = relabel tip"
        )

        print(
            "Q = quit"
        )

        window = (
            "RIGHT WRIST WIRE IMAGE V6"
        )

        cv2.namedWindow(
            window,
            cv2.WINDOW_NORMAL
        )

        last_print = 0.0

        while True:

            frames = (
                pipeline
                .wait_for_frames(
                    100
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
                frame_to_bgr(
                    color
                )
            )

            if frame is None:
                continue

            # --------------------------------------------
            # CURRENT IMAGE detection
            # --------------------------------------------

            result = detector.detect(
                frame,
                active_roi
            )

            # --------------------------------------------
            # Dynamic ROI:
            #
            # valid image detection -> FOLLOW current line
            #
            # invalid -> EXPAND search area
            #
            # no FK, no Depth, no geometric prediction.
            # --------------------------------------------

            if (
                result["valid"]
                and
                len(
                    result[
                        "centerline"
                    ]
                )
                >=
                2
            ):

                miss_count = 0

                active_roi = (
                    roi_from_path(
                        result[
                            "centerline"
                        ],
                        frame.shape,
                        margin=65
                    )
                )
                anchor_roi = active_roi


            else:

                miss_count += 1

                expansion = min(
                    20
                    *
                    miss_count,
                    70
                )

                active_roi = (
                    expand_roi(
                        anchor_roi,
                        frame.shape,
                        expansion
                    )
                )

            # --------------------------------------------
            # Draw
            # --------------------------------------------

            vis = frame.copy()

            x, y, w, h = (
                active_roi
            )

            cv2.rectangle(
                vis,
                (
                    x,
                    y
                ),
                (
                    x + w,
                    y + h
                ),
                (255, 255, 0),
                2
            )

            cv2.putText(
                vis,
                "ACTIVE SEARCH ROI",
                (
                    x + 5,
                    max(
                        20,
                        y - 8
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 0),
                2
            )

            if result[
                "valid"
            ]:

                pts = np.asarray(
                    result[
                        "centerline"
                    ],
                    dtype=np.int32
                )

                if len(pts) >= 2:

                    cv2.polylines(
                        vis,
                        [
                            pts.reshape(
                                -1,
                                1,
                                2
                            )
                        ],
                        False,
                        (0, 255, 0),
                        3,
                        cv2.LINE_AA
                    )

                tip = result[
                    "wire_tip_uv"
                ]

                target = result[
                    "grasp_target_uv"
                ]

                if tip is not None:

                    cv2.circle(
                        vis,
                        tuple(tip),
                        8,
                        (0, 0, 255),
                        -1
                    )

                    cv2.putText(
                        vis,
                        "WIRE TIP",
                        (
                            tip[0] + 10,
                            tip[1] - 10
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2
                    )

                if target is not None:

                    cv2.circle(
                        vis,
                        tuple(target),
                        8,
                        (0, 255, 255),
                        -1
                    )

                    cv2.putText(
                        vis,
                        "GRASP",
                        (
                            target[0] + 10,
                            target[1] + 20
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2
                    )

            state = result[
                "state"
            ]

            cv2.putText(
                vis,
                (
                    f"{state} "
                    f"score={result.get('score',0):.2f} "
                    f"profile={result.get('profile_score',0):.2f}"
                ),
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (
                    (0, 255, 0)
                    if result["valid"]
                    else
                    (0, 165, 255)
                ),
                2
            )

            cv2.putText(
                vis,
                (
                    "two-side="
                    f"{result.get('two_sided_fraction',0):.2f} "
                    "edge-overlap="
                    f"{result.get('straight_edge_overlap',0):.2f} "
                    f"miss={miss_count}"
                ),
                (20, 67),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            output = {
                "enabled":
                    False,

                "mode":
                    "IMAGE_ONLY_V61_DYNAMIC_ROI",

                "camera_serial":
                    SERIAL,

                "timestamp":
                    time.time(),

                "active_roi":
                    list(
                        active_roi
                    ),

                "miss_count":
                    int(
                        miss_count
                    ),

                **result
            }

            OUTPUT.write_text(
                json.dumps(
                    output,
                    indent=2
                )
            )

            now = time.time()

            if (
                now -
                last_print
                >
                1.0
            ):

                print(
                    "[V6]"
                    f" state={state}"
                    f" valid={result['valid']}"
                    f" score={result.get('score',0):.2f}"
                    f" ridge={result.get('ridge_score',0):.2f}"
                    f" profile={result.get('profile_score',0):.2f}"
                    f" two={result.get('two_sided_fraction',0):.2f}"
                    f" edge={result.get('straight_edge_overlap',0):.2f}"
                    f" tip={result.get('wire_tip_uv')}"
                    f" miss={miss_count}",
                    flush=True
                )

                last_print = now

            cv2.imshow(
                window,
                vis
            )

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            if key in (
                ord("q"),
                ord("Q"),
                27
            ):

                break

            if key in (
                ord("r"),
                ord("R")
            ):

                seed = click_tip(
                    frame
                )

                if seed is not None:

                    detector = (
                        WireLocalDetectorV61()
                    )

                    detector.set_seed_tip(
                        seed
                    )

                    active_roi = (
                        roi_from_tip(
                            seed,
                            frame.shape
                        )
                    )
                    anchor_roi = active_roi


                    miss_count = 0

    finally:

        pipeline.stop()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
