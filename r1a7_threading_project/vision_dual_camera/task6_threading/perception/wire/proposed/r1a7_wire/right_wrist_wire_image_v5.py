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

from wire_local_detector_v5 import (
    WireLocalDetectorV5,
)


SERIAL = "CPCBC53000C5"

OUTPUT = Path(
    "/tmp/r1a7_right_wrist_wire_v5.json"
)


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
            dev.get_device_info()
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


def frame_to_bgr(frame):

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


def click_tip(
    frame,
    roi
):

    x, y, w, h = [
        int(v)
        for v in roi
    ]

    point = {
        "uv": None
    }

    win = "CLICK WIRE FREE TIP"

    cv2.namedWindow(
        win,
        cv2.WINDOW_NORMAL
    )

    def cb(
        event,
        px,
        py,
        flags,
        param
    ):

        if (
            event
            ==
            cv2.EVENT_LBUTTONDOWN
        ):

            point["uv"] = (
                int(px),
                int(py)
            )

    cv2.setMouseCallback(
        win,
        cb
    )

    while True:

        vis = frame.copy()

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

        if point["uv"] is not None:

            cv2.circle(
                vis,
                point["uv"],
                8,
                (0, 0, 255),
                -1
            )

        cv2.putText(
            vis,
            "LEFT: wire free tip | ENTER confirm",
            (20, 35),
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
            point["uv"] is not None
        ):

            cv2.destroyWindow(
                win
            )

            return point["uv"]

        if key == 27:

            cv2.destroyWindow(
                win
            )

            return None


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
        WireLocalDetectorV5()
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

        # ----------------------------------------------------
        # Select SEARCH AREA.
        #
        # This is not a wire coordinate.
        # It defines the right-wrist grasp workspace.
        # ----------------------------------------------------

        print(
            "[INIT] Select wire SEARCH ROI."
        )

        roi = cv2.selectROI(
            "SELECT WIRE SEARCH ROI",
            frame,
            False,
            False
        )

        cv2.destroyWindow(
            "SELECT WIRE SEARCH ROI"
        )

        if (
            roi[2] <= 0
            or
            roi[3] <= 0
        ):
            return

        # Development only:
        # one click labels which endpoint is the free tip.
        seed = click_tip(
            frame,
            roi
        )

        if seed is None:
            return

        detector.set_seed_tip(
            seed
        )

        print(
            "[INIT] ROI  =",
            tuple(
                int(v)
                for v in roi
            )
        )

        print(
            "[INIT] TIP  =",
            seed
        )

        print()
        print(
            "V5 IMAGE-ONLY DETECTOR"
        )

        print(
            "R = relabel free tip"
        )

        print(
            "B = redefine search ROI"
        )

        print(
            "Q = quit"
        )

        win = (
            "RIGHT WRIST WIRE IMAGE V5"
        )

        cv2.namedWindow(
            win,
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

            result = detector.detect(
                frame,
                roi
            )

            vis = frame.copy()

            x, y, w, h = [
                int(v)
                for v in roi
            ]

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

            if result["valid"]:

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

            state = (
                result["state"]
            )

            cv2.putText(
                vis,
                (
                    f"{state} "
                    f"score={result.get('score',0):.2f} "
                    f"N={result.get('candidate_count',0)}"
                ),
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (
                    (0, 255, 0)
                    if result["valid"]
                    else
                    (0, 165, 255)
                ),
                2
            )

            output = {
                "enabled":
                    False,

                "mode":
                    "IMAGE_ONLY_V5",

                "camera_serial":
                    SERIAL,

                "timestamp":
                    time.time(),

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
                    "[V5]"
                    f" state={state}"
                    f" valid={result['valid']}"
                    f" score={result.get('score',0):.2f}"
                    f" ridge={result.get('ridge_score',0):.2f}"
                    f" thickness={result.get('thickness',0):.2f}"
                    f" tip={result.get('wire_tip_uv')}",
                    flush=True
                )

                last_print = now

            cv2.imshow(
                win,
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
                    frame,
                    roi
                )

                if seed is not None:

                    detector = (
                        WireLocalDetectorV5()
                    )

                    detector.set_seed_tip(
                        seed
                    )

            if key in (
                ord("b"),
                ord("B")
            ):

                new_roi = cv2.selectROI(
                    "SELECT WIRE SEARCH ROI",
                    frame,
                    False,
                    False
                )

                cv2.destroyWindow(
                    "SELECT WIRE SEARCH ROI"
                )

                if (
                    new_roi[2] > 0
                    and
                    new_roi[3] > 0
                ):

                    roi = new_roi

                    seed = click_tip(
                        frame,
                        roi
                    )

                    if seed is not None:

                        detector = (
                            WireLocalDetectorV5()
                        )

                        detector.set_seed_tip(
                            seed
                        )

    finally:

        pipeline.stop()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
