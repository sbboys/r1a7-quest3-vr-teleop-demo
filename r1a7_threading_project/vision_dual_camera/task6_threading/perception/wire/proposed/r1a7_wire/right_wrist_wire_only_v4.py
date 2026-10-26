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

from wire_ego_tracker_v4 import (
    EgoMotionWireTrackerV4,
    grasp_target_from_centerline,
)


RIGHT_WRIST_SERIAL = "CPCBC53000C5"

OUTPUT_FILE = Path(
    "/tmp/r1a7_right_wrist_wire_v4.json"
)


def find_device(serial):

    ctx = Context()

    devices = ctx.query_devices()

    print(
        "[CAMERA] count:",
        devices.get_count()
    )

    for i in range(
        devices.get_count()
    ):

        device = (
            devices.get_device_by_index(
                i
            )
        )

        info = (
            device.get_device_info()
        )

        sn = (
            info.get_serial_number()
        )

        print(
            f"[CAMERA] #{i} SN={sn}"
        )

        if sn == serial:

            print(
                "[CAMERA] RIGHT wrist locked:",
                sn
            )

            return ctx, device

    raise RuntimeError(
        f"Cannot find Orbbec {serial}"
    )


def frame_to_bgr(frame):

    if frame is None:
        return None

    w = frame.get_width()
    h = frame.get_height()

    data = np.frombuffer(
        frame.get_data(),
        dtype=np.uint8
    )

    if data.size != (
        w * h * 3
    ):
        return None

    rgb = data.reshape(
        h,
        w,
        3
    )

    return cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )


def select_wire_polyline(frame):

    points = []

    win = (
        "WIRE V4 INIT - CLICK FROM FREE TIP INWARD"
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

            points.append(
                (
                    int(x),
                    int(y)
                )
            )

        elif (
            event
            ==
            cv2.EVENT_RBUTTONDOWN
        ):

            if points:
                points.pop()

    cv2.setMouseCallback(
        win,
        callback
    )

    while True:

        vis = frame.copy()

        if len(points) >= 2:

            cv2.polylines(
                vis,
                [
                    np.asarray(
                        points,
                        dtype=np.int32
                    )
                ],
                False,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

        for i, p in enumerate(
            points
        ):

            cv2.circle(
                vis,
                p,
                6,
                (
                    (0, 0, 255)
                    if i == 0
                    else
                    (0, 255, 255)
                ),
                -1
            )

            cv2.putText(
                vis,
                str(
                    i + 1
                ),
                (
                    p[0] + 7,
                    p[1] - 7
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        cv2.putText(
            vis,
            "Click 4-12 points FROM FREE TIP inward",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            vis,
            "ENTER confirm | RIGHT undo | ESC cancel",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
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
            len(points) >= 4
        ):

            cv2.destroyWindow(
                win
            )

            return points

        if key == 27:

            cv2.destroyWindow(
                win
            )

            return None


def main():

    ctx, device = find_device(
        RIGHT_WRIST_SERIAL
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

    tracker = None

    try:

        init_frame = None

        while init_frame is None:

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

            init_frame = (
                frame_to_bgr(
                    color
                )
            )

        control_points = (
            select_wire_polyline(
                init_frame
            )
        )

        if control_points is None:
            return

        tracker = EgoMotionWireTrackerV4()

        tracker.initialize(
            init_frame,
            control_points
        )

        win = (
            "R1-A7 RIGHT WRIST WIRE V4"
        )

        cv2.namedWindow(
            win,
            cv2.WINDOW_NORMAL
        )

        print(
            "[WIRE V4] running"
        )

        print(
            "[WIRE V4] R = reinitialize"
        )

        print(
            "[WIRE V4] Q/ESC = quit"
        )

        last_print = 0.0

        while True:

            frames = (
                pipeline.wait_for_frames(
                    100
                )
            )

            if frames is None:
                continue

            color = (
                frames.get_color_frame()
            )

            frame = (
                frame_to_bgr(
                    color
                )
            )

            if frame is None:
                continue

            result = tracker.update(
                frame
            )

            vis = frame.copy()

            pts = np.asarray(
                result["centerline"],
                dtype=np.int32
            )

            line_color = (
                (0, 255, 0)
                if result["valid"]
                else
                (120, 120, 120)
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
                    line_color,
                    2,
                    cv2.LINE_AA
                )

            tip = result["tip"]

            if tip is not None:

                cv2.circle(
                    vis,
                    tuple(tip),
                    7,
                    (0, 0, 255),
                    -1
                )

            target = None

            if result["valid"]:
                target = grasp_target_from_centerline(
                    result["centerline"],
                    30.0
                )

            if target is not None:

                gt = (
                    int(round(
                        target[0]
                    )),
                    int(round(
                        target[1]
                    ))
                )

                cv2.circle(
                    vis,
                    gt,
                    7,
                    (0, 255, 255),
                    -1
                )

            state_text = (
                f"{result['state']} "
                f"conf={result['confidence']:.2f} "
                f"flow={result['flow_fraction']:.2f} "
                f"ridge={result['ridge_score']:.2f}"
            )

            cv2.putText(
                vis,
                state_text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (
                    (0, 255, 0)
                    if result["valid"]
                    else
                    (120, 120, 120)
                ),
                2,
                cv2.LINE_AA
            )

            if result[
                "needs_reinit"
            ]:

                cv2.putText(
                    vis,
                    "TRACK LOST - PRESS R",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

            output = {
                "enabled":
                    False,

                "camera_serial":
                    RIGHT_WRIST_SERIAL,

                "timestamp":
                    time.time(),

                **result,

                "grasp_target_uv":
                    (
                        [
                            int(round(
                                target[0]
                            )),
                            int(round(
                                target[1]
                            ))
                        ]
                        if target is not None
                        else None
                    )
            }

            OUTPUT_FILE.write_text(
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
                    "[WIRE V4]"
                    f" state={result['state']}"
                    f" valid={result['valid']}"
                    f" conf={result['confidence']:.2f}"
                    f" flow={result['flow_fraction']:.2f}"
                    f" ridge={result['ridge_score']:.2f}"
                    f" tip={result['tip']}",
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
                27
            ):
                break

            if key in (
                ord("r"),
                ord("R")
            ):

                new_points = (
                    select_wire_polyline(
                        frame
                    )
                )

                if (
                    new_points
                    is not None
                ):

                    tracker = (
                        EgoMotionWireTrackerV4()
                    )

                    tracker.initialize(
                        frame,
                        new_points
                    )

    finally:

        pipeline.stop()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
