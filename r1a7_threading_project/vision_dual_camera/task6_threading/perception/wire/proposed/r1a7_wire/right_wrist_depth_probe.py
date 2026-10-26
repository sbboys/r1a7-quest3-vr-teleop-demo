#!/usr/bin/env python3

import cv2
import numpy as np

from pyorbbecsdk import (
    Context,
    Pipeline,
    Config,
    AlignFilter,
    OBFormat,
    OBSensorType,
    OBStreamType,
    OBFrameAggregateOutputMode,
)

SERIAL = "CPCBC53000C5"


def find_device():

    ctx = Context()

    devices = ctx.query_devices()

    print("Orbbec count:", devices.get_count())

    for i in range(devices.get_count()):

        dev = devices.get_device_by_index(i)

        info = dev.get_device_info()

        sn = str(
            info.get_serial_number()
        )

        print(
            f"device {i}: "
            f"{info.get_name()} "
            f"SN={sn}"
        )

        if sn == SERIAL:
            return ctx, dev

    raise RuntimeError(
        f"Cannot find {SERIAL}"
    )


def color_to_bgr(frame):

    w = int(frame.get_width())
    h = int(frame.get_height())

    raw = np.frombuffer(
        frame.get_data(),
        dtype=np.uint8
    )

    if raw.size != w * h * 3:
        raise RuntimeError(
            f"Unexpected color bytes: {raw.size}"
        )

    rgb = raw.reshape(
        h,
        w,
        3
    )

    return cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )


def raw_depth(frame):

    w = int(frame.get_width())
    h = int(frame.get_height())

    raw = np.frombuffer(
        frame.get_data(),
        dtype=np.uint16
    )

    if raw.size != w * h:
        return None

    return raw.reshape(
        h,
        w
    )


def main():

    ctx, device = find_device()

    pipeline = Pipeline(
        device
    )

    config = Config()

    cp = pipeline.get_stream_profile_list(
        OBSensorType.COLOR_SENSOR
    )

    dp = pipeline.get_stream_profile_list(
        OBSensorType.DEPTH_SENSOR
    )

    color_profile = (
        cp.get_video_stream_profile(
            1280,
            800,
            OBFormat.RGB,
            30
        )
    )

    depth_profile = (
        dp.get_default_video_stream_profile()
    )

    print()
    print(
        "Color:",
        color_profile.get_width(),
        "x",
        color_profile.get_height(),
        "@",
        color_profile.get_fps(),
        color_profile.get_format()
    )

    print(
        "Depth:",
        depth_profile.get_width(),
        "x",
        depth_profile.get_height(),
        "@",
        depth_profile.get_fps(),
        depth_profile.get_format()
    )

    config.enable_stream(
        color_profile
    )

    config.enable_stream(
        depth_profile
    )

    try:
        config.set_frame_aggregate_output_mode(
            OBFrameAggregateOutputMode
            .FULL_FRAME_REQUIRE
        )
    except Exception as exc:
        print(
            "aggregate mode warning:",
            exc
        )

    try:
        pipeline.enable_frame_sync()
    except Exception as exc:
        print(
            "frame sync warning:",
            exc
        )

    pipeline.start(
        config
    )

    align = AlignFilter(
        align_to_stream=
        OBStreamType.COLOR_STREAM
    )

    clicked = {
        "uv": None
    }

    win = "RIGHT WRIST RGB-D PROBE"

    cv2.namedWindow(
        win,
        cv2.WINDOW_NORMAL
    )

    def mouse_cb(
        event,
        x,
        y,
        flags,
        param
    ):

        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["uv"] = (
                int(x),
                int(y)
            )

    cv2.setMouseCallback(
        win,
        mouse_cb
    )

    frame_id = 0

    try:

        while True:

            frames = pipeline.wait_for_frames(
                1000
            )

            if frames is None:
                continue

            aligned = align.process(
                frames
            )

            if aligned is not None:
                frames = aligned

            color = frames.get_color_frame()
            depth = frames.get_depth_frame()

            if color is None or depth is None:
                print(
                    "missing color/depth frame"
                )
                continue

            image = color_to_bgr(
                color
            )

            depth_raw = raw_depth(
                depth
            )

            if depth_raw is None:
                print(
                    "depth reshape failed"
                )
                continue

            scale = float(
                depth.get_depth_scale()
            )

            frame_id += 1

            if frame_id % 30 == 0:

                valid = depth_raw[
                    depth_raw > 0
                ]

                print()
                print(
                    "================================"
                )
                print(
                    "color size:",
                    image.shape[1],
                    "x",
                    image.shape[0]
                )

                print(
                    "depth size:",
                    depth_raw.shape[1],
                    "x",
                    depth_raw.shape[0]
                )

                print(
                    "depth scale:",
                    scale
                )

                print(
                    "valid depth pixels:",
                    valid.size,
                    "/",
                    depth_raw.size,
                    f"({100.0 * valid.size / depth_raw.size:.1f}%)"
                )

                if valid.size:

                    print(
                        "RAW min/median/max:",
                        int(np.min(valid)),
                        float(np.median(valid)),
                        int(np.max(valid))
                    )

                    scaled = (
                        valid.astype(np.float64)
                        *
                        scale
                    )

                    print(
                        "SCALED min/median/max:",
                        float(np.min(scaled)),
                        float(np.median(scaled)),
                        float(np.max(scaled))
                    )

            if clicked["uv"] is not None:

                u, v = clicked["uv"]

                clicked["uv"] = None

                h, w = (
                    depth_raw.shape
                )

                x0 = max(
                    0,
                    u - 20
                )

                x1 = min(
                    w,
                    u + 21
                )

                y0 = max(
                    0,
                    v - 20
                )

                y1 = min(
                    h,
                    v + 21
                )

                patch = depth_raw[
                    y0:y1,
                    x0:x1
                ]

                valid = patch[
                    patch > 0
                ]

                print()
                print(
                    "******** CLICK ********"
                )

                print(
                    "uv:",
                    u,
                    v
                )

                print(
                    "patch shape:",
                    patch.shape
                )

                print(
                    "patch valid:",
                    valid.size,
                    "/",
                    patch.size
                )

                if valid.size:

                    print(
                        "patch raw min/med/max:",
                        int(np.min(valid)),
                        float(np.median(valid)),
                        int(np.max(valid))
                    )

                    scaled = (
                        valid.astype(np.float64)
                        *
                        scale
                    )

                    print(
                        "patch scaled min/med/max:",
                        float(np.min(scaled)),
                        float(np.median(scaled)),
                        float(np.max(scaled))
                    )

                else:

                    print(
                        "PATCH HAS NO VALID DEPTH"
                    )

            # --------------------------------------------
            # Depth visualization
            # --------------------------------------------

            valid_mask = (
                depth_raw > 0
            )

            depth_vis = np.zeros_like(
                depth_raw,
                dtype=np.uint8
            )

            if np.any(valid_mask):

                vals = depth_raw[
                    valid_mask
                ]

                lo = np.percentile(
                    vals,
                    5
                )

                hi = np.percentile(
                    vals,
                    95
                )

                if hi > lo:

                    scaled_vis = (
                        255.0
                        *
                        (
                            depth_raw.astype(
                                np.float32
                            )
                            -
                            lo
                        )
                        /
                        (
                            hi
                            -
                            lo
                        )
                    )

                    depth_vis = np.clip(
                        scaled_vis,
                        0,
                        255
                    ).astype(
                        np.uint8
                    )

                    depth_vis[
                        ~valid_mask
                    ] = 0

            depth_color = (
                cv2.applyColorMap(
                    depth_vis,
                    cv2.COLORMAP_JET
                )
            )

            if (
                depth_color.shape[:2]
                ==
                image.shape[:2]
            ):

                display = np.hstack(
                    [
                        image,
                        depth_color
                    ]
                )

            else:

                display = image

            cv2.putText(
                display,
                (
                    f"depth={depth_raw.shape[1]}x"
                    f"{depth_raw.shape[0]} "
                    f"scale={scale:g}"
                ),
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                "LEFT CLICK tabletop near wire",
                (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                win,
                display
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

    finally:

        pipeline.stop()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
