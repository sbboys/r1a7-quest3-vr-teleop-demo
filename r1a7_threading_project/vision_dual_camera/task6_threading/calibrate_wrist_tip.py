#!/usr/bin/env python3

import sys
import cv2
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "vision_dual_camera" / "handeye")
)

import collect_handeye_samples as hc

from pyorbbecsdk import (
    Config,
    OBFormat,
    OBSensorType,
    Pipeline,
)


tip_pixel = None
frame_now = None


def mouse_callback(event,x,y,flags,param):

    global tip_pixel

    if event == cv2.EVENT_LBUTTONDOWN:

        tip_pixel = (
            int(x),
            int(y)
        )

        print(
            "[TIP SELECT]",
            tip_pixel,
            flush=True
        )


def main():

    global frame_now


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


    window="CALIB TIP"


    cv2.namedWindow(
        window
    )

    cv2.setMouseCallback(
        window,
        mouse_callback
    )


    print("==============================")
    print("WRIST TIP CALIBRATION")
    print("==============================")
    print("Click fixed tweezer tip")
    print("S save")
    print("Q quit")


    while True:


        frames = pipeline.wait_for_frames(
            1000
        )


        if frames is None:
            continue


        color = frames.get_color_frame()


        if color is None:
            continue


        frame = hc.color_frame_to_bgr(
            color
        )


        frame_now = frame.copy()


        if tip_pixel:

            cv2.drawMarker(
                frame,
                tip_pixel,
                (0,0,255),
                cv2.MARKER_CROSS,
                30,
                2
            )


        cv2.imshow(
            window,
            frame
        )


        key=cv2.waitKey(1)&0xff


        if key==ord("s"):

            if tip_pixel:

                data={
                    "tip_pixel":
                        list(tip_pixel),

                    "camera":
                        hc.RIGHT_WRIST_SN,

                    "width":
                        hc.COLOR_WIDTH,

                    "height":
                        hc.COLOR_HEIGHT
                }


                Path(
                    "vision_dual_camera/task6_threading/"
                    "wrist_tip_calibration.json"
                ).write_text(
                    json.dumps(
                        data,
                        indent=2
                    )
                )


                print(
                    "[SAVE] wrist_tip_calibration.json",
                    flush=True
                )


        if key==ord("q"):
            break


    pipeline.stop()

    cv2.destroyAllWindows()


if __name__=="__main__":
    main()
