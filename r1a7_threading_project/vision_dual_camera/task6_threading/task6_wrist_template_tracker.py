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


hole_template = None
tip_template = None

hole_center = None
tip_center = None

select_mode = "hole"


frame_now = None


# 模板尺寸
HOLE_SIZE = 50
TIP_SIZE = 40



def mouse_callback(event,x,y,flags,param):

    global hole_template
    global tip_template

    global hole_center
    global tip_center

    global select_mode
    global frame_now


    if event != cv2.EVENT_LBUTTONDOWN:
        return


    gray=cv2.cvtColor(
        frame_now,
        cv2.COLOR_BGR2GRAY
    )


    if select_mode=="hole":

        h=HOLE_SIZE

        hole_template = gray[
            y-h:y+h,
            x-h:x+h
        ].copy()


        hole_center=(
            x,
            y
        )

        select_mode="tip"

        print(
            "[SELECT] HOLE",
            hole_center,
            flush=True
        )


    elif select_mode=="tip":

        h=TIP_SIZE

        tip_template = gray[
            y-h:y+h,
            x-h:x+h
        ].copy()


        tip_center=(
            x,
            y
        )

        select_mode="done"


        print(
            "[SELECT] TIP",
            tip_center,
            flush=True
        )



def template_track(
    gray,
    template,
    center,
    search=80
):

    if template is None:
        return None


    x,y=center


    x0=max(
        0,
        x-search
    )

    y0=max(
        0,
        y-search
    )

    x1=min(
        gray.shape[1],
        x+search
    )

    y1=min(
        gray.shape[0],
        y+search
    )


    roi=gray[
        y0:y1,
        x0:x1
    ]


    result=cv2.matchTemplate(
        roi,
        template,
        cv2.TM_CCOEFF_NORMED
    )


    _,score,_,loc=cv2.minMaxLoc(
        result
    )


    if score < 0.5:

        return None


    th,tw=template.shape


    nx=x0+loc[0]+tw//2
    ny=y0+loc[1]+th//2


    return (
        int(nx),
        int(ny),
        float(score)
    )



def main():

    global frame_now
    global hole_center
    global tip_center
    global hole_template
    global tip_template
    global select_mode


    print(
        "=============================="
    )

    print(
        "TASK6 WRIST TEMPLATE TRACKER"
    )

    print(
        "NO ROBOT CONTROL"
    )

    print(
        "=============================="
    )


    context,device=(
        hc.find_orbbec_device(
            hc.RIGHT_WRIST_SN
        )
    )


    pipeline=Pipeline(
        device
    )


    config=Config()


    profiles=(
        pipeline
        .get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        )
    )


    profile=(
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
        "[WRIST CAMERA] started",
        flush=True
    )

    print(
        "Click hole first"
    )

    print(
        "Click tip second"
    )


    cv2.namedWindow(
        "WRIST TEMPLATE"
    )


    cv2.setMouseCallback(
        "WRIST TEMPLATE",
        mouse_callback
    )


    last_print=0


    while True:


        frames=(
            pipeline.wait_for_frames(
                1000
            )
        )


        if frames is None:
            continue


        color=(
            frames.get_color_frame()
        )


        if color is None:
            continue


        frame=(
            hc.color_frame_to_bgr(
                color
            )
        )


        frame_now=frame.copy()


        gray=cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # hole tracking

        if hole_template is not None:

            r=template_track(
                gray,
                hole_template,
                hole_center
            )

            if r:

                hole_center=(
                    r[0],
                    r[1]
                )


        # tip tracking

        if tip_template is not None:

            r=template_track(
                gray,
                tip_template,
                tip_center
            )

            if r:

                tip_center=(
                    r[0],
                    r[1]
                )


        img=frame.copy()


        if hole_center:

            cv2.drawMarker(
                img,
                hole_center,
                (0,255,0),
                cv2.MARKER_CROSS,
                30,
                2
            )


        if tip_center:

            cv2.drawMarker(
                img,
                tip_center,
                (0,0,255),
                cv2.MARKER_CROSS,
                30,
                2
            )


        if (
            hole_center
            and
            tip_center
        ):

            du=(
                hole_center[0]
                -
                tip_center[0]
            )

            dv=(
                hole_center[1]
                -
                tip_center[1]
            )


            err=np.sqrt(
                du*du+dv*dv
            )


            cv2.putText(
                img,
                f"du={du:+.1f} dv={dv:+.1f}",
                (30,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,0),
                2
            )


            now=time.time()

            if now-last_print>0.2:

                print(
                    "[WRIST TEMPLATE] "
                    f"H={hole_center} "
                    f"T={tip_center} "
                    f"du={du:+.1f} "
                    f"dv={dv:+.1f} "
                    f"err={err:.1f}",
                    flush=True
                )

                last_print=now


        cv2.imshow(
            "WRIST TEMPLATE",
            img
        )


        key=cv2.waitKey(1)&0xff


        if key==ord("q"):
            break


        if key==ord("c"):

            hole_template=None
            tip_template=None

            hole_center=None
            tip_center=None

            select_mode="hole"

            print(
                "[CLEAR]"
            )


    pipeline.stop()

    cv2.destroyAllWindows()



if __name__=="__main__":
    main()
