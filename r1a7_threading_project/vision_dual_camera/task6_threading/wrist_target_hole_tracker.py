#!/usr/bin/env python3

import sys
import json
import cv2
import json
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


TIP_FILE = (
    ROOT /
    "vision_dual_camera" /
    "task6_threading" /
    "wrist_tip_calibration.json"
)


# ============================================================
# State
# ============================================================

hole_roi = None

select_done = False


hole_center = None
hole_radius = None

hole_template = None


tip_pixel = None


last_print = 0


# ============================================================
# Mouse
# ============================================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    global hole_roi
    global select_done


    if event != cv2.EVENT_LBUTTONDOWN:
        return


    if select_done:
        return


    size = 60


    hole_roi = (
        x-size,
        y-size,
        x+size,
        y+size
    )


    select_done=True


    print(
        "[SELECT TARGET HOLE ROI]",
        hole_roi,
        flush=True
    )


# ============================================================
# Load fixed tip
# ============================================================

def load_tip():

    global tip_pixel


    if not TIP_FILE.exists():

        raise RuntimeError(
            "tip calibration missing"
        )


    data=json.loads(
        TIP_FILE.read_text()
    )


    tip_pixel=tuple(
        data["tip_pixel"]
    )


    print(
        "[TIP FIXED]",
        tip_pixel,
        flush=True
    )


# ============================================================
# Hole initialization
# ============================================================

def init_hole(gray):

    global hole_center
    global hole_radius
    global hole_template


    if hole_roi is None:
        return False


    x0,y0,x1,y1=hole_roi


    crop=gray[
        y0:y1,
        x0:x1
    ]


    if crop.size==0:
        return False


    blur=cv2.GaussianBlur(
        crop,
        (5,5),
        0
    )


    circles=cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=80,
        param2=12,
        minRadius=5,
        maxRadius=40
    )


    if circles is None:
        return False


    circles=np.round(
        circles[0]
    ).astype(int)


    c=circles[0]


    hole_center=(
        int(c[0]+x0),
        int(c[1]+y0)
    )


    hole_radius=int(c[2])


    size=40


    hx,hy=hole_center


    hole_template=gray[
        hy-size:hy+size,
        hx-size:hx+size
    ].copy()


    print(
        "[HOLE INIT]",
        hole_center,
        "radius=",
        hole_radius,
        flush=True
    )


    return True



# ============================================================
# Hole tracking
# ============================================================

def track_hole(gray):

    global hole_center


    if hole_center is None:
        return None


    hx,hy=hole_center


    search=35


    x0=max(
        0,
        hx-search
    )

    y0=max(
        0,
        hy-search
    )

    x1=min(
        gray.shape[1],
        hx+search
    )

    y1=min(
        gray.shape[0],
        hy+search
    )


    roi=gray[
        y0:y1,
        x0:x1
    ]


    blur=cv2.GaussianBlur(
        roi,
        (5,5),
        0
    )


    circles=cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=15,
        param1=80,
        param2=12,
        minRadius=max(
            3,
            hole_radius-5
        ),
        maxRadius=hole_radius+5
    )


    if circles is None:

        return hole_center


    circles=np.round(
        circles[0]
    ).astype(int)


    best=None
    best_score=999999


    for c in circles:


        cx=c[0]+x0
        cy=c[1]+y0
        r=c[2]


        d=np.hypot(
            cx-hx,
            cy-hy
        )


        if d<best_score:

            best_score=d

            best=(
                int(cx),
                int(cy),
                int(r)
            )


    if best is not None:

        hole_center=(
            best[0],
            best[1]
        )


    return hole_center



# ============================================================
# Main
# ============================================================

def main():

    global last_print


    load_tip()


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


    window="WRIST HOLE LOCK"


    cv2.namedWindow(
        window
    )


    cv2.setMouseCallback(
        window,
        mouse_callback
    )


    initialized=False


    while True:


        frames=pipeline.wait_for_frames(
            1000
        )


        if frames is None:
            continue


        color=frames.get_color_frame()


        if color is None:
            continue


        frame=hc.color_frame_to_bgr(
            color
        )


        gray=cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        if (
            select_done
            and
            not initialized
        ):

            initialized=init_hole(
                gray
            )


        if initialized:

            H=track_hole(
                gray
            )

        else:

            H=None



        img=frame.copy()


        if H:

            cv2.drawMarker(
                img,
                H,
                (0,255,0),
                cv2.MARKER_CROSS,
                30,
                2
            )


        if tip_pixel:

            cv2.drawMarker(
                img,
                tip_pixel,
                (0,0,255),
                cv2.MARKER_CROSS,
                30,
                2
            )


        if H and tip_pixel:


            du=H[0]-tip_pixel[0]

            dv=H[1]-tip_pixel[1]


            now=time.time()


            if now-last_print>0.2:


                Path(
                    "vision_dual_camera/task6_threading/"
                    "wrist_error.json"
                ).write_text(
                    json.dumps(
                        {
                            "du":du,
                            "dv":dv,
                            "timestamp":time.time()
                        }
                    )
                )


                print(
                    "[WRIST HOLE LOCK] "
                    f"H={H} "
                    f"T={tip_pixel} "
                    f"du={du:+.1f} "
                    f"dv={dv:+.1f}",
                    flush=True
                )


                last_print=now


        cv2.imshow(
            window,
            img
        )


        key=cv2.waitKey(1)&0xff


        if key==ord("q"):
            break


    pipeline.stop()

    cv2.destroyAllWindows()



if __name__=="__main__":
    main()
