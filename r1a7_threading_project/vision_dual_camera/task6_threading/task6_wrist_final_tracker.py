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


# ============================================================
# Global state
# ============================================================

hole_roi = None
tip_roi = None

select_state = 0


# hole memory

hole_center = None
hole_radius = None

hole_template = None


# tip memory

tip_points_prev = None

tip_reference_offset = None

tip_center = None


frame_now = None


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
    global tip_roi
    global select_state

    global hole_template

    global tip_points_prev
    global tip_reference_offset
    global tip_center

    global frame_now


    if event != cv2.EVENT_LBUTTONDOWN:
        return


    if frame_now is None:
        return


    if select_state == 0:

        size = 60

        hole_roi = (
            x-size,
            y-size,
            x+size,
            y+size
        )

        gray=cv2.cvtColor(
            frame_now,
            cv2.COLOR_BGR2GRAY
        )

        x0,y0,x1,y1=hole_roi

        hole_template = gray[
            y0:y1,
            x0:x1
        ].copy()


        select_state=1


        print(
            "[SELECT HOLE]",
            hole_roi,
            flush=True
        )


    elif select_state == 1:

        size=50

        tip_roi=(
            x-size,
            y-size,
            x+size,
            y+size
        )


        gray=cv2.cvtColor(
            frame_now,
            cv2.COLOR_BGR2GRAY
        )


        mask=np.zeros_like(
            gray
        )


        x0,y0,x1,y1=tip_roi


        mask[
            y0:y1,
            x0:x1
        ]=255


        tip_points_prev=cv2.goodFeaturesToTrack(
            gray,
            maxCorners=40,
            qualityLevel=0.01,
            minDistance=5,
            mask=mask
        )


        tip_center=(
            x,
            y
        )


        if tip_points_prev is not None:

            tip_reference_offset = (
                np.mean(
                    tip_points_prev.reshape(
                        -1,2
                    ),
                    axis=0
                )
                -
                np.array(
                    [
                        x,
                        y
                    ],
                    dtype=float
                )
            )


        select_state=2


        count=0

        if tip_points_prev is not None:
            count=len(tip_points_prev)


        print(
            "[SELECT TIP]",
            tip_roi,
            "features=",
            count,
            flush=True
        )



# ============================================================
# Hole detector
# ============================================================

def detect_hole(gray):

    global hole_center
    global hole_radius


    if hole_roi is None:
        return None


    image_h,image_w = gray.shape[:2]


    x0,y0,x1,y1=hole_roi


    # Clamp ROI inside image boundary

    x0=max(
        0,
        min(
            x0,
            image_w-1
        )
    )

    y0=max(
        0,
        min(
            y0,
            image_h-1
        )
    )

    x1=max(
        x0+1,
        min(
            x1,
            image_w
        )
    )

    y1=max(
        y0+1,
        min(
            y1,
            image_h
        )
    )


    crop=gray[
        y0:y1,
        x0:x1
    ]


    if crop.size == 0:
        return hole_center


    blur=cv2.GaussianBlur(
        crop,
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
        minRadius=5,
        maxRadius=40
    )


    if circles is None:

        return hole_center


    circles=np.round(
        circles[0]
    ).astype(int)


    cx0=(x1-x0)/2
    cy0=(y1-y0)/2


    best=None
    score=999999


    for c in circles:

        d=(
            (c[0]-cx0)**2
            +
            (c[1]-cy0)**2
        )


        if d < score:

            score=d
            best=c


    if best is None:
        return hole_center


    cx=int(
        best[0]+x0
    )

    cy=int(
        best[1]+y0
    )

    r=int(
        best[2]
    )


    if hole_radius is None:

        hole_radius=r


    # radius lock

    if abs(
        r-hole_radius
    ) > max(
        5,
        hole_radius*0.5
    ):

        return hole_center


    hole_center=(
        cx,
        cy
    )


    return hole_center



# ============================================================
# Tip LK rigid tracking
# ============================================================

def track_tip(
    prev_gray,
    gray
):

    global tip_points_prev
    global tip_center


    if tip_points_prev is None:
        return


    new,st,err=cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        tip_points_prev,
        None,
        winSize=(21,21),
        maxLevel=3
    )


    if new is None:
        return


    good_old=[]
    good_new=[]


    for a,b,s in zip(
        tip_points_prev,
        new,
        st
    ):

        if s[0]:

            good_old.append(
                a[0]
            )

            good_new.append(
                b[0]
            )


    if len(good_old)<8:

        return


    good_old=np.asarray(
        good_old,
        dtype=np.float32
    )

    good_new=np.asarray(
        good_new,
        dtype=np.float32
    )


    M,inliers=cv2.estimateAffinePartial2D(
        good_old,
        good_new,
        method=cv2.RANSAC
    )


    if M is None:
        return


    old_point=np.array(
        [
            tip_center[0],
            tip_center[1],
            1
        ],
        dtype=np.float32
    )


    new_point=M @ old_point


    tip_center=(
        int(
            round(
                new_point[0]
            )
        ),
        int(
            round(
                new_point[1]
            )
        )
    )


    tip_points_prev=(
        good_new
        .reshape(-1,1,2)
    )



# ============================================================
# Main
# ============================================================

def main():

    global frame_now
    global last_print


    print(
        "================================"
    )
    print(
        "TASK6 WRIST FINAL TRACKER"
    )
    print(
        "NO ROBOT CONTROL"
    )
    print(
        "================================"
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


    cv2.namedWindow(
        "WRIST FINAL"
    )

    cv2.setMouseCallback(
        "WRIST FINAL",
        mouse_callback
    )


    prev_gray=None


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


        frame_now=frame.copy()


        gray=cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        if prev_gray is not None:

            track_tip(
                prev_gray,
                gray
            )


        H=detect_hole(
            gray
        )


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


        if tip_center:

            cv2.drawMarker(
                img,
                tip_center,
                (0,0,255),
                cv2.MARKER_CROSS,
                30,
                2
            )


        if H and tip_center:


            du=H[0]-tip_center[0]

            dv=H[1]-tip_center[1]


            err=np.hypot(
                du,
                dv
            )


            now=time.time()


            if now-last_print>0.2:

                print(
                    "[WRIST FINAL] "
                    f"H={H} "
                    f"r={hole_radius} "
                    f"T={tip_center} "
                    f"du={du:+.1f} "
                    f"dv={dv:+.1f} "
                    f"err={err:.1f}",
                    flush=True
                )

                last_print=now


        cv2.imshow(
            "WRIST FINAL",
            img
        )


        key=cv2.waitKey(1)&0xff


        if key==ord("q"):
            break


        if key==ord("c"):

            global hole_roi
            global tip_roi
            global select_state

            hole_roi=None
            tip_roi=None

            select_state=0


        prev_gray=gray.copy()


    pipeline.stop()

    cv2.destroyAllWindows()



if __name__=="__main__":
    main()
