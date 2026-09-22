#!/usr/bin/env python3

import sys
import cv2
import time
import json
import numpy as np
from pathlib import Path


# ============================================================
# Orbbec import
# ============================================================

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
hole_center = None

tip_center = None
tip_roi = None
tip_features = None


frame_now = None


# ============================================================
# Mouse selection
# ============================================================

select_mode = "hole"


def mouse_callback(event, x, y, flags, param):

    global hole_roi
    global tip_roi
    global select_mode

    global frame_now

    global tip_features
    global tip_center


    if event != cv2.EVENT_LBUTTONDOWN:
        return


    if frame_now is None:
        return


    # ----------------------------------------
    # First selection:
    # Hole ROI
    # ----------------------------------------

    if select_mode == "hole":

        size = 60

        hole_roi = (
            x-size,
            y-size,
            x+size,
            y+size
        )

        select_mode = "tip"

        print(
            "[SELECT] HOLE ROI",
            hole_roi,
            flush=True
        )

        return


    # ----------------------------------------
    # Second selection:
    # Tip ROI
    # ----------------------------------------

    if select_mode == "tip":

        size = 50

        tip_roi = (
            x-size,
            y-size,
            x+size,
            y+size
        )

        tip_center = (
            x,
            y
        )

        gray = cv2.cvtColor(
            frame_now,
            cv2.COLOR_BGR2GRAY
        )


        mask = np.zeros_like(
            gray
        )


        x0,y0,x1,y1 = tip_roi


        x0=max(0,x0)
        y0=max(0,y0)

        x1=min(
            gray.shape[1],
            x1
        )

        y1=min(
            gray.shape[0],
            y1
        )


        mask[
            y0:y1,
            x0:x1
        ]=255


        tip_features = (
            cv2.goodFeaturesToTrack(
                gray,
                maxCorners=50,
                qualityLevel=0.01,
                minDistance=5,
                mask=mask
            )
        )


        select_mode="done"


        if tip_features is None:

            print(
                "[TIP] no features",
                flush=True
            )

        else:

            print(
                f"[SELECT] TIP ROI {tip_roi} "
                f"features={len(tip_features)}",
                flush=True
            )


# ============================================================
# Hole detection
# ============================================================

def detect_hole(gray):

    global hole_roi


    if hole_roi is None:
        return None


    x0,y0,x1,y1 = hole_roi


    x0=max(0,x0)
    y0=max(0,y0)

    x1=min(
        gray.shape[1],
        x1
    )

    y1=min(
        gray.shape[0],
        y1
    )


    roi = gray[
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
        minRadius=3,
        maxRadius=35
    )


    if circles is None:
        return None


    circles=np.uint16(
        np.around(
            circles
        )
    )


    # choose nearest center of ROI

    cx = (
        roi.shape[1]
        /
        2
    )

    cy = (
        roi.shape[0]
        /
        2
    )


    best=None
    best_d=999999


    for c in circles[0]:

        d = (
            (c[0]-cx)**2
            +
            (c[1]-cy)**2
        )


        if d < best_d:

            best_d=d
            best=c


    if best is None:
        return None


    return (
        int(best[0]+x0),
        int(best[1]+y0)
    )


# ============================================================
# Tip LK tracking
# ============================================================

def track_tip(
    prev_gray,
    gray
):

    global tip_features
    global tip_center


    if tip_features is None:
        return


    p0=tip_features


    p1,st,err=(
        cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            p0,
            None,
            winSize=(21,21),
            maxLevel=3
        )
    )


    if p1 is None:
        return


    old=[]
    new=[]


    for a,b,s in zip(
        p0,
        p1,
        st
    ):

        if s[0]:

            old.append(
                a[0]
            )

            new.append(
                b[0]
            )


    if len(old)<3:

        return


    old=np.asarray(old)
    new=np.asarray(new)


    delta=new-old


    move=np.median(
        delta,
        axis=0
    )


    tip_center=(
        int(
            round(
                tip_center[0]+move[0]
            )
        ),

        int(
            round(
                tip_center[1]+move[1]
            )
        )
    )


    tip_features=(
        new
        .reshape(-1,1,2)
        .astype(np.float32)
    )


# ============================================================
# Main
# ============================================================

def main():


    global frame_now
    global hole_center
    global hole_roi
    global tip_roi
    global tip_features
    global tip_center


    print("==============================")
    print("TASK6 WRIST HOLE + TIP TRACKER")
    print("NO ROBOT CONTROL")
    print("==============================")


    context,device = (
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
        "[WRIST CAMERA] RGB started",
        hc.COLOR_WIDTH,
        hc.COLOR_HEIGHT,
        hc.COLOR_FPS
    )


    print("")
    print("First click : hole ROI")
    print("Second click: tip ROI")
    print("")


    win="WRIST HOLE TIP"


    cv2.namedWindow(
        win
    )


    cv2.setMouseCallback(
        win,
        mouse_callback
    )


    prev_gray=None


    last_print=0


    while True:


        frames=(
            pipeline.wait_for_frames(
                1000
            )
        )


        if frames is None:
            continue


        cf=(
            frames.get_color_frame()
        )


        if cf is None:
            continue


        frame=(
            hc.color_frame_to_bgr(
                cf
            )
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


        hole_center=detect_hole(
            gray
        )


        img=frame.copy()


        if hole_roi:

            x0,y0,x1,y1=hole_roi

            cv2.rectangle(
                img,
                (x0,y0),
                (x1,y1),
                (0,255,0),
                2
            )


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


            e=np.sqrt(
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
                    "[WRIST IBVS] "
                    f"H={hole_center} "
                    f"T={tip_center} "
                    f"du={du:+.1f} "
                    f"dv={dv:+.1f} "
                    f"err={e:.1f}",
                    flush=True
                )

                last_print=now


        cv2.imshow(
            win,
            img
        )


        key=cv2.waitKey(1)&0xff


        if key==ord("q"):
            break


        if key==ord("c"):

            hole_roi=None
            tip_roi=None
            tip_features=None
            tip_center=None

            print(
                "[CLEAR]"
            )


        prev_gray=gray.copy()


    pipeline.stop()

    cv2.destroyAllWindows()



if __name__=="__main__":
    main()
