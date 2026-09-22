#!/usr/bin/env python3

import cv2
import numpy as np
import time
import json
from pathlib import Path
import argparse


hole = None
tip = None
tip_template = None
last_print = 0


TEMPLATE_SIZE = 40
SEARCH_SIZE = 120


def mouse_callback(event,x,y,flags,param):

    global hole
    global tip
    global tip_template
    global current_frame


    if event != cv2.EVENT_LBUTTONDOWN:
        return


    if hole is None:

        hole=(x,y)

        print(
            f"[SELECT] HOLE u={x} v={y}",
            flush=True
        )


    elif tip is None:

        tip=(x,y)

        gray=cv2.cvtColor(
            current_frame,
            cv2.COLOR_BGR2GRAY
        )

        h=TEMPLATE_SIZE

        tip_template=gray[
            y-h:y+h,
            x-h:x+h
        ].copy()


        print(
            f"[SELECT] TIP u={x} v={y}",
            flush=True
        )


def track_tip(frame):

    global tip

    global tip_template


    if (
        tip is None
        or
        tip_template is None
    ):
        return


    gray=cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    x,y=tip


    s=SEARCH_SIZE

    x0=max(
        0,
        x-s
    )

    y0=max(
        0,
        y-s
    )

    x1=min(
        gray.shape[1],
        x+s
    )

    y1=min(
        gray.shape[0],
        y+s
    )


    roi=gray[
        y0:y1,
        x0:x1
    ]


    result=cv2.matchTemplate(
        roi,
        tip_template,
        cv2.TM_CCOEFF_NORMED
    )


    _,score,_,loc=cv2.minMaxLoc(
        result
    )


    if score < 0.5:
        return


    th,tw=tip_template.shape


    nx=x0+loc[0]+tw//2
    ny=y0+loc[1]+th//2


    tip=(
        int(nx),
        int(ny)
    )


    return score



def main():

    global hole
    global tip
    global tip_template
    global current_frame
    global last_print


    parser=argparse.ArgumentParser()


    parser.add_argument(
        "--device",
        default="/dev/video16"
    )


    args=parser.parse_args()


    cap=cv2.VideoCapture(
        args.device,
        cv2.CAP_V4L2
    )


    if not cap.isOpened():

        raise RuntimeError(
            "camera open failed"
        )


    win="TASK6 IBVS Tracker"


    cv2.namedWindow(
        win
    )

    cv2.setMouseCallback(
        win,
        mouse_callback
    )


    print("====================")
    print("TASK6 IBVS Tracker")
    print("NO ROBOT CONTROL")
    print("====================")
    print("click hole")
    print("click tip")
    print("s save")
    print("q quit")


    while True:


        ret,frame=cap.read()


        if not ret:
            continue


        current_frame=frame.copy()


        score=track_tip(
            frame
        )


        img=frame.copy()


        if hole:

            cv2.drawMarker(
                img,
                hole,
                (0,255,0),
                cv2.MARKER_CROSS,
                30,
                2
            )


        if tip:

            cv2.drawMarker(
                img,
                tip,
                (0,0,255),
                cv2.MARKER_CROSS,
                30,
                2
            )


        if (
            hole
            and
            tip
        ):


            du=hole[0]-tip[0]
            dv=hole[1]-tip[1]

            err=np.sqrt(
                du*du+dv*dv
            )


            cv2.putText(
                img,
                f"du={du:+d} dv={dv:+d}",
                (30,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,0),
                2
            )


            cv2.putText(
                img,
                f"error={err:.1f}px",
                (30,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,0),
                2
            )


            now=time.time()

            if now-last_print>0.2:

                print(
                    "[IBVS]"
                    f" hole={hole}"
                    f" tip={tip}"
                    f" du={du:+d}"
                    f" dv={dv:+d}"
                    f" err={err:.1f}"
                    ,
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

            hole=None
            tip=None


        if key==ord("s"):


            if hole and tip:


                data={

                    "time":time.time(),

                    "hole":hole,

                    "tip":tip,

                    "error":[
                        hole[0]-tip[0],
                        hole[1]-tip[1]
                    ]

                }


                Path(
                    "vision_dual_camera/task6_threading/"
                    "ibvs_tracking.json"
                ).write_text(
                    json.dumps(
                        data,
                        indent=2
                    )
                )


                print(
                    "[SAVE]"
                )



    cap.release()

    cv2.destroyAllWindows()



if __name__=="__main__":

    main()
