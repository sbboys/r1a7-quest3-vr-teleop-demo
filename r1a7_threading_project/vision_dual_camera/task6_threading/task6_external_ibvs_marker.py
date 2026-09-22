#!/usr/bin/env python3

import cv2
import json
import time
from pathlib import Path
import argparse


hole_point = None
tip_point = None


def mouse_callback(event, x, y, flags, param):
    global hole_point
    global tip_point

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if hole_point is None:
        hole_point = (int(x), int(y))
        print(
            f"[SELECT] HOLE u={x} v={y}",
            flush=True
        )

    elif tip_point is None:
        tip_point = (int(x), int(y))
        print(
            f"[SELECT] TIP u={x} v={y}",
            flush=True
        )


def draw(img):

    if hole_point is not None:
        cv2.drawMarker(
            img,
            hole_point,
            (0,255,0),
            cv2.MARKER_CROSS,
            30,
            2
        )

        cv2.putText(
            img,
            "HOLE",
            (
                hole_point[0]+10,
                hole_point[1]
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )


    if tip_point is not None:
        cv2.drawMarker(
            img,
            tip_point,
            (0,0,255),
            cv2.MARKER_CROSS,
            30,
            2
        )

        cv2.putText(
            img,
            "TIP",
            (
                tip_point[0]+10,
                tip_point[1]
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,255),
            2
        )


    if (
        hole_point is not None
        and
        tip_point is not None
    ):

        du = hole_point[0]-tip_point[0]
        dv = hole_point[1]-tip_point[1]

        err = (
            du*du+dv*dv
        )**0.5

        cv2.putText(
            img,
            f"du={du:+d} dv={dv:+d}",
            (30,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255,255,0),
            2
        )

        cv2.putText(
            img,
            f"pixel_error={err:.1f}px",
            (30,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255,255,0),
            2
        )


def main():

    global hole_point
    global tip_point


    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        default="/dev/video16"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1280
    )

    parser.add_argument(
        "--height",
        type=int,
        default=720
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30
    )

    args = parser.parse_args()


    cap = cv2.VideoCapture(
        args.device,
        cv2.CAP_V4L2
    )


    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open {args.device}"
        )


    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        args.width
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        args.height
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        args.fps
    )


    print("==============================")
    print("TASK6 External IBVS Marker")
    print("NO ROBOT CONTROL")
    print("==============================")
    print("First click : HOLE")
    print("Second click: TIP")
    print("C: clear")
    print("S: save")
    print("Q: quit")


    window="TASK6 IBVS"

    cv2.namedWindow(
        window,
        cv2.WINDOW_NORMAL
    )

    cv2.setMouseCallback(
        window,
        mouse_callback
    )


    while True:

        ret, frame = cap.read()

        if not ret:
            continue


        img = frame.copy()

        draw(img)


        cv2.imshow(
            window,
            img
        )


        key = (
            cv2.waitKey(1)
            &
            0xff
        )


        if key == ord("q"):
            break


        if key == ord("c"):

            hole_point=None
            tip_point=None

            print(
                "[CLEAR]",
                flush=True
            )


        if key == ord("s"):

            if (
                hole_point is not None
                and
                tip_point is not None
            ):

                data={

                    "timestamp":
                    time.time(),

                    "hole_pixel":
                    list(hole_point),

                    "tip_pixel":
                    list(tip_point),

                    "error_pixel":
                    [
                        hole_point[0]-tip_point[0],
                        hole_point[1]-tip_point[1]
                    ]

                }


                out=Path(
                    "vision_dual_camera/"
                    "task6_threading/"
                    "ibvs_target_pixel.json"
                )

                out.write_text(
                    json.dumps(
                        data,
                        indent=2
                    )
                )

                print(
                    "[SAVE]",
                    out,
                    flush=True
                )


    cap.release()

    cv2.destroyAllWindows()



if __name__=="__main__":
    main()
