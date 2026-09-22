#!/usr/bin/env python3

import os
import socket
import json

import time
import math
import cv2
import numpy as np

from r1a7_rigid_tool_tracker import RigidRoiTracker


# ==============================
# Camera
# ==============================

CAMERA = (
"/dev/v4l/by-id/"
"usb-ALP_USB_ZOOM_Camera_20685204b12d5283-video-index0"
)


# ==============================
# Pixel Jacobian
# pixel/mm
# ==============================

J_PIXEL = np.array([
    [ 2.9, -0.63],
    [ 0.76,-1.73]
])


J_PINV = np.linalg.pinv(
    J_PIXEL
)


# ==============================
# Controller
# ==============================

GAIN = 0.25

MAX_STEP_MM = 0.5

STOP_PIXEL_ERROR = 10


def select_roi(title,frame):

    roi=cv2.selectROI(
        title,
        frame,
        False,
        True
    )

    cv2.destroyWindow(title)

    x,y,w,h=[
        int(v)
        for v in roi
    ]

    return np.array(
        [x,y,w,h],
        dtype=float
    )


def limit_step(dx,dy):

    norm=np.sqrt(
        dx*dx+dy*dy
    )

    if norm>MAX_STEP_MM:

        scale=MAX_STEP_MM/norm

        dx*=scale
        dy*=scale

    return dx,dy




class RobotServoClient:

    def __init__(self):

        self.sock=socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM
        )

        self.addr=(
            "127.0.0.1",
            5005
        )


    def move_xy(self,dx,dy):

        msg={
            "dx_mm":float(dx),
            "dy_mm":float(dy),
            "dz_mm":0.0
        }

        self.sock.sendto(
            json.dumps(msg).encode(),
            self.addr
        )

        print(
            "[VISUAL SERVO SEND]",
            msg
        )


def select_roi(title,frame):

    roi=cv2.selectROI(
        title,
        frame,
        False,
        True
    )

    cv2.destroyWindow(title)

    x,y,w,h=[
        int(v)
        for v in roi
    ]

    return np.array(
        [x,y,w,h],
        dtype=float
    )


def limit_step(dx,dy):

    norm=np.sqrt(
        dx*dx+dy*dy
    )

    if norm>MAX_STEP_MM:

        scale=MAX_STEP_MM/norm

        dx*=scale
        dy*=scale

    return dx,dy





def main():


    robot=RobotServoClient()


    cam=os.path.realpath(
        CAMERA
    )


    cap=cv2.VideoCapture(
        cam
    )


    if not cap.isOpened():

        raise RuntimeError(
            "camera failed"
        )


    ok,frame=cap.read()

    if not ok:
        raise RuntimeError(
            "frame failed"
        )


    print(
        "Select TIP"
    )


    tip_roi=select_roi(
        "TIP",
        frame
    )


    tracker=RigidRoiTracker()


    if not tracker.initialize(
        frame,
        tip_roi
    ):

        raise RuntimeError(
            tracker.status
        )


    print(
        "Select HOLE"
    )


    hole_roi=select_roi(
        "HOLE",
        frame
    )


    hx,hy,hw,hh=hole_roi


    hole=np.array([
        hx+hw/2,
        hy+hh/2
    ])


    print(
        "START SERVO"
    )


    while True:


        ok,frame=cap.read()

        if not ok:
            break


        tracker.update(frame)


        if tracker.center is None:

            print(
                "TIP LOST"
            )

            continue


        tip=tracker.center.copy()


        error=np.array(
            [
                hole[0]-tip[0],
                hole[1]-tip[1]
            ]
        )


        pixel_error=np.linalg.norm(
            error
        )


        print(
            "error=",
            pixel_error,
            "px",
            "tip=",
            tip
        )


        if pixel_error < STOP_PIXEL_ERROR:

            print(
                "ALIGN SUCCESS"
            )

            break



        # --------------------
        # pixel -> robot XY
        # --------------------

        delta_xy = (
            GAIN
            *
            J_PINV
            @
            error
        )


        dx=float(
            delta_xy[0]
        )

        dy=float(
            delta_xy[1]
        )


        dx,dy=limit_step(
            dx,
            dy
        )


        robot.move_xy(
            dx,
            dy
        )


        time.sleep(
            1.0
        )


        cv2.imshow(
            "servo",
            frame
        )


        if cv2.waitKey(1)&0xff==ord("q"):

            break



    cap.release()
    cv2.destroyAllWindows()



if __name__=="__main__":

    main()
