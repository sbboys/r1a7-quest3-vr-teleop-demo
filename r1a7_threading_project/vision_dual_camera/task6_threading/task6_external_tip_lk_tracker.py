#!/usr/bin/env python3

import cv2
import numpy as np
import time


hole = None
roi = None
initialized = False

feature_points = None
last_tip = None
frame_now = None


def mouse_callback(event,x,y,flags,param):

    global hole
    global roi
    global initialized
    global feature_points
    global frame_now


    if event != cv2.EVENT_LBUTTONDOWN:
        return


    if hole is None:

        hole=(x,y)

        print(
            f"[SELECT] HOLE u={x} v={y}",
            flush=True
        )

    elif roi is None:

        roi=[
            x-40,
            y-40,
            x+40,
            y+40
        ]

        print(
            "[SELECT] TIP ROI",
            roi,
            flush=True
        )

        gray=cv2.cvtColor(
            frame_now,
            cv2.COLOR_BGR2GRAY
        )


        mask=np.zeros_like(gray)

        x0,y0,x1,y1=roi

        mask[
            max(y0,0):min(y1,gray.shape[0]),
            max(x0,0):min(x1,gray.shape[1])
        ]=255


        feature_points=cv2.goodFeaturesToTrack(
            gray,
            maxCorners=30,
            qualityLevel=0.01,
            minDistance=5,
            mask=mask
        )


        if feature_points is None:

            print(
                "[ERROR] no features",
                flush=True
            )

        else:

            initialized=True

            print(
                f"[TRACK] features={len(feature_points)}",
                flush=True
            )



def lk_track(prev_gray, gray):

    global feature_points
    global last_tip
    global roi

    if feature_points is None:
        return None

    p0 = np.asarray(
        feature_points,
        dtype=np.float32
    ).reshape(-1, 1, 2)

    # --------------------------------------------------
    # Forward LK
    # --------------------------------------------------

    p1, st1, err1 = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        p0,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS
            |
            cv2.TERM_CRITERIA_COUNT,
            30,
            0.01
        )
    )

    if p1 is None or st1 is None:
        print(
            "[TRACK] LOST: forward LK failed",
            flush=True
        )
        return None

    # --------------------------------------------------
    # Backward LK
    # --------------------------------------------------

    p0_back, st2, err2 = cv2.calcOpticalFlowPyrLK(
        gray,
        prev_gray,
        p1,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS
            |
            cv2.TERM_CRITERIA_COUNT,
            30,
            0.01
        )
    )

    if p0_back is None or st2 is None:
        print(
            "[TRACK] LOST: backward LK failed",
            flush=True
        )
        return None

    # --------------------------------------------------
    # Convert:
    #
    # OpenCV:
    #   N x 1 x 2
    #
    # into:
    #   N x 2
    # --------------------------------------------------

    p0_xy = p0.reshape(-1, 2)
    p1_xy = p1.reshape(-1, 2)
    pb_xy = p0_back.reshape(-1, 2)

    st1 = st1.reshape(-1).astype(bool)
    st2 = st2.reshape(-1).astype(bool)

    # Forward/backward consistency error.
    fb_error = np.linalg.norm(
        p0_xy - pb_xy,
        axis=1
    )

    good = (
        st1
        &
        st2
        &
        np.isfinite(fb_error)
        &
        (fb_error < 1.5)
    )

    good_count = int(
        np.count_nonzero(good)
    )

    if good_count < 3:

        print(
            f"[TRACK] LOST: only "
            f"{good_count} reliable features",
            flush=True
        )

        return None

    good_old = p0_xy[good]
    good_new = p1_xy[good]

    # --------------------------------------------------
    # Robust rigid-region displacement.
    # --------------------------------------------------

    delta = (
        good_new
        -
        good_old
    )

    median_delta = np.median(
        delta,
        axis=0
    )

    mdx = float(
        median_delta[0]
    )

    mdy = float(
        median_delta[1]
    )

    # Keep only reliable features for next frame.
    feature_points = (
        good_new
        .reshape(-1, 1, 2)
        .astype(np.float32)
    )

    # First tracked frame:
    # initialize the virtual tip from clicked ROI center.
    if last_tip is None:

        if roi is not None:

            last_tip = (
                int(
                    round(
                        0.5
                        *
                        (
                            roi[0]
                            +
                            roi[2]
                        )
                    )
                ),
                int(
                    round(
                        0.5
                        *
                        (
                            roi[1]
                            +
                            roi[3]
                        )
                    )
                )
            )

        else:

            center = np.median(
                good_old,
                axis=0
            )

            last_tip = (
                int(round(center[0])),
                int(round(center[1]))
            )

    last_tip = (
        int(
            round(
                last_tip[0]
                +
                mdx
            )
        ),
        int(
            round(
                last_tip[1]
                +
                mdy
            )
        )
    )

    return last_tip



def main():

    global frame_now
    global last_tip


    cap=cv2.VideoCapture(
        "/dev/video16",
        cv2.CAP_V4L2
    )


    if not cap.isOpened():

        raise RuntimeError(
            "camera open failed"
        )


    cv2.namedWindow(
        "TASK6 LK IBVS"
    )


    cv2.setMouseCallback(
        "TASK6 LK IBVS",
        mouse_callback
    )


    print("====================")
    print("TASK6 LK TIP TRACKER")
    print("NO ROBOT CONTROL")
    print("====================")
    print("click hole")
    print("click tip center")
    print("q quit")


    prev_gray=None


    while True:


        ret,frame=cap.read()

        if not ret:
            continue


        frame_now=frame.copy()


        gray=cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        if (
            prev_gray is not None
            and
            initialized
        ):

            tip=lk_track(
                prev_gray,
                gray
            )

            if tip is not None:

                last_tip=tip



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


        if last_tip:

            cv2.drawMarker(
                img,
                last_tip,
                (0,0,255),
                cv2.MARKER_CROSS,
                30,
                2
            )


        if (
            hole
            and
            last_tip
        ):

            du=hole[0]-last_tip[0]
            dv=hole[1]-last_tip[1]


            err=np.sqrt(
                du*du+dv*dv
            )


            print(
                f"[IBVS]"
                f" R={last_tip}"
                f" T={hole}"
                f" e=({du:+.1f},{dv:+.1f})"
                f" err={err:.1f}",
                flush=True
            )


            cv2.putText(
                img,
                f"du={du:.1f} dv={dv:.1f}",
                (30,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,0),
                2
            )


        cv2.imshow(
            "TASK6 LK IBVS",
            img
        )


        key=cv2.waitKey(1)&255


        if key==ord("q"):
            break


        prev_gray=gray.copy()


    cap.release()

    cv2.destroyAllWindows()



if __name__=="__main__":
    main()
