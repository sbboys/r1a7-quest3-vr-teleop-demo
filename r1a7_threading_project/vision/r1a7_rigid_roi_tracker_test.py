#!/usr/bin/env python3

import time
import zlib

import cv2
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient


INTERFACE = "enp6s0"
WINDOW = "R1-A7 Rigid ROI Multi-Point Tracker"

MAX_CORNERS = 40
MIN_FEATURES = 6


def detect_features(gray, roi):
    x, y, w, h = roi

    mask = np.zeros_like(gray)
    mask[y:y+h, x:x+w] = 255

    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=MAX_CORNERS,
        qualityLevel=0.01,
        minDistance=5,
        mask=mask,
        blockSize=5,
    )

    return pts


def track_features(prev_gray, gray, prev_pts):
    if prev_pts is None or len(prev_pts) == 0:
        return None, None, None

    next_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        prev_pts,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS
            | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )

    if next_pts is None or status is None:
        return None, None, None

    # Forward-backward check
    back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
        gray,
        prev_gray,
        next_pts,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS
            | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )

    if back_pts is None:
        return None, None, None

    p0 = prev_pts.reshape(-1, 2)
    p1 = next_pts.reshape(-1, 2)
    pb = back_pts.reshape(-1, 2)

    st = status.reshape(-1).astype(bool)

    fb_error = np.linalg.norm(
        p0 - pb,
        axis=1,
    )

    good = st & (fb_error < 1.0)

    p0_good = p0[good]
    p1_good = p1[good]

    if len(p0_good) < MIN_FEATURES:
        return None, None, None

    displacement = p1_good - p0_good

    # 中位值比平均值更抗异常点
    median_delta = np.median(
        displacement,
        axis=0,
    )

    return (
        p1_good.reshape(-1, 1, 2).astype(np.float32),
        median_delta,
        p1_good,
    )


print("[1] DDS:", INTERFACE)

ChannelFactoryInitialize(
    0,
    INTERFACE,
)

video = VideoClient()
video.SetTimeout(3.0)
video.Init()

last_crc = None

prev_gray = None
points = None
roi = None

total_u = 0.0
total_v = 0.0

history = []
last_stats_print = time.time()

cv2.namedWindow(
    WINDOW,
    cv2.WINDOW_NORMAL,
)

print()
print("Controls:")
print("R : select rigid tweezer ROI")
print("C : clear accumulated displacement")
print("ESC/Q : quit")
print()

while True:

    ret, data = video.GetImageSample()

    if ret != 0 or data is None or len(data) == 0:
        continue

    raw = bytes(data)
    crc = zlib.crc32(raw)

    if crc == last_crc:
        continue

    last_crc = crc

    frame = cv2.imdecode(
        np.frombuffer(
            raw,
            dtype=np.uint8,
        ),
        cv2.IMREAD_COLOR,
    )

    if frame is None:
        continue

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    vis = frame.copy()

    if (
        prev_gray is not None
        and points is not None
    ):

        new_points, delta, visible_points = track_features(
            prev_gray,
            gray,
            points,
        )

        if new_points is None:
            print("[TRACK] lost, please press R again")
            points = None

        else:
            points = new_points

            du = float(delta[0])
            dv = float(delta[1])

            total_u += du
            total_v += dv

            history.append((du, dv))

            if len(history) > 150:
                history.pop(0)

            for px, py in visible_points:
                cv2.circle(
                    vis,
                    (int(round(px)), int(round(py))),
                    3,
                    (0, 255, 0),
                    -1,
                )

            cv2.putText(
                vis,
                f"frame du={du:+.3f} dv={dv:+.3f} px",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                vis,
                f"accum du={total_u:+.3f} dv={total_v:+.3f} px",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            if len(history) >= 20:
                arr = np.asarray(history)

                std_u = float(np.std(arr[:, 0]))
                std_v = float(np.std(arr[:, 1]))

                if time.time() - last_stats_print >= 1.0:
                    print(
                        f"[TRACK] features={len(points)} "
                        f"du={du:+.3f} dv={dv:+.3f} "
                        f"accum=({total_u:+.3f},{total_v:+.3f}) "
                        f"jitter_std=({std_u:.3f},{std_v:.3f}) px",
                        flush=True,
                    )
                    last_stats_print = time.time()

                cv2.putText(
                    vis,
                    f"frame jitter std=({std_u:.3f},{std_v:.3f}) px",
                    (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )

    if roi is not None:
        x, y, w, h = roi

        cv2.rectangle(
            vis,
            (x, y),
            (x+w, y+h),
            (255, 255, 0),
            2,
        )

    cv2.putText(
        vis,
        "R: rigid ROI   C: reset   Q/ESC: quit",
        (20, vis.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.imshow(
        WINDOW,
        vis,
    )

    key = cv2.waitKey(1) & 0xFF

    if key in (27, ord('q'), ord('Q')):
        break

    elif key in (ord('r'), ord('R')):

        selected = cv2.selectROI(
            "Select rigid tweezer region",
            frame,
            fromCenter=False,
            showCrosshair=True,
        )

        cv2.destroyWindow(
            "Select rigid tweezer region"
        )

        x, y, w, h = [
            int(v)
            for v in selected
        ]

        if w > 0 and h > 0:

            roi = (x, y, w, h)

            points = detect_features(
                gray,
                roi,
            )

            if points is None:
                print(
                    "[ROI] no usable corners; choose another area"
                )
            else:
                print(
                    "[ROI] selected:",
                    roi,
                    "features:",
                    len(points),
                )

            total_u = 0.0
            total_v = 0.0
            history.clear()

    elif key in (ord('c'), ord('C')):
        total_u = 0.0
        total_v = 0.0
        history.clear()
        print("[TRACK] displacement reset")

    prev_gray = gray.copy()

cv2.destroyAllWindows()
