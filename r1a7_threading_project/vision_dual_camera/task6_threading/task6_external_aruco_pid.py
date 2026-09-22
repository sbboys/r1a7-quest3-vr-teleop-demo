#!/usr/bin/env python3

import cv2
import json
import os
import time
import argparse
import numpy as np
from pathlib import Path


hole = None


class PID:
    def __init__(self, kp, ki, kd):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)

        self.integral = 0.0
        self.prev_error = None
        self.prev_time = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = None
        self.prev_time = None

    def update(self, error, now):
        error = float(error)

        if self.prev_time is None:
            self.prev_time = now
            self.prev_error = error
            return self.kp * error

        dt = now - self.prev_time

        if dt <= 0.0:
            return self.kp * error

        self.integral += error * dt

        derivative = (
            error - self.prev_error
        ) / dt

        output = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        self.prev_error = error
        self.prev_time = now

        return output


def mouse_callback(event, x, y, flags, param):
    global hole

    if event == cv2.EVENT_LBUTTONDOWN:
        hole = (int(x), int(y))

        print(
            f"[SELECT] HOLE u={hole[0]} v={hole[1]}",
            flush=True
        )


def write_command(path, data):
    path = Path(path)

    tmp = Path(
        str(path) + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            data,
            indent=2
        )
    )

    os.replace(
        tmp,
        path
    )


def main():
    global hole

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        default="/dev/video16"
    )

    parser.add_argument(
        "--tag-id",
        type=int,
        default=2
    )

    parser.add_argument(
        "--kp-x",
        type=float,
        default=0.03
    )

    parser.add_argument(
        "--ki-x",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--kd-x",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--kp-z",
        type=float,
        default=0.03
    )

    parser.add_argument(
        "--ki-z",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--kd-z",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--max-speed-mm-s",
        type=float,
        default=5.0
    )

    parser.add_argument(
        "--command-file",
        default="/tmp/r1a7_ibvs_velocity.json"
    )

    args = parser.parse_args()

    dictionary = (
        cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_5X5_50
        )
    )

    if hasattr(
        cv2.aruco,
        "DetectorParameters"
    ):
        parameters = (
            cv2.aruco.DetectorParameters()
        )
    else:
        parameters = (
            cv2.aruco.DetectorParameters_create()
        )

    if hasattr(
        cv2.aruco,
        "ArucoDetector"
    ):
        detector = cv2.aruco.ArucoDetector(
            dictionary,
            parameters
        )

        def detect(gray):
            return detector.detectMarkers(gray)

    else:
        def detect(gray):
            return cv2.aruco.detectMarkers(
                gray,
                dictionary,
                parameters=parameters
            )

    pid_x = PID(
        args.kp_x,
        args.ki_x,
        args.kd_x
    )

    pid_z = PID(
        args.kp_z,
        args.ki_z,
        args.kd_z
    )

    cap = cv2.VideoCapture(
        args.device,
        cv2.CAP_V4L2
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"camera open failed: {args.device}"
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        30
    )

    win = "TASK6 ARUCO PID"

    cv2.namedWindow(win)

    cv2.setMouseCallback(
        win,
        mouse_callback
    )

    pid_enabled = False
    last_print = 0.0

    print("==============================")
    print("TASK6 ARUCO PID")
    print("==============================")
    print("dictionary : DICT_5X5_50")
    print(f"marker id  : {args.tag_id}")
    print("")
    print("Mapping:")
    print("  +U -> +X")
    print("  +V -> -Z")
    print("")
    print("Left click : select hole")
    print("P          : PID ON/OFF")
    print("R          : reset PID")
    print("C          : clear hole")
    print("Q / ESC    : quit")
    print("")
    print("PID initially OFF")

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        corners, ids, rejected = detect(
            gray
        )

        marker_center = None
        marker_corners = None

        if ids is not None:

            ids_flat = ids.reshape(-1)

            matches = np.where(
                ids_flat == args.tag_id
            )[0]

            if len(matches) > 0:

                i = int(matches[0])

                pts = np.asarray(
                    corners[i],
                    dtype=np.float32
                ).reshape(4, 2)

                marker_corners = pts

                center = np.mean(
                    pts,
                    axis=0
                )

                marker_center = (
                    int(round(center[0])),
                    int(round(center[1]))
                )

        img = frame.copy()

        if marker_corners is not None:

            cv2.polylines(
                img,
                [
                    marker_corners
                    .astype(np.int32)
                    .reshape(-1, 1, 2)
                ],
                True,
                (0, 0, 255),
                2
            )

            cv2.drawMarker(
                img,
                marker_center,
                (0, 0, 255),
                cv2.MARKER_CROSS,
                30,
                2
            )

        if hole is not None:

            cv2.drawMarker(
                img,
                hole,
                (0, 255, 0),
                cv2.MARKER_CROSS,
                30,
                2
            )

        now = time.time()

        vx = 0.0
        vy = 0.0
        vz = 0.0

        du = None
        dv = None

        command_valid = (
            hole is not None
            and
            marker_center is not None
        )

        if command_valid:

            du = float(
                hole[0]
                -
                marker_center[0]
            )

            dv = float(
                hole[1]
                -
                marker_center[1]
            )

            # ------------------------------------------------
            # User-confirmed mapping
            #
            # image +U error -> robot +X
            #
            # image +V error -> robot DOWN
            #                -> Base -Z
            #
            # Therefore:
            #
            # ex = du
            # ez = -dv
            # ------------------------------------------------

            ex = du
            ez = -dv

            if pid_enabled:

                vx = pid_x.update(
                    ex,
                    now
                )

                vz = pid_z.update(
                    ez,
                    now
                )

                vx = float(
                    np.clip(
                        vx,
                        -args.max_speed_mm_s,
                        args.max_speed_mm_s
                    )
                )

                vz = float(
                    np.clip(
                        vz,
                        -args.max_speed_mm_s,
                        args.max_speed_mm_s
                    )
                )

            else:
                pid_x.reset()
                pid_z.reset()

            cv2.line(
                img,
                marker_center,
                hole,
                (255, 255, 0),
                2
            )

            cv2.putText(
                img,
                (
                    f"du={du:+.0f}px "
                    f"dv={dv:+.0f}px"
                ),
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            cv2.putText(
                img,
                (
                    f"vx={vx:+.2f} "
                    f"vz={vz:+.2f} mm/s"
                ),
                (25, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

        else:
            pid_x.reset()
            pid_z.reset()

        state = {
            "timestamp": now,
            "enabled": bool(
                pid_enabled
                and
                command_valid
            ),
            "dictionary": "DICT_5X5_50",
            "marker_id": args.tag_id,
            "hole_pixel": (
                list(hole)
                if hole is not None
                else None
            ),
            "aruco_pixel": (
                list(marker_center)
                if marker_center is not None
                else None
            ),
            "du_px": du,
            "dv_px": dv,
            "vx_mm_s": vx,
            "vy_mm_s": 0.0,
            "vz_mm_s": vz
        }

        write_command(
            args.command_file,
            state
        )

        mode_text = (
            "PID ON"
            if pid_enabled
            else
            "PID OFF"
        )

        cv2.putText(
            img,
            mode_text,
            (25, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (
                (0, 255, 0)
                if pid_enabled
                else
                (0, 0, 255)
            ),
            2
        )

        if now - last_print >= 0.2:

            if command_valid:

                print(
                    "[PID] "
                    f"R={marker_center} "
                    f"T={hole} "
                    f"du={du:+.0f} "
                    f"dv={dv:+.0f} "
                    f"=> "
                    f"vx={vx:+.3f} "
                    f"vz={vz:+.3f} mm/s "
                    f"enabled={pid_enabled}",
                    flush=True
                )

            last_print = now

        cv2.imshow(
            win,
            img
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("p"):

            pid_enabled = (
                not pid_enabled
            )

            pid_x.reset()
            pid_z.reset()

            print(
                "[PID] "
                + (
                    "ENABLED"
                    if pid_enabled
                    else
                    "DISABLED"
                ),
                flush=True
            )

        elif key == ord("r"):

            pid_x.reset()
            pid_z.reset()

            print(
                "[PID] RESET",
                flush=True
            )

        elif key == ord("c"):

            hole = None

            pid_enabled = False

            pid_x.reset()
            pid_z.reset()

            print(
                "[CLEAR] hole, PID OFF",
                flush=True
            )

        elif (
            key == ord("q")
            or
            key == 27
        ):
            break

    write_command(
        args.command_file,
        {
            "timestamp": time.time(),
            "enabled": False,
            "vx_mm_s": 0.0,
            "vy_mm_s": 0.0,
            "vz_mm_s": 0.0
        }
    )

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
