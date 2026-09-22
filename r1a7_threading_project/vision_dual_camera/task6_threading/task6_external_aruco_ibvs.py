#!/usr/bin/env python3

import cv2
import numpy as np
import argparse
import time
import json
from pathlib import Path


hole = None


ARUCO_DICT_NAMES = [
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_4X4_1000",

    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_5X5_250",
    "DICT_5X5_1000",

    "DICT_6X6_50",
    "DICT_6X6_100",
    "DICT_6X6_250",
    "DICT_6X6_1000",

    "DICT_7X7_50",
    "DICT_7X7_100",
    "DICT_7X7_250",
    "DICT_7X7_1000",

    "DICT_ARUCO_ORIGINAL",
]


def mouse_callback(event, x, y, flags, param):
    global hole

    if event == cv2.EVENT_LBUTTONDOWN:
        hole = (int(x), int(y))

        print(
            f"[SELECT] HOLE u={hole[0]} v={hole[1]}",
            flush=True
        )


def make_dictionary(name):

    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco not available"
        )

    if not hasattr(cv2.aruco, name):
        return None

    return cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, name)
    )


def make_parameters():

    if hasattr(
        cv2.aruco,
        "DetectorParameters"
    ):
        return cv2.aruco.DetectorParameters()

    return cv2.aruco.DetectorParameters_create()


def detect_one_dictionary(
    gray,
    dictionary,
    parameters
):

    if hasattr(
        cv2.aruco,
        "ArucoDetector"
    ):

        detector = cv2.aruco.ArucoDetector(
            dictionary,
            parameters
        )

        corners, ids, rejected = (
            detector.detectMarkers(gray)
        )

    else:

        corners, ids, rejected = (
            cv2.aruco.detectMarkers(
                gray,
                dictionary,
                parameters=parameters
            )
        )

    return corners, ids, rejected


def build_dictionary_table():

    table = {}

    for name in ARUCO_DICT_NAMES:

        d = make_dictionary(name)

        if d is not None:
            table[name] = d

    return table


def auto_detect(
    gray,
    dictionaries,
    parameters,
    requested_id
):

    best = None

    for name, dictionary in dictionaries.items():

        corners, ids, rejected = (
            detect_one_dictionary(
                gray,
                dictionary,
                parameters
            )
        )

        if ids is None:
            continue

        ids_flat = ids.reshape(-1)

        for i, marker_id in enumerate(
            ids_flat
        ):

            marker_id = int(marker_id)

            if (
                requested_id >= 0
                and
                marker_id != requested_id
            ):
                continue

            pts = np.asarray(
                corners[i],
                dtype=np.float32
            ).reshape(4, 2)

            area = abs(
                cv2.contourArea(
                    pts.astype(np.float32)
                )
            )

            if (
                best is None
                or
                area > best["area"]
            ):

                best = {
                    "dictionary": name,
                    "id": marker_id,
                    "corners": pts,
                    "area": float(area)
                }

    return best


def main():

    global hole


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

    parser.add_argument(
        "--dict",
        default="auto",
        help=(
            "auto or OpenCV dictionary name, "
            "e.g. DICT_4X4_50"
        )
    )

    parser.add_argument(
        "--tag-id",
        type=int,
        default=-1
    )

    parser.add_argument(
        "--save",
        default=(
            "vision_dual_camera/task6_threading/"
            "aruco_ibvs_state.json"
        )
    )

    args = parser.parse_args()


    parameters = make_parameters()

    all_dicts = build_dictionary_table()


    if args.dict != "auto":

        if args.dict not in all_dicts:
            raise RuntimeError(
                f"Unsupported dictionary: "
                f"{args.dict}"
            )

        active_dicts = {
            args.dict:
            all_dicts[args.dict]
        }

    else:

        active_dicts = all_dicts


    print("==============================")
    print("TASK6 ARUCO IBVS TRACKER")
    print("NO ROBOT CONTROL")
    print("==============================")

    print(
        f"camera : {args.device}"
    )

    print(
        f"dict   : {args.dict}"
    )

    print(
        f"tag id : {args.tag_id}"
    )

    print()

    print("Left click : select hole")
    print("C          : clear hole")
    print("S          : save state")
    print("Q / ESC    : quit")

    print()

    print("Mapping:")
    print("  +U -> +X")
    print("  +V -> -Z")

    print()

    print(
        "[INFO] available ArUco dictionaries:"
    )

    for name in active_dicts:
        print(
            " ",
            name
        )


    cap = cv2.VideoCapture(
        args.device,
        cv2.CAP_V4L2
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"camera open failed: "
            f"{args.device}"
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


    win = "TASK6 ARUCO IBVS"

    cv2.namedWindow(win)

    cv2.setMouseCallback(
        win,
        mouse_callback
    )


    last_print = 0.0
    current_state = None
    last_dictionary = None
    last_id = None


    while True:

        ret, frame = cap.read()

        if not ret:
            continue


        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        detected = auto_detect(
            gray,
            active_dicts,
            parameters,
            args.tag_id
        )


        img = frame.copy()

        marker_center = None


        if detected is not None:

            pts = detected["corners"]

            center = np.mean(
                pts,
                axis=0
            )

            marker_center = (
                int(round(center[0])),
                int(round(center[1]))
            )


            p = pts.astype(
                np.int32
            ).reshape(-1, 1, 2)


            cv2.polylines(
                img,
                [p],
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


            dictionary_name = (
                detected["dictionary"]
            )

            marker_id = (
                detected["id"]
            )


            if (
                dictionary_name
                != last_dictionary
                or
                marker_id
                != last_id
            ):

                print(
                    "[ARUCO FOUND] "
                    f"dict={dictionary_name} "
                    f"id={marker_id}",
                    flush=True
                )

                last_dictionary = (
                    dictionary_name
                )

                last_id = marker_id


            cv2.putText(
                img,
                (
                    f"{dictionary_name} "
                    f"ID={marker_id}"
                ),
                (
                    marker_center[0] + 15,
                    marker_center[1] - 15
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )


        else:

            cv2.putText(
                img,
                "ARUCO NOT DETECTED",
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
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

            cv2.putText(
                img,
                "HOLE",
                (
                    hole[0] + 15,
                    hole[1] - 15
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


        if (
            hole is not None
            and
            marker_center is not None
        ):

            # R = ArUco center
            # T = hole

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


            error_px = float(
                np.hypot(
                    du,
                    dv
                )
            )


            # User-confirmed mapping:
            #
            # image +U -> robot +X
            # image +V -> robot -Z

            x_direction = du

            z_direction = -dv


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
                    f"du={du:+.0f} "
                    f"dv={dv:+.0f} "
                    f"err={error_px:.1f}px"
                ),
                (25, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )


            cv2.putText(
                img,
                (
                    f"Robot direction: "
                    f"X={x_direction:+.0f} "
                    f"Z={z_direction:+.0f}"
                ),
                (25, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 0),
                2
            )


            current_state = {

                "timestamp":
                time.time(),

                "dictionary":
                detected["dictionary"],

                "marker_id":
                detected["id"],

                "aruco_center_pixel":
                [
                    marker_center[0],
                    marker_center[1]
                ],

                "hole_pixel":
                [
                    hole[0],
                    hole[1]
                ],

                "image_error_pixel":
                [
                    du,
                    dv
                ],

                "robot_direction":
                {
                    "x": x_direction,
                    "z": z_direction
                }
            }


            now = time.time()

            if (
                now
                -
                last_print
                >= 0.2
            ):

                print(
                    "[ARUCO_IBVS] "
                    f"dict="
                    f"{detected['dictionary']} "
                    f"id="
                    f"{detected['id']} "
                    f"R="
                    f"{marker_center} "
                    f"T="
                    f"{hole} "
                    f"du="
                    f"{du:+.0f} "
                    f"dv="
                    f"{dv:+.0f} "
                    f"err="
                    f"{error_px:.1f}px "
                    f"=> Xsign="
                    f"{x_direction:+.0f} "
                    f"Zsign="
                    f"{z_direction:+.0f}",
                    flush=True
                )

                last_print = now


        cv2.imshow(
            win,
            img
        )


        key = (
            cv2.waitKey(1)
            &
            0xFF
        )


        if (
            key == ord("q")
            or
            key == 27
        ):
            break


        if key == ord("c"):

            hole = None

            print(
                "[CLEAR] hole",
                flush=True
            )


        if key == ord("s"):

            if current_state is None:

                print(
                    "[SAVE] no complete state",
                    flush=True
                )

            else:

                save_path = Path(
                    args.save
                )

                save_path.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                save_path.write_text(
                    json.dumps(
                        current_state,
                        indent=2
                    )
                )

                print(
                    f"[SAVE] {save_path}",
                    flush=True
                )


    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
