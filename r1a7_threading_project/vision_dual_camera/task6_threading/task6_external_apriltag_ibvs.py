#!/usr/bin/env python3

import cv2
import json
import time
import argparse
import numpy as np
from pathlib import Path


hole = None


def mouse_callback(event, x, y, flags, param):
    global hole

    if event == cv2.EVENT_LBUTTONDOWN:
        hole = (int(x), int(y))

        print(
            f"[SELECT] HOLE u={hole[0]} v={hole[1]}",
            flush=True
        )


def get_dictionary(name):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "OpenCV has no aruco module. "
            "Install opencv-contrib-python."
        )

    if not hasattr(cv2.aruco, name):
        raise RuntimeError(
            f"AprilTag dictionary not found: {name}"
        )

    dict_id = getattr(cv2.aruco, name)

    return cv2.aruco.getPredefinedDictionary(
        dict_id
    )


def create_detector(dictionary):
    if hasattr(cv2.aruco, "DetectorParameters"):
        params = cv2.aruco.DetectorParameters()
    else:
        params = cv2.aruco.DetectorParameters_create()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(
            dictionary,
            params
        )

        def detect(gray):
            corners, ids, rejected = detector.detectMarkers(
                gray
            )
            return corners, ids, rejected

        return detect

    def detect(gray):
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray,
            dictionary,
            parameters=params
        )
        return corners, ids, rejected

    return detect


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
        default="DICT_APRILTAG_36h11"
    )

    parser.add_argument(
        "--tag-id",
        type=int,
        default=-1,
        help="-1 means use the first detected tag"
    )

    parser.add_argument(
        "--save",
        default=(
            "vision_dual_camera/task6_threading/"
            "apriltag_ibvs_state.json"
        )
    )

    args = parser.parse_args()

    dictionary = get_dictionary(
        args.dict
    )

    detect_markers = create_detector(
        dictionary
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

    win = "TASK6 APRILTAG IBVS"

    cv2.namedWindow(win)

    cv2.setMouseCallback(
        win,
        mouse_callback
    )

    print("==============================")
    print("TASK6 APRILTAG IBVS TRACKER")
    print("NO ROBOT CONTROL")
    print("==============================")
    print(f"camera : {args.device}")
    print(f"dict   : {args.dict}")
    print(f"tag id : {args.tag_id}")
    print("")
    print("Left click : select hole")
    print("C          : clear hole")
    print("S          : save current state")
    print("Q / ESC    : quit")
    print("")
    print("Mapping:")
    print("  +U -> +X")
    print("  +V -> -Z")
    print("")

    last_print = 0.0
    current_state = None

    while True:
        ret, frame = cap.read()

        if not ret:
            continue

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        corners, ids, rejected = detect_markers(
            gray
        )

        img = frame.copy()

        tag_center = None
        detected_id = None

        if ids is not None and len(ids) > 0:

            ids_flat = ids.reshape(-1)

            selected_index = None

            if args.tag_id >= 0:
                matches = np.where(
                    ids_flat == args.tag_id
                )[0]

                if len(matches) > 0:
                    selected_index = int(
                        matches[0]
                    )

            else:
                selected_index = 0

            cv2.aruco.drawDetectedMarkers(
                img,
                corners,
                ids
            )

            if selected_index is not None:

                pts = np.asarray(
                    corners[selected_index],
                    dtype=np.float32
                ).reshape(4, 2)

                center = np.mean(
                    pts,
                    axis=0
                )

                tag_center = (
                    int(round(center[0])),
                    int(round(center[1]))
                )

                detected_id = int(
                    ids_flat[selected_index]
                )

                cv2.drawMarker(
                    img,
                    tag_center,
                    (0, 0, 255),
                    cv2.MARKER_CROSS,
                    30,
                    2
                )

                cv2.putText(
                    img,
                    f"TAG {detected_id}",
                    (
                        tag_center[0] + 15,
                        tag_center[1] - 15
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
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
            tag_center is not None
        ):

            # Image error:
            # target - current
            du = float(
                hole[0]
                -
                tag_center[0]
            )

            dv = float(
                hole[1]
                -
                tag_center[1]
            )

            error_px = float(
                np.hypot(
                    du,
                    dv
                )
            )

            # User-confirmed image -> robot direction:
            #
            # +U -> +X
            # +V -> robot downward -> -Z
            #
            robot_x_error = du
            robot_z_error = -dv

            cv2.line(
                img,
                tag_center,
                hole,
                (255, 255, 0),
                2
            )

            text1 = (
                f"du={du:+.0f}px "
                f"dv={dv:+.0f}px "
                f"err={error_px:.1f}px"
            )

            text2 = (
                f"robot: X~{robot_x_error:+.0f} "
                f"Z~{robot_z_error:+.0f}"
            )

            cv2.putText(
                img,
                text1,
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            cv2.putText(
                img,
                text2,
                (25, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            current_state = {
                "timestamp": time.time(),
                "tag_id": detected_id,
                "tag_pixel": [
                    tag_center[0],
                    tag_center[1]
                ],
                "hole_pixel": [
                    hole[0],
                    hole[1]
                ],
                "image_error_pixel": [
                    du,
                    dv
                ],
                "robot_direction_error": {
                    "x_from_u": robot_x_error,
                    "z_from_v": robot_z_error
                }
            }

            now = time.time()

            if now - last_print >= 0.2:

                print(
                    "[APRILTAG_IBVS] "
                    f"id={detected_id} "
                    f"R={tag_center} "
                    f"T={hole} "
                    f"du={du:+.0f} "
                    f"dv={dv:+.0f} "
                    f"err={error_px:.1f}px "
                    f"=> Xsign={robot_x_error:+.0f} "
                    f"Zsign={robot_z_error:+.0f}",
                    flush=True
                )

                last_print = now

        elif tag_center is None:

            cv2.putText(
                img,
                "APRILTAG NOT DETECTED",
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            win,
            img
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
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
