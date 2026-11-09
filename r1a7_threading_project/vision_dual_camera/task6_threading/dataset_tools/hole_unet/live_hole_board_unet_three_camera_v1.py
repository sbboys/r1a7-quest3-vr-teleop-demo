#!/usr/bin/env python3

from pathlib import Path
import argparse
import time

import cv2
import numpy as np
import torch

from pyorbbecsdk import (
    Context,
    Pipeline,
    Config,
    OBSensorType,
    OBFormat,
)

from train_hole_board_unet_v1 import (
    UNet,
    predict_sliding,
)


# ============================================================
# Camera configuration
# ============================================================

EXTERNAL_DEVICE = (
    "/dev/v4l/by-id/"
    "usb-ALP_USB_ZOOM_Camera_20685204b12d5283-"
    "video-index0"
)

RIGHT_WRIST_SERIAL = "CPCBC53000C5"
LEFT_WRIST_SERIAL = "CPCBC530007Z"

CHECKPOINT_DEFAULT = (
    "/home/robot/r1a7_datasets/wire_unet/"
    "hole_tube_001/runs/"
    "hole_board_unet_v1/best.pt"
)


# ============================================================
# External USB camera
# ============================================================

class ExternalCamera:

    def __init__(self):

        self.cap = cv2.VideoCapture(
            EXTERNAL_DEVICE,
            cv2.CAP_V4L2,
        )

        if not self.cap.isOpened():

            raise RuntimeError(
                "Failed to open External camera: "
                + EXTERNAL_DEVICE
            )

        self.cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(
                *"MJPG"
            ),
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            1280,
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            720,
        )

        self.cap.set(
            cv2.CAP_PROP_FPS,
            30,
        )

        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1,
        )

        # Flush initial frames.
        for _ in range(4):
            self.cap.read()

        print(
            "[CAMERA] External opened"
        )

    def read(self):

        ok, frame = self.cap.read()

        if not ok or frame is None:
            return None

        return frame

    def close(self):

        self.cap.release()


# ============================================================
# Orbbec camera
# ============================================================

class OrbbecCamera:

    def __init__(
        self,
        serial,
        name,
        width=1280,
        height=800,
        fps=30,
    ):

        self.serial = serial
        self.name = name
        self.width = width
        self.height = height

        self.context = Context()

        devices = (
            self.context.query_devices()
        )

        count = devices.get_count()

        print(
            f"[ORBBEC] found {count} devices"
        )

        self.device = None

        for i in range(count):

            dev = (
                devices.get_device_by_index(
                    i
                )
            )

            try:

                info = (
                    dev.get_device_info()
                )

                sn = (
                    info.get_serial_number()
                )

            except Exception:

                continue

            print(
                "[ORBBEC]",
                i,
                "serial=",
                sn,
            )

            if str(sn) == str(serial):

                self.device = dev
                break

        if self.device is None:

            raise RuntimeError(
                f"Cannot find {name} "
                f"Orbbec serial={serial}"
            )

        self.pipeline = Pipeline(
            self.device
        )

        self.config = Config()

        profile_list = (
            self.pipeline
            .get_stream_profile_list(
                OBSensorType.COLOR_SENSOR
            )
        )

        try:

            profile = (
                profile_list
                .get_video_stream_profile(
                    width,
                    height,
                    OBFormat.RGB,
                    fps,
                )
            )

        except Exception as e:

            raise RuntimeError(
                f"{name}: failed to request "
                f"{width}x{height}@{fps} RGB: {e}"
            )

        self.config.enable_stream(
            profile
        )

        self.pipeline.start(
            self.config
        )

        # Warm up.
        for _ in range(5):

            try:
                self.pipeline.wait_for_frames(
                    100
                )
            except Exception:
                pass

        print(
            f"[CAMERA] {name} opened "
            f"{width}x{height}@{fps}"
        )

    def read(self):

        try:

            frames = (
                self.pipeline
                .wait_for_frames(
                    100
                )
            )

        except Exception:

            return None

        if frames is None:
            return None

        color_frame = (
            frames.get_color_frame()
        )

        if color_frame is None:
            return None

        try:

            width = (
                color_frame.get_width()
            )

            height = (
                color_frame.get_height()
            )

            data = np.asanyarray(
                color_frame.get_data()
            )

            data = np.asarray(
                data,
                dtype=np.uint8,
            )

            expected = (
                width
                *
                height
                *
                3
            )

            if data.size != expected:

                print(
                    f"[WARNING] {self.name}: "
                    f"unexpected RGB buffer "
                    f"size={data.size} "
                    f"expected={expected}"
                )

                return None

            rgb = data.reshape(
                (
                    height,
                    width,
                    3,
                )
            )

            # Requested Orbbec stream format is RGB.
            bgr = cv2.cvtColor(
                rgb,
                cv2.COLOR_RGB2BGR,
            )

            return bgr

        except Exception as e:

            print(
                f"[WARNING] {self.name} "
                f"decode failed: {e}"
            )

            return None

    def close(self):

        try:
            self.pipeline.stop()
        except Exception:
            pass


# ============================================================
# Model
# ============================================================

def load_model(
    checkpoint,
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    print(
        "[MODEL] device =",
        device,
    )

    if device.type == "cuda":

        print(
            "[MODEL] gpu =",
            torch.cuda.get_device_name(
                0
            ),
        )

        torch.backends.cudnn.benchmark = (
            True
        )

    ckpt = torch.load(
        checkpoint,
        map_location=device,
    )

    model = UNet(
        in_channels=3,
        num_classes=3,
        base=32,
    ).to(
        device
    )

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    model.eval()

    print(
        "[MODEL] checkpoint =",
        checkpoint,
    )

    print(
        "[MODEL] epoch =",
        ckpt.get(
            "epoch",
            "unknown",
        ),
    )

    metrics = ckpt.get(
        "metrics",
        {}
    )

    if metrics:

        print(
            "[MODEL] val board =",
            metrics.get(
                "board_dice"
            ),
        )

        print(
            "[MODEL] val hole =",
            metrics.get(
                "hole_dice"
            ),
        )

    return (
        model,
        device,
    )


# ============================================================
# Hole geometry
# ============================================================

def extract_hole_centers(
    mask,
    min_area,
    max_area,
):

    binary = (
        mask == 2
    ).astype(
        np.uint8
    )

    n, labels, stats, centers = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    holes = []

    for i in range(
        1,
        n,
    ):

        area = int(
            stats[
                i,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < min_area:
            continue

        if (
            max_area > 0
            and
            area > max_area
        ):
            continue

        cx = float(
            centers[i][0]
        )

        cy = float(
            centers[i][1]
        )

        x = int(
            stats[
                i,
                cv2.CC_STAT_LEFT,
            ]
        )

        y = int(
            stats[
                i,
                cv2.CC_STAT_TOP,
            ]
        )

        w = int(
            stats[
                i,
                cv2.CC_STAT_WIDTH,
            ]
        )

        h = int(
            stats[
                i,
                cv2.CC_STAT_HEIGHT,
            ]
        )

        holes.append({
            "center_uv":
                (
                    cx,
                    cy,
                ),

            "area":
                area,

            "bbox":
                (
                    x,
                    y,
                    w,
                    h,
                ),
        })

    return holes


# ============================================================
# Visualization
# ============================================================

def render_prediction(
    image,
    pred,
    camera_name,
    min_area,
    max_area,
    fps,
):

    overlay = image.copy()

    # Board surface.
    overlay[
        pred == 1
    ] = (
        80,
        180,
        80,
    )

    # Hole interior.
    overlay[
        pred == 2
    ] = (
        0,
        180,
        255,
    )

    shown = cv2.addWeighted(
        image,
        0.62,
        overlay,
        0.38,
        0,
    )

    holes = extract_hole_centers(
        pred,
        min_area=min_area,
        max_area=max_area,
    )

    # No hole IDs: only center crosses and contours/bboxes.
    for hole in holes:

        cx, cy = hole[
            "center_uv"
        ]

        x, y, w, h = hole[
            "bbox"
        ]

        c = (
            int(round(cx)),
            int(round(cy)),
        )

        cv2.drawMarker(
            shown,
            c,
            (
                0,
                0,
                255,
            ),
            markerType=cv2.MARKER_CROSS,
            markerSize=16,
            thickness=2,
        )

        cv2.rectangle(
            shown,
            (
                x,
                y,
            ),
            (
                x + w,
                y + h,
            ),
            (
                255,
                255,
                255,
            ),
            1,
        )

    cv2.rectangle(
        shown,
        (0, 0),
        (
            shown.shape[1],
            42,
        ),
        (
            0,
            0,
            0,
        ),
        -1,
    )

    text = (
        f"{camera_name}  "
        f"holes={len(holes)}  "
        f"FPS={fps:.1f}"
    )

    cv2.putText(
        shown,
        text,
        (
            10,
            28,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (
            255,
            255,
            255,
        ),
        2,
        cv2.LINE_AA,
    )

    return (
        shown,
        holes,
    )


def make_display_tile(
    image,
    width=640,
    height=400,
):

    if image is None:

        return np.zeros(
            (
                height,
                width,
                3,
            ),
            dtype=np.uint8,
        )

    return cv2.resize(
        image,
        (
            width,
            height,
        ),
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default=CHECKPOINT_DEFAULT,
    )

    parser.add_argument(
        "--tile",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=384,
    )

    parser.add_argument(
        "--min-hole-area",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--max-hole-area",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    checkpoint = Path(
        args.checkpoint
    )

    if not checkpoint.exists():

        raise FileNotFoundError(
            checkpoint
        )

    model, device = load_model(
        checkpoint
    )

    external = None
    right = None
    left = None

    try:

        external = ExternalCamera()

        right = OrbbecCamera(
            RIGHT_WRIST_SERIAL,
            "right_wrist",
        )

        left = OrbbecCamera(
            LEFT_WRIST_SERIAL,
            "left_wrist",
        )

        print()
        print(
            "=============================================="
        )

        print(
            "LIVE HOLE BOARD U-NET V1"
        )

        print(
            "Q / ESC : quit"
        )

        print(
            "Green    : predicted board"
        )

        print(
            "Orange   : predicted hole"
        )

        print(
            "Red cross: connected-component center"
        )

        print(
            "=============================================="
        )

        frame_index = 0

        last_time = time.time()

        fps_smooth = 0.0

        while True:

            frame_index += 1

            ext_frame = external.read()
            rw_frame = right.read()
            lw_frame = left.read()

            frames = {
                "external":
                    ext_frame,

                "right_wrist":
                    rw_frame,

                "left_wrist":
                    lw_frame,
            }

            rendered = {}

            counts = {}

            infer_start = time.time()

            for camera, frame in frames.items():

                if frame is None:

                    blank = np.zeros(
                        (
                            400,
                            640,
                            3,
                        ),
                        dtype=np.uint8,
                    )

                    cv2.putText(
                        blank,
                        f"{camera}: NO FRAME",
                        (
                            30,
                            200,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (
                            255,
                            255,
                            255,
                        ),
                        2,
                    )

                    rendered[
                        camera
                    ] = blank

                    counts[
                        camera
                    ] = 0

                    continue

                pred = predict_sliding(
                    model,
                    frame,
                    device,
                    tile=args.tile,
                    stride=args.stride,
                )

                shown, holes = (
                    render_prediction(
                        frame,
                        pred,
                        camera,
                        args.min_hole_area,
                        args.max_hole_area,
                        fps_smooth,
                    )
                )

                rendered[
                    camera
                ] = shown

                counts[
                    camera
                ] = len(
                    holes
                )

            infer_dt = (
                time.time()
                -
                infer_start
            )

            fps_now = (
                1.0 / infer_dt
                if infer_dt > 0
                else 0.0
            )

            if fps_smooth <= 0:
                fps_smooth = fps_now
            else:
                fps_smooth = (
                    0.90
                    *
                    fps_smooth
                    +
                    0.10
                    *
                    fps_now
                )

            ext_tile = make_display_tile(
                rendered[
                    "external"
                ]
            )

            rw_tile = make_display_tile(
                rendered[
                    "right_wrist"
                ]
            )

            lw_tile = make_display_tile(
                rendered[
                    "left_wrist"
                ]
            )

            info = np.zeros(
                (
                    400,
                    640,
                    3,
                ),
                dtype=np.uint8,
            )

            lines = [
                "R1-A7 Hole U-Net V1",
                "",
                f"Live inference: {fps_smooth:.2f} FPS",
                "",
                (
                    "External holes: "
                    f"{counts.get('external', 0)}"
                ),
                (
                    "Right wrist holes: "
                    f"{counts.get('right_wrist', 0)}"
                ),
                (
                    "Left wrist holes: "
                    f"{counts.get('left_wrist', 0)}"
                ),
                "",
                "Q / ESC = quit",
                "",
                "VISION ONLY",
                "NO ROBOT COMMANDS",
            ]

            y = 42

            for line in lines:

                cv2.putText(
                    info,
                    line,
                    (
                        25,
                        y,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (
                        255,
                        255,
                        255,
                    ),
                    2,
                    cv2.LINE_AA,
                )

                y += 30

            top = np.hstack(
                [
                    ext_tile,
                    rw_tile,
                ]
            )

            bottom = np.hstack(
                [
                    lw_tile,
                    info,
                ]
            )

            canvas = np.vstack(
                [
                    top,
                    bottom,
                ]
            )

            cv2.imshow(
                "R1-A7 Live Hole Board U-Net V1",
                canvas,
            )

            if (
                args.print_every > 0
                and
                frame_index
                %
                args.print_every
                ==
                0
            ):

                print(
                    "[LIVE]",
                    f"fps={fps_smooth:.2f}",
                    "external=",
                    counts.get(
                        "external",
                        0,
                    ),
                    "right_wrist=",
                    counts.get(
                        "right_wrist",
                        0,
                    ),
                    "left_wrist=",
                    counts.get(
                        "left_wrist",
                        0,
                    ),
                )

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            if key in (
                ord("q"),
                27,
            ):
                break

    finally:

        print(
            "[SHUTDOWN] closing cameras"
        )

        if external is not None:
            external.close()

        if right is not None:
            right.close()

        if left is not None:
            left.close()

        cv2.destroyAllWindows()

        print(
            "[SHUTDOWN] done"
        )


if __name__ == "__main__":
    main()
