#!/usr/bin/env python3

from pathlib import Path
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

from overfit_wire_unet import UNet

from eval_wire_threshold_sweep import (
    tiled_logits,
    threshold_prediction,
)

from eval_wire_main_cluster_v2 import (
    keep_main_cluster,
)

from eval_wire_skeleton_endpoints_v2 import (
    skeletonize,
)

from eval_wire_connected_path_v3 import (
    connect_skeleton_components,
    longest_geodesic_path,
)


# ============================================================
# CONFIG
# ============================================================

EXTERNAL_DEV = (
    "/dev/v4l/by-id/"
    "usb-ALP_USB_ZOOM_Camera_20685204b12d5283-video-index0"
)

RIGHT_SERIAL = "CPCBC53000C5"
LEFT_SERIAL = "CPCBC530007Z"

CHECKPOINT = Path(
    "/home/robot/r1a7_datasets/wire_unet/pilot_001/"
    "runs/unet_train_val_v2_resume/best.pt"
)

# Frozen segmentation / geometry parameters.
WIRE_THRESHOLD = 0.86
CLUSTER_BRIDGE = 24
COMPONENT_MAX_GAP = 50.0

TILE = 512
STRIDE = 384

EXT_W = 1280
EXT_H = 720
EXT_FPS = 30

ORB_W = 1280
ORB_H = 800
ORB_FPS = 30

DISPLAY_W = 560
DISPLAY_H = 350

WINDOW_NAME = "R1-A7 THREE CAMERA LIVE WIRE DETECTION"


# ============================================================
# EXTERNAL USB CAMERA
# ============================================================

def open_external():

    cap = cv2.VideoCapture(
        EXTERNAL_DEV,
        cv2.CAP_V4L2,
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open external camera: {EXTERNAL_DEV}"
        )

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"),
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        EXT_W,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        EXT_H,
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        EXT_FPS,
    )

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1,
    )

    print(
        "[External] opened:",
        EXTERNAL_DEV,
    )

    return cap


# ============================================================
# ORBBEC
# ============================================================

def get_device_by_serial(
    ctx,
    serial,
):

    devices = ctx.query_devices()

    count = devices.get_count()

    for i in range(count):

        dev = devices.get_device_by_index(i)
        info = dev.get_device_info()

        sn = info.get_serial_number()

        if sn == serial:
            return dev

    raise RuntimeError(
        f"Orbbec serial not found: {serial}"
    )


def open_orbbec(
    ctx,
    serial,
):

    device = get_device_by_serial(
        ctx,
        serial,
    )

    pipeline = Pipeline(
        device
    )

    config = Config()

    profiles = (
        pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        )
    )

    profile = (
        profiles.get_video_stream_profile(
            ORB_W,
            ORB_H,
            OBFormat.RGB,
            ORB_FPS,
        )
    )

    config.enable_stream(
        profile
    )

    pipeline.start(
        config
    )

    print(
        f"[Orbbec] started "
        f"{serial} "
        f"{ORB_W}x{ORB_H}@{ORB_FPS}"
    )

    return pipeline


def read_orbbec(
    pipeline,
):

    frames = pipeline.wait_for_frames(
        100
    )

    if frames is None:
        return None

    color = frames.get_color_frame()

    if color is None:
        return None

    data = np.frombuffer(
        color.get_data(),
        dtype=np.uint8,
    )

    expected = (
        ORB_W
        * ORB_H
        * 3
    )

    if data.size != expected:

        print(
            "[WARN] Orbbec RGB size:",
            data.size,
            "expected:",
            expected,
        )

        return None

    rgb = data.reshape(
        ORB_H,
        ORB_W,
        3,
    )

    return cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )


# ============================================================
# MODEL
# ============================================================

def load_model():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "[Model] device =",
        device,
    )

    model = UNet(
        in_channels=3,
        num_classes=3,
        base=32,
    ).to(device)

    ckpt = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    model.load_state_dict(
        ckpt["model"]
    )

    model.eval()

    print(
        "[Model] checkpoint =",
        CHECKPOINT,
    )

    return model, device


# ============================================================
# WIRE GEOMETRY
# ============================================================

def empty_detection(
    shape,
    tool,
):

    z = np.zeros(
        shape,
        dtype=bool,
    )

    return {
        "wire_mask": z,
        "skeleton": z,
        "connected": z,
        "wire_path": z,
        "tool_mask": tool,
        "connect_ok": False,
        "components": 0,
        "bridge_count": 0,
        "path_length_px": 0.0,
    }


@torch.inference_mode()
def detect_objects(
    model,
    device,
    image,
):

    # --------------------------------------------------------
    # U-Net
    # --------------------------------------------------------

    logits = tiled_logits(
        model,
        image,
        device,
        tile=TILE,
        stride=STRIDE,
    )

    # Keep the actual live wire probability for stability diagnosis.
    wire_probability = torch.softmax(
        logits,
        dim=0,
    )[1].detach().cpu().numpy()

    pred = threshold_prediction(
        logits,
        WIRE_THRESHOLD,
    )

    if torch.is_tensor(pred):
        pred = (
            pred
            .detach()
            .cpu()
            .numpy()
        )

    pred = np.asarray(pred)

    wire = (
        pred == 1
    )

    tool = (
        pred == 2
    )

    # --------------------------------------------------------
    # Main wire cluster
    #
    # bridge=24 is used only to determine which wire pieces
    # belong to the principal cluster.
    # Original pixels are preserved.
    # --------------------------------------------------------

    if not np.any(wire):
        return empty_detection(
            wire.shape,
            tool,
        )

    wire_main = keep_main_cluster(
        wire,
        CLUSTER_BRIDGE,
    )

    wire_main = np.asarray(
        wire_main,
        dtype=bool,
    )

    if not np.any(wire_main):
        return empty_detection(
            wire.shape,
            tool,
        )

    # --------------------------------------------------------
    # Skeletonization
    # --------------------------------------------------------

    skel = skeletonize(
        wire_main
    )

    skel = np.asarray(
        skel,
        dtype=bool,
    )

    if not np.any(skel):

        result = empty_detection(
            wire.shape,
            tool,
        )

        result["wire_mask"] = (
            wire_main
        )

        return result

    # --------------------------------------------------------
    # Connect broken skeleton components.
    #
    # This is the previously validated V3/V4 operation.
    # It only connects gaps <= COMPONENT_MAX_GAP.
    # --------------------------------------------------------

    try:

        (
            connected,
            bridges,
            connect_ok,
            components,
        ) = connect_skeleton_components(
            skel,
            COMPONENT_MAX_GAP,
        )

    except Exception as e:

        print(
            "[WARN] skeleton connection failed:",
            repr(e),
        )

        connected = skel.copy()
        bridges = []
        connect_ok = False
        components = -1

    connected = np.asarray(
        connected,
        dtype=bool,
    )

    # --------------------------------------------------------
    # Longest geodesic center path
    # --------------------------------------------------------

    path_mask = np.zeros_like(
        wire_main,
        dtype=bool,
    )

    path_length = 0.0

    if np.any(connected):

        try:

            (
                path,
                endpoint_a,
                endpoint_b,
                path_length,
            ) = longest_geodesic_path(
                connected
            )

            if path is not None:

                for p in path:

                    y = int(p[0])
                    x = int(p[1])

                    if (
                        0 <= y < path_mask.shape[0]
                        and
                        0 <= x < path_mask.shape[1]
                    ):
                        path_mask[
                            y,
                            x,
                        ] = True

        except Exception as e:

            print(
                "[WARN] longest path failed:",
                repr(e),
            )

            path_length = 0.0

    try:
        bridge_count = len(
            bridges
        )
    except Exception:
        bridge_count = 0

    try:
        component_count = int(
            components
        )
    except Exception:
        try:
            component_count = len(
                components
            )
        except Exception:
            component_count = -1

    return {
        "wire_mask":
            wire_main,

        "skeleton":
            skel,

        "connected":
            connected,

        "wire_path":
            path_mask,

        "tool_mask":
            tool,

        "wire_probability":
            wire_probability,

        "connect_ok":
            bool(connect_ok),

        "components":
            component_count,

        "bridge_count":
            int(bridge_count),

        "path_length_px":
            float(path_length),
    }


# ============================================================
# DISPLAY
# ============================================================

def dilate_mask(
    mask,
    size,
):

    if not np.any(mask):
        return mask

    return cv2.dilate(
        mask.astype(
            np.uint8
        ),
        np.ones(
            (size, size),
            dtype=np.uint8,
        ),
        iterations=1,
    ).astype(bool)


def make_display(
    image,
    det,
    name,
    fps,
):

    wire = det[
        "wire_mask"
    ]

    path = det[
        "wire_path"
    ]

    tool = det[
        "tool_mask"
    ]

    # Display-only thickness.
    wire_vis = dilate_mask(
        wire,
        3,
    )

    path_vis = dilate_mask(
        path,
        5,
    )

    tool_vis = dilate_mask(
        tool,
        3,
    )

    overlay = image.copy()

    # --------------------------------------------------------
    # Tool = blue
    # --------------------------------------------------------

    overlay[
        tool_vis
    ] = (
        255,
        0,
        0,
    )

    # --------------------------------------------------------
    # Raw/main wire mask = pink/magenta
    # --------------------------------------------------------

    overlay[
        wire_vis
    ] = (
        255,
        80,
        255,
    )

    # --------------------------------------------------------
    # Final connected center path = bright red
    #
    # Draw last so it remains visible above the raw mask.
    # --------------------------------------------------------

    overlay[
        path_vis
    ] = (
        0,
        0,
        255,
    )

    vis = cv2.addWeighted(
        image,
        0.68,
        overlay,
        0.32,
        0,
    )

    wire_pixels = int(
        np.count_nonzero(
            wire
        )
    )

    tool_pixels = int(
        np.count_nonzero(
            tool
        )
    )

    path_pixels = int(
        np.count_nonzero(
            path
        )
    )

    wire_found = (
        wire_pixels > 0
    )

    tool_found = (
        tool_pixels > 0
    )

    path_found = (
        path_pixels > 0
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    line1 = (
        f"{name}  "
        f"WIRE={'YES' if wire_found else 'NO'}  "
        f"TOOL={'YES' if tool_found else 'NO'}"
    )

    line2 = (
        f"PATH={'YES' if path_found else 'NO'}  "
        f"CONNECT={'YES' if det['connect_ok'] else 'NO'}  "
        f"COMP={det['components']}  "
        f"BRIDGE={det['bridge_count']}  "
        f"LEN={det['path_length_px']:.0f}px"
    )

    line3 = (
        f"wire={wire_pixels}  "
        f"tool={tool_pixels}  "
        f"infer={fps:.2f} FPS"
    )

    cv2.rectangle(
        vis,
        (0, 0),
        (
            vis.shape[1],
            86,
        ),
        (
            0,
            0,
            0,
        ),
        -1,
    )

    cv2.putText(
        vis,
        line1,
        (14, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (
            255,
            255,
            255,
        ),
        2,
        cv2.LINE_AA,
    )

    connect_color = (
        (0, 255, 0)
        if det["connect_ok"]
        else (0, 180, 255)
    )

    cv2.putText(
        vis,
        line2,
        (14, 51),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        connect_color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        vis,
        line3,
        (14, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (
            210,
            210,
            210,
        ),
        1,
        cv2.LINE_AA,
    )

    return vis


def resize_panel(
    image,
):

    return cv2.resize(
        image,
        (
            DISPLAY_W,
            DISPLAY_H,
        ),
        interpolation=cv2.INTER_AREA,
    )


def run_one(
    model,
    device,
    frame,
):

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    det = detect_objects(
        model,
        device,
        frame,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    dt = (
        time.perf_counter()
        - t0
    )

    fps = (
        1.0 / dt
        if dt > 0
        else 0.0
    )

    return det, fps


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "R1-A7 THREE-CAMERA LIVE WIRE + TOOL + CONNECTED PATH"
    )
    print("=" * 78)

    print(
        "WIRE threshold      =",
        WIRE_THRESHOLD,
    )

    print(
        "main cluster bridge =",
        CLUSTER_BRIDGE,
    )

    print(
        "skeleton max gap    =",
        COMPONENT_MAX_GAP,
    )

    print()
    print(
        "DISPLAY:"
    )

    print(
        "  PINK = raw/main U-Net wire mask"
    )

    print(
        "  RED  = connected longest center path"
    )

    print(
        "  BLUE = tool/tweezer"
    )

    print()
    print(
        "NO DDS / NO LowCmd / NO ROBOT CONTROL"
    )

    print(
        "Q / ESC / Ctrl+C = quit"
    )

    print("=" * 78)

    model, device = (
        load_model()
    )

    external = None
    right = None
    left = None

    try:

        # ----------------------------------------------------
        # Cameras
        # ----------------------------------------------------

        external = (
            open_external()
        )

        ctx = Context()

        right = open_orbbec(
            ctx,
            RIGHT_SERIAL,
        )

        left = open_orbbec(
            ctx,
            LEFT_SERIAL,
        )

        # Camera exposure stabilization.
        time.sleep(
            1.0
        )

        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL,
        )

        loop_counter = 0
        loop_t0 = time.perf_counter()
        loop_fps = 0.0

        while True:

            # ------------------------------------------------
            # Capture
            # ------------------------------------------------

            ok_ext, frame_ext = (
                external.read()
            )

            frame_right = (
                read_orbbec(
                    right
                )
            )

            frame_left = (
                read_orbbec(
                    left
                )
            )

            if not ok_ext:

                print(
                    "[WARN] external frame missing"
                )

            if frame_right is None:

                print(
                    "[WARN] right wrist frame missing"
                )

            if frame_left is None:

                print(
                    "[WARN] left wrist frame missing"
                )

            if (
                not ok_ext
                or frame_right is None
                or frame_left is None
            ):

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key in (
                    ord("q"),
                    27,
                ):
                    break

                continue

            # ------------------------------------------------
            # Inference
            # ------------------------------------------------

            det_ext, fps_ext = (
                run_one(
                    model,
                    device,
                    frame_ext,
                )
            )

            det_right, fps_right = (
                run_one(
                    model,
                    device,
                    frame_right,
                )
            )

            det_left, fps_left = (
                run_one(
                    model,
                    device,
                    frame_left,
                )
            )

            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            vis_ext = make_display(
                frame_ext,
                det_ext,
                "EXTERNAL",
                fps_ext,
            )

            vis_right = make_display(
                frame_right,
                det_right,
                "RIGHT WRIST",
                fps_right,
            )

            vis_left = make_display(
                frame_left,
                det_left,
                "LEFT WRIST",
                fps_left,
            )

            panel = np.hstack(
                [
                    resize_panel(
                        vis_ext
                    ),
                    resize_panel(
                        vis_right
                    ),
                    resize_panel(
                        vis_left
                    ),
                ]
            )

            # ------------------------------------------------
            # Overall loop FPS
            # ------------------------------------------------

            loop_counter += 1

            elapsed = (
                time.perf_counter()
                - loop_t0
            )

            if elapsed >= 1.0:

                loop_fps = (
                    loop_counter
                    / elapsed
                )

                loop_counter = 0
                loop_t0 = (
                    time.perf_counter()
                )

            cv2.putText(
                panel,
                f"3-CAMERA LOOP = {loop_fps:.2f} FPS",
                (
                    15,
                    panel.shape[0] - 13,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (
                    255,
                    255,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                WINDOW_NAME,
                panel,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):
                break

    except KeyboardInterrupt:

        print(
            "\n[INFO] Ctrl+C"
        )

    finally:

        print(
            "\n[INFO] stopping cameras..."
        )

        if external is not None:

            try:
                external.release()
            except Exception:
                pass

        if right is not None:

            try:
                right.stop()
            except Exception:
                pass

        if left is not None:

            try:
                left.stop()
            except Exception:
                pass

        cv2.destroyAllWindows()

        print(
            "[INFO] stopped"
        )


if __name__ == "__main__":
    main()
