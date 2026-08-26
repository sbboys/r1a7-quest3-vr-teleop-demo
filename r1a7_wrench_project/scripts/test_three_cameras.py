#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R1-A7 three-camera test viewer

Camera mapping:
    Head:
        Intel RealSense D435i
        SN = 109622072668

    Left wrist:
        Orbbec Gemini 336L
        SN = CPCBC530007Z

    Right wrist:
        Orbbec Gemini 336L
        SN = CPCBC53000C5

Current test mode:
    - RGB only
    - 3 cameras
    - aspect-ratio-preserving preview
    - three-column display
    - sequential camera startup

Important:
    Preview scaling is ONLY for visualization.
    The original frame stored in LatestFrame is never resized.
"""

import sys
import threading
import time
import traceback

import cv2
import numpy as np
import pyrealsense2 as rs

from pyorbbecsdk import (
    Config,
    Context,
    OBFormat,
    OBSensorType,
    Pipeline,
)


# ============================================================
# Camera serial numbers
# ============================================================

HEAD_SN = "109622072668"

LEFT_WRIST_SN = "CPCBC530007Z"

RIGHT_WRIST_SN = "CPCBC53000C5"


# ============================================================
# RealSense configuration
# ============================================================

HEAD_COLOR_WIDTH = 1280
HEAD_COLOR_HEIGHT = 720
HEAD_COLOR_FPS = 30


# ============================================================
# Preview configuration
#
# This is only the DISPLAY canvas size.
# It does NOT change the original camera frame.
#
# 600 x 400 * 3 = 1800 x 400 total window
# Suitable for a 1920-wide monitor.
# ============================================================

PREVIEW_WIDTH = 600
PREVIEW_HEIGHT = 400


# ============================================================
# Startup timeout
# ============================================================

CAMERA_START_TIMEOUT = 10.0

CAMERA_START_INTERVAL = 1.0


# ============================================================
# Latest frame buffer
# ============================================================

class LatestFrame:
    """
    Thread-safe latest-frame buffer.

    Original camera image is stored here.
    No preview resize is performed in this class.
    """

    def __init__(self):
        self.lock = threading.Lock()

        self.image = None
        self.timestamp = None

        self.ready = threading.Event()

        self.frame_count = 0

    def update(self, image, timestamp=None):
        with self.lock:
            self.image = image
            self.timestamp = timestamp
            self.frame_count += 1

        # First valid frame means the camera is ready.
        self.ready.set()

    def get(self):
        with self.lock:
            if self.image is None:
                return None, None

            return self.image.copy(), self.timestamp

    def wait_ready(self, timeout=10.0):
        return self.ready.wait(timeout)

    def is_ready(self):
        return self.ready.is_set()


# ============================================================
# Orbbec image conversion
# ============================================================

def orbbec_color_to_bgr(frame):
    """
    Convert an Orbbec color frame into OpenCV BGR.

    Supports several common Gemini output formats.
    """

    if frame is None:
        return None

    width = frame.get_width()
    height = frame.get_height()
    fmt = frame.get_format()

    raw_data = np.asanyarray(frame.get_data())

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    if fmt == OBFormat.RGB:
        image = raw_data.reshape(
            height,
            width,
            3,
        )

        return cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR,
        )

    # --------------------------------------------------------
    # BGR
    # --------------------------------------------------------

    if fmt == OBFormat.BGR:
        return raw_data.reshape(
            height,
            width,
            3,
        )

    # --------------------------------------------------------
    # MJPEG
    # --------------------------------------------------------

    if fmt == OBFormat.MJPG:
        image = cv2.imdecode(
            raw_data,
            cv2.IMREAD_COLOR,
        )

        return image

    # --------------------------------------------------------
    # YUYV
    # --------------------------------------------------------

    if fmt == OBFormat.YUYV:
        image = raw_data.reshape(
            height,
            width,
            2,
        )

        return cv2.cvtColor(
            image,
            cv2.COLOR_YUV2BGR_YUY2,
        )

    # --------------------------------------------------------
    # UYVY
    # --------------------------------------------------------

    if fmt == OBFormat.UYVY:
        image = raw_data.reshape(
            height,
            width,
            2,
        )

        return cv2.cvtColor(
            image,
            cv2.COLOR_YUV2BGR_UYVY,
        )

    print(
        "[ORBBEC WARNING] "
        f"Unsupported color format: {fmt}"
    )

    return None


# ============================================================
# RealSense camera thread
# ============================================================

class RealSenseCamera(threading.Thread):

    def __init__(
        self,
        serial,
        output_buffer,
    ):
        super().__init__(
            daemon=True
        )

        self.serial = serial
        self.output = output_buffer

        self.running = True

        self.pipeline = None

        self.error = None

    def run(self):

        print()
        print(
            f"[HEAD] Searching D435i: "
            f"{self.serial}"
        )

        try:
            # ------------------------------------------------
            # Enumerate RealSense devices first
            # ------------------------------------------------

            ctx = rs.context()

            devices = ctx.query_devices()

            print(
                f"[HEAD] RealSense device count: "
                f"{len(devices)}"
            )

            found = False

            for dev in devices:

                try:
                    name = dev.get_info(
                        rs.camera_info.name
                    )

                    sn = dev.get_info(
                        rs.camera_info.serial_number
                    )

                    print(
                        "[HEAD] Found RealSense: "
                        f"{name}, SN={sn}"
                    )

                    if sn == self.serial:
                        found = True

                except Exception as exc:
                    print(
                        "[HEAD WARNING] "
                        f"Unable to read device info: {exc}"
                    )

            if not found:
                raise RuntimeError(
                    "Requested D435i not found: "
                    f"SN={self.serial}"
                )

            # ------------------------------------------------
            # Build pipeline
            # ------------------------------------------------

            self.pipeline = rs.pipeline()

            config = rs.config()

            config.enable_device(
                self.serial
            )

            config.enable_stream(
                rs.stream.color,
                HEAD_COLOR_WIDTH,
                HEAD_COLOR_HEIGHT,
                rs.format.bgr8,
                HEAD_COLOR_FPS,
            )

            print(
                "[HEAD] Starting D435i: "
                f"{self.serial}"
            )

            self.pipeline.start(
                config
            )

            print(
                "[HEAD] D435i pipeline started"
            )

            # ------------------------------------------------
            # Capture loop
            # ------------------------------------------------

            while self.running:

                try:
                    frames = (
                        self.pipeline.wait_for_frames(
                            1000
                        )
                    )

                except RuntimeError as exc:

                    if self.running:
                        print(
                            "[HEAD WARNING] "
                            f"Frame timeout: {exc}"
                        )

                    continue

                color_frame = (
                    frames.get_color_frame()
                )

                if not color_frame:
                    continue

                image = np.asanyarray(
                    color_frame.get_data()
                )

                timestamp = (
                    color_frame.get_timestamp()
                )

                self.output.update(
                    image,
                    timestamp,
                )

        except Exception as exc:

            self.error = exc

            print()
            print(
                "[HEAD ERROR]",
                repr(exc),
            )

            traceback.print_exc()

        finally:

            if self.pipeline is not None:

                try:
                    self.pipeline.stop()

                except Exception:
                    pass

            print(
                "[HEAD] stopped"
            )

    def stop(self):
        self.running = False


# ============================================================
# Orbbec camera thread
# ============================================================

class OrbbecCamera(threading.Thread):

    def __init__(
        self,
        device,
        serial,
        name,
        output_buffer,
    ):
        super().__init__(
            daemon=True
        )

        self.device = device

        self.serial = serial

        self.name = name

        self.output = output_buffer

        self.running = True

        self.pipeline = None

        self.config = None

        self.error = None

    def run(self):

        print()
        print(
            f"[{self.name}] "
            "Starting Gemini 336L: "
            f"{self.serial}"
        )

        try:

            # ------------------------------------------------
            # Pipeline
            # ------------------------------------------------

            self.pipeline = Pipeline(
                self.device
            )

            self.config = Config()

            # ------------------------------------------------
            # Get color profiles
            #
            # width=0
            # height=0
            # fps=0
            #
            # means SDK selects a compatible/default profile.
            #
            # First test RGB only.
            # ------------------------------------------------

            profile_list = (
                self.pipeline.get_stream_profile_list(
                    OBSensorType.COLOR_SENSOR
                )
            )

            color_profile = (
                profile_list.get_video_stream_profile(
                    0,
                    0,
                    OBFormat.RGB,
                    0,
                )
            )

            self.config.enable_stream(
                color_profile
            )

            # ------------------------------------------------
            # Start camera
            # ------------------------------------------------

            self.pipeline.start(
                self.config
            )

            print(
                f"[{self.name}] "
                "Gemini pipeline started"
            )

            # ------------------------------------------------
            # Capture loop
            # ------------------------------------------------

            while self.running:

                try:

                    frames = (
                        self.pipeline.wait_for_frames(
                            1000
                        )
                    )

                except Exception as exc:

                    if self.running:
                        print(
                            f"[{self.name} WARNING] "
                            f"Frame timeout: {exc}"
                        )

                    continue

                if frames is None:
                    continue

                color_frame = (
                    frames.get_color_frame()
                )

                if color_frame is None:
                    continue

                image = orbbec_color_to_bgr(
                    color_frame
                )

                if image is None:
                    continue

                try:

                    timestamp = (
                        color_frame.get_timestamp_us()
                    )

                except Exception:

                    timestamp = time.time()

                self.output.update(
                    image,
                    timestamp,
                )

        except Exception as exc:

            self.error = exc

            print()
            print(
                f"[{self.name} ERROR]",
                repr(exc),
            )

            traceback.print_exc()

        finally:

            if self.pipeline is not None:

                try:
                    self.pipeline.stop()

                except Exception:
                    pass

            print(
                f"[{self.name}] stopped"
            )

    def stop(self):
        self.running = False


# ============================================================
# Find Orbbec devices by serial number
# ============================================================

def find_orbbec_devices():

    context = Context()

    device_list = (
        context.query_devices()
    )

    device_count = (
        device_list.get_count()
    )

    print(
        "Orbbec device count:",
        device_count,
    )

    found = {}

    for i in range(device_count):

        device = (
            device_list.get_device_by_index(
                i
            )
        )

        info = (
            device.get_device_info()
        )

        serial = (
            info.get_serial_number()
        )

        print(
            f"Orbbec #{i}: "
            f"SN={serial}"
        )

        found[serial] = device

    return context, found


# ============================================================
# Aspect-ratio-preserving preview
# ============================================================

def make_preview(
    image,
    title,
    canvas_width=PREVIEW_WIDTH,
    canvas_height=PREVIEW_HEIGHT,
):
    """
    Create preview while preserving original aspect ratio.

    IMPORTANT:
        This function ONLY creates the GUI preview.

        It does NOT modify the original image stored in
        LatestFrame.

    If source aspect ratio does not match preview canvas,
    black bars are added rather than stretching the image.
    """

    canvas = np.zeros(
        (
            canvas_height,
            canvas_width,
            3,
        ),
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # No frame yet
    # --------------------------------------------------------

    if image is None:

        cv2.putText(
            canvas,
            title,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            "WAITING FOR FRAME...",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return canvas

    # --------------------------------------------------------
    # Read original dimensions
    # --------------------------------------------------------

    source_height, source_width = (
        image.shape[:2]
    )

    # --------------------------------------------------------
    # Calculate uniform scale
    #
    # NEVER independently stretch X and Y.
    # --------------------------------------------------------

    scale = min(
        canvas_width / source_width,
        canvas_height / source_height,
    )

    new_width = max(
        1,
        int(
            round(
                source_width * scale
            )
        ),
    )

    new_height = max(
        1,
        int(
            round(
                source_height * scale
            )
        ),
    )

    # --------------------------------------------------------
    # Resize preview with correct aspect ratio
    # --------------------------------------------------------

    if scale < 1.0:
        interpolation = (
            cv2.INTER_AREA
        )
    else:
        interpolation = (
            cv2.INTER_LINEAR
        )

    resized = cv2.resize(
        image,
        (
            new_width,
            new_height,
        ),
        interpolation=interpolation,
    )

    # --------------------------------------------------------
    # Center image inside black canvas
    # --------------------------------------------------------

    x0 = (
        canvas_width - new_width
    ) // 2

    y0 = (
        canvas_height - new_height
    ) // 2

    canvas[
        y0:y0 + new_height,
        x0:x0 + new_width,
    ] = resized

    # --------------------------------------------------------
    # Camera title
    # --------------------------------------------------------

    cv2.rectangle(
        canvas,
        (0, 0),
        (
            canvas_width,
            48,
        ),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        canvas,
        title,
        (15, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Show ORIGINAL source resolution
    # --------------------------------------------------------

    resolution_text = (
        f"source: "
        f"{source_width}x{source_height}"
    )

    cv2.putText(
        canvas,
        resolution_text,
        (
            15,
            canvas_height - 15,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return canvas


# ============================================================
# Camera startup helper
# ============================================================

def start_camera_and_wait(
    camera,
    frame_buffer,
    camera_name,
    timeout=CAMERA_START_TIMEOUT,
):
    """
    Start one camera and wait until its first frame arrives.

    This prevents all three RGB-D cameras from initializing
    simultaneously on the same USB hub.
    """

    print()
    print(
        f"[START] {camera_name}"
    )

    camera.start()

    ready = frame_buffer.wait_ready(
        timeout
    )

    if not ready:

        if camera.error is not None:

            raise RuntimeError(
                f"{camera_name} failed: "
                f"{camera.error}"
            )

        raise RuntimeError(
            f"{camera_name} did not produce "
            f"a frame within {timeout:.1f}s"
        )

    print(
        f"[OK] {camera_name} ready"
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Frame buffers
    # --------------------------------------------------------

    head_buffer = LatestFrame()

    left_buffer = LatestFrame()

    right_buffer = LatestFrame()

    # --------------------------------------------------------
    # Find both Orbbec cameras
    # --------------------------------------------------------

    orbbec_context, devices = (
        find_orbbec_devices()
    )

    # Keep context alive for whole program lifetime.
    _ = orbbec_context

    # --------------------------------------------------------
    # Verify both wrist cameras exist
    # --------------------------------------------------------

    if LEFT_WRIST_SN not in devices:

        raise RuntimeError(
            "Left wrist Gemini 336L "
            "not found.\n"
            f"Expected SN: {LEFT_WRIST_SN}"
        )

    if RIGHT_WRIST_SN not in devices:

        raise RuntimeError(
            "Right wrist Gemini 336L "
            "not found.\n"
            f"Expected SN: {RIGHT_WRIST_SN}"
        )

    # --------------------------------------------------------
    # Create camera objects
    # --------------------------------------------------------

    head_camera = RealSenseCamera(
        HEAD_SN,
        head_buffer,
    )

    left_camera = OrbbecCamera(
        devices[LEFT_WRIST_SN],
        LEFT_WRIST_SN,
        "LEFT WRIST",
        left_buffer,
    )

    right_camera = OrbbecCamera(
        devices[RIGHT_WRIST_SN],
        RIGHT_WRIST_SN,
        "RIGHT WRIST",
        right_buffer,
    )

    cameras = [
        head_camera,
        left_camera,
        right_camera,
    ]

    # --------------------------------------------------------
    # Sequential startup
    # --------------------------------------------------------

    try:

        print()
        print(
            "=========================================="
        )
        print(
            " R1-A7 Three Camera Startup"
        )
        print(
            "=========================================="
        )

        # ----------------------------------------------------
        # 1. Head D435i
        # ----------------------------------------------------

        start_camera_and_wait(
            head_camera,
            head_buffer,
            "HEAD D435i",
        )

        time.sleep(
            CAMERA_START_INTERVAL
        )

        # ----------------------------------------------------
        # 2. Left wrist Gemini
        # ----------------------------------------------------

        start_camera_and_wait(
            left_camera,
            left_buffer,
            "LEFT WRIST Gemini 336L",
        )

        time.sleep(
            CAMERA_START_INTERVAL
        )

        # ----------------------------------------------------
        # 3. Right wrist Gemini
        # ----------------------------------------------------

        start_camera_and_wait(
            right_camera,
            right_buffer,
            "RIGHT WRIST Gemini 336L",
        )

        print()
        print(
            "=========================================="
        )
        print(
            " ALL THREE CAMERAS READY"
        )
        print(
            "=========================================="
        )
        print(
            f"HEAD  : D435i "
            f"SN={HEAD_SN}"
        )
        print(
            f"LEFT  : Gemini 336L "
            f"SN={LEFT_WRIST_SN}"
        )
        print(
            f"RIGHT : Gemini 336L "
            f"SN={RIGHT_WRIST_SN}"
        )
        print()
        print(
            "Press Q or ESC in the camera window "
            "to quit."
        )
        print()

        # ----------------------------------------------------
        # Create OpenCV window
        # ----------------------------------------------------

        window_name = (
            "R1-A7 Three Camera Viewer"
        )

        cv2.namedWindow(
            window_name,
            cv2.WINDOW_NORMAL,
        )

        # Initial GUI size:
        # 3 preview columns.
        cv2.resizeWindow(
            window_name,
            PREVIEW_WIDTH * 3,
            PREVIEW_HEIGHT,
        )

        # ----------------------------------------------------
        # GUI loop
        # ----------------------------------------------------

        while True:

            # ------------------------------------------------
            # Get ORIGINAL latest frames
            # ------------------------------------------------

            head_image, head_timestamp = (
                head_buffer.get()
            )

            left_image, left_timestamp = (
                left_buffer.get()
            )

            right_image, right_timestamp = (
                right_buffer.get()
            )

            # Timestamps are currently preserved for future
            # synchronization but not yet displayed/used.
            _ = (
                head_timestamp,
                left_timestamp,
                right_timestamp,
            )

            # ------------------------------------------------
            # Produce aspect-ratio-preserving previews
            # ------------------------------------------------

            head_view = make_preview(
                head_image,
                "HEAD - D435i",
            )

            left_view = make_preview(
                left_image,
                "LEFT WRIST - Gemini 336L",
            )

            right_view = make_preview(
                right_image,
                "RIGHT WRIST - Gemini 336L",
            )

            # ------------------------------------------------
            # Three-column display
            #
            # HEAD | LEFT | RIGHT
            #
            # Each preview canvas has exactly the same size.
            # The image itself keeps its own aspect ratio.
            # ------------------------------------------------

            display = np.hstack(
                (
                    head_view,
                    left_view,
                    right_view,
                )
            )

            cv2.imshow(
                window_name,
                display,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                print(
                    "[INFO] Q pressed, exiting..."
                )
                break

            if key == 27:
                print(
                    "[INFO] ESC pressed, exiting..."
                )
                break

    except KeyboardInterrupt:

        print()
        print(
            "[INFO] Ctrl+C received, exiting..."
        )

    except Exception as exc:

        print()
        print(
            "[MAIN ERROR]",
            repr(exc),
        )

        traceback.print_exc()

    finally:

        # ----------------------------------------------------
        # Stop all cameras
        # ----------------------------------------------------

        print()
        print(
            "[INFO] Stopping cameras..."
        )

        for camera in cameras:
            camera.stop()

        # ----------------------------------------------------
        # Wait for camera threads
        # ----------------------------------------------------

        for camera in cameras:

            if camera.is_alive():

                camera.join(
                    timeout=3.0
                )

        # ----------------------------------------------------
        # Close GUI
        # ----------------------------------------------------

        cv2.destroyAllWindows()

        print(
            "[INFO] All cameras stopped."
        )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    main()