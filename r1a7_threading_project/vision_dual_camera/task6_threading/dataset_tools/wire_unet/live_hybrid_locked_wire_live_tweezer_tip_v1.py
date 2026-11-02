#!/usr/bin/env python3

import time
from collections import deque

import cv2
import numpy as np

import live_wire_unet_three_camera as base
import live_wire_temporal_validation_v1 as temporal
import live_manual_free_grasp_validation_v1 as manual


# ============================================================
# CONFIG
# ============================================================

WINDOW = "R1-A7 HYBRID WIRE LOCK + LIVE TWEEZER TIP V1"

CAMERA_ORDER = (
    "external",
    "right_wrist",
    "left_wrist",
)

FIXED_WIRE_CAMERAS = {
    "external",
    "left_wrist",
}

LIVE_WIRE_CAMERA = "right_wrist"

# Pixels close to the nearest tool boundary are regarded
# as the tweezer leading-tip region.
TIP_FRONT_BAND_PX = 12.0

TIP_MIN_TOOL_PIXELS = 10
TIP_MIN_FRONT_PIXELS = 4

# Median stabilization.
TIP_HISTORY = 5

# Visualization.
TIP_COLOR = (
    255,
    0,
    255,
)  # BGR magenta


# ============================================================
# BASIC HELPERS
# ============================================================

def copy_dict_arrays(src):

    if src is None:
        return None

    out = {}

    for k, v in src.items():

        if isinstance(
            v,
            np.ndarray,
        ):
            out[k] = v.copy()

        else:
            out[k] = v

    return out


# ============================================================
# TWEEZER TIP TRACKER
# ============================================================

class TweezerTipTracker:

    def __init__(
        self,
        camera_name,
    ):

        self.camera_name = (
            camera_name
        )

        self.history = deque(
            maxlen=TIP_HISTORY
        )


    def clear(
        self,
    ):

        self.history.clear()


    def update(
        self,
        tool_mask,
        grasp_yx,
    ):

        result = {
            "valid":
                False,

            "tip_yx":
                None,

            "raw_tip_yx":
                None,

            "grasp_yx":
                grasp_yx,

            "tip_grasp_distance_px":
                None,

            "tool_pixels":
                0,

            "front_pixels":
                0,

            "status":
                "INVALID",
        }

        if grasp_yx is None:

            result[
                "status"
            ] = "NO_GRASP_TARGET"

            return result

        tool = np.asarray(
            tool_mask,
            dtype=bool,
        )

        tool_pixels = int(
            np.count_nonzero(
                tool
            )
        )

        result[
            "tool_pixels"
        ] = tool_pixels

        if (
            tool_pixels
            < TIP_MIN_TOOL_PIXELS
        ):

            result[
                "status"
            ] = "NO_TOOL"

            return result

        ys, xs = np.where(
            tool
        )

        gy = float(
            grasp_yx[0]
        )

        gx = float(
            grasp_yx[1]
        )

        dy = (
            ys.astype(
                np.float32
            )
            - gy
        )

        dx = (
            xs.astype(
                np.float32
            )
            - gx
        )

        distance = np.sqrt(
            dx * dx
            +
            dy * dy
        )

        if len(distance) == 0:

            result[
                "status"
            ] = "NO_TOOL"

            return result

        min_distance = float(
            np.min(
                distance
            )
        )

        # Tool pixels belonging to the front-most band
        # facing the current GRASP target.
        front = (
            distance
            <=
            (
                min_distance
                +
                TIP_FRONT_BAND_PX
            )
        )

        front_count = int(
            np.count_nonzero(
                front
            )
        )

        result[
            "front_pixels"
        ] = front_count

        if (
            front_count
            <
            TIP_MIN_FRONT_PIXELS
        ):

            # Fallback:
            # use a few nearest pixels.
            order = np.argsort(
                distance
            )

            n = min(
                20,
                len(order),
            )

            idx = order[:n]

            front_ys = ys[
                idx
            ]

            front_xs = xs[
                idx
            ]

        else:

            front_ys = ys[
                front
            ]

            front_xs = xs[
                front
            ]

        # Median instead of mean:
        # less sensitive to irregular mask edges.
        raw_y = float(
            np.median(
                front_ys
            )
        )

        raw_x = float(
            np.median(
                front_xs
            )
        )

        raw_tip = (
            raw_y,
            raw_x,
        )

        self.history.append(
            raw_tip
        )

        hist = np.asarray(
            self.history,
            dtype=np.float32,
        )

        smooth_y = float(
            np.median(
                hist[:, 0]
            )
        )

        smooth_x = float(
            np.median(
                hist[:, 1]
            )
        )

        smooth_tip = (
            smooth_y,
            smooth_x,
        )

        tip_grasp_distance = float(
            np.sqrt(
                (
                    smooth_x
                    - gx
                ) ** 2
                +
                (
                    smooth_y
                    - gy
                ) ** 2
            )
        )

        result.update(
            {
                "valid":
                    True,

                "tip_yx":
                    smooth_tip,

                "raw_tip_yx":
                    raw_tip,

                "tip_grasp_distance_px":
                    tip_grasp_distance,

                "status":
                    "TIP_VALID",
            }
        )

        return result


# ============================================================
# TIP DISPLAY
# ============================================================

def draw_tip(
    vis,
    tip_result,
):

    if (
        not tip_result[
            "valid"
        ]
    ):

        cv2.putText(
            vis,
            "TIP INVALID: "
            + tip_result[
                "status"
            ],
            (
                12,
                133,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (
                0,
                165,
                255,
            ),
            2,
            cv2.LINE_AA,
        )

        return vis

    tip_y, tip_x = (
        tip_result[
            "tip_yx"
        ]
    )

    grasp_y, grasp_x = (
        tip_result[
            "grasp_yx"
        ]
    )

    tx = int(
        round(
            tip_x
        )
    )

    ty = int(
        round(
            tip_y
        )
    )

    gx = int(
        round(
            grasp_x
        )
    )

    gy = int(
        round(
            grasp_y
        )
    )

    # --------------------------------------------------------
    # TIP point
    # --------------------------------------------------------

    cv2.circle(
        vis,
        (
            tx,
            ty,
        ),
        9,
        TIP_COLOR,
        3,
        cv2.LINE_AA,
    )

    cv2.circle(
        vis,
        (
            tx,
            ty,
        ),
        3,
        TIP_COLOR,
        -1,
        cv2.LINE_AA,
    )

    cv2.putText(
        vis,
        "TIP",
        (
            tx + 10,
            ty - 10,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        TIP_COLOR,
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # TIP -> GRASP line
    # --------------------------------------------------------

    cv2.line(
        vis,
        (
            tx,
            ty,
        ),
        (
            gx,
            gy,
        ),
        TIP_COLOR,
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Numeric position
    # --------------------------------------------------------

    dist = (
        tip_result[
            "tip_grasp_distance_px"
        ]
    )

    text1 = (
        f"TIP=({tx},{ty})"
    )

    text2 = (
        f"TIP->GRASP={dist:.1f}px"
    )

    cv2.putText(
        vis,
        text1,
        (
            12,
            133,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        TIP_COLOR,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        vis,
        text2,
        (
            12,
            158,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (
            255,
            255,
            255,
        ),
        1,
        cv2.LINE_AA,
    )

    return vis


def add_mode_label(
    vis,
    text,
    color,
):

    cv2.rectangle(
        vis,
        (
            0,
            vis.shape[0]
            - 32,
        ),
        (
            vis.shape[1],
            vis.shape[0],
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
        text,
        (
            12,
            vis.shape[0]
            - 9,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        2,
        cv2.LINE_AA,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 82)

    print(
        "R1-A7 HYBRID WIRE LOCK + LIVE TWEEZER TIP V1"
    )

    print("=" * 82)

    print()
    print(
        "EXTERNAL:"
    )

    print(
        "  wire/FREE/GRASP = locked after manual FREE click"
    )

    print(
        "  tool/TIP        = LIVE every frame"
    )

    print()
    print(
        "LEFT WRIST:"
    )

    print(
        "  wire/FREE/GRASP = locked after manual FREE click"
    )

    print(
        "  tool/TIP        = LIVE every frame"
    )

    print()
    print(
        "RIGHT WRIST:"
    )

    print(
        "  wire/FREE/GRASP = LIVE"
    )

    print(
        "  tool/TIP        = LIVE"
    )

    print()
    print(
        "BLUE    = live tool mask"
    )

    print(
        "MAGENTA = tweezer TIP"
    )

    print(
        "GREEN   = FREE"
    )

    print(
        "YELLOW  = GRASP region"
    )

    print(
        "WHITE   = GRASP reference point"
    )

    print()
    print(
        "LEFT CLICK = manually initialize FREE"
    )

    print(
        "P          = print all three TIP coordinates"
    )

    print(
        "C          = clear all calibration"
    )

    print(
        "Q / ESC    = quit"
    )

    print()
    print(
        "NO DDS / NO LowCmd / NO ROBOT CONTROL"
    )

    print("=" * 82)

    model, device = (
        base.load_model()
    )

    temporal_trackers = {
        camera:
            temporal.TemporalTracker(
                camera
            )
        for camera
        in CAMERA_ORDER
    }

    free_trackers = {
        camera:
            manual.ManualFreeTracker(
                camera
            )
        for camera
        in CAMERA_ORDER
    }

    tip_trackers = {
        camera:
            TweezerTipTracker(
                camera
            )
        for camera
        in CAMERA_ORDER
    }

    locked = {
        "external":
            False,

        "left_wrist":
            False,
    }

    locked_tracks = {
        "external":
            None,

        "left_wrist":
            None,
    }

    locked_geometry = {
        "external":
            None,

        "left_wrist":
            None,
    }

    latest_tracks = {
        camera:
            None
        for camera
        in CAMERA_ORDER
    }

    latest_shapes = {
        camera:
            None
        for camera
        in CAMERA_ORDER
    }

    latest_tips = {
        camera:
            None
        for camera
        in CAMERA_ORDER
    }

    # Dynamic display-panel geometry for mouse mapping.
    panel_layout = {}

    # --------------------------------------------------------
    # MOUSE
    # --------------------------------------------------------

    def on_mouse(
        event,
        x,
        y,
        flags,
        param,
    ):

        if (
            event
            !=
            cv2.EVENT_LBUTTONDOWN
        ):
            return

        selected_camera = None
        local_x = None
        local_y = None

        for camera in CAMERA_ORDER:

            layout = (
                panel_layout.get(
                    camera
                )
            )

            if layout is None:
                continue

            x0 = layout[
                "x0"
            ]

            x1 = layout[
                "x1"
            ]

            h = layout[
                "h"
            ]

            if (
                x0 <= x < x1
                and
                0 <= y < h
            ):

                selected_camera = (
                    camera
                )

                local_x = (
                    x - x0
                )

                local_y = y

                break

        if selected_camera is None:
            return

        camera = (
            selected_camera
        )

        track = (
            latest_tracks[
                camera
            ]
        )

        shape = (
            latest_shapes[
                camera
            ]
        )

        layout = (
            panel_layout[
                camera
            ]
        )

        if (
            track is None
            or
            shape is None
        ):

            print(
                f"[{camera}] "
                f"no tracked wire available"
            )

            return

        if track[
            "state"
        ] not in (
            "INIT",
            "TRACK",
            "RECOVER",
            "LOCKED",
        ):

            print(
                f"[{camera}] "
                f"click rejected, "
                f"state={track['state']}"
            )

            return

        native_h = int(
            shape[0]
        )

        native_w = int(
            shape[1]
        )

        display_w = int(
            layout[
                "w"
            ]
        )

        display_h = int(
            layout[
                "h"
            ]
        )

        original_x = int(
            np.clip(
                local_x
                * native_w
                / display_w,
                0,
                native_w - 1,
            )
        )

        original_y = int(
            np.clip(
                local_y
                * native_h
                / display_h,
                0,
                native_h - 1,
            )
        )

        tracker = (
            free_trackers[
                camera
            ]
        )

        tracker.current_shape = (
            track[
                "path"
            ].shape
        )

        ok = tracker.calibrate(
            track[
                "path"
            ],
            (
                original_y,
                original_x,
            ),
        )

        if not ok:
            return

        # ----------------------------------------------------
        # EXTERNAL / LEFT:
        # freeze ONLY wire/FREE/GRASP geometry.
        #
        # TOOL remains live.
        # ----------------------------------------------------

        if (
            camera
            in FIXED_WIRE_CAMERAS
        ):

            frozen_track = {
                "path":
                    track[
                        "path"
                    ].copy(),

                "length":
                    float(
                        track[
                            "length"
                        ]
                    ),

                "state":
                    "LOCKED",

                "support":
                    float(
                        track.get(
                            "support",
                            1.0,
                        )
                    ),

                "raw_jump":
                    0.0,

                "candidate_jump":
                    0.0,

                "hold_count":
                    0,
            }

            geom = copy_dict_arrays(
                tracker.last_geometry
            )

            geom[
                "state"
            ] = "LOCKED"

            geom[
                "status"
            ] = "LOCKED_WIRE_REFERENCE"

            geom[
                "measurement_valid"
            ] = False

            geom[
                "display_valid"
            ] = True

            locked_tracks[
                camera
            ] = frozen_track

            locked_geometry[
                camera
            ] = geom

            locked[
                camera
            ] = True

            latest_tracks[
                camera
            ] = frozen_track

            print()
            print(
                "=" * 72
            )

            print(
                f"[{camera}] "
                f"WIRE/FREE/GRASP LOCKED"
            )

            print(
                "FREE  =",
                geom[
                    "free_endpoint_yx"
                ],
            )

            print(
                "GRASP =",
                geom[
                    "grasp_point_yx"
                ],
            )

            print(
                "TOOL/TIP will remain LIVE."
            )

            print(
                "=" * 72
            )

        else:

            print()
            print(
                "=" * 72
            )

            print(
                "[right_wrist] "
                "LIVE FREE identity initialized"
            )

            print(
                "wire/FREE/GRASP/tool/TIP "
                "will all remain LIVE."
            )

            print(
                "=" * 72
            )

    # --------------------------------------------------------
    # CAMERAS
    # --------------------------------------------------------

    external = None
    right = None
    left = None

    try:

        external = (
            base.open_external()
        )

        ctx = base.Context()

        right = (
            base.open_orbbec(
                ctx,
                base.RIGHT_SERIAL,
            )
        )

        left = (
            base.open_orbbec(
                ctx,
                base.LEFT_SERIAL,
            )
        )

        time.sleep(
            1.0
        )

        cv2.namedWindow(
            WINDOW,
            cv2.WINDOW_AUTOSIZE,
        )

        cv2.setMouseCallback(
            WINDOW,
            on_mouse,
        )

        while True:

            # ------------------------------------------------
            # Capture
            # ------------------------------------------------

            ok_ext, frame_ext = (
                external.read()
            )

            frame_right = (
                base.read_orbbec(
                    right
                )
            )

            frame_left = (
                base.read_orbbec(
                    left
                )
            )

            if (
                not ok_ext
                or
                frame_right is None
                or
                frame_left is None
            ):

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

                continue

            frames = {
                "external":
                    frame_ext,

                "right_wrist":
                    frame_right,

                "left_wrist":
                    frame_left,
            }

            detections = {}
            tracks = {}
            geometries = {}
            fps_values = {}

            # ------------------------------------------------
            # IMPORTANT:
            #
            # Run U-Net on ALL THREE cameras because the
            # tool/tweezer must remain live everywhere.
            # ------------------------------------------------

            for camera in CAMERA_ORDER:

                (
                    detections[
                        camera
                    ],
                    fps_values[
                        camera
                    ],
                ) = base.run_one(
                    model,
                    device,
                    frames[
                        camera
                    ],
                )

            # ------------------------------------------------
            # External / Left
            #
            # Wire becomes frozen after calibration,
            # but detection["tool_mask"] remains LIVE.
            # ------------------------------------------------

            for camera in (
                "external",
                "left_wrist",
            ):

                if not locked[
                    camera
                ]:

                    tracks[
                        camera
                    ] = temporal_trackers[
                        camera
                    ].update(
                        detections[
                            camera
                        ]
                    )

                    geometries[
                        camera
                    ] = free_trackers[
                        camera
                    ].update(
                        tracks[
                            camera
                        ]
                    )

                    latest_tracks[
                        camera
                    ] = tracks[
                        camera
                    ]

                else:

                    tracks[
                        camera
                    ] = locked_tracks[
                        camera
                    ]

                    geometries[
                        camera
                    ] = locked_geometry[
                        camera
                    ]

                    latest_tracks[
                        camera
                    ] = tracks[
                        camera
                    ]

                latest_shapes[
                    camera
                ] = frames[
                    camera
                ].shape

            # ------------------------------------------------
            # Right wrist always live.
            # ------------------------------------------------

            camera = LIVE_WIRE_CAMERA

            tracks[
                camera
            ] = temporal_trackers[
                camera
            ].update(
                detections[
                    camera
                ]
            )

            geometries[
                camera
            ] = free_trackers[
                camera
            ].update(
                tracks[
                    camera
                ]
            )

            latest_tracks[
                camera
            ] = tracks[
                camera
            ]

            latest_shapes[
                camera
            ] = frames[
                camera
            ].shape

            # ------------------------------------------------
            # Live tweezer tip on ALL THREE cameras.
            # ------------------------------------------------

            tip_results = {}

            for camera in CAMERA_ORDER:

                geom = (
                    geometries[
                        camera
                    ]
                )

                grasp_yx = (
                    geom.get(
                        "grasp_point_yx"
                    )
                    if geom is not None
                    else None
                )

                tip_results[
                    camera
                ] = tip_trackers[
                    camera
                ].update(
                    detections[
                        camera
                    ][
                        "tool_mask"
                    ],
                    grasp_yx,
                )

                latest_tips[
                    camera
                ] = tip_results[
                    camera
                ]

            # ------------------------------------------------
            # Draw
            # ------------------------------------------------

            visuals = {}

            for camera in CAMERA_ORDER:

                vis = manual.draw_scene(
                    frames[
                        camera
                    ],
                    detections[
                        camera
                    ],
                    tracks[
                        camera
                    ],
                    geometries[
                        camera
                    ],
                    camera.upper(),
                    fps_values[
                        camera
                    ],
                )

                vis = draw_tip(
                    vis,
                    tip_results[
                        camera
                    ],
                )

                if (
                    camera
                    in FIXED_WIRE_CAMERAS
                ):

                    if locked[
                        camera
                    ]:

                        add_mode_label(
                            vis,
                            "WIRE LOCKED | TWEEZER + TIP LIVE",
                            (
                                0,
                                255,
                                255,
                            ),
                        )

                    else:

                        add_mode_label(
                            vis,
                            "CLICK FREE TO LOCK WIRE | TWEEZER LIVE",
                            (
                                0,
                                165,
                                255,
                            ),
                        )

                else:

                    if free_trackers[
                        camera
                    ].calibrated:

                        add_mode_label(
                            vis,
                            "RIGHT WRIST: ALL LIVE",
                            (
                                0,
                                255,
                                0,
                            ),
                        )

                    else:

                        add_mode_label(
                            vis,
                            "RIGHT WRIST: CLICK FREE | TOOL LIVE",
                            (
                                0,
                                165,
                                255,
                            ),
                        )

                visuals[
                    camera
                ] = vis

            # ------------------------------------------------
            # Resize + dynamic mouse layout
            # ------------------------------------------------

            resized = {}

            for camera in CAMERA_ORDER:

                resized[
                    camera
                ] = base.resize_panel(
                    visuals[
                        camera
                    ]
                )

            x_offset = 0

            panel_layout.clear()

            for camera in CAMERA_ORDER:

                h, w = (
                    resized[
                        camera
                    ].shape[:2]
                )

                panel_layout[
                    camera
                ] = {
                    "x0":
                        x_offset,

                    "x1":
                        x_offset + w,

                    "w":
                        w,

                    "h":
                        h,
                }

                x_offset += w

            panel = np.hstack(
                [
                    resized[
                        camera
                    ]
                    for camera
                    in CAMERA_ORDER
                ]
            )

            cv2.imshow(
                WINDOW,
                panel,
            )

            # ------------------------------------------------
            # Keyboard
            # ------------------------------------------------

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # Print tip coordinates.
            if key in (
                ord("p"),
                ord("P"),
            ):

                print()
                print("=" * 72)
                print("LIVE TWEEZER TIP POSITIONS")
                print("=" * 72)

                for camera in CAMERA_ORDER:

                    result = (
                        latest_tips[
                            camera
                        ]
                    )

                    if (
                        result is not None
                        and
                        result[
                            "valid"
                        ]
                    ):

                        ty, tx = (
                            result[
                                "tip_yx"
                            ]
                        )

                        print(
                            f"{camera:12s} "
                            f"TIP=(x={tx:.1f}, y={ty:.1f}) "
                            f"d_tip_grasp="
                            f"{result['tip_grasp_distance_px']:.1f}px"
                        )

                    else:

                        status = (
                            result[
                                "status"
                            ]
                            if result
                            is not None
                            else
                            "NO_RESULT"
                        )

                        print(
                            f"{camera:12s} "
                            f"TIP INVALID "
                            f"({status})"
                        )

                print("=" * 72)
                print()

            # Clear calibration.
            elif key in (
                ord("c"),
                ord("C"),
            ):

                print()
                print(
                    "CLEAR / UNLOCK ALL"
                )

                for tracker in (
                    free_trackers.values()
                ):
                    tracker.clear()

                for tracker in (
                    tip_trackers.values()
                ):
                    tracker.clear()

                temporal_trackers = {
                    camera:
                        temporal.TemporalTracker(
                            camera
                        )
                    for camera
                    in CAMERA_ORDER
                }

                for camera in (
                    "external",
                    "left_wrist",
                ):

                    locked[
                        camera
                    ] = False

                    locked_tracks[
                        camera
                    ] = None

                    locked_geometry[
                        camera
                    ] = None

                print(
                    "All FREE calibration cleared."
                )

            elif key in (
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
            "[INFO] HYBRID WIRE + TWEEZER TIP V1 stopped"
        )


if __name__ == "__main__":
    main()
