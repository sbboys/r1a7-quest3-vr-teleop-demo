#!/usr/bin/env python3

import time

import cv2
import numpy as np

import live_wire_unet_three_camera as base
import live_wire_temporal_validation_v1 as temporal
import live_manual_free_grasp_validation_v1 as manual


# ============================================================
# MODE
# ============================================================

WINDOW = "R1-A7 HYBRID LOCKED FREE/GRASP V1"

CAMERA_ORDER = (
    "external",
    "right_wrist",
    "left_wrist",
)

FIXED_CAMERAS = {
    "external",
    "left_wrist",
}

LIVE_CAMERA = "right_wrist"

PANEL_W = base.DISPLAY_W
PANEL_H = base.DISPLAY_H


# ============================================================
# HELPERS
# ============================================================

def copy_dict_arrays(src):

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


def make_blank_detection(
    shape,
):

    h, w = shape[:2]

    return {
        "tool_mask":
            np.zeros(
                (h, w),
                dtype=bool,
            ),
    }


def add_mode_label(
    vis,
    text,
    color,
):

    cv2.rectangle(
        vis,
        (
            0,
            vis.shape[0] - 32,
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
            vis.shape[0] - 9,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "R1-A7 HYBRID LOCKED FREE/GRASP V1"
    )

    print("=" * 80)

    print()
    print(
        "EXTERNAL:"
    )

    print(
        "  click FREE once -> LOCK wire/FREE/GRASP"
    )

    print()
    print(
        "RIGHT WRIST:"
    )

    print(
        "  click FREE once -> CONTINUOUS LIVE tracking"
    )

    print()
    print(
        "LEFT WRIST:"
    )

    print(
        "  click FREE once -> LOCK wire/FREE/GRASP"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  External/Left locked coordinates do NOT update."
    )

    print(
        "  Right wrist FREE/GRASP follows the live tracked wire."
    )

    print()
    print(
        "C       = clear/unlock all and recalibrate"
    )

    print(
        "Q / ESC = quit"
    )

    print()
    print(
        "NO DDS / NO LowCmd / NO ROBOT CONTROL"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model, device = (
        base.load_model()
    )

    # Temporal trackers.
    temporal_trackers = {
        name:
            temporal.TemporalTracker(
                name
            )
        for name in CAMERA_ORDER
    }

    # Manual semantic identity trackers.
    free_trackers = {
        name:
            manual.ManualFreeTracker(
                name
            )
        for name in CAMERA_ORDER
    }

    # Fixed camera lock state.
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

    # Latest live state used by mouse callback.
    latest_tracks = {
        name:
            None
        for name in CAMERA_ORDER
    }

    latest_shapes = {
        name:
            None
        for name in CAMERA_ORDER
    }

    # --------------------------------------------------------
    # Mouse callback
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
            != cv2.EVENT_LBUTTONDOWN
        ):
            return

        if (
            y < 0
            or
            y >= PANEL_H
        ):
            return

        panel_index = int(
            x // PANEL_W
        )

        if (
            panel_index < 0
            or
            panel_index
            >= len(
                CAMERA_ORDER
            )
        ):
            return

        camera = (
            CAMERA_ORDER[
                panel_index
            ]
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

        if (
            track is None
            or
            shape is None
        ):

            print(
                f"[{camera}] "
                f"no path available"
            )

            return

        if track[
            "state"
        ] not in (
            "INIT",
            "TRACK",
            "RECOVER",
        ):

            print(
                f"[{camera}] "
                f"click rejected: "
                f"state={track['state']}"
            )

            return

        # Position inside selected panel.
        local_x = (
            x
            -
            panel_index
            * PANEL_W
        )

        local_y = y

        h, w = shape[:2]

        # Map display coordinate back to native camera image.
        original_x = int(
            np.clip(
                local_x
                * w
                / PANEL_W,
                0,
                w - 1,
            )
        )

        original_y = int(
            np.clip(
                local_y
                * h
                / PANEL_H,
                0,
                h - 1,
            )
        )

        click_yx = (
            original_y,
            original_x,
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
            click_yx,
        )

        if not ok:
            return

        # ----------------------------------------------------
        # EXTERNAL / LEFT WRIST:
        # lock current wire geometry permanently in image
        # coordinates until user presses C.
        # ----------------------------------------------------

        if camera in FIXED_CAMERAS:

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

            geom = (
                copy_dict_arrays(
                    tracker.last_geometry
                )
            )

            geom[
                "state"
            ] = "LOCKED"

            geom[
                "status"
            ] = "LOCKED_REFERENCE"

            # It remains a valid reference geometry,
            # but is no longer a new live measurement.
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
                "=" * 70
            )

            print(
                f"[{camera}] STATIC REFERENCE LOCKED"
            )

            print(
                "FREE =",
                geom[
                    "free_endpoint_yx"
                ],
            )

            print(
                "FIXED =",
                geom[
                    "fixed_endpoint_yx"
                ],
            )

            print(
                "GRASP =",
                geom[
                    "grasp_point_yx"
                ],
            )

            print(
                "No more U-Net inference "
                "for this camera."
            )

            print(
                "=" * 70
            )

        # ----------------------------------------------------
        # RIGHT WRIST:
        # manual click initializes FREE identity only.
        # It continues to move with the live path.
        # ----------------------------------------------------

        else:

            print()
            print(
                "=" * 70
            )

            print(
                "[right_wrist] LIVE FREE IDENTITY INITIALIZED"
            )

            print(
                "FREE will now follow the live tracked wire."
            )

            print(
                "=" * 70
            )

    # --------------------------------------------------------
    # Cameras
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

            # =================================================
            # CAPTURE ALL THREE CAMERAS
            # =================================================

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

            # =================================================
            # EXTERNAL + LEFT
            #
            # Before calibration:
            #   live inference
            #
            # After calibration:
            #   NO inference
            #   use locked reference
            # =================================================

            for camera in (
                "external",
                "left_wrist",
            ):

                if not locked[
                    camera
                ]:

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

                    # No model inference.
                    fps_values[
                        camera
                    ] = 0.0

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

                    detections[
                        camera
                    ] = make_blank_detection(
                        frames[
                            camera
                        ].shape
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

            # =================================================
            # RIGHT WRIST
            #
            # ALWAYS LIVE
            # =================================================

            (
                detections[
                    LIVE_CAMERA
                ],
                fps_values[
                    LIVE_CAMERA
                ],
            ) = base.run_one(
                model,
                device,
                frames[
                    LIVE_CAMERA
                ],
            )

            tracks[
                LIVE_CAMERA
            ] = temporal_trackers[
                LIVE_CAMERA
            ].update(
                detections[
                    LIVE_CAMERA
                ]
            )

            geometries[
                LIVE_CAMERA
            ] = free_trackers[
                LIVE_CAMERA
            ].update(
                tracks[
                    LIVE_CAMERA
                ]
            )

            latest_tracks[
                LIVE_CAMERA
            ] = tracks[
                LIVE_CAMERA
            ]

            latest_shapes[
                LIVE_CAMERA
            ] = frames[
                LIVE_CAMERA
            ].shape

            # =================================================
            # DISPLAY
            # =================================================

            visuals = {}

            for camera in CAMERA_ORDER:

                visuals[
                    camera
                ] = manual.draw_scene(
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

                if (
                    camera
                    in FIXED_CAMERAS
                ):

                    if locked[
                        camera
                    ]:

                        add_mode_label(
                            visuals[
                                camera
                            ],
                            "STATIC LOCKED REFERENCE",
                            (
                                0,
                                255,
                                255,
                            ),
                        )

                    else:

                        add_mode_label(
                            visuals[
                                camera
                            ],
                            "LIVE INITIALIZATION - CLICK FREE TO LOCK",
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
                            visuals[
                                camera
                            ],
                            "RIGHT WRIST LIVE TRACKING",
                            (
                                0,
                                255,
                                0,
                            ),
                        )

                    else:

                        add_mode_label(
                            visuals[
                                camera
                            ],
                            "RIGHT WRIST - CLICK FREE TO INITIALIZE",
                            (
                                0,
                                165,
                                255,
                            ),
                        )

            panel = np.hstack(
                [
                    base.resize_panel(
                        visuals[
                            camera
                        ]
                    )
                    for camera
                    in CAMERA_ORDER
                ]
            )

            cv2.imshow(
                WINDOW,
                panel,
            )

            # =================================================
            # KEYBOARD
            # =================================================

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # -------------------------------------------------
            # Clear everything and reinitialize.
            # -------------------------------------------------

            if key in (
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

                # Re-create temporal state so old path history
                # cannot contaminate a new calibration.
                temporal_trackers = {
                    name:
                        temporal.TemporalTracker(
                            name
                        )
                    for name in CAMERA_ORDER
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
                    "All cameras returned to initialization mode."
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
            "[INFO] HYBRID FREE/GRASP V1 stopped"
        )


if __name__ == "__main__":
    main()
