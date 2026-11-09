#!/usr/bin/env python3

import sys
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Paths
# ============================================================

THIS_DIR = Path(__file__).resolve().parent

WIRE_DIR = (
    THIS_DIR.parent
    / "wire_unet"
)

if str(WIRE_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(WIRE_DIR),
    )


# ============================================================
# Existing tested modules
# ============================================================

import live_visual_guidance_shadow_v1 as wire_live
import live_hole_board_unet_three_camera_v1 as hole_live


CAMERAS = (
    "external",
    "right_wrist",
    "left_wrist",
)

WINDOW = (
    "R1-A7 INSERTION PERCEPTION V1 "
    "- HOLE + LIVE FREE + TIP"
)

HOLE_CHECKPOINT = Path(
    "/home/robot/r1a7_datasets/wire_unet/"
    "hole_tube_001/runs/"
    "hole_board_unet_v1/best.pt"
)

OUT_JSON = Path(
    "/tmp/r1a7_insertion_perception_v1.json"
)


# Display geometry.
TILE_W = 640
TILE_H = 400


# ============================================================
# JSON
# ============================================================

def atomic_json(
    path,
    data,
):

    tmp = Path(
        str(path) + ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
        )

        f.flush()

        os.fsync(
            f.fileno()
        )

    os.replace(
        tmp,
        path,
    )


# ============================================================
# Small helpers
# ============================================================

def uv_from_yx(
    p,
):

    if p is None:
        return None

    return [
        float(p[1]),
        float(p[0]),
    ]


def format_uv(
    p,
):

    if p is None:
        return "(--,--)"

    return (
        f"({p[0]:.1f},"
        f"{p[1]:.1f})"
    )


def resize_tile(
    image,
):

    return cv2.resize(
        image,
        (
            TILE_W,
            TILE_H,
        ),
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# Hole overlay
# ============================================================

def draw_holes(
    image,
    pred,
    min_area,
    max_area,
):

    if pred is None:
        return image, []

    out = image.copy()

    holes = (
        hole_live.extract_hole_centers(
            pred,
            min_area=min_area,
            max_area=max_area,
        )
    )

    # Only overlay Class-2 holes here.
    #
    # Do not tint the whole Class-1 board, otherwise the
    # wire/tool visualization becomes unnecessarily obscured.
    hole_mask = (
        pred == 2
    )

    if np.any(
        hole_mask
    ):

        overlay = out.copy()

        overlay[
            hole_mask
        ] = (
            0,
            150,
            255,
        )

        out = cv2.addWeighted(
            out,
            0.72,
            overlay,
            0.28,
            0,
        )

    for hole in holes:

        u, v = hole[
            "center_uv"
        ]

        center = (
            int(round(u)),
            int(round(v)),
        )

        cv2.drawMarker(
            out,
            center,
            (
                0,
                0,
                255,
            ),
            cv2.MARKER_CROSS,
            18,
            2,
        )

        cv2.circle(
            out,
            center,
            6,
            (
                255,
                255,
                255,
            ),
            1,
        )

    return out, holes


# ============================================================
# Status overlay
# ============================================================

def draw_insertion_status(
    image,
    camera,
    tracker,
    geometry,
    tip_result,
    holes,
):

    free_yx = (
        geometry.get(
            "free_point_yx"
        )
        if geometry
        else None
    )

    grasp_yx = (
        geometry.get(
            "grasp_point_yx"
        )
        if geometry
        else None
    )

    tip_yx = (
        tip_result.get(
            "tip_yx"
        )
        if tip_result
        else None
    )

    free_uv = uv_from_yx(
        free_yx
    )

    grasp_uv = uv_from_yx(
        grasp_yx
    )

    tip_uv = uv_from_yx(
        tip_yx
    )

    measurement_valid = bool(
        geometry.get(
            "measurement_valid",
            False,
        )
        if geometry
        else False
    )

    calibrated = bool(
        getattr(
            tracker,
            "calibrated",
            False,
        )
    )

    tip_valid = bool(
        tip_result.get(
            "valid",
            False,
        )
        if tip_result
        else False
    )

    lines = [
        (
            f"INSERTION LIVE  "
            f"{camera.upper()}"
        ),
        (
            f"FREE={format_uv(free_uv)}  "
            f"live={int(measurement_valid)}"
        ),
        (
            f"TIP ={format_uv(tip_uv)}  "
            f"valid={int(tip_valid)}"
        ),
        (
            f"GRASP={format_uv(grasp_uv)}  "
            f"cal={int(calibrated)}"
        ),
        (
            f"HOLES={len(holes)}"
        ),
    ]

    x = 10
    y = 24

    for i, line in enumerate(
        lines
    ):

        cv2.putText(
            image,
            line,
            (
                x,
                y + 22 * i,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (
                255,
                255,
                255,
            ),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            line,
            (
                x,
                y + 22 * i,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (
                20,
                20,
                20,
            ),
            1,
            cv2.LINE_AA,
        )

    return image


# ============================================================
# Main
# ============================================================

def main():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hole-checkpoint",
        default=str(
            HOLE_CHECKPOINT
        ),
    )

    parser.add_argument(
        "--hole-every",
        type=int,
        default=2,
        help=(
            "Run Hole U-Net every N frames. "
            "Use 1 for every frame."
        ),
    )

    parser.add_argument(
        "--hole-tile",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--hole-stride",
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

    args = parser.parse_args()


    print("=" * 82)
    print(
        "R1-A7 INSERTION PERCEPTION V1"
    )
    print("=" * 82)
    print(
        "HOLE + LIVE FREE + TWEEZER TIP"
    )
    print()
    print(
        "VISION ONLY - NO ROBOT COMMANDS"
    )
    print()
    print(
        "LEFT CLICK each camera: initialize FREE identity"
    )
    print(
        "C: clear FREE/TIP calibration"
    )
    print(
        "P: print current perception JSON"
    )
    print(
        "Q / ESC: quit"
    )
    print("=" * 82)


    # ========================================================
    # Wire models
    # ========================================================

    # Existing V2 model:
    # External + Left Wrist.
    wire_model, device = (
        wire_live.base.load_model()
    )

    # Existing V3 model:
    # Right Wrist only.
    wire_model_right = (
        wire_live.load_model_from_checkpoint(
            wire_live.RIGHT_WRIST_V3_CHECKPOINT,
            device,
        )
    )


    # ========================================================
    # Hole model
    # ========================================================

    hole_model, hole_device = (
        hole_live.load_model(
            Path(
                args.hole_checkpoint
            )
        )
    )


    # ========================================================
    # Trackers
    # ========================================================

    temporal_trackers = {
        camera:
            wire_live.temporal.TemporalTracker(
                camera
            )
        for camera in CAMERAS
    }

    free_trackers = {
        camera:
            wire_live.manual.ManualFreeTracker(
                camera
            )
        for camera in CAMERAS
    }

    tip_trackers = {}

    for camera in CAMERAS:

        if camera == "right_wrist":

            tip_trackers[
                camera
            ] = (
                wire_live.wrist_tip_mid.
                RightWristTweezerTipMidpointV1(
                    camera
                )
            )

        else:

            tip_trackers[
                camera
            ] = (
                wire_live.TweezerTipTracker(
                    camera
                )
            )


    # ========================================================
    # Runtime state
    # ========================================================

    latest_tracks = {
        camera: None
        for camera in CAMERAS
    }

    latest_shapes = {
        camera: None
        for camera in CAMERAS
    }

    latest_geometry = {
        camera: {}
        for camera in CAMERAS
    }

    latest_tips = {
        camera: {}
        for camera in CAMERAS
    }

    latest_holes = {
        camera: []
        for camera in CAMERAS
    }

    latest_hole_pred = {
        camera: None
        for camera in CAMERAS
    }

    latest_output = {}

    frame_index = 0

    external = None
    right = None
    left = None


    # ========================================================
    # Mouse FREE calibration
    # ========================================================

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

        # 2 x 2 panel:
        #
        # External    | Right
        # Left        | Info

        camera = None
        local_x = None
        local_y = None

        if (
            0 <= x < TILE_W
            and
            0 <= y < TILE_H
        ):

            camera = "external"
            local_x = x
            local_y = y

        elif (
            TILE_W <= x < 2 * TILE_W
            and
            0 <= y < TILE_H
        ):

            camera = "right_wrist"
            local_x = x - TILE_W
            local_y = y

        elif (
            0 <= x < TILE_W
            and
            TILE_H <= y < 2 * TILE_H
        ):

            camera = "left_wrist"
            local_x = x
            local_y = y - TILE_H

        else:
            return

        track = latest_tracks[
            camera
        ]

        shape = latest_shapes[
            camera
        ]

        if (
            track is None
            or
            shape is None
            or
            track.get(
                "path"
            )
            is None
        ):

            print(
                f"[FREE INIT] {camera}: "
                "no valid tracked path"
            )

            return

        h = int(
            shape[0]
        )

        w = int(
            shape[1]
        )

        native_x = int(
            np.clip(
                local_x
                *
                w
                /
                TILE_W,
                0,
                w - 1,
            )
        )

        native_y = int(
            np.clip(
                local_y
                *
                h
                /
                TILE_H,
                0,
                h - 1,
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
                native_y,
                native_x,
            ),
        )

        if ok:

            g = (
                tracker.last_geometry
            )

            print()
            print(
                f"[FREE INIT OK] {camera}"
            )

            print(
                "  FREE =",
                g.get(
                    "free_point_yx"
                ),
            )

            print(
                "  GRASP =",
                g.get(
                    "grasp_point_yx"
                ),
            )

            print(
                "  NOTE: insertion mode keeps this tracker LIVE"
            )

        else:

            print(
                f"[FREE INIT FAILED] {camera}"
            )


    # ========================================================
    # Cameras
    # ========================================================

    try:

        external = (
            wire_live.base.open_external()
        )

        ctx = (
            wire_live.base.Context()
        )

        right = (
            wire_live.base.open_orbbec(
                ctx,
                wire_live.base.RIGHT_SERIAL,
            )
        )

        left = (
            wire_live.base.open_orbbec(
                ctx,
                wire_live.base.LEFT_SERIAL,
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

        print()
        print(
            "[READY] cameras opened"
        )

        print(
            "[ACTION] click actual FREE endpoint "
            "once in each camera view"
        )

        last_print = 0.0


        # ====================================================
        # Loop
        # ====================================================

        while True:

            frame_index += 1

            ok_ext, frame_ext = (
                external.read()
            )

            frame_right = (
                wire_live.base.read_orbbec(
                    right
                )
            )

            frame_left = (
                wire_live.base.read_orbbec(
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
                continue

            frames = {
                "external":
                    frame_ext,

                "right_wrist":
                    frame_right,

                "left_wrist":
                    frame_left,
            }


            # =================================================
            # 1. Wire / Tool U-Net
            # =================================================

            detections = {}
            wire_fps = {}

            for camera in CAMERAS:

                active_model = (
                    wire_model_right
                    if camera
                    ==
                    "right_wrist"
                    else wire_model
                )

                (
                    detections[
                        camera
                    ],
                    wire_fps[
                        camera
                    ],
                ) = (
                    wire_live.base.run_one(
                        active_model,
                        device,
                        frames[
                            camera
                        ],
                    )
                )


            # =================================================
            # 2. ALL THREE WIRES ARE LIVE
            #
            # No locked_tracks.
            # No locked_geometry.
            # =================================================

            tracks = {}
            geometries = {}

            for camera in CAMERAS:

                tracks[
                    camera
                ] = (
                    temporal_trackers[
                        camera
                    ].update(
                        detections[
                            camera
                        ]
                    )
                )

                geometries[
                    camera
                ] = (
                    free_trackers[
                        camera
                    ].update(
                        tracks[
                            camera
                        ]
                    )
                )

                latest_tracks[
                    camera
                ] = tracks[
                    camera
                ]

                latest_shapes[
                    camera
                ] = (
                    frames[
                        camera
                    ].shape
                )

                latest_geometry[
                    camera
                ] = (
                    geometries[
                        camera
                    ]
                )


            # =================================================
            # 3. Tweezer TIP
            # =================================================

            tip_results = {}

            for camera in CAMERAS:

                grasp_yx = (
                    geometries[
                        camera
                    ].get(
                        "grasp_point_yx"
                    )
                )

                tip_results[
                    camera
                ] = (
                    tip_trackers[
                        camera
                    ].update(
                        detections[
                            camera
                        ][
                            "tool_mask"
                        ],
                        grasp_yx,
                    )
                )

                latest_tips[
                    camera
                ] = (
                    tip_results[
                        camera
                    ]
                )


            # =================================================
            # 4. Hole U-Net
            #
            # Cache prediction between updates to reduce load.
            # =================================================

            run_hole = (
                frame_index == 1
                or
                frame_index
                %
                max(
                    1,
                    args.hole_every,
                )
                ==
                0
            )

            if run_hole:

                for camera in CAMERAS:

                    latest_hole_pred[
                        camera
                    ] = (
                        hole_live.predict_sliding(
                            hole_model,
                            frames[
                                camera
                            ],
                            hole_device,
                            tile=args.hole_tile,
                            stride=args.hole_stride,
                        )
                    )


            # =================================================
            # 5. Display
            # =================================================

            visuals = {}

            current_holes = {}

            for camera in CAMERAS:

                # Existing proven wire/FREE/GRASP display.
                vis = (
                    wire_live.draw_scene(
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
                        wire_fps[
                            camera
                        ],
                    )
                )

                # Existing proven TIP display.
                vis = (
                    wire_live.draw_tip(
                        vis,
                        tip_results[
                            camera
                        ],
                    )
                )

                # Hole overlay.
                vis, holes = (
                    draw_holes(
                        vis,
                        latest_hole_pred[
                            camera
                        ],
                        args.min_hole_area,
                        args.max_hole_area,
                    )
                )

                latest_holes[
                    camera
                ] = holes

                current_holes[
                    camera
                ] = holes

                vis = (
                    draw_insertion_status(
                        vis,
                        camera,
                        free_trackers[
                            camera
                        ],
                        geometries[
                            camera
                        ],
                        tip_results[
                            camera
                        ],
                        holes,
                    )
                )

                visuals[
                    camera
                ] = resize_tile(
                    vis
                )


            # =================================================
            # 6. Publish perception-only JSON
            # =================================================

            camera_output = {}

            for camera in CAMERAS:

                geometry = (
                    geometries[
                        camera
                    ]
                )

                tip = (
                    tip_results[
                        camera
                    ]
                )

                free_yx = (
                    geometry.get(
                        "free_point_yx"
                    )
                )

                grasp_yx = (
                    geometry.get(
                        "grasp_point_yx"
                    )
                )

                tip_yx = (
                    tip.get(
                        "tip_yx"
                    )
                )

                holes_json = []

                for h in (
                    current_holes[
                        camera
                    ]
                ):

                    holes_json.append({
                        "center_uv": [
                            float(
                                h[
                                    "center_uv"
                                ][0]
                            ),
                            float(
                                h[
                                    "center_uv"
                                ][1]
                            ),
                        ],

                        "area_px":
                            int(
                                h[
                                    "area"
                                ]
                            ),
                    })

                camera_output[
                    camera
                ] = {

                    "free_calibrated":
                        bool(
                            getattr(
                                free_trackers[
                                    camera
                                ],
                                "calibrated",
                                False,
                            )
                        ),

                    "free_measurement_valid":
                        bool(
                            geometry.get(
                                "measurement_valid",
                                False,
                            )
                        ),

                    "track_state":
                        (
                            tracks[
                                camera
                            ].get(
                                "state"
                            )
                            if tracks[
                                camera
                            ]
                            else None
                        ),

                    "geometry_status":
                        geometry.get(
                            "status"
                        ),

                    "free_uv":
                        uv_from_yx(
                            free_yx
                        ),

                    "grasp_uv":
                        uv_from_yx(
                            grasp_yx
                        ),

                    "tip_valid":
                        bool(
                            tip.get(
                                "valid",
                                False,
                            )
                        ),

                    "tip_status":
                        tip.get(
                            "status"
                        ),

                    "tip_uv":
                        uv_from_yx(
                            tip_yx
                        ),

                    "holes":
                        holes_json,
                }

            latest_output = {
                "timestamp":
                    time.time(),

                "enabled":
                    False,

                "shadow_mode":
                    True,

                "source":
                    "live_insertion_perception_v1",

                "robot_commands":
                    False,

                "cameras":
                    camera_output,
            }

            atomic_json(
                OUT_JSON,
                latest_output,
            )


            # =================================================
            # 7. Info panel
            # =================================================

            info = np.zeros(
                (
                    TILE_H,
                    TILE_W,
                    3,
                ),
                dtype=np.uint8,
            )

            lines = [
                "R1-A7 INSERTION PERCEPTION V1",
                "",
                "HOLE + LIVE FREE + TIP",
                "",
                "ALL 3 WIRE TRACKERS = LIVE",
                "",
                "Left click:",
                " initialize FREE in camera panel",
                "",
                "C = clear all calibration",
                "P = print current JSON",
                "Q / ESC = quit",
                "",
                "VISION ONLY",
                "NO ROBOT COMMANDS",
            ]

            yy = 32

            for line in lines:

                cv2.putText(
                    info,
                    line,
                    (
                        20,
                        yy,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (
                        255,
                        255,
                        255,
                    ),
                    1,
                    cv2.LINE_AA,
                )

                yy += 25


            top = np.hstack(
                [
                    visuals[
                        "external"
                    ],
                    visuals[
                        "right_wrist"
                    ],
                ]
            )

            bottom = np.hstack(
                [
                    visuals[
                        "left_wrist"
                    ],
                    info,
                ]
            )

            panel = np.vstack(
                [
                    top,
                    bottom,
                ]
            )

            cv2.imshow(
                WINDOW,
                panel,
            )


            # =================================================
            # Terminal status
            # =================================================

            now = time.monotonic()

            if (
                now
                -
                last_print
                >=
                1.0
            ):

                pieces = []

                for camera in CAMERAS:

                    g = geometries[
                        camera
                    ]

                    t = tip_results[
                        camera
                    ]

                    free_uv = (
                        uv_from_yx(
                            g.get(
                                "free_point_yx"
                            )
                        )
                    )

                    tip_uv = (
                        uv_from_yx(
                            t.get(
                                "tip_yx"
                            )
                        )
                    )

                    pieces.append(
                        (
                            f"{camera}: "
                            f"F={format_uv(free_uv)} "
                            f"T={format_uv(tip_uv)} "
                            f"H={len(current_holes[camera])}"
                        )
                    )

                print(
                    "[INSERTION] "
                    +
                    " | ".join(
                        pieces
                    ),
                    flush=True,
                )

                last_print = now


            # =================================================
            # Keyboard
            # =================================================

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            if key in (
                ord("c"),
                ord("C"),
            ):

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
                        wire_live.temporal.TemporalTracker(
                            camera
                        )
                    for camera in CAMERAS
                }

                print(
                    "[CLEAR] FREE/TIP calibration cleared"
                )

            elif key in (
                ord("p"),
                ord("P"),
            ):

                print()
                print(
                    json.dumps(
                        latest_output,
                        indent=2,
                    )
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

        atomic_json(
            OUT_JSON,
            {
                "timestamp":
                    time.time(),

                "enabled":
                    False,

                "shadow_mode":
                    True,

                "source":
                    "live_insertion_perception_v1",

                "robot_commands":
                    False,

                "valid":
                    False,

                "reason":
                    "PROGRAM_EXIT",
            },
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
            "[INFO] insertion perception stopped"
        )


if __name__ == "__main__":
    main()
