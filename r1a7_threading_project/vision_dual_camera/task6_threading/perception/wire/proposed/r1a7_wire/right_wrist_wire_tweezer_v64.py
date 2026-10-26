#!/usr/bin/env python3

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from pyorbbecsdk import (
    Pipeline,
    Config,
    OBSensorType,
    OBFormat,
)

from wire_local_detector_v64 import (
    WireLocalDetectorV64,
)

# Reuse camera / ROI helpers from V6.2.
from right_wrist_wire_image_v62 import (
    find_device,
    frame_to_bgr,
    click_tip,
    roi_from_tip,
    roi_from_path,
    expand_roi,
)


# ============================================================
# Locate Task6 and reuse the October SAM2 tweezer detector
# ============================================================

HERE = Path(__file__).resolve().parent

TASK6_DIR = None

for p in [HERE] + list(HERE.parents):
    candidate = (
        p /
        "task6_wrist_ibvs_controller_v2_tip.py"
    )

    if candidate.exists():
        TASK6_DIR = p
        break

if TASK6_DIR is None:
    raise RuntimeError(
        "Cannot find task6_wrist_ibvs_controller_v2_tip.py"
    )

sys.path.insert(
    0,
    str(TASK6_DIR)
)

import task6_wrist_ibvs_controller_v2_tip as wrist


SERIAL = "CPCBC53000C5"

OUTPUT = Path(
    "/tmp/r1a7_right_wrist_wire_v64.json"
)


# ============================================================
# Tweezer semantic mask tracker
# ============================================================

class TweezerMaskTracker:

    def __init__(
        self,
        init_frame
    ):

        print()
        print(
            "[TWEEZER] Loading SAM2...",
            flush=True
        )

        self.detector = (
            wrist.Sam2PcaTweezerDetector(
                checkpoint_path=
                    wrist.SAM2_CHECKPOINT
            )
        )

        self.refiner = (
            wrist.TweezerTipRefinerV2()
        )

        print(
            "[TWEEZER] SAM2 ready.",
            flush=True
        )

        init = wrist.initialize_live_tip(
            init_frame
        )

        if init is None:
            raise RuntimeError(
                "Tweezer initialization cancelled"
            )

        (
            self.tip_box,
            self.positive_points,
            self.tip_hint
        ) = init

        self.last_tip = None
        self.last_mask = None


    def update(
        self,
        frame
    ):

        try:

            coarse = self.detector.detect(
                frame,
                box_xyxy=
                    self.tip_box,
                tip_hint=
                    self.tip_hint,
                positive_points=
                    self.positive_points,
            )

        except Exception as exc:

            return {
                "valid": False,
                "state":
                    "SAM2_ERROR:"
                    +
                    type(exc).__name__,
                "tip": None,
                "raw_mask": None,
                "exclusion_mask": None,
            }

        if not coarse.get(
            "valid",
            False
        ):

            return {
                "valid": False,
                "state": "SAM2_LOST",
                "tip": None,
                "raw_mask": None,
                "exclusion_mask": None,
            }

        mask = np.asarray(
            coarse["mask"]
        )

        mask = np.squeeze(
            mask
        )

        if (
            mask.shape[:2]
            !=
            frame.shape[:2]
        ):

            mask = cv2.resize(
                mask.astype(
                    np.uint8
                ),
                (
                    frame.shape[1],
                    frame.shape[0]
                ),
                interpolation=
                    cv2.INTER_NEAREST
            )

        raw_mask = (
            mask > 0
        ).astype(
            np.uint8
        ) * 255

        tip = np.asarray(
            coarse["tip_uv"],
            dtype=np.float64
        )

        # Refine the tip if the already-tested refiner succeeds.
        try:

            refined = self.refiner.refine(
                frame,
                raw_mask > 0,
                coarse["tip_uv"],
                coarse["direction"],
            )

            if refined.get(
                "valid",
                False
            ):

                tip = np.asarray(
                    refined[
                        "refined_tip_uv"
                    ],
                    dtype=np.float64
                )

        except Exception:
            pass

        tip_xy = (
            int(round(
                float(tip[0])
            )),
            int(round(
                float(tip[1])
            ))
        )

        self.tip_hint = (
            float(tip[0]),
            float(tip[1])
        )

        # ----------------------------------------------------
        # Build hard negative mask for wire detector.
        #
        # Expand tool mask so jaw boundaries are also removed.
        # ----------------------------------------------------

        kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (13, 13)
            )
        )

        exclusion = cv2.dilate(
            raw_mask,
            kernel,
            iterations=1
        )

        # ----------------------------------------------------
        # Small aperture at the physical tweezer tip.
        #
        # This allows the real wire to enter the grasp point,
        # but only exposes a very short piece of the tool tip.
        # ----------------------------------------------------

        cv2.circle(
            exclusion,
            tip_xy,
            16,
            0,
            -1
        )

        self.last_tip = (
            tip_xy
        )

        self.last_mask = (
            exclusion
        )

        return {
            "valid": True,
            "state": "TWEEZER_TRACK",
            "tip": tip_xy,
            "raw_mask": raw_mask,
            "exclusion_mask": exclusion,
        }


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 72)
    print("R1-A7 RIGHT WRIST")
    print("WIRE V6.3 + SAM2 TWEEZER EXCLUSION")
    print("SHADOW ONLY - NO ROBOT CONTROL")
    print("=" * 72)

    ctx, device = (
        find_device()
    )

    pipeline = Pipeline(
        device
    )

    config = Config()

    profiles = (
        pipeline
        .get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        )
    )

    profile = (
        profiles
        .get_video_stream_profile(
            1280,
            800,
            OBFormat.RGB,
            30
        )
    )

    config.enable_stream(
        profile
    )

    pipeline.start(
        config
    )

    try:

        frame = None

        print(
            "[CAMERA] Waiting for frame..."
        )

        while frame is None:

            frames = (
                pipeline.wait_for_frames(
                    1000
                )
            )

            if frames is None:
                continue

            color = (
                frames.get_color_frame()
            )

            if color is None:
                continue

            frame = (
                frame_to_bgr(
                    color
                )
            )

        print(
            "[CAMERA] Ready:",
            frame.shape
        )

        # ----------------------------------------------------
        # 1. Initialize TWEEZER first.
        # ----------------------------------------------------

        tweezer = (
            TweezerMaskTracker(
                frame
            )
        )

        # ----------------------------------------------------
        # 2. Initialize WIRE free tip.
        # ----------------------------------------------------

        seed = click_tip(
            frame
        )

        if seed is None:
            return

        wire = (
            WireLocalDetectorV64(local_length_px=180.0, grasp_offset_px=55.0, grasp_band_start_px=35.0, grasp_band_end_px=90.0)
        )

        wire.set_seed_tip(
            seed
        )

        active_roi = (
            roi_from_tip(
                seed,
                frame.shape,
                half_w=130,
                half_h=130,
            )
        )

        anchor_roi = (
            active_roi
        )

        miss_count = 0

        print()
        print(
            "[WIRE] Initial tip:",
            seed
        )

        print(
            "[RUN] Q/ESC = quit"
        )

        print(
            "[RUN] R = relabel wire tip"
        )

        print(
            "[RUN] NO LowCmd / NO gripper command"
        )

        window = (
            "RIGHT WRIST "
            "WIRE V6.3 + TWEEZER MASK"
        )

        cv2.namedWindow(
            window,
            cv2.WINDOW_NORMAL
        )

        last_log = 0.0

        while True:

            frames = (
                pipeline.wait_for_frames(
                    200
                )
            )

            if frames is None:
                continue

            color = (
                frames.get_color_frame()
            )

            if color is None:
                continue

            frame = (
                frame_to_bgr(
                    color
                )
            )

            if frame is None:
                continue

            # =================================================
            # TWEEZER semantic detection
            # =================================================

            tweezer_result = (
                tweezer.update(
                    frame
                )
            )

            tweezer_valid = bool(
                tweezer_result[
                    "valid"
                ]
            )

            exclusion = (
                tweezer_result[
                    "exclusion_mask"
                ]
            )

            # =================================================
            # WIRE detection
            #
            # Safety:
            # if the tweezer segmentation is lost, do NOT
            # allow an unmasked detector to relabel the jaw
            # as wire.
            # =================================================

            if tweezer_valid:

                result = wire.detect(
                    frame,
                    active_roi,
                    tweezer_exclusion_mask=exclusion,
                    tweezer_tip=tweezer_result["tip"],
                )

            else:

                result = {
                    "valid": False,
                    "state":
                        "TWEEZER_MASK_REQUIRED",
                    "candidate_count": 0,
                    "centerline": [],
                    "debug_centerline": [],
                    "wire_tip_uv": None,
                    "grasp_target_uv": None,
                    "wire_direction_uv": None,
                    "score": 0.0,
                    "ridge_score": 0.0,
                    "rank_score": 0.0,
                    "profile_score": 0.0,
                    "two_sided_fraction": 0.0,
                    "straight_edge_overlap": 0.0,
                    "tweezer_mask_fraction": 0.0,
                }

            debug_line = result.get(
                "debug_centerline",
                []
            )

            # =================================================
            # Dynamic search ROI
            # =================================================

            wire_geometry_valid = bool(result.get("valid", False) or result.get("grasp_guidance_valid", False))
            if (
                wire_geometry_valid
                and
                len(
                    result[
                        "centerline"
                    ]
                )
                >=
                2
            ):

                miss_count = 0

                active_roi = (
                    roi_from_path(
                        result[
                            "centerline"
                        ],
                        frame.shape,
                        margin=60,
                        min_w=240,
                        min_h=240,
                    )
                )

                if result.get("tip_valid", False):
                    anchor_roi = active_roi

            elif (
                len(debug_line) >= 2
                and
                result.get(
                    "rank_score",
                    0.0
                )
                >=
                0.30
            ):

                miss_count = min(
                    miss_count + 1,
                    20
                )

                active_roi = (
                    roi_from_path(
                        debug_line,
                        frame.shape,
                        margin=70,
                        min_w=260,
                        min_h=260,
                    )
                )

            else:

                miss_count = min(
                    miss_count + 1,
                    20
                )

                expansion = min(
                    12 * miss_count,
                    60
                )

                active_roi = (
                    expand_roi(
                        anchor_roi,
                        frame.shape,
                        expansion
                    )
                )

            # =================================================
            # Draw
            # =================================================

            vis = frame.copy()

            # ----------------------------------------------
            # Draw tweezer exclusion mask boundary.
            # ----------------------------------------------

            if (
                tweezer_valid
                and
                exclusion is not None
            ):

                contours, _ = (
                    cv2.findContours(
                        exclusion,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                )

                cv2.drawContours(
                    vis,
                    contours,
                    -1,
                    (255, 0, 0),
                    2
                )

                tt = (
                    tweezer_result[
                        "tip"
                    ]
                )

                if tt is not None:

                    cv2.circle(
                        vis,
                        tuple(tt),
                        7,
                        (255, 255, 0),
                        -1
                    )

                    cv2.putText(
                        vis,
                        "TWEEZER TIP",
                        (
                            tt[0] + 10,
                            tt[1] - 10
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (255, 255, 0),
                        2
                    )

            # ----------------------------------------------
            # Search ROI
            # ----------------------------------------------

            x, y, w, h = (
                active_roi
            )

            cv2.rectangle(
                vis,
                (
                    x,
                    y
                ),
                (
                    x + w,
                    y + h
                ),
                (255, 255, 0),
                2
            )

            cv2.putText(
                vis,
                "ACTIVE SEARCH ROI",
                (
                    x + 4,
                    max(
                        20,
                        y - 8
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 0),
                2
            )

            # ----------------------------------------------
            # Invalid candidate = purple.
            # ----------------------------------------------

            debug_pts = np.asarray(
                debug_line,
                dtype=np.int32
            )

            if (
                not result["valid"]
                and
                len(debug_pts) >= 2
            ):

                cv2.polylines(
                    vis,
                    [
                        debug_pts.reshape(
                            -1,
                            1,
                            2
                        )
                    ],
                    False,
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.putText(
                    vis,
                    "CANDIDATE ONLY",
                    (
                        int(
                            debug_pts[0][0]
                        )
                        +
                        8,
                        int(
                            debug_pts[0][1]
                        )
                        -
                        8
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 0, 255),
                    2
                )

            # ----------------------------------------------
            # Confirmed wire = green.
            # ----------------------------------------------

            if result["valid"]:

                pts = np.asarray(
                    result[
                        "centerline"
                    ],
                    dtype=np.int32
                )

                if len(pts) >= 2:

                    cv2.polylines(
                        vis,
                        [
                            pts.reshape(
                                -1,
                                1,
                                2
                            )
                        ],
                        False,
                        (0, 255, 0),
                        3,
                        cv2.LINE_AA
                    )

                tip = result[
                    "wire_tip_uv"
                ]

                grasp = result[
                    "grasp_target_uv"
                ]

                if tip is not None:

                    cv2.circle(
                        vis,
                        tuple(tip),
                        8,
                        (0, 0, 255),
                        -1
                    )

                    cv2.putText(
                        vis,
                        "WIRE TIP",
                        (
                            tip[0] + 10,
                            tip[1] - 10
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (0, 0, 255),
                        2
                    )

                if grasp is not None:

                    cv2.circle(
                        vis,
                        tuple(grasp),
                        8,
                        (0, 255, 255),
                        -1
                    )

                    cv2.putText(
                        vis,
                        "GRASP",
                        (
                            grasp[0] + 10,
                            grasp[1] + 18
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (0, 255, 255),
                        2
                    )

            # ----------------------------------------------
            grasp_segment = np.asarray(
                result.get("grasp_segment_uv", []),
                dtype=np.int32
            )

            if len(grasp_segment) >= 2:
                cv2.polylines(
                    vis,
                    [grasp_segment.reshape(-1, 1, 2)],
                    False,
                    (0, 255, 255),
                    6,
                    cv2.LINE_AA
                )

            # Status
            # ----------------------------------------------

            state = result[
                "state"
            ]

            cv2.putText(
                vis,
                (
                    f"WIRE={state} "
                    f"TWEEZER="
                    f"{tweezer_result['state']}"
                ),
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (
                    (0, 255, 0)
                    if result["valid"]
                    else
                    (0, 165, 255)
                ),
                2
            )

            cv2.putText(
                vis,
                (
                    f"score="
                    f"{result.get('score',0):.2f} "
                    f"ridge="
                    f"{result.get('ridge_score',0):.2f} "
                    f"mask="
                    f"{result.get('tweezer_mask_fraction',0):.2f}"
                ),
                (20, 67),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.53,
                (255, 255, 255),
                2
            )

            # =================================================
            # Publish
            # =================================================

            output = {
                "enabled": False,

                "mode":
                    "RIGHT_WRIST_WIRE_V63_TWEEZER_MASK",

                "camera_serial":
                    SERIAL,

                "timestamp":
                    time.time(),

                "active_roi":
                    list(
                        active_roi
                    ),

                "miss_count":
                    int(
                        miss_count
                    ),

                "tweezer_valid":
                    tweezer_valid,

                "tweezer_state":
                    tweezer_result[
                        "state"
                    ],

                "tweezer_tip_uv":
                    (
                        list(
                            tweezer_result[
                                "tip"
                            ]
                        )
                        if
                        tweezer_result[
                            "tip"
                        ]
                        is not None
                        else
                        None
                    ),

                **result
            }

            OUTPUT.write_text(
                json.dumps(
                    output,
                    indent=2
                )
            )

            now = time.time()

            if (
                now -
                last_log
                >
                1.0
            ):

                print(
                    "[V63]"
                    f" wire={state}"
                    f" valid={result['valid']}"
                    f" tip={result.get('wire_tip_uv')}"
                    f" grasp={result.get('grasp_target_uv')}"
                    f" tweezer="
                    f"{tweezer_result.get('tip')}"
                    f" mask="
                    f"{result.get('tweezer_mask_fraction',0):.2f}"
                    f" miss={miss_count}",
                    flush=True
                )

                last_log = now

            cv2.imshow(
                window,
                vis
            )

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            if key in (
                ord("q"),
                ord("Q"),
                27
            ):
                break

            if key in (
                ord("r"),
                ord("R")
            ):

                seed = click_tip(
                    frame
                )

                if seed is not None:

                    wire = (
                        WireLocalDetectorV64(local_length_px=180.0, grasp_offset_px=55.0, grasp_band_start_px=35.0, grasp_band_end_px=90.0)
                    )

                    wire.set_seed_tip(
                        seed
                    )

                    active_roi = (
                        roi_from_tip(
                            seed,
                            frame.shape,
                            half_w=130,
                            half_h=130,
                        )
                    )

                    anchor_roi = (
                        active_roi
                    )

                    miss_count = 0

    finally:

        try:
            pipeline.stop()
        except Exception:
            pass

        cv2.destroyAllWindows()

        print(
            "[STOP] V6.3 stopped."
        )


if __name__ == "__main__":
    main()
