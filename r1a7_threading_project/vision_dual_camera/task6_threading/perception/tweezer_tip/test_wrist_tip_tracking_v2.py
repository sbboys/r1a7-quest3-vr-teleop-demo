#!/usr/bin/env python3

import argparse
import csv
import json
import sys
import time

from pathlib import Path

import cv2
import numpy as np


THIS_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[4]
)

sys.path.insert(
    0,
    str(THIS_DIR),
)

sys.path.insert(
    0,
    str(
        PROJECT_ROOT
        /
        "vision_dual_camera"
        /
        "handeye"
    ),
)

from sam2_pca_detector import (
    Sam2PcaTweezerDetector,
)

from tip_refiner_v2 import (
    TweezerTipRefinerV2,
)

from tip_temporal_filter import (
    TipTemporalFilter,
)

import collect_handeye_samples as hc

from pyorbbecsdk import (
    Config,
    OBFormat,
    OBSensorType,
    Pipeline,
)


CHECKPOINT = (
    PROJECT_ROOT
    /
    "third_party"
    /
    "sam2"
    /
    "checkpoints"
    /
    "sam2.1_hiera_small.pt"
)

OUTPUT_DIR = (
    THIS_DIR
    /
    "output"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def select_positive_points(
    image,
):

    selected = []

    win = "SELECT TWO SAM2 POSITIVE POINTS"

    def callback(
        event,
        x,
        y,
        flags,
        param,
    ):

        if (
            event
            ==
            cv2.EVENT_LBUTTONDOWN
        ):

            if len(selected) < 2:

                selected.append(
                    (x, y)
                )

        elif (
            event
            ==
            cv2.EVENT_RBUTTONDOWN
        ):

            if selected:
                selected.pop()

    cv2.namedWindow(
        win,
        cv2.WINDOW_NORMAL,
    )

    cv2.setMouseCallback(
        win,
        callback,
    )

    while True:

        vis = image.copy()

        cv2.putText(
            vis,
            "Left click JAW1 and JAW2 metal",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            vis,
            "Right click = undo   ENTER = confirm",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        for i, p in enumerate(
            selected
        ):

            cv2.circle(
                vis,
                p,
                7,
                (0, 0, 255),
                -1,
            )

            cv2.putText(
                vis,
                f"P{i + 1}",
                (
                    p[0] + 10,
                    p[1] - 10,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        if len(selected) == 2:

            midpoint = (
                int(
                    round(
                        (
                            selected[0][0]
                            +
                            selected[1][0]
                        )
                        / 2.0
                    )
                ),
                int(
                    round(
                        (
                            selected[0][1]
                            +
                            selected[1][1]
                        )
                        / 2.0
                    )
                ),
            )

            cv2.drawMarker(
                vis,
                midpoint,
                (0, 255, 0),
                cv2.MARKER_CROSS,
                25,
                2,
            )

            cv2.putText(
                vis,
                "TIP SIDE HINT",
                (
                    midpoint[0] + 10,
                    midpoint[1] + 25,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imshow(
            win,
            vis,
        )

        key = (
            cv2.waitKey(20)
            &
            0xFF
        )

        if (
            key in (10, 13)
            and
            len(selected) == 2
        ):

            cv2.destroyWindow(
                win
            )

            return selected

        if key == 27:

            cv2.destroyWindow(
                win
            )

            return None



def calculate_stats(
    records,
):

    valid = [
        r for r in records
        if r["valid"]
    ]

    summary = {
        "total_frames":
        len(records),

        "valid_frames":
        len(valid),

        "valid_rate":
        (
            len(valid)
            /
            max(
                1,
                len(records),
            )
        ),
    }

    if not valid:
        return summary

    refined = np.asarray(
        [
            [
                r["refined_u"],
                r["refined_v"],
            ]
            for r in valid
        ],
        dtype=np.float64,
    )

    coarse = np.asarray(
        [
            [
                r["coarse_u"],
                r["coarse_v"],
            ]
            for r in valid
        ],
        dtype=np.float64,
    )

    gaps = np.asarray(
        [
            r["jaw_gap_px"]
            for r in valid
        ],
        dtype=np.float64,
    )

    runtimes = np.asarray(
        [
            r["runtime_ms"]
            for r in valid
        ],
        dtype=np.float64,
    )

    mean_refined = (
        refined.mean(
            axis=0
        )
    )

    std_refined = (
        refined.std(
            axis=0
        )
    )

    mean_coarse = (
        coarse.mean(
            axis=0
        )
    )

    std_coarse = (
        coarse.std(
            axis=0
        )
    )

    if len(refined) >= 2:

        jumps = np.linalg.norm(
            np.diff(
                refined,
                axis=0,
            ),
            axis=1,
        )

        max_jump = float(
            np.max(jumps)
        )

        mean_jump = float(
            np.mean(jumps)
        )

    else:

        max_jump = 0.0
        mean_jump = 0.0

    summary.update(
        {
            "refined_mean_uv": [
                float(mean_refined[0]),
                float(mean_refined[1]),
            ],

            "refined_std_uv": [
                float(std_refined[0]),
                float(std_refined[1]),
            ],

            "refined_radial_std_px":
            float(
                np.sqrt(
                    std_refined[0] ** 2
                    +
                    std_refined[1] ** 2
                )
            ),

            "refined_peak_to_peak_uv": [
                float(
                    refined[:, 0].max()
                    -
                    refined[:, 0].min()
                ),
                float(
                    refined[:, 1].max()
                    -
                    refined[:, 1].min()
                ),
            ],

            "coarse_mean_uv": [
                float(mean_coarse[0]),
                float(mean_coarse[1]),
            ],

            "coarse_std_uv": [
                float(std_coarse[0]),
                float(std_coarse[1]),
            ],

            "jaw_gap_mean_px":
            float(
                gaps.mean()
            ),

            "jaw_gap_std_px":
            float(
                gaps.std()
            ),

            "mean_step_jump_px":
            mean_jump,

            "max_step_jump_px":
            max_jump,

            "runtime_mean_ms":
            float(
                runtimes.mean()
            ),

            "runtime_p95_ms":
            float(
                np.percentile(
                    runtimes,
                    95,
                )
            ),
        }
    )

    return summary


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-frames",
        type=int,
        default=1000000,
    )

    parser.add_argument(
        "--label",
        type=str,
        default="test",
    )

    parser.add_argument(
        "--save-debug",
        action="store_true",
    )

    args = parser.parse_args()

    print("")
    print("==========================================")
    print("R1-A7 TWEEZER TIP V2 LIVE TRACKING")
    print("==========================================")
    print("")
    print(
        "LIVE TRACKING: robot may move slowly after initialization."
    )
    print("")

    detector = (
        Sam2PcaTweezerDetector(
            checkpoint_path=CHECKPOINT
        )
    )

    refiner = TweezerTipRefinerV2()

    tip_filter = TipTemporalFilter(
        alpha=0.4,
        max_jump_px=8.0,
    )

    context, device = (
        hc.find_orbbec_device(
            hc.RIGHT_WRIST_SN
        )
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
            hc.COLOR_WIDTH,
            hc.COLOR_HEIGHT,
            OBFormat.RGB,
            hc.COLOR_FPS,
        )
    )

    config.enable_stream(
        profile
    )

    pipeline.start(
        config
    )

    win = "TIP STABILITY INITIALIZATION"

    cv2.namedWindow(
        win,
        cv2.WINDOW_NORMAL,
    )

    try:

        print(
            "Press S to freeze one frame and initialize."
        )

        init_frame = None

        while True:

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
                hc.color_frame_to_bgr(
                    color
                )
            )

            vis = frame.copy()

            cv2.putText(
                vis,
                "S = initialize   Q = quit",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            cv2.imshow(
                win,
                vis,
            )

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):
                return

            if key in (
                ord("s"),
                ord("S"),
            ):
                init_frame = (
                    frame.copy()
                )
                break

        cv2.destroyWindow(
            win
        )

        roi = cv2.selectROI(
            "SELECT TWEEZER ROI",
            init_frame,
            fromCenter=False,
            showCrosshair=True,
        )

        cv2.destroyWindow(
            "SELECT TWEEZER ROI"
        )

        x, y, rw, rh = roi

        if (
            rw <= 0
            or
            rh <= 0
        ):
            print("ROI cancelled.")
            return

        box = [
            int(x),
            int(y),
            int(x + rw),
            int(y + rh),
        ]

        positive_points = (
            select_positive_points(
                init_frame
            )
        )

        if positive_points is None:
            print(
                "Point selection cancelled."
            )
            return

        tip_hint_array = np.mean(
            np.asarray(
                positive_points,
                dtype=np.float64,
            ),
            axis=0,
        )

        tip_hint = (
            float(tip_hint_array[0]),
            float(tip_hint_array[1]),
        )

        print("")
        print("ROI =", box)

        print(
            "SAM2 positive points =",
            positive_points,
        )

        print(
            "PCA tip-side hint =",
            tip_hint,
        )

        print("")
        print(
            f"Collecting {args.num_frames} frames..."
        )
        print(
            "Do NOT move robot, tweezer or camera."
        )
        print("")

        records = []

        debug_dir = (
            OUTPUT_DIR
            /
            f"debug_{args.label}"
        )

        debug_low_count = 0
        debug_mid_count = 0
        debug_high_count = 0

        if args.save_debug:
            debug_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            print(
                "Debug image saving:",
                debug_dir,
            )

        display_win = (
            "REFINED TIP STABILITY"
        )

        cv2.namedWindow(
            display_win,
            cv2.WINDOW_NORMAL,
        )

        for i in range(
            args.num_frames
        ):

            frames = (
                pipeline.wait_for_frames(
                    1000
                )
            )

            if frames is None:

                records.append(
                    {
                        "frame": i,
                        "valid": False,
                        "reason": "no_frames",
                    }
                )

                continue

            color = (
                frames.get_color_frame()
            )

            if color is None:

                records.append(
                    {
                        "frame": i,
                        "valid": False,
                        "reason": "no_color",
                    }
                )

                continue

            frame = (
                hc.color_frame_to_bgr(
                    color
                )
            )

            t0 = time.perf_counter()

            coarse = detector.detect(
                frame,
                box_xyxy=box,
                tip_hint=tip_hint,
                positive_points=positive_points,
            )

            if not coarse["valid"]:

                records.append(
                    {
                        "frame": i,
                        "valid": False,
                        "reason": "sam2_pca_failed",
                    }
                )

                continue

            refined = refiner.refine(
                frame,
                coarse["mask"],
                coarse["tip_uv"],
                coarse["direction"],
            )

            runtime_ms = (
                time.perf_counter()
                -
                t0
            ) * 1000.0

            if not refined["valid"]:

                records.append(
                    {
                        "frame": i,
                        "valid": False,
                        "reason": refined.get(
                            "reason",
                            "refine_failed",
                        ),
                    }
                )

                continue

            filtered = tip_filter.update(
                refined["refined_tip_uv"],
                refined["jaw_gap_px"],
            )

            filtered_u = float(
                filtered[
                    "filtered_tip_uv"
                ][0]
            )

            filtered_v = float(
                filtered[
                    "filtered_tip_uv"
                ][1]
            )

            filtered_gap = (
                filtered.get(
                    "filtered_gap_px"
                )
            )

            filter_rejected = bool(
                filtered.get(
                    "rejected",
                    False,
                )
            )

            coarse_u = float(
                coarse["tip_uv"][0]
            )

            coarse_v = float(
                coarse["tip_uv"][1]
            )

            refined_u = float(
                refined[
                    "refined_tip_uv"
                ][0]
            )

            refined_v = float(
                refined[
                    "refined_tip_uv"
                ][1]
            )

            record = {
                "frame": i,
                "valid": True,

                "coarse_u":
                coarse_u,

                "coarse_v":
                coarse_v,

                "refined_u":
                refined_u,

                "refined_v":
                refined_v,

                "filtered_u":
                filtered_u,

                "filtered_v":
                filtered_v,

                "filtered_gap_px":
                filtered_gap,

                "filter_rejected":
                filter_rejected,

                "jaw1_u":
                refined[
                    "jaw_tip_1_uv"
                ][0],

                "jaw1_v":
                refined[
                    "jaw_tip_1_uv"
                ][1],

                "jaw2_u":
                refined[
                    "jaw_tip_2_uv"
                ][0],

                "jaw2_v":
                refined[
                    "jaw_tip_2_uv"
                ][1],

                "jaw_gap_px":
                refined[
                    "jaw_gap_px"
                ],

                "v2_confidence":
                refined.get(
                    "confidence",
                    0.0,
                ),

                "v2_longitudinal_diff_px":
                refined.get(
                    "longitudinal_diff_px",
                    0.0,
                ),

                "v2_mean_forward_px":
                refined.get(
                    "mean_forward_px",
                    0.0,
                ),

                "sam_score":
                coarse[
                    "sam_score"
                ],

                "pca_linearity":
                coarse[
                    "pca_linearity"
                ],

                "runtime_ms":
                runtime_ms,
            }

            records.append(
                record
            )

            vis = frame.copy()

            jaw1 = tuple(
                np.round(
                    refined[
                        "jaw_tip_1_uv"
                    ]
                ).astype(int)
            )

            jaw2 = tuple(
                np.round(
                    refined[
                        "jaw_tip_2_uv"
                    ]
                ).astype(int)
            )

            refined_tip = (
                int(round(refined_u)),
                int(round(refined_v)),
            )

            coarse_tip = (
                int(round(coarse_u)),
                int(round(coarse_v)),
            )

            cv2.circle(
                vis,
                jaw1,
                5,
                (255, 255, 0),
                -1,
            )

            cv2.circle(
                vis,
                jaw2,
                5,
                (255, 0, 255),
                -1,
            )

            cv2.drawMarker(
                vis,
                coarse_tip,
                (0, 0, 255),
                cv2.MARKER_CROSS,
                22,
                2,
            )

            cv2.drawMarker(
                vis,
                refined_tip,
                (0, 255, 0),
                cv2.MARKER_CROSS,
                24,
                2,
            )

            filtered_tip = (
                int(round(filtered_u)),
                int(round(filtered_v)),
            )

            cv2.drawMarker(
                vis,
                filtered_tip,
                (0, 255, 255),
                cv2.MARKER_CROSS,
                34,
                3,
            )

            cv2.putText(
                vis,
                (
                    f"{i + 1}/{args.num_frames}  "
                    f"RAW=({refined_u:.2f},"
                    f"{refined_v:.2f})  "
                    f"EMA=({filtered_u:.2f},"
                    f"{filtered_v:.2f})"
                ),
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            if args.save_debug:

                gap_now = float(
                    refined["jaw_gap_px"]
                )

                save_this = False
                category = None

                # Three bands:
                # low  = typical narrow jaw solution
                # mid  = intermediate candidate
                # high = suspicious / wide candidate
                #
                # Save only a few per band so we do not create
                # hundreds of images.

                if (
                    gap_now < 25.0
                    and
                    debug_low_count < 5
                ):
                    category = "LOW"
                    debug_low_count += 1
                    save_this = True

                elif (
                    25.0 <= gap_now < 40.0
                    and
                    debug_mid_count < 5
                ):
                    category = "MID"
                    debug_mid_count += 1
                    save_this = True

                elif (
                    gap_now >= 40.0
                    and
                    debug_high_count < 8
                ):
                    category = "HIGH"
                    debug_high_count += 1
                    save_this = True

                if save_this:

                    prefix = (
                        f"{i + 1:03d}_"
                        f"{category}_"
                        f"gap_{gap_now:.1f}"
                    )

                    raw_path = (
                        debug_dir
                        /
                        f"{prefix}_raw.png"
                    )

                    mask_path = (
                        debug_dir
                        /
                        f"{prefix}_mask.png"
                    )

                    overlay_path = (
                        debug_dir
                        /
                        f"{prefix}_overlay.png"
                    )

                    cv2.imwrite(
                        str(raw_path),
                        frame,
                    )

                    mask_to_save = (
                        coarse["mask"]
                    )

                    if (
                        mask_to_save.dtype
                        != np.uint8
                    ):
                        mask_to_save = (
                            mask_to_save
                            > 0
                        ).astype(
                            np.uint8
                        ) * 255

                    elif (
                        mask_to_save.max()
                        <= 1
                    ):
                        mask_to_save = (
                            mask_to_save
                            * 255
                        ).astype(
                            np.uint8
                        )

                    cv2.imwrite(
                        str(mask_path),
                        mask_to_save,
                    )

                    cv2.imwrite(
                        str(overlay_path),
                        vis,
                    )

                    print(
                        f"[DEBUG] saved {category} "
                        f"frame={i + 1} "
                        f"gap={gap_now:.2f}px"
                    )

            cv2.imshow(
                display_win,
                vis,
            )

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):
                print("")
                print("Tracking stopped by user.")
                break

            print(
                f"[{i + 1:03d}/"
                f"{args.num_frames:03d}] "
                f"coarse=({coarse_u:.2f},"
                f"{coarse_v:.2f}) "
                f"raw=({refined_u:.2f},"
                f"{refined_v:.2f}) "
                f"ema=({filtered_u:.2f},"
                f"{filtered_v:.2f}) "
                f"gap={refined['jaw_gap_px']:.2f}px "
                f"reject={filter_rejected} "
                f"runtime={runtime_ms:.1f}ms"
            )

        cv2.destroyWindow(
            display_win
        )

        summary = (
            calculate_stats(
                records
            )
        )

        valid_records = [
            r
            for r in records
            if r.get("valid", False)
        ]

        if valid_records:

            filtered_xy = np.asarray(
                [
                    [
                        r["filtered_u"],
                        r["filtered_v"],
                    ]
                    for r in valid_records
                ],
                dtype=np.float64,
            )

            filtered_mean = (
                filtered_xy.mean(
                    axis=0
                )
            )

            filtered_std = (
                filtered_xy.std(
                    axis=0
                )
            )

            filtered_ptp = (
                filtered_xy.max(
                    axis=0
                )
                -
                filtered_xy.min(
                    axis=0
                )
            )

            if len(filtered_xy) >= 2:

                filtered_jumps = (
                    np.linalg.norm(
                        np.diff(
                            filtered_xy,
                            axis=0,
                        ),
                        axis=1,
                    )
                )

                filtered_mean_jump = float(
                    filtered_jumps.mean()
                )

                filtered_max_jump = float(
                    filtered_jumps.max()
                )

            else:

                filtered_mean_jump = 0.0
                filtered_max_jump = 0.0

            filtered_gaps = np.asarray(
                [
                    r["filtered_gap_px"]
                    for r in valid_records
                    if r["filtered_gap_px"]
                    is not None
                ],
                dtype=np.float64,
            )

            summary.update(
                {
                    "filtered_mean_uv": [
                        float(filtered_mean[0]),
                        float(filtered_mean[1]),
                    ],

                    "filtered_std_uv": [
                        float(filtered_std[0]),
                        float(filtered_std[1]),
                    ],

                    "filtered_radial_std_px":
                    float(
                        np.sqrt(
                            filtered_std[0] ** 2
                            +
                            filtered_std[1] ** 2
                        )
                    ),

                    "filtered_peak_to_peak_uv": [
                        float(filtered_ptp[0]),
                        float(filtered_ptp[1]),
                    ],

                    "filtered_mean_step_jump_px":
                    filtered_mean_jump,

                    "filtered_max_step_jump_px":
                    filtered_max_jump,

                    "filtered_gap_mean_px":
                    float(
                        filtered_gaps.mean()
                    )
                    if len(filtered_gaps)
                    else None,

                    "filtered_gap_std_px":
                    float(
                        filtered_gaps.std()
                    )
                    if len(filtered_gaps)
                    else None,

                    "filter_rejected_frames":
                    int(
                        sum(
                            bool(
                                r.get(
                                    "filter_rejected",
                                    False,
                                )
                            )
                            for r in valid_records
                        )
                    ),

                    "filter_alpha":
                    0.4,

                    "filter_max_jump_px":
                    8.0,
                }
            )

        stamp = time.strftime(
            "%Y%m%d_%H%M%S"
        )

        safe_label = "".join(
            c if (
                c.isalnum()
                or c in ("-", "_")
            )
            else "_"
            for c in args.label
        )

        csv_path = (
            OUTPUT_DIR
            /
            f"{stamp}_{safe_label}_stability.csv"
        )

        json_path = (
            OUTPUT_DIR
            /
            f"{stamp}_{safe_label}_stability_summary.json"
        )

        fieldnames = [
            "frame",
            "valid",
            "reason",
            "coarse_u",
            "coarse_v",
            "refined_u",
            "refined_v",
            "filtered_u",
            "filtered_v",
            "filtered_gap_px",
            "filter_rejected",
            "jaw1_u",
            "jaw1_v",
            "jaw2_u",
            "jaw2_v",
            "jaw_gap_px",
            "v2_confidence",
            "v2_longitudinal_diff_px",
            "v2_mean_forward_px",
            "sam_score",
            "pca_linearity",
            "runtime_ms",
        ]

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )

            writer.writeheader()

            for r in records:

                row = {
                    key: r.get(
                        key,
                        ""
                    )
                    for key
                    in fieldnames
                }

                writer.writerow(
                    row
                )

        with open(
            json_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                {
                    "label":
                    args.label,

                    "roi_xyxy":
                    box,

                    "sam2_positive_points":
                    [
                        list(p)
                        for p
                        in positive_points
                    ],

                    "pca_tip_side_hint":
                    list(
                        tip_hint
                    ),

                    "summary":
                    summary,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print("")
        print("==========================================")
        print("STABILITY SUMMARY")
        print("==========================================")

        for k, v in summary.items():
            print(
                f"{k}: {v}"
            )

        print("")
        print("Saved:")
        print(csv_path)
        print(json_path)

    finally:

        try:
            pipeline.stop()
        except Exception:
            pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
