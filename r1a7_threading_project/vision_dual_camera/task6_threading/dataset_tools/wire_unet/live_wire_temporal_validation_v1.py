#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime
import csv
import json
import time

import cv2
import numpy as np

import live_wire_unet_three_camera as base

from eval_wire_skeleton_endpoints_v2 import (
    skeletonize,
)

from eval_wire_connected_path_v3 import (
    connect_skeleton_components,
    longest_geodesic_path,
)


# ============================================================
# TEMPORAL PARAMETERS
# ============================================================

TEMPORAL_RADIUS = 7

PATH_JUMP_TRIGGER = 0.35

TEMPORAL_SUPPORT_MIN = 0.65

TEMPORAL_LENGTH_JUMP_MAX = 0.35

HOLD_MAX_FRAMES = 2

COMPONENT_MAX_GAP = 50.0


OUT_ROOT = Path(
    "/home/robot/r1a7_datasets/"
    "wire_live_validation"
)

WINDOW = (
    "R1-A7 TEMPORAL PATH TRACKER V1"
)


# ============================================================
# BASIC HELPERS
# ============================================================

def rel_jump(
    cur,
    prev,
):

    if prev is None:
        return 0.0

    if prev <= 1e-6:
        return 0.0

    return abs(
        cur - prev
    ) / prev


def dilate_radius(
    mask,
    radius,
):

    if radius <= 0:
        return mask.copy()

    k = (
        2 * radius
        + 1
    )

    kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (k, k),
        )
    )

    return cv2.dilate(
        mask.astype(
            np.uint8
        ),
        kernel,
    ).astype(bool)


def path_from_wire(
    wire_mask,
):

    out = np.zeros_like(
        wire_mask,
        dtype=bool,
    )

    if not np.any(
        wire_mask
    ):
        return (
            out,
            0.0,
            False,
        )

    skel = skeletonize(
        wire_mask
    )

    skel = np.asarray(
        skel,
        dtype=bool,
    )

    if not np.any(
        skel
    ):
        return (
            out,
            0.0,
            False,
        )

    try:

        (
            connected,
            _,
            connect_ok,
            _,
        ) = connect_skeleton_components(
            skel,
            COMPONENT_MAX_GAP,
        )

    except Exception:

        return (
            out,
            0.0,
            False,
        )

    if (
        not connect_ok
        or not np.any(
            connected
        )
    ):

        return (
            out,
            0.0,
            False,
        )

    try:

        (
            path,
            _,
            _,
            length,
        ) = longest_geodesic_path(
            connected
        )

    except Exception:

        return (
            out,
            0.0,
            False,
        )

    if path is None:
        return (
            out,
            0.0,
            False,
        )

    for p in path:

        y = int(
            p[0]
        )

        x = int(
            p[1]
        )

        if (
            0 <= y < out.shape[0]
            and
            0 <= x < out.shape[1]
        ):

            out[
                y,
                x,
            ] = True

    return (
        out,
        float(length),
        True,
    )


# ============================================================
# TEMPORAL TRACKER
# ============================================================

class TemporalTracker:

    def __init__(
        self,
        name,
    ):

        self.name = name

        self.prev_path = None
        self.prev_length = None

        self.hold_count = 0


    def update(
        self,
        det,
    ):

        raw_path = np.asarray(
            det["wire_path"],
            dtype=bool,
        )

        wire = np.asarray(
            det["wire_mask"],
            dtype=bool,
        )

        raw_length = float(
            det["path_length_px"]
        )

        raw_found = bool(
            np.any(
                raw_path
            )
        )

        # -----------------------------------------------
        # Initialization
        # -----------------------------------------------

        if (
            self.prev_path is None
            or
            self.prev_length is None
            or
            not np.any(
                self.prev_path
            )
        ):

            if raw_found:

                self.prev_path = (
                    raw_path.copy()
                )

                self.prev_length = (
                    raw_length
                )

                self.hold_count = 0

                return {
                    "path":
                        raw_path.copy(),

                    "length":
                        raw_length,

                    "state":
                        "INIT",

                    "support":
                        1.0,

                    "raw_jump":
                        0.0,

                    "candidate_jump":
                        0.0,

                    "hold_count":
                        0,
                }

            return {
                "path":
                    raw_path.copy(),

                "length":
                    0.0,

                "state":
                    "LOST",

                "support":
                    0.0,

                "raw_jump":
                    0.0,

                "candidate_jump":
                    0.0,

                "hold_count":
                    0,
            }

        # -----------------------------------------------
        # Compare current observation with previous path.
        # -----------------------------------------------

        raw_jump = rel_jump(
            raw_length,
            self.prev_length,
        )

        # Current wire expanded by 7 px:
        # "Does the current image still see the previous path
        # somewhere nearby?"
        current_wire_near = (
            dilate_radius(
                wire,
                TEMPORAL_RADIUS,
            )
        )

        prev_pixels = int(
            np.count_nonzero(
                self.prev_path
            )
        )

        if prev_pixels > 0:

            support = float(
                np.mean(
                    current_wire_near[
                        self.prev_path
                    ]
                )
            )

        else:

            support = 0.0

        # -----------------------------------------------
        # Normal case.
        # -----------------------------------------------

        if (
            raw_found
            and
            raw_jump
            <= PATH_JUMP_TRIGGER
        ):

            accepted = (
                raw_path.copy()
            )

            accepted_length = (
                raw_length
            )

            self.prev_path = (
                accepted.copy()
            )

            self.prev_length = (
                accepted_length
            )

            self.hold_count = 0

            return {
                "path":
                    accepted,

                "length":
                    accepted_length,

                "state":
                    "TRACK",

                "support":
                    support,

                "raw_jump":
                    raw_jump,

                "candidate_jump":
                    raw_jump,

                "hold_count":
                    0,
            }

        # -----------------------------------------------
        # Raw path jumped.
        #
        # Restrict current wire to a corridor around the
        # previous accepted path and rebuild the path.
        # -----------------------------------------------

        corridor = dilate_radius(
            self.prev_path,
            TEMPORAL_RADIUS,
        )

        temporal_wire = (
            wire
            &
            corridor
        )

        (
            candidate_path,
            candidate_length,
            candidate_ok,
        ) = path_from_wire(
            temporal_wire
        )

        candidate_jump = rel_jump(
            candidate_length,
            self.prev_length,
        )

        # -----------------------------------------------
        # RECOVER:
        # current image still strongly supports previous
        # physical path position and rebuilt path remains
        # geometrically plausible.
        # -----------------------------------------------

        if (
            candidate_ok
            and
            support
            >= TEMPORAL_SUPPORT_MIN
            and
            candidate_jump
            <= TEMPORAL_LENGTH_JUMP_MAX
        ):

            accepted = (
                candidate_path.copy()
            )

            accepted_length = (
                candidate_length
            )

            self.prev_path = (
                accepted.copy()
            )

            self.prev_length = (
                accepted_length
            )

            self.hold_count = 0

            return {
                "path":
                    accepted,

                "length":
                    accepted_length,

                "state":
                    "RECOVER",

                "support":
                    support,

                "raw_jump":
                    raw_jump,

                "candidate_jump":
                    candidate_jump,

                "hold_count":
                    0,
            }

        # -----------------------------------------------
        # HOLD:
        # Do not invent a new measurement.
        # Only keep previous visualization for <= 2 frames.
        # -----------------------------------------------

        if (
            self.hold_count
            <
            HOLD_MAX_FRAMES
        ):

            self.hold_count += 1

            return {
                "path":
                    self.prev_path.copy(),

                "length":
                    float(
                        self.prev_length
                    ),

                "state":
                    "HOLD",

                "support":
                    support,

                "raw_jump":
                    raw_jump,

                "candidate_jump":
                    candidate_jump,

                "hold_count":
                    self.hold_count,
            }

        # -----------------------------------------------
        # RESET:
        # persistent disagreement.
        #
        # Accept current raw observation and establish a
        # new temporal state.
        # -----------------------------------------------

        if raw_found:

            self.prev_path = (
                raw_path.copy()
            )

            self.prev_length = (
                raw_length
            )

            self.hold_count = 0

            return {
                "path":
                    raw_path.copy(),

                "length":
                    raw_length,

                "state":
                    "RESET",

                "support":
                    support,

                "raw_jump":
                    raw_jump,

                "candidate_jump":
                    candidate_jump,

                "hold_count":
                    0,
            }

        # No raw path after HOLD exhausted.
        self.prev_path = None
        self.prev_length = None
        self.hold_count = 0

        return {
            "path":
                np.zeros_like(
                    raw_path
                ),

            "length":
                0.0,

            "state":
                "LOST",

            "support":
                support,

            "raw_jump":
                raw_jump,

            "candidate_jump":
                candidate_jump,

            "hold_count":
                0,
        }


# ============================================================
# DISPLAY
# ============================================================

def display_temporal(
    image,
    det,
    track,
    name,
    fps,
):

    wire = np.asarray(
        det["wire_mask"],
        dtype=bool,
    )

    tool = np.asarray(
        det["tool_mask"],
        dtype=bool,
    )

    raw_path = np.asarray(
        det["wire_path"],
        dtype=bool,
    )

    tracked_path = np.asarray(
        track["path"],
        dtype=bool,
    )

    overlay = image.copy()

    wire_vis = dilate_radius(
        wire,
        1,
    )

    tool_vis = dilate_radius(
        tool,
        1,
    )

    raw_vis = dilate_radius(
        raw_path,
        2,
    )

    tracked_vis = dilate_radius(
        tracked_path,
        2,
    )

    # Tool = blue
    overlay[
        tool_vis
    ] = (
        255,
        0,
        0,
    )

    # U-Net wire = pink
    overlay[
        wire_vis
    ] = (
        255,
        80,
        255,
    )

    # Raw current path = yellow
    overlay[
        raw_vis
    ] = (
        0,
        255,
        255,
    )

    # Tracked path = red
    overlay[
        tracked_vis
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

    state = (
        track["state"]
    )

    if state in (
        "TRACK",
        "INIT",
    ):

        state_color = (
            0,
            255,
            0,
        )

    elif state == "RECOVER":

        state_color = (
            0,
            255,
            255,
        )

    elif state == "HOLD":

        state_color = (
            0,
            165,
            255,
        )

    else:

        state_color = (
            0,
            0,
            255,
        )

    cv2.rectangle(
        vis,
        (
            0,
            0,
        ),
        (
            vis.shape[1],
            92,
        ),
        (
            0,
            0,
            0,
        ),
        -1,
    )

    raw_len = float(
        det["path_length_px"]
    )

    tracked_len = float(
        track["length"]
    )

    line1 = (
        f"{name}  "
        f"STATE={state}"
    )

    line2 = (
        f"RAW={raw_len:.0f}px  "
        f"TRACK={tracked_len:.0f}px  "
        f"SUPPORT={track['support']:.2f}"
    )

    line3 = (
        f"RAW_JUMP={track['raw_jump']:.2f}  "
        f"CAND_JUMP={track['candidate_jump']:.2f}  "
        f"FPS={fps:.1f}"
    )

    cv2.putText(
        vis,
        line1,
        (
            12,
            27,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        state_color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        vis,
        line2,
        (
            12,
            57,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (
            255,
            255,
            255,
        ),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        vis,
        line3,
        (
            12,
            83,
        ),
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


# ============================================================
# RECORDER
# ============================================================

class Recorder:

    def __init__(
        self,
    ):

        self.active = False

        self.session = None

        self.fp = None
        self.writer = None

        self.start_time = None
        self.frame_index = 0

        self.prev_tracked_len = {
            "external":
                None,

            "right_wrist":
                None,

            "left_wrist":
                None,
        }

        self.rows = []


    def start(
        self,
    ):

        if self.active:
            return

        stamp = (
            datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        self.session = (
            OUT_ROOT
            /
            f"temporal_v1_{stamp}"
        )

        self.session.mkdir(
            parents=True,
            exist_ok=False,
        )

        (
            self.session
            /
            "anomaly"
        ).mkdir()

        self.fp = open(
            self.session
            /
            "temporal_metrics.csv",
            "w",
            newline="",
        )

        fields = [
            "frame_index",
            "elapsed_s",
            "camera",

            "state",

            "raw_path_length_px",
            "tracked_path_length_px",

            "raw_jump",
            "tracked_jump",

            "temporal_support",
            "candidate_jump",

            "hold_count",

            "wire_pixels",
            "tool_pixels",

            "infer_fps",
        ]

        self.writer = (
            csv.DictWriter(
                self.fp,
                fieldnames=fields,
            )
        )

        self.writer.writeheader()

        self.start_time = (
            time.perf_counter()
        )

        self.frame_index = 0

        self.rows = []

        for k in self.prev_tracked_len:
            self.prev_tracked_len[k] = None

        self.active = True

        print()
        print("=" * 72)
        print("TEMPORAL V1 RECORDING STARTED")
        print("SESSION =", self.session)
        print("=" * 72)
        print()


    def add(
        self,
        camera,
        image,
        vis,
        det,
        track,
        fps,
    ):

        if not self.active:
            return

        tracked_len = float(
            track["length"]
        )

        tracked_jump = rel_jump(
            tracked_len,
            self.prev_tracked_len[
                camera
            ],
        )

        row = {
            "frame_index":
                self.frame_index,

            "elapsed_s":
                time.perf_counter()
                - self.start_time,

            "camera":
                camera,

            "state":
                track["state"],

            "raw_path_length_px":
                float(
                    det[
                        "path_length_px"
                    ]
                ),

            "tracked_path_length_px":
                tracked_len,

            "raw_jump":
                float(
                    track["raw_jump"]
                ),

            "tracked_jump":
                tracked_jump,

            "temporal_support":
                float(
                    track["support"]
                ),

            "candidate_jump":
                float(
                    track[
                        "candidate_jump"
                    ]
                ),

            "hold_count":
                int(
                    track["hold_count"]
                ),

            "wire_pixels":
                int(
                    np.count_nonzero(
                        det["wire_mask"]
                    )
                ),

            "tool_pixels":
                int(
                    np.count_nonzero(
                        det["tool_mask"]
                    )
                ),

            "infer_fps":
                float(fps),
        }

        self.writer.writerow(
            row
        )

        self.rows.append(
            row
        )

        self.prev_tracked_len[
            camera
        ] = tracked_len

        # Save important transitions only.
        if (
            track["raw_jump"]
            >
            PATH_JUMP_TRIGGER
            or
            track["state"]
            in (
                "RECOVER",
                "HOLD",
                "RESET",
                "LOST",
            )
        ):

            stem = (
                f"{self.frame_index:06d}_"
                f"{camera}_"
                f"{track['state']}"
            )

            cv2.imwrite(
                str(
                    self.session
                    /
                    "anomaly"
                    /
                    f"{stem}.png"
                ),
                vis,
                [
                    cv2.IMWRITE_PNG_COMPRESSION,
                    1,
                ],
            )


    def next_frame(
        self,
    ):

        if self.active:

            self.frame_index += 1

            if (
                self.frame_index
                % 10
                == 0
            ):

                self.fp.flush()


    def stop(
        self,
    ):

        if not self.active:
            return

        self.active = False

        self.fp.flush()
        self.fp.close()

        summary = {
            "parameters": {
                "temporal_radius":
                    TEMPORAL_RADIUS,

                "path_jump_trigger":
                    PATH_JUMP_TRIGGER,

                "support_min":
                    TEMPORAL_SUPPORT_MIN,

                "candidate_length_jump_max":
                    TEMPORAL_LENGTH_JUMP_MAX,

                "hold_max_frames":
                    HOLD_MAX_FRAMES,
            },

            "cameras": {},
        }

        for camera in (
            "external",
            "right_wrist",
            "left_wrist",
        ):

            rows = [
                r
                for r in self.rows
                if r["camera"] == camera
            ]

            if not rows:
                continue

            raw_jump = np.asarray(
                [
                    r["raw_jump"]
                    for r in rows
                ],
                dtype=float,
            )

            tracked_jump = np.asarray(
                [
                    r["tracked_jump"]
                    for r in rows
                ],
                dtype=float,
            )

            states = [
                r["state"]
                for r in rows
            ]

            support = np.asarray(
                [
                    r[
                        "temporal_support"
                    ]
                    for r in rows
                ],
                dtype=float,
            )

            fps = np.asarray(
                [
                    r["infer_fps"]
                    for r in rows
                ],
                dtype=float,
            )

            n = len(rows)

            def state_rate(
                name,
            ):
                return float(
                    np.mean(
                        [
                            x == name
                            for x in states
                        ]
                    )
                )

            summary[
                "cameras"
            ][camera] = {

                "frames":
                    n,

                "raw_jump_gt35_rate":
                    float(
                        np.mean(
                            raw_jump
                            >
                            PATH_JUMP_TRIGGER
                        )
                    ),

                "tracked_jump_gt35_rate":
                    float(
                        np.mean(
                            tracked_jump
                            >
                            PATH_JUMP_TRIGGER
                        )
                    ),

                "raw_jump_p95":
                    float(
                        np.percentile(
                            raw_jump,
                            95,
                        )
                    ),

                "tracked_jump_p95":
                    float(
                        np.percentile(
                            tracked_jump,
                            95,
                        )
                    ),

                "support_median":
                    float(
                        np.median(
                            support
                        )
                    ),

                "track_rate":
                    state_rate(
                        "TRACK"
                    ),

                "recover_rate":
                    state_rate(
                        "RECOVER"
                    ),

                "hold_rate":
                    state_rate(
                        "HOLD"
                    ),

                "reset_rate":
                    state_rate(
                        "RESET"
                    ),

                "lost_rate":
                    state_rate(
                        "LOST"
                    ),

                "fps_median":
                    float(
                        np.median(
                            fps
                        )
                    ),
            }

        summary_file = (
            self.session
            /
            "summary.json"
        )

        summary_file.write_text(
            json.dumps(
                summary,
                indent=2,
            )
        )

        print()
        print("=" * 72)
        print("TEMPORAL V1 SUMMARY")
        print("=" * 72)

        for camera, s in (
            summary[
                "cameras"
            ].items()
        ):

            print()
            print(
                camera.upper()
            )

            print(
                "  frames                  =",
                s["frames"],
            )

            print(
                "  RAW jump >35%           =",
                f"{s['raw_jump_gt35_rate']:.4f}",
            )

            print(
                "  TRACKED jump >35%       =",
                f"{s['tracked_jump_gt35_rate']:.4f}",
            )

            print(
                "  RAW jump P95            =",
                f"{s['raw_jump_p95']:.4f}",
            )

            print(
                "  TRACKED jump P95        =",
                f"{s['tracked_jump_p95']:.4f}",
            )

            print(
                "  temporal support median =",
                f"{s['support_median']:.4f}",
            )

            print(
                "  TRACK rate              =",
                f"{s['track_rate']:.4f}",
            )

            print(
                "  RECOVER rate            =",
                f"{s['recover_rate']:.4f}",
            )

            print(
                "  HOLD rate               =",
                f"{s['hold_rate']:.4f}",
            )

            print(
                "  RESET rate              =",
                f"{s['reset_rate']:.4f}",
            )

            print(
                "  LOST rate               =",
                f"{s['lost_rate']:.4f}",
            )

            print(
                "  FPS median              =",
                f"{s['fps_median']:.2f}",
            )

        print()
        print(
            "SESSION =",
            self.session,
        )

        print(
            "SUMMARY =",
            summary_file,
        )

        print("=" * 72)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)

    print(
        "R1-A7 TEMPORAL PATH TRACKER V1 VALIDATION"
    )

    print("=" * 78)

    print(
        "radius        =",
        TEMPORAL_RADIUS,
        "px",
    )

    print(
        "jump trigger  =",
        PATH_JUMP_TRIGGER,
    )

    print(
        "support min   =",
        TEMPORAL_SUPPORT_MIN,
    )

    print(
        "hold max      =",
        HOLD_MAX_FRAMES,
        "frames",
    )

    print()

    print(
        "PINK   = U-Net wire"
    )

    print(
        "YELLOW = raw current path"
    )

    print(
        "RED    = temporal tracked path"
    )

    print(
        "BLUE   = tool"
    )

    print()

    print(
        "R = start / stop recording"
    )

    print(
        "Q / ESC = quit"
    )

    print()

    print(
        "NO DDS / NO LowCmd / NO ROBOT CONTROL"
    )

    print("=" * 78)

    model, device = (
        base.load_model()
    )

    trackers = {
        "external":
            TemporalTracker(
                "external"
            ),

        "right_wrist":
            TemporalTracker(
                "right_wrist"
            ),

        "left_wrist":
            TemporalTracker(
                "left_wrist"
            ),
    }

    recorder = Recorder()

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
            cv2.WINDOW_NORMAL,
        )

        while True:

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

            det_ext, fps_ext = (
                base.run_one(
                    model,
                    device,
                    frame_ext,
                )
            )

            det_right, fps_right = (
                base.run_one(
                    model,
                    device,
                    frame_right,
                )
            )

            det_left, fps_left = (
                base.run_one(
                    model,
                    device,
                    frame_left,
                )
            )

            tr_ext = (
                trackers[
                    "external"
                ].update(
                    det_ext
                )
            )

            tr_right = (
                trackers[
                    "right_wrist"
                ].update(
                    det_right
                )
            )

            tr_left = (
                trackers[
                    "left_wrist"
                ].update(
                    det_left
                )
            )

            vis_ext = (
                display_temporal(
                    frame_ext,
                    det_ext,
                    tr_ext,
                    "EXTERNAL",
                    fps_ext,
                )
            )

            vis_right = (
                display_temporal(
                    frame_right,
                    det_right,
                    tr_right,
                    "RIGHT WRIST",
                    fps_right,
                )
            )

            vis_left = (
                display_temporal(
                    frame_left,
                    det_left,
                    tr_left,
                    "LEFT WRIST",
                    fps_left,
                )
            )

            if recorder.active:

                recorder.add(
                    "external",
                    frame_ext,
                    vis_ext,
                    det_ext,
                    tr_ext,
                    fps_ext,
                )

                recorder.add(
                    "right_wrist",
                    frame_right,
                    vis_right,
                    det_right,
                    tr_right,
                    fps_right,
                )

                recorder.add(
                    "left_wrist",
                    frame_left,
                    vis_left,
                    det_left,
                    tr_left,
                    fps_left,
                )

                recorder.next_frame()

            panel = np.hstack(
                [
                    base.resize_panel(
                        vis_ext
                    ),

                    base.resize_panel(
                        vis_right
                    ),

                    base.resize_panel(
                        vis_left
                    ),
                ]
            )

            if recorder.active:

                cv2.putText(
                    panel,
                    "REC",
                    (
                        panel.shape[1]
                        - 70,
                        28,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (
                        0,
                        0,
                        255,
                    ),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(
                WINDOW,
                panel,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key in (
                ord("r"),
                ord("R"),
            ):

                if recorder.active:
                    recorder.stop()

                else:
                    recorder.start()

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

        if recorder.active:
            recorder.stop()

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
            "[INFO] Temporal V1 stopped"
        )


if __name__ == "__main__":
    main()
