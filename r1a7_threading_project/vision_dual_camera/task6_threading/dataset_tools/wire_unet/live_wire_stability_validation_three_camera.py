#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime
import csv
import json
import time

import cv2
import numpy as np

import live_wire_unet_three_camera as base


OUT_ROOT = Path(
    "/home/robot/r1a7_datasets/wire_live_validation"
)

PATH_JUMP_THRESHOLD = 0.35
WIRE_PIXEL_JUMP_THRESHOLD = 0.50

WINDOW = "R1-A7 WIRE STABILITY VALIDATION"


def rel_jump(cur, prev):
    if prev is None:
        return 0.0
    if prev <= 1e-6:
        return 0.0
    return abs(cur - prev) / prev


def longest_false_streak(values):
    best = 0
    cur = 0

    for v in values:
        if not v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0

    return best


def percentile(values, q):
    if not values:
        return None
    return float(
        np.percentile(
            np.asarray(values, dtype=np.float64),
            q,
        )
    )


class Recorder:

    def __init__(self):
        self.active = False
        self.session_dir = None
        self.csv_fp = None
        self.writer = None
        self.start_mono = None
        self.frame_index = 0
        self.rows = []

        self.prev = {
            "external": {},
            "right_wrist": {},
            "left_wrist": {},
        }

        # Exact previous live input/probability for transition diagnosis.
        self.prev_frame = {
            "external": None,
            "right_wrist": None,
            "left_wrist": None,
        }

        self.prev_pwire = {
            "external": None,
            "right_wrist": None,
            "left_wrist": None,
        }

    def start(self):

        if self.active:
            return

        stamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        self.session_dir = (
            OUT_ROOT
            / f"motion_stability_{stamp}"
        )

        self.session_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        for d in (
            "anomaly_raw",
            "anomaly_overlay",
            "anomaly_masks",
        ):
            (
                self.session_dir / d
            ).mkdir()

        self.csv_fp = open(
            self.session_dir
            / "frame_metrics.csv",
            "w",
            newline="",
        )

        fields = [
            "frame_index",
            "elapsed_s",
            "wall_time",
            "camera",

            "wire_found",
            "tool_found",
            "path_found",
            "connect_ok",

            "wire_pixels",
            "tool_pixels",
            "path_pixels",

            "components",
            "bridge_count",
            "path_length_px",

            "infer_fps",

            "path_rel_jump",
            "wire_pixel_rel_jump",

            "anomaly",
            "anomaly_reason",
        ]

        self.writer = csv.DictWriter(
            self.csv_fp,
            fieldnames=fields,
        )

        self.writer.writeheader()
        self.csv_fp.flush()

        self.start_mono = (
            time.perf_counter()
        )

        self.frame_index = 0
        self.rows = []

        for k in self.prev:
            self.prev[k] = {}
            self.prev_frame[k] = None
            self.prev_pwire[k] = None

        self.active = True

        print()
        print("=" * 72)
        print("RECORDING STARTED")
        print("SESSION =", self.session_dir)
        print("=" * 72)
        print()

    def record_one(
        self,
        camera,
        frame,
        overlay,
        det,
        infer_fps,
    ):

        if not self.active:
            return

        wire = det["wire_mask"]
        tool = det["tool_mask"]
        path = det["wire_path"]

        wire_pixels = int(
            np.count_nonzero(wire)
        )

        tool_pixels = int(
            np.count_nonzero(tool)
        )

        path_pixels = int(
            np.count_nonzero(path)
        )

        wire_found = wire_pixels > 0
        tool_found = tool_pixels > 0
        path_found = path_pixels > 0
        connect_ok = bool(
            det["connect_ok"]
        )

        path_length = float(
            det["path_length_px"]
        )

        prev = self.prev[camera]

        path_jump = rel_jump(
            path_length,
            prev.get("path_length_px"),
        )

        wire_jump = rel_jump(
            float(wire_pixels),
            prev.get("wire_pixels"),
        )

        reasons = []

        if not wire_found:
            reasons.append("WIRE_LOST")

        if not tool_found:
            reasons.append("TOOL_LOST")

        if not path_found:
            reasons.append("PATH_LOST")

        if not connect_ok:
            reasons.append("CONNECT_FAIL")

        if (
            prev.get("path_length_px") is not None
            and
            path_jump > PATH_JUMP_THRESHOLD
        ):
            reasons.append(
                "PATH_JUMP"
            )

        if (
            prev.get("wire_pixels") is not None
            and
            wire_jump > WIRE_PIXEL_JUMP_THRESHOLD
        ):
            reasons.append(
                "WIRE_PIXEL_JUMP"
            )

        anomaly = bool(reasons)

        row = {
            "frame_index":
                self.frame_index,

            "elapsed_s":
                time.perf_counter()
                - self.start_mono,

            "wall_time":
                datetime.now().isoformat(
                    timespec="milliseconds"
                ),

            "camera":
                camera,

            "wire_found":
                int(wire_found),

            "tool_found":
                int(tool_found),

            "path_found":
                int(path_found),

            "connect_ok":
                int(connect_ok),

            "wire_pixels":
                wire_pixels,

            "tool_pixels":
                tool_pixels,

            "path_pixels":
                path_pixels,

            "components":
                det["components"],

            "bridge_count":
                det["bridge_count"],

            "path_length_px":
                path_length,

            "infer_fps":
                float(infer_fps),

            "path_rel_jump":
                path_jump,

            "wire_pixel_rel_jump":
                wire_jump,

            "anomaly":
                int(anomaly),

            "anomaly_reason":
                "|".join(reasons),
        }

        self.writer.writerow(
            row
        )

        self.rows.append(
            row
        )

        self.prev[camera] = {
            "path_length_px":
                path_length,

            "wire_pixels":
                float(wire_pixels),
        }

        # Do not JPEG-compress these. They are the diagnostic truth
        # from the actual live inference cycle.
        current_prev_frame = frame.copy()

        current_prev_pwire = None

        if "wire_probability" in det:
            current_prev_pwire = (
                det["wire_probability"].copy()
            )

        if anomaly:

            stem = (
                f"{self.frame_index:06d}_"
                f"{camera}"
            )

            # LOSSLESS current live frame.
            cv2.imwrite(
                str(
                    self.session_dir
                    / "anomaly_raw"
                    / f"{stem}.png"
                ),
                frame,
                [
                    cv2.IMWRITE_PNG_COMPRESSION,
                    1,
                ],
            )

            # Previous exact live frame, so a collapse transition
            # can be compared without JPEG recompression.
            if self.prev_frame[camera] is not None:

                prev_stem = (
                    f"{self.frame_index - 1:06d}_"
                    f"{camera}_PREV"
                )

                cv2.imwrite(
                    str(
                        self.session_dir
                        / "anomaly_raw"
                        / f"{prev_stem}.png"
                    ),
                    self.prev_frame[camera],
                    [
                        cv2.IMWRITE_PNG_COMPRESSION,
                        1,
                    ],
                )

            cv2.imwrite(
                str(
                    self.session_dir
                    / "anomaly_overlay"
                    / f"{stem}.jpg"
                ),
                overlay,
            )

            # Save the ACTUAL live U-Net probability map.
            if "wire_probability" in det:

                np.save(
                    self.session_dir
                    / "anomaly_masks"
                    / f"{stem}_pwire.npy",
                    det["wire_probability"].astype(
                        np.float32
                    ),
                )

            if self.prev_pwire[camera] is not None:

                np.save(
                    self.session_dir
                    / "anomaly_masks"
                    / (
                        f"{self.frame_index - 1:06d}_"
                        f"{camera}_PREV_pwire.npy"
                    ),
                    self.prev_pwire[camera].astype(
                        np.float32
                    ),
                )

            cv2.imwrite(
                str(
                    self.session_dir
                    / "anomaly_masks"
                    / f"{stem}_wire.png"
                ),
                wire.astype(
                    np.uint8
                ) * 255,
            )

            cv2.imwrite(
                str(
                    self.session_dir
                    / "anomaly_masks"
                    / f"{stem}_path.png"
                ),
                path.astype(
                    np.uint8
                ) * 255,
            )

            cv2.imwrite(
                str(
                    self.session_dir
                    / "anomaly_masks"
                    / f"{stem}_tool.png"
                ),
                tool.astype(
                    np.uint8
                ) * 255,
            )

        self.prev_frame[camera] = (
            current_prev_frame
        )

        self.prev_pwire[camera] = (
            current_prev_pwire
        )

    def next_cycle(self):

        if self.active:
            self.frame_index += 1

            if (
                self.frame_index % 10
                == 0
            ):
                self.csv_fp.flush()

    def stop(self):

        if not self.active:
            return None

        self.active = False

        if self.csv_fp is not None:
            self.csv_fp.flush()
            self.csv_fp.close()
            self.csv_fp = None

        summary = self.make_summary()

        summary_path = (
            self.session_dir
            / "summary.json"
        )

        summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False,
            )
        )

        print()
        print("=" * 72)
        print("RECORDING STOPPED")
        print("=" * 72)

        self.print_summary(
            summary
        )

        print()
        print(
            "SESSION =",
            self.session_dir,
        )

        print(
            "CSV     =",
            self.session_dir
            / "frame_metrics.csv",
        )

        print(
            "SUMMARY =",
            summary_path,
        )

        print("=" * 72)
        print()

        return summary

    def make_summary(self):

        summary = {
            "session":
                str(self.session_dir),

            "thresholds": {
                "path_rel_jump":
                    PATH_JUMP_THRESHOLD,

                "wire_pixel_rel_jump":
                    WIRE_PIXEL_JUMP_THRESHOLD,
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

            n = len(rows)

            if n == 0:
                continue

            wire_flags = [
                bool(r["wire_found"])
                for r in rows
            ]

            tool_flags = [
                bool(r["tool_found"])
                for r in rows
            ]

            path_flags = [
                bool(r["path_found"])
                for r in rows
            ]

            connect_flags = [
                bool(r["connect_ok"])
                for r in rows
            ]

            fps_values = [
                float(r["infer_fps"])
                for r in rows
            ]

            path_jumps = [
                float(r["path_rel_jump"])
                for r in rows
            ]

            anomalies = [
                bool(r["anomaly"])
                for r in rows
            ]

            summary["cameras"][
                camera
            ] = {
                "frames":
                    n,

                "wire_found_rate":
                    float(
                        np.mean(
                            wire_flags
                        )
                    ),

                "tool_found_rate":
                    float(
                        np.mean(
                            tool_flags
                        )
                    ),

                "path_found_rate":
                    float(
                        np.mean(
                            path_flags
                        )
                    ),

                "connect_rate":
                    float(
                        np.mean(
                            connect_flags
                        )
                    ),

                "wire_longest_dropout_frames":
                    longest_false_streak(
                        wire_flags
                    ),

                "tool_longest_dropout_frames":
                    longest_false_streak(
                        tool_flags
                    ),

                "path_longest_dropout_frames":
                    longest_false_streak(
                        path_flags
                    ),

                "infer_fps_median":
                    percentile(
                        fps_values,
                        50,
                    ),

                "infer_fps_p05":
                    percentile(
                        fps_values,
                        5,
                    ),

                "path_jump_median":
                    percentile(
                        path_jumps,
                        50,
                    ),

                "path_jump_p95":
                    percentile(
                        path_jumps,
                        95,
                    ),

                "path_jump_gt_threshold_rate":
                    float(
                        np.mean(
                            np.asarray(
                                path_jumps
                            )
                            >
                            PATH_JUMP_THRESHOLD
                        )
                    ),

                "anomaly_frames":
                    int(
                        np.sum(
                            anomalies
                        )
                    ),

                "anomaly_rate":
                    float(
                        np.mean(
                            anomalies
                        )
                    ),
            }

        return summary

    def print_summary(
        self,
        summary,
    ):

        print()
        print(
            "===== MOTION STABILITY SUMMARY ====="
        )

        for camera, s in (
            summary["cameras"].items()
        ):

            print()
            print(
                camera.upper()
            )

            print(
                "  frames             =",
                s["frames"],
            )

            print(
                "  wire found rate    =",
                f"{s['wire_found_rate']:.4f}",
            )

            print(
                "  tool found rate    =",
                f"{s['tool_found_rate']:.4f}",
            )

            print(
                "  path found rate    =",
                f"{s['path_found_rate']:.4f}",
            )

            print(
                "  connect rate       =",
                f"{s['connect_rate']:.4f}",
            )

            print(
                "  wire max dropout   =",
                s[
                    "wire_longest_dropout_frames"
                ],
                "frames",
            )

            print(
                "  tool max dropout   =",
                s[
                    "tool_longest_dropout_frames"
                ],
                "frames",
            )

            print(
                "  path max dropout   =",
                s[
                    "path_longest_dropout_frames"
                ],
                "frames",
            )

            print(
                "  FPS median / P05   =",
                f"{s['infer_fps_median']:.2f}",
                "/",
                f"{s['infer_fps_p05']:.2f}",
            )

            print(
                "  path jump P95      =",
                f"{s['path_jump_p95']:.4f}",
            )

            print(
                "  path jump >35%     =",
                f"{s['path_jump_gt_threshold_rate']:.4f}",
            )

            print(
                "  anomaly rate       =",
                f"{s['anomaly_rate']:.4f}",
            )


def draw_rec(
    panel,
    active,
):

    if active:

        cv2.circle(
            panel,
            (
                panel.shape[1] - 38,
                27,
            ),
            9,
            (
                0,
                0,
                255,
            ),
            -1,
        )

        cv2.putText(
            panel,
            "REC",
            (
                panel.shape[1] - 95,
                34,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (
                255,
                255,
                255,
            ),
            2,
            cv2.LINE_AA,
        )

    else:

        cv2.putText(
            panel,
            "R = START RECORD",
            (
                panel.shape[1] - 210,
                34,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (
                255,
                255,
                255,
            ),
            2,
            cv2.LINE_AA,
        )


def main():

    print("=" * 78)
    print(
        "R1-A7 THREE-CAMERA MOTION STABILITY VALIDATION"
    )
    print("=" * 78)

    print(
        "R       = start / stop recording"
    )

    print(
        "Q / ESC = quit"
    )

    print()
    print(
        "NO DDS / NO LowCmd / NO ROBOT CONTROL"
    )

    print("=" * 78)

    recorder = Recorder()

    model, device = (
        base.load_model()
    )

    external = None
    right = None
    left = None

    try:

        external = (
            base.open_external()
        )

        ctx = base.Context()

        right = base.open_orbbec(
            ctx,
            base.RIGHT_SERIAL,
        )

        left = base.open_orbbec(
            ctx,
            base.LEFT_SERIAL,
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
                or frame_right is None
                or frame_left is None
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

            vis_ext = (
                base.make_display(
                    frame_ext,
                    det_ext,
                    "EXTERNAL",
                    fps_ext,
                )
            )

            vis_right = (
                base.make_display(
                    frame_right,
                    det_right,
                    "RIGHT WRIST",
                    fps_right,
                )
            )

            vis_left = (
                base.make_display(
                    frame_left,
                    det_left,
                    "LEFT WRIST",
                    fps_left,
                )
            )

            if recorder.active:

                recorder.record_one(
                    "external",
                    frame_ext,
                    vis_ext,
                    det_ext,
                    fps_ext,
                )

                recorder.record_one(
                    "right_wrist",
                    frame_right,
                    vis_right,
                    det_right,
                    fps_right,
                )

                recorder.record_one(
                    "left_wrist",
                    frame_left,
                    vis_left,
                    det_left,
                    fps_left,
                )

                recorder.next_cycle()

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

            draw_rec(
                panel,
                recorder.active,
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
            "[INFO] validation stopped"
        )


if __name__ == "__main__":
    main()
