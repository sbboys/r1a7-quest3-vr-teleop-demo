#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime
import argparse
import json
import os
import threading
import time

import cv2
import numpy as np

# Reuse the existing camera drivers.
# No checkpoint loading and no U-Net inference.
from live_hole_board_unet_three_camera_v1 import (
    ExternalCamera,
    OrbbecCamera,
    RIGHT_WRIST_SERIAL,
    LEFT_WRIST_SERIAL,
)

CAMERAS = (
    "external",
    "right_wrist",
    "left_wrist",
)

WINDOW = "R1-A7 MANUAL NEW BOARD CAPTURE V1"


def iso_time(t=None):
    if t is None:
        t = time.time()

    return datetime.fromtimestamp(
        t
    ).astimezone().isoformat(
        timespec="milliseconds"
    )


def default_session():

    p = Path(
        "/data/threading/current_new_board_session.txt"
    )

    if p.is_file():
        value = p.read_text(
            encoding="utf-8"
        ).strip()

        if value:
            return value

    return None


class FrameWorker:

    def __init__(self, camera_name, camera):

        self.name = camera_name
        self.camera = camera

        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        self.frame = None
        self.sequence = 0

        self.wall_time = None
        self.mono_time = None

        self.fps = 0.0
        self.last_error = None

        self.thread = threading.Thread(
            target=self.run,
            daemon=True,
            name=f"capture_{camera_name}",
        )

    def start(self):
        self.thread.start()

    def run(self):

        previous_time = None

        while not self.stop_event.is_set():

            try:

                frame = self.camera.read()

                wall = time.time()
                mono = time.monotonic()

                if frame is None:
                    time.sleep(0.002)
                    continue

                if previous_time is not None:

                    dt = mono - previous_time

                    if dt > 0:

                        instant = 1.0 / dt

                        self.fps = (
                            instant
                            if self.fps <= 0
                            else
                            0.85 * self.fps
                            + 0.15 * instant
                        )

                previous_time = mono

                with self.lock:

                    self.frame = frame.copy()

                    self.sequence += 1

                    self.wall_time = wall
                    self.mono_time = mono

                    self.last_error = None

            except Exception as exc:

                with self.lock:
                    self.last_error = str(exc)

                time.sleep(0.02)

    def snapshot(self):

        with self.lock:

            if self.frame is None:
                return None

            return {
                "frame": self.frame.copy(),
                "sequence": self.sequence,
                "wall_time": self.wall_time,
                "mono_time": self.mono_time,
                "fps": self.fps,
                "last_error": self.last_error,
            }

    def stop(self):

        self.stop_event.set()

        self.thread.join(
            timeout=3.0
        )


class CaptureStore:

    def __init__(
        self,
        session,
        max_age,
        max_skew,
    ):

        self.session = Path(session)

        self.max_age = float(max_age)
        self.max_skew = float(max_skew)

        self.image_dirs = {}

        for camera in CAMERAS:

            folder = (
                self.session
                / "images"
                / camera
            )

            folder.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.image_dirs[camera] = folder

        self.metadata = (
            self.session
            / "metadata"
        )

        self.metadata.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.manifest = (
            self.metadata
            / "capture_manifest.jsonl"
        )

        self.active = []

        self.next_number = 1
        self.scene_number = 0

        self.restore()

        self.create_config()

    @property
    def scene_id(self):

        return (
            f"scene_{self.scene_number:03d}"
        )

    def append_record(self, record):

        with self.manifest.open(
            "a",
            encoding="utf-8",
            buffering=1,
        ) as f:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            f.flush()
            os.fsync(f.fileno())

    def restore(self):

        if not self.manifest.exists():
            return

        active = {}
        max_number = 0

        for line in self.manifest.read_text(
            encoding="utf-8"
        ).splitlines():

            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except Exception:
                continue

            event = record.get("event")
            group_id = record.get("group_id")

            if event == "CAPTURE" and group_id:

                active[group_id] = record

                try:
                    number = int(
                        group_id.split("_")[-1]
                    )

                    max_number = max(
                        max_number,
                        number,
                    )

                except Exception:
                    pass

            elif event == "UNDO" and group_id:

                active.pop(
                    group_id,
                    None,
                )

            elif event == "SCENE_CHANGE":

                try:
                    self.scene_number = int(
                        record["scene_id"].split("_")[-1]
                    )
                except Exception:
                    pass

        # Only restore groups whose three files still exist.
        for group_id, record in active.items():

            valid = all(
                (
                    self.session
                    / record["frames"][camera]["path"]
                ).is_file()
                for camera in CAMERAS
            )

            if valid:
                self.active.append(record)

        self.next_number = max_number + 1

    def create_config(self):

        path = (
            self.metadata
            / "capture_config.json"
        )

        if path.exists():
            return

        config = {
            "board_id": "NEW_BOARD_01",
            "camera_order": list(CAMERAS),
            "external": {
                "device": "ALP USB Zoom Camera",
                "resolution": [1280, 720],
                "fps": 30,
            },
            "right_wrist": {
                "device": "Orbbec",
                "serial": RIGHT_WRIST_SERIAL,
                "resolution": [1280, 800],
                "fps": 30,
            },
            "left_wrist": {
                "device": "Orbbec",
                "serial": LEFT_WRIST_SERIAL,
                "resolution": [1280, 800],
                "fps": 30,
            },
            "model_inference": False,
            "robot_commands": False,
            "image_format": "PNG",
            "timestamp_type": "host_read_completion",
            "created_at": iso_time(),
        }

        path.write_text(
            json.dumps(
                config,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    def change_scene(self):

        self.scene_number += 1

        self.append_record({
            "event": "SCENE_CHANGE",
            "scene_id": self.scene_id,
            "timestamp": iso_time(),
            "timestamp_epoch": time.time(),
        })

        print(
            "[SCENE]",
            self.scene_id,
            flush=True,
        )

    def capture(self, workers):

        trigger_wall = time.time()
        trigger_mono = time.monotonic()

        snapshots = {}

        for camera in CAMERAS:

            snap = workers[camera].snapshot()

            if snap is None:

                print(
                    "[REJECT] no frame:",
                    camera,
                    flush=True,
                )

                return False

            snapshots[camera] = snap

        ages = {
            camera:
                trigger_mono
                - snapshots[camera]["mono_time"]
            for camera in CAMERAS
        }

        if any(
            age < 0 or age > self.max_age
            for age in ages.values()
        ):

            print(
                "[REJECT] stale frame:",
                ages,
                flush=True,
            )

            return False

        times = [
            snapshots[camera]["mono_time"]
            for camera in CAMERAS
        ]

        skew = max(times) - min(times)

        if skew > self.max_skew:

            print(
                f"[REJECT] camera timestamp skew "
                f"{skew:.3f}s > {self.max_skew:.3f}s",
                flush=True,
            )

            return False

        group_id = (
            f"group_{self.next_number:06d}"
        )

        written = []
        frame_info = {}

        try:

            for camera in CAMERAS:

                snap = snapshots[camera]
                frame = snap["frame"]

                folder = self.image_dirs[camera]

                final_path = (
                    folder
                    / f"{group_id}.png"
                )

                temp_path = (
                    folder
                    / f"{group_id}.tmp.png"
                )

                ok = cv2.imwrite(
                    str(temp_path),
                    frame,
                    [
                        cv2.IMWRITE_PNG_COMPRESSION,
                        3,
                    ],
                )

                if not ok:
                    raise RuntimeError(
                        f"Failed writing {temp_path}"
                    )

                os.replace(
                    temp_path,
                    final_path,
                )

                written.append(final_path)

                height, width = frame.shape[:2]

                frame_info[camera] = {
                    "path": str(
                        final_path.relative_to(
                            self.session
                        )
                    ),
                    "sequence": int(
                        snap["sequence"]
                    ),
                    "timestamp_epoch": float(
                        snap["wall_time"]
                    ),
                    "timestamp_iso": iso_time(
                        snap["wall_time"]
                    ),
                    "monotonic": float(
                        snap["mono_time"]
                    ),
                    "age_at_trigger_s": float(
                        ages[camera]
                    ),
                    "width": int(width),
                    "height": int(height),
                    "fps": float(
                        snap["fps"]
                    ),
                }

            record = {
                "event": "CAPTURE",
                "group_id": group_id,
                "scene_id": self.scene_id,
                "trigger_timestamp": iso_time(
                    trigger_wall
                ),
                "trigger_epoch": trigger_wall,
                "frame_skew_s": float(skew),
                "max_frame_age_s": float(
                    max(ages.values())
                ),
                "frames": frame_info,
            }

            self.append_record(record)

            self.active.append(record)

            self.next_number += 1

            print(
                f"[CAPTURE OK] {group_id} "
                f"scene={self.scene_id} "
                f"skew={skew:.3f}s "
                f"age={max(ages.values()):.3f}s",
                flush=True,
            )

            return True

        except Exception as exc:

            for path in written:

                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass

            print(
                "[CAPTURE ERROR]",
                repr(exc),
                flush=True,
            )

            return False

    def undo(self):

        if not self.active:

            print(
                "[UNDO] no capture group",
                flush=True,
            )

            return

        record = self.active.pop()

        group_id = record["group_id"]

        for camera in CAMERAS:

            path = (
                self.session
                / record["frames"][camera]["path"]
            )

            path.unlink(missing_ok=True)

        self.append_record({
            "event": "UNDO",
            "group_id": group_id,
            "timestamp": iso_time(),
            "note": "Image files removed; CAPTURE history retained.",
        })

        print(
            "[UNDO OK]",
            group_id,
            flush=True,
        )

    def stats(self):

        count = len(self.active)

        print()
        print("========== CAPTURE STATISTICS ==========")

        print("active_groups =", count)

        print("scene =", self.scene_id)

        print(
            "next_group =",
            f"group_{self.next_number:06d}",
        )

        for camera in CAMERAS:

            print(
                camera,
                "=",
                count,
            )

        print("=========================================")
        print(flush=True)


def make_tile(frame, label):

    width = 640
    height = 400

    canvas = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    canvas[:] = (30, 30, 30)

    if frame is None:

        cv2.putText(
            canvas,
            f"{label}: NO FRAME",
            (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        return canvas

    h, w = frame.shape[:2]

    scale = min(
        width / w,
        height / h,
    )

    nw = int(w * scale)
    nh = int(h * scale)

    resized = cv2.resize(
        frame,
        (nw, nh),
    )

    x = (width - nw) // 2
    y = (height - nh) // 2

    canvas[
        y:y + nh,
        x:x + nw,
    ] = resized

    cv2.rectangle(
        canvas,
        (0, 0),
        (639, 32),
        (20, 20, 20),
        -1,
    )

    cv2.putText(
        canvas,
        label,
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    return canvas


def make_info(store, workers):

    tile = np.zeros(
        (400, 640, 3),
        dtype=np.uint8,
    )

    tile[:] = (25, 25, 25)

    lines = [
        "R1-A7 MANUAL NEW BOARD CAPTURE",
        f"Scene: {store.scene_id}",
        f"Saved groups: {len(store.active)}",
        f"Next: group_{store.next_number:06d}",
        "",
        "S : capture 3 cameras",
        "R : next scene",
        "U : undo last group",
        "P : print statistics",
        "Q / ESC : quit",
        "",
        "NO MODEL INFERENCE",
        "NO ROBOT COMMANDS",
        "NO LOWCMD",
    ]

    y = 35

    for line in lines:

        color = (230, 230, 230)

        if "NO " in line:
            color = (80, 220, 80)

        cv2.putText(
            tile,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )

        y += 26

    y += 8

    for camera in CAMERAS:

        snap = workers[camera].snapshot()

        if snap is None:

            text = f"{camera}: waiting"

            color = (0, 0, 255)

        else:

            age = (
                time.monotonic()
                - snap["mono_time"]
            )

            text = (
                f"{camera}: "
                f"seq={snap['sequence']} "
                f"fps={snap['fps']:.1f} "
                f"age={age:.3f}s"
            )

            color = (
                (80, 220, 80)
                if age < 0.5
                else (0, 165, 255)
            )

        cv2.putText(
            tile,
            text,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            color,
            1,
        )

        y += 24

    return tile


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--session",
        default=os.environ.get(
            "R1A7_NEW_BOARD_SESSION"
        ) or default_session(),
    )

    parser.add_argument(
        "--max-frame-age",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--max-frame-skew",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--min-trigger-interval",
        type=float,
        default=0.30,
    )

    args = parser.parse_args()

    if not args.session:

        parser.error(
            "Missing capture session path."
        )

    session = Path(
        args.session
    ).expanduser().resolve()

    print(
        "========== MANUAL NEW BOARD CAPTURE V1 ==========",
        flush=True,
    )

    print(
        "SESSION =",
        session,
        flush=True,
    )

    print(
        "MODEL INFERENCE = DISABLED",
        flush=True,
    )

    print(
        "ROBOT COMMANDS = DISABLED",
        flush=True,
    )

    cameras = {}
    workers = {}

    try:

        print("[OPEN] External", flush=True)

        cameras["external"] = ExternalCamera()

        print("[OPEN] Right Wrist", flush=True)

        cameras["right_wrist"] = OrbbecCamera(
            RIGHT_WRIST_SERIAL,
            "right_wrist",
        )

        print("[OPEN] Left Wrist", flush=True)

        cameras["left_wrist"] = OrbbecCamera(
            LEFT_WRIST_SERIAL,
            "left_wrist",
        )

        for name in CAMERAS:

            workers[name] = FrameWorker(
                name,
                cameras[name],
            )

            workers[name].start()

        store = CaptureStore(
            session,
            args.max_frame_age,
            args.max_frame_skew,
        )

        cv2.namedWindow(
            WINDOW,
            cv2.WINDOW_NORMAL,
        )

        cv2.resizeWindow(
            WINDOW,
            1280,
            800,
        )

        print(
            "[READY] S=capture R=scene U=undo P=stats Q=quit",
            flush=True,
        )

        last_trigger = 0.0

        while True:

            snapshots = {
                name: workers[name].snapshot()
                for name in CAMERAS
            }

            ext = make_tile(
                snapshots["external"]["frame"]
                if snapshots["external"]
                else None,
                "EXTERNAL",
            )

            right = make_tile(
                snapshots["right_wrist"]["frame"]
                if snapshots["right_wrist"]
                else None,
                "RIGHT WRIST",
            )

            left = make_tile(
                snapshots["left_wrist"]["frame"]
                if snapshots["left_wrist"]
                else None,
                "LEFT WRIST",
            )

            info = make_info(
                store,
                workers,
            )

            canvas = np.vstack([
                np.hstack([ext, right]),
                np.hstack([left, info]),
            ])

            cv2.imshow(
                WINDOW,
                canvas,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):

                break

            elif key in (
                ord("s"),
                ord("S"),
            ):

                now = time.monotonic()

                if (
                    now - last_trigger
                    < args.min_trigger_interval
                ):

                    print(
                        "[SKIP] trigger debounce",
                        flush=True,
                    )

                    continue

                last_trigger = now

                store.capture(workers)

            elif key in (
                ord("r"),
                ord("R"),
            ):

                store.change_scene()

            elif key in (
                ord("u"),
                ord("U"),
            ):

                store.undo()

            elif key in (
                ord("p"),
                ord("P"),
            ):

                store.stats()

        store.stats()

    finally:

        print(
            "[SHUTDOWN] stopping frame workers",
            flush=True,
        )

        for worker in workers.values():
            worker.stop()

        for name in CAMERAS:

            camera = cameras.get(name)

            if camera is not None:

                try:
                    camera.close()

                except Exception as exc:

                    print(
                        "[CLOSE WARN]",
                        name,
                        repr(exc),
                    )

        cv2.destroyAllWindows()

        print(
            "[SHUTDOWN] all cameras closed",
            flush=True,
        )


if __name__ == "__main__":
    main()

