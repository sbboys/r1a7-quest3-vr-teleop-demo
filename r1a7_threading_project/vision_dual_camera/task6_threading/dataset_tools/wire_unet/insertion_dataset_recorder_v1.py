#!/usr/bin/env python3
from __future__ import annotations

import atexit
import copy
import json
import math
import os
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np


CAMERAS = (
    "external",
    "right_wrist",
    "left_wrist",
)

ACTIVE_PHASES = {
    "WAIT_HOLE_TARGET",
    "INSERTION_TRACKING",
}

DEFAULT_ROOT = Path(
    os.environ.get(
        "R1A7_INSERTION_DATASET_ROOT",
        "/home/robot/r1a7_datasets/wire_unet/"
        "insertion_runtime_001/raw_sessions",
    )
)

FUSION_JSON = Path(
    "/tmp/r1a7_wire_grasp_fusion_shadow_v1.json"
)

GATE_JSON = Path(
    "/tmp/r1a7_dual_ibvs_velocity.json"
)

TARGET_JSON = Path(
    "/tmp/r1a7_insertion_target_state_v1.json"
)


def _read_json_quiet(path: Path):
    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return {}


def _jsonable(value):
    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        if (
            isinstance(value, float)
            and
            not math.isfinite(value)
        ):
            return None

        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return _jsonable(
            value.item()
        )

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, dict):
        return {
            str(k): _jsonable(v)
            for k, v in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return [
            _jsonable(v)
            for v in value
        ]

    return str(value)


def _uv(value):
    if value is None:
        return None

    try:
        if len(value) < 2:
            return None

        u = float(value[0])
        v = float(value[1])

        if not (
            math.isfinite(u)
            and
            math.isfinite(v)
        ):
            return None

        return (
            u,
            v,
        )

    except Exception:
        return None


def _distance(a, b):
    a = _uv(a)
    b = _uv(b)

    if (
        a is None
        or
        b is None
    ):
        return None

    return math.hypot(
        a[0] - b[0],
        a[1] - b[1],
    )


def _free_uv_from_guidance(d):
    if not isinstance(d, dict):
        return None

    for key in (
        "uv",
        "free_uv",
        "free_target_uv",
    ):
        value = _uv(
            d.get(key)
        )

        if value is not None:
            return value

    return None


class InsertionDatasetRecorder:

    def __init__(
        self,
        root=DEFAULT_ROOT,
        cameras=CAMERAS,
        normal_hz=5.0,
        event_hz=10.0,
        jpeg_quality=95,
        max_queue=128,
    ):
        self.root = Path(root)
        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.cameras = tuple(
            cameras
        )

        self.normal_period = (
            1.0
            /
            max(
                float(normal_hz),
                0.1,
            )
        )

        self.event_period = (
            1.0
            /
            max(
                float(event_hz),
                0.1,
            )
        )

        self.jpeg_quality = int(
            jpeg_quality
        )

        self.q = queue.Queue(
            maxsize=int(
                max_queue
            )
        )

        self.worker = threading.Thread(
            target=self._writer_loop,
            name="InsertionDatasetWriter",
            daemon=True,
        )

        self.worker.start()

        self.session_active = False
        self.session_id = None
        self.session_dir = None

        self.session_started_wall = None

        self.last_normal_save = 0.0
        self.last_event_save = 0.0

        self.last_phase = None
        self.last_event_signature = ()

        self.prev_free_uv = {
            camera: None
            for camera in self.cameras
        }

        self.prev_hole_uv = {
            camera: None
            for camera in self.cameras
        }

        self.prev_free_source = {
            camera: None
            for camera in self.cameras
        }

        self.prev_selected = {
            camera: False
            for camera in self.cameras
        }

        self.prev_valid = {
            camera: False
            for camera in self.cameras
        }

        self.runtime_cache = {
            "fusion": {},
            "gate": {},
            "target_state": {},
        }

        self.last_runtime_poll = 0.0

        self.dropped_queue = 0
        self.closed = False

        atexit.register(
            self.close
        )

    # --------------------------------------------------------
    # Session management
    # --------------------------------------------------------

    def _start_session(
        self,
        phase,
        frame_counter,
        extra=None,
    ):
        if self.session_active:
            return

        now = time.time()

        stamp = time.strftime(
            "%Y%m%d_%H%M%S",
            time.localtime(now),
        )

        millis = int(
            (now % 1.0) * 1000.0
        )

        self.session_id = (
            f"{stamp}_{millis:03d}"
        )

        self.session_dir = (
            self.root
            /
            self.session_id
        )

        for camera in self.cameras:
            (
                self.session_dir
                /
                "images"
                /
                camera
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

        self.session_started_wall = now
        self.session_active = True

        self.last_normal_save = 0.0
        self.last_event_save = 0.0
        self.last_event_signature = ()

        self.prev_free_uv = {
            camera: None
            for camera in self.cameras
        }

        self.prev_hole_uv = {
            camera: None
            for camera in self.cameras
        }

        self.prev_free_source = {
            camera: None
            for camera in self.cameras
        }

        self.prev_selected = {
            camera: False
            for camera in self.cameras
        }

        self.prev_valid = {
            camera: False
            for camera in self.cameras
        }

        item = {
            "kind": "session_start",
            "session_id":
                self.session_id,
            "session_dir":
                str(
                    self.session_dir
                ),
            "timestamp":
                now,
            "phase":
                phase,
            "frame_counter":
                int(
                    frame_counter
                ),
            "extra":
                _jsonable(
                    extra or {}
                ),
        }

        self._put_control(
            item
        )

        print(
            "[DATA RECORDER] START",
            self.session_dir,
            flush=True,
        )

    def _stop_session(
        self,
        reason,
    ):
        if not self.session_active:
            return

        item = {
            "kind": "session_stop",
            "timestamp":
                time.time(),
            "reason":
                str(reason),
            "dropped_queue":
                int(
                    self.dropped_queue
                ),
        }

        self._put_control(
            item
        )

        print(
            "[DATA RECORDER] STOP",
            self.session_dir,
            "reason=",
            reason,
            flush=True,
        )

        self.session_active = False
        self.session_id = None
        self.session_dir = None
        self.session_started_wall = None

    # --------------------------------------------------------
    # Runtime state polling
    # --------------------------------------------------------

    def _poll_runtime(
        self,
        now_mono,
    ):
        if (
            now_mono
            -
            self.last_runtime_poll
            <
            0.10
        ):
            return

        self.last_runtime_poll = (
            now_mono
        )

        self.runtime_cache = {
            "fusion":
                _read_json_quiet(
                    FUSION_JSON
                ),
            "gate":
                _read_json_quiet(
                    GATE_JSON
                ),
            "target_state":
                _read_json_quiet(
                    TARGET_JSON
                ),
        }

    # --------------------------------------------------------
    # Event detection
    # --------------------------------------------------------

    def _detect_events(
        self,
        phase,
        free_guidance,
        target_uv,
        target_selected,
        target_valid,
        external_guidance,
        wrist_guidance,
    ):
        events = []

        if (
            self.last_phase
            is not None
            and
            phase
            !=
            self.last_phase
        ):
            events.append(
                "PHASE_CHANGE"
            )

        fusion = (
            self.runtime_cache.get(
                "fusion",
                {},
            )
        )

        gate = (
            self.runtime_cache.get(
                "gate",
                {},
            )
        )

        target_state = (
            self.runtime_cache.get(
                "target_state",
                {},
            )
        )

        fusion_mode = str(
            fusion.get(
                "mode",
                fusion.get(
                    "fusion_mode",
                    "",
                ),
            )
            or
            ""
        ).upper()

        gate_state = str(
            gate.get(
                "gate_state",
                gate.get(
                    "mode",
                    "",
                ),
            )
            or
            ""
        ).upper()

        if "STOP" in fusion_mode:
            events.append(
                "FUSION_STOP"
            )

        if "BLIND_GRACE" in fusion_mode:
            events.append(
                "BLIND_GRACE"
            )

        if (
            "DUAL_Z_CONFLICT"
            in fusion_mode
            or
            "DUAL_Z_CONFLICT"
            in gate_state
        ):
            events.append(
                "DUAL_Z_CONFLICT"
            )

        if (
            phase
            ==
            "INSERTION_TRACKING"
            and
            gate
            and
            gate.get(
                "enabled"
            )
            is False
        ):
            events.append(
                "GATE_DISABLED"
            )

        if target_state.get(
            "degraded_single_view",
            False,
        ):
            events.append(
                "SINGLE_VIEW"
            )

        for camera in self.cameras:

            guidance = (
                free_guidance.get(
                    camera,
                    {},
                )
                if isinstance(
                    free_guidance,
                    dict,
                )
                else {}
            )

            source = str(
                guidance.get(
                    "source",
                    "",
                )
                or
                ""
            ).upper()

            if (
                source
                and
                source
                !=
                "MEASURED"
            ):
                events.append(
                    f"{camera}:FREE_SOURCE_{source}"
                )

            prev_source = (
                self.prev_free_source[
                    camera
                ]
            )

            if (
                prev_source
                is not None
                and
                source
                !=
                prev_source
            ):
                events.append(
                    f"{camera}:FREE_SOURCE_CHANGE"
                )

            current_free = (
                _free_uv_from_guidance(
                    guidance
                )
            )

            jump = _distance(
                current_free,
                self.prev_free_uv[
                    camera
                ],
            )

            if (
                jump is not None
                and
                jump
                >
                35.0
            ):
                events.append(
                    f"{camera}:FREE_JUMP"
                )

            current_hole = _uv(
                (
                    target_uv.get(
                        camera
                    )
                    if isinstance(
                        target_uv,
                        dict,
                    )
                    else None
                )
            )

            hole_jump = _distance(
                current_hole,
                self.prev_hole_uv[
                    camera
                ],
            )

            if (
                hole_jump is not None
                and
                hole_jump
                >
                25.0
            ):
                events.append(
                    f"{camera}:HOLE_JUMP"
                )

            selected = bool(
                (
                    target_selected.get(
                        camera,
                        False,
                    )
                    if isinstance(
                        target_selected,
                        dict,
                    )
                    else False
                )
            )

            valid = bool(
                (
                    target_valid.get(
                        camera,
                        False,
                    )
                    if isinstance(
                        target_valid,
                        dict,
                    )
                    else False
                )
            )

            if (
                selected
                and
                not valid
            ):
                events.append(
                    f"{camera}:TARGET_INVALID"
                )

            if (
                selected
                !=
                self.prev_selected[
                    camera
                ]
            ):
                events.append(
                    f"{camera}:TARGET_SELECTED_CHANGE"
                )

            if (
                valid
                !=
                self.prev_valid[
                    camera
                ]
            ):
                events.append(
                    f"{camera}:TARGET_VALID_CHANGE"
                )

            self.prev_free_uv[
                camera
            ] = current_free

            self.prev_hole_uv[
                camera
            ] = current_hole

            self.prev_free_source[
                camera
            ] = source

            self.prev_selected[
                camera
            ] = selected

            self.prev_valid[
                camera
            ] = valid

        for name, guidance in (
            (
                "external",
                external_guidance,
            ),
            (
                "right_wrist",
                wrist_guidance,
            ),
        ):
            if (
                isinstance(
                    guidance,
                    dict,
                )
                and
                guidance
                and
                guidance.get(
                    "guidance_valid"
                )
                is False
            ):
                events.append(
                    f"{name}:GUIDANCE_INVALID"
                )

        return sorted(
            set(events)
        )

    # --------------------------------------------------------
    # Main entry point called by visual loop
    # --------------------------------------------------------

    def record(
        self,
        frame_counter,
        phase,
        frames,
        free_guidance,
        target_uv,
        target_selected,
        target_valid,
        external_guidance=None,
        wrist_guidance=None,
        extra=None,
    ):
        if self.closed:
            return

        now_wall = time.time()
        now_mono = time.monotonic()

        if (
            phase
            in
            ACTIVE_PHASES
        ):
            if not self.session_active:
                self._start_session(
                    phase,
                    frame_counter,
                    extra=extra,
                )

        else:
            if self.session_active:
                self._stop_session(
                    "phase_left_insertion_dataset_window"
                )

            self.last_phase = phase
            return

        self._poll_runtime(
            now_mono
        )

        events = self._detect_events(
            phase=phase,
            free_guidance=(
                free_guidance
                or {}
            ),
            target_uv=(
                target_uv
                or {}
            ),
            target_selected=(
                target_selected
                or {}
            ),
            target_valid=(
                target_valid
                or {}
            ),
            external_guidance=(
                external_guidance
                or {}
            ),
            wrist_guidance=(
                wrist_guidance
                or {}
            ),
        )

        signature = tuple(
            events
        )

        recovery = bool(
            self.last_event_signature
            and
            not signature
        )

        if recovery:
            events = [
                "RECOVERY"
            ]

            signature = (
                "RECOVERY",
            )

        event_changed = (
            signature
            !=
            self.last_event_signature
        )

        normal_due = bool(
            now_mono
            -
            self.last_normal_save
            >=
            self.normal_period
        )

        event_due = bool(
            events
            and
            (
                event_changed
                or
                now_mono
                -
                self.last_event_save
                >=
                self.event_period
            )
        )

        phase_changed = bool(
            self.last_phase
            is not None
            and
            phase
            !=
            self.last_phase
        )

        should_save = bool(
            normal_due
            or
            event_due
            or
            phase_changed
        )

        self.last_phase = phase
        self.last_event_signature = (
            signature
        )

        if not should_save:
            return

        if event_due:
            self.last_event_save = (
                now_mono
            )

        if normal_due:
            self.last_normal_save = (
                now_mono
            )

        save_reason = (
            "EVENT"
            if events
            else
            "PERIODIC"
        )

        frame_copies = {}

        if isinstance(
            frames,
            dict,
        ):
            for camera in self.cameras:
                image = frames.get(
                    camera
                )

                if (
                    isinstance(
                        image,
                        np.ndarray,
                    )
                    and
                    image.size
                    >
                    0
                ):
                    frame_copies[
                        camera
                    ] = image.copy()

        record = {
            "timestamp":
                now_wall,

            "monotonic":
                now_mono,

            "frame_counter":
                int(
                    frame_counter
                ),

            "phase":
                str(
                    phase
                ),

            "save_reason":
                save_reason,

            "events":
                list(
                    events
                ),

            "free_guidance":
                _jsonable(
                    free_guidance
                    or {}
                ),

            "target_uv":
                _jsonable(
                    target_uv
                    or {}
                ),

            "target_selected":
                _jsonable(
                    target_selected
                    or {}
                ),

            "target_valid":
                _jsonable(
                    target_valid
                    or {}
                ),

            "external_guidance":
                _jsonable(
                    external_guidance
                    or {}
                ),

            "wrist_guidance":
                _jsonable(
                    wrist_guidance
                    or {}
                ),

            "fusion":
                _jsonable(
                    self.runtime_cache.get(
                        "fusion",
                        {},
                    )
                ),

            "gate":
                _jsonable(
                    self.runtime_cache.get(
                        "gate",
                        {},
                    )
                ),

            "target_state":
                _jsonable(
                    self.runtime_cache.get(
                        "target_state",
                        {},
                    )
                ),

            "extra":
                _jsonable(
                    extra
                    or {}
                ),
        }

        item = {
            "kind":
                "frame",

            "session_dir":
                str(
                    self.session_dir
                ),

            "frame_counter":
                int(
                    frame_counter
                ),

            "timestamp":
                now_wall,

            "frames":
                frame_copies,

            "record":
                record,
        }

        try:
            self.q.put_nowait(
                item
            )

        except queue.Full:
            self.dropped_queue += 1

            if (
                self.dropped_queue
                %
                20
                ==
                1
            ):
                print(
                    "[DATA RECORDER] WARNING "
                    "writer queue full; dropped=",
                    self.dropped_queue,
                    flush=True,
                )

    # --------------------------------------------------------
    # Writer thread
    # --------------------------------------------------------

    def _put_control(
        self,
        item,
    ):
        try:
            self.q.put(
                item,
                timeout=1.0,
            )
        except queue.Full:
            print(
                "[DATA RECORDER] ERROR "
                "control queue full",
                flush=True,
            )

    def _writer_loop(
        self,
    ):
        record_fp = None
        active_session_dir = None

        saved_records = 0
        saved_images = 0
        event_records = 0
        event_counts = {}
        session_start_ts = None

        while True:

            item = self.q.get()

            try:
                kind = item.get(
                    "kind"
                )

                if (
                    kind
                    ==
                    "session_start"
                ):
                    if record_fp is not None:
                        record_fp.close()

                    active_session_dir = Path(
                        item[
                            "session_dir"
                        ]
                    )

                    active_session_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    record_fp = (
                        active_session_dir
                        /
                        "records.jsonl"
                    ).open(
                        "a",
                        encoding="utf-8",
                        buffering=1,
                    )

                    session_start_ts = (
                        item.get(
                            "timestamp"
                        )
                    )

                    saved_records = 0
                    saved_images = 0
                    event_records = 0
                    event_counts = {}

                    (
                        active_session_dir
                        /
                        "session_start.json"
                    ).write_text(
                        json.dumps(
                            _jsonable(
                                item
                            ),
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )

                elif (
                    kind
                    ==
                    "frame"
                ):
                    if (
                        record_fp is None
                        or
                        active_session_dir
                        is None
                    ):
                        continue

                    frame_counter = int(
                        item[
                            "frame_counter"
                        ]
                    )

                    ts_ms = int(
                        float(
                            item[
                                "timestamp"
                            ]
                        )
                        *
                        1000.0
                    )

                    image_paths = {}

                    for (
                        camera,
                        image,
                    ) in item[
                        "frames"
                    ].items():

                        filename = (
                            f"{frame_counter:08d}_"
                            f"{ts_ms}.jpg"
                        )

                        path = (
                            active_session_dir
                            /
                            "images"
                            /
                            camera
                            /
                            filename
                        )

                        ok = cv2.imwrite(
                            str(path),
                            image,
                            [
                                cv2.IMWRITE_JPEG_QUALITY,
                                self.jpeg_quality,
                            ],
                        )

                        if ok:
                            image_paths[
                                camera
                            ] = str(
                                path.relative_to(
                                    active_session_dir
                                )
                            )

                            saved_images += 1

                    record = copy.deepcopy(
                        item[
                            "record"
                        ]
                    )

                    record[
                        "image_paths"
                    ] = image_paths

                    record_fp.write(
                        json.dumps(
                            _jsonable(
                                record
                            ),
                            ensure_ascii=False,
                        )
                        +
                        "\n"
                    )

                    saved_records += 1

                    events = (
                        record.get(
                            "events",
                            [],
                        )
                        or
                        []
                    )

                    if events:
                        event_records += 1

                    for event in events:
                        event_counts[
                            event
                        ] = (
                            event_counts.get(
                                event,
                                0,
                            )
                            +
                            1
                        )

                elif (
                    kind
                    ==
                    "session_stop"
                ):
                    if (
                        active_session_dir
                        is not None
                    ):
                        summary = {
                            "session_start_timestamp":
                                session_start_ts,

                            "session_stop_timestamp":
                                item.get(
                                    "timestamp"
                                ),

                            "stop_reason":
                                item.get(
                                    "reason"
                                ),

                            "saved_records":
                                saved_records,

                            "saved_images":
                                saved_images,

                            "event_records":
                                event_records,

                            "event_counts":
                                event_counts,

                            "dropped_queue":
                                item.get(
                                    "dropped_queue",
                                    0,
                                ),
                        }

                        (
                            active_session_dir
                            /
                            "session_summary.json"
                        ).write_text(
                            json.dumps(
                                summary,
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )

                    if record_fp is not None:
                        record_fp.flush()
                        record_fp.close()

                    record_fp = None
                    active_session_dir = None

                elif (
                    kind
                    ==
                    "shutdown"
                ):
                    if record_fp is not None:
                        record_fp.flush()
                        record_fp.close()

                    break

            except Exception as exc:
                print(
                    "[DATA RECORDER] WRITER ERROR:",
                    repr(exc),
                    flush=True,
                )

            finally:
                self.q.task_done()

    # --------------------------------------------------------
    # Shutdown
    # --------------------------------------------------------

    def close(
        self,
    ):
        if self.closed:
            return

        self.closed = True

        if self.session_active:
            self._stop_session(
                "recorder_close"
            )

        try:
            self.q.join()
        except Exception:
            pass

        try:
            self.q.put(
                {
                    "kind":
                        "shutdown"
                },
                timeout=1.0,
            )

            self.q.join()

        except Exception:
            pass

        try:
            self.worker.join(
                timeout=2.0
            )
        except Exception:
            pass


def _self_test():

    root = Path(
        "/tmp/r1a7_insertion_recorder_selftest"
    )

    recorder = InsertionDatasetRecorder(
        root=root,
        normal_hz=20.0,
        event_hz=20.0,
    )

    frames = {
        camera:
            np.zeros(
                (
                    64,
                    96,
                    3,
                ),
                dtype=np.uint8,
            )
        for camera
        in CAMERAS
    }

    free = {
        camera: {
            "source":
                "MEASURED",
            "uv":
                [
                    40.0,
                    30.0,
                ],
        }
        for camera
        in CAMERAS
    }

    target_uv = {
        "external":
            [
                50.0,
                30.0,
            ],
        "right_wrist":
            [
                50.0,
                30.0,
            ],
        "left_wrist":
            None,
    }

    selected = {
        "external":
            True,
        "right_wrist":
            True,
        "left_wrist":
            False,
    }

    valid = {
        "external":
            True,
        "right_wrist":
            True,
        "left_wrist":
            False,
    }

    for frame_counter in range(
        1,
        8,
    ):
        recorder.record(
            frame_counter=
                frame_counter,

            phase=
                "WAIT_HOLE_TARGET"
                if
                frame_counter
                <
                4
                else
                "INSERTION_TRACKING",

            frames=
                frames,

            free_guidance=
                free,

            target_uv=
                target_uv,

            target_selected=
                selected,

            target_valid=
                valid,

            external_guidance={
                "guidance_valid":
                    True,
            },

            wrist_guidance={
                "guidance_valid":
                    True,
            },

            extra={
                "self_test":
                    True,
            },
        )

        time.sleep(
            0.06
        )

    recorder.record(
        frame_counter=8,
        phase="GRASP",
        frames=frames,
        free_guidance=free,
        target_uv=target_uv,
        target_selected=selected,
        target_valid=valid,
    )

    recorder.close()

    print(
        "SELF_TEST_OK root=",
        root,
    )


if __name__ == "__main__":
    _self_test()
