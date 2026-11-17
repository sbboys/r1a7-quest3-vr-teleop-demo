#!/usr/bin/env python3

import csv
import json
import os
import queue
import threading
import time
from pathlib import Path


class FullChainAsyncRecorder:
    """
    Passive runtime recorder.

    IMPORTANT:
      - no robot commands
      - no IK changes
      - no velocity changes
      - no LowCmd changes

    Controller thread only calls submit().
    File IO and upstream JSON reads happen in the worker thread.
    """

    SOURCE_PATHS = {
        "external":
            Path("/tmp/r1a7_external_wire_grasp_v1.json"),

        "wrist":
            Path("/tmp/r1a7_wrist_wire_grasp_guidance.json"),

        "fusion":
            Path("/tmp/r1a7_wire_grasp_fusion_shadow_v1.json"),

        "gate":
            Path("/tmp/r1a7_dual_ibvs_velocity.json"),

        "insertion":
            Path("/tmp/r1a7_insertion_target_state_v1.json"),

        "settle":
            Path("/tmp/r1a7_auto_grasp_close_settle_v1.json"),
    }

    FIELDNAMES = [
        "sample_wall",
        "sample_monotonic",
        "loop_dt_s",

        # Visual External.
        "ext_timestamp",
        "ext_guidance_valid",
        "ext_source_frame_seq",
        "ext_visual_cycle_id",
        "ext_source_age_publish_ms",
        "ext_du_px",
        "ext_dv_px",
        "ext_vx_mm_s",
        "ext_vy_mm_s",
        "ext_vz_mm_s",

        # Visual Right Wrist.
        "wrist_timestamp",
        "wrist_guidance_valid",
        "wrist_source_frame_seq",
        "wrist_visual_cycle_id",
        "wrist_source_age_publish_ms",
        "wrist_du_px",
        "wrist_dv_px",
        "wrist_tip_grasp_distance_px",
        "wrist_jaw_corridor_ok",
        "wrist_jaw_corridor_latched",
        "wrist_vx_mm_s",
        "wrist_vy_mm_s",
        "wrist_vz_mm_s",

        # Fusion.
        "fusion_timestamp",
        "fusion_mode",
        "fusion_grasp_sequence_stage",
        "fusion_z_control_source",
        "fusion_external_valid",
        "fusion_wrist_valid",
        "fusion_vx_mm_s",
        "fusion_vy_mm_s",
        "fusion_vz_mm_s",
        "fusion_forward_nudge_active",
        "fusion_forward_nudge_done",
        "fusion_close_ready",

        # Gate file observed by recorder.
        "gate_current_timestamp",
        "gate_current_enabled",
        "gate_current_state",
        "gate_current_vx_mm_s",
        "gate_current_vy_mm_s",
        "gate_current_vz_mm_s",

        # EXACT Gate JSON consumed by Controller.
        "gate_consumed_timestamp",
        "gate_consumed_enabled",
        "gate_consumed_state",
        "gate_consumed_vx_mm_s",
        "gate_consumed_vy_mm_s",
        "gate_consumed_vz_mm_s",
        "gate_consumed_proposed_vx_mm_s",
        "gate_consumed_proposed_vy_mm_s",
        "gate_consumed_proposed_vz_mm_s",

        # Controller Cartesian path.
        "controller_ibvs_active",
        "controller_ibvs_vx_mm_s",
        "controller_ibvs_vy_mm_s",
        "controller_ibvs_vz_mm_s",

        "controller_cart_vx_mm_s",
        "controller_cart_vy_mm_s",
        "controller_cart_vz_mm_s",

        "command_x_mm",
        "command_y_mm",
        "command_z_mm",

        "actual_x_mm",
        "actual_y_mm",
        "actual_z_mm",

        "cartesian_lag_mm",
        "ik_solve_ms",

        "jacobian_min_singular",
        "jacobian_condition",

        "lowcmd_publish_count",
        "mode_machine",

        # Robot / IK / LowCmd arrays.
        "arm_q_actual",
        "arm_dq_actual",
        "arm_tau_est_actual",

        "arm_q_ik_raw",
        "arm_q_stream_goal",

        "arm_q_lowcmd_sent",
        "arm_dq_lowcmd_sent",
        "arm_tau_ff_sent",

        "gripper_q_actual",
        "gripper_dq_actual",

        # Auto grasp.
        "auto_grasp_state",
        "auto_grasp_open_issued",
        "auto_grasp_close_issued",

        # Task state.
        "insertion_phase",
        "insertion_target_ready",
        "insertion_target_confirmed",

        "close_settle_active",

        # Full raw snapshots.
        "external_json",
        "wrist_json",
        "fusion_json",
        "gate_current_json",
        "gate_consumed_json",
        "insertion_json",
        "settle_json",
    ]

    def __init__(
        self,
        path,
        metadata=None,
    ):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.meta_path = self.path.with_suffix(
            ".meta.json"
        )

        self._queue = queue.Queue(
            maxsize=8192
        )

        self._stop = threading.Event()

        self._cache = {}

        self.rows = 0
        self.dropped = 0
        self.error = None

        self._file = self.path.open(
            "w",
            newline="",
            encoding="utf-8",
            buffering=1,
        )

        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self.FIELDNAMES,
        )

        self._writer.writeheader()
        self._file.flush()

        meta = {
            "created_at":
                time.time(),

            "recording_version":
                "FULL_CHAIN_ASYNC_V1",

            "csv":
                str(self.path),

            "source_paths": {
                name: str(path)
                for name, path
                in self.SOURCE_PATHS.items()
            },
        }

        if metadata:
            meta.update(metadata)

        self.meta_path.write_text(
            json.dumps(
                meta,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        self._last_fsync = (
            time.monotonic()
        )

        self._thread = threading.Thread(
            target=self._run,
            name="r1a7-full-chain-recorder",
            daemon=True,
        )

        self._thread.start()

    @staticmethod
    def _json_text(value):
        try:
            return json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        except Exception:
            return "{}"

    @staticmethod
    def _array_text(value):
        try:
            return json.dumps(
                list(value),
                separators=(",", ":"),
            )
        except Exception:
            return "[]"

    @staticmethod
    def _xyz(value):
        try:
            return (
                float(value[0]),
                float(value[1]),
                float(value[2]),
            )
        except Exception:
            return (
                None,
                None,
                None,
            )

    def _read_json_cached(
        self,
        name,
    ):
        path = self.SOURCE_PATHS[name]

        previous = self._cache.get(
            name,
            (None, {}),
        )

        try:
            stat = path.stat()
            mtime = stat.st_mtime_ns

            if mtime == previous[0]:
                return previous[1]

            data = json.loads(
                path.read_text()
            )

            self._cache[name] = (
                mtime,
                data,
            )

            return data

        except Exception:
            return previous[1]

    def submit(
        self,
        sample,
    ):
        if self.error is not None:
            return False

        try:
            self._queue.put_nowait(
                sample
            )
            return True

        except queue.Full:
            self.dropped += 1
            return False

    def _build_row(
        self,
        sample,
    ):
        ext = self._read_json_cached(
            "external"
        )

        wrist = self._read_json_cached(
            "wrist"
        )

        fusion = self._read_json_cached(
            "fusion"
        )

        gate = self._read_json_cached(
            "gate"
        )

        insertion = self._read_json_cached(
            "insertion"
        )

        settle = self._read_json_cached(
            "settle"
        )

        gate_used = sample.get(
            "gate_consumed_json",
            {},
        )

        ibvs = self._xyz(
            sample.get(
                "ibvs_velocity_mm_s",
                [0, 0, 0],
            )
        )

        cart = self._xyz(
            sample.get(
                "cartesian_velocity_mm_s",
                [0, 0, 0],
            )
        )

        cmd = self._xyz(
            sample.get(
                "command_xyz_mm",
                [0, 0, 0],
            )
        )

        actual = self._xyz(
            sample.get(
                "actual_xyz_mm",
                [0, 0, 0],
            )
        )

        return {
            "sample_wall":
                sample.get(
                    "sample_wall"
                ),

            "sample_monotonic":
                sample.get(
                    "sample_monotonic"
                ),

            "loop_dt_s":
                sample.get(
                    "loop_dt_s"
                ),

            # External.
            "ext_timestamp":
                ext.get("timestamp"),

            "ext_guidance_valid":
                ext.get("guidance_valid"),

            "ext_source_frame_seq":
                ext.get("source_frame_seq"),

            "ext_visual_cycle_id":
                ext.get("visual_cycle_id"),

            "ext_source_age_publish_ms":
                ext.get(
                    "source_frame_age_ms_at_publish"
                ),

            "ext_du_px":
                ext.get("du_px"),

            "ext_dv_px":
                ext.get("dv_px"),

            "ext_vx_mm_s":
                ext.get("proposed_vx_mm_s"),

            "ext_vy_mm_s":
                ext.get("proposed_vy_mm_s"),

            "ext_vz_mm_s":
                ext.get("proposed_vz_mm_s"),

            # Wrist.
            "wrist_timestamp":
                wrist.get("timestamp"),

            "wrist_guidance_valid":
                wrist.get("guidance_valid"),

            "wrist_source_frame_seq":
                wrist.get("source_frame_seq"),

            "wrist_visual_cycle_id":
                wrist.get("visual_cycle_id"),

            "wrist_source_age_publish_ms":
                wrist.get(
                    "source_frame_age_ms_at_publish"
                ),

            "wrist_du_px":
                wrist.get("du_px"),

            "wrist_dv_px":
                wrist.get("dv_px"),

            "wrist_tip_grasp_distance_px":
                wrist.get(
                    "tip_grasp_distance_px"
                ),

            "wrist_jaw_corridor_ok":
                wrist.get(
                    "jaw_corridor_ok"
                ),

            "wrist_jaw_corridor_latched":
                wrist.get(
                    "jaw_corridor_latched"
                ),

            "wrist_vx_mm_s":
                wrist.get("proposed_vx_mm_s"),

            "wrist_vy_mm_s":
                wrist.get("proposed_vy_mm_s"),

            "wrist_vz_mm_s":
                wrist.get("proposed_vz_mm_s"),

            # Fusion.
            "fusion_timestamp":
                fusion.get("timestamp"),

            "fusion_mode":
                fusion.get("fusion_mode"),

            "fusion_grasp_sequence_stage":
                fusion.get(
                    "grasp_sequence_stage"
                ),

            "fusion_z_control_source":
                fusion.get(
                    "z_control_source"
                ),

            "fusion_external_valid":
                fusion.get(
                    "external_valid"
                ),

            "fusion_wrist_valid":
                fusion.get(
                    "wrist_valid"
                ),

            "fusion_vx_mm_s":
                fusion.get(
                    "proposed_vx_mm_s"
                ),

            "fusion_vy_mm_s":
                fusion.get(
                    "proposed_vy_mm_s"
                ),

            "fusion_vz_mm_s":
                fusion.get(
                    "proposed_vz_mm_s"
                ),

            "fusion_forward_nudge_active":
                fusion.get(
                    "grasp_forward_nudge_active"
                ),

            "fusion_forward_nudge_done":
                fusion.get(
                    "grasp_forward_nudge_done"
                ),

            "fusion_close_ready":
                fusion.get(
                    "grasp_close_ready"
                ),

            # Gate current.
            "gate_current_timestamp":
                gate.get("timestamp"),

            "gate_current_enabled":
                gate.get("enabled"),

            "gate_current_state":
                gate.get("gate_state"),

            "gate_current_vx_mm_s":
                gate.get("vx_mm_s"),

            "gate_current_vy_mm_s":
                gate.get("vy_mm_s"),

            "gate_current_vz_mm_s":
                gate.get("vz_mm_s"),

            # Gate actually consumed.
            "gate_consumed_timestamp":
                gate_used.get("timestamp"),

            "gate_consumed_enabled":
                gate_used.get("enabled"),

            "gate_consumed_state":
                gate_used.get("gate_state"),

            "gate_consumed_vx_mm_s":
                gate_used.get("vx_mm_s"),

            "gate_consumed_vy_mm_s":
                gate_used.get("vy_mm_s"),

            "gate_consumed_vz_mm_s":
                gate_used.get("vz_mm_s"),

            "gate_consumed_proposed_vx_mm_s":
                gate_used.get(
                    "proposed_vx_mm_s"
                ),

            "gate_consumed_proposed_vy_mm_s":
                gate_used.get(
                    "proposed_vy_mm_s"
                ),

            "gate_consumed_proposed_vz_mm_s":
                gate_used.get(
                    "proposed_vz_mm_s"
                ),

            # Controller.
            "controller_ibvs_active":
                sample.get(
                    "ibvs_velocity_active"
                ),

            "controller_ibvs_vx_mm_s":
                ibvs[0],

            "controller_ibvs_vy_mm_s":
                ibvs[1],

            "controller_ibvs_vz_mm_s":
                ibvs[2],

            "controller_cart_vx_mm_s":
                cart[0],

            "controller_cart_vy_mm_s":
                cart[1],

            "controller_cart_vz_mm_s":
                cart[2],

            "command_x_mm":
                cmd[0],

            "command_y_mm":
                cmd[1],

            "command_z_mm":
                cmd[2],

            "actual_x_mm":
                actual[0],

            "actual_y_mm":
                actual[1],

            "actual_z_mm":
                actual[2],

            "cartesian_lag_mm":
                sample.get(
                    "cartesian_lag_mm"
                ),

            "ik_solve_ms":
                sample.get(
                    "ik_solve_ms"
                ),

            "jacobian_min_singular":
                sample.get(
                    "jacobian_min_singular"
                ),

            "jacobian_condition":
                sample.get(
                    "jacobian_condition"
                ),

            "lowcmd_publish_count":
                sample.get(
                    "lowcmd_publish_count"
                ),

            "mode_machine":
                sample.get("mode_machine"),

            # Arrays.
            "arm_q_actual":
                self._array_text(
                    sample.get(
                        "arm_q_actual",
                        [],
                    )
                ),

            "arm_dq_actual":
                self._array_text(
                    sample.get(
                        "arm_dq_actual",
                        [],
                    )
                ),

            "arm_tau_est_actual":
                self._array_text(
                    sample.get(
                        "arm_tau_est_actual",
                        [],
                    )
                ),

            "arm_q_ik_raw":
                self._array_text(
                    sample.get(
                        "arm_q_ik_raw",
                        [],
                    )
                ),

            "arm_q_stream_goal":
                self._array_text(
                    sample.get(
                        "arm_q_stream_goal",
                        [],
                    )
                ),

            "arm_q_lowcmd_sent":
                self._array_text(
                    sample.get(
                        "arm_q_lowcmd_sent",
                        [],
                    )
                ),

            "arm_dq_lowcmd_sent":
                self._array_text(
                    sample.get(
                        "arm_dq_lowcmd_sent",
                        [],
                    )
                ),

            "arm_tau_ff_sent":
                self._array_text(
                    sample.get(
                        "arm_tau_ff_sent",
                        [],
                    )
                ),

            "gripper_q_actual":
                self._array_text(
                    sample.get(
                        "gripper_q_actual",
                        [],
                    )
                ),

            "gripper_dq_actual":
                self._array_text(
                    sample.get(
                        "gripper_dq_actual",
                        [],
                    )
                ),

            # Auto grasp.
            "auto_grasp_state":
                sample.get(
                    "auto_grasp_state"
                ),

            "auto_grasp_open_issued":
                sample.get(
                    "auto_grasp_open_issued"
                ),

            "auto_grasp_close_issued":
                sample.get(
                    "auto_grasp_close_issued"
                ),

            # Task state.
            "insertion_phase":
                insertion.get("phase"),

            "insertion_target_ready":
                insertion.get("target_ready"),

            "insertion_target_confirmed":
                insertion.get(
                    "target_confirmed"
                ),

            "close_settle_active":
                settle.get("active"),

            # Raw snapshots.
            "external_json":
                self._json_text(ext),

            "wrist_json":
                self._json_text(wrist),

            "fusion_json":
                self._json_text(fusion),

            "gate_current_json":
                self._json_text(gate),

            "gate_consumed_json":
                self._json_text(gate_used),

            "insertion_json":
                self._json_text(insertion),

            "settle_json":
                self._json_text(settle),
        }

    def _run(self):
        try:
            while (
                not self._stop.is_set()
                or
                not self._queue.empty()
            ):
                try:
                    sample = self._queue.get(
                        timeout=0.10
                    )
                except queue.Empty:
                    continue

                row = self._build_row(
                    sample
                )

                self._writer.writerow(
                    row
                )

                self.rows += 1

                # Continuous userspace flush.
                self._file.flush()

                now = time.monotonic()

                if (
                    now
                    -
                    self._last_fsync
                    >=
                    1.0
                ):
                    try:
                        os.fsync(
                            self._file.fileno()
                        )
                    except Exception:
                        pass

                    self._last_fsync = now

                self._queue.task_done()

        except Exception as exc:
            self.error = (
                f"{type(exc).__name__}: {exc}"
            )

    def close(self):
        self._stop.set()

        self._thread.join(
            timeout=5.0
        )

        try:
            self._file.flush()
            os.fsync(
                self._file.fileno()
            )
        except Exception:
            pass

        try:
            self._file.close()
        except Exception:
            pass
