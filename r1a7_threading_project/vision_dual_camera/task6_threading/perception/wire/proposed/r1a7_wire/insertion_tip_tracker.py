import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

TASK6_DIR = Path(__file__).resolve().parents[4]
if str(TASK6_DIR) not in sys.path:
    sys.path.insert(0, str(TASK6_DIR))

import task6_wrist_ibvs_controller_v2_tip as wrist


TIP_EMA_ALPHA = wrist.TIP_EMA_ALPHA
TIP_MAX_JUMP_PX = wrist.TIP_MAX_JUMP_PX
TIP_RELOCK_FRAMES = wrist.TIP_RELOCK_FRAMES
TIP_RELOCK_CLUSTER_RADIUS_PX = wrist.TIP_RELOCK_CLUSTER_RADIUS_PX
TIP_RELOCK_MAX_ANCHOR_DIST_PX = wrist.TIP_RELOCK_MAX_ANCHOR_DIST_PX
TIP_HOLD_S = wrist.TIP_HOLD_S
TIP_DETECT_EVERY_N_FRAMES = 1

OUTPUT = Path("/tmp/r1a7_dynamic_tweezer_tip.json")


def atomic_write_json(path, data):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


class InsertionTipTracker:

    def __init__(self, init_frame):
        print()
        print("[INSERTION TIP] Loading original wrist Tip detector...", flush=True)

        self.detector = wrist.Sam2PcaTweezerDetector(
            checkpoint_path=wrist.SAM2_CHECKPOINT
        )
        self.refiner = wrist.TweezerTipRefinerV2()

        init = wrist.initialize_live_tip(init_frame)

        if init is None:
            raise RuntimeError("Tweezer Tip initialization cancelled")

        (
            self.tip_box,
            self.positive_points,
            self.tip_hint,
        ) = init

        self.tip_ema = None
        self.last_valid_tip = None
        self.last_valid_time = None
        self.tip_velocity = np.zeros(2, dtype=np.float64)

        self.relock_candidates = []
        self.frame_counter = 0

        self.last_mask = None
        self.last_mask_time = None

        print("[INSERTION TIP] Dynamic Tip tracking enabled", flush=True)
        print("[INSERTION TIP] Fixed (739,530) control point DISABLED", flush=True)


    def _result(
        self,
        tip,
        state,
        mask,
        measurement_valid=False,
    ):
        now = time.time()
        measurement_valid = bool(
            measurement_valid and tip is not None
        )

        mask_fresh = bool(
            mask is not None
            and self.last_mask_time is not None
            and 0.0 <= now - self.last_mask_time <= TIP_HOLD_S
        )

        exclusion = None
        if mask_fresh:
            m = np.squeeze(np.asarray(mask))
            if m.ndim == 2:
                m = (m > 0).astype(np.uint8) * 255
                kernel = np.ones((9, 9), np.uint8)
                exclusion = cv2.dilate(
                    m, kernel, iterations=1
                )
            else:
                mask_fresh = False

        if not mask_fresh:
            exclusion = None

        if measurement_valid:
            source = "MEASURED"
        elif tip is not None and str(state).startswith("HOLD"):
            source = "HOLD"
        elif tip is not None and str(state).startswith("PREDICT"):
            source = "PREDICT"
        else:
            source = "LOST"

        age = (
            max(0.0, now - self.last_valid_time)
            if self.last_valid_time is not None
            else None
        )

        data = {
            "timestamp": now,
            "valid": measurement_valid,
            "measurement_valid": measurement_valid,
            "mask_valid": bool(
                mask_fresh and exclusion is not None
            ),
            "tip_uv": list(tip) if tip is not None else None,
            "tip_source": source,
            "last_observation_timestamp": self.last_valid_time,
            "tip_age_s": age,
            "state": state,
        }

        atomic_write_json(OUTPUT, data)

        return {
            "valid": measurement_valid,
            "measurement_valid": measurement_valid,
            "mask_valid": bool(
                mask_fresh and exclusion is not None
            ),
            "tip_source": source,
            "tip_age_s": age,
            "last_observation_timestamp": self.last_valid_time,
            "state": state,
            "tip": tip,
            "raw_mask": mask if mask_fresh else None,
            "exclusion_mask": exclusion,
        }


    def update(self, frame):
        self.frame_counter += 1

        now = time.time()

        run_detector = (
            self.tip_ema is None
            or self.last_valid_tip is None
            or self.last_valid_time is None
            or (
                self.frame_counter
                % TIP_DETECT_EVERY_N_FRAMES
                == 0
            )
        )

        tip_pixel = None
        state = "LOST"
        current_mask = self.last_mask
        measurement_valid = False

        if run_detector:

            try:
                coarse = self.detector.detect(
                    frame,
                    box_xyxy=self.tip_box,
                    tip_hint=self.tip_hint,
                    positive_points=self.positive_points,
                )
            except Exception as exc:
                self.relock_candidates.clear()
                return self._result(
                    None,
                    "SAM2_ERROR:" + type(exc).__name__,
                    self.last_mask,
                )

            if coarse.get("valid", False):

                current_mask = coarse.get("mask")
                self.last_mask = current_mask
                self.last_mask_time = (
                    now if current_mask is not None else None
                )

                refined = self.refiner.refine(
                    frame,
                    coarse["mask"],
                    coarse["tip_uv"],
                    coarse["direction"],
                )

                if refined.get("valid", False):

                    raw_tip = np.asarray(
                        refined["refined_tip_uv"],
                        dtype=np.float64,
                    )

                    accepted = False
                    relocked = False

                    if self.tip_ema is None:

                        self.tip_ema = raw_tip.copy()
                        self.relock_candidates.clear()

                        accepted = True
                        state = "INIT"

                    else:

                        jump = float(
                            np.linalg.norm(
                                raw_tip - self.tip_ema
                            )
                        )

                        if jump <= TIP_MAX_JUMP_PX:

                            self.tip_ema = (
                                TIP_EMA_ALPHA * raw_tip
                                +
                                (1.0 - TIP_EMA_ALPHA)
                                * self.tip_ema
                            )

                            self.relock_candidates.clear()

                            accepted = True
                            state = f"OK jump={jump:.1f}"

                        else:

                            self.relock_candidates.append(
                                raw_tip.copy()
                            )

                            if (
                                len(self.relock_candidates)
                                >
                                TIP_RELOCK_FRAMES
                            ):
                                self.relock_candidates = (
                                    self.relock_candidates[
                                        -TIP_RELOCK_FRAMES:
                                    ]
                                )

                            state = (
                                f"RELOCK WAIT "
                                f"{len(self.relock_candidates)}/"
                                f"{TIP_RELOCK_FRAMES} "
                                f"jump={jump:.1f}"
                            )

                            if (
                                len(self.relock_candidates)
                                ==
                                TIP_RELOCK_FRAMES
                            ):

                                arr = np.stack(
                                    self.relock_candidates,
                                    axis=0,
                                )

                                center = np.mean(
                                    arr,
                                    axis=0,
                                )

                                spread = float(
                                    np.max(
                                        np.linalg.norm(
                                            arr - center,
                                            axis=1,
                                        )
                                    )
                                )

                                anchor_dist = 0.0

                                if self.last_valid_tip is not None:

                                    anchor = (
                                        self.last_valid_tip.copy()
                                    )

                                    if self.last_valid_time is not None:

                                        age = max(
                                            0.0,
                                            now
                                            -
                                            self.last_valid_time,
                                        )

                                        anchor = (
                                            anchor
                                            +
                                            self.tip_velocity
                                            *
                                            min(
                                                age,
                                                TIP_HOLD_S,
                                            )
                                        )

                                    anchor_dist = float(
                                        np.linalg.norm(
                                            center - anchor
                                        )
                                    )

                                if (
                                    spread
                                    <=
                                    TIP_RELOCK_CLUSTER_RADIUS_PX
                                    and
                                    anchor_dist
                                    <=
                                    TIP_RELOCK_MAX_ANCHOR_DIST_PX
                                ):

                                    self.tip_ema = center.copy()

                                    self.relock_candidates.clear()

                                    accepted = True
                                    relocked = True

                                    state = (
                                        f"RELOCK "
                                        f"spread={spread:.1f}"
                                    )

                                    print(
                                        "[TIP RELOCK] "
                                        f"tip=({center[0]:.1f},"
                                        f"{center[1]:.1f}) "
                                        f"spread={spread:.1f}px",
                                        flush=True,
                                    )

                                elif (
                                    anchor_dist
                                    >
                                    TIP_RELOCK_MAX_ANCHOR_DIST_PX
                                ):

                                    state = (
                                        f"RELOCK REJECT "
                                        f"anchor={anchor_dist:.1f} "
                                        f"spread={spread:.1f}"
                                    )

                                    print(
                                        "[TIP RELOCK REJECT] "
                                        f"candidate="
                                        f"({center[0]:.1f},"
                                        f"{center[1]:.1f}) "
                                        f"anchor_dist="
                                        f"{anchor_dist:.1f}px",
                                        flush=True,
                                    )

                                    self.relock_candidates.clear()

                    if accepted:
                        measurement_valid = True

                        old_tip = (
                            None
                            if self.last_valid_tip is None
                            else self.last_valid_tip.copy()
                        )

                        old_time = self.last_valid_time

                        if relocked:
                            self.tip_velocity[:] = 0.0

                        elif (
                            old_tip is not None
                            and
                            old_time is not None
                        ):

                            dt = max(
                                now - old_time,
                                1e-3,
                            )

                            self.tip_velocity = (
                                self.tip_ema - old_tip
                            ) / dt

                        self.last_valid_tip = (
                            self.tip_ema.copy()
                        )

                        self.last_valid_time = now

                        self.tip_hint = (
                            float(self.tip_ema[0]),
                            float(self.tip_ema[1]),
                        )

                        tip_pixel = (
                            int(round(self.tip_ema[0])),
                            int(round(self.tip_ema[1])),
                        )

                    else:

                        if (
                            self.last_valid_tip is not None
                            and
                            self.last_valid_time is not None
                        ):

                            lost_time = (
                                now - self.last_valid_time
                            )

                            if lost_time < TIP_HOLD_S:

                                hold_tip = (
                                    self.last_valid_tip
                                    +
                                    self.tip_velocity
                                    *
                                    lost_time
                                )

                                tip_pixel = (
                                    int(round(hold_tip[0])),
                                    int(round(hold_tip[1])),
                                )

                                self.tip_hint = (
                                    float(hold_tip[0]),
                                    float(hold_tip[1]),
                                )

                                state = (
                                    f"HOLD {lost_time:.3f}s"
                                )

                else:

                    self.relock_candidates.clear()

                    if (
                        self.last_valid_tip is not None
                        and
                        self.last_valid_time is not None
                    ):

                        lost_time = (
                            now - self.last_valid_time
                        )

                        if lost_time < TIP_HOLD_S:

                            hold_tip = (
                                self.last_valid_tip
                                +
                                self.tip_velocity
                                *
                                lost_time
                            )

                            tip_pixel = (
                                int(round(hold_tip[0])),
                                int(round(hold_tip[1])),
                            )

                            state = (
                                f"HOLD {lost_time:.3f}s"
                            )

                        else:
                            state = "REFINER LOST"

                    else:
                        state = "REFINER LOST"

            else:

                self.relock_candidates.clear()
                state = "SAM2 LOST"

        else:

            if (
                self.last_valid_tip is not None
                and
                self.last_valid_time is not None
            ):

                predict_dt = max(
                    0.0,
                    now - self.last_valid_time,
                )

                if predict_dt < TIP_HOLD_S:

                    predicted_tip = (
                        self.last_valid_tip
                        +
                        self.tip_velocity
                        *
                        predict_dt
                    )

                    if np.all(
                        np.isfinite(predicted_tip)
                    ):

                        tip_pixel = (
                            int(round(predicted_tip[0])),
                            int(round(predicted_tip[1])),
                        )

                        self.tip_hint = (
                            float(predicted_tip[0]),
                            float(predicted_tip[1]),
                        )

                        state = (
                            f"PREDICT {predict_dt:.3f}s"
                        )

                else:
                    state = (
                        f"PREDICT EXPIRED "
                        f"{predict_dt:.3f}s"
                    )

        return self._result(
            tip_pixel,
            state,
            current_mask,
            measurement_valid=measurement_valid,
        )
