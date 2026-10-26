#!/usr/bin/env python3

import cv2
import json
import os
import time
from pathlib import Path

from wire_detector_core import WireDetectorCore


class ExternalWireAdapter:

    def __init__(
        self,
        out_file="/tmp/r1a7_external_wire.json",
        process_every=3,
    ):
        self.out_file = Path(out_file)

        self.detector = WireDetectorCore(
            roi_margin=100,
            keypoint_step=15,
        )

        self.process_every = int(process_every)

        self.frame_index = 0

        # Manually initialized free wire tip.
        self.wire_tip_seed = None

        self.last_result = None


    # ========================================================
    # Mouse
    #
    # LEFT  : keep original hole-selection behavior
    # RIGHT : initialize wire free tip
    # MIDDLE: clear wire initialization
    # ========================================================

    def mouse_callback(
        self,
        event,
        x,
        y,
        flags,
        hole_callback,
    ):

        if event == cv2.EVENT_RBUTTONDOWN:

            self.wire_tip_seed = (
                int(x),
                int(y),
            )

            self.last_result = None

            print(
                "[WIRE] free-tip seed =",
                self.wire_tip_seed,
                flush=True,
            )

            return


        if event == cv2.EVENT_MBUTTONDOWN:

            self.wire_tip_seed = None
            self.last_result = None

            print(
                "[WIRE] seed cleared",
                flush=True,
            )

            return


        # Preserve the existing LEFT-click hole selector.
        if hole_callback is not None:

            hole_callback(
                event,
                x,
                y,
                flags,
                None,
            )


    # ========================================================
    # JSON
    # ========================================================

    def _write_json(
        self,
        data,
    ):

        tmp = Path(
            str(self.out_file)
            +
            ".tmp"
        )

        tmp.write_text(
            json.dumps(
                data,
                indent=2,
            )
        )

        os.replace(
            tmp,
            self.out_file,
        )


    def _invalid_state(
        self,
        reason,
        grasp_uv,
        hole_uv,
    ):

        state = {
            "timestamp":
                time.time(),

            "source":
                "external_wire",

            "wire_valid":
                False,

            "reason":
                str(reason),

            "wire_tip_seed_uv":
                (
                    list(self.wire_tip_seed)
                    if self.wire_tip_seed is not None
                    else None
                ),

            "wire_tip_uv":
                None,

            "wire_grasp_uv":
                (
                    list(grasp_uv)
                    if grasp_uv is not None
                    else None
                ),

            "hole_uv":
                (
                    list(hole_uv)
                    if hole_uv is not None
                    else None
                ),

            "centerline":
                [],

            "keypoints":
                [],

            "confidence":
                0.0,
        }

        self._write_json(
            state
        )

        return state


    # ========================================================
    # Drawing
    # ========================================================

    def _draw(
        self,
        img,
        result,
    ):

        if self.wire_tip_seed is not None:

            cv2.circle(
                img,
                self.wire_tip_seed,
                8,
                (0,255,255),
                2,
            )

            cv2.putText(
                img,
                "WIRE SEED",
                (
                    self.wire_tip_seed[0] + 10,
                    self.wire_tip_seed[1] - 10,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,255),
                2,
            )


        if (
            result is None
            or
            not result.get(
                "wire_valid",
                False,
            )
        ):

            cv2.putText(
                img,
                "WIRE: WAIT RIGHT CLICK",
                (20,95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0,165,255),
                2,
            )

            return


        centerline = result.get(
            "centerline",
            [],
        )


        for i in range(
            1,
            len(centerline),
        ):

            p0 = tuple(
                map(
                    int,
                    centerline[i-1]
                )
            )

            p1 = tuple(
                map(
                    int,
                    centerline[i]
                )
            )

            cv2.line(
                img,
                p0,
                p1,
                (0,255,0),
                2,
            )


        for p in result.get(
            "keypoints",
            [],
        ):

            cv2.circle(
                img,
                tuple(map(int,p)),
                3,
                (0,255,255),
                -1,
            )


        wire_tip = result.get(
            "wire_tip_uv"
        )

        wire_grasp = result.get(
            "wire_grasp_uv"
        )


        if wire_tip is not None:

            cv2.circle(
                img,
                tuple(map(int,wire_tip)),
                7,
                (0,255,255),
                -1,
            )


        if wire_grasp is not None:

            cv2.circle(
                img,
                tuple(map(int,wire_grasp)),
                7,
                (255,0,0),
                -1,
            )


        roi = result.get(
            "roi"
        )

        if roi is not None:

            x0,y0,x1,y1 = map(
                int,
                roi
            )

            cv2.rectangle(
                img,
                (x0,y0),
                (x1,y1),
                (255,255,0),
                1,
            )


        cv2.putText(
            img,
            (
                "WIRE OK "
                f"C={result.get('confidence',0.0):.2f} "
                f"N={len(centerline)}"
            ),
            (20,95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0,255,0),
            2,
        )


    # ========================================================
    # Runtime
    # ========================================================

    def process(
        self,
        frame,
        overlay,
        grasp_uv,
        hole_uv=None,
    ):

        self.frame_index += 1


        if self.wire_tip_seed is None:

            self._invalid_state(
                "wire_tip_not_initialized",
                grasp_uv,
                hole_uv,
            )

            self._draw(
                overlay,
                None,
            )

            return None


        if grasp_uv is None:

            self._invalid_state(
                "tweezer_tip_lost",
                grasp_uv,
                hole_uv,
            )

            self._draw(
                overlay,
                self.last_result,
            )

            return self.last_result


        # Heavy wire detector does not need to run at full 30 Hz.
        run_detector = (
            self.last_result is None
            or
            self.frame_index
            %
            self.process_every
            ==
            0
        )


        if run_detector:

            result = self.detector.detect(
                frame,
                self.wire_tip_seed,
                grasp_uv,
            )


            result["timestamp"] = (
                time.time()
            )

            result["source"] = (
                "external_wire"
            )

            result[
                "wire_tip_seed_uv"
            ] = list(
                self.wire_tip_seed
            )

            result[
                "tweezer_tip_uv"
            ] = list(
                grasp_uv
            )

            result[
                "hole_uv"
            ] = (
                list(hole_uv)
                if hole_uv is not None
                else None
            )


            self.last_result = result

            self._write_json(
                result
            )


        self._draw(
            overlay,
            self.last_result,
        )

        return self.last_result
