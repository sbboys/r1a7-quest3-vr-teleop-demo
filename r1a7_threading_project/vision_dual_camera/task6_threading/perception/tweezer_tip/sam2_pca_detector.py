#!/usr/bin/env python3

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class Sam2PcaTweezerDetector:

    def __init__(
        self,
        checkpoint_path,
        config_name="configs/sam2.1/sam2.1_hiera_s.yaml",
        device=None,
    ):

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = device

        print(
            f"[SAM2] loading model on {self.device} ..."
        )

        self.model = build_sam2(
            config_file=config_name,
            ckpt_path=str(checkpoint_path),
            device=self.device,
        )

        self.predictor = SAM2ImagePredictor(
            self.model
        )

        print("[SAM2] model ready")

    @staticmethod
    def _select_contour(
        mask_u8,
        tip_hint=None,
        min_area=50.0,
    ):

        contours, _ = cv2.findContours(
            mask_u8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )

        contours = [
            c for c in contours
            if cv2.contourArea(c) >= min_area
        ]

        if not contours:
            return None

        # 如果点击位置确实落在某个连通区域内，
        # 优先使用包含 tip hint 的轮廓。
        if tip_hint is not None:

            p = (
                float(tip_hint[0]),
                float(tip_hint[1]),
            )

            containing = []

            for c in contours:

                if cv2.pointPolygonTest(
                    c,
                    p,
                    False,
                ) >= 0:

                    containing.append(c)

            if containing:

                return max(
                    containing,
                    key=cv2.contourArea,
                )

        # 否则退回最大连通轮廓
        return max(
            contours,
            key=cv2.contourArea,
        )

    @staticmethod
    def _pca_tip_tail(
        contour,
        tip_hint,
        endpoint_fraction=0.02,
    ):

        pts = contour.reshape(
            -1,
            2,
        ).astype(np.float64)

        if len(pts) < 10:
            raise RuntimeError(
                "Too few contour points"
            )

        mean, eigenvectors, eigenvalues = (
            cv2.PCACompute2(
                pts,
                mean=np.empty((0)),
            )
        )

        center = mean[0]

        axis = eigenvectors[0]
        axis /= (
            np.linalg.norm(axis) + 1e-12
        )

        projection = (
            pts - center
        ) @ axis

        order = np.argsort(
            projection
        )

        k = max(
            5,
            int(
                len(pts)
                * endpoint_fraction
            ),
        )

        low_cluster = pts[
            order[:k]
        ]

        high_cluster = pts[
            order[-k:]
        ]

        # 用一组极值点中位数，
        # 比直接拿一个最远像素抗噪声更强。
        low_endpoint = np.median(
            low_cluster,
            axis=0,
        )

        high_endpoint = np.median(
            high_cluster,
            axis=0,
        )

        hint = np.asarray(
            tip_hint,
            dtype=np.float64,
        )

        d_low = np.linalg.norm(
            low_endpoint - hint
        )

        d_high = np.linalg.norm(
            high_endpoint - hint
        )

        if d_low <= d_high:

            tip = low_endpoint
            tail = high_endpoint

        else:

            tip = high_endpoint
            tail = low_endpoint

        direction = (
            tip - tail
        )

        direction /= (
            np.linalg.norm(direction)
            + 1e-12
        )

        eig = eigenvalues.reshape(-1)

        if len(eig) >= 2:

            linearity = float(
                eig[0]
                /
                (
                    eig[0]
                    + eig[1]
                    + 1e-12
                )
            )

        else:

            linearity = 0.0

        return {
            "tip": tip,
            "tail": tail,
            "center": center,
            "axis": axis,
            "direction": direction,
            "linearity": linearity,
        }

    def detect(
        self,
        frame_bgr,
        box_xyxy,
        tip_hint,
        positive_points=None,
    ):

        h, w = frame_bgr.shape[:2]

        box = np.asarray(
            box_xyxy,
            dtype=np.float32,
        )

        x1, y1, x2, y2 = box

        x1 = int(
            np.clip(x1, 0, w - 1)
        )
        y1 = int(
            np.clip(y1, 0, h - 1)
        )
        x2 = int(
            np.clip(x2, 0, w - 1)
        )
        y2 = int(
            np.clip(y2, 0, h - 1)
        )

        if (
            x2 <= x1
            or
            y2 <= y1
        ):
            raise ValueError(
                "Invalid ROI box"
            )

        image_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )

        if positive_points is None:
            positive_points = [tip_hint]

        point_coords = np.asarray(
            positive_points,
            dtype=np.float32,
        )

        point_labels = np.ones(
            len(point_coords),
            dtype=np.int32,
        )

        with torch.inference_mode():

            self.predictor.set_image(
                image_rgb
            )

            masks, scores, _ = (
                self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box,
                    multimask_output=True,
                )
            )

        best_idx = int(
            np.argmax(scores)
        )

        mask = masks[
            best_idx
        ].astype(bool)

        sam_score = float(
            scores[best_idx]
        )

        # --------------------------------------------------
        # 强制裁掉 ROI 外部区域
        # 防止 SAM2 把附近机器人结构也连接进去。
        # --------------------------------------------------

        roi_mask = np.zeros(
            mask.shape,
            dtype=bool,
        )

        roi_mask[
            y1:y2 + 1,
            x1:x2 + 1
        ] = True

        mask = (
            mask
            &
            roi_mask
        )

        mask_u8 = (
            mask.astype(np.uint8)
            * 255
        )

        contour = self._select_contour(
            mask_u8,
            tip_hint=tip_hint,
        )

        if contour is None:

            return {
                "valid": False,
                "reason": "no_contour",
                "mask": mask_u8,
                "sam_score": sam_score,
            }

        area = float(
            cv2.contourArea(
                contour
            )
        )

        pca = self._pca_tip_tail(
            contour,
            tip_hint,
        )

        tip = pca["tip"]
        tail = pca["tail"]

        # 这里只作为实验质量指标，
        # 不是统计学概率。
        confidence = float(
            np.clip(
                sam_score,
                0.0,
                1.0,
            )
            *
            np.clip(
                pca["linearity"],
                0.0,
                1.0,
            )
        )

        return {
            "valid": True,

            "tip_uv": [
                float(tip[0]),
                float(tip[1]),
            ],

            "tail_uv": [
                float(tail[0]),
                float(tail[1]),
            ],

            "center_uv": [
                float(pca["center"][0]),
                float(pca["center"][1]),
            ],

            "axis": [
                float(pca["axis"][0]),
                float(pca["axis"][1]),
            ],

            "direction": [
                float(
                    pca["direction"][0]
                ),
                float(
                    pca["direction"][1]
                ),
            ],

            "sam_score": sam_score,

            "pca_linearity": float(
                pca["linearity"]
            ),

            "confidence": confidence,

            "mask_area": area,

            "mask": mask_u8,

            "contour": contour,

            "box_xyxy": [
                int(x1),
                int(y1),
                int(x2),
                int(y2),
            ],
        }
