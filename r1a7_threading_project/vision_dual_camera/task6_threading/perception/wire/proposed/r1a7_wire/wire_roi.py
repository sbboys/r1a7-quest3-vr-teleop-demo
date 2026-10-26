#!/usr/bin/env python3

import cv2
import numpy as np


def make_wire_roi(
    frame_shape,
    wire_tip_uv,
    grasp_uv,
    margin=80,
):
    """
    根据 wire free tip 和 tweezer grasp 建立 ROI。

    Parameters
    ----------
    frame_shape : H,W or H,W,C
    wire_tip_uv : (u,v)
    grasp_uv    : (u,v)
    margin      : ROI expansion pixels

    Returns
    -------
    (x0,y0,x1,y1)
    """

    h, w = frame_shape[:2]

    if wire_tip_uv is None or grasp_uv is None:
        return (0, 0, w, h)

    tx, ty = map(float, wire_tip_uv)
    gx, gy = map(float, grasp_uv)

    x0 = int(max(0, min(tx, gx) - margin))
    y0 = int(max(0, min(ty, gy) - margin))

    x1 = int(min(w, max(tx, gx) + margin))
    y1 = int(min(h, max(ty, gy) + margin))

    return (x0, y0, x1, y1)


def draw_wire_roi(
    image,
    roi,
    color=(255,255,0),
):
    out = image.copy()

    x0,y0,x1,y1 = roi

    cv2.rectangle(
        out,
        (x0,y0),
        (x1,y1),
        color,
        1,
    )

    return out
