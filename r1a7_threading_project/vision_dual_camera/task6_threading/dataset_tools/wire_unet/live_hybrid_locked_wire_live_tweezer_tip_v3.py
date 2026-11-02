#!/usr/bin/env python3

import cv2
import numpy as np

import live_hybrid_locked_wire_live_tweezer_tip_v2 as v2


# ============================================================
# COLORS BGR
# ============================================================

COLOR_WIRE = (
    0,
    0,
    255,
)

COLOR_TOOL = (
    255,
    0,
    0,
)

COLOR_FREE = (
    0,
    255,
    0,
)

COLOR_FIXED = (
    255,
    255,
    0,
)

COLOR_GRASP = (
    0,
    255,
    255,
)

COLOR_TEXT = (
    255,
    255,
    255,
)


# ============================================================
# EXACT SNAP TO TRACKED CENTERLINE
# ============================================================

def snap_to_path(
    path_mask,
    point_yx,
):

    if point_yx is None:
        return None

    path = np.asarray(
        path_mask,
        dtype=bool,
    )

    ys, xs = np.where(
        path
    )

    if len(xs) == 0:
        return point_yx

    py = float(
        point_yx[0]
    )

    px = float(
        point_yx[1]
    )

    d2 = (
        (
            ys.astype(
                np.float32
            )
            - py
        ) ** 2
        +
        (
            xs.astype(
                np.float32
            )
            - px
        ) ** 2
    )

    i = int(
        np.argmin(
            d2
        )
    )

    return (
        int(
            ys[i]
        ),
        int(
            xs[i]
        ),
    )


# ============================================================
# THIN MASK DRAWING
# ============================================================

def overlay_mask(
    image,
    mask,
    color,
    radius=0,
):

    m = np.asarray(
        mask,
        dtype=bool,
    )

    if not np.any(m):
        return

    if radius > 0:

        k = (
            2 * radius
            + 1
        )

        kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    k,
                    k,
                ),
            )
        )

        m = cv2.dilate(
            m.astype(
                np.uint8
            ),
            kernel,
        ).astype(bool)

    image[m] = color


# ============================================================
# SMALL PRECISE CROSS
# ============================================================

def draw_precise_cross(
    image,
    point_yx,
    color,
):

    if point_yx is None:
        return

    y = int(
        point_yx[0]
    )

    x = int(
        point_yx[1]
    )

    cv2.drawMarker(
        image,
        (
            x,
            y,
        ),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=9,
        thickness=1,
        line_type=cv2.LINE_AA,
    )

    # Exact center pixel.
    cv2.circle(
        image,
        (
            x,
            y,
        ),
        1,
        color,
        -1,
        cv2.LINE_AA,
    )


# ============================================================
# PRECISE FREE / GRASP DISPLAY
# ============================================================

def draw_scene_precise(
    frame,
    det,
    track,
    geom,
    name,
    fps,
):

    overlay = (
        frame.copy()
    )

    path = np.asarray(
        track["path"],
        dtype=bool,
    )

    tool = np.asarray(
        det["tool_mask"],
        dtype=bool,
    )

    # --------------------------------------------------------
    # BLUE TOOL
    # --------------------------------------------------------

    overlay_mask(
        overlay,
        tool,
        COLOR_TOOL,
        radius=0,
    )

    # --------------------------------------------------------
    # RED CENTERLINE
    #
    # Keep it thin so marker position can be judged precisely.
    # --------------------------------------------------------

    overlay_mask(
        overlay,
        path,
        COLOR_WIRE,
        radius=1,
    )

    vis = cv2.addWeighted(
        frame,
        0.62,
        overlay,
        0.38,
        0,
    )

    # --------------------------------------------------------
    # SNAP GEOMETRY TO EXACT TRACKED PATH
    # --------------------------------------------------------

    free_yx = snap_to_path(
        path,
        geom.get(
            "free_endpoint_yx"
        ),
    )

    fixed_yx = snap_to_path(
        path,
        geom.get(
            "fixed_endpoint_yx"
        ),
    )

    grasp_yx = snap_to_path(
        path,
        geom.get(
            "grasp_point_yx"
        ),
    )

    # --------------------------------------------------------
    # SMALL CROSSES
    # --------------------------------------------------------

    if geom.get(
        "display_valid",
        False,
    ):

        draw_precise_cross(
            vis,
            free_yx,
            COLOR_FREE,
        )

        draw_precise_cross(
            vis,
            fixed_yx,
            COLOR_FIXED,
        )

        draw_precise_cross(
            vis,
            grasp_yx,
            COLOR_GRASP,
        )

    # --------------------------------------------------------
    # HEADER
    #
    # Put labels here instead of next to the points.
    # This prevents text from hiding the actual wire.
    # --------------------------------------------------------

    cv2.rectangle(
        vis,
        (
            0,
            0,
        ),
        (
            vis.shape[1],
            108,
        ),
        (
            0,
            0,
            0,
        ),
        -1,
    )

    calibrated = (
        geom.get(
            "calibrated",
            False,
        )
    )

    line1 = (
        f"{name}  "
        f"CAL={'YES' if calibrated else 'NO'}  "
        f"STATE={track['state']}"
    )

    line2 = (
        f"STATUS="
        f"{geom.get('status', 'N/A')}"
    )

    if free_yx is not None:

        line3 = (
            f"FREE=({free_yx[1]},{free_yx[0]})"
        )

    else:

        line3 = (
            "FREE=(--,--)"
        )

    if grasp_yx is not None:

        line3 += (
            f"  GRASP="
            f"({grasp_yx[1]},{grasp_yx[0]})"
        )

    else:

        line3 += (
            "  GRASP=(--,--)"
        )

    line4 = (
        f"FPS={fps:.1f}"
    )

    cv2.putText(
        vis,
        line1,
        (
            10,
            23,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        COLOR_TEXT,
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        vis,
        line2,
        (
            10,
            48,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        COLOR_TEXT,
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        vis,
        line3,
        (
            10,
            73,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        COLOR_TEXT,
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        vis,
        line4,
        (
            10,
            98,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        COLOR_TEXT,
        1,
        cv2.LINE_AA,
    )

    return vis


# ============================================================
# PATCH ONLY FREE / GRASP DISPLAY
#
# TIP V2 remains unchanged.
# ============================================================

v2.v1.manual.draw_scene = (
    draw_scene_precise
)


if __name__ == "__main__":

    print()
    print("=" * 72)

    print(
        "PRECISE MARKER V3"
    )

    print(
        "GREEN CROSS  = FREE"
    )

    print(
        "YELLOW CROSS = GRASP"
    )

    print(
        "CYAN CROSS   = FIXED"
    )

    print(
        "MAGENTA CROSS= TIP"
    )

    print()
    print(
        "FREE/GRASP are snapped to exact tracked-path pixels."
    )

    print(
        "Large endpoint/grasp circles are disabled."
    )

    print("=" * 72)
    print()

    v2.v1.main()
