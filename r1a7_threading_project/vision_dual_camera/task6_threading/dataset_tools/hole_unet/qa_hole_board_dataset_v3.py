#!/usr/bin/env python3

from pathlib import Path
import csv

import cv2
import numpy as np


ROOT = Path(
    "/home/robot/r1a7_datasets/wire_unet/"
    "hole_tube_001"
)

SESSION = (
    ROOT
    /
    "images_raw"
    /
    "session_20261105_105208"
)

MASK_ROOT = (
    ROOT
    /
    "masks_v3"
)

OUT = (
    ROOT
    /
    "qa_v3"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

CAMERAS = (
    "external",
    "right_wrist",
    "left_wrist",
)


def hole_components(mask):

    binary = (
        mask == 2
    ).astype(
        np.uint8
    )

    n, labels, stats, centers = (
        cv2.connectedComponentsWithStats(
            binary,
            8,
        )
    )

    areas = []

    for i in range(
        1,
        n,
    ):
        areas.append(
            int(
                stats[
                    i,
                    cv2.CC_STAT_AREA,
                ]
            )
        )

    return areas


rows = []

fatal_errors = []

all_ids = sorted(
    p.stem
    for p in (
        SESSION
        /
        "external"
    ).glob(
        "*.png"
    )
)

print(
    "=============================================="
)

print(
    "R1-A7 HOLE DATASET V3 QA"
)

print(
    "samples =",
    len(all_ids),
)

print(
    "expected images =",
    len(all_ids)
    *
    len(CAMERAS),
)

print(
    "=============================================="
)


for camera in CAMERAS:

    for sid in all_ids:

        image_path = (
            SESSION
            /
            camera
            /
            f"{sid}.png"
        )

        mask_path = (
            MASK_ROOT
            /
            camera
            /
            f"{sid}.png"
        )

        status = []

        if not image_path.exists():

            fatal_errors.append(
                f"MISSING IMAGE {camera} {sid}"
            )

            continue

        if not mask_path.exists():

            fatal_errors.append(
                f"MISSING MASK {camera} {sid}"
            )

            continue

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_UNCHANGED,
        )

        if image is None:

            fatal_errors.append(
                f"BAD IMAGE {camera} {sid}"
            )

            continue

        if mask is None:

            fatal_errors.append(
                f"BAD MASK {camera} {sid}"
            )

            continue

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        if (
            image.shape[:2]
            !=
            mask.shape[:2]
        ):

            fatal_errors.append(
                f"SHAPE MISMATCH "
                f"{camera} {sid} "
                f"image={image.shape[:2]} "
                f"mask={mask.shape[:2]}"
            )

            continue

        labels = sorted(
            int(x)
            for x in np.unique(
                mask
            )
        )

        bad_labels = [
            x
            for x in labels
            if x not in (
                0,
                1,
                2,
                255,
            )
        ]

        if bad_labels:

            fatal_errors.append(
                f"BAD LABEL "
                f"{camera} {sid}: "
                f"{bad_labels}"
            )

        bg_px = int(
            np.count_nonzero(
                mask == 0
            )
        )

        board_px = int(
            np.count_nonzero(
                mask == 1
            )
        )

        hole_px = int(
            np.count_nonzero(
                mask == 2
            )
        )

        ignore_px = int(
            np.count_nonzero(
                mask == 255
            )
        )

        components = (
            hole_components(
                mask
            )
        )

        if board_px == 0:
            status.append(
                "NO_BOARD"
            )

        if hole_px == 0:
            status.append(
                "NO_HOLE"
            )

        if len(components) > 0:

            min_hole = min(
                components
            )

            median_hole = float(
                np.median(
                    components
                )
            )

            max_hole = max(
                components
            )

        else:

            min_hole = 0
            median_hole = 0.0
            max_hole = 0

        rows.append({
            "sample":
                sid,

            "camera":
                camera,

            "height":
                mask.shape[0],

            "width":
                mask.shape[1],

            "labels":
                ",".join(
                    map(
                        str,
                        labels,
                    )
                ),

            "background_pixels":
                bg_px,

            "board_pixels":
                board_px,

            "hole_pixels":
                hole_px,

            "ignore_pixels":
                ignore_px,

            "hole_components":
                len(components),

            "hole_component_min_px":
                min_hole,

            "hole_component_median_px":
                median_hole,

            "hole_component_max_px":
                max_hole,

            "status":
                ";".join(
                    status
                ),
        })


csv_path = (
    OUT
    /
    "qa_mask_stats.csv"
)

if rows:

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


print()

print(
    "===== SUMMARY ====="
)

for camera in CAMERAS:

    rr = [
        r
        for r in rows
        if r["camera"]
        ==
        camera
    ]

    hole_pixels = [
        r["hole_pixels"]
        for r in rr
    ]

    hole_counts = [
        r["hole_components"]
        for r in rr
    ]

    print()

    print(
        camera
    )

    print(
        "  masks       =",
        len(rr),
    )

    print(
        "  no-hole     =",
        sum(
            x == 0
            for x in hole_pixels
        ),
    )

    if hole_pixels:

        print(
            "  hole pixels =",
            "min",
            min(hole_pixels),
            "median",
            round(
                float(
                    np.median(
                        hole_pixels
                    )
                ),
                1,
            ),
            "max",
            max(hole_pixels),
        )

        print(
            "  components  =",
            "min",
            min(hole_counts),
            "median",
            round(
                float(
                    np.median(
                        hole_counts
                    )
                ),
                1,
            ),
            "max",
            max(hole_counts),
        )


print()

print(
    "CSV =",
    csv_path,
)

print()

if fatal_errors:

    print(
        "===== FATAL QA ERRORS ====="
    )

    for x in fatal_errors:
        print(
            x
        )

    raise SystemExit(
        1
    )

else:

    print(
        "BASIC_DATASET_QA_OK"
    )
