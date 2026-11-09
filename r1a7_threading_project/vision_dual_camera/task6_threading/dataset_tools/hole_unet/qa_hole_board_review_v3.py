#!/usr/bin/env python3

from pathlib import Path
import csv
import json

import cv2
import numpy as np


ROOT = Path(
    "/home/robot/r1a7_datasets/wire_unet/hole_tube_001"
)

SESSION = (
    ROOT
    / "images_raw"
    / "session_20261105_105208"
)

MASK_ROOT = ROOT / "masks_v3"

QA_CSV = ROOT / "qa_v3" / "qa_mask_stats.csv"

GEO_JSON = (
    ROOT
    / "annotations_v3"
    / "hole_geometry.jsonl"
)

OUT = ROOT / "qa_v3" / "visual_review"

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

CAMERAS = (
    "external",
    "right_wrist",
    "left_wrist",
)


# ------------------------------------------------------------
# Read QA CSV
# ------------------------------------------------------------

rows = []

with QA_CSV.open() as f:

    reader = csv.DictReader(f)

    for r in reader:

        r["hole_pixels"] = int(
            r["hole_pixels"]
        )

        r["hole_components"] = int(
            r["hole_components"]
        )

        rows.append(r)


# ------------------------------------------------------------
# Read geometry JSON
# ------------------------------------------------------------

geometry = {}

if GEO_JSON.exists():

    for line in GEO_JSON.read_text(
        encoding="utf-8"
    ).splitlines():

        if not line.strip():
            continue

        r = json.loads(line)

        geometry[
            (
                r["sample_id"],
                r["camera"],
            )
        ] = len(
            r.get("holes", [])
        )


# ------------------------------------------------------------
# Add annotation hole count
# ------------------------------------------------------------

for r in rows:

    key = (
        r["sample"],
        r["camera"],
    )

    r["geometry_holes"] = (
        geometry.get(
            key,
            -1,
        )
    )

    r["component_delta"] = (
        r["geometry_holes"]
        -
        r["hole_components"]
        if r["geometry_holes"] >= 0
        else -999
    )


# ------------------------------------------------------------
# Print rankings
# ------------------------------------------------------------

print()
print("=" * 70)
print("HOLE BOARD V3 VISUAL QA")
print("=" * 70)

for camera in CAMERAS:

    rr = [
        r
        for r in rows
        if r["camera"] == camera
    ]

    print()
    print(
        "=================================================="
    )

    print(
        camera.upper()
    )

    print(
        "=================================================="
    )

    print()
    print("LOWEST COMPONENT COUNTS")

    for r in sorted(
        rr,
        key=lambda x: (
            x["hole_components"],
            x["hole_pixels"],
        ),
    )[:10]:

        print(
            r["sample"],
            "components=",
            r["hole_components"],
            "geometry=",
            r["geometry_holes"],
            "pixels=",
            r["hole_pixels"],
        )

    print()
    print("HIGHEST HOLE PIXELS")

    for r in sorted(
        rr,
        key=lambda x:
            x["hole_pixels"],
        reverse=True,
    )[:10]:

        print(
            r["sample"],
            "pixels=",
            r["hole_pixels"],
            "components=",
            r["hole_components"],
            "geometry=",
            r["geometry_holes"],
        )


print()
print(
    "=================================================="
)
print("GEOMETRY / COMPONENT MISMATCH")
print(
    "=================================================="
)

mismatch = [
    r
    for r in rows
    if (
        r["geometry_holes"] >= 0
        and
        r["geometry_holes"]
        !=
        r["hole_components"]
    )
]

mismatch = sorted(
    mismatch,
    key=lambda r:
        abs(r["component_delta"]),
    reverse=True,
)

for r in mismatch:

    print(
        r["sample"],
        r["camera"],
        "geometry=",
        r["geometry_holes"],
        "components=",
        r["hole_components"],
        "delta=",
        r["component_delta"],
        "pixels=",
        r["hole_pixels"],
    )


# ------------------------------------------------------------
# Select visual-review candidates
# ------------------------------------------------------------

selected = set()


# No-hole
for r in rows:

    if r["hole_components"] == 0:

        selected.add(
            (
                r["sample"],
                r["camera"],
            )
        )


# Lowest 6 component-count samples per camera
for camera in CAMERAS:

    rr = [
        r
        for r in rows
        if r["camera"] == camera
    ]

    rr = sorted(
        rr,
        key=lambda x: (
            x["hole_components"],
            x["hole_pixels"],
        ),
    )

    for r in rr[:6]:

        selected.add(
            (
                r["sample"],
                r["camera"],
            )
        )


# Highest 6 hole-pixel samples per camera
for camera in CAMERAS:

    rr = [
        r
        for r in rows
        if r["camera"] == camera
    ]

    rr = sorted(
        rr,
        key=lambda x:
            x["hole_pixels"],
        reverse=True,
    )

    for r in rr[:6]:

        selected.add(
            (
                r["sample"],
                r["camera"],
            )
        )


# All geometry/component mismatches
for r in mismatch:

    selected.add(
        (
            r["sample"],
            r["camera"],
        )
    )


# ------------------------------------------------------------
# Generate visual overlays
# ------------------------------------------------------------

tiles = []

for sid, camera in sorted(
    selected,
    key=lambda x: (
        x[0],
        CAMERAS.index(x[1]),
    ),
):

    image_path = (
        SESSION
        / camera
        / f"{sid}.png"
    )

    mask_path = (
        MASK_ROOT
        / camera
        / f"{sid}.png"
    )

    image = cv2.imread(
        str(image_path)
    )

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_UNCHANGED,
    )

    if (
        image is None
        or mask is None
    ):
        continue

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    overlay = image.copy()

    # Class 1 = board
    overlay[
        mask == 1
    ] = (
        80,
        180,
        80,
    )

    # Class 2 = hole
    overlay[
        mask == 2
    ] = (
        0,
        180,
        255,
    )

    # Ignore
    overlay[
        mask == 255
    ] = (
        255,
        0,
        255,
    )

    shown = cv2.addWeighted(
        image,
        0.60,
        overlay,
        0.40,
        0,
    )

    row = next(
        r
        for r in rows
        if (
            r["sample"] == sid
            and
            r["camera"] == camera
        )
    )

    text = (
        f"{sid} {camera} "
        f"comp={row['hole_components']} "
        f"geo={row['geometry_holes']} "
        f"holepx={row['hole_pixels']}"
    )

    cv2.rectangle(
        shown,
        (0, 0),
        (shown.shape[1], 38),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        shown,
        text,
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    shown = cv2.resize(
        shown,
        (480, 300),
    )

    tiles.append(
        (
            sid,
            camera,
            shown,
        )
    )


# ------------------------------------------------------------
# Contact sheets
# ------------------------------------------------------------

PER_PAGE = 12
COLS = 3

for start in range(
    0,
    len(tiles),
    PER_PAGE,
):

    page = tiles[
        start:
        start + PER_PAGE
    ]

    while (
        len(page)
        %
        COLS
        != 0
    ):

        blank = np.zeros(
            (300, 480, 3),
            dtype=np.uint8,
        )

        page.append(
            (
                "",
                "",
                blank,
            )
        )

    rows_img = []

    for i in range(
        0,
        len(page),
        COLS,
    ):

        rows_img.append(
            np.hstack(
                [
                    page[j][2]
                    for j in range(
                        i,
                        i + COLS,
                    )
                ]
            )
        )

    sheet = np.vstack(
        rows_img
    )

    out_path = (
        OUT
        /
        f"review_{start // PER_PAGE + 1:02d}.jpg"
    )

    cv2.imwrite(
        str(out_path),
        sheet,
    )

    print(
        "SAVED:",
        out_path,
    )


print()
print(
    "SELECTED REVIEW IMAGES =",
    len(tiles),
)

print(
    "REVIEW DIR =",
    OUT,
)

print()
print(
    "VISUAL_QA_REPORT_READY"
)
