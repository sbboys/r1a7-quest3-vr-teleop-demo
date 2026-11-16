#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path

import cv2
import numpy as np


CAMERAS = (
    "external",
    "right_wrist",
    "left_wrist",
)

DEFAULT_QUOTA = {
    "external": 40,
    "right_wrist": 35,
    "left_wrist": 25,
}


def free_source(record, camera):
    return str(
        record.get(
            "free_guidance",
            {},
        ).get(
            camera,
            {},
        ).get(
            "source",
            "",
        )
        or
        ""
    ).upper()


def camera_score(record, camera):
    events = [
        str(x)
        for x in (
            record.get(
                "events",
                [],
            )
            or
            []
        )
    ]

    source = free_source(
        record,
        camera,
    )

    selected = bool(
        (
            record.get(
                "target_selected",
                {},
            )
            or
            {}
        ).get(
            camera,
            False,
        )
    )

    valid = bool(
        (
            record.get(
                "target_valid",
                {},
            )
            or
            {}
        ).get(
            camera,
            False,
        )
    )

    score = 0.0
    reasons = []
    wire_reasons = []
    hole_reasons = []

    # --------------------------------------------------------
    # Wire / FREE
    # --------------------------------------------------------

    if source == "DISPLAY_ONLY":
        score += 10.0
        reasons.append("DISPLAY_ONLY")
        wire_reasons.append("DISPLAY_ONLY")

    elif source == "TIP_RELATIVE_PRED":
        score += 7.0
        reasons.append("TIP_RELATIVE_PRED")
        wire_reasons.append("TIP_RELATIVE_PRED")

    elif source == "MEASURED":
        # Normal samples still have value.
        score += 1.0
        reasons.append("MEASURED")

    for event in events:
        upper = event.upper()

        if (
            event.startswith(
                camera + ":"
            )
            and
            "FREE_JUMP"
            in upper
        ):
            score += 9.0
            reasons.append("FREE_JUMP")
            wire_reasons.append("FREE_JUMP")

        if (
            event.startswith(
                camera + ":"
            )
            and
            "FREE_SOURCE_CHANGE"
            in upper
        ):
            score += 5.0
            reasons.append("FREE_SOURCE_CHANGE")
            wire_reasons.append("FREE_SOURCE_CHANGE")

        if (
            event.startswith(
                camera + ":"
            )
            and
            "GUIDANCE_INVALID"
            in upper
        ):
            score += 6.0
            reasons.append("GUIDANCE_INVALID")
            wire_reasons.append("GUIDANCE_INVALID")

    # --------------------------------------------------------
    # Hole
    # --------------------------------------------------------

    if selected and not valid:
        score += 10.0
        reasons.append("SELECTED_TARGET_INVALID")
        hole_reasons.append("SELECTED_TARGET_INVALID")

    for event in events:
        upper = event.upper()

        if (
            event.startswith(
                camera + ":"
            )
            and
            "TARGET_INVALID"
            in upper
        ):
            score += 9.0
            reasons.append("TARGET_INVALID")
            hole_reasons.append("TARGET_INVALID")

        if (
            event.startswith(
                camera + ":"
            )
            and
            "TARGET_VALID_CHANGE"
            in upper
        ):
            score += 5.0
            reasons.append("TARGET_VALID_CHANGE")
            hole_reasons.append("TARGET_VALID_CHANGE")

        if (
            event.startswith(
                camera + ":"
            )
            and
            "HOLE_JUMP"
            in upper
        ):
            score += 10.0
            reasons.append("HOLE_JUMP")
            hole_reasons.append("HOLE_JUMP")

    # --------------------------------------------------------
    # System-level context
    # --------------------------------------------------------

    upper_events = [
        e.upper()
        for e in events
    ]

    if any(
        "SINGLE_VIEW" in e
        for e in upper_events
    ):
        score += 3.0
        reasons.append("SINGLE_VIEW")

    if any(
        "FUSION_STOP" in e
        for e in upper_events
    ):
        score += 2.0
        reasons.append("FUSION_STOP")

    if any(
        "BLIND_GRACE" in e
        for e in upper_events
    ):
        score += 2.0
        reasons.append("BLIND_GRACE")

    if any(
        "DUAL_Z_CONFLICT" in e
        for e in upper_events
    ):
        score += 3.0
        reasons.append("DUAL_Z_CONFLICT")

    # Right Wrist hole failures are currently especially useful.
    if (
        camera == "right_wrist"
        and
        hole_reasons
    ):
        score += 3.0

    # External FREE failures are currently highest priority.
    if (
        camera == "external"
        and
        wire_reasons
    ):
        score += 3.0

    if wire_reasons and hole_reasons:
        task_hint = "wire+hole"
    elif wire_reasons:
        task_hint = "wire"
    elif hole_reasons:
        task_hint = "hole"
    else:
        task_hint = "normal"

    return {
        "score": round(
            score,
            3,
        ),
        "reasons": sorted(
            set(
                reasons
            )
        ),
        "wire_reasons": sorted(
            set(
                wire_reasons
            )
        ),
        "hole_reasons": sorted(
            set(
                hole_reasons
            )
        ),
        "task_hint": task_hint,
        "source": source,
        "selected": selected,
        "valid": valid,
    }


def phash64(image_path):
    image = cv2.imread(
        str(
            image_path
        ),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        return None

    image = cv2.resize(
        image,
        (
            32,
            32,
        ),
        interpolation=cv2.INTER_AREA,
    )

    image = np.float32(
        image
    )

    dct = cv2.dct(
        image
    )

    block = dct[
        :8,
        :8,
    ].copy()

    flat = block.flatten()

    # Ignore DC coefficient for threshold.
    threshold = float(
        np.median(
            flat[1:]
        )
    )

    bits = (
        flat
        >
        threshold
    )

    value = 0

    for bit in bits:
        value = (
            value << 1
        ) | int(
            bool(bit)
        )

    return int(
        value
    )


def hamming64(a, b):
    return (
        int(a)
        ^
        int(b)
    ).bit_count()


def load_candidates(session):
    records_path = (
        session
        /
        "records.jsonl"
    )

    candidates = {
        camera: []
        for camera in CAMERAS
    }

    with records_path.open(
        "r",
        encoding="utf-8",
    ) as fp:

        for line_no, line in enumerate(
            fp,
            1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(
                    line
                )
            except Exception as exc:
                print(
                    "SKIP BAD JSON",
                    line_no,
                    repr(exc),
                )
                continue

            image_paths = (
                record.get(
                    "image_paths",
                    {},
                )
                or
                {}
            )

            for camera in CAMERAS:
                rel = image_paths.get(
                    camera
                )

                if not rel:
                    continue

                image_path = (
                    session
                    /
                    rel
                )

                if not image_path.is_file():
                    continue

                info = camera_score(
                    record,
                    camera,
                )

                candidates[
                    camera
                ].append(
                    {
                        "camera": camera,
                        "record": record,
                        "image_path": image_path,
                        "score": info[
                            "score"
                        ],
                        "reasons": info[
                            "reasons"
                        ],
                        "task_hint": info[
                            "task_hint"
                        ],
                        "source": info[
                            "source"
                        ],
                        "selected": info[
                            "selected"
                        ],
                        "valid": info[
                            "valid"
                        ],
                    }
                )

    return candidates


def select_diverse(
    candidates,
    quota,
    min_time_gap,
    hash_threshold,
    normal_fraction,
):
    if not candidates:
        return []

    # Precompute hashes once.
    for item in candidates:
        item["phash"] = phash64(
            item[
                "image_path"
            ]
        )

    # Hard / normal split.
    hard = [
        x
        for x in candidates
        if x[
            "task_hint"
        ]
        !=
        "normal"
    ]

    normal = [
        x
        for x in candidates
        if x[
            "task_hint"
        ]
        ==
        "normal"
    ]

    hard.sort(
        key=lambda x: (
            x["score"],
            float(
                x[
                    "record"
                ].get(
                    "timestamp",
                    0.0,
                )
                or
                0.0
            ),
        ),
        reverse=True,
    )

    # Spread normal samples over time instead of taking adjacent frames.
    normal.sort(
        key=lambda x: float(
            x[
                "record"
            ].get(
                "timestamp",
                0.0,
            )
            or
            0.0
        )
    )

    normal_target = int(
        round(
            quota
            *
            normal_fraction
        )
    )

    hard_target = max(
        0,
        quota
        -
        normal_target
    )

    selected = []

    def accept(item, time_gap, hash_gap):
        ts = float(
            item[
                "record"
            ].get(
                "timestamp",
                0.0,
            )
            or
            0.0
        )

        ph = item.get(
            "phash"
        )

        for old in selected:
            old_ts = float(
                old[
                    "record"
                ].get(
                    "timestamp",
                    0.0,
                )
                or
                0.0
            )

            if abs(
                ts
                -
                old_ts
            ) < time_gap:
                return False

            old_ph = old.get(
                "phash"
            )

            if (
                ph is not None
                and
                old_ph is not None
                and
                hamming64(
                    ph,
                    old_ph,
                )
                <
                hash_gap
            ):
                return False

        return True

    # --------------------------------------------------------
    # Pass 1: hard samples with strict dedup.
    # --------------------------------------------------------

    for item in hard:
        if len(
            selected
        ) >= hard_target:
            break

        if accept(
            item,
            min_time_gap,
            hash_threshold,
        ):
            selected.append(
                item
            )

    # --------------------------------------------------------
    # Pass 2: normal samples distributed through time.
    # --------------------------------------------------------

    if normal_target > 0 and normal:

        if len(normal) <= normal_target:
            normal_order = normal
        else:
            indices = np.linspace(
                0,
                len(normal) - 1,
                normal_target * 3,
            )

            normal_order = [
                normal[
                    int(
                        round(i)
                    )
                ]
                for i in indices
            ]

        for item in normal_order:
            if len(
                selected
            ) >= quota:
                break

            if accept(
                item,
                min_time_gap,
                hash_threshold,
            ):
                selected.append(
                    item
                )

    # --------------------------------------------------------
    # Pass 3: relax hash threshold slightly if quota not reached.
    # --------------------------------------------------------

    pool = sorted(
        candidates,
        key=lambda x: x[
            "score"
        ],
        reverse=True,
    )

    for relaxed_hash in (
        max(
            4,
            hash_threshold - 2,
        ),
        3,
        1,
    ):
        if len(
            selected
        ) >= quota:
            break

        for item in pool:
            if len(
                selected
            ) >= quota:
                break

            if item in selected:
                continue

            if accept(
                item,
                min_time_gap
                *
                0.70,
                relaxed_hash,
            ):
                selected.append(
                    item
                )

    # --------------------------------------------------------
    # Pass 4: final fill by temporal diversity.
    # --------------------------------------------------------

    if len(selected) < quota:

        for item in pool:
            if len(
                selected
            ) >= quota:
                break

            if item in selected:
                continue

            ts = float(
                item[
                    "record"
                ].get(
                    "timestamp",
                    0.0,
                )
                or
                0.0
            )

            if any(
                abs(
                    ts
                    -
                    float(
                        old[
                            "record"
                        ].get(
                            "timestamp",
                            0.0,
                        )
                        or
                        0.0
                    )
                )
                <
                0.15
                for old in selected
            ):
                continue

            selected.append(
                item
            )

    selected = selected[
        :quota
    ]

    selected.sort(
        key=lambda x: float(
            x[
                "record"
            ].get(
                "timestamp",
                0.0,
            )
            or
            0.0
        )
    )

    return selected


def link_or_copy(src, dst):
    src = Path(src)
    dst = Path(dst)

    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if dst.exists():
        dst.unlink()

    try:
        os.link(
            src,
            dst,
        )
        return "hardlink"

    except Exception:
        shutil.copy2(
            src,
            dst,
        )
        return "copy"


def build_contact_sheet(
    items,
    output_path,
    camera,
):
    thumbs = []

    for index, item in enumerate(
        items,
        1,
    ):
        image = cv2.imread(
            str(
                item[
                    "output_path"
                ]
            )
        )

        if image is None:
            continue

        h, w = image.shape[:2]

        target_w = 320
        scale = target_w / max(
            w,
            1,
        )

        target_h = max(
            1,
            int(
                round(
                    h
                    *
                    scale
                )
            ),
        )

        image = cv2.resize(
            image,
            (
                target_w,
                target_h,
            ),
        )

        title = (
            f"{camera} #{index:02d} "
            f"{item['task_hint']} "
            f"{item['free_source']}"
        )

        cv2.rectangle(
            image,
            (
                0,
                0,
            ),
            (
                target_w,
                32,
            ),
            (
                0,
                0,
                0,
            ),
            -1,
        )

        cv2.putText(
            image,
            title,
            (
                5,
                22,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (
                255,
                255,
                255,
            ),
            1,
            cv2.LINE_AA,
        )

        thumbs.append(
            image
        )

    if not thumbs:
        return

    cols = 5

    max_h = max(
        x.shape[0]
        for x in thumbs
    )

    rows = int(
        math.ceil(
            len(thumbs)
            /
            cols
        )
    )

    sheet = np.zeros(
        (
            rows * max_h,
            cols * 320,
            3,
        ),
        dtype=np.uint8,
    )

    for i, image in enumerate(
        thumbs
    ):
        row = i // cols
        col = i % cols

        y0 = row * max_h
        x0 = col * 320

        sheet[
            y0:y0 + image.shape[0],
            x0:x0 + image.shape[1],
        ] = image

    cv2.imwrite(
        str(
            output_path
        ),
        sheet,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "session",
        type=Path,
    )

    parser.add_argument(
        "--external",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--right-wrist",
        type=int,
        default=35,
    )

    parser.add_argument(
        "--left-wrist",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--min-time-gap",
        type=float,
        default=0.55,
    )

    parser.add_argument(
        "--hash-threshold",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--normal-fraction",
        type=float,
        default=0.25,
    )

    args = parser.parse_args()

    session = args.session.resolve()

    if not (
        session
        /
        "records.jsonl"
    ).is_file():
        raise SystemExit(
            f"records.jsonl missing: {session}"
        )

    quota = {
        "external":
            args.external,
        "right_wrist":
            args.right_wrist,
        "left_wrist":
            args.left_wrist,
    }

    output_root = (
        session
        /
        "dedup_annotation_v1"
    )

    if output_root.exists():
        shutil.rmtree(
            output_root
        )

    (
        output_root
        /
        "images"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates = load_candidates(
        session
    )

    all_selected = []

    summary = {
        "session":
            str(
                session
            ),
        "output_root":
            str(
                output_root
            ),
        "requested_total":
            int(
                sum(
                    quota.values()
                )
            ),
        "min_time_gap":
            args.min_time_gap,
        "hash_threshold":
            args.hash_threshold,
        "normal_fraction":
            args.normal_fraction,
        "cameras": {},
    }

    for camera in CAMERAS:

        chosen = select_diverse(
            candidates[
                camera
            ],
            quota[
                camera
            ],
            args.min_time_gap,
            args.hash_threshold,
            args.normal_fraction,
        )

        camera_dir = (
            output_root
            /
            "images"
            /
            camera
        )

        camera_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        exported = []

        for index, item in enumerate(
            chosen,
            1,
        ):
            frame = int(
                item[
                    "record"
                ].get(
                    "frame_counter",
                    -1,
                )
            )

            dst = (
                camera_dir
                /
                (
                    f"{camera}_"
                    f"{index:03d}_"
                    f"frame_{frame:08d}.jpg"
                )
            )

            method = link_or_copy(
                item[
                    "image_path"
                ],
                dst,
            )

            row = {
                "global_id": None,
                "camera": camera,
                "camera_index": index,
                "frame_counter": frame,
                "timestamp":
                    item[
                        "record"
                    ].get(
                        "timestamp"
                    ),
                "phase":
                    item[
                        "record"
                    ].get(
                        "phase"
                    ),
                "score":
                    item[
                        "score"
                    ],
                "task_hint":
                    item[
                        "task_hint"
                    ],
                "free_source":
                    item[
                        "source"
                    ],
                "target_selected":
                    int(
                        item[
                            "selected"
                        ]
                    ),
                "target_valid":
                    int(
                        item[
                            "valid"
                        ]
                    ),
                "reasons":
                    ";".join(
                        item[
                            "reasons"
                        ]
                    ),
                "source_image":
                    str(
                        item[
                            "image_path"
                        ]
                    ),
                "selected_image":
                    str(
                        dst
                    ),
                "storage_method":
                    method,
                "output_path":
                    dst,
            }

            exported.append(
                row
            )

            all_selected.append(
                row
            )

        build_contact_sheet(
            exported,
            output_root
            /
            f"contact_sheet_{camera}.jpg",
            camera,
        )

        task_counts = {}

        for x in exported:
            task_counts[
                x[
                    "task_hint"
                ]
            ] = (
                task_counts.get(
                    x[
                        "task_hint"
                    ],
                    0,
                )
                +
                1
            )

        summary[
            "cameras"
        ][
            camera
        ] = {
            "candidate_count":
                len(
                    candidates[
                        camera
                    ]
                ),
            "requested":
                quota[
                    camera
                ],
            "selected":
                len(
                    exported
                ),
            "task_counts":
                task_counts,
        }

    # Stable global numbering.
    all_selected.sort(
        key=lambda x: (
            x[
                "camera"
            ],
            x[
                "camera_index"
            ],
        )
    )

    for global_id, row in enumerate(
        all_selected,
        1,
    ):
        row[
            "global_id"
        ] = global_id

    manifest_path = (
        output_root
        /
        "manifest.csv"
    )

    fields = [
        "global_id",
        "camera",
        "camera_index",
        "frame_counter",
        "timestamp",
        "phase",
        "score",
        "task_hint",
        "free_source",
        "target_selected",
        "target_valid",
        "reasons",
        "source_image",
        "selected_image",
        "storage_method",
    ]

    with manifest_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as fp:

        writer = csv.DictWriter(
            fp,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in all_selected:
            writer.writerow(
                row
            )

    summary[
        "selected_total"
    ] = len(
        all_selected
    )

    (
        output_root
        /
        "selection_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "DEDUP_ANNOTATION_SELECTION_OK"
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
