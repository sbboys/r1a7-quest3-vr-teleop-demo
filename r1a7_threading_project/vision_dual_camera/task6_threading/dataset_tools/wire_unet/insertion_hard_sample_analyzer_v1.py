#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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


def source_of(record, camera):
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


def event_contains(events, text):
    text = text.upper()

    return any(
        text in str(e).upper()
        for e in events
    )


def score_record(record):

    events = (
        record.get(
            "events",
            [],
        )
        or
        []
    )

    wire_score = 0
    hole_score = 0
    control_score = 0

    wire_reasons = []
    hole_reasons = []
    control_reasons = []

    # --------------------------------------------------------
    # Wire / FREE quality
    # --------------------------------------------------------

    sources = {
        c: source_of(
            record,
            c,
        )
        for c in CAMERAS
    }

    for camera, source in sources.items():

        if source == "DISPLAY_ONLY":
            wire_score += 8
            wire_reasons.append(
                f"{camera}:DISPLAY_ONLY"
            )

        elif source == "TIP_RELATIVE_PRED":
            wire_score += 5
            wire_reasons.append(
                f"{camera}:TIP_RELATIVE_PRED"
            )

    if (
        sources["external"]
        ==
        "DISPLAY_ONLY"
        and
        sources["right_wrist"]
        ==
        "DISPLAY_ONLY"
    ):
        wire_score += 12
        wire_reasons.append(
            "BOTH_CONTROL_FREE_LOST"
        )

    for event in events:

        e = str(event).upper()

        if "FREE_JUMP" in e:
            wire_score += 8
            wire_reasons.append(
                str(event)
            )

        if "FREE_SOURCE_CHANGE" in e:
            wire_score += 4
            wire_reasons.append(
                str(event)
            )

        if "GUIDANCE_INVALID" in e:
            wire_score += 4
            wire_reasons.append(
                str(event)
            )

    # --------------------------------------------------------
    # Hole / target quality
    # --------------------------------------------------------

    selected = (
        record.get(
            "target_selected",
            {},
        )
        or
        {}
    )

    valid = (
        record.get(
            "target_valid",
            {},
        )
        or
        {}
    )

    for camera in CAMERAS:

        if (
            bool(
                selected.get(
                    camera,
                    False,
                )
            )
            and
            not bool(
                valid.get(
                    camera,
                    False,
                )
            )
        ):
            hole_score += 10
            hole_reasons.append(
                f"{camera}:SELECTED_BUT_INVALID"
            )

    for event in events:

        e = str(event).upper()

        if "TARGET_INVALID" in e:
            hole_score += 10
            hole_reasons.append(
                str(event)
            )

        if "HOLE_JUMP" in e:
            hole_score += 10
            hole_reasons.append(
                str(event)
            )

        if "TARGET_VALID_CHANGE" in e:
            hole_score += 5
            hole_reasons.append(
                str(event)
            )

        if "SINGLE_VIEW" in e:
            hole_score += 7
            hole_reasons.append(
                "SINGLE_VIEW"
            )

    target_state = (
        record.get(
            "target_state",
            {},
        )
        or
        {}
    )

    if target_state.get(
        "degraded_single_view",
        False,
    ):
        hole_score += 5
        hole_reasons.append(
            "DEGRADED_SINGLE_VIEW"
        )

    control_count = (
        target_state.get(
            "control_target_count"
        )
    )

    if (
        record.get("phase")
        ==
        "INSERTION_TRACKING"
        and
        control_count
        ==
        0
    ):
        hole_score += 15
        hole_reasons.append(
            "NO_VALID_CONTROL_TARGET"
        )

    # --------------------------------------------------------
    # Control / system failures
    # --------------------------------------------------------

    if event_contains(
        events,
        "FUSION_STOP",
    ):
        control_score += 10
        control_reasons.append(
            "FUSION_STOP"
        )

    if event_contains(
        events,
        "GATE_DISABLED",
    ):
        control_score += 8
        control_reasons.append(
            "GATE_DISABLED"
        )

    if event_contains(
        events,
        "BLIND_GRACE",
    ):
        control_score += 6
        control_reasons.append(
            "BLIND_GRACE"
        )

    if event_contains(
        events,
        "DUAL_Z_CONFLICT",
    ):
        control_score += 12
        control_reasons.append(
            "DUAL_Z_CONFLICT"
        )

    # Control failures increase the value of the corresponding
    # perception sample, but perception remains the primary score.

    wire_total = (
        wire_score
        +
        0.35
        *
        control_score
    )

    hole_total = (
        hole_score
        +
        0.35
        *
        control_score
    )

    return {
        "wire_score":
            round(
                wire_total,
                2,
            ),

        "hole_score":
            round(
                hole_total,
                2,
            ),

        "control_score":
            control_score,

        "wire_reasons":
            sorted(
                set(
                    wire_reasons
                )
            ),

        "hole_reasons":
            sorted(
                set(
                    hole_reasons
                )
            ),

        "control_reasons":
            sorted(
                set(
                    control_reasons
                )
            ),
    }


def temporal_select(
    scored,
    score_key,
    top_k,
    min_gap_sec,
):

    ranked = sorted(
        scored,
        key=lambda x: (
            x["scores"][
                score_key
            ],
            x["record"].get(
                "timestamp",
                0.0,
            ),
        ),
        reverse=True,
    )

    selected = []

    for item in ranked:

        score = item[
            "scores"
        ][
            score_key
        ]

        if score <= 0:
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

        too_close = any(
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
            min_gap_sec
            for old in selected
        )

        if too_close:
            continue

        selected.append(
            item
        )

        if len(
            selected
        ) >= top_k:
            break

    return selected


def link_or_copy(
    src,
    dst,
):

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


def make_triplet(
    images,
    out_path,
):

    loaded = []

    for camera in CAMERAS:

        path = images.get(
            camera
        )

        if not path:
            continue

        image = cv2.imread(
            str(path)
        )

        if image is None:
            continue

        cv2.putText(
            image,
            camera,
            (
                15,
                35,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (
                255,
                255,
                255,
            ),
            2,
            cv2.LINE_AA,
        )

        loaded.append(
            image
        )

    if not loaded:
        return False

    h = min(
        x.shape[0]
        for x in loaded
    )

    resized = []

    for image in loaded:

        scale = (
            h
            /
            image.shape[0]
        )

        width = int(
            round(
                image.shape[1]
                *
                scale
            )
        )

        resized.append(
            cv2.resize(
                image,
                (
                    width,
                    h,
                ),
            )
        )

    triplet = np.hstack(
        resized
    )

    out_path = Path(
        out_path
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return cv2.imwrite(
        str(
            out_path
        ),
        triplet,
    )


def export_queue(
    session,
    selected,
    queue_name,
    score_key,
    output_root,
):

    queue_dir = (
        output_root
        /
        "annotation_queue"
        /
        queue_name
    )

    queue_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_root
        /
        f"{queue_name}_hard_samples.csv"
    )

    csv_rows = []

    for rank, item in enumerate(
        selected,
        1,
    ):

        record = item[
            "record"
        ]

        scores = item[
            "scores"
        ]

        frame = int(
            record.get(
                "frame_counter",
                -1,
            )
        )

        sample_dir = (
            queue_dir
            /
            (
                f"rank_{rank:03d}_"
                f"frame_{frame:08d}"
            )
        )

        sample_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        linked_images = {}

        for camera in CAMERAS:

            rel = (
                record.get(
                    "image_paths",
                    {},
                )
                or
                {}
            ).get(
                camera
            )

            if not rel:
                continue

            src = (
                session
                /
                rel
            )

            if not src.is_file():
                continue

            dst = (
                sample_dir
                /
                f"{camera}.jpg"
            )

            link_or_copy(
                src,
                dst,
            )

            linked_images[
                camera
            ] = dst

        make_triplet(
            linked_images,
            sample_dir
            /
            "triplet.jpg",
        )

        metadata = {
            "rank":
                rank,

            "queue":
                queue_name,

            "score":
                scores[
                    score_key
                ],

            "scores":
                scores,

            "record":
                record,
        }

        (
            sample_dir
            /
            "metadata.json"
        ).write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        free = (
            record.get(
                "free_guidance",
                {},
            )
            or
            {}
        )

        selected_state = (
            record.get(
                "target_selected",
                {},
            )
            or
            {}
        )

        valid_state = (
            record.get(
                "target_valid",
                {},
            )
            or
            {}
        )

        reasons_key = (
            "wire_reasons"
            if
            queue_name
            ==
            "wire"
            else
            "hole_reasons"
        )

        csv_rows.append(
            {
                "rank":
                    rank,

                "score":
                    scores[
                        score_key
                    ],

                "frame":
                    frame,

                "timestamp":
                    record.get(
                        "timestamp"
                    ),

                "phase":
                    record.get(
                        "phase"
                    ),

                "reasons":
                    ";".join(
                        scores[
                            reasons_key
                        ]
                    ),

                "control_reasons":
                    ";".join(
                        scores[
                            "control_reasons"
                        ]
                    ),

                "external_free":
                    free.get(
                        "external",
                        {},
                    ).get(
                        "source"
                    ),

                "right_wrist_free":
                    free.get(
                        "right_wrist",
                        {},
                    ).get(
                        "source"
                    ),

                "left_wrist_free":
                    free.get(
                        "left_wrist",
                        {},
                    ).get(
                        "source"
                    ),

                "external_target":
                    (
                        f"{int(bool(selected_state.get('external')))}"
                        "/"
                        f"{int(bool(valid_state.get('external')))}"
                    ),

                "right_wrist_target":
                    (
                        f"{int(bool(selected_state.get('right_wrist')))}"
                        "/"
                        f"{int(bool(valid_state.get('right_wrist')))}"
                    ),

                "left_wrist_target":
                    (
                        f"{int(bool(selected_state.get('left_wrist')))}"
                        "/"
                        f"{int(bool(valid_state.get('left_wrist')))}"
                    ),

                "sample_dir":
                    str(
                        sample_dir
                    ),
            }
        )

    fields = [
        "rank",
        "score",
        "frame",
        "timestamp",
        "phase",
        "reasons",
        "control_reasons",
        "external_free",
        "right_wrist_free",
        "left_wrist_free",
        "external_target",
        "right_wrist_target",
        "left_wrist_target",
        "sample_dir",
    ]

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as fp:

        writer = csv.DictWriter(
            fp,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            csv_rows
        )

    return csv_path


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "session",
        type=Path,
    )

    parser.add_argument(
        "--top-wire",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--top-hole",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--min-gap-sec",
        type=float,
        default=0.75,
    )

    args = parser.parse_args()

    session = args.session.resolve()

    records_path = (
        session
        /
        "records.jsonl"
    )

    if not records_path.is_file():
        raise SystemExit(
            f"records.jsonl not found: {records_path}"
        )

    output_root = (
        session
        /
        "hard_sample_analysis_v1"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

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

            records.append(
                {
                    "record":
                        record,

                    "scores":
                        score_record(
                            record
                        ),
                }
            )

    wire_selected = temporal_select(
        records,
        "wire_score",
        args.top_wire,
        args.min_gap_sec,
    )

    hole_selected = temporal_select(
        records,
        "hole_score",
        args.top_hole,
        args.min_gap_sec,
    )

    wire_csv = export_queue(
        session,
        wire_selected,
        "wire",
        "wire_score",
        output_root,
    )

    hole_csv = export_queue(
        session,
        hole_selected,
        "hole",
        "hole_score",
        output_root,
    )

    summary = {
        "session":
            str(
                session
            ),

        "total_records":
            len(
                records
            ),

        "wire_selected":
            len(
                wire_selected
            ),

        "hole_selected":
            len(
                hole_selected
            ),

        "min_gap_sec":
            args.min_gap_sec,

        "wire_csv":
            str(
                wire_csv
            ),

        "hole_csv":
            str(
                hole_csv
            ),
    }

    (
        output_root
        /
        "analysis_summary.json"
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
        "HARD_SAMPLE_ANALYSIS_OK"
    )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
