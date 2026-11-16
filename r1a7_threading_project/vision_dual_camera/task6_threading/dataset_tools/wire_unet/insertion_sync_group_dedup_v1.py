#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


CAMERAS = (
    "external",
    "right_wrist",
    "left_wrist",
)

CONTROL_CAMERAS = (
    "external",
    "right_wrist",
)


def image_path(session, record, camera):
    rel = (
        record.get("image_paths", {})
        or {}
    ).get(camera)

    if not rel:
        return None

    p = Path(rel)

    if not p.is_absolute():
        p = session / p

    return p


def source_of(record, camera):
    return str(
        (
            record.get("free_guidance", {})
            or {}
        ).get(camera, {}).get(
            "source",
            "",
        )
        or ""
    ).upper()


def selected_of(record, camera):
    return bool(
        (
            record.get("target_selected", {})
            or {}
        ).get(camera, False)
    )


def valid_of(record, camera):
    return bool(
        (
            record.get("target_valid", {})
            or {}
        ).get(camera, False)
    )


def score_record(record):
    events = [
        str(x)
        for x in (
            record.get("events", [])
            or []
        )
    ]

    upper_events = [
        x.upper()
        for x in events
    ]

    score = 0.0
    reasons = []

    # ---------------------------------------------------------
    # FREE / wire quality
    # ---------------------------------------------------------

    for camera in CONTROL_CAMERAS:
        source = source_of(
            record,
            camera,
        )

        if source == "DISPLAY_ONLY":
            score += 9.0
            reasons.append(
                f"{camera}:DISPLAY_ONLY"
            )

        elif source == "TIP_RELATIVE_PRED":
            score += 6.0
            reasons.append(
                f"{camera}:TIP_RELATIVE_PRED"
            )

    # Left Wrist is still useful, but is not a control camera.
    left_source = source_of(
        record,
        "left_wrist",
    )

    if left_source == "DISPLAY_ONLY":
        score += 2.0
        reasons.append(
            "left_wrist:DISPLAY_ONLY"
        )

    elif left_source == "TIP_RELATIVE_PRED":
        score += 2.0
        reasons.append(
            "left_wrist:TIP_RELATIVE_PRED"
        )

    if (
        source_of(record, "external")
        == "DISPLAY_ONLY"
        and
        source_of(record, "right_wrist")
        == "DISPLAY_ONLY"
    ):
        score += 12.0
        reasons.append(
            "BOTH_CONTROL_FREE_LOST"
        )

    # ---------------------------------------------------------
    # Hole / target quality
    # ---------------------------------------------------------

    for camera in CONTROL_CAMERAS:
        if (
            selected_of(record, camera)
            and
            not valid_of(record, camera)
        ):
            score += 10.0
            reasons.append(
                f"{camera}:TARGET_INVALID"
            )

    # ---------------------------------------------------------
    # Events
    # ---------------------------------------------------------

    # PLAY_ONLY_VISUAL_SCORING_V2
    # Visual-only acquisition: ignore controller/fusion inactive
    # states when ranking perception samples.
    weights = {
        "FREE_JUMP": 8.0,
        "FREE_SOURCE_CHANGE": 5.0,
        "GUIDANCE_INVALID": 5.0,
        "TARGET_INVALID": 10.0,
        "TARGET_VALID_CHANGE": 5.0,
        "HOLE_JUMP": 10.0,
        "SINGLE_VIEW": 7.0,
        "RECOVERY": 4.0,
    }

    for key, weight in weights.items():
        if any(
            key in e
            for e in upper_events
        ):
            score += weight
            reasons.append(key)

    # ---------------------------------------------------------
    # Determine whether this is a relatively stable sample.
    #
    # Left Wrist DISPLAY_ONLY is deliberately NOT enough to
    # classify the group as hard.
    # ---------------------------------------------------------

    stable = (
        source_of(record, "external")
        == "MEASURED"
        and
        source_of(record, "right_wrist")
        == "MEASURED"
    )

    # PLAY_ONLY_VISUAL_SCORING_V2
    # Stable/hard uses perception failures from control cameras only.
    perception_severe_keys = (
        "FREE_JUMP",
        "FREE_SOURCE_CHANGE",
        "GUIDANCE_INVALID",
        "TARGET_INVALID",
        "TARGET_VALID_CHANGE",
        "HOLE_JUMP",
    )

    for event in events:
        event_upper = event.upper()

        control_camera_event = any(
            event.startswith(camera + ":")
            for camera in CONTROL_CAMERAS
        )

        if (
            control_camera_event
            and
            any(
                key in event_upper
                for key in perception_severe_keys
            )
        ):
            stable = False

    for camera in CONTROL_CAMERAS:
        if (
            selected_of(record, camera)
            and
            not valid_of(record, camera)
        ):
            stable = False

    category = (
        "stable"
        if stable
        else "hard"
    )

    # Stable samples are still useful for training balance.
    if stable:
        score += 1.0
        reasons.append(
            "STABLE_CONTROL_VIEW"
        )

    return {
        "score": round(score, 3),
        "category": category,
        "reasons": sorted(
            set(reasons)
        ),
    }


def phash64(path):
    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        return None

    image = cv2.resize(
        image,
        (32, 32),
        interpolation=cv2.INTER_AREA,
    )

    image = np.float32(image)

    dct = cv2.dct(image)

    flat = dct[:8, :8].flatten()

    threshold = float(
        np.median(
            flat[1:]
        )
    )

    bits = flat > threshold

    value = 0

    for bit in bits:
        value = (
            value << 1
        ) | int(bool(bit))

    return int(value)


def hamming(a, b):
    return (
        int(a) ^ int(b)
    ).bit_count()


def visually_duplicate(
    a,
    b,
    hash_threshold,
):
    distances = []

    for camera in CAMERAS:
        ha = a["hashes"].get(camera)
        hb = b["hashes"].get(camera)

        if (
            ha is None
            or hb is None
        ):
            continue

        distances.append(
            hamming(ha, hb)
        )

    if not distances:
        return False

    # Consider a synchronized group duplicate when at least
    # two camera views are perceptually almost identical.
    close_count = sum(
        d < hash_threshold
        for d in distances
    )

    return close_count >= 2


def accept_candidate(
    item,
    selected,
    min_time_gap,
    hash_threshold,
):
    ts = float(
        item["record"].get(
            "timestamp",
            0.0,
        )
        or 0.0
    )

    for old in selected:
        old_ts = float(
            old["record"].get(
                "timestamp",
                0.0,
            )
            or 0.0
        )

        if abs(
            ts - old_ts
        ) < min_time_gap:
            return False

        if visually_duplicate(
            item,
            old,
            hash_threshold,
        ):
            return False

    return True


def select_phase(
    candidates,
    quota,
    normal_fraction,
    min_time_gap,
    hash_threshold,
):
    if quota <= 0:
        return []

    hard = [
        x
        for x in candidates
        if x["category"] == "hard"
    ]

    stable = [
        x
        for x in candidates
        if x["category"] == "stable"
    ]

    hard.sort(
        key=lambda x: (
            x["score"],
            float(
                x["record"].get(
                    "timestamp",
                    0.0,
                )
                or 0.0
            ),
        ),
        reverse=True,
    )

    stable.sort(
        key=lambda x: float(
            x["record"].get(
                "timestamp",
                0.0,
            )
            or 0.0
        )
    )

    stable_target = int(
        round(
            quota
            * normal_fraction
        )
    )

    hard_target = max(
        0,
        quota - stable_target,
    )

    selected = []

    # Pass 1: high-value hard samples.
    for item in hard:
        if len(selected) >= hard_target:
            break

        if accept_candidate(
            item,
            selected,
            min_time_gap,
            hash_threshold,
        ):
            selected.append(item)

    # Pass 2: stable samples spread across time.
    if stable_target > 0 and stable:
        n_probe = min(
            len(stable),
            max(
                stable_target * 4,
                stable_target,
            ),
        )

        indices = np.linspace(
            0,
            len(stable) - 1,
            n_probe,
        )

        for idx in indices:
            if len(selected) >= quota:
                break

            item = stable[
                int(round(idx))
            ]

            if accept_candidate(
                item,
                selected,
                min_time_gap,
                hash_threshold,
            ):
                selected.append(item)

    # Pass 3: relax pHash slightly if quota not reached.
    pool = sorted(
        candidates,
        key=lambda x: x["score"],
        reverse=True,
    )

    for relaxed_hash in (
        max(4, hash_threshold - 2),
        3,
        1,
    ):
        if len(selected) >= quota:
            break

        for item in pool:
            if len(selected) >= quota:
                break

            if item in selected:
                continue

            if accept_candidate(
                item,
                selected,
                min_time_gap,
                relaxed_hash,
            ):
                selected.append(item)

    # Pass 4: only temporal separation.
    if len(selected) < quota:
        for item in pool:
            if len(selected) >= quota:
                break

            if item in selected:
                continue

            ts = float(
                item["record"].get(
                    "timestamp",
                    0.0,
                )
                or 0.0
            )

            if any(
                abs(
                    ts
                    -
                    float(
                        old["record"].get(
                            "timestamp",
                            0.0,
                        )
                        or 0.0
                    )
                )
                < min_time_gap
                for old in selected
            ):
                continue

            selected.append(item)

    return selected[:quota]


def make_triplet(
    paths,
    sample_id,
    phase,
    score,
):
    images = []

    for camera in CAMERAS:
        image = cv2.imread(
            str(paths[camera])
        )

        if image is None:
            return None

        h, w = image.shape[:2]

        target_w = 420
        scale = target_w / max(w, 1)
        target_h = int(
            round(h * scale)
        )

        image = cv2.resize(
            image,
            (
                target_w,
                target_h,
            ),
        )

        cv2.rectangle(
            image,
            (0, 0),
            (target_w, 35),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            image,
            camera,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        images.append(image)

    h = min(
        x.shape[0]
        for x in images
    )

    resized = [
        cv2.resize(
            x,
            (
                x.shape[1],
                h,
            ),
        )
        for x in images
    ]

    triplet = np.hstack(resized)

    header = np.zeros(
        (
            45,
            triplet.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    text = (
        f"{sample_id} | {phase} | score={score:.1f}"
    )

    cv2.putText(
        header,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return np.vstack(
        [
            header,
            triplet,
        ]
    )


def build_contact_sheets(
    selected_rows,
    out_dir,
):
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_page = 5

    for page_start in range(
        0,
        len(selected_rows),
        per_page,
    ):
        page_rows = selected_rows[
            page_start:
            page_start + per_page
        ]

        strips = []

        for row in page_rows:
            strip = make_triplet(
                row["output_paths"],
                row["sample_id"],
                row["phase"],
                row["score"],
            )

            if strip is not None:
                strips.append(strip)

        if not strips:
            continue

        max_w = max(
            x.shape[1]
            for x in strips
        )

        padded = []

        for image in strips:
            if image.shape[1] < max_w:
                pad = np.zeros(
                    (
                        image.shape[0],
                        max_w - image.shape[1],
                        3,
                    ),
                    dtype=np.uint8,
                )

                image = np.hstack(
                    [
                        image,
                        pad,
                    ]
                )

            padded.append(image)

        page = np.vstack(padded)

        page_id = (
            page_start // per_page
            + 1
        )

        cv2.imwrite(
            str(
                out_dir
                /
                f"sync35_page_{page_id:02d}.jpg"
            ),
            page,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "session",
        type=Path,
    )

    parser.add_argument(
        "output_root",
        type=Path,
    )

    parser.add_argument(
        "--total-groups",
        type=int,
        default=35,
    )

    parser.add_argument(
        "--wait-groups",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--normal-fraction",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--min-time-gap",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--hash-threshold",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    session = args.session.resolve()
    output_root = args.output_root.resolve()

    records_path = (
        session
        /
        "records.jsonl"
    )

    if not records_path.is_file():
        raise SystemExit(
            f"records.jsonl missing: {records_path}"
        )

    insertion_groups = (
        args.total_groups
        -
        args.wait_groups
    )

    raw = []

    with records_path.open(
        "r",
        encoding="utf-8",
    ) as fp:
        for line in fp:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            paths = {}

            valid_group = True

            for camera in CAMERAS:
                p = image_path(
                    session,
                    record,
                    camera,
                )

                if (
                    p is None
                    or
                    not p.is_file()
                ):
                    valid_group = False
                    break

                paths[camera] = p

            if not valid_group:
                continue

            info = score_record(record)

            raw.append(
                {
                    "record": record,
                    "paths": paths,
                    "score": info["score"],
                    "category": info["category"],
                    "reasons": info["reasons"],
                    "hashes": {
                        camera:
                            phash64(
                                paths[camera]
                            )
                        for camera
                        in CAMERAS
                    },
                }
            )

    wait_candidates = [
        x
        for x in raw
        if x["record"].get("phase")
        ==
        "WAIT_HOLE_TARGET"
    ]

    insertion_candidates = [
        x
        for x in raw
        if x["record"].get("phase")
        ==
        "INSERTION_TRACKING"
    ]

    selected_wait = select_phase(
        wait_candidates,
        args.wait_groups,
        args.normal_fraction,
        args.min_time_gap,
        args.hash_threshold,
    )

    selected_insertion = select_phase(
        insertion_candidates,
        insertion_groups,
        args.normal_fraction,
        args.min_time_gap,
        args.hash_threshold,
    )

    selected = (
        selected_wait
        +
        selected_insertion
    )

    # If one phase did not have enough groups, fill globally.
    if len(selected) < args.total_groups:
        remaining = sorted(
            raw,
            key=lambda x: x["score"],
            reverse=True,
        )

        for item in remaining:
            if len(selected) >= args.total_groups:
                break

            if item in selected:
                continue

            if accept_candidate(
                item,
                selected,
                args.min_time_gap,
                2,
            ):
                selected.append(item)

    selected = selected[
        :args.total_groups
    ]

    selected.sort(
        key=lambda x: float(
            x["record"].get(
                "timestamp",
                0.0,
            )
            or 0.0
        )
    )

    session_name = (
        "session_20261110_152101_sync35"
    )

    out_session = (
        output_root
        /
        "images_raw"
        /
        session_name
    )

    for camera in CAMERAS:
        (
            out_session
            /
            camera
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    (
        output_root
        /
        "selection"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    contact_dir = (
        output_root
        /
        "selection"
        /
        "contact_sheets"
    )

    manifest_rows = []
    output_rows = []

    selected_jsonl = (
        output_root
        /
        "selection"
        /
        "records_selected.jsonl"
    )

    with selected_jsonl.open(
        "w",
        encoding="utf-8",
    ) as fp_json:

        for index, item in enumerate(
            selected,
            1,
        ):
            sid = f"{index:06d}"

            output_paths = {}

            for camera in CAMERAS:
                src = item["paths"][camera]

                dst = (
                    out_session
                    /
                    camera
                    /
                    f"{sid}.png"
                )

                image = cv2.imread(
                    str(src),
                    cv2.IMREAD_COLOR,
                )

                if image is None:
                    raise RuntimeError(
                        f"Cannot read {src}"
                    )

                ok = cv2.imwrite(
                    str(dst),
                    image,
                    [
                        cv2.IMWRITE_PNG_COMPRESSION,
                        3,
                    ],
                )

                if not ok:
                    raise RuntimeError(
                        f"Cannot write {dst}"
                    )

                output_paths[
                    camera
                ] = dst

            record = dict(
                item["record"]
            )

            record[
                "sync_sample_id"
            ] = sid

            record[
                "selection_score"
            ] = item["score"]

            record[
                "selection_category"
            ] = item["category"]

            record[
                "selection_reasons"
            ] = item["reasons"]

            fp_json.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                +
                "\n"
            )

            row = {
                "sample_id":
                    sid,

                "source_frame":
                    item["record"].get(
                        "frame_counter"
                    ),

                "timestamp":
                    item["record"].get(
                        "timestamp"
                    ),

                "phase":
                    item["record"].get(
                        "phase"
                    ),

                "score":
                    item["score"],

                "category":
                    item["category"],

                "reasons":
                    ";".join(
                        item["reasons"]
                    ),

                "external_free":
                    source_of(
                        item["record"],
                        "external",
                    ),

                "right_wrist_free":
                    source_of(
                        item["record"],
                        "right_wrist",
                    ),

                "left_wrist_free":
                    source_of(
                        item["record"],
                        "left_wrist",
                    ),

                "external_target":
                    (
                        f"{int(selected_of(item['record'], 'external'))}"
                        "/"
                        f"{int(valid_of(item['record'], 'external'))}"
                    ),

                "right_wrist_target":
                    (
                        f"{int(selected_of(item['record'], 'right_wrist'))}"
                        "/"
                        f"{int(valid_of(item['record'], 'right_wrist'))}"
                    ),

                "left_wrist_target":
                    (
                        f"{int(selected_of(item['record'], 'left_wrist'))}"
                        "/"
                        f"{int(valid_of(item['record'], 'left_wrist'))}"
                    ),
            }

            manifest_rows.append(
                row
            )

            output_rows.append(
                {
                    **row,
                    "output_paths":
                        output_paths,
                }
            )

    manifest = (
        output_root
        /
        "selection"
        /
        "sync35_manifest.csv"
    )

    fields = list(
        manifest_rows[0].keys()
    )

    with manifest.open(
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
            manifest_rows
        )

    build_contact_sheets(
        output_rows,
        contact_dir,
    )

    phase_counts = {}

    category_counts = {}

    for row in manifest_rows:
        phase_counts[
            row["phase"]
        ] = (
            phase_counts.get(
                row["phase"],
                0,
            )
            +
            1
        )

        category_counts[
            row["category"]
        ] = (
            category_counts.get(
                row["category"],
                0,
            )
            +
            1
        )

    summary = {
        "source_session":
            str(session),

        "output_root":
            str(output_root),

        "output_session":
            str(out_session),

        "candidate_groups":
            len(raw),

        "selected_groups":
            len(selected),

        "selected_images":
            len(selected)
            *
            len(CAMERAS),

        "phase_counts":
            phase_counts,

        "category_counts":
            category_counts,

        "normal_fraction":
            args.normal_fraction,

        "min_time_gap":
            args.min_time_gap,

        "hash_threshold":
            args.hash_threshold,
    }

    summary_path = (
        output_root
        /
        "selection"
        /
        "selection_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "SYNC_GROUP_SELECTION_OK"
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
