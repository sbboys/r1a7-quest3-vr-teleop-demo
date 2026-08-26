#!/usr/bin/env python3

import argparse
import sqlite3
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Analyze RealSense/ROS2 DB3 recording"
    )
    parser.add_argument("db3", help="Path to .db3 file")
    args = parser.parse_args()

    path = Path(args.db3).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(path)

    print("=" * 80)
    print("D435i DB3 Analysis")
    print("=" * 80)
    print("File :", path)
    print("Size :", f"{path.stat().st_size / 1024 / 1024:.2f} MB")

    con = sqlite3.connect(str(path))
    cur = con.cursor()

    print("\n[1] SQLite integrity")
    result = cur.execute(
        "PRAGMA integrity_check;"
    ).fetchone()

    print("Integrity:", result[0])

    tables = {
        row[0]
        for row in cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table';"
        )
    }

    print("\n[2] Tables")
    for table in sorted(tables):
        print(" ", table)

    if "topics" not in tables or "messages" not in tables:
        print("\nNo standard rosbag2 topics/messages tables found.")
        con.close()
        return

    print("\n[3] Topics")

    topics = cur.execute(
        """
        SELECT
            id,
            name,
            type,
            serialization_format
        FROM topics
        ORDER BY id
        """
    ).fetchall()

    for topic_id, name, msg_type, serialization in topics:
        print()
        print(f"Topic ID       : {topic_id}")
        print(f"Topic          : {name}")
        print(f"Type           : {msg_type}")
        print(f"Serialization  : {serialization}")

    print("\n[4] Message statistics")
    print("-" * 80)

    rows = cur.execute(
        """
        SELECT
            topics.id,
            topics.name,
            topics.type,
            COUNT(messages.id),
            MIN(messages.timestamp),
            MAX(messages.timestamp)
        FROM topics
        LEFT JOIN messages
            ON messages.topic_id = topics.id
        GROUP BY topics.id
        ORDER BY topics.id
        """
    ).fetchall()

    global_first = None
    global_last = None

    for (
        topic_id,
        topic_name,
        topic_type,
        count,
        first_ns,
        last_ns,
    ) in rows:

        print()
        print(f"Topic : {topic_name}")
        print(f"Type  : {topic_type}")
        print(f"Count : {count}")

        if first_ns is not None and last_ns is not None:
            duration_s = (last_ns - first_ns) / 1e9

            print(f"First timestamp : {first_ns}")
            print(f"Last timestamp  : {last_ns}")
            print(f"Topic duration  : {duration_s:.3f} s")

            if duration_s > 0 and count > 1:
                approx_rate = (count - 1) / duration_s
                print(
                    f"Approx rate     : "
                    f"{approx_rate:.2f} Hz"
                )

            global_first = (
                first_ns
                if global_first is None
                else min(global_first, first_ns)
            )

            global_last = (
                last_ns
                if global_last is None
                else max(global_last, last_ns)
            )

    if global_first is not None and global_last is not None:
        duration = (global_last - global_first) / 1e9

        print("\n" + "=" * 80)
        print("Recording summary")
        print("=" * 80)

        print(f"Start timestamp : {global_first}")
        print(f"End timestamp   : {global_last}")
        print(f"Duration        : {duration:.3f} s")

        total_messages = cur.execute(
            "SELECT COUNT(*) FROM messages"
        ).fetchone()[0]

        print(f"Total messages  : {total_messages}")

    con.close()


if __name__ == "__main__":
    main()
