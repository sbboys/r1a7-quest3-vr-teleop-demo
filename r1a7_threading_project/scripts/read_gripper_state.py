#!/usr/bin/env python3
"""Read R1-A7 DEX1 gripper q/dq from rt/lowstate without publishing commands."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


class LowStateReader:
    def __init__(self, left_index: int, right_index: int):
        self.left_index = left_index
        self.right_index = right_index
        self.msg: Optional[LowState_] = None
        self.received_at: Optional[float] = None

    def callback(self, msg: LowState_) -> None:
        self.msg = msg
        self.received_at = time.time()

    def snapshot(self) -> dict:
        if self.msg is None or self.received_at is None:
            raise RuntimeError("no lowstate received")
        max_index = max(self.left_index, self.right_index)
        if len(self.msg.motor_state) <= max_index:
            raise RuntimeError(f"lowstate has {len(self.msg.motor_state)} motors, requested {max_index}")
        left = self.msg.motor_state[self.left_index]
        right = self.msg.motor_state[self.right_index]
        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "received_at_unix": self.received_at,
            "state_topic": None,
            "left_index": self.left_index,
            "right_index": self.right_index,
            "left_q": float(left.q),
            "left_dq": float(left.dq),
            "right_q": float(right.q),
            "right_dq": float(right.dq),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read R1-A7 gripper state without publishing commands")
    parser.add_argument("--interface", default="enp6s0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--state-topic", default="rt/lowstate")
    parser.add_argument("--left-index", type=int, default=31)
    parser.add_argument("--right-index", type=int, default=33)
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    ChannelFactoryInitialize(args.domain_id, args.interface)
    reader = LowStateReader(args.left_index, args.right_index)
    sub = ChannelSubscriber(args.state_topic, LowState_)
    sub.Init(reader.callback, 10)

    deadline = time.time() + args.timeout_s
    while reader.msg is None and time.time() < deadline:
        time.sleep(0.02)
    payload = reader.snapshot()
    payload["state_topic"] = args.state_topic
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
