#!/usr/bin/env python3
"""Print selected R1-A7 lowstate motor fields without publishing commands."""

from __future__ import annotations

import argparse
import time
from typing import Iterable

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


def _get(obj, name: str, default="n/a"):
    return getattr(obj, name, default)


def _parse_indices(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def print_motors(motor_state: Iterable[object], indices: list[int]) -> None:
    motors = list(motor_state)
    print(f"motor_count={len(motors)}")
    for idx in indices:
        if idx >= len(motors):
            print(f"{idx}: missing")
            continue
        motor = motors[idx]
        print(
            f"{idx}: "
            f"q={float(_get(motor, 'q', 0.0)):.6f} "
            f"dq={float(_get(motor, 'dq', 0.0)):.6f} "
            f"tau={float(_get(motor, 'tau_est', 0.0)):.6f} "
            f"mode={_get(motor, 'mode')} "
            f"merror={_get(motor, 'merror')} "
            f"temperature={_get(motor, 'temperature')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", default="enx9c69d37d0967")
    parser.add_argument("--domain_id", type=int, default=0)
    parser.add_argument("--state_topic", default="rt/lowstate")
    parser.add_argument("--indices", default="29,30,31,32,33,34")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--period", type=float, default=0.5)
    args = parser.parse_args()

    ChannelFactoryInitialize(args.domain_id, args.interface)
    sub = ChannelSubscriber(args.state_topic, LowState_)
    sub.Init()

    indices = _parse_indices(args.indices)
    for sample in range(args.samples):
        state = sub.Read(1.0)
        if state is None:
            print(f"sample {sample}: lowstate timeout")
            continue
        print(f"sample {sample}:")
        print_motors(state.motor_state, indices)
        time.sleep(args.period)


if __name__ == "__main__":
    main()
