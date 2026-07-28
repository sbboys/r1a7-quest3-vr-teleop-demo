#!/usr/bin/env python3
"""Query or set real robot arm_sdk status through Unitree sport/loco RPC."""

from __future__ import annotations

import argparse
import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.h2.loco.h2_loco_client import LocoClient


def main() -> int:
    parser = argparse.ArgumentParser(description="R1-A/H2-style arm_sdk status helper")
    parser.add_argument("--interface", default="enx9c69d37d0967")
    parser.add_argument("--domain_id", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--enable", action="store_true", help="set arm_sdk status true")
    parser.add_argument("--disable", action="store_true", help="set arm_sdk status false")
    parser.add_argument("--stand_up", action="store_true", help="call StandUp before enabling arm_sdk")
    args = parser.parse_args()

    if args.enable and args.disable:
        print("[R1-A7 ARM SDK] choose only one of --enable or --disable")
        return 2

    ChannelFactoryInitialize(args.domain_id, args.interface)
    client = LocoClient()
    client.SetTimeout(args.timeout)
    client.Init()

    code, fsm_id = client.GetFsmId()
    print(f"[R1-A7 ARM SDK] fsm_id: code={code} value={fsm_id}")
    code, fsm_mode = client.GetFsmMode()
    print(f"[R1-A7 ARM SDK] fsm_mode: code={code} value={fsm_mode}")
    code, arm_status = client.GetArmSdkStatus()
    print(f"[R1-A7 ARM SDK] arm_sdk_status: code={code} value={arm_status}")

    if args.stand_up:
        print("[R1-A7 ARM SDK] calling StandUp()")
        code = client.StandUp()
        print(f"[R1-A7 ARM SDK] StandUp code={code}")
        time.sleep(1.0)

    if args.enable:
        print("[R1-A7 ARM SDK] calling SetArmSdkStatus(True)")
        code = client.SetArmSdkStatus(True)
        print(f"[R1-A7 ARM SDK] enable code={code}")
        time.sleep(0.2)
    elif args.disable:
        print("[R1-A7 ARM SDK] calling SetArmSdkStatus(False)")
        code = client.SetArmSdkStatus(False)
        print(f"[R1-A7 ARM SDK] disable code={code}")
        time.sleep(0.2)

    code, arm_status = client.GetArmSdkStatus()
    print(f"[R1-A7 ARM SDK] arm_sdk_status_after: code={code} value={arm_status}")
    return 0 if code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
