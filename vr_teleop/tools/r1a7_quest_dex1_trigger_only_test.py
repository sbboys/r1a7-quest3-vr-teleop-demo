#!/usr/bin/env python3
"""Quest 3 controller-trigger test for the real Unitree Dex1-1 grippers.

This script does NOT create a LowCmd publisher and does NOT command any arm,
waist, or leg joint. It only forwards the Quest left/right index-trigger values
to the matching Dex1-1 gripper controller from xr_teleoperate.

TeleVuer controller convention:
    trigger released     -> value about 10 -> gripper opens
    trigger fully pulled -> value about 0  -> gripper closes
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from multiprocessing import Array, Lock, Value
from pathlib import Path

import numpy as np

XR_TELEOP = Path(os.getenv("XR_TELEOP_ROOT", "/home/robot/xr_teleoperate"))
XR_TELEOP_TELEOP = XR_TELEOP / "teleop"
XR_TELEOP_TV_SRC = XR_TELEOP_TELEOP / "televuer" / "src"
for path in (XR_TELEOP_TV_SRC, XR_TELEOP_TELEOP, XR_TELEOP):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from televuer import TeleVuerWrapper  # noqa: E402
from robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller  # noqa: E402
from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: E402


def port_is_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def cleanup_port_8012() -> None:
    try:
        result = subprocess.run(
            ["lsof", "-t", "-iTCP:8012", "-sTCP:LISTEN"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print("[DEX1 TEST] lsof is not installed; skipping port cleanup")
        return

    pids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not pids:
        return

    print("[DEX1 TEST] stopping old Vuer listener(s):", ", ".join(pids))
    for pid in pids:
        subprocess.run(["kill", "-TERM", pid], check=False)

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if not port_is_open(8012):
            return
        time.sleep(0.1)

    for pid in pids:
        subprocess.run(["kill", "-KILL", pid], check=False)
    time.sleep(0.3)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quest trigger-only control test for real Dex1-1 grippers")
    parser.add_argument("--interface", default="enx9c69d37d0967")
    parser.add_argument("--domain_id", type=int, default=0)
    parser.add_argument("--host_ip", default=os.getenv("HOST_IP", "192.168.1.127"))
    parser.add_argument("--gripper_fps", type=float, default=200.0)
    parser.add_argument("--print_period", type=float, default=0.25)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--disable_gripper_filter", action="store_true")
    parser.add_argument("--assume_yes", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cleanup_port_8012()

    print("WARNING: This test commands both real Dex1-1 grippers.")
    print("It does NOT publish rt/lowcmd and does NOT move the robot arms.")
    print("Released triggers command OPEN; pulled triggers command CLOSE.")
    if not args.assume_yes:
        answer = input("Type ENABLE_GRIPPER to continue: ").strip()
        if answer != "ENABLE_GRIPPER":
            print("[DEX1 TEST] aborted")
            return 2

    ChannelFactoryInitialize(args.domain_id, args.interface)

    old_cwd = Path.cwd()
    os.chdir(XR_TELEOP_TELEOP)
    try:
        tv = TeleVuerWrapper(
            use_hand_tracking=False,
            binocular=False,
            img_shape=(480, 640),
            display_mode="pass-through",
            zmq=False,
            webrtc=False,
            arm_reference_mode="head_yaw",
        )
    finally:
        os.chdir(old_cwd)

    left_value = Value("d", 10.0, lock=True)
    right_value = Value("d", 10.0, lock=True)
    xr_ready = Value("b", False, lock=True)
    data_lock = Lock()
    state_array = Array("d", 2, lock=False)
    action_array = Array("d", 2, lock=False)

    controller_holder: dict[str, object] = {}
    init_error: list[str] = []

    def controller_worker() -> None:
        try:
            controller_holder["controller"] = Dex1_1_Gripper_Controller(
                left_value,
                right_value,
                data_lock,
                state_array,
                action_array,
                filter=not args.disable_gripper_filter,
                fps=args.gripper_fps,
                simulation_mode=False,
                xr_motion_data_ready_in=xr_ready,
            )
            print("[DEX1 TEST] Dex1 DDS ready")
        except Exception as exc:
            init_error.append(f"{type(exc).__name__}: {exc}")
            print("[DEX1 TEST] controller init error:", init_error[-1])

    threading.Thread(target=controller_worker, daemon=True).start()

    done = False

    def stop_handler(_signum, _frame) -> None:
        nonlocal done
        done = True

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    print("[DEX1 TEST] waiting for rt/dex1/left/state and rt/dex1/right/state")
    print("[DEX1 TEST] open Quest URL:")
    print(f"https://{args.host_ip}:8012/?ws=wss://{args.host_ip}:8012")
    print("[DEX1 TEST] enter VR, then use each index trigger slowly")

    deadline = time.monotonic() + max(0.0, args.duration)
    last_print = 0.0

    try:
        while not done:
            now = time.monotonic()
            if args.duration > 0 and now >= deadline:
                break

            tele = tv.get_tele_data()
            motion_ready = bool(getattr(tele, "motion_data_ready", False))
            fields_ok = hasattr(tele, "left_ctrl_triggerValue") and hasattr(tele, "right_ctrl_triggerValue")
            active = motion_ready and fields_ok and not init_error

            if active:
                left_trigger = float(np.clip(float(tele.left_ctrl_triggerValue), 0.0, 10.0))
                right_trigger = float(np.clip(float(tele.right_ctrl_triggerValue), 0.0, 10.0))
                with left_value.get_lock():
                    left_value.value = left_trigger
                with right_value.get_lock():
                    right_value.value = right_trigger
            else:
                left_trigger = float("nan")
                right_trigger = float("nan")

            with xr_ready.get_lock():
                xr_ready.value = bool(active)

            if now - last_print >= args.print_period:
                connected = "controller" in controller_holder
                with data_lock:
                    states = list(state_array[:])
                    actions = list(action_array[:])
                print(
                    "[DEX1 TEST] "
                    f"connected={connected} motion_ready={motion_ready} active={active} "
                    f"trigger_L={left_trigger:.3f} trigger_R={right_trigger:.3f} "
                    f"state_L={states[0]:.3f} state_R={states[1]:.3f} "
                    f"cmd_L={actions[0]:.3f} cmd_R={actions[1]:.3f}"
                )
                if motion_ready and not fields_ok:
                    print("  TeleVuer has no controller trigger fields; update the televuer submodule")
                last_print = now

            time.sleep(0.02)
    finally:
        with xr_ready.get_lock():
            xr_ready.value = False
        controller = controller_holder.get("controller")
        if controller is not None and hasattr(controller, "running"):
            controller.running = False
        try:
            tv.close()
        except Exception:
            pass
        print("[DEX1 TEST] stopped; no more gripper commands will be published")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
