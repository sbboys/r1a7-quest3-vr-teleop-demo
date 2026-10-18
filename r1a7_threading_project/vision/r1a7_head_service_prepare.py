#!/usr/bin/env python3
"""Switch R1-A head video services for the official Stereo patch PC1 stream."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient


RUNNING = 0
STOPPED = 1
STATE_PATH = Path("/tmp/r1a7_head_service_state.json")
RECEIVER_IP = "192.168.123.164"


def service_states(client: RobotStateClient) -> dict[str, int]:
    code, services = client.ServiceList()
    if code != 0 or services is None:
        raise RuntimeError(f"ServiceList failed: code={code}")
    return {service.name: service.status for service in services}


def wait_for_state(
    client: RobotStateClient, name: str, expected: int, timeout: float
) -> None:
    deadline = time.monotonic() + timeout
    last_status: int | None = None
    while time.monotonic() < deadline:
        try:
            last_status = service_states(client).get(name)
        except RuntimeError:
            time.sleep(0.5)
            continue
        if last_status == expected:
            return
        time.sleep(0.5)
    raise RuntimeError(
        f"service {name} did not reach status={expected} "
        f"(last={last_status}) within {timeout:.1f}s"
    )


def set_service(
    client: RobotStateClient, name: str, enabled: bool, timeout: float
) -> None:
    expected = RUNNING if enabled else STOPPED
    current = service_states(client).get(name)
    if current is None:
        raise RuntimeError(f"robot service not found: {name}")
    if current == expected:
        return
    code = client.ServiceSwitch(name, enabled)
    # The RPC can time out while an asynchronous transition still completes.
    # Accept it only when the requested state is actually observed afterward.
    try:
        wait_for_state(client, name, expected, timeout)
    except RuntimeError as exc:
        raise RuntimeError(
            f"ServiceSwitch({name}, {enabled}) failed: code={code}; {exc}"
        ) from exc
    if code != 0:
        reason = "（RPC 调用超时）" if code == 3104 else ""
        print(
            f"[HEAD SERVICE WARNING] ServiceSwitch({name}, {enabled}) "
            f"returned code={code}{reason}, but the requested state was reached",
            flush=True,
        )


def interface_addresses(interface: str) -> list[str]:
    result = subprocess.run(
        ["ip", "-4", "-o", "addr", "show", "dev", interface],
        check=False,
        capture_output=True,
        text=True,
        timeout=3.0,
    )
    addresses: list[str] = []
    for line in result.stdout.splitlines():
        fields = line.split()
        for index, field in enumerate(fields[:-1]):
            if field == "inet":
                addresses.append(fields[index + 1].split("/", 1)[0])
    return addresses


def save_original_state(states: dict[str, int], interface: str) -> None:
    if STATE_PATH.exists():
        try:
            saved = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = {}
        if (
            saved.get("video_hub") in (RUNNING, STOPPED)
            and saved.get("stereo_patch_pc1") in (RUNNING, STOPPED)
        ):
            print(
                "[HEAD SERVICE STATE] retaining the original snapshot "
                f"from {saved.get('saved_at', 'unknown')}",
                flush=True,
            )
            return

    payload = {
        "interface": interface,
        "video_hub": states.get("video_hub"),
        "stereo_patch_pc1": states.get("stereo_patch_pc1"),
        "saved_at": time.time(),
    }
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(STATE_PATH)


def restore_original_state(client: RobotStateClient, timeout: float) -> None:
    if not STATE_PATH.exists():
        print("[HEAD SERVICE RESTORE] no saved state", flush=True)
        return
    saved = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    errors: list[str] = []
    desired_stereo = saved.get("stereo_patch_pc1") == RUNNING
    desired_hub = saved.get("video_hub") == RUNNING
    # Release the UDP streamer before restoring video_hub.
    for name, enabled in (
        ("stereo_patch_pc1", desired_stereo),
        ("video_hub", desired_hub),
    ):
        try:
            set_service(client, name, enabled, timeout)
        except RuntimeError as exc:
            errors.append(str(exc))
    if errors:
        raise RuntimeError("; ".join(errors))
    STATE_PATH.unlink(missing_ok=True)
    states = service_states(client)
    print(
        "[HEAD SERVICE RESTORED] "
        f"video_hub={states.get('video_hub')} "
        f"stereo_patch_pc1={states.get('stereo_patch_pc1')}",
        flush=True,
    )


def prepare_official_stream(
    client: RobotStateClient, interface: str, timeout: float
) -> None:
    addresses = interface_addresses(interface)
    print(
        "[HEAD NETWORK] "
        f"interface={interface} ipv4={','.join(addresses) or 'none'} "
        f"official_receiver_ip={'ready' if RECEIVER_IP in addresses else 'missing'}",
        flush=True,
    )
    if RECEIVER_IP not in addresses:
        raise RuntimeError(
            f"official R1-A stream requires {RECEIVER_IP} on {interface}; "
            f"found {addresses or 'no IPv4 address'}"
        )

    before = service_states(client)
    print(
        "[HEAD SERVICE] before "
        f"video_hub={before.get('video_hub')} "
        f"stereo_patch_pc1={before.get('stereo_patch_pc1')} "
        f"videohub_rtsp_server={before.get('videohub_rtsp_server')}",
        flush=True,
    )
    save_original_state(before, interface)
    try:
        # Required order from Unitree's official R1-A video-service page.
        set_service(client, "video_hub", False, timeout)
        set_service(client, "stereo_patch_pc1", True, timeout)
    except RuntimeError:
        try:
            restore_original_state(client, timeout)
        except RuntimeError as restore_exc:
            print(f"[HEAD SERVICE RESTORE ERROR] {restore_exc}", flush=True)
        raise

    after = service_states(client)
    print(
        "[HEAD SERVICE READY] "
        f"video_hub={after.get('video_hub')} "
        f"stereo_patch_pc1={after.get('stereo_patch_pc1')} "
        "udp_ports=5001,5002,5003",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare R1-A official head stream")
    parser.add_argument("--interface", default="enp6s0")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--mode", choices=("start", "restore"), default="start")
    args = parser.parse_args()

    ChannelFactoryInitialize(0, args.interface)
    client = RobotStateClient()
    client.SetTimeout(5.0)
    client.Init()

    if args.mode == "restore":
        restore_original_state(client, args.timeout)
    else:
        prepare_official_stream(client, args.interface, args.timeout)


if __name__ == "__main__":
    main()
