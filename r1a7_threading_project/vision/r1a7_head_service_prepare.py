#!/usr/bin/env python3
"""Prepare the verified R1-A7 VideoClient service without touching App video."""

from __future__ import annotations

import argparse
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient


RUNNING = 0
STOPPED = 1


def service_states(client: RobotStateClient) -> dict[str, int]:
    code, services = client.ServiceList()
    if code != 0 or services is None:
        raise RuntimeError(f"ServiceList failed: code={code}")
    return {service.name: service.status for service in services}


def wait_for_state(
    client: RobotStateClient,
    name: str,
    expected: int,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = service_states(client).get(name)
        if status == expected:
            return
        time.sleep(0.5)
    raise RuntimeError(
        f"service {name} did not reach status={expected} within {timeout:.1f}s"
    )


def set_service(
    client: RobotStateClient,
    name: str,
    enabled: bool,
    timeout: float,
) -> None:
    expected = RUNNING if enabled else STOPPED
    current = service_states(client).get(name)
    if current is None:
        raise RuntimeError(f"robot service not found: {name}")
    if current == expected:
        return
    code = client.ServiceSwitch(name, enabled)
    if code != 0:
        raise RuntimeError(f"ServiceSwitch({name}, {enabled}) failed: code={code}")
    wait_for_state(client, name, expected, timeout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare R1-A7 head-camera services")
    parser.add_argument("--interface", default="enp6s0")
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    ChannelFactoryInitialize(0, args.interface)
    client = RobotStateClient()
    client.SetTimeout(5.0)
    client.Init()

    before = service_states(client)
    print(
        "[HEAD SERVICE] before "
        f"video_hub={before.get('video_hub')} "
        f"videohub_rtsp_server={before.get('videohub_rtsp_server')}",
        flush=True,
    )

    # The September GitHub baseline received 1280x720 RGB through VideoClient
    # while the robot-managed App services were left alone.  Do not disable the
    # RTSP/WebRTC path here: doing so makes the phone App lose its preview and is
    # not part of the previously verified acquisition procedure.
    set_service(client, "video_hub", True, args.timeout)

    after = service_states(client)
    print(
        "[HEAD SERVICE READY] "
        f"video_hub={after.get('video_hub')} "
        f"videohub_rtsp_server={after.get('videohub_rtsp_server')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
