#!/usr/bin/env python3
"""Check SteamVR/OpenVR controller visibility for R1-A7 teleoperation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


STEAMVR_ROOT = Path.home() / ".local/share/Steam/steamapps/common/SteamVR"


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except Exception as exc:
        return str(exc)


def _print_system_probe() -> None:
    print("[VR CHECK] USB devices")
    usb = _run(["lsusb"])
    for line in usb.splitlines():
        low = line.lower()
        if any(key in low for key in ("vive", "valve", "watchman", "lighthouse", "htc")):
            print("  " + line)

    print("[VR CHECK] SteamVR processes")
    ps = _run(["ps", "-eo", "pid,comm,args"])
    found = False
    for line in ps.splitlines():
        low = line.lower()
        if any(key in low for key in ("vrserver", "vrmonitor", "vrcompositor")):
            print("  " + line)
            found = True
    if not found:
        print("  no SteamVR runtime process found")


def _check_openvr(samples: int) -> int:
    try:
        import openvr  # type: ignore
    except Exception as exc:
        print("[VR CHECK] Python openvr module is not installed:", exc)
        print("[VR CHECK] Install in isaaclab env: python -m pip install openvr")
        return 4

    vr = None
    try:
        vr = openvr.init(openvr.VRApplication_Other)
        print("[VR CHECK] OpenVR initialized")
        for _ in range(max(1, samples)):
            poses = vr.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding,
                0.0,
                openvr.k_unMaxTrackedDeviceCount,
            )
            controllers = 0
            for idx, pose in enumerate(poses):
                device_class = vr.getTrackedDeviceClass(idx)
                if device_class != openvr.TrackedDeviceClass_Controller:
                    continue
                controllers += 1
                role = vr.getControllerRoleForTrackedDeviceIndex(idx)
                role_name = {
                    openvr.TrackedControllerRole_LeftHand: "left",
                    openvr.TrackedControllerRole_RightHand: "right",
                }.get(role, f"role_{role}")
                connected = vr.isTrackedDeviceConnected(idx)
                valid = bool(pose.bPoseIsValid)
                matrix = pose.mDeviceToAbsoluteTracking
                pos = (matrix[0][3], matrix[1][3], matrix[2][3])
                state = vr.getControllerState(idx)[1]
                print(
                    "[VR CHECK] controller"
                    f" index={idx} role={role_name}"
                    f" connected={connected} pose_valid={valid}"
                    f" pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})"
                    f" buttons=0x{state.ulButtonPressed:x}"
                    f" trigger={state.rAxis[1].x:.3f}"
                )
            if controllers == 0:
                print("[VR CHECK] no tracked controller reported by OpenVR")
                return 2
        return 0
    except Exception as exc:
        print("[VR CHECK] OpenVR runtime check failed:", exc)
        print("[VR CHECK] Start SteamVR, wake/pair both controllers, then retry.")
        return 3
    finally:
        if vr is not None:
            try:
                openvr.shutdown()  # type: ignore[name-defined]
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Check VR controller connection")
    parser.add_argument("--samples", type=int, default=1)
    args = parser.parse_args()

    os.environ.setdefault("VR_OVERRIDE", str(STEAMVR_ROOT))
    _print_system_probe()
    return _check_openvr(args.samples)


if __name__ == "__main__":
    sys.exit(main())
