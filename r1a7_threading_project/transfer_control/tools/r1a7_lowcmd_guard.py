#!/usr/bin/env python3
from __future__ import annotations

import fcntl
import os
import subprocess
from pathlib import Path


LOCK_PATH = Path("/tmp/r1a7_vr_g1ik_real.lock")

KNOWN_UNGUARDED_PUBLISHERS = (
    "/home/version/桌面/r1a7_vr_dual_arm_g1ik_real.py",
    "/home/version/桌面/r1a7_vr_dual_arm_g1ik_real(1).py",
    "r1a7_hold_current_only.py",
    "r1a7_vr_dual_arm_g1ik_real.py",
    "r1a7_vr_g1ik_real_only_start_a.py",
    "r1a7_vr_g1ik_real_with_mujoco_mirror.py",
    "r1A7_lowlevel_hold_record",
    "r1A7_lowlevel_replay",
    "r1A7_lowlevel_replay_tune",
    "r1A7_lowlevel_right_arm_servo",
)


def _ancestor_pids(pid: int) -> set[int]:
    ancestors = {pid}
    current = pid
    while current > 1:
        try:
            fields = Path(f"/proc/{current}/stat").read_text().split()
            current = int(fields[3])
        except (FileNotFoundError, IndexError, OSError, ValueError):
            break
        ancestors.add(current)
    return ancestors


def _find_unguarded_publishers() -> list[tuple[int, str]]:
    current_pid = os.getpid()
    protected = _ancestor_pids(current_pid)
    try:
        own_cmdline = Path("/proc/self/cmdline").read_bytes().replace(b"\0", b" ").decode(
            errors="replace"
        ).strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        own_cmdline = ""
    conflicts: list[tuple[int, str]] = []
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        if pid == current_pid:
            continue
        if pid in protected:
            continue
        try:
            cmdline = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if own_cmdline and cmdline.strip() == own_cmdline:
            continue
        if any(marker in cmdline for marker in KNOWN_UNGUARDED_PUBLISHERS):
            conflicts.append((pid, cmdline.strip()))
    return conflicts


def acquire_lowcmd_guard(owner: str, topic: str = "rt/lowcmd"):
    lock_file = open(LOCK_PATH, "a+", encoding="ascii")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_file.seek(0)
        holder = lock_file.read().strip() or "unknown owner"
        lock_file.close()
        raise RuntimeError(
            f"another R1-A7 lowcmd publisher owns {LOCK_PATH}: {holder}"
        ) from exc

    conflicts = _find_unguarded_publishers()
    if conflicts:
        details = "; ".join(f"pid={pid} cmd={cmd}" for pid, cmd in conflicts)
        lock_file.close()
        raise RuntimeError(
            "refusing to publish because another known unguarded R1-A7 "
            f"controller is running: {details}"
        )

    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(f"pid={os.getpid()} owner={owner} topic={topic}\n")
    lock_file.flush()
    return lock_file


def ensure_tcp_port_available(port: int) -> None:
    try:
        output = subprocess.check_output(
            ["lsof", f"-tiTCP:{port}", "-sTCP:LISTEN"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return
    pids = sorted({int(text) for text in output.split() if text.isdigit()})
    if not pids:
        return

    owners: list[str] = []
    for pid in pids:
        try:
            cmdline = (Path(f"/proc/{pid}") / "cmdline").read_bytes()
            command = cmdline.replace(b"\0", b" ").decode(errors="replace").strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            command = "unknown"
        owners.append(f"pid={pid} cmd={command}")
    raise RuntimeError(
        f"TCP port {port} is already in use; refusing to terminate it automatically: "
        + "; ".join(owners)
    )
