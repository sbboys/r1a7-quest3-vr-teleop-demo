#!/usr/bin/env python3
"""Camera discovery and preview-process isolation for the R1-A7 console.

The manager never kills arbitrary camera users.  It only signals process groups
that it created itself, and it rediscovers /dev/video* nodes before every start.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
from typing import Any


TV_PYTHON = Path("/home/robot/miniconda3/envs/tv/bin/python")


@dataclass(frozen=True)
class CameraSpec:
    camera_id: str
    label: str
    group: str
    kind: str
    serial: str = ""
    description: str = ""


@dataclass(frozen=True)
class VideoDevice:
    node: str
    product: str
    serial: str
    formats: tuple[str, ...]


@dataclass
class CameraRuntime:
    process: asyncio.subprocess.Process | None = None
    reader_task: asyncio.Task[None] | None = None
    logs: list[str] = field(default_factory=list)
    last_error: str = ""
    node: str | None = None
    pixel_format: str | None = None
    ready: bool = False
    started_at: float | None = None


CAMERAS: tuple[CameraSpec, ...] = (
    CameraSpec(
        "head",
        "头部相机",
        "head",
        "head",
        description="GitHub 已验证 Unitree SDK VideoClient（1280×720 RGB）",
    ),
    CameraSpec(
        "wrist_left",
        "左腕相机",
        "wrist",
        "uvc",
        "JR0001",
        "JR0001 腕部 RGB 相机",
    ),
    CameraSpec(
        "wrist_right",
        "右腕相机",
        "wrist",
        "uvc",
        "JR0002",
        "JR0002 腕部 RGB 相机",
    ),
    CameraSpec(
        "external",
        "外部相机",
        "external",
        "uvc",
        "20685204b12d5283",
        "USB ZOOM Camera 固定视角",
    ),
    CameraSpec(
        "orbbec_left",
        "奥比中光 007Z",
        "orbbec",
        "uvc",
        "CPCBC530007Z",
        "Gemini 336L 彩色流",
    ),
    CameraSpec(
        "orbbec_right",
        "奥比中光 00C5",
        "orbbec",
        "uvc",
        "CPCBC53000C5",
        "Gemini 336L 彩色流",
    ),
)


def _video_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def _properties(node: Path) -> dict[str, str]:
    try:
        result = subprocess.run(
            ["udevadm", "info", "--query=property", f"--name={node}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    props: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            props[key] = value
    return props


def _formats(node: Path) -> tuple[str, ...]:
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", str(node), "--list-formats-ext"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ()
    return tuple(dict.fromkeys(re.findall(r"'([A-Z0-9]{4})'", result.stdout)))


def discover_video_devices(dev_root: Path = Path("/dev")) -> list[VideoDevice]:
    """Enumerate capture-capable nodes with stable identity metadata."""
    devices: list[VideoDevice] = []
    for node in sorted(dev_root.glob("video*"), key=_video_sort_key):
        formats = _formats(node)
        if not formats:
            continue
        props = _properties(node)
        devices.append(
            VideoDevice(
                node=str(node),
                product=props.get("ID_V4L_PRODUCT", "unknown"),
                serial=props.get("ID_SERIAL_SHORT", ""),
                formats=formats,
            )
        )
    return devices


def choose_device(spec: CameraSpec, devices: list[VideoDevice]) -> VideoDevice | None:
    """Choose an RGB node by serial; reject depth-only Orbbec nodes."""
    candidates = [device for device in devices if device.serial == spec.serial]
    color = [device for device in candidates if "MJPG" in device.formats]
    if color:
        return color[0]
    raw = [device for device in candidates if "YUYV" in device.formats]
    return raw[0] if raw else None


def preview_command(device: VideoDevice) -> tuple[list[str], str]:
    if "MJPG" in device.formats:
        return (
            [
                "gst-launch-1.0",
                "-q",
                "v4l2src",
                f"device={device.node}",
                "do-timestamp=true",
                "!",
                "image/jpeg,width=640,height=480,framerate=30/1",
                "!",
                "jpegdec",
                "!",
                "videoconvert",
                "!",
                "autovideosink",
                "sync=false",
            ],
            "MJPG",
        )
    return (
        [
            "gst-launch-1.0",
            "-q",
            "v4l2src",
            f"device={device.node}",
            "do-timestamp=true",
            "!",
            "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1",
            "!",
            "videoconvert",
            "!",
            "autovideosink",
            "sync=false",
        ],
        "YUYV",
    )


class CameraManager:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.head_script = (
            project_root / "r1a7_threading_project" / "vision" / "r1a7_head_visual_baseline.py"
        )
        self.head_service_prepare = (
            project_root
            / "r1a7_threading_project"
            / "vision"
            / "r1a7_head_service_prepare.py"
        )
        self.specs = {spec.camera_id: spec for spec in CAMERAS}
        self.runtime = {camera_id: CameraRuntime() for camera_id in self.specs}
        self.devices: list[VideoDevice] = []
        self.lock = asyncio.Lock()
        self.rescan_sync()

    def rescan_sync(self) -> None:
        self.devices = discover_video_devices()

    def _refresh_process(self, runtime: CameraRuntime) -> None:
        if runtime.process and runtime.process.returncode is not None:
            if runtime.process.returncode != 0 and not runtime.last_error:
                runtime.last_error = f"预览进程退出，返回码 {runtime.process.returncode}"
            runtime.process = None
            runtime.node = None
            runtime.pixel_format = None
            runtime.started_at = None

    def snapshot(self) -> dict[str, Any]:
        items = []
        for spec in CAMERAS:
            runtime = self.runtime[spec.camera_id]
            self._refresh_process(runtime)
            if (
                spec.kind == "head"
                and runtime.process
                and runtime.process.returncode is None
                and not runtime.ready
                and runtime.started_at is not None
                and time.monotonic() - runtime.started_at >= 12.0
                and not runtime.last_error
            ):
                runtime.last_error = (
                    "video_hub 已启动，但 12 秒内未收到 JPEG；"
                    "请先检查手机 App 与机器人头部相机"
                )
            device = choose_device(spec, self.devices) if spec.kind == "uvc" else None
            detected = (
                bool(device)
                if spec.kind == "uvc"
                else self.head_script.exists() and TV_PYTHON.exists()
            )
            items.append(
                {
                    "id": spec.camera_id,
                    "label": spec.label,
                    "group": spec.group,
                    "kind": spec.kind,
                    "serial": spec.serial,
                    "description": spec.description,
                    "source": (
                        "Unitree SDK VideoClient · enp6s0 · 目标 1280×720 RGB"
                        if spec.kind == "head"
                        else "USB UVC"
                    ),
                    "backend": "sdk_videoclient" if spec.kind == "head" else "uvc",
                    "detected": detected,
                    "node": runtime.node or (device.node if device else None),
                    "formats": list(device.formats) if device else [],
                    "running": bool(runtime.process and runtime.process.returncode is None),
                    "ready": bool(
                        runtime.process
                        and runtime.process.returncode is None
                        and (spec.kind == "uvc" or runtime.ready)
                    ),
                    "pid": (
                        runtime.process.pid
                        if runtime.process and runtime.process.returncode is None
                        else None
                    ),
                    "pixel_format": runtime.pixel_format,
                    "last_error": runtime.last_error,
                    "logs": runtime.logs[-8:],
                }
            )
        return {"cameras": items}

    async def rescan(self) -> dict[str, Any]:
        async with self.lock:
            self.rescan_sync()
            return self.snapshot()

    async def _read_output(self, camera_id: str, process: asyncio.subprocess.Process) -> None:
        assert process.stdout is not None
        runtime = self.runtime[camera_id]
        while line := await process.stdout.readline():
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                runtime.logs.append(text)
                del runtime.logs[:-40]
                if camera_id == "head" and "[HEAD FRAME READY]" in text:
                    runtime.ready = True
                    runtime.last_error = ""
                elif camera_id == "head" and "send request error" in text.lower():
                    runtime.last_error = "VideoClient 已启动，但暂未收到头部图像"

    async def _prepare_head_camera(
        self,
        runtime: CameraRuntime,
        head_interface: str,
        env: dict[str, str],
    ) -> None:
        if not self.head_service_prepare.exists():
            raise RuntimeError("头部相机服务准备程序不存在")
        process = await asyncio.create_subprocess_exec(
            str(TV_PYTHON),
            "-u",
            str(self.head_service_prepare),
            "--interface",
            head_interface,
            cwd=str(self.project_root),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            output, _ = await asyncio.wait_for(process.communicate(), timeout=35.0)
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.wait()
            raise RuntimeError("头部相机服务准备超时") from exc
        lines = output.decode("utf-8", errors="replace").splitlines()
        runtime.logs.extend(line for line in lines if line)
        del runtime.logs[:-40]
        if process.returncode != 0:
            detail = lines[-1] if lines else f"返回码 {process.returncode}"
            raise RuntimeError(f"头部相机服务准备失败：{detail}")

    async def start(self, camera_id: str, head_interface: str = "enp6s0") -> dict[str, Any]:
        if camera_id not in self.specs:
            raise KeyError(camera_id)
        async with self.lock:
            spec = self.specs[camera_id]
            runtime = self.runtime[camera_id]
            self._refresh_process(runtime)
            if runtime.process:
                return self.snapshot()

            self.rescan_sync()
            env = os.environ.copy()
            env.setdefault("DISPLAY", ":0")
            runtime.logs.clear()
            runtime.last_error = ""
            runtime.node = None
            runtime.pixel_format = None
            runtime.ready = False
            runtime.started_at = None

            if spec.kind == "head":
                if (
                    not self.head_script.exists()
                    or not self.head_service_prepare.exists()
                    or not TV_PYTHON.exists()
                ):
                    runtime.last_error = "头部相机程序或 tv Python 环境不存在"
                    raise RuntimeError(runtime.last_error)
                env["R1A7_INTERFACE"] = head_interface
                env["PYTHONNOUSERSITE"] = "1"
                try:
                    await self._prepare_head_camera(runtime, head_interface, env)
                except RuntimeError as exc:
                    runtime.last_error = str(exc)
                    raise
                command = [str(TV_PYTHON), "-u", str(self.head_script)]
            else:
                device = choose_device(spec, self.devices)
                if not device:
                    runtime.last_error = f"未检测到 {spec.label} 的彩色采集节点"
                    raise RuntimeError(runtime.last_error)
                if not shutil.which("gst-launch-1.0"):
                    runtime.last_error = "系统中未找到 gst-launch-1.0"
                    raise RuntimeError(runtime.last_error)
                command, runtime.pixel_format = preview_command(device)
                runtime.node = device.node

            runtime.process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(self.project_root),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            runtime.started_at = time.monotonic()
            runtime.reader_task = asyncio.create_task(
                self._read_output(camera_id, runtime.process)
            )
            await asyncio.sleep(0.35)
            if runtime.process.returncode is not None:
                returncode = runtime.process.returncode
                runtime.process = None
                runtime.node = None
                runtime.pixel_format = None
                runtime.ready = False
                runtime.started_at = None
                runtime.last_error = (
                    runtime.logs[-1] if runtime.logs else f"预览启动失败，返回码 {returncode}"
                )
                raise RuntimeError(runtime.last_error)
            return self.snapshot()

    async def stop(self, camera_id: str) -> dict[str, Any]:
        if camera_id not in self.specs:
            raise KeyError(camera_id)
        async with self.lock:
            await self._stop_unlocked(camera_id)
            return self.snapshot()

    async def _stop_unlocked(self, camera_id: str) -> None:
        runtime = self.runtime[camera_id]
        process = runtime.process
        if not process or process.returncode is not None:
            runtime.process = None
            runtime.node = None
            runtime.pixel_format = None
            runtime.ready = False
            runtime.started_at = None
            return
        try:
            os.killpg(process.pid, signal.SIGINT)
            await asyncio.wait_for(process.wait(), timeout=2.0)
        except (ProcessLookupError, asyncio.TimeoutError):
            if process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await process.wait()
        runtime.process = None
        runtime.node = None
        runtime.pixel_format = None
        runtime.ready = False
        runtime.started_at = None

    async def start_group(self, group: str, head_interface: str = "enp6s0") -> dict[str, Any]:
        errors: dict[str, str] = {}
        for spec in CAMERAS:
            if spec.group != group:
                continue
            try:
                await self.start(spec.camera_id, head_interface)
            except RuntimeError as exc:
                errors[spec.camera_id] = str(exc)
        result = self.snapshot()
        result["errors"] = errors
        return result

    async def stop_group(self, group: str) -> dict[str, Any]:
        for spec in CAMERAS:
            if spec.group == group:
                await self.stop(spec.camera_id)
        return self.snapshot()

    async def stop_all(self) -> dict[str, Any]:
        for spec in CAMERAS:
            await self.stop(spec.camera_id)
        return self.snapshot()
