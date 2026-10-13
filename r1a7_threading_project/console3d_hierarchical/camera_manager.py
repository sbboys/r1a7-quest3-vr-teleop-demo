#!/usr/bin/env python3
"""Camera discovery and preview-process isolation for the R1-A7 console.

The manager never kills arbitrary camera users.  It only signals process groups
that it created itself, and it rediscovers /dev/video* nodes before every start.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
from typing import Any


TV_PYTHON = Path("/home/robot/miniconda3/envs/tv/bin/python")
SAM2_PYTHON = Path(
    os.environ.get(
        "R1A7_SAM2_PYTHON",
        "/home/robot/miniconda3/envs/tv_sam2/bin/python",
    )
)
SAM2_RUNTIME_ROOT = Path(
    os.environ.get(
        "R1A7_SAM2_ROOT",
        "/home/robot/unitree_sim_isaaclab_threading/"
        "r1a7_threading_project/third_party/sam2",
    )
)
THREADING_OUTPUTS = {
    "external": Path("/tmp/r1a7_ibvs_velocity.json"),
    "wrist": Path("/tmp/r1a7_wrist_ibvs_velocity.json"),
    "fusion": Path("/tmp/r1a7_dual_ibvs_velocity.json"),
}


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


@dataclass
class ThreadingVisionRuntime:
    processes: dict[str, asyncio.subprocess.Process] = field(default_factory=dict)
    reader_tasks: dict[str, asyncio.Task[None]] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    last_error: str = ""
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


def threading_vision_commands(
    project_root: Path, external_node: str
) -> dict[str, list[str]]:
    """Build the latest Task 6 perception stack in read-only shadow mode."""
    task_dir = (
        project_root
        / "r1a7_threading_project"
        / "vision_dual_camera"
        / "task6_threading"
    )
    return {
        "external": [
            str(SAM2_PYTHON),
            "-u",
            str(task_dir / "task6_external_ibvs_v4_aruco_edge.py"),
            "--device",
            external_node,
        ],
        "wrist": [
            str(SAM2_PYTHON),
            "-u",
            str(task_dir / "task6_wrist_ibvs_controller_v2_tip.py"),
        ],
        "fusion": [
            str(SAM2_PYTHON),
            "-u",
            str(task_dir / "task6_dual_ibvs_fusion.py"),
            "--shadow-mode",
        ],
    }


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
        self.threading_vision = ThreadingVisionRuntime()
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

    def _threading_requirements(self) -> tuple[bool, str]:
        commands = threading_vision_commands(self.project_root, "/dev/video0")
        required = [SAM2_PYTHON, *[Path(command[2]) for command in commands.values()]]
        required.extend(
            [
                SAM2_RUNTIME_ROOT,
                SAM2_RUNTIME_ROOT / "checkpoints" / "sam2.1_hiera_small.pt",
            ]
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            return False, "缺少视觉运行文件：" + "；".join(missing)
        external = choose_device(self.specs["external"], self.devices)
        if external is None:
            return False, "未检测到外部相机彩色节点"
        return True, ""

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _refresh_threading_vision(self) -> None:
        runtime = self.threading_vision
        for name, process in list(runtime.processes.items()):
            if process.returncode is None:
                continue
            if not runtime.last_error:
                tail = runtime.logs[-1] if runtime.logs else "无日志"
                runtime.last_error = f"{name} 识别进程退出（{process.returncode}）：{tail}"
            runtime.processes.pop(name, None)

    def threading_vision_snapshot(self) -> dict[str, Any]:
        self._refresh_threading_vision()
        runtime = self.threading_vision
        available, requirement_error = self._threading_requirements()
        expected = {"external", "wrist", "fusion"}
        running_names = {
            name
            for name, process in runtime.processes.items()
            if process.returncode is None
        }
        fusion = self._read_json(THREADING_OUTPUTS["fusion"])
        external = self._read_json(THREADING_OUTPUTS["external"])
        wrist = self._read_json(THREADING_OUTPUTS["wrist"])
        proposed = [
            fusion.get("proposed_vx_mm_s"),
            fusion.get("proposed_vy_mm_s"),
            fusion.get("proposed_vz_mm_s"),
        ]
        proposed = [
            float(value) if isinstance(value, (int, float)) else 0.0
            for value in proposed
        ]
        external_valid = bool(fusion.get("external_valid", False))
        wrist_valid = bool(fusion.get("wrist_valid", False))
        sources_live = bool(running_names)
        return {
            "available": available,
            "running": bool(running_names),
            "ready": running_names == expected,
            "processes": sorted(running_names),
            "pids": {
                name: process.pid
                for name, process in runtime.processes.items()
                if process.returncode is None
            },
            "mode": fusion.get("fusion_mode", "WAITING"),
            "shadow_mode": True,
            "motion_output_enabled": False,
            "ready_for_guidance": bool(fusion.get("ready_for_motion", False))
            and (external_valid or wrist_valid),
            "external_valid": external_valid,
            "wrist_valid": wrist_valid,
            # Do not present detections left behind by a previous manual run as
            # current console data when this stack is stopped.
            "external_tip": external.get("tweezer_tip_pixel") if sources_live else None,
            "external_hole": external.get("hole_pixel") if sources_live else None,
            "wrist_tip": wrist.get("tweezer_tip_pixel") if sources_live else None,
            "wrist_hole": wrist.get("hole_pixel") if sources_live else None,
            "proposed_velocity_mm_s": proposed,
            "last_error": runtime.last_error or requirement_error,
            "logs": runtime.logs[-12:],
        }

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
        return {
            "cameras": items,
            "threading_vision": self.threading_vision_snapshot(),
        }

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

    async def _read_threading_output(
        self, name: str, process: asyncio.subprocess.Process
    ) -> None:
        assert process.stdout is not None
        runtime = self.threading_vision
        while line := await process.stdout.readline():
            message = line.decode("utf-8", errors="replace").rstrip()
            if message:
                runtime.logs.append(f"[{name}] {message}")
                del runtime.logs[:-120]

    @staticmethod
    def _write_threading_disabled_outputs(reason: str) -> None:
        now = time.time()
        source_payload = {
            "enabled": False,
            "timestamp": now,
            "reason": reason,
        }
        fusion_payload = {
            **source_payload,
            "shadow_mode": True,
            "ready_for_motion": False,
            "fusion_mode": "STOP",
            "external_valid": False,
            "wrist_valid": False,
            "vx_mm_s": 0.0,
            "vy_mm_s": 0.0,
            "vz_mm_s": 0.0,
            "proposed_vx_mm_s": 0.0,
            "proposed_vy_mm_s": 0.0,
            "proposed_vz_mm_s": 0.0,
        }
        for name, path in THREADING_OUTPUTS.items():
            payload = fusion_payload if name == "fusion" else source_payload
            temporary = path.with_suffix(path.suffix + ".console.tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(temporary, path)

    async def start_threading_vision(self) -> dict[str, Any]:
        """Start Task 6 recognition and guidance with robot output forced off."""
        async with self.lock:
            self._refresh_threading_vision()
            if self.threading_vision.processes:
                return self.snapshot()

            self.rescan_sync()
            available, requirement_error = self._threading_requirements()
            if not available:
                self.threading_vision.last_error = requirement_error
                raise RuntimeError(requirement_error)
            external = choose_device(self.specs["external"], self.devices)
            assert external is not None

            # These two ordinary previews compete with the Task 6 capture
            # backends. Only stop preview processes owned by this console.
            await self._stop_unlocked("external")
            await self._stop_unlocked("orbbec_right")

            runtime = self.threading_vision
            runtime.logs.clear()
            runtime.last_error = ""
            runtime.started_at = time.monotonic()
            self._write_threading_disabled_outputs("recognition_starting")

            env = os.environ.copy()
            env.setdefault("DISPLAY", ":0")
            env["PYTHONNOUSERSITE"] = "1"
            env["R1A7_SAM2_ROOT"] = str(SAM2_RUNTIME_ROOT)
            env["R1A7_SAM2_CHECKPOINT"] = str(
                SAM2_RUNTIME_ROOT / "checkpoints" / "sam2.1_hiera_small.pt"
            )
            commands = threading_vision_commands(self.project_root, external.node)
            vision_workdir = self.project_root / "r1a7_threading_project"

            try:
                for name in ("external", "wrist", "fusion"):
                    process = await asyncio.create_subprocess_exec(
                        *commands[name],
                        # The validated Task 6 scripts resolve calibration files
                        # relative to r1a7_threading_project.
                        cwd=str(vision_workdir),
                        env=env,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                        start_new_session=True,
                    )
                    runtime.processes[name] = process
                    runtime.reader_tasks[name] = asyncio.create_task(
                        self._read_threading_output(name, process)
                    )
                    await asyncio.sleep(0.25)
                    if process.returncode is not None:
                        raise RuntimeError(
                            f"{name} 识别进程启动失败，返回码 {process.returncode}"
                        )
            except Exception as exc:
                await self._stop_threading_unlocked("recognition_start_failed")
                runtime.last_error = str(exc)
                raise
            return self.snapshot()

    async def stop_threading_vision(self) -> dict[str, Any]:
        async with self.lock:
            await self._stop_threading_unlocked("operator_stop")
            return self.snapshot()

    async def _stop_threading_unlocked(self, reason: str) -> None:
        runtime = self.threading_vision
        for process in list(runtime.processes.values()):
            if process.returncode is not None:
                continue
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        for process in list(runtime.processes.values()):
            if process.returncode is not None:
                continue
            try:
                await asyncio.wait_for(process.wait(), timeout=3.0)
            except asyncio.TimeoutError:
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
        runtime.processes.clear()
        runtime.reader_tasks.clear()
        runtime.started_at = None
        self._write_threading_disabled_outputs(reason)

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
            self._refresh_threading_vision()
            if camera_id in ("external", "orbbec_right") and self.threading_vision.processes:
                runtime.last_error = "细线穿孔识别正在占用此相机，请先停止识别"
                raise RuntimeError(runtime.last_error)

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
        await self.stop_threading_vision()
        for spec in CAMERAS:
            await self.stop(spec.camera_id)
        return self.snapshot()
