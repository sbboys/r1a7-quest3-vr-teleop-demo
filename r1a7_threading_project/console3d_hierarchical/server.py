#!/usr/bin/env python3
"""Web console aligned with the verified terminal-control baseline."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable

from aiohttp import web

from camera_manager import CameraManager


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
BASE_CONSOLE = PROJECT / "console3d"
BASELINE_CONTROL = (
    PROJECT
    / "transfer_control"
    / "experimental"
    / "r1a7_cartesian_stream_control_kd30_posvel_wu_dqik_u010_v1.py"
)
CONTROL_TOOLS = PROJECT / "transfer_control" / "tools"
CAMERAS = CameraManager(PROJECT.parent)
FUSION_GUIDANCE_PATH = Path("/tmp/r1a7_dual_ibvs_velocity.json")
CONSOLE_IBVS_PATH = Path("/tmp/r1a7_console_ibvs_velocity.json")
VISUAL_SERVO_MAX_SPEED_MM_S = 2.0
VISUAL_SERVO_MAX_SOURCE_AGE_S = 0.20
VISUAL_SERVO_DEADMAN_S = 0.75
VISUAL_SERVO_MAX_TELEMETRY_AGE_S = 1.0
VISUAL_SERVO_MAX_TRACKING_LAG_MM = 15.0
VISUAL_SERVO_DUAL_MODES = {
    "DUAL",
    "DUAL_EXT_PRIORITY",
    "DUAL_WRIST_PRIORITY",
}


class VisualServoGate:
    """Fail-closed bridge from shadow guidance to the robot controller."""

    def __init__(
        self,
        source_path: Path = FUSION_GUIDANCE_PATH,
        output_path: Path = CONSOLE_IBVS_PATH,
    ) -> None:
        self.source_path = source_path
        self.output_path = output_path
        self.control_snapshot: Callable[[], dict[str, Any]] | None = None
        self.authorized_until = 0.0
        self.inhibit_until = 0.0
        self.enabled = False
        self.last_reason = "not_authorized"
        self.last_velocity = [0.0, 0.0, 0.0]
        self.task: asyncio.Task[None] | None = None

    def bind_control_snapshot(
        self, provider: Callable[[], dict[str, Any]]
    ) -> None:
        self.control_snapshot = provider

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write(self, payload: dict[str, Any]) -> None:
        temporary = self.output_path.with_suffix(
            self.output_path.suffix + ".gate.tmp"
        )
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, self.output_path)

    def _disabled_payload(self, reason: str) -> dict[str, Any]:
        return {
            "timestamp": time.time(),
            "enabled": False,
            "reason": reason,
            "vx_mm_s": 0.0,
            "vy_mm_s": 0.0,
            "vz_mm_s": 0.0,
            "source": "console_visual_servo_gate",
        }

    def _evaluate(self, require_deadman: bool = True) -> tuple[bool, str, list[float]]:
        now_monotonic = time.monotonic()
        if now_monotonic < self.inhibit_until:
            return False, "deadman_stop_settling", [0.0, 0.0, 0.0]
        if require_deadman and now_monotonic >= self.authorized_until:
            return False, "deadman_released", [0.0, 0.0, 0.0]

        if self.control_snapshot is None:
            return False, "controller_unavailable", [0.0, 0.0, 0.0]
        control = self.control_snapshot()
        telemetry = control.get("telemetry") or {}
        if control.get("phase") != "enabled" or not control.get("running"):
            return False, "controller_not_enabled", [0.0, 0.0, 0.0]
        if telemetry.get("control_mode") != "cartesian":
            return False, "controller_not_cartesian", [0.0, 0.0, 0.0]
        if telemetry.get("cartesian_arm") != "right":
            return False, "right_arm_not_selected", [0.0, 0.0, 0.0]
        try:
            telemetry_age_s = time.time() - float(telemetry["timestamp"])
        except (KeyError, TypeError, ValueError):
            return False, "controller_telemetry_missing", [0.0, 0.0, 0.0]
        if (
            not math.isfinite(telemetry_age_s)
            or telemetry_age_s < -0.05
            or telemetry_age_s > VISUAL_SERVO_MAX_TELEMETRY_AGE_S
        ):
            return False, "controller_telemetry_stale", [0.0, 0.0, 0.0]
        if telemetry.get("motion_state") not in ("holding", "visual_servo"):
            return False, "manual_motion_not_stopped", [0.0, 0.0, 0.0]
        try:
            tracking_lag_mm = float(telemetry.get("lag_mm", float("inf")))
        except (TypeError, ValueError):
            tracking_lag_mm = float("inf")
        if (
            not math.isfinite(tracking_lag_mm)
            or tracking_lag_mm > VISUAL_SERVO_MAX_TRACKING_LAG_MM
        ):
            return False, "tracking_lag_too_large", [0.0, 0.0, 0.0]

        vision = CAMERAS.threading_vision_snapshot()
        if not vision.get("ready"):
            return False, "vision_stack_not_ready", [0.0, 0.0, 0.0]

        fusion = self._read_json(self.source_path)
        try:
            age_s = time.time() - float(fusion["timestamp"])
        except (KeyError, TypeError, ValueError):
            return False, "fusion_timestamp_missing", [0.0, 0.0, 0.0]
        if (
            not math.isfinite(age_s)
            or age_s < -0.05
            or age_s > VISUAL_SERVO_MAX_SOURCE_AGE_S
        ):
            return False, "fusion_stale", [0.0, 0.0, 0.0]
        if not bool(fusion.get("shadow_mode", False)):
            return False, "fusion_not_shadow_mode", [0.0, 0.0, 0.0]
        if not bool(fusion.get("ready_for_motion", False)):
            return False, "guidance_not_ready", [0.0, 0.0, 0.0]
        if not (
            bool(fusion.get("external_valid", False))
            and bool(fusion.get("wrist_valid", False))
        ):
            return False, "dual_camera_required", [0.0, 0.0, 0.0]
        if fusion.get("fusion_mode") not in VISUAL_SERVO_DUAL_MODES:
            return False, "dual_fusion_mode_required", [0.0, 0.0, 0.0]

        try:
            velocity = [
                float(fusion.get("proposed_vx_mm_s", 0.0)),
                float(fusion.get("proposed_vy_mm_s", 0.0)),
                float(fusion.get("proposed_vz_mm_s", 0.0)),
            ]
        except (TypeError, ValueError):
            return False, "guidance_not_numeric", [0.0, 0.0, 0.0]
        if not all(math.isfinite(value) for value in velocity):
            return False, "guidance_not_finite", [0.0, 0.0, 0.0]
        norm = math.sqrt(sum(value * value for value in velocity))
        if norm > VISUAL_SERVO_MAX_SPEED_MM_S and norm > 0.0:
            scale = VISUAL_SERVO_MAX_SPEED_MM_S / norm
            velocity = [value * scale for value in velocity]
        return True, "deadman_active", velocity

    def publish_once(self) -> None:
        enabled, reason, velocity = self._evaluate(require_deadman=True)
        self.enabled = enabled
        self.last_reason = reason
        self.last_velocity = velocity if enabled else [0.0, 0.0, 0.0]
        payload = {
            "timestamp": time.time(),
            "enabled": enabled,
            "reason": reason,
            "vx_mm_s": self.last_velocity[0],
            "vy_mm_s": self.last_velocity[1],
            "vz_mm_s": self.last_velocity[2],
            "source": "console_visual_servo_gate",
            "max_speed_mm_s": VISUAL_SERVO_MAX_SPEED_MM_S,
        }
        self._write(payload)

    def keepalive(self) -> None:
        if time.monotonic() < self.inhibit_until:
            raise RuntimeError("deadman_stop_settling")
        available, reason, _velocity = self._evaluate(require_deadman=False)
        if not available:
            self.stop(reason)
            raise RuntimeError(reason)
        self.authorized_until = time.monotonic() + VISUAL_SERVO_DEADMAN_S
        self.publish_once()

    def stop(self, reason: str = "operator_stop", inhibit_s: float = 0.0) -> None:
        self.authorized_until = 0.0
        self.inhibit_until = max(
            self.inhibit_until,
            time.monotonic() + max(0.0, inhibit_s),
        )
        self.enabled = False
        self.last_reason = reason
        self.last_velocity = [0.0, 0.0, 0.0]
        self._write(self._disabled_payload(reason))

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "deadman_active": time.monotonic() < self.authorized_until,
            "stop_settling": time.monotonic() < self.inhibit_until,
            "reason": self.last_reason,
            "velocity_mm_s": self.last_velocity,
            "max_speed_mm_s": VISUAL_SERVO_MAX_SPEED_MM_S,
            "requires_dual_camera": True,
            "max_tracking_lag_mm": VISUAL_SERVO_MAX_TRACKING_LAG_MM,
        }

    async def run(self) -> None:
        self.stop("server_start")
        try:
            while True:
                self.publish_once()
                await asyncio.sleep(0.02)
        except asyncio.CancelledError:
            self.stop("server_stop")
            raise

    async def start(self) -> None:
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self.run())

    async def close(self) -> None:
        self.stop("server_stop")
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None


SERVO = VisualServoGate()


def camera_payload(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    result = payload if payload is not None else CAMERAS.snapshot()
    result.setdefault("threading_vision", {})["servo"] = SERVO.snapshot()
    return result


def build_control_arguments(config: dict[str, object]) -> list[str]:
    """Build only the web-facing overrides on top of the verified CLI defaults."""
    interface = str(config.get("interface", "enp6s0"))
    host_ip = str(config.get("host_ip", "192.168.123.223"))
    xy_step = max(0.5, min(30.0, float(config.get("xy_step_mm", 30.0))))
    z_step = max(0.5, min(10.0, float(config.get("z_step_mm", 10.0))))
    joint_step = max(0.1, min(2.0, float(config.get("joint_step_deg", 1.0))))
    gripper_speed = max(0.1, min(2.0, float(config.get("gripper_speed", 1.5))))
    cartesian_speed = max(
        1.0, min(12.0, float(config.get("cartesian_speed_mm_s", 10.0)))
    )
    cartesian_accel = max(
        5.0, min(40.0, float(config.get("cartesian_accel_mm_s2", 20.0)))
    )
    return [
        "--interface",
        interface,
        "--domain-id",
        "0",
        "--host-ip",
        host_ip,
        "--gripper-speed",
        str(gripper_speed),
        "--no-record-d435i",
        "--jog-step-mm",
        str(xy_step),
        "--jog-vertical-step-mm",
        str(z_step),
        "--joint-jog-step-deg",
        str(joint_step),
        "--cartesian-jog-speed-mm-s",
        str(cartesian_speed),
        "--cartesian-jog-accel-mm-s2",
        str(cartesian_accel),
        "--ibvs-command-path",
        str(CONSOLE_IBVS_PATH),
        "--ibvs-max-age-s",
        str(VISUAL_SERVO_MAX_SOURCE_AGE_S),
        "--ibvs-max-speed-mm-s",
        str(VISUAL_SERVO_MAX_SPEED_MM_S),
        "--ibvs-min-speed-mm-s",
        "0.0",
        "--no-udp-visual-servo",
    ]


@web.middleware
async def no_cache_console_ui(
    request: web.Request, handler: web.RequestHandler
) -> web.StreamResponse:
    """Keep the desktop launcher from reusing an obsolete console bundle."""
    response = await handler(request)
    if request.path == "/" or request.path.startswith("/experiment/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


def load_base_server():
    spec = importlib.util.spec_from_file_location(
        "r1a7_console3d_base_server",
        BASE_CONSOLE / "server.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load the existing console server")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CONTROL = BASELINE_CONTROL

    async def baseline_start(self, config):
        async with self.lock:
            if self.process and self.process.returncode is None:
                raise module.web.HTTPConflict(text="controller already running")

            command = [
                str(module.TV_PYTHON),
                "-u",
                str(module.CONTROL),
                *build_control_arguments(config),
            ]
            env = os.environ.copy()
            env["PYTHONNOUSERSITE"] = "1"
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = os.pathsep.join(
                part for part in (str(CONTROL_TOOLS), existing_pythonpath) if part
            )

            self.logs.clear()
            self.telemetry = {}
            self.phase = "starting"
            self.process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(PROJECT.parent),
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            self.reader_task = asyncio.create_task(self.read_output())
            await self.broadcast({"type": "status", "data": self.snapshot()})

    module.ControlSession.start = baseline_start

    async def experimental_index(_request):
        return module.web.FileResponse(HERE / "index.html")

    module.index = experimental_index
    return module


def camera_id(request: web.Request) -> str:
    value = request.match_info["camera_id"]
    if value not in CAMERAS.specs:
        raise web.HTTPNotFound(text="unknown camera")
    return value


async def cameras_status(_request: web.Request) -> web.Response:
    return web.json_response(camera_payload())


async def cameras_rescan(_request: web.Request) -> web.Response:
    return web.json_response(camera_payload(await CAMERAS.rescan()))


async def camera_start(request: web.Request) -> web.Response:
    body = await request.json() if request.can_read_body else {}
    try:
        result = await CAMERAS.start(
            camera_id(request), str(body.get("interface", "enp6s0"))
        )
    except RuntimeError as exc:
        raise web.HTTPConflict(text=str(exc)) from exc
    return web.json_response(camera_payload(result))


async def camera_stop(request: web.Request) -> web.Response:
    return web.json_response(camera_payload(await CAMERAS.stop(camera_id(request))))


async def camera_group_start(request: web.Request) -> web.Response:
    body = await request.json() if request.can_read_body else {}
    return web.json_response(
        camera_payload(await CAMERAS.start_group(
            request.match_info["group"], str(body.get("interface", "enp6s0"))
        ))
    )


async def camera_group_stop(request: web.Request) -> web.Response:
    return web.json_response(
        camera_payload(await CAMERAS.stop_group(request.match_info["group"]))
    )


async def cameras_stop_all(_request: web.Request) -> web.Response:
    SERVO.stop("camera_stop_all", inhibit_s=VISUAL_SERVO_DEADMAN_S)
    return web.json_response(camera_payload(await CAMERAS.stop_all()))


async def threading_vision_start(_request: web.Request) -> web.Response:
    try:
        result = await CAMERAS.start_threading_vision()
    except RuntimeError as exc:
        raise web.HTTPConflict(text=str(exc)) from exc
    return web.json_response(camera_payload(result))


async def threading_vision_stop(_request: web.Request) -> web.Response:
    SERVO.stop("vision_stopped", inhibit_s=VISUAL_SERVO_DEADMAN_S)
    return web.json_response(camera_payload(await CAMERAS.stop_threading_vision()))


async def threading_servo_keepalive(_request: web.Request) -> web.Response:
    try:
        SERVO.keepalive()
    except RuntimeError as exc:
        raise web.HTTPConflict(text=str(exc)) from exc
    return web.json_response(camera_payload())


async def threading_servo_stop(_request: web.Request) -> web.Response:
    SERVO.stop("operator_stop", inhibit_s=0.50)
    return web.json_response(camera_payload())


async def servo_startup(_app: web.Application) -> None:
    await SERVO.start()


async def servo_cleanup(_app: web.Application) -> None:
    await SERVO.close()


async def camera_cleanup(_app: web.Application) -> None:
    await CAMERAS.stop_all()


def add_camera_routes(app: web.Application) -> None:
    app.router.add_get("/api/cameras", cameras_status)
    app.router.add_post("/api/cameras/rescan", cameras_rescan)
    app.router.add_post("/api/cameras/{camera_id}/start", camera_start)
    app.router.add_post("/api/cameras/{camera_id}/stop", camera_stop)
    app.router.add_post("/api/cameras/groups/{group}/start", camera_group_start)
    app.router.add_post("/api/cameras/groups/{group}/stop", camera_group_stop)
    app.router.add_post("/api/cameras/stop-all", cameras_stop_all)
    app.router.add_post("/api/threading-vision/start", threading_vision_start)
    app.router.add_post("/api/threading-vision/stop", threading_vision_stop)
    app.router.add_post(
        "/api/threading-vision/servo/keepalive",
        threading_servo_keepalive,
    )
    app.router.add_post(
        "/api/threading-vision/servo/stop",
        threading_servo_stop,
    )
    app.on_cleanup.append(camera_cleanup)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="R1-A7 terminal-baseline web console"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8098)
    args = parser.parse_args()
    base = load_base_server()
    SERVO.bind_control_snapshot(base.SESSION.snapshot)
    app = base.create_app()

    async def close_control_websockets(_app: web.Application) -> None:
        clients = list(base.SESSION.clients)
        if clients:
            await asyncio.gather(
                *(client.close() for client in clients),
                return_exceptions=True,
            )

    app.middlewares.append(no_cache_console_ui)
    app.on_shutdown.append(close_control_websockets)
    app.on_startup.append(servo_startup)
    app.on_cleanup.insert(0, servo_cleanup)
    add_camera_routes(app)
    app.router.add_static("/experiment", HERE, show_index=False)
    base.web.run_app(
        app,
        host=args.host,
        port=args.port,
        print=None,
    )


if __name__ == "__main__":
    main()
