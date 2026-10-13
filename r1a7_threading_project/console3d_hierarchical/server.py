#!/usr/bin/env python3
"""Web console aligned with the verified terminal-control baseline."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
from pathlib import Path

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
    return web.json_response(CAMERAS.snapshot())


async def cameras_rescan(_request: web.Request) -> web.Response:
    return web.json_response(await CAMERAS.rescan())


async def camera_start(request: web.Request) -> web.Response:
    body = await request.json() if request.can_read_body else {}
    try:
        result = await CAMERAS.start(
            camera_id(request), str(body.get("interface", "enp6s0"))
        )
    except RuntimeError as exc:
        raise web.HTTPConflict(text=str(exc)) from exc
    return web.json_response(result)


async def camera_stop(request: web.Request) -> web.Response:
    return web.json_response(await CAMERAS.stop(camera_id(request)))


async def camera_group_start(request: web.Request) -> web.Response:
    body = await request.json() if request.can_read_body else {}
    return web.json_response(
        await CAMERAS.start_group(
            request.match_info["group"], str(body.get("interface", "enp6s0"))
        )
    )


async def camera_group_stop(request: web.Request) -> web.Response:
    return web.json_response(await CAMERAS.stop_group(request.match_info["group"]))


async def cameras_stop_all(_request: web.Request) -> web.Response:
    return web.json_response(await CAMERAS.stop_all())


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
    app.on_cleanup.append(camera_cleanup)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="R1-A7 terminal-baseline web console"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8098)
    args = parser.parse_args()
    base = load_base_server()
    app = base.create_app()
    app.middlewares.append(no_cache_console_ui)
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
