#!/usr/bin/env python3
"""Local web server for the R1-A7 real-robot and URDF digital-twin console."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any

from aiohttp import web


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
CONTROL = PROJECT / "transfer_control" / "tools" / "r1a7_cartesian_stream_control.py"
TV_PYTHON = Path("/home/robot/miniconda3/envs/tv/bin/python")
ALLOWED_COMMANDS = {"w", "s", "a", "d", "u", "j", "i", "h", "q", "ENABLE"}
JOG_DIRECTIONS = {"w", "s", "a", "d", "u", "j", "i"}


def command_allowed(command: str) -> bool:
    if command in ALLOWED_COMMANDS:
        return True
    parts = command.split(":")
    if parts == ["jog", "stop"]:
        return True
    if len(parts) == 3 and parts[0] == "jog":
        return parts[1] in ("start", "keepalive") and parts[2] in JOG_DIRECTIONS
    if len(parts) != 3:
        return False
    if parts[0] == "gripper":
        return (
            parts[1] in ("left", "right")
            and parts[2] in ("open", "hold", "close")
        )
    if parts[0] != "joint" or parts[2] not in ("+", "-"):
        return False
    try:
        return 0 <= int(parts[1]) < 14
    except ValueError:
        return False


class ControlSession:
    def __init__(self) -> None:
        self.process: asyncio.subprocess.Process | None = None
        self.reader_task: asyncio.Task[None] | None = None
        self.logs: list[str] = []
        self.telemetry: dict[str, Any] = {}
        self.phase = "stopped"
        self.clients: set[web.WebSocketResponse] = set()
        self.lock = asyncio.Lock()

    def snapshot(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "running": bool(self.process and self.process.returncode is None),
            "pid": self.process.pid if self.process and self.process.returncode is None else None,
            "telemetry": self.telemetry,
            "logs": self.logs[-120:],
        }

    async def broadcast(self, payload: dict[str, Any]) -> None:
        stale = []
        for ws in self.clients:
            try:
                await ws.send_json(payload)
            except (ConnectionResetError, RuntimeError):
                stale.append(ws)
        for ws in stale:
            self.clients.discard(ws)

    async def start(self, config: dict[str, Any]) -> None:
        async with self.lock:
            if self.process and self.process.returncode is None:
                raise web.HTTPConflict(text="controller already running")
            interface = str(config.get("interface", "enp6s0"))
            host_ip = str(config.get("host_ip", "192.168.123.223"))
            xy_step = max(0.5, min(10.0, float(config.get("xy_step_mm", 5.0))))
            z_step = max(0.5, min(5.0, float(config.get("z_step_mm", 2.0))))
            joint_step = max(0.1, min(2.0, float(config.get("joint_step_deg", 0.5))))
            gripper_speed = max(0.1, min(2.0, float(config.get("gripper_speed", 0.6))))
            cartesian_speed = max(
                1.0, min(12.0, float(config.get("cartesian_speed_mm_s", 8.0)))
            )
            cartesian_accel = max(
                5.0, min(40.0, float(config.get("cartesian_accel_mm_s2", 20.0)))
            )
            # VR-equivalent baseline: the verified VR executor publishes zero
            # arm feed-forward torque.  Keep this forced off until a separate
            # gravity-compensation qualification is completed.
            gravity_ff_scale = 0.0
            command = [
                str(TV_PYTHON), "-u", str(CONTROL),
                "--interface", interface,
                "--domain-id", "0",
                "--host-ip", host_ip,
                "--enable-gripper",
                "--gripper-initial-mode", "current",
                "--gripper-speed", str(gripper_speed),
                "--jog-step-mm", str(xy_step),
                "--jog-vertical-step-mm", str(z_step),
                "--joint-jog-step-deg", str(joint_step),
                "--cartesian-jog-speed-mm-s", str(cartesian_speed),
                "--cartesian-jog-accel-mm-s2", str(cartesian_accel),
                "--max-joint-speed", "0.8",
                "--right-arm-gravity-feedforward-scale", str(gravity_ff_scale),
                "--gravity-feedforward-max-torque", "2.0",
                "--gravity-feedforward-slew", "2.0",
            ]
            env = os.environ.copy()
            env["PYTHONNOUSERSITE"] = "1"
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

    async def read_output(self) -> None:
        assert self.process and self.process.stdout
        while line := await self.process.stdout.readline():
            text = line.decode("utf-8", errors="replace").rstrip()
            if text.startswith("[TELEMETRY_JSON] "):
                try:
                    self.telemetry = json.loads(text.removeprefix("[TELEMETRY_JSON] "))
                    if self.phase != "enabled":
                        self.phase = "enabled"
                        await self.broadcast({"type": "status", "data": self.snapshot()})
                    await self.broadcast({"type": "telemetry", "data": self.telemetry})
                except json.JSONDecodeError:
                    pass
            else:
                self.logs.append(text)
                del self.logs[:-300]
                if "Type ENABLE" in text:
                    self.phase = "waiting_enable"
                elif (
                    "lowcmd enabled" in text.lower()
                    or "R1-A7 CONTINUOUS CARTESIAN STREAM ACTIVE" in text
                ):
                    self.phase = "enabled"
                await self.broadcast({"type": "log", "data": text, "phase": self.phase})
        returncode = await self.process.wait()
        self.logs.append(f"[controller exited: {returncode}]")
        self.phase = "stopped" if returncode == 0 else "fault"
        await self.broadcast({"type": "status", "data": self.snapshot()})

    async def send(self, command: str) -> None:
        if not command_allowed(command):
            raise web.HTTPBadRequest(text="unsupported command")
        if not self.process or self.process.returncode is not None or not self.process.stdin:
            raise web.HTTPConflict(text="controller is not running")
        self.process.stdin.write((command + "\n").encode())
        await self.process.stdin.drain()

    async def stop(self) -> None:
        if not self.process or self.process.returncode is not None:
            self.phase = "stopped"
            return
        self.phase = "stopping"
        try:
            await self.send("q")
            await asyncio.wait_for(self.process.wait(), timeout=3.0)
        except (asyncio.TimeoutError, web.HTTPException):
            self.process.send_signal(signal.SIGTERM)
            await self.process.wait()
        self.phase = "stopped"


SESSION = ControlSession()


async def index(_request: web.Request) -> web.FileResponse:
    return web.FileResponse(HERE / "static" / "index.html")


async def status(_request: web.Request) -> web.Response:
    return web.json_response(SESSION.snapshot())


async def start(request: web.Request) -> web.Response:
    await SESSION.start(await request.json())
    return web.json_response(SESSION.snapshot())


async def command(request: web.Request) -> web.Response:
    body = await request.json()
    await SESSION.send(str(body.get("command", "")))
    return web.json_response({"ok": True})


async def stop(_request: web.Request) -> web.Response:
    await SESSION.stop()
    return web.json_response(SESSION.snapshot())


async def websocket(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    SESSION.clients.add(ws)
    await ws.send_json({"type": "status", "data": SESSION.snapshot()})
    try:
        async for _message in ws:
            pass
    finally:
        SESSION.clients.discard(ws)
    return ws


async def cleanup(_app: web.Application) -> None:
    await SESSION.stop()


def create_app() -> web.Application:
    app = web.Application(client_max_size=1024 * 1024)
    app.router.add_get("/", index)
    app.router.add_get("/api/status", status)
    app.router.add_post("/api/start", start)
    app.router.add_post("/api/command", command)
    app.router.add_post("/api/stop", stop)
    app.router.add_get("/ws", websocket)
    app.router.add_static("/static", HERE / "static", show_index=False)
    app.router.add_static("/model", HERE / "model", show_index=False)
    app.on_cleanup.append(cleanup)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="R1-A7 3D Cartesian jog console")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8097)
    args = parser.parse_args()
    web.run_app(create_app(), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
