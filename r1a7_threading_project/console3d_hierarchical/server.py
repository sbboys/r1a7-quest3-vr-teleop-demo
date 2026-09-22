#!/usr/bin/env python3
"""Web console aligned with the verified terminal-control baseline."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
BASE_CONSOLE = PROJECT / "console3d"
BASELINE_CONTROL = (
    PROJECT
    / "transfer_control"
    / "experimental"
    / "r1a7_cartesian_stream_control_experimental.py"
)
CONTROL_TOOLS = PROJECT / "transfer_control" / "tools"


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

            interface = str(config.get("interface", "enp6s0"))
            host_ip = str(config.get("host_ip", "192.168.123.223"))
            xy_step = max(0.5, min(10.0, float(config.get("xy_step_mm", 5.0))))
            z_step = max(0.5, min(5.0, float(config.get("z_step_mm", 2.0))))
            joint_step = max(
                0.1, min(2.0, float(config.get("joint_step_deg", 0.5)))
            )
            cartesian_speed = max(
                1.0, min(12.0, float(config.get("cartesian_speed_mm_s", 10.0)))
            )
            cartesian_accel = max(
                5.0, min(40.0, float(config.get("cartesian_accel_mm_s2", 30.0)))
            )

            command = [
                str(module.TV_PYTHON),
                "-u",
                str(module.CONTROL),
                "--interface",
                interface,
                "--domain-id",
                "0",
                "--host-ip",
                host_ip,
                "--no-enable-gripper",
                "--print-period",
                "0.5",
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
                "--right-arm-gravity-feedforward-scale",
                "0.25",
                "--gravity-feedforward-max-torque",
                "2.0",
                "--gravity-feedforward-slew",
                "2.0",
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="R1-A7 terminal-baseline web console"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8098)
    args = parser.parse_args()
    base = load_base_server()
    app = base.create_app()
    app.router.add_static("/experiment", HERE, show_index=False)
    base.web.run_app(
        app,
        host=args.host,
        port=args.port,
        print=None,
    )


if __name__ == "__main__":
    main()
