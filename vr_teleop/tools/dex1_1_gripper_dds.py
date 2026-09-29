#!/usr/bin/env python3
"""DDS helper for Unitree Dex1_1 gripper service."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_


@dataclass
class Dex1State:
    q: float
    dq: float
    stamp: float


class Dex1GripperDDS:
    def __init__(
        self,
        side: str = "right",
        open_q: float = 5.40,
        close_q: float = 0.0,
        kp: float = 5.0,
        kd: float = 0.05,
        max_step: float = 0.18,
    ):
        self.side = side
        if self.side not in ("left", "right"):
            raise ValueError("side must be left or right")
        self.command_topic = f"rt/dex1/{self.side}/cmd"
        self.state_topic = f"rt/dex1/{self.side}/state"
        self.open_q = float(open_q)
        self.close_q = float(close_q)
        self.kp = float(kp)
        self.kd = float(kd)
        self.max_step = float(max_step)
        self.state: Optional[Dex1State] = None
        self.command_q: Optional[float] = None

        self.publisher = ChannelPublisher(self.command_topic, MotorCmds_)
        self.publisher.Init()
        self.subscriber = ChannelSubscriber(self.state_topic, MotorStates_)
        self.subscriber.Init(self._state_handler, 10)

        self.msg = MotorCmds_()
        self.msg.cmds = [unitree_go_msg_dds__MotorCmd_()]
        self.msg.cmds[0].mode = 1
        self.msg.cmds[0].dq = 0.0
        self.msg.cmds[0].tau = 0.0
        self.msg.cmds[0].kp = self.kp
        self.msg.cmds[0].kd = self.kd

    def _state_handler(self, msg: MotorStates_) -> None:
        if not msg.states:
            return
        state = msg.states[0]
        self.state = Dex1State(q=float(state.q), dq=float(state.dq), stamp=time.monotonic())

    def wait_state(self, timeout_s: float = 3.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_s)
        while time.monotonic() < deadline:
            if self.state is not None:
                if self.command_q is None:
                    self.command_q = self.state.q
                return True
            time.sleep(0.01)
        return False

    def grip_to_q(self, grip: float) -> float:
        grip = float(np.clip(grip, 0.0, 1.0))
        return self.open_q + grip * (self.close_q - self.open_q)

    def publish_q(self, target_q: float) -> float:
        if self.command_q is None:
            self.command_q = self.state.q if self.state is not None else self.open_q
        target_q = float(np.clip(target_q, min(self.open_q, self.close_q), max(self.open_q, self.close_q)))
        step = float(np.clip(target_q - self.command_q, -self.max_step, self.max_step))
        self.command_q += step
        self.msg.cmds[0].q = self.command_q
        self.msg.cmds[0].mode = 1
        self.msg.cmds[0].kp = self.kp
        self.msg.cmds[0].kd = self.kd
        self.publisher.Write(self.msg)
        return self.command_q

    def publish_grip(self, grip: float) -> float:
        return self.publish_q(self.grip_to_q(grip))


def main() -> int:
    parser = argparse.ArgumentParser(description="Unitree Dex1_1 DDS gripper test")
    parser.add_argument("--interface", default="enx9c69d37d0967")
    parser.add_argument("--domain_id", type=int, default=0)
    parser.add_argument("--side", choices=["left", "right"], default="right")
    parser.add_argument("--mode", choices=["monitor", "open", "close", "cycle"], default="monitor")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--open_q", type=float, default=5.40)
    parser.add_argument("--close_q", type=float, default=0.0)
    parser.add_argument("--kp", type=float, default=5.0)
    parser.add_argument("--kd", type=float, default=0.05)
    parser.add_argument("--max_step", type=float, default=0.18)
    args = parser.parse_args()

    ChannelFactoryInitialize(args.domain_id, args.interface)
    gripper = Dex1GripperDDS(
        side=args.side,
        open_q=args.open_q,
        close_q=args.close_q,
        kp=args.kp,
        kd=args.kd,
        max_step=args.max_step,
    )
    print(f"[DEX1] command topic: {gripper.command_topic}")
    print(f"[DEX1] state topic: {gripper.state_topic}")
    print("[DEX1] waiting for state ...")
    if not gripper.wait_state(3.0):
        print("[DEX1] no state received. Start dex1_1_gripper_server first.")
        return 2

    done = False

    def _handle_signal(_signum, _frame):
        nonlocal done
        done = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    deadline = time.monotonic() + max(0.0, args.duration)
    while not done and (args.duration <= 0 or time.monotonic() < deadline):
        if args.mode == "open":
            cmd = gripper.publish_grip(0.0)
        elif args.mode == "close":
            cmd = gripper.publish_grip(1.0)
        elif args.mode == "cycle":
            phase = 0.5 + 0.5 * np.sin(2.0 * np.pi * 0.2 * time.monotonic())
            cmd = gripper.publish_grip(float(phase))
        else:
            cmd = gripper.command_q if gripper.command_q is not None else float("nan")
        state_q = gripper.state.q if gripper.state is not None else float("nan")
        err = cmd - state_q
        print(f"[DEX1] side={args.side} state_q={state_q:+.3f} cmd_q={cmd:+.3f} err={err:+.3f}")
        time.sleep(0.2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
