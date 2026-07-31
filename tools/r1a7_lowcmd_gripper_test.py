#!/usr/bin/env python3
"""Conservative lowcmd test for internal R1-A7 gripper motors 31 and 33."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from typing import Optional

import numpy as np

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


HOLD_INDICES = [13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]


class LowcmdGripperTest:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.done = False
        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state: Optional[LowState_] = None
        self.lowstate_count = 0
        self.publisher = None
        self.subscriber = None
        self.home_q: Optional[np.ndarray] = None
        self.command_q: Optional[np.ndarray] = None

    def _enter_debug_mode(self) -> None:
        msc = MotionSwitcherClient()
        msc.SetTimeout(2.0)
        msc.Init()
        status, result = msc.CheckMode()
        print(f"[R1-A7 GRIPPER TEST] motion_switcher CheckMode: status={status} result={result}")
        while result and result.get("name"):
            print("[R1-A7 GRIPPER TEST] releasing active mode:", result)
            msc.ReleaseMode()
            time.sleep(0.5)
            status, result = msc.CheckMode()
            print(f"[R1-A7 GRIPPER TEST] motion_switcher CheckMode: status={status} result={result}")

    def init(self) -> None:
        ChannelFactoryInitialize(self.args.domain_id, self.args.interface)
        if self.args.enter_debug_mode:
            self._enter_debug_mode()
        self.subscriber = ChannelSubscriber(self.args.state_topic, LowState_)
        self.subscriber.Init(self._lowstate_handler, 10)
        self.publisher = ChannelPublisher(self.args.command_topic, LowCmd_)
        self.publisher.Init()
        print("[R1-A7 GRIPPER TEST] DDS initialized")
        print("[R1-A7 GRIPPER TEST] gripper indices:", self.args.left_index, self.args.right_index)

    def _lowstate_handler(self, msg: LowState_) -> None:
        self.low_state = msg
        self.lowstate_count += 1

    def _read_q(self) -> Optional[np.ndarray]:
        if self.low_state is None:
            return None
        ms = self.low_state.motor_state
        max_idx = max(self.args.left_index, self.args.right_index)
        if len(ms) <= max_idx:
            raise RuntimeError(f"lowstate has {len(ms)} motors, requested {max_idx}")
        return np.array([ms[self.args.left_index].q, ms[self.args.right_index].q], dtype=float)

    def _init_low_cmd_stop(self) -> None:
        for motor in self.low_cmd.motor_cmd:
            motor.tau = 0.0
            motor.q = 0.0
            motor.dq = 0.0
            motor.kp = 0.0
            motor.kd = 0.0

    def _publish(self, target_q: np.ndarray) -> None:
        assert self.publisher is not None
        self._init_low_cmd_stop()
        if self.low_state is not None:
            count = min(len(self.low_cmd.motor_cmd), len(self.low_state.motor_state))
            if hasattr(self.low_cmd, "mode_pr"):
                self.low_cmd.mode_pr = 0
            if hasattr(self.low_cmd, "mode_machine") and hasattr(self.low_state, "mode_machine"):
                self.low_cmd.mode_machine = self.low_state.mode_machine
            for i in HOLD_INDICES:
                if i >= count:
                    continue
                motor = self.low_cmd.motor_cmd[i]
                motor.mode = 1
                motor.tau = 0.0
                motor.q = float(self.low_state.motor_state[i].q)
                motor.dq = 0.0
                motor.kp = self.args.hold_kp
                motor.kd = self.args.hold_kd

        for idx, q in zip([self.args.left_index, self.args.right_index], target_q):
            motor = self.low_cmd.motor_cmd[idx]
            motor.mode = 1
            motor.tau = 0.0
            motor.q = float(q)
            motor.dq = 0.0
            motor.kp = self.args.kp
            motor.kd = self.args.kd

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)

    def _release(self) -> None:
        if self.publisher is None:
            return
        self._init_low_cmd_stop()
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)
        print("[R1-A7 GRIPPER TEST] released lowcmd gains")

    def run(self) -> int:
        self.init()
        print("[R1-A7 GRIPPER TEST] waiting lowstate ...")
        while not self.done and self.home_q is None:
            q = self._read_q()
            if q is not None:
                if self.args.left_home_q is not None or self.args.right_home_q is not None:
                    self.home_q = np.array(
                        [
                            q[0] if self.args.left_home_q is None else self.args.left_home_q,
                            q[1] if self.args.right_home_q is None else self.args.right_home_q,
                        ],
                        dtype=float,
                    )
                else:
                    self.home_q = q.copy()
                self.command_q = q.copy()
                print("[R1-A7 GRIPPER TEST] state q:", np.round(q, 4).tolist())
                print("[R1-A7 GRIPPER TEST] home q:", np.round(self.home_q, 4).tolist())
                break
            time.sleep(0.02)
        if self.home_q is None:
            return 1

        cycle_offsets = [
            np.array([self.args.left_offset, self.args.right_offset], dtype=float),
            np.zeros(2, dtype=float),
        ]
        cycle_names = ["open_offset", "home"]
        if self.args.cycles <= 0:
            offsets = cycle_offsets
            names = cycle_names
        else:
            offsets = []
            names = []
            if self.args.start_home:
                offsets.append(np.zeros(2, dtype=float))
                names.append("initial_home")
            for cycle in range(self.args.cycles):
                for name, offset in zip(cycle_names, cycle_offsets):
                    offsets.append(offset)
                    names.append(f"cycle{cycle + 1}_{name}")
        try:
            start = time.monotonic()
            stage = 0
            last_print = 0.0
            while not self.done:
                now = time.monotonic()
                if self.args.cycles <= 0 and self.args.duration > 0 and now - start >= self.args.duration:
                    break
                if now - start >= (stage + 1) * self.args.stage_time and stage < len(offsets) - 1:
                    stage += 1
                    print("[R1-A7 GRIPPER TEST] switching target:", names[stage])
                if self.args.cycles > 0 and stage >= len(offsets) - 1 and now - start >= len(offsets) * self.args.stage_time:
                    break
                state_q = self._read_q()
                if state_q is None:
                    time.sleep(0.02)
                    continue
                target_q = self.home_q + offsets[stage]
                max_delta = max(0.0, self.args.velocity_limit) * 0.02
                self.command_q = self.command_q + np.clip(target_q - self.command_q, -max_delta, max_delta)
                self._publish(self.command_q)
                if now - last_print >= self.args.print_period:
                    last_print = now
                    print(
                        f"[R1-A7 GRIPPER TEST] {names[stage]} "
                        f"state={np.round(state_q, 4).tolist()} "
                        f"cmd={np.round(self.command_q, 4).tolist()} "
                        f"target={np.round(target_q, 4).tolist()}"
                    )
                time.sleep(0.02)
        finally:
            self._release()
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Test R1-A7 internal gripper motors through rt/lowcmd")
    parser.add_argument("--interface", default="enx9c69d37d0967")
    parser.add_argument("--domain_id", type=int, default=0)
    parser.add_argument("--state_topic", default="rt/lowstate")
    parser.add_argument("--command_topic", default="rt/lowcmd")
    parser.add_argument("--left_index", type=int, default=31)
    parser.add_argument("--right_index", type=int, default=33)
    parser.add_argument("--left_offset", type=float, default=0.35)
    parser.add_argument("--right_offset", type=float, default=0.35)
    parser.add_argument("--left_home_q", type=float)
    parser.add_argument("--right_home_q", type=float)
    parser.add_argument("--velocity_limit", type=float, default=0.25)
    parser.add_argument("--kp", type=float, default=8.0)
    parser.add_argument("--kd", type=float, default=0.4)
    parser.add_argument("--hold_kp", type=float, default=10.0)
    parser.add_argument("--hold_kd", type=float, default=0.8)
    parser.add_argument("--stage_time", type=float, default=4.0)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--cycles", type=int, default=0, help="number of open/home cycles; 0 uses duration")
    parser.add_argument("--start_home", action="store_true", help="move to home before the first open cycle")
    parser.add_argument("--print_period", type=float, default=0.25)
    parser.add_argument("--enter_debug_mode", action="store_true")
    parser.add_argument("--assume_yes", action="store_true")
    args = parser.parse_args()

    print("WARNING: This directly commands R1-A7 internal gripper motors through rt/lowcmd.")
    print("Keep fingers and objects clear of both grippers.")
    print(f"Test motion: [{args.left_index}, {args.right_index}] offsets [{args.left_offset}, {args.right_offset}] then home.")
    if not args.assume_yes:
        if not sys.stdin.isatty():
            print("[R1-A7 GRIPPER TEST] interactive confirmation required")
            return 2
        if input("Type ENABLE to continue: ").strip() != "ENABLE":
            print("[R1-A7 GRIPPER TEST] aborted")
            return 2

    node = LowcmdGripperTest(args)

    def _handle_signal(_signum, _frame):
        node.done = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    return node.run()


if __name__ == "__main__":
    raise SystemExit(main())
