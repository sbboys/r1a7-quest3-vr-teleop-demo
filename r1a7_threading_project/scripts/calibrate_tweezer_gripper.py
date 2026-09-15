#!/usr/bin/env python3
"""Step the R1-A7 right gripper through fixed openings for tweezer calibration.

This script is intentionally narrow: it commands only the right DEX1 gripper
motor index 33 through rt/lowcmd and prints the measured gripper q. Measure the
tweezer tip gap externally at each hold point and write the result into
docs/gripper_tweezer_binding_notes_zh.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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


HOLD_INDICES = [13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]


class TweezerGripperCalibration:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.done = False
        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state: Optional[LowState_] = None
        self.publisher = None
        self.current_command_q: Optional[float] = None

    def init(self) -> None:
        ChannelFactoryInitialize(self.args.domain_id, self.args.interface)
        if self.args.enter_debug_mode:
            self._enter_debug_mode()
        self.subscriber = ChannelSubscriber(self.args.state_topic, LowState_)
        self.subscriber.Init(self._lowstate_handler, 10)
        self.publisher = ChannelPublisher(self.args.command_topic, LowCmd_)
        self.publisher.Init()
        print("[TWEEZER CAL] DDS initialized")
        print(f"[TWEEZER CAL] right gripper index: {self.args.right_index}")

    def _enter_debug_mode(self) -> None:
        msc = MotionSwitcherClient()
        msc.SetTimeout(2.0)
        msc.Init()
        status, result = msc.CheckMode()
        print(f"[TWEEZER CAL] motion_switcher CheckMode: status={status} result={result}")
        while result and result.get("name"):
            print("[TWEEZER CAL] releasing active mode:", result)
            msc.ReleaseMode()
            time.sleep(0.5)
            status, result = msc.CheckMode()
            print(f"[TWEEZER CAL] motion_switcher CheckMode: status={status} result={result}")

    def _lowstate_handler(self, msg: LowState_) -> None:
        self.low_state = msg

    def _read_right_q(self) -> Optional[float]:
        if self.low_state is None:
            return None
        if len(self.low_state.motor_state) <= self.args.right_index:
            raise RuntimeError(
                f"lowstate has {len(self.low_state.motor_state)} motors, requested {self.args.right_index}"
            )
        return float(self.low_state.motor_state[self.args.right_index].q)

    def _read_right_tau_est(self) -> float:
        if self.low_state is None:
            return float("nan")
        idx = self.args.right_index
        if idx >= len(self.low_state.motor_state):
            return float("nan")
        state = self.low_state.motor_state[idx]
        if not hasattr(state, "tau_est"):
            return float("nan")
        try:
            return float(state.tau_est)
        except (TypeError, ValueError):
            return float("nan")

    def _norm_to_q(self, norm: float) -> float:
        norm = min(1.0, max(0.0, norm))
        return self.args.right_close_q + norm * (self.args.right_open_q - self.args.right_close_q)

    def _init_low_cmd_stop(self) -> None:
        for motor in self.low_cmd.motor_cmd:
            motor.tau = 0.0
            motor.q = 0.0
            motor.dq = 0.0
            motor.kp = 0.0
            motor.kd = 0.0

    def _publish(self, target_q: float) -> None:
        assert self.publisher is not None
        self._init_low_cmd_stop()

        if self.low_state is not None:
            count = min(len(self.low_cmd.motor_cmd), len(self.low_state.motor_state))
            if hasattr(self.low_cmd, "mode_pr"):
                self.low_cmd.mode_pr = 0
            if hasattr(self.low_cmd, "mode_machine") and hasattr(self.low_state, "mode_machine"):
                self.low_cmd.mode_machine = self.low_state.mode_machine
            for idx in HOLD_INDICES:
                if idx >= count:
                    continue
                motor = self.low_cmd.motor_cmd[idx]
                motor.mode = 1
                motor.tau = 0.0
                motor.q = float(self.low_state.motor_state[idx].q)
                motor.dq = 0.0
                motor.kp = self.args.hold_kp
                motor.kd = self.args.hold_kd

        motor = self.low_cmd.motor_cmd[self.args.right_index]
        motor.mode = 1
        motor.tau = 0.0
        motor.q = float(target_q)
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
        print("[TWEEZER CAL] released lowcmd gains")

    def run(self) -> int:
        self.init()
        print("[TWEEZER CAL] waiting for rt/lowstate ...")
        while not self.done:
            q = self._read_right_q()
            if q is not None:
                self.current_command_q = q
                print(f"[TWEEZER CAL] initial right gripper q={q:.4f}")
                break
            time.sleep(0.02)

        if self.current_command_q is None:
            return 1

        try:
            if self.args.absolute_q:
                targets = [
                    (f"absolute_q={target_q:.4f}", float(target_q))
                    for target_q in self.args.absolute_q
                ]
            elif self.args.relative_offsets_q:
                baseline_q = self.current_command_q
                self._write_baseline(baseline_q)
                targets = [
                    (f"relative_q={offset:+.4f}", baseline_q + offset)
                    for offset in self.args.relative_offsets_q
                ]
            else:
                targets = [
                    (f"norm={norm:.3f}", self._norm_to_q(norm))
                    for norm in self.args.openings
                ]

            for label, target_q in targets:
                if self.done:
                    break
                print(f"[TWEEZER CAL] moving to {label}, target_q={target_q:.4f}")
                stage_start = time.monotonic()
                last_print = 0.0
                while not self.done:
                    now = time.monotonic()
                    elapsed = now - stage_start
                    measured_q = self._read_right_q()
                    tau_est = self._read_right_tau_est()
                    if measured_q is None:
                        time.sleep(0.02)
                        continue
                    max_delta = max(0.0, self.args.velocity_limit_q_per_s) * 0.02
                    self.current_command_q += float(np.clip(target_q - self.current_command_q, -max_delta, max_delta))
                    self._publish(self.current_command_q)
                    if now - last_print >= self.args.print_period:
                        last_print = now
                        print(
                            f"[TWEEZER CAL] hold {label} "
                            f"measured_q={measured_q:.4f} cmd_q={self.current_command_q:.4f} "
                            f"target_q={target_q:.4f} tau_est={tau_est:.4f}"
                        )
                    if elapsed >= self.args.hold_s and abs(self.current_command_q - target_q) < 1e-3:
                        break
                    time.sleep(0.02)

                if self.args.pause_for_measurement and not self.done:
                    measured_q = self._read_right_q()
                    print(
                        f"[TWEEZER CAL] OBSERVE NOW: {label}, "
                        f"measured_q={measured_q if measured_q is not None else float('nan'):.4f}. "
                        "Record tweezer motion, wire hold, slip, and deformation."
                    )
                    if sys.stdin.isatty():
                        input("Press Enter for next opening, or Ctrl+C to stop.")
            if self.args.hold_after_move and targets and not self.done:
                hold_q = float(self.current_command_q)
                print(
                    f"[TWEEZER CAL] HOLDING final q={hold_q:.4f}. "
                    "Press Ctrl+C to release."
                )
                last_print = 0.0
                while not self.done:
                    now = time.monotonic()
                    measured_q = self._read_right_q()
                    tau_est = self._read_right_tau_est()
                    self._publish(hold_q)

                    if now - last_print >= self.args.print_period:
                        last_print = now
                        if measured_q is not None:
                            print(
                                f"[TWEEZER CAL] HOLD "
                                f"measured_q={measured_q:.4f} "
                                f"cmd_q={hold_q:.4f} "
                                f"tau_est={tau_est:.4f}"
                            )

                    time.sleep(0.02)

            return 0
        finally:
            self._release()

    def _write_baseline(self, baseline_q: float) -> None:
        if not self.args.baseline_out:
            return
        out = Path(self.args.baseline_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "mode": "relative_from_contact",
            "right_gripper_index": self.args.right_index,
            "baseline_q": float(baseline_q),
            "relative_offsets_q": [float(v) for v in self.args.relative_offsets_q],
            "state_topic": self.args.state_topic,
            "command_topic": self.args.command_topic,
            "interface": self.args.interface,
        }
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"[TWEEZER CAL] wrote baseline: {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate tweezer opening using R1-A7 right gripper")
    parser.add_argument("--interface", default="enp6s0")
    parser.add_argument("--domain_id", type=int, default=0)
    parser.add_argument("--state_topic", default="rt/lowstate")
    parser.add_argument("--command_topic", default="rt/lowcmd")
    parser.add_argument("--right_index", type=int, default=33)
    parser.add_argument("--right_open_q", type=float, default=4.80)
    parser.add_argument("--right_close_q", type=float, default=-0.20)
    parser.add_argument("--openings", type=float, nargs="+", default=[0.65, 0.55, 0.45, 0.35, 0.25, 0.20])
    parser.add_argument(
        "--relative_offsets_q",
        type=float,
        nargs="+",
        default=None,
        help="Use current measured q as baseline and move to baseline + each offset.",
    )
    parser.add_argument("--baseline_out", default=None, help="Write relative baseline metadata as JSON")
    parser.add_argument("--velocity_limit_q_per_s", type=float, default=0.20)
    parser.add_argument("--kp", type=float, default=8.0)
    parser.add_argument("--kd", type=float, default=0.4)
    parser.add_argument("--hold_kp", type=float, default=10.0)
    parser.add_argument("--hold_kd", type=float, default=0.8)
    parser.add_argument(
        "--absolute_q",
        type=float,
        nargs="+",
        default=None,
        help="Command one or more absolute right-gripper q targets.",
    )
    parser.add_argument(
        "--hold-after-move",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Continuously hold the final target until interrupted.",
    )
    parser.add_argument("--hold_s", type=float, default=3.0)
    parser.add_argument("--print_period", type=float, default=0.25)
    parser.add_argument("--pause_for_measurement", action="store_true")
    parser.add_argument("--enter_debug_mode", action="store_true")
    parser.add_argument("--assume_yes", action="store_true")
    args = parser.parse_args()

    print("WARNING: This commands the real R1-A7 right gripper through rt/lowcmd.")
    print("Keep hands clear. Mount tweezers securely before running motor motion.")
    if args.relative_offsets_q:
        print(f"Relative q offsets from current baseline: {args.relative_offsets_q}")
    else:
        print(f"Openings: {args.openings}")
    if not args.assume_yes:
        if not sys.stdin.isatty():
            print("[TWEEZER CAL] interactive confirmation required")
            return 2
        if input("Type ENABLE to continue: ").strip() != "ENABLE":
            print("[TWEEZER CAL] aborted")
            return 2

    node = TweezerGripperCalibration(args)

    def _handle_signal(_signum, _frame):
        node.done = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    return node.run()


if __name__ == "__main__":
    raise SystemExit(main())
