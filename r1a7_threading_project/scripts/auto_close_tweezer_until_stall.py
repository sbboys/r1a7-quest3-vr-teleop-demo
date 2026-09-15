#!/usr/bin/env python3

import argparse
import json
import signal
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


RIGHT_GRIPPER_INDEX = 33


class AutoClose:
    def __init__(self, args):
        self.args = args
        self.low_state = None
        self.last_state_time = None
        self.running = True

        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()

        self.publisher = None
        self.subscriber = None

    def state_cb(self, msg):
        self.low_state = msg
        self.last_state_time = time.monotonic()

    def init(self):
        ChannelFactoryInitialize(
            self.args.domain_id,
            self.args.interface,
        )

        self.subscriber = ChannelSubscriber(
            "rt/lowstate",
            LowState_,
        )
        self.subscriber.Init(self.state_cb, 10)

        self.publisher = ChannelPublisher(
            "rt/lowcmd",
            LowCmd_,
        )
        self.publisher.Init()

        print("[AUTO CLOSE] DDS initialized")
        print(
            f"[AUTO CLOSE] right gripper motor index = "
            f"{RIGHT_GRIPPER_INDEX}"
        )

    def wait_state(self):
        print("[AUTO CLOSE] waiting for rt/lowstate ...")

        deadline = time.monotonic() + 5.0

        while self.running:
            if self.low_state is not None:
                q = self.get_q()
                print(f"[AUTO CLOSE] initial measured_q={q:.6f}")
                return

            if time.monotonic() > deadline:
                raise RuntimeError("timeout waiting for rt/lowstate")

            time.sleep(0.01)

    def get_motor(self):
        if self.low_state is None:
            raise RuntimeError("lowstate unavailable")

        return self.low_state.motor_state[RIGHT_GRIPPER_INDEX]

    def get_q(self):
        return float(self.get_motor().q)

    def get_dq(self):
        return float(self.get_motor().dq)

    def get_tau(self):
        return float(self.get_motor().tau_est)

    def init_stop_command(self):
        for motor in self.low_cmd.motor_cmd:
            motor.mode = 0
            motor.tau = 0.0
            motor.q = 0.0
            motor.dq = 0.0
            motor.kp = 0.0
            motor.kd = 0.0

    def publish(self, q_cmd, kp=None, kd=None):
        if self.low_state is None:
            return

        self.init_stop_command()

        if hasattr(self.low_cmd, "mode_pr"):
            self.low_cmd.mode_pr = 0

        if (
            hasattr(self.low_cmd, "mode_machine")
            and hasattr(self.low_state, "mode_machine")
        ):
            self.low_cmd.mode_machine = self.low_state.mode_machine

        motor = self.low_cmd.motor_cmd[RIGHT_GRIPPER_INDEX]

        motor.mode = 1
        motor.tau = 0.0
        motor.q = float(q_cmd)
        motor.dq = 0.0
        motor.kp = float(
            self.args.kp if kp is None else kp
        )
        motor.kd = float(
            self.args.kd if kd is None else kd
        )

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)

    def release(self):
        if self.publisher is None:
            return

        self.init_stop_command()
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)

        print("[AUTO CLOSE] released motor 33 lowcmd gains")

    def move_to_open(self):
        start_q = self.get_q()
        target = self.args.open_q

        print()
        print("[AUTO CLOSE] moving to tweezer OPEN command")
        print(
            f"[AUTO CLOSE] current={start_q:.6f} "
            f"target_cmd={target:.6f}"
        )

        cmd = start_q
        last = time.monotonic()
        last_print = 0.0

        while self.running:
            now = time.monotonic()
            dt = max(now - last, 1e-4)
            last = now

            delta = target - cmd
            step = self.args.open_speed * dt

            if abs(delta) <= step:
                cmd = target
            else:
                cmd += np.sign(delta) * step

            self.publish(cmd)

            if now - last_print >= 0.1:
                last_print = now
                print(
                    f"[OPEN] "
                    f"cmd={cmd:.6f} "
                    f"q={self.get_q():.6f} "
                    f"dq={self.get_dq():+.6f} "
                    f"tau={self.get_tau():+.4f}"
                )

            if abs(cmd - target) < 1e-5:
                break

            time.sleep(0.004)

        # Let the mechanism settle at the open command.
        settle_until = time.monotonic() + self.args.open_settle

        while self.running and time.monotonic() < settle_until:
            self.publish(target)
            time.sleep(0.004)

        print(
            "[AUTO CLOSE] OPEN settled: "
            f"cmd={target:.6f} "
            f"measured={self.get_q():.6f} "
            f"tau={self.get_tau():+.4f}"
        )

    def close_until_stall(self):
        cmd = float(self.args.open_q)

        q_open = self.get_q()
        tau_open = self.get_tau()

        history = deque()
        motion_seen = False
        stall_started = None
        last_print = 0.0
        last = time.monotonic()

        print()
        print("[AUTO CLOSE] starting automatic closing")
        print(
            f"[AUTO CLOSE] q decreases while closing; "
            f"hard software floor={self.args.min_q:.6f}"
        )
        print(
            f"[AUTO CLOSE] q_open_measured={q_open:.6f} "
            f"tau_open={tau_open:+.4f}"
        )

        while self.running:
            now = time.monotonic()
            dt = max(now - last, 1e-4)
            last = now

            # Closing direction: smaller q.
            cmd -= self.args.close_speed * dt

            if cmd < self.args.min_q:
                cmd = self.args.min_q

            self.publish(cmd)

            q = self.get_q()
            dq = self.get_dq()
            tau = self.get_tau()

            history.append((now, q))

            while history and (
                now - history[0][0] > self.args.stall_window
            ):
                history.popleft()

            total_motion = q_open - q

            if total_motion >= self.args.motion_required:
                motion_seen = True

            if len(history) >= 3:
                q_values = [x[1] for x in history]
                q_range = max(q_values) - min(q_values)
            else:
                q_range = float("inf")

            # Stall is only considered after:
            # 1. the gripper has actually moved in the closing direction;
            # 2. the commanded position is below measured position;
            # 3. measured q stays almost unchanged for the whole window.
            asking_to_close = (
                cmd < q - self.args.min_command_error
            )

            stalled_now = (
                motion_seen
                and asking_to_close
                and q_range <= self.args.stall_q_range
                and abs(dq) <= self.args.stall_dq
            )

            if stalled_now:
                if stall_started is None:
                    stall_started = now
            else:
                stall_started = None

            if now - last_print >= 0.05:
                last_print = now
                print(
                    f"[CLOSE] "
                    f"cmd={cmd:.6f} "
                    f"q={q:.6f} "
                    f"dq={dq:+.6f} "
                    f"tau={tau:+.4f} "
                    f"moved={total_motion:.6f} "
                    f"window={q_range:.6f}"
                )

            if (
                stall_started is not None
                and now - stall_started
                >= self.args.stall_confirm
            ):
                print()
                print("========================================")
                print("[AUTO CLOSE] STALL DETECTED")
                print(f"CLOSE_CMD_Q      = {cmd:.6f}")
                print(f"CLOSE_MEASURED_Q = {q:.6f}")
                print(f"CLOSE_DQ         = {dq:+.6f}")
                print(f"CLOSE_TAU_EST    = {tau:+.4f}")
                print(
                    f"TOTAL_Q_MOTION   = "
                    f"{total_motion:.6f}"
                )
                print("========================================")

                result = {
                    "timestamp": datetime.now().isoformat(),
                    "motor_index": RIGHT_GRIPPER_INDEX,
                    "open_cmd_q": self.args.open_q,
                    "open_measured_q": q_open,
                    "close_cmd_q": cmd,
                    "close_measured_q": q,
                    "close_dq": dq,
                    "open_tau_est": tau_open,
                    "close_tau_est": tau,
                    "total_measured_motion": total_motion,
                    "stall_q_range": q_range,
                }

                out = Path(self.args.output)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(
                    json.dumps(result, indent=2),
                    encoding="utf-8",
                )

                print(f"[AUTO CLOSE] saved: {out}")

                # Hold only briefly so the closed state can be observed.
                hold_until = (
                    time.monotonic()
                    + self.args.final_hold
                )

                while (
                    self.running
                    and time.monotonic() < hold_until
                ):
                    self.publish(cmd)
                    time.sleep(0.004)

                return

            if cmd <= self.args.min_q + 1e-8:
                print()
                print("[AUTO CLOSE] MIN_Q reached; holding command and waiting for settling.")

                hold_history = deque()
                hold_start = time.monotonic()
                last_hold_print = 0.0

                while self.running:
                    now_hold = time.monotonic()

                    self.publish(self.args.min_q)

                    q_hold = self.get_q()
                    dq_hold = self.get_dq()
                    tau_hold = self.get_tau()

                    hold_history.append((now_hold, q_hold))

                    while (
                        hold_history
                        and now_hold - hold_history[0][0]
                        > self.args.stall_window
                    ):
                        hold_history.popleft()

                    if len(hold_history) >= 3:
                        values = [x[1] for x in hold_history]
                        q_range_hold = max(values) - min(values)
                    else:
                        q_range_hold = float("inf")

                    if now_hold - last_hold_print >= 0.05:
                        last_hold_print = now_hold
                        print(
                            f"[HOLD ZERO] "
                            f"cmd={self.args.min_q:.6f} "
                            f"q={q_hold:.6f} "
                            f"dq={dq_hold:+.6f} "
                            f"tau={tau_hold:+.4f} "
                            f"window={q_range_hold:.6f}"
                        )

                    settled = (
                        q_range_hold <= self.args.stall_q_range
                        and abs(dq_hold) <= self.args.stall_dq
                    )

                    if settled and now_hold - hold_start >= 0.5:
                        print()
                        print("========================================")
                        print("[AUTO CLOSE] FINAL CLOSED STATE DETECTED")
                        print(f"CLOSE_CMD_Q      = {self.args.min_q:.6f}")
                        print(f"CLOSE_MEASURED_Q = {q_hold:.6f}")
                        print(f"CLOSE_DQ         = {dq_hold:+.6f}")
                        print(f"CLOSE_TAU_EST    = {tau_hold:+.4f}")
                        print("========================================")
                        return

                    if now_hold - hold_start >= 3.0:
                        print()
                        print("[AUTO CLOSE] zero-command hold timeout")
                        print(f"CLOSE_CMD_Q      = {self.args.min_q:.6f}")
                        print(f"CLOSE_MEASURED_Q = {q_hold:.6f}")
                        print(f"CLOSE_DQ         = {dq_hold:+.6f}")
                        print(f"CLOSE_TAU_EST    = {tau_hold:+.4f}")
                        return

                    time.sleep(0.004)

            time.sleep(0.004)

    def run(self):
        self.init()
        self.wait_state()

        self.move_to_open()

        print()
        print(
            "WARNING: automatic closing will now start."
        )
        print(
            "Keep fingers, wire and tools clear for this endpoint test."
        )
        print("Press Ctrl+C at any time to abort.")
        print()

        time.sleep(1.0)

        self.close_until_stall()


def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--interface", default="enp6s0")
    p.add_argument("--domain-id", type=int, default=0)

    # Known tweezer-open command from the current mounting.
    p.add_argument("--open-q", type=float, default=0.1585)

    # Do not command below this during the first endpoint search.
    p.add_argument("--min-q", type=float, default=0.0)

    p.add_argument("--open-speed", type=float, default=1.0)

    # This only spans about 0.16 q, so 0.10 q/s is a few seconds
    # from full open to the hard software floor.
    p.add_argument("--close-speed", type=float, default=0.10)

    p.add_argument("--open-settle", type=float, default=0.8)

    p.add_argument("--kp", type=float, default=8.0)
    p.add_argument("--kd", type=float, default=0.4)

    # Stall detection.
    p.add_argument("--stall-window", type=float, default=0.25)
    p.add_argument("--stall-confirm", type=float, default=0.20)
    p.add_argument("--stall-q-range", type=float, default=0.00035)
    p.add_argument("--stall-dq", type=float, default=0.01)

    # Do not declare stall before real closing motion was observed.
    p.add_argument("--motion-required", type=float, default=0.0015)

    # Command must genuinely be asking for further closing.
    p.add_argument("--min-command-error", type=float, default=0.003)

    p.add_argument("--final-hold", type=float, default=1.0)

    p.add_argument(
        "--output",
        default=(
            "r1a7_threading_project/docs/"
            "tweezer_close_stall_latest.json"
        ),
    )

    return p


def main():
    args = build_parser().parse_args()
    node = AutoClose(args)

    def stop_handler(_sig, _frame):
        node.running = False

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    print(
        "WARNING: REAL R1-A7 RIGHT GRIPPER MOTOR 33 "
        "WILL MOVE."
    )
    print(
        "This test searches for the tweezer closed endpoint."
    )

    try:
        node.run()
    finally:
        node.release()


if __name__ == "__main__":
    main()
