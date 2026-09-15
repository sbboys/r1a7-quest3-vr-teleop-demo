#!/usr/bin/env python3

import argparse
import signal
import time

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


LEFT_GRIPPER_INDEX = 31


class LeftGripperCycle:
    def __init__(self, args):
        self.args = args
        self.state = None
        self.running = True
        self.crc = CRC()
        self.cmd = unitree_hg_msg_dds__LowCmd_()
        self.publisher = None

    def state_cb(self, msg):
        self.state = msg

    def initialize(self):
        ChannelFactoryInitialize(
            self.args.domain_id,
            self.args.interface,
        )

        subscriber = ChannelSubscriber(
            "rt/lowstate",
            LowState_,
        )
        subscriber.Init(self.state_cb, 10)

        self.publisher = ChannelPublisher(
            "rt/lowcmd",
            LowCmd_,
        )
        self.publisher.Init()

        print("[LEFT CYCLE] waiting for rt/lowstate ...")

        deadline = time.monotonic() + 5.0

        while self.running and self.state is None:
            if time.monotonic() > deadline:
                raise RuntimeError("timeout waiting for lowstate")
            time.sleep(0.01)

    def motor_state(self):
        return self.state.motor_state[LEFT_GRIPPER_INDEX]

    def reset_command(self):
        for motor in self.cmd.motor_cmd:
            motor.mode = 0
            motor.tau = 0.0
            motor.q = 0.0
            motor.dq = 0.0
            motor.kp = 0.0
            motor.kd = 0.0

    def publish(self, q_cmd):
        self.reset_command()

        if hasattr(self.cmd, "mode_pr"):
            self.cmd.mode_pr = 0

        if (
            hasattr(self.cmd, "mode_machine")
            and hasattr(self.state, "mode_machine")
        ):
            self.cmd.mode_machine = self.state.mode_machine

        motor = self.cmd.motor_cmd[LEFT_GRIPPER_INDEX]

        motor.mode = 1
        motor.tau = 0.0
        motor.q = float(q_cmd)
        motor.dq = 0.0
        motor.kp = self.args.kp
        motor.kd = self.args.kd

        self.cmd.crc = self.crc.Crc(self.cmd)
        self.publisher.Write(self.cmd)

    def release(self):
        if self.publisher is None:
            return

        self.reset_command()
        self.cmd.crc = self.crc.Crc(self.cmd)
        self.publisher.Write(self.cmd)

        print("[LEFT CYCLE] released motor 31")

    def move(self, target, label):
        q_cmd = float(self.motor_state().q)

        print()
        print(
            f"[LEFT CYCLE] {label}: "
            f"start={q_cmd:.6f} target={target:.6f}"
        )

        last_time = time.monotonic()
        last_print = 0.0

        while self.running:
            now = time.monotonic()
            dt = max(now - last_time, 0.001)
            last_time = now

            delta = target - q_cmd
            max_step = self.args.speed * dt

            if abs(delta) <= max_step:
                q_cmd = target
            else:
                q_cmd += max_step if delta > 0 else -max_step

            self.publish(q_cmd)

            state = self.motor_state()
            q = float(state.q)
            dq = float(state.dq)
            tau = float(state.tau_est)

            if now - last_print >= 0.1:
                last_print = now
                print(
                    f"[{label}] "
                    f"cmd={q_cmd:.5f} "
                    f"q={q:.5f} "
                    f"dq={dq:+.5f} "
                    f"tau={tau:+.4f}"
                )

            if abs(q_cmd - target) < 1e-5:
                break

            time.sleep(0.004)

        hold_end = time.monotonic() + self.args.hold

        while self.running and time.monotonic() < hold_end:
            self.publish(target)

            now = time.monotonic()

            if now - last_print >= 0.1:
                last_print = now

                state = self.motor_state()

                print(
                    f"[{label} HOLD] "
                    f"cmd={target:.5f} "
                    f"q={float(state.q):.5f} "
                    f"dq={float(state.dq):+.5f} "
                    f"tau={float(state.tau_est):+.4f}"
                )

            time.sleep(0.004)

    def run(self):
        self.initialize()

        open_q = float(self.motor_state().q)

        print()
        print("========================================")
        print("[LEFT CYCLE] REAL LEFT GRIPPER TEST")
        print(f"motor index = {LEFT_GRIPPER_INDEX}")
        print(f"OPEN_Q      = {open_q:.6f}")
        print(f"CLOSE_Q     = {self.args.close_q:.6f}")
        print(f"cycles      = {self.args.cycles}")
        print("========================================")
        print()
        print("Keep fingers and objects clear.")
        print("Ctrl+C stops immediately.")

        time.sleep(1.0)

        for i in range(self.args.cycles):
            if not self.running:
                break

            print()
            print(f"========== CYCLE {i + 1}/{self.args.cycles} ==========")

            self.move(
                self.args.close_q,
                "CLOSE",
            )

            if not self.running:
                break

            self.move(
                open_q,
                "OPEN",
            )

        print()
        print("[LEFT CYCLE] test finished")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--interface", default="enp6s0")
    parser.add_argument("--domain-id", type=int, default=0)

    parser.add_argument("--close-q", type=float, default=0.0)

    parser.add_argument(
        "--speed",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--hold",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--cycles",
        type=int,
        default=3,
    )

    parser.add_argument("--kp", type=float, default=8.0)
    parser.add_argument("--kd", type=float, default=0.4)

    args = parser.parse_args()

    node = LeftGripperCycle(args)

    def stop_handler(_sig, _frame):
        node.running = False

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    try:
        node.run()
    finally:
        node.release()


if __name__ == "__main__":
    main()
