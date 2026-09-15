#!/usr/bin/env python3

import argparse
import time

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


IDX = 31


class Probe:
    def __init__(self, args):
        self.args = args
        self.state = None
        self.crc = CRC()
        self.cmd = unitree_hg_msg_dds__LowCmd_()

    def cb(self, msg):
        self.state = msg

    def stop_cmd(self):
        for m in self.cmd.motor_cmd:
            m.mode = 0
            m.tau = 0.0
            m.q = 0.0
            m.dq = 0.0
            m.kp = 0.0
            m.kd = 0.0

    def publish(self, q):
        self.stop_cmd()

        if hasattr(self.cmd, "mode_pr"):
            self.cmd.mode_pr = 0

        if hasattr(self.cmd, "mode_machine") and hasattr(self.state, "mode_machine"):
            self.cmd.mode_machine = self.state.mode_machine

        m = self.cmd.motor_cmd[IDX]
        m.mode = 1
        m.q = float(q)
        m.dq = 0.0
        m.tau = 0.0
        m.kp = 8.0
        m.kd = 0.4

        self.cmd.crc = self.crc.Crc(self.cmd)
        self.pub.Write(self.cmd)

    def release(self):
        self.stop_cmd()
        self.cmd.crc = self.crc.Crc(self.cmd)
        self.pub.Write(self.cmd)

    def run(self):
        ChannelFactoryInitialize(0, self.args.interface)

        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(self.cb, 10)

        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()

        print("[LEFT PROBE] waiting lowstate...")

        while self.state is None:
            time.sleep(0.01)

        q0 = float(self.state.motor_state[IDX].q)
        target = q0 + self.args.delta

        print(f"[LEFT PROBE] initial q = {q0:.6f}")
        print(f"[LEFT PROBE] target q  = {target:.6f}")
        print(f"[LEFT PROBE] delta     = {self.args.delta:+.6f}")

        cmd = q0
        t0 = time.monotonic()
        last = t0

        try:
            while time.monotonic() - t0 < 0.6:
                now = time.monotonic()
                dt = now - last
                last = now

                direction = 1.0 if target > cmd else -1.0
                cmd += direction * min(
                    abs(target - cmd),
                    0.05 * max(dt, 0.001),
                )

                self.publish(cmd)

                m = self.state.motor_state[IDX]
                print(
                    f"cmd={cmd:.6f} "
                    f"q={float(m.q):.6f} "
                    f"dq={float(m.dq):+.6f} "
                    f"tau={float(m.tau_est):+.4f}"
                )

                time.sleep(0.05)

            time.sleep(0.5)

        finally:
            self.release()
            print("[LEFT PROBE] released")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--interface", default="enp6s0")
    p.add_argument("--delta", type=float, default=0.02)
    args = p.parse_args()

    Probe(args).run()


if __name__ == "__main__":
    main()
