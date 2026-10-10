#!/usr/bin/env python3

from pathlib import Path
import csv
import fcntl
import socket
import struct
import sys
import threading
import time

import numpy as np

from r1a7_lowcmd_guard import acquire_lowcmd_guard


# ============================================================
# R1-A7 R_SHOULDER_PITCH
# PRE-BUFFERED 500 Hz LOWCMD DIAGNOSTIC
#
# No Cartesian
# No IK
# No online trajectory generation
# No import from r1a7_cartesian_stream_control.py
# ============================================================

INTERFACE = "enp6s0"
EXPECTED_HOST_IP = "192.168.123.223"
DOMAIN_ID = 0

STATE_TOPIC = "rt/lowstate"
COMMAND_TOPIC = "rt/lowcmd"

FREQ_HZ = 500.0
DT = 1.0 / FREQ_HZ

ARM_INDICES = tuple(range(15, 29))
UPPER_BODY_INDICES = (13, *range(15, 31))

# arm_q index 7 = motor 22 = R_SHOULDER_PITCH
TEST_JOINT = 7

ARM_NAMES = (
    "L_SHOULDER_PITCH",
    "L_SHOULDER_ROLL",
    "L_SHOULDER_YAW",
    "L_ELBOW",
    "L_WRIST_ROLL",
    "L_WRIST_PITCH",
    "L_WRIST_YAW",
    "R_SHOULDER_PITCH",
    "R_SHOULDER_ROLL",
    "R_SHOULDER_YAW",
    "R_ELBOW",
    "R_WRIST_ROLL",
    "R_WRIST_PITCH",
    "R_WRIST_YAW",
)

ARM_LOWER = np.asarray(
    [
        -3.1416, -0.22689, -1.9199, -0.97564,
        -1.9199, -1.6144, -1.6144,
        -3.1416, -2.47849, -1.9199, -0.97564,
        -1.9199, -1.6144, -1.6144,
    ],
    dtype=float,
)

ARM_UPPER = np.asarray(
    [
        2.0944, 2.4784, 1.9199, 2.1852,
        1.9199, 1.6144, 1.6144,
        2.0944, 0.2268, 1.9199, 2.1852,
        1.9199, 1.6144, 1.6144,
    ],
    dtype=float,
)

JOINT_MARGIN_RAD = np.deg2rad(2.0)

# Official A7-style arm gains:
# shoulder/elbow = 100/2
# wrists         = 50/2
UPPER_KP = np.asarray(
    [
        60.0,
        100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 50.0,
        100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 50.0,
        50.0, 10.0,
    ],
    dtype=float,
)

UPPER_KD = np.asarray(
    [
        2.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        2.0, 0.1,
    ],
    dtype=float,
)


# ============================================================
# Test trajectory
#
# -0.5 deg in 1 second minimum-jerk.
#
# Full q trajectory is generated BEFORE LowCmd starts.
# ============================================================

DELTA_RAD = np.deg2rad(+0.5)

PRE_HOLD_S = 1.0
OUT_S = 1.0
MID_HOLD_S = 0.5
RETURN_S = 1.0
POST_HOLD_S = 1.0

MAX_TRACKING_ERROR_RAD = np.deg2rad(0.75)
MAX_STATE_AGE_S = 0.050

# If Python stops publishing for >20 ms, abort the experiment.
MAX_PUBLISH_GAP_S = 0.020


def validate_robot_interface(interface: str) -> str:
    path = Path("/sys/class/net") / interface

    if not path.exists():
        raise RuntimeError(
            f"network interface {interface!r} does not exist"
        )

    operstate = (
        path / "operstate"
    ).read_text(
        encoding="ascii"
    ).strip()

    carrier_path = path / "carrier"

    carrier = (
        carrier_path.read_text(
            encoding="ascii"
        ).strip()
        if carrier_path.exists()
        else "unknown"
    )

    if operstate != "up" or carrier == "0":
        raise RuntimeError(
            f"{interface} disconnected: "
            f"operstate={operstate}, carrier={carrier}"
        )

    with socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    ) as sock:

        request = struct.pack(
            "256s",
            interface[:15].encode("ascii"),
        )

        address = socket.inet_ntoa(
            fcntl.ioctl(
                sock.fileno(),
                0x8915,
                request,
            )[20:24]
        )

    return address


class StateBuffer:

    def __init__(self, crc):
        self.crc = crc
        self.lock = threading.Lock()
        self.message = None
        self.received_at = 0.0
        self.received_count = 0
        self.crc_errors = 0

    def callback(self, message):

        self.received_count += 1

        try:
            if int(message.crc) != int(
                self.crc.Crc(message)
            ):
                self.crc_errors += 1
                return
        except Exception:
            self.crc_errors += 1
            return

        with self.lock:
            self.message = message
            self.received_at = time.monotonic()

    def snapshot(self):

        with self.lock:

            if self.message is None:
                return None

            message = self.message

            upper_q = np.asarray(
                [
                    message.motor_state[i].q
                    for i in UPPER_BODY_INDICES
                ],
                dtype=float,
            )

            arm_q = np.asarray(
                [
                    message.motor_state[i].q
                    for i in ARM_INDICES
                ],
                dtype=float,
            )

            arm_dq = np.asarray(
                [
                    message.motor_state[i].dq
                    for i in ARM_INDICES
                ],
                dtype=float,
            )

            return (
                upper_q,
                arm_q,
                arm_dq,
                int(message.mode_machine),
                self.received_at,
            )


def check_debug_mode(MotionSwitcherClient):

    switcher = MotionSwitcherClient()
    switcher.SetTimeout(1.0)
    switcher.Init()

    for attempt in range(1, 4):

        status, result = (
            switcher.CheckMode()
        )

        if status == 0:

            mode_name = str(
                (result or {}).get(
                    "name",
                    "",
                )
            )

            if mode_name:

                raise RuntimeError(
                    "MotionSwitcher service active: "
                    f"{mode_name!r}"
                )

            print(
                "MotionSwitcher: debug mode confirmed "
                "(no active motion service).",
                flush=True,
            )

            return True

        print(
            "MotionSwitcher check "
            f"{attempt}/3 failed; retrying...",
            flush=True,
        )

        time.sleep(0.5)

    return False


def minimum_jerk(u):

    u = np.clip(
        np.asarray(u, dtype=float),
        0.0,
        1.0,
    )

    return (
        10.0 * u**3
        - 15.0 * u**4
        + 6.0 * u**5
    )


def make_trajectory(start_arm_q):

    start_arm_q = np.asarray(
        start_arm_q,
        dtype=float,
    ).copy()

    blocks = []
    phases = []

    def add_hold(seconds, offset, name):

        n = int(
            round(seconds * FREQ_HZ)
        )

        q = np.repeat(
            start_arm_q[None, :],
            n,
            axis=0,
        )

        q[:, TEST_JOINT] += offset

        blocks.append(q)
        phases.extend(
            [name] * n
        )

    def add_move(seconds, a, b, name):

        n = int(
            round(seconds * FREQ_HZ)
        )

        # Exact 500 Hz sample grid.
        # Endpoint is supplied by the next block.
        u = (
            np.arange(
                n,
                dtype=float,
            )
            / n
        )

        blend = minimum_jerk(u)

        q = np.repeat(
            start_arm_q[None, :],
            n,
            axis=0,
        )

        q[:, TEST_JOINT] += (
            a
            + (b - a) * blend
        )

        blocks.append(q)
        phases.extend(
            [name] * n
        )

    add_hold(
        PRE_HOLD_S,
        0.0,
        "pre_hold",
    )

    add_move(
        OUT_S,
        0.0,
        DELTA_RAD,
        "out",
    )

    add_hold(
        MID_HOLD_S,
        DELTA_RAD,
        "mid_hold",
    )

    add_move(
        RETURN_S,
        DELTA_RAD,
        0.0,
        "return",
    )

    add_hold(
        POST_HOLD_S,
        0.0,
        "post_hold",
    )

    trajectory = np.vstack(
        blocks
    )

    qdot_ref = np.zeros(
        len(trajectory),
        dtype=float,
    )

    qdot_ref[1:] = (
        np.diff(
            trajectory[
                :,
                TEST_JOINT,
            ]
        )
        / DT
    )

    if len(qdot_ref) > 1:
        qdot_ref[0] = qdot_ref[1]

    return (
        trajectory,
        phases,
        qdot_ref,
    )


def build_lowcmd(
    factory,
    crc,
    startup_upper_q,
    arm_ref,
    mode_machine,
):

    upper_ref = np.asarray(
        startup_upper_q,
        dtype=float,
    ).copy()

    upper_ref[1:15] = arm_ref

    message = factory()

    message.mode_pr = 0
    message.mode_machine = int(
        mode_machine
    )

    for local_index, motor_index in enumerate(
        UPPER_BODY_INDICES
    ):

        motor = (
            message.motor_cmd[
                motor_index
            ]
        )

        motor.mode = 1
        motor.q = float(
            upper_ref[
                local_index
            ]
        )

        # Official position control convention.
        motor.dq = 0.0
        motor.tau = 0.0

        motor.kp = float(
            UPPER_KP[
                local_index
            ]
        )

        motor.kd = float(
            UPPER_KD[
                local_index
            ]
        )

    message.crc = crc.Crc(
        message
    )

    return message


def write_csv(
    path,
    rows,
):

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "sample",
                "trajectory_time_s",
                "phase",
                "publish_dt_ms",
                "state_age_ms",
                "q_ref_rad",
                "qdot_ref_rad_s",
                "q_actual_rad",
                "q_error_deg",
                "raw_dq_rad_s",
            ]
        )

        writer.writerows(
            rows
        )


def measured_hold(
    publisher,
    factory,
    crc,
    state_buffer,
    startup_upper_q,
    duration_s=0.5,
):

    next_tick = time.monotonic()

    deadline = (
        next_tick
        + duration_s
    )

    while time.monotonic() < deadline:

        state = (
            state_buffer.snapshot()
        )

        if state is None:
            break

        (
            _,
            arm_q,
            _,
            mode_machine,
            received_at,
        ) = state

        if (
            time.monotonic()
            - received_at
        ) > MAX_STATE_AGE_S:
            break

        publisher.Write(
            build_lowcmd(
                factory,
                crc,
                startup_upper_q,
                arm_q,
                mode_machine,
            )
        )

        next_tick += DT

        sleep_s = (
            next_tick
            - time.monotonic()
        )

        if sleep_s > 0.0:
            time.sleep(
                sleep_s
            )


def main():

    interface_ip = (
        validate_robot_interface(
            INTERFACE
        )
    )

    print(
        f"Robot DDS interface: "
        f"{INTERFACE} ({interface_ip})"
    )

    if (
        interface_ip
        != EXPECTED_HOST_IP
    ):

        raise RuntimeError(
            "host IP mismatch: "
            f"expected {EXPECTED_HOST_IP}"
        )

    sdk_root = Path(
        "/home/robot/"
        "R1A7_VR_dual_arm_transfer_20260831_001/"
        "robot_dev/unitree_sdk2_python"
    )

    if not (
        sdk_root
        / "unitree_sdk2py"
    ).is_dir():

        raise RuntimeError(
            f"Unitree SDK not found: "
            f"{sdk_root}"
        )

    sys.path.insert(
        0,
        str(sdk_root),
    )

    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
        MotionSwitcherClient,
    )

    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
    )

    from unitree_sdk2py.idl.default import (
        unitree_hg_msg_dds__LowCmd_,
    )

    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
        LowCmd_,
        LowState_,
    )

    from unitree_sdk2py.utils.crc import CRC

    guard = acquire_lowcmd_guard(
        Path(__file__).name,
        topic=COMMAND_TOPIC,
    )

    publisher = None
    state_buffer = None
    startup_upper_q = None
    rows = []

    csv_path = Path(
        "/tmp/"
        "r1a7_r_shoulder_pitch_buffered_"
        + time.strftime(
            "%Y%m%d_%H%M%S"
        )
        + ".csv"
    )

    crc = CRC()

    try:

        ChannelFactoryInitialize(
            DOMAIN_ID,
            INTERFACE,
        )

        state_buffer = StateBuffer(
            crc
        )

        subscriber = ChannelSubscriber(
            STATE_TOPIC,
            LowState_,
        )

        # Latest state only.
        subscriber.Init(
            state_buffer.callback,
            1,
        )

        deadline = (
            time.monotonic()
            + 10.0
        )

        state = (
            state_buffer.snapshot()
        )

        while (
            state is None
            and time.monotonic()
            < deadline
        ):

            time.sleep(0.02)

            state = (
                state_buffer.snapshot()
            )

        if state is None:

            raise RuntimeError(
                "no valid rt/lowstate; "
                "no command sent"
            )

        (
            startup_upper_q,
            startup_arm_q,
            _,
            startup_mode_machine,
            received_at,
        ) = state

        if (
            time.monotonic()
            - received_at
        ) > MAX_STATE_AGE_S:

            raise RuntimeError(
                "startup LowState stale"
            )

        if not check_debug_mode(
            MotionSwitcherClient
        ):

            raise RuntimeError(
                "MotionSwitcher debug mode "
                "could not be confirmed"
            )

        target_q = (
            startup_arm_q[
                TEST_JOINT
            ]
            + DELTA_RAD
        )

        lower = (
            ARM_LOWER[
                TEST_JOINT
            ]
            + JOINT_MARGIN_RAD
        )

        upper = (
            ARM_UPPER[
                TEST_JOINT
            ]
            - JOINT_MARGIN_RAD
        )

        if not (
            lower
            <= target_q
            <= upper
        ):

            raise RuntimeError(
                "requested shoulder target "
                "outside soft joint limit"
            )

        # ====================================================
        # Entire q trajectory is pre-generated here.
        # No online trajectory computation during movement.
        # ====================================================

        (
            trajectory,
            phases,
            qdot_ref,
        ) = make_trajectory(
            startup_arm_q
        )

        total_s = (
            len(trajectory)
            * DT
        )

        print()
        print(
            "========================================"
        )
        print(
            "BUFFERED SINGLE-JOINT LOWCMD TEST"
        )
        print(
            "========================================"
        )
        print(
            "Joint:",
            ARM_NAMES[
                TEST_JOINT
            ],
            "(motor 22)",
        )
        print(
            f"Startup q = "
            f"{np.rad2deg(startup_arm_q[TEST_JOINT]):+.4f} deg"
        )
        print(
            f"Displacement = "
            f"{np.rad2deg(DELTA_RAD):+.4f} deg"
        )
        print(
            f"LowCmd = "
            f"{FREQ_HZ:.1f} Hz"
        )
        print(
            f"dt = "
            f"{DT * 1000.0:.1f} ms"
        )
        print(
            f"Preloaded samples = "
            f"{len(trajectory)}"
        )
        print(
            f"Total trajectory = "
            f"{total_s:.3f} s"
        )
        print(
            "motor.dq = 0"
        )
        print(
            "tau_ff = 0"
        )
        print(
            "Other 13 arm joints = "
            "startup measured hold"
        )
        print(
            "No Cartesian / IK / "
            "online q trajectory generation"
        )
        print(
            "CSV is buffered in RAM "
            "and written AFTER movement."
        )
        print(
            f"Tracking stop = "
            f"{np.rad2deg(MAX_TRACKING_ERROR_RAD):.3f} deg"
        )
        print(
            f"State stale stop = "
            f"{MAX_STATE_AGE_S * 1000:.0f} ms"
        )
        print(
            f"Publish-gap stop = "
            f"{MAX_PUBLISH_GAP_S * 1000:.0f} ms"
        )
        print(
            f"CSV: {csv_path}"
        )
        print(
            "========================================"
        )
        print()

        answer = input(
            "Type RUNBUFFER to start: "
        ).strip()

        if answer != "RUNBUFFER":

            print(
                "Cancelled; "
                "no LowCmd publisher created."
            )

            return 0

        publisher = ChannelPublisher(
            COMMAND_TOPIC,
            LowCmd_,
        )

        publisher.Init()

        print(
            "[START] buffered playback",
            flush=True,
        )

        next_tick = time.monotonic()
        last_publish = None
        overruns = 0

        for sample_index in range(
            len(trajectory)
        ):

            arm_ref = (
                trajectory[
                    sample_index
                ]
            )

            state = (
                state_buffer.snapshot()
            )

            if state is None:

                raise RuntimeError(
                    "LowState disappeared"
                )

            (
                _,
                arm_actual,
                arm_dq,
                mode_machine,
                received_at,
            ) = state

            now = time.monotonic()

            state_age = (
                now
                - received_at
            )

            if (
                state_age
                > MAX_STATE_AGE_S
            ):

                raise RuntimeError(
                    "LowState stale: "
                    f"{state_age * 1000:.3f} ms"
                )

            error_rad = (
                arm_ref[
                    TEST_JOINT
                ]
                - arm_actual[
                    TEST_JOINT
                ]
            )

            if (
                abs(error_rad)
                > MAX_TRACKING_ERROR_RAD
            ):

                raise RuntimeError(
                    "R_SHOULDER_PITCH "
                    "tracking error: "
                    f"{np.rad2deg(error_rad):+.4f} deg"
                )

            publisher.Write(
                build_lowcmd(
                    unitree_hg_msg_dds__LowCmd_,
                    crc,
                    startup_upper_q,
                    arm_ref,
                    mode_machine,
                )
            )

            published_at = (
                time.monotonic()
            )

            if last_publish is None:

                publish_dt = DT

            else:

                publish_dt = (
                    published_at
                    - last_publish
                )

                if (
                    publish_dt
                    > MAX_PUBLISH_GAP_S
                ):

                    raise RuntimeError(
                        "LowCmd publish gap: "
                        f"{publish_dt * 1000:.3f} ms"
                    )

            last_publish = (
                published_at
            )

            # RAM only. No disk I/O here.
            rows.append(
                [
                    sample_index,
                    f"{sample_index * DT:.6f}",
                    phases[
                        sample_index
                    ],
                    f"{publish_dt * 1000.0:.6f}",
                    f"{state_age * 1000.0:.6f}",
                    f"{arm_ref[TEST_JOINT]:.9f}",
                    f"{qdot_ref[sample_index]:.9f}",
                    f"{arm_actual[TEST_JOINT]:.9f}",
                    f"{np.rad2deg(error_rad):.6f}",
                    f"{arm_dq[TEST_JOINT]:.9f}",
                ]
            )

            next_tick += DT

            wait_s = (
                next_tick
                - time.monotonic()
            )

            if wait_s > 0.0:

                time.sleep(
                    wait_s
                )

            else:

                overruns += 1

                next_tick = (
                    time.monotonic()
                )

        print(
            "[PLAYBACK DONE]",
            flush=True,
        )

        measured_hold(
            publisher,
            unitree_hg_msg_dds__LowCmd_,
            crc,
            state_buffer,
            startup_upper_q,
            duration_s=0.5,
        )

        print(
            "[SAFE HOLD DONE]",
            flush=True,
        )

        write_csv(
            csv_path,
            rows,
        )

        dt_values = np.asarray(
            [
                float(row[3])
                for row in rows[1:]
            ],
            dtype=float,
        )

        error_values = np.asarray(
            [
                abs(float(row[8]))
                for row in rows
            ],
            dtype=float,
        )

        print()
        print(
            "========== TIMING SUMMARY =========="
        )

        if len(dt_values):

            print(
                f"publish dt median = "
                f"{np.median(dt_values):.4f} ms"
            )

            print(
                f"publish dt p99    = "
                f"{np.percentile(dt_values, 99):.4f} ms"
            )

            print(
                f"publish dt max    = "
                f"{np.max(dt_values):.4f} ms"
            )

        print(
            f"scheduler overruns = "
            f"{overruns}"
        )

        if len(error_values):

            print(
                f"max |tracking error| = "
                f"{np.max(error_values):.4f} deg"
            )

        print(
            f"CSV = {csv_path}"
        )

        print(
            "===================================="
        )

        return 0

    except KeyboardInterrupt:

        print(
            "\n[ABORT] Ctrl+C",
            flush=True,
        )

        return 130

    except Exception as exc:

        print(
            "\n[SAFETY STOP] "
            f"{type(exc).__name__}: "
            f"{exc}",
            flush=True,
        )

        return 2

    finally:

        if (
            publisher is not None
            and state_buffer is not None
            and startup_upper_q is not None
        ):

            try:

                measured_hold(
                    publisher,
                    unitree_hg_msg_dds__LowCmd_,
                    crc,
                    state_buffer,
                    startup_upper_q,
                    duration_s=0.3,
                )

            except Exception:
                pass

        if rows:

            try:

                write_csv(
                    csv_path,
                    rows,
                )

            except Exception:
                pass

        guard.close()


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
