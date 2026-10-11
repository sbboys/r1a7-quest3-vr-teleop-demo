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


INTERFACE = "enp6s0"
EXPECTED_HOST_IP = "192.168.123.223"
DOMAIN_ID = 0

STATE_TOPIC = "rt/lowstate"
COMMAND_TOPIC = "rt/lowcmd"

FREQ_HZ = 500.0
DT = 1.0 / FREQ_HZ

ARM_INDICES = tuple(range(15, 29))
UPPER_BODY_INDICES = (13, *range(15, 31))

# arm index 7 = motor 22 = R_SHOULDER_PITCH
TEST_JOINT = 7

# ------------------------------------------------------------
# Frozen-q torque diagnostic
# ------------------------------------------------------------

MAX_TAU_NM = -0.30

PRE_HOLD_S = 1.0
RAMP_UP_S = 2.0
TAU_HOLD_S = 1.0
RAMP_DOWN_S = 2.0
POST_HOLD_S = 1.0

# Stop the torque experiment as soon as clear motion is seen.
BREAKAWAY_STOP_RAD = np.deg2rad(0.10)

# Absolute final safety bound.
MAX_Q_DEVIATION_RAD = np.deg2rad(0.25)

MAX_STATE_AGE_S = 0.050
MAX_PUBLISH_GAP_S = 0.020

# Torque removal slew after abort/breakaway.
SAFE_TAU_SLEW_NM_S = 2.0


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


def validate_robot_interface(interface):

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
        self.crc_errors = 0

    def callback(self, message):

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

            m = self.message

            upper_q = np.asarray(
                [
                    m.motor_state[i].q
                    for i in UPPER_BODY_INDICES
                ],
                dtype=float,
            )

            arm_q = np.asarray(
                [
                    m.motor_state[i].q
                    for i in ARM_INDICES
                ],
                dtype=float,
            )

            arm_dq = np.asarray(
                [
                    m.motor_state[i].dq
                    for i in ARM_INDICES
                ],
                dtype=float,
            )

            arm_tau_est = np.asarray(
                [
                    m.motor_state[i].tau_est
                    for i in ARM_INDICES
                ],
                dtype=float,
            )

            return (
                upper_q,
                arm_q,
                arm_dq,
                arm_tau_est,
                int(m.mode_machine),
                self.received_at,
            )


def check_debug_mode(MotionSwitcherClient):

    switcher = MotionSwitcherClient()
    switcher.SetTimeout(1.0)
    switcher.Init()

    for attempt in range(1, 4):

        status, result = switcher.CheckMode()

        if status == 0:

            mode_name = str(
                (result or {}).get(
                    "name",
                    "",
                )
            )

            if mode_name:
                raise RuntimeError(
                    "MotionSwitcher active: "
                    f"{mode_name!r}"
                )

            print(
                "MotionSwitcher: debug mode confirmed "
                "(no active motion service).",
                flush=True,
            )

            return True

        print(
            f"MotionSwitcher check {attempt}/3 failed",
            flush=True,
        )

        time.sleep(0.5)

    return False


def make_tau_profile():

    values = []
    phases = []

    def hold(seconds, tau, name):

        n = int(
            round(
                seconds * FREQ_HZ
            )
        )

        values.extend(
            [float(tau)] * n
        )

        phases.extend(
            [name] * n
        )

    def ramp(seconds, start, end, name):

        n = int(
            round(
                seconds * FREQ_HZ
            )
        )

        # Endpoint supplied by following phase.
        x = np.arange(
            n,
            dtype=float,
        ) / n

        tau = (
            start
            + (end - start) * x
        )

        values.extend(
            tau.tolist()
        )

        phases.extend(
            [name] * n
        )

    hold(
        PRE_HOLD_S,
        0.0,
        "pre_hold",
    )

    ramp(
        RAMP_UP_S,
        0.0,
        MAX_TAU_NM,
        "ramp_up",
    )

    hold(
        TAU_HOLD_S,
        MAX_TAU_NM,
        "tau_hold",
    )

    ramp(
        RAMP_DOWN_S,
        MAX_TAU_NM,
        0.0,
        "ramp_down",
    )

    hold(
        POST_HOLD_S,
        0.0,
        "post_hold",
    )

    return (
        np.asarray(
            values,
            dtype=float,
        ),
        phases,
    )


def build_lowcmd(
    factory,
    crc,
    upper_ref,
    arm_tau_ref,
    mode_machine,
):

    upper_ref = np.asarray(
        upper_ref,
        dtype=float,
    ).reshape(
        len(UPPER_BODY_INDICES)
    )

    tau_ref = np.asarray(
        arm_tau_ref,
        dtype=float,
    ).reshape(14)

    msg = factory()

    msg.mode_pr = 0
    msg.mode_machine = int(
        mode_machine
    )

    for local_index, motor_index in enumerate(
        UPPER_BODY_INDICES
    ):

        motor = msg.motor_cmd[
            motor_index
        ]

        motor.mode = 1
        motor.q = float(
            upper_ref[
                local_index
            ]
        )

        motor.dq = 0.0

        if 1 <= local_index <= 14:

            motor.tau = float(
                tau_ref[
                    local_index - 1
                ]
            )

        else:

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

    msg.crc = crc.Crc(
        msg
    )

    return msg


def ramp_tau_to_zero(
    publisher,
    factory,
    crc,
    state_buffer,
    current_tau,
):

    tau = float(
        current_tau
    )

    max_step = (
        SAFE_TAU_SLEW_NM_S
        * DT
    )

    next_tick = time.monotonic()

    while abs(tau) > 1.0e-9:

        state = state_buffer.snapshot()

        if state is None:
            break

        (
            upper_q,
            arm_q,
            _,
            _,
            mode_machine,
            received_at,
        ) = state

        if (
            time.monotonic()
            - received_at
        ) > MAX_STATE_AGE_S:
            break

        # Hold the CURRENT measured posture while tau is removed.
        hold_upper = upper_q.copy()
        hold_upper[1:15] = arm_q

        if tau > 0.0:
            tau = max(
                0.0,
                tau - max_step,
            )
        else:
            tau = min(
                0.0,
                tau + max_step,
            )

        arm_tau = np.zeros(
            14,
            dtype=float,
        )

        arm_tau[
            TEST_JOINT
        ] = tau

        publisher.Write(
            build_lowcmd(
                factory,
                crc,
                hold_upper,
                arm_tau,
                mode_machine,
            )
        )

        next_tick += DT

        wait_s = (
            next_tick
            - time.monotonic()
        )

        if wait_s > 0:
            time.sleep(
                wait_s
            )


def measured_hold_zero_tau(
    publisher,
    factory,
    crc,
    state_buffer,
    duration_s=0.5,
):

    next_tick = time.monotonic()
    deadline = (
        next_tick
        + duration_s
    )

    zero_tau = np.zeros(
        14,
        dtype=float,
    )

    while time.monotonic() < deadline:

        state = state_buffer.snapshot()

        if state is None:
            break

        (
            upper_q,
            arm_q,
            _,
            _,
            mode_machine,
            received_at,
        ) = state

        if (
            time.monotonic()
            - received_at
        ) > MAX_STATE_AGE_S:
            break

        hold_upper = upper_q.copy()
        hold_upper[1:15] = arm_q

        publisher.Write(
            build_lowcmd(
                factory,
                crc,
                hold_upper,
                zero_tau,
                mode_machine,
            )
        )

        next_tick += DT

        wait_s = (
            next_tick
            - time.monotonic()
        )

        if wait_s > 0:
            time.sleep(
                wait_s
            )


def write_csv(path, rows):

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "sample",
                "time_s",
                "phase",
                "publish_dt_ms",
                "state_age_ms",
                "tau_ff_sent_nm",
                "q_ref_rad",
                "q_actual_rad",
                "q_delta_deg",
                "q_error_deg",
                "raw_dq_rad_s",
                "tau_est",
            ]
        )

        writer.writerows(
            rows
        )


def main():

    interface_ip = validate_robot_interface(
        INTERFACE
    )

    print(
        f"Robot DDS interface: "
        f"{INTERFACE} ({interface_ip})"
    )

    if interface_ip != EXPECTED_HOST_IP:
        raise RuntimeError(
            "host IP mismatch: "
            f"expected {EXPECTED_HOST_IP}"
        )

    sdk_root = Path(
        "/home/robot/"
        "R1A7_VR_dual_arm_transfer_20260831_001/"
        "robot_dev/unitree_sdk2_python"
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
    rows = []

    last_tau_sent = 0.0

    csv_path = Path(
        "/tmp/"
        "r1a7_r_shoulder_pitch_frozen_q_tau_"
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

        subscriber.Init(
            state_buffer.callback,
            1,
        )

        deadline = (
            time.monotonic()
            + 10.0
        )

        state = state_buffer.snapshot()

        while (
            state is None
            and time.monotonic()
            < deadline
        ):

            time.sleep(
                0.02
            )

            state = state_buffer.snapshot()

        if state is None:
            raise RuntimeError(
                "no valid rt/lowstate"
            )

        (
            startup_upper_q,
            startup_arm_q,
            _,
            startup_tau_est,
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

        tau_profile, phases = (
            make_tau_profile()
        )

        total_s = (
            len(tau_profile)
            * DT
        )

        print()
        print(
            "========================================"
        )
        print(
            "FROZEN-Q TORQUE RAMP DIAGNOSTIC"
        )
        print(
            "========================================"
        )
        print(
            "Joint: R_SHOULDER_PITCH (motor 22)"
        )
        print(
            f"Startup q = "
            f"{np.rad2deg(startup_arm_q[TEST_JOINT]):+.4f} deg"
        )
        print(
            "q_ref = startup measured q, FIXED"
        )
        print(
            "motor.dq = 0"
        )
        print(
            f"tau_ff = 0 -> {MAX_TAU_NM:+.3f} Nm -> 0"
        )
        print(
            f"LowCmd = {FREQ_HZ:.1f} Hz"
        )
        print(
            f"Total planned time = {total_s:.3f} s"
        )
        print(
            f"Breakaway early-stop = "
            f"{np.rad2deg(BREAKAWAY_STOP_RAD):.3f} deg"
        )
        print(
            f"Absolute displacement stop = "
            f"{np.rad2deg(MAX_Q_DEVIATION_RAD):.3f} deg"
        )
        print(
            "Other 13 arm joints = startup measured hold"
        )
        print(
            "No IK / Cartesian / gravity model"
        )
        print(
            "CSV buffered in RAM during motion"
        )
        print(
            f"Startup tau_est motor22 = "
            f"{startup_tau_est[TEST_JOINT]:+.6f}"
        )
        print(
            "NOTE: tau_est is diagnostic telemetry, "
            "NOT measured motor current."
        )
        print(
            f"CSV: {csv_path}"
        )
        print(
            "========================================"
        )
        print()

        answer = input(
            "Type RUNTORQUE to start: "
        ).strip()

        if answer != "RUNTORQUE":

            print(
                "Cancelled; no LowCmd publisher created."
            )

            return 0

        publisher = ChannelPublisher(
            COMMAND_TOPIC,
            LowCmd_,
        )

        publisher.Init()

        fixed_upper_q = (
            startup_upper_q.copy()
        )

        fixed_upper_q[1:15] = (
            startup_arm_q
        )

        print(
            "[START] frozen-q torque ramp",
            flush=True,
        )

        next_tick = (
            time.monotonic()
        )

        last_publish = None
        overruns = 0
        breakaway_detected = False

        for sample_index, tau_cmd in enumerate(
            tau_profile
        ):

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
                arm_tau_est,
                mode_machine,
                received_at,
            ) = state

            now = (
                time.monotonic()
            )

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

            q_delta = (
                arm_actual[
                    TEST_JOINT
                ]
                - startup_arm_q[
                    TEST_JOINT
                ]
            )

            if (
                abs(q_delta)
                > MAX_Q_DEVIATION_RAD
            ):

                raise RuntimeError(
                    "R_SHOULDER_PITCH displacement "
                    "safety stop: "
                    f"{np.rad2deg(q_delta):+.4f} deg"
                )

            arm_tau = np.zeros(
                14,
                dtype=float,
            )

            arm_tau[
                TEST_JOINT
            ] = float(
                tau_cmd
            )

            publisher.Write(
                build_lowcmd(
                    unitree_hg_msg_dds__LowCmd_,
                    crc,
                    fixed_upper_q,
                    arm_tau,
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

            last_publish = published_at
            last_tau_sent = float(
                tau_cmd
            )

            q_error = (
                startup_arm_q[
                    TEST_JOINT
                ]
                - arm_actual[
                    TEST_JOINT
                ]
            )

            rows.append(
                [
                    sample_index,
                    f"{sample_index * DT:.6f}",
                    phases[
                        sample_index
                    ],
                    f"{publish_dt * 1000.0:.6f}",
                    f"{state_age * 1000.0:.6f}",
                    f"{tau_cmd:.9f}",
                    f"{startup_arm_q[TEST_JOINT]:.9f}",
                    f"{arm_actual[TEST_JOINT]:.9f}",
                    f"{np.rad2deg(q_delta):.6f}",
                    f"{np.rad2deg(q_error):.6f}",
                    f"{arm_dq[TEST_JOINT]:.9f}",
                    f"{arm_tau_est[TEST_JOINT]:.9f}",
                ]
            )

            if (
                abs(q_delta)
                >= BREAKAWAY_STOP_RAD
            ):

                print(
                    "[BREAKAWAY] "
                    f"q_delta={np.rad2deg(q_delta):+.4f} deg "
                    f"at tau_ff={tau_cmd:+.4f} Nm",
                    flush=True,
                )

                breakaway_detected = True
                break

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
            "[TORQUE PROFILE DONE]"
            if not breakaway_detected
            else "[EARLY STOP AFTER BREAKAWAY]",
            flush=True,
        )

        # Remove torque smoothly while holding CURRENT measured q.
        ramp_tau_to_zero(
            publisher,
            unitree_hg_msg_dds__LowCmd_,
            crc,
            state_buffer,
            last_tau_sent,
        )

        measured_hold_zero_tau(
            publisher,
            unitree_hg_msg_dds__LowCmd_,
            crc,
            state_buffer,
            duration_s=0.5,
        )

        print(
            "[ZERO-TAU MEASURED HOLD DONE]",
            flush=True,
        )

        write_csv(
            csv_path,
            rows,
        )

        pubdt = np.asarray(
            [
                float(r[3])
                for r in rows[1:]
            ],
            dtype=float,
        )

        qdelta = np.asarray(
            [
                float(r[8])
                for r in rows
            ],
            dtype=float,
        )

        tau_sent = np.asarray(
            [
                float(r[5])
                for r in rows
            ],
            dtype=float,
        )

        print()
        print(
            "========== SUMMARY =========="
        )

        if len(pubdt):

            print(
                f"publish dt median = "
                f"{np.median(pubdt):.4f} ms"
            )

            print(
                f"publish dt p99    = "
                f"{np.percentile(pubdt, 99):.4f} ms"
            )

            print(
                f"publish dt max    = "
                f"{np.max(pubdt):.4f} ms"
            )

        print(
            f"scheduler overruns = {overruns}"
        )

        print(
            f"max +q displacement = "
            f"{np.max(qdelta):+.6f} deg"
        )

        print(
            f"min q displacement  = "
            f"{np.min(qdelta):+.6f} deg"
        )

        for threshold in (
            0.02,
            0.05,
            0.10,
        ):

            hit = np.where(
                np.abs(qdelta)
                >= threshold
            )[0]

            if len(hit):

                i = int(
                    hit[0]
                )

                print(
                    f"first |dq| >= {threshold:.2f} deg: "
                    f"t={float(rows[i][1]):.3f}s "
                    f"tau_ff={tau_sent[i]:+.5f} Nm "
                    f"dq={qdelta[i]:+.5f} deg"
                )

            else:

                print(
                    f"first |dq| >= {threshold:.2f} deg: NONE"
                )

        print(
            f"CSV = {csv_path}"
        )

        print(
            "============================="
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
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        return 2

    finally:

        if (
            publisher is not None
            and state_buffer is not None
        ):

            try:

                ramp_tau_to_zero(
                    publisher,
                    unitree_hg_msg_dds__LowCmd_,
                    crc,
                    state_buffer,
                    last_tau_sent,
                )

                measured_hold_zero_tau(
                    publisher,
                    unitree_hg_msg_dds__LowCmd_,
                    crc,
                    state_buffer,
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
