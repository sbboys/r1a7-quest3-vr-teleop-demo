#!/usr/bin/env python3

from pathlib import Path
from collections import deque
import csv
import fcntl
import os
import socket
import struct
import sys
import threading
import time

import numpy as np
import pinocchio as pin

from r1a7_lowcmd_guard import acquire_lowcmd_guard


# ============================================================
# DDS / robot
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

# arm index 7 = motor 22
TEST_JOINT = 7

XR_ROOT = Path(
    "/home/robot/"
    "R1A7_VR_dual_arm_transfer_20260831_001/"
    "robot_dev/xr_teleoperate"
)

SDK_ROOT = XR_ROOT.parent / "unitree_sdk2_python"


# ============================================================
# AUTO_BREAKAWAY_V1
#
# Positive R_SHOULDER_PITCH only.
#
# q_ref moves slowly.
# Extra assist starts at zero and increases ONLY when needed.
#
# ============================================================

DESIRED_SPEED_DEG_S = 0.08
DESIRED_SPEED_RAD_S = np.deg2rad(
    DESIRED_SPEED_DEG_S
)

# q_ref may never get too far in front of actual.
QREF_MAX_LEAD_DEG = 0.15
QREF_MAX_LEAD_RAD = np.deg2rad(
    QREF_MAX_LEAD_DEG
)

# Total nominal q_ref travel available to the experiment.
QREF_MAX_TRAVEL_DEG = 0.20
QREF_MAX_TRAVEL_RAD = np.deg2rad(
    QREF_MAX_TRAVEL_DEG
)

# Once actual reaches this, experiment is successful.
EVAL_TARGET_DEG = 0.15
EVAL_TARGET_RAD = np.deg2rad(
    EVAL_TARGET_DEG
)

# Absolute physical displacement stop.
HARD_DISPLACEMENT_DEG = 0.25
HARD_DISPLACEMENT_RAD = np.deg2rad(
    HARD_DISPLACEMENT_DEG
)

# Additional tracking safety.
MAX_TRACKING_ERROR_DEG = 0.25
MAX_TRACKING_ERROR_RAD = np.deg2rad(
    MAX_TRACKING_ERROR_DEG
)


# ============================================================
# Gravity + adaptive assist
# ============================================================

GRAVITY_SCALE = 0.50

# Total shoulder tau_ff may NEVER exceed this.
TOTAL_TAU_FF_CAP_NM = 0.40

# Extra assist search rate.
#
# 0 -> 0.03 Nm takes 1 second.
# Search is deliberately slow.
ASSIST_SEARCH_RATE_NM_S = 0.03

# Even if gravity leaves more headroom, V1 will not request
# more than this additional assist.
ASSIST_ABSOLUTE_MAX_NM = 0.12

# When motion becomes too fast, back assist down slowly.
ASSIST_DECAY_RATE_NM_S = 0.05

# Global torque slew during safe shutdown.
SAFE_TAU_SLEW_NM_S = 2.0


# ============================================================
# Actual-motion detector
#
# Do NOT use raw dq as the primary low-speed detector.
# Use finite-difference actual q over ~200 ms.
# ============================================================

VELOCITY_WINDOW_S = 0.20

BREAKAWAY_SPEED_DEG_S = 0.04
BREAKAWAY_CONFIRM_S = 0.10

STALL_SPEED_DEG_S = 0.02
STALL_CONFIRM_S = 0.25

FAST_SPEED_DEG_S = 0.20

MAX_TEST_S = 8.0
CAP_NO_MOTION_TIMEOUT_S = 2.0

MAX_STATE_AGE_S = 0.050
MAX_PUBLISH_GAP_S = 0.020


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

    switcher.SetTimeout(
        1.0
    )

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

        time.sleep(
            0.5
        )

    return False


def compute_startup_gravity_tau(
    startup_arm_q,
):

    sys.path.insert(
        0,
        str(XR_ROOT),
    )

    from teleop.robot_control.robot_arm_ik import (
        R1A7_ArmIK,
    )

    old_cwd = os.getcwd()

    try:

        os.chdir(
            XR_ROOT / "teleop"
        )

        ik = R1A7_ArmIK(
            Unit_Test=False,
            Visualization=False,
        )

    finally:

        os.chdir(
            old_cwd
        )

    model = ik.reduced_robot.model
    data = ik.reduced_robot.data

    if model.nq != 14 or model.nv != 14:

        raise RuntimeError(
            "unexpected R1A7 reduced model size"
        )

    names = list(
        model.names
    )

    if (
        len(names) <= 8
        or names[8]
        != "right_shoulder_pitch_joint"
    ):

        raise RuntimeError(
            "unexpected R1A7 model joint order"
        )

    zero_v = np.zeros(
        model.nv,
        dtype=float,
    )

    zero_a = np.zeros(
        model.nv,
        dtype=float,
    )

    raw = np.asarray(
        pin.rnea(
            model,
            data,
            startup_arm_q,
            zero_v,
            zero_a,
        ),
        dtype=float,
    ).reshape(14)

    shoulder_raw = float(
        raw[TEST_JOINT]
    )

    shoulder_base = float(
        np.clip(
            shoulder_raw
            * GRAVITY_SCALE,
            -TOTAL_TAU_FF_CAP_NM,
            TOTAL_TAU_FF_CAP_NM,
        )
    )

    return (
        shoulder_raw,
        shoulder_base,
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

    if not np.all(
        np.isfinite(
            upper_ref
        )
    ):

        raise ValueError(
            "non-finite q reference"
        )

    if not np.all(
        np.isfinite(
            tau_ref
        )
    ):

        raise ValueError(
            "non-finite tau reference"
        )

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


def actual_velocity_from_history(
    history,
):

    if len(history) < 2:
        return 0.0

    newest_t, newest_q = history[-1]

    target_t = (
        newest_t
        - VELOCITY_WINDOW_S
    )

    old_t, old_q = history[0]

    for t, q in history:

        old_t = t
        old_q = q

        if t >= target_t:
            break

    dt = (
        newest_t
        - old_t
    )

    if dt < 0.10:
        return 0.0

    return (
        newest_q
        - old_q
    ) / dt


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

    next_tick = (
        time.monotonic()
    )

    while abs(tau) > 1.0e-9:

        state = (
            state_buffer.snapshot()
        )

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

        hold_upper = (
            upper_q.copy()
        )

        hold_upper[1:15] = (
            arm_q
        )

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

        if wait_s > 0.0:

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

    next_tick = (
        time.monotonic()
    )

    deadline = (
        next_tick
        + duration_s
    )

    zero_tau = np.zeros(
        14,
        dtype=float,
    )

    while time.monotonic() < deadline:

        state = (
            state_buffer.snapshot()
        )

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

        hold_upper = (
            upper_q.copy()
        )

        hold_upper[1:15] = (
            arm_q
        )

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

        if wait_s > 0.0:

            time.sleep(
                wait_s
            )


def write_csv(
    path,
    rows,
):

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "sample",
                "time_s",
                "controller_state",
                "publish_dt_ms",
                "state_age_ms",
                "q_ref_rad",
                "q_actual_rad",
                "q_delta_deg",
                "tracking_error_deg",
                "actual_vel_200ms_deg_s",
                "gravity_raw_nm",
                "gravity_base_nm",
                "assist_nm",
                "total_tau_ff_nm",
                "raw_dq_rad_s",
                "tau_est",
            ]
        )

        writer.writerows(
            rows
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

    sys.path.insert(
        0,
        str(SDK_ROOT),
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

    from unitree_sdk2py.utils.crc import (
        CRC,
    )

    guard = acquire_lowcmd_guard(
        Path(__file__).name,
        topic=COMMAND_TOPIC,
    )

    crc = CRC()

    publisher = None
    state_buffer = None

    rows = []

    total_tau_sent = 0.0

    csv_path = Path(
        "/tmp/"
        "r1a7_r_shoulder_pitch_auto_breakaway_"
        + time.strftime(
            "%Y%m%d_%H%M%S"
        )
        + ".csv"
    )

    try:

        ChannelFactoryInitialize(
            DOMAIN_ID,
            INTERFACE,
        )

        state_buffer = (
            StateBuffer(
                crc
            )
        )

        subscriber = (
            ChannelSubscriber(
                STATE_TOPIC,
                LowState_,
            )
        )

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

            time.sleep(
                0.02
            )

            state = (
                state_buffer.snapshot()
            )

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

        print(
            "[MODEL] computing startup gravity...",
            flush=True,
        )

        (
            gravity_raw_nm,
            gravity_base_nm,
        ) = compute_startup_gravity_tau(
            startup_arm_q
        )

        available_headroom = (
            TOTAL_TAU_FF_CAP_NM
            - gravity_base_nm
        )

        assist_max_nm = min(
            ASSIST_ABSOLUTE_MAX_NM,
            max(
                0.0,
                available_headroom,
            ),
        )

        if assist_max_nm < 0.01:

            raise RuntimeError(
                "insufficient positive tau_ff headroom: "
                f"{assist_max_nm:+.4f} Nm"
            )

        print()
        print(
            "========================================"
        )
        print(
            "AUTO_BREAKAWAY_V1"
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
            f"Desired q_ref speed = "
            f"+{DESIRED_SPEED_DEG_S:.3f} deg/s"
        )

        print(
            f"q_ref max lead = "
            f"{QREF_MAX_LEAD_DEG:.3f} deg"
        )

        print(
            f"Actual evaluation target = "
            f"+{EVAL_TARGET_DEG:.3f} deg"
        )

        print(
            f"Actual HARD STOP = "
            f"{HARD_DISPLACEMENT_DEG:.3f} deg"
        )

        print(
            f"gravity raw = "
            f"{gravity_raw_nm:+.6f} Nm"
        )

        print(
            f"gravity x0.50 = "
            f"{gravity_base_nm:+.6f} Nm"
        )

        print(
            "extra assist = 0 -> automatic increase"
        )

        print(
            f"assist search rate = "
            f"{ASSIST_SEARCH_RATE_NM_S:.3f} Nm/s"
        )

        print(
            f"assist max = "
            f"{assist_max_nm:.6f} Nm"
        )

        print(
            f"TOTAL tau_ff HARD CAP = "
            f"+{TOTAL_TAU_FF_CAP_NM:.3f} Nm"
        )

        print(
            f"breakaway detection = "
            f"{BREAKAWAY_SPEED_DEG_S:.3f} deg/s "
            f"for {BREAKAWAY_CONFIRM_S:.2f} s"
        )

        print(
            f"stall detection = "
            f"{STALL_SPEED_DEG_S:.3f} deg/s "
            f"for {STALL_CONFIRM_S:.2f} s"
        )

        print(
            f"LowCmd = {FREQ_HZ:.1f} Hz"
        )

        print(
            "Kp/Kd unchanged = 100/2"
        )

        print(
            "Other 13 arm joints = startup measured hold"
        )

        print(
            "CSV buffered in RAM during motion"
        )

        print(
            "tau_est is telemetry only; "
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
            "Type RUNAUTO to start: "
        ).strip()

        if answer != "RUNAUTO":

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

        fixed_upper_q = (
            startup_upper_q.copy()
        )

        fixed_upper_q[1:15] = (
            startup_arm_q
        )

        start_q = float(
            startup_arm_q[
                TEST_JOINT
            ]
        )

        assist_nm = 0.0

        controller_state = "SEEK"

        history = deque()

        breakaway_confirm = 0.0
        stall_confirm = 0.0

        breakaway_assist_nm = None
        breakaway_total_tau_nm = None
        breakaway_time_s = None

        cap_reached_at = None

        next_tick = (
            time.monotonic()
        )

        start_time = (
            next_tick
        )

        last_publish = None

        overruns = 0

        status = "TIMEOUT"

        sample_index = 0

        max_tracking_error = 0.0
        max_actual_speed_deg_s = 0.0

        print(
            "[START] automatic breakaway search",
            flush=True,
        )

        while True:

            now = (
                time.monotonic()
            )

            elapsed = (
                now
                - start_time
            )

            if elapsed >= MAX_TEST_S:

                status = "MAX_TEST_TIME"
                break

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
                    f"{state_age * 1000.0:.3f} ms"
                )

            q_actual = float(
                arm_actual[
                    TEST_JOINT
                ]
            )

            actual_delta = (
                q_actual
                - start_q
            )

            if (
                abs(actual_delta)
                >= HARD_DISPLACEMENT_RAD
            ):

                status = "HARD_DISPLACEMENT_STOP"
                break

            # -----------------------------------------------
            # Desired reference moves slowly, but may not run
            # farther than QREF_MAX_LEAD ahead of actual.
            # -----------------------------------------------

            nominal_ref = (
                start_q
                + min(
                    DESIRED_SPEED_RAD_S
                    * elapsed,
                    QREF_MAX_TRAVEL_RAD,
                )
            )

            q_ref = min(
                nominal_ref,
                q_actual
                + QREF_MAX_LEAD_RAD,
            )

            tracking_error = (
                q_ref
                - q_actual
            )

            max_tracking_error = max(
                max_tracking_error,
                abs(
                    tracking_error
                ),
            )

            if (
                abs(tracking_error)
                > MAX_TRACKING_ERROR_RAD
            ):

                status = "TRACKING_ERROR_STOP"
                break

            # -----------------------------------------------
            # Actual velocity from q finite difference.
            # -----------------------------------------------

            history.append(
                (
                    now,
                    q_actual,
                )
            )

            while (
                history
                and now
                - history[0][0]
                > VELOCITY_WINDOW_S
                + 0.05
            ):

                history.popleft()

            actual_vel_rad_s = (
                actual_velocity_from_history(
                    history
                )
            )

            actual_vel_deg_s = float(
                np.rad2deg(
                    actual_vel_rad_s
                )
            )

            max_actual_speed_deg_s = max(
                max_actual_speed_deg_s,
                actual_vel_deg_s,
            )

            # -----------------------------------------------
            # Adaptive assist state machine.
            # -----------------------------------------------

            if controller_state == "SEEK":

                if (
                    actual_vel_deg_s
                    >= BREAKAWAY_SPEED_DEG_S
                    and actual_delta
                    >= np.deg2rad(0.005)
                ):

                    breakaway_confirm += DT

                else:

                    breakaway_confirm = 0.0

                if (
                    breakaway_confirm
                    >= BREAKAWAY_CONFIRM_S
                ):

                    controller_state = "RUN"

                    breakaway_assist_nm = float(
                        assist_nm
                    )

                    breakaway_total_tau_nm = float(
                        np.clip(
                            gravity_base_nm
                            + assist_nm,
                            -TOTAL_TAU_FF_CAP_NM,
                            TOTAL_TAU_FF_CAP_NM,
                        )
                    )

                    breakaway_time_s = (
                        elapsed
                    )

                    print(
                        "[BREAKAWAY CONFIRMED] "
                        f"t={elapsed:.3f}s "
                        f"actual={np.rad2deg(actual_delta):+.4f}deg "
                        f"vel={actual_vel_deg_s:+.4f}deg/s "
                        f"assist={assist_nm:+.5f}Nm "
                        f"total_tau={breakaway_total_tau_nm:+.5f}Nm",
                        flush=True,
                    )

                    stall_confirm = 0.0

                else:

                    assist_nm = min(
                        assist_max_nm,
                        assist_nm
                        + ASSIST_SEARCH_RATE_NM_S
                        * DT,
                    )

                    if (
                        assist_nm
                        >= assist_max_nm
                        - 1.0e-9
                    ):

                        if cap_reached_at is None:

                            cap_reached_at = (
                                elapsed
                            )

                        elif (
                            elapsed
                            - cap_reached_at
                            >= CAP_NO_MOTION_TIMEOUT_S
                        ):

                            status = "NO_BREAKAWAY_AT_ASSIST_CAP"
                            break

                    else:

                        cap_reached_at = None

            elif controller_state == "RUN":

                # If it becomes stuck again, resume SEEK.
                if (
                    actual_vel_deg_s
                    < STALL_SPEED_DEG_S
                ):

                    stall_confirm += DT

                else:

                    stall_confirm = 0.0

                if (
                    stall_confirm
                    >= STALL_CONFIRM_S
                ):

                    controller_state = "SEEK"

                    breakaway_confirm = 0.0
                    stall_confirm = 0.0

                    print(
                        "[RE-STALL] "
                        f"t={elapsed:.3f}s "
                        f"actual={np.rad2deg(actual_delta):+.4f}deg "
                        f"assist={assist_nm:+.5f}Nm; "
                        "resuming slow search",
                        flush=True,
                    )

                # Catch-up too fast: gently reduce assist.
                elif (
                    actual_vel_deg_s
                    > FAST_SPEED_DEG_S
                ):

                    assist_nm = max(
                        0.0,
                        assist_nm
                        - ASSIST_DECAY_RATE_NM_S
                        * DT,
                    )

            # -----------------------------------------------
            # Success: actual arm has moved enough to evaluate
            # whether motion remains continuous.
            # -----------------------------------------------

            if (
                actual_delta
                >= EVAL_TARGET_RAD
            ):

                status = "EVAL_TARGET_REACHED"
                break

            total_tau_sent = float(
                np.clip(
                    gravity_base_nm
                    + assist_nm,
                    -TOTAL_TAU_FF_CAP_NM,
                    TOTAL_TAU_FF_CAP_NM,
                )
            )

            arm_ref = (
                startup_arm_q.copy()
            )

            arm_ref[
                TEST_JOINT
            ] = q_ref

            upper_ref = (
                fixed_upper_q.copy()
            )

            upper_ref[1:15] = (
                arm_ref
            )

            arm_tau = np.zeros(
                14,
                dtype=float,
            )

            arm_tau[
                TEST_JOINT
            ] = total_tau_sent

            publisher.Write(
                build_lowcmd(
                    unitree_hg_msg_dds__LowCmd_,
                    crc,
                    upper_ref,
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
                        f"{publish_dt * 1000.0:.3f} ms"
                    )

            last_publish = (
                published_at
            )

            rows.append(
                [
                    sample_index,
                    f"{elapsed:.6f}",
                    controller_state,
                    f"{publish_dt * 1000.0:.6f}",
                    f"{state_age * 1000.0:.6f}",
                    f"{q_ref:.9f}",
                    f"{q_actual:.9f}",
                    f"{np.rad2deg(actual_delta):.6f}",
                    f"{np.rad2deg(tracking_error):.6f}",
                    f"{actual_vel_deg_s:.6f}",
                    f"{gravity_raw_nm:.9f}",
                    f"{gravity_base_nm:.9f}",
                    f"{assist_nm:.9f}",
                    f"{total_tau_sent:.9f}",
                    f"{arm_dq[TEST_JOINT]:.9f}",
                    f"{arm_tau_est[TEST_JOINT]:.9f}",
                ]
            )

            sample_index += 1

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
            f"[AUTO TEST END] {status}",
            flush=True,
        )

        # ====================================================
        # Safe exit:
        # hold CURRENT measured posture while total tau_ff
        # is slewed to zero.
        # ====================================================

        ramp_tau_to_zero(
            publisher,
            unitree_hg_msg_dds__LowCmd_,
            crc,
            state_buffer,
            total_tau_sent,
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

        state = (
            state_buffer.snapshot()
        )

        final_delta_deg = float("nan")

        if state is not None:

            final_arm_q = state[1]

            final_delta_deg = float(
                np.rad2deg(
                    final_arm_q[
                        TEST_JOINT
                    ]
                    - start_q
                )
            )

        pubdt = np.asarray(
            [
                float(r[3])
                for r in rows[1:]
            ],
            dtype=float,
        )

        print()
        print(
            "========== AUTO_BREAKAWAY SUMMARY =========="
        )

        print(
            f"status = {status}"
        )

        print(
            f"startup q = "
            f"{np.rad2deg(start_q):+.6f} deg"
        )

        print(
            f"final actual displacement = "
            f"{final_delta_deg:+.6f} deg"
        )

        if breakaway_assist_nm is None:

            print(
                "breakaway = NOT CONFIRMED"
            )

        else:

            print(
                f"breakaway time = "
                f"{breakaway_time_s:.3f} s"
            )

            print(
                f"breakaway assist = "
                f"{breakaway_assist_nm:+.6f} Nm"
            )

            print(
                f"breakaway total tau_ff = "
                f"{breakaway_total_tau_nm:+.6f} Nm"
            )

        print(
            f"final assist = "
            f"{assist_nm:+.6f} Nm"
        )

        print(
            f"max finite-diff speed = "
            f"{max_actual_speed_deg_s:+.6f} deg/s"
        )

        print(
            f"max tracking error = "
            f"{np.rad2deg(max_tracking_error):.6f} deg"
        )

        if len(pubdt):

            print(
                f"publish dt median = "
                f"{np.median(pubdt):.4f} ms"
            )

            print(
                f"publish dt p99 = "
                f"{np.percentile(pubdt, 99):.4f} ms"
            )

            print(
                f"publish dt max = "
                f"{np.max(pubdt):.4f} ms"
            )

        print(
            f"scheduler overruns = {overruns}"
        )

        print(
            f"CSV = {csv_path}"
        )

        print(
            "============================================"
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
                    total_tau_sent,
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
