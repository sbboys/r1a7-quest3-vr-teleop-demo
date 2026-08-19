#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
ORBBEC_EXAMPLES = ROOT / "doc" / "orbbec_gemini_336l" / "pyorbbecsdk_src" / "examples"
PYORBBEC_SITE = Path.home() / ".local" / "lib" / "python3.10" / "site-packages"
if PYORBBEC_SITE.is_dir() and str(PYORBBEC_SITE) not in sys.path:
    sys.path.append(str(PYORBBEC_SITE))
sys.path.insert(0, str(ORBBEC_EXAMPLES))

from pyorbbecsdk import OBFormat, OBSensorType, Pipeline  # noqa: E402
from utils import frame_to_bgr_image  # noqa: E402
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber  # noqa: E402
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_  # noqa: E402
from unitree_sdk2py.utils.crc import CRC  # noqa: E402


RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
RIGHT_ARM_INDICES = [22, 23, 24, 25, 26, 27, 28]


def detect_tag_pose(image, camera_matrix, dist_coeffs, tag_size_m, tag_id):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    ids_flat = ids.flatten().tolist()
    if tag_id not in ids_flat:
        return None
    idx = ids_flat.index(tag_id)
    image_points = corners[idx].reshape(4, 2).astype(np.float64)
    half = tag_size_m / 2.0
    object_points = np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float64,
    )
    ok, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    t_tag_to_camera = np.eye(4)
    t_tag_to_camera[:3, :3] = rmat
    t_tag_to_camera[:3, 3] = tvec.reshape(3)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    reproj = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    return t_tag_to_camera, float(reproj.mean())


def observe_tag_in_base(calibration_path: Path, samples: int, timeout_s: float, tag_id: int, tag_size_m: float):
    calib = json.loads(calibration_path.read_text(encoding="utf-8"))
    t_base_camera = np.array(calib["T_base_camera"], dtype=float)
    pipeline = Pipeline()
    profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = profile_list.get_video_stream_profile(0, 0, OBFormat.RGB, 0)
    config = __import__("pyorbbecsdk").Config()
    config.enable_stream(color_profile)
    pipeline.start(config)
    try:
        cam = pipeline.get_camera_param()
        intr = cam.rgb_intrinsic
        dist = cam.rgb_distortion
        camera_matrix = np.array([[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]])
        dist_coeffs = np.array([dist.k1, dist.k2, dist.p1, dist.p2, dist.k3], dtype=np.float64)
        observed = []
        reproj_errors = []
        deadline = time.monotonic() + timeout_s
        while len(observed) < samples and time.monotonic() < deadline:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
            image = frame_to_bgr_image(color_frame)
            if image is None:
                continue
            detected = detect_tag_pose(image, camera_matrix, dist_coeffs, tag_size_m, tag_id)
            if detected is None:
                continue
            t_tag_to_camera, reproj = detected
            observed.append(t_base_camera @ t_tag_to_camera)
            reproj_errors.append(reproj)
        if not observed:
            raise RuntimeError("no AprilTag observations collected")
        trans = np.array([m[:3, 3] for m in observed])
        t_base_tag = observed[-1].copy()
        t_base_tag[:3, 3] = trans.mean(axis=0)
        return t_base_tag, float(np.mean(reproj_errors)), trans.std(axis=0)
    finally:
        pipeline.stop()


class LowStateReader:
    def __init__(self):
        self.msg = None
        self.t = 0.0

    def cb(self, msg):
        self.msg = msg
        self.t = time.monotonic()


def q_model_from_lowstate(model, lowstate, waist_index: int = 13):
    q = pin.neutral(model)
    motor_state = lowstate.motor_state
    q[model.joints[model.getJointId("waist_yaw_joint")].idx_q] = float(motor_state[waist_index].q)
    for name, idx in zip(RIGHT_ARM_JOINTS, RIGHT_ARM_INDICES):
        q[model.joints[model.getJointId(name)].idx_q] = float(motor_state[idx].q)
    return q


def right_q_from_model(model, q):
    return np.array([q[model.joints[model.getJointId(n)].idx_q] for n in RIGHT_ARM_JOINTS], dtype=float)


def frame_pose(model, data, q, frame: str, offset: np.ndarray):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return data.oMf[model.getFrameId(frame)] * pin.SE3(np.eye(3), offset)


def frame_jacobian(model, data, q, frame: str, offset: np.ndarray):
    fid = model.getFrameId(frame)
    jac = pin.computeFrameJacobian(model, data, q, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    if np.linalg.norm(offset) > 1e-12:
        jac = jac.copy()
        jac[:3, :] += -pin.skew(offset) @ jac[3:, :]
    return jac


def solve_right_probe_position(model, q0, target_pos, tcp_offset, max_iters=120):
    data = model.createData()
    q = q0.copy()
    active_v = np.array([model.joints[model.getJointId(n)].idx_v for n in RIGHT_ARM_JOINTS], dtype=int)
    lower = model.lowerPositionLimit.copy()
    upper = model.upperPositionLimit.copy()
    for it in range(max_iters):
        pose = frame_pose(model, data, q, "right_wrist_yaw_link", tcp_offset)
        err = target_pos - pose.translation
        if np.linalg.norm(err) < 0.003:
            return q, float(np.linalg.norm(err)), it
        j = frame_jacobian(model, data, q, "right_wrist_yaw_link", tcp_offset)[:3, active_v]
        lhs = j @ j.T + (0.04**2) * np.eye(3)
        dq_active = j.T @ np.linalg.solve(lhs, err)
        dq = np.zeros(model.nv)
        dq[active_v] = np.clip(0.45 * dq_active, -0.035, 0.035)
        q = pin.integrate(model, q, dq)
        q = np.clip(q, lower, upper)
    pose = frame_pose(model, data, q, "right_wrist_yaw_link", tcp_offset)
    return q, float(np.linalg.norm(target_pos - pose.translation)), max_iters


def publish_right_arm(lowstate, publisher, crc, command_q7, waist_index, fixed_waist_q, kp, kd, hold_kp, hold_kd):
    low_cmd = unitree_hg_msg_dds__LowCmd_()
    for motor in low_cmd.motor_cmd:
        motor.tau = 0.0
        motor.q = 0.0
        motor.dq = 0.0
        motor.kp = 0.0
        motor.kd = 0.0
    if hasattr(low_cmd, "mode_machine") and hasattr(lowstate, "mode_machine"):
        low_cmd.mode_machine = lowstate.mode_machine
    if hasattr(low_cmd, "mode_pr"):
        low_cmd.mode_pr = 0
    count = min(len(low_cmd.motor_cmd), len(lowstate.motor_state))
    for i in [waist_index, *RIGHT_ARM_INDICES]:
        if i >= count:
            continue
        motor = low_cmd.motor_cmd[i]
        motor.mode = 1
        motor.tau = 0.0
        motor.q = fixed_waist_q if i == waist_index else float(lowstate.motor_state[i].q)
        motor.dq = 0.0
        motor.kp = hold_kp
        motor.kd = hold_kd
    for idx, q in zip(RIGHT_ARM_INDICES, command_q7):
        motor = low_cmd.motor_cmd[idx]
        motor.mode = 1
        motor.tau = 0.0
        motor.q = float(q)
        motor.dq = 0.0
        motor.kp = kp
        motor.kd = kd
    low_cmd.crc = crc.Crc(low_cmd)
    publisher.Write(low_cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely move R1-A7 right probe above observed AprilTag center")
    parser.add_argument("--interface", default="enp6s0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--state-topic", default="rt/lowstate")
    parser.add_argument("--command-topic", default="rt/lowcmd")
    parser.add_argument("--calibration", default="calibration/r1a7_base_camera_from_apriltag.json")
    parser.add_argument("--probe-tcp", default="data/r1a7_teach/20260819_212932/probe_tcp_result.json")
    parser.add_argument("--urdf", default="/home/robot/IsaacLab/bolt_nut_assembly/R1_A7_official/A7.urdf")
    parser.add_argument("--tag-size-m", type=float, default=0.092)
    parser.add_argument("--tag-id", type=int, default=0)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--timeout-s", type=float, default=12.0)
    parser.add_argument("--above-m", type=float, default=0.20)
    parser.add_argument("--max-reference-tag-error-m", type=float, default=0.025)
    parser.add_argument("--max-target-distance-m", type=float, default=0.12)
    parser.add_argument("--max-joint-delta-rad", type=float, default=0.35)
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--max-speed-rad-s", type=float, default=0.12)
    parser.add_argument("--closed-loop-segment-m", type=float, default=0.05)
    parser.add_argument("--replan-period-s", type=float, default=0.5)
    parser.add_argument("--position-tolerance-m", type=float, default=0.015)
    parser.add_argument("--hold-s", type=float, default=1.0)
    parser.add_argument("--kp", type=float, default=16.0)
    parser.add_argument("--kd", type=float, default=0.9)
    parser.add_argument("--hold-kp", type=float, default=18.0)
    parser.add_argument("--hold-kd", type=float, default=1.2)
    parser.add_argument("--execute", action="store_true", help="actually publish rt/lowcmd; default only prints dry-run result")
    args = parser.parse_args()

    t_base_tag, reproj, obs_std = observe_tag_in_base(
        Path(args.calibration), args.samples, args.timeout_s, args.tag_id, args.tag_size_m
    )
    calib = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    ref_t_base_tag = np.array(calib["T_base_tag"], dtype=float)
    ref_tag_error = float(np.linalg.norm(t_base_tag[:3, 3] - ref_t_base_tag[:3, 3]))
    if ref_tag_error > args.max_reference_tag_error_m:
        raise RuntimeError(
            "observed tag center is inconsistent with fixed calibration: "
            f"{ref_tag_error:.3f} m > {args.max_reference_tag_error_m:.3f} m"
        )
    target_pos = t_base_tag[:3, 3].copy()
    target_pos[2] += args.above_m

    ChannelFactoryInitialize(args.domain_id, args.interface)
    reader = LowStateReader()
    sub = ChannelSubscriber(args.state_topic, LowState_)
    sub.Init(reader.cb, 10)
    deadline = time.monotonic() + 3.0
    while reader.msg is None and time.monotonic() < deadline:
        time.sleep(0.02)
    if reader.msg is None:
        raise RuntimeError("no rt/lowstate received")

    model = pin.buildModelFromUrdf(args.urdf)
    q0 = q_model_from_lowstate(model, reader.msg)
    tcp_offset = np.array(json.loads(Path(args.probe_tcp).read_text())["probe_tcp_in_frame_m"], dtype=float)
    data = model.createData()
    current_probe = frame_pose(model, data, q0, "right_wrist_yaw_link", tcp_offset).translation
    target_dist = float(np.linalg.norm(target_pos - current_probe))
    if target_dist > args.max_target_distance_m:
        raise RuntimeError(
            f"target too far from current probe: {target_dist:.3f} m > {args.max_target_distance_m:.3f} m"
        )

    q_sol, ik_err, iters = solve_right_probe_position(model, q0, target_pos, tcp_offset)
    right_q0 = right_q_from_model(model, q0)
    right_q_sol = right_q_from_model(model, q_sol)
    joint_delta = right_q_sol - right_q0
    max_joint_delta = float(np.max(np.abs(joint_delta)))
    if max_joint_delta > args.max_joint_delta_rad:
        raise RuntimeError(
            f"IK joint delta too large: {max_joint_delta:.3f} rad > {args.max_joint_delta_rad:.3f} rad"
        )
    if ik_err > 0.025:
        raise RuntimeError(f"IK error too large: {ik_err:.3f} m")

    print("R1-A7 AprilTag safe target")
    print(f"  execute: {args.execute}")
    print(f"  observed tag center base: {np.array2string(t_base_tag[:3, 3], precision=6)}")
    print(f"  reference tag error: {ref_tag_error:.6f} m")
    print(f"  target probe position base: {np.array2string(target_pos, precision=6)}")
    print(f"  current probe position base: {np.array2string(current_probe, precision=6)}")
    print(f"  target distance: {target_dist:.6f} m")
    print(f"  mean reprojection error: {reproj:.3f} px")
    print(f"  observed tag std: {np.array2string(obs_std, precision=6)} m")
    print(f"  ik_err: {ik_err:.6f} m, iters: {iters}")
    print("  right_q_current:", np.array2string(right_q0, precision=5))
    print("  right_q_target: ", np.array2string(right_q_sol, precision=5))
    print("  right_q_delta:  ", np.array2string(joint_delta, precision=5))

    if not args.execute:
        print("  dry-run only: add --execute after stopping VR control to move the robot")
        return 0

    pub = ChannelPublisher(args.command_topic, LowCmd_)
    pub.Init()
    crc = CRC()
    fixed_waist_q = float(reader.msg.motor_state[13].q)
    command = right_q0.copy()
    start = time.monotonic()
    last = start
    last_report = start
    next_replan = start
    reached = False
    segment_q = command.copy()
    last_remaining_dist = target_dist
    last_segment_ik_err = ik_err
    try:
        while time.monotonic() - start < args.duration_s:
            now = time.monotonic()
            dt = now - last
            last = now
            if reader.msg is None or now - reader.t > 0.5:
                raise RuntimeError("stale rt/lowstate while executing")
            q_now = q_model_from_lowstate(model, reader.msg)
            current_right_q = right_q_from_model(model, q_now)
            current_probe_now = frame_pose(model, data, q_now, "right_wrist_yaw_link", tcp_offset).translation
            remaining = target_pos - current_probe_now
            remaining_dist = float(np.linalg.norm(remaining))
            last_remaining_dist = remaining_dist
            if remaining_dist <= args.position_tolerance_m:
                reached = True
                command = current_right_q.copy()
                break
            if now >= next_replan:
                segment_pos = target_pos.copy()
                if remaining_dist > args.closed_loop_segment_m:
                    segment_pos = current_probe_now + remaining / remaining_dist * args.closed_loop_segment_m
                q_segment, segment_ik_err, _ = solve_right_probe_position(model, q_now, segment_pos, tcp_offset)
                segment_q = right_q_from_model(model, q_segment)
                segment_delta = segment_q - current_right_q
                max_segment_delta = float(np.max(np.abs(segment_delta)))
                if max_segment_delta > args.max_joint_delta_rad:
                    raise RuntimeError(
                        f"closed-loop segment joint delta too large: "
                        f"{max_segment_delta:.3f} rad > {args.max_joint_delta_rad:.3f} rad"
                    )
                if segment_ik_err > 0.025:
                    raise RuntimeError(f"closed-loop segment IK error too large: {segment_ik_err:.3f} m")
                last_segment_ik_err = segment_ik_err
                next_replan = now + args.replan_period_s
            step = np.clip(segment_q - command, -args.max_speed_rad_s * dt, args.max_speed_rad_s * dt)
            command = command + step
            publish_right_arm(
                reader.msg, pub, crc, command, 13, fixed_waist_q, args.kp, args.kd, args.hold_kp, args.hold_kd
            )
            if now - last_report >= 1.0:
                print(
                    "  closed-loop remaining: "
                    f"{last_remaining_dist:.4f} m, segment_ik_err: {last_segment_ik_err:.4f} m"
                )
                last_report = now
            time.sleep(max(0.0, 1.0 / args.hz))
        hold_until = time.monotonic() + max(0.0, args.hold_s)
        while time.monotonic() < hold_until:
            if reader.msg is not None:
                q_now = q_model_from_lowstate(model, reader.msg)
                current_probe_now = frame_pose(model, data, q_now, "right_wrist_yaw_link", tcp_offset).translation
                q_hold, _, _ = solve_right_probe_position(model, q_now, target_pos, tcp_offset)
                publish_right_arm(
                    reader.msg,
                    pub,
                    crc,
                    right_q_from_model(model, q_hold),
                    13,
                    fixed_waist_q,
                    args.kp,
                    args.kd,
                    args.hold_kp,
                    args.hold_kd,
                )
            time.sleep(max(0.0, 1.0 / args.hz))
        if reader.msg is not None:
            q_final = q_model_from_lowstate(model, reader.msg)
            final_probe = frame_pose(model, data, q_final, "right_wrist_yaw_link", tcp_offset).translation
            final_error = float(np.linalg.norm(target_pos - final_probe))
            print(f"  final probe position base: {np.array2string(final_probe, precision=6)}")
            print(f"  final target error: {final_error:.6f} m")
            print(f"  reached tolerance: {reached or final_error <= args.position_tolerance_m}")
    finally:
        low_cmd = unitree_hg_msg_dds__LowCmd_()
        low_cmd.crc = crc.Crc(low_cmd)
        pub.Write(low_cmd)
        print("  released lowcmd gains")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
