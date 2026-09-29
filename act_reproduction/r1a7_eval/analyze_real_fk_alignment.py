import os
import csv
import json
import numpy as np
import pinocchio as pin


URDF = "/home/robot/R1A7_VR_dual_arm_transfer_20260831_001/unitree_sdk2-main/sim/mujoco_r1/models/A7.urdf"
MODEL_DIR = "/home/robot/R1A7_VR_dual_arm_transfer_20260831_001/unitree_sdk2-main/sim/mujoco_r1/models"

DATA_ROOT = "/data/R1A7/wrench_pilot_v1"


ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",

    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


print("="*80)
print("R1-A7 REAL FK RIGHT EE ALIGNMENT")
print("="*80)


robot = pin.RobotWrapper.BuildFromURDF(
    URDF,
    MODEL_DIR,
)

reduced = robot.buildReducedRobot(
    [
        "waist_yaw_joint",
        "head_pitch_joint",
        "head_yaw_joint",
    ],
    np.zeros(robot.model.nq)
)


reduced.model.addFrame(
    pin.Frame(
        "R_ee",
        reduced.model.getJointId(
            "right_wrist_yaw_joint"
        ),
        pin.SE3(
            np.eye(3),
            np.array([0.05,0,0])
        ),
        pin.FrameType.OP_FRAME,
    )
)


data = reduced.model.createData()

R_id = reduced.model.getFrameId(
    "R_ee"
)


def fk_right_arm(arm_q):

    q = np.zeros(
        reduced.model.nq
    )

    # reduced model has only arm DOF
    q[:] = arm_q[:]

    pin.forwardKinematics(
        reduced.model,
        data,
        q
    )

    pin.updateFramePlacements(
        reduced.model,
        data
    )

    return data.oMf[R_id].homogeneous.copy()



def find_grasp_ts(path):

    with open(path) as f:

        reader = csv.DictReader(f)

        rows=list(reader)


    for i,r in enumerate(rows):

        grip=np.array(
            json.loads(
                r["sent_gripper_q"]
            )
        )

        # right gripper
        if grip[1] < 0.5:

            return i, rows


    raise RuntimeError(
        "grasp not found"
    )



poses=[]


for ep in range(1,6):

    csv_path = (
        f"{DATA_ROOT}/"
        f"wrench_pilot_{ep:03d}/states.csv"
    )


    ts, rows = find_grasp_ts(
        csv_path
    )


    arm_q=np.array(
        json.loads(
            rows[ts]["arm_q"]
        ),
        dtype=float
    )


    # 14 arm joints
    T=fk_right_arm(
        arm_q
    )


    poses.append(T)


    print()
    print(
        f"episode_{ep-1}"
    )
    print(
        "grasp_ts=",
        ts
    )
    print(
        "position=",
        T[:3,3]
    )


print()
print("="*80)
print("POSITION VARIATION")
print("="*80)


P=np.array(
    [
        x[:3,3]
        for x in poses
    ]
)


mean=P.mean(axis=0)


for i,p in enumerate(P):

    print(
        f"episode_{i}: "
        f"{np.linalg.norm(p-mean):.5f} m"
    )


print()
print(
    "range_xyz=",
    np.ptp(P,axis=0)
)


print()
print("="*80)
print("ROTATION VARIATION")
print("="*80)


R0=poses[0][:3,:3]


for i,T in enumerate(poses):

    Rerr=R0.T@T[:3,:3]

    angle=np.arccos(
        np.clip(
            (np.trace(Rerr)-1)/2,
            -1,
            1
        )
    )

    print(
        f"episode_{i}: "
        f"{np.degrees(angle):.3f} deg"
    )


print()
print(
"R1A7_REAL_FK_ALIGNMENT=PASS"
)
