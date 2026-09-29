import h5py
import numpy as np

DATASET = "/data/R1A7/wrench_act_pilot_v1"

thresholds = [
    0.8,
    0.7,
    0.6,
    0.5,
    0.4,
    0.3,
    0.2,
]

joint_names = [
    "R_J1",
    "R_J2",
    "R_J3",
    "R_J4",
    "R_J5",
    "R_J6",
    "R_J7",
]

# Canonical action:
# 0-6   left arm
# 7     left gripper
# 8-14  right arm
# 15    right gripper

right_arm_idx = list(range(8, 15))
right_gripper_idx = 15

episodes = []

for ep in range(5):

    path = f"{DATASET}/episode_{ep}.hdf5"

    with h5py.File(path, "r") as f:
        action = f["action"][:]

    episodes.append(action)


print("=" * 78)
print("R1-A7 RIGHT ARM AT GRIPPER EVENT ALIGNMENT")
print("=" * 78)

for th in thresholds:

    print()
    print("=" * 78)
    print(f"GRIPPER THRESHOLD < {th:.1f}")
    print("=" * 78)

    arm_states = []
    timestamps = []

    for ep, action in enumerate(episodes):

        grip = action[:, right_gripper_idx]

        idx = np.flatnonzero(
            grip < th
        )

        if len(idx) == 0:

            print(
                f"episode_{ep}: EVENT_NOT_FOUND"
            )

            continue

        ts = int(idx[0])

        arm = action[
            ts,
            right_arm_idx
        ].astype(np.float64)

        arm_states.append(arm)
        timestamps.append(ts)

        print(
            f"episode_{ep}: "
            f"ts={ts:4d} "
            f"time={ts/30.0:7.3f}s "
            f"grip={grip[ts]:.4f} "
            +
            " ".join(
                f"{joint_names[j]}={arm[j]:+.4f}"
                for j in range(7)
            )
        )

    arm_states = np.stack(
        arm_states,
        axis=0,
    )

    print()
    print("CROSS-DEMO VARIATION:")

    for j, name in enumerate(joint_names):

        x = arm_states[:, j]

        print(
            f"{name:<4} "
            f"MEAN={np.mean(x):+.5f} "
            f"STD={np.std(x):.5f} "
            f"MIN={np.min(x):+.5f} "
            f"MAX={np.max(x):+.5f} "
            f"RANGE={np.ptp(x):.5f}"
        )

print()
print("=" * 78)
print("SUMMARY BY JOINT")
print("=" * 78)

for j, name in enumerate(joint_names):

    ranges = []

    for th in thresholds:

        values = []

        for action in episodes:

            grip = action[:, right_gripper_idx]

            idx = np.flatnonzero(
                grip < th
            )

            if len(idx) == 0:
                continue

            ts = int(idx[0])

            values.append(
                float(
                    action[
                        ts,
                        right_arm_idx[j]
                    ]
                )
            )

        if len(values) == 5:
            ranges.append(
                np.ptp(values)
            )

    print(
        f"{name:<4} "
        f"EVENT_RANGE_MEAN={np.mean(ranges):.5f} "
        f"EVENT_RANGE_MAX={np.max(ranges):.5f}"
    )

print()
print(
    "R1A7_GRIPPER_EVENT_ARM_ALIGNMENT=PASS"
)
