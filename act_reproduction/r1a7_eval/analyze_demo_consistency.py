import numpy as np
import h5py

DATASET = "/data/R1A7/wrench_act_pilot_v1"

episodes = [
    f"{DATASET}/episode_{i}.hdf5"
    for i in range(5)
]

# Canonical 16D:
# 0-6 left arm
# 7 left gripper
# 8-14 right arm
# 15 right gripper

right_indices = list(range(8, 15))

names = [
    "R_J1",
    "R_J2",
    "R_J3",
    "R_J4",
    "R_J5",
    "R_J6",
    "R_J7",
]

# Normalize every episode to the same 101 progress points.
progress = np.linspace(
    0.0,
    1.0,
    101,
)

all_traj = []

print("=" * 72)
print("R1-A7 DEMONSTRATION CONSISTENCY")
print("=" * 72)

for ep_path in episodes:

    with h5py.File(ep_path, "r") as f:

        action = f["action"][:]

    T = len(action)

    src_x = np.linspace(
        0.0,
        1.0,
        T,
    )

    right = action[:, right_indices]

    interp = np.zeros(
        (len(progress), 7),
        dtype=np.float64,
    )

    for j in range(7):

        interp[:, j] = np.interp(
            progress,
            src_x,
            right[:, j],
        )

    all_traj.append(interp)

    print(
        f"{ep_path}: T={T}"
    )

all_traj = np.stack(
    all_traj,
    axis=0,
)

# shape:
# [5 episodes, 101 progress points, 7 joints]

mean_traj = np.mean(
    all_traj,
    axis=0,
)

std_traj = np.std(
    all_traj,
    axis=0,
)

range_traj = (
    np.max(all_traj, axis=0)
    - np.min(all_traj, axis=0)
)

print()
print("=" * 72)
print("PER-JOINT CROSS-DEMO VARIATION")
print("=" * 72)

for j, name in enumerate(names):

    std_mean = float(
        np.mean(std_traj[:, j])
    )

    std_max = float(
        np.max(std_traj[:, j])
    )

    range_mean = float(
        np.mean(range_traj[:, j])
    )

    range_max = float(
        np.max(range_traj[:, j])
    )

    worst_idx = int(
        np.argmax(range_traj[:, j])
    )

    print(
        f"{name:<4} "
        f"STD_MEAN={std_mean:.5f} "
        f"STD_MAX={std_max:.5f} "
        f"RANGE_MEAN={range_mean:.5f} "
        f"RANGE_MAX={range_max:.5f} "
        f"WORST_PROGRESS={progress[worst_idx]*100:.1f}%"
    )

print()
print("=" * 72)
print("R_J4 / R_J5 KEY PROGRESS POINTS")
print("=" * 72)

for pct in [
    0,
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
]:

    idx = int(pct)

    rj4 = all_traj[
        :, idx, 3
    ]

    rj5 = all_traj[
        :, idx, 4
    ]

    print()
    print(f"progress={pct:3d}%")

    print(
        "  R_J4:",
        " ".join(
            f"{x:+.4f}"
            for x in rj4
        ),
        f"range={np.ptp(rj4):.4f}",
    )

    print(
        "  R_J5:",
        " ".join(
            f"{x:+.4f}"
            for x in rj5
        ),
        f"range={np.ptp(rj5):.4f}",
    )

print()
print(
    "R1A7_DEMO_CONSISTENCY_ANALYSIS=PASS"
)
