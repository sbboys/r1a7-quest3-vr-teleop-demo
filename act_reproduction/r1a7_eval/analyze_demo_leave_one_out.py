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

episodes = []

for ep in range(5):

    path = f"{DATASET}/episode_{ep}.hdf5"

    with h5py.File(path, "r") as f:
        action = f["action"][:]

    episodes.append(action)

# ------------------------------------------------------------
# Build:
# states[threshold_index, episode, joint]
# ------------------------------------------------------------

states = np.zeros(
    (len(thresholds), 5, 7),
    dtype=np.float64,
)

for ti, th in enumerate(thresholds):

    for ep, action in enumerate(episodes):

        grip = action[:, 15]

        idx = np.flatnonzero(
            grip < th
        )

        if len(idx) == 0:
            raise RuntimeError(
                f"event not found: episode={ep}, threshold={th}"
            )

        ts = int(idx[0])

        states[ti, ep] = action[
            ts,
            8:15,
        ]

print("=" * 78)
print("R1-A7 EVENT-ALIGNED LEAVE-ONE-OUT ANALYSIS")
print("=" * 78)

print()
print("ALL-5 BASELINE")

for j, name in enumerate(joint_names):

    ranges = np.ptp(
        states[:, :, j],
        axis=1,
    )

    print(
        f"{name:<4} "
        f"RANGE_MEAN={np.mean(ranges):.5f} "
        f"RANGE_MAX={np.max(ranges):.5f}"
    )

print()
print("=" * 78)
print("LEAVE-ONE-OUT")
print("=" * 78)

for drop_ep in range(5):

    keep = [
        i for i in range(5)
        if i != drop_ep
    ]

    print()
    print(f"DROP episode_{drop_ep}")

    for j, name in enumerate(joint_names):

        ranges = np.ptp(
            states[:, keep, j],
            axis=1,
        )

        print(
            f"{name:<4} "
            f"RANGE_MEAN={np.mean(ranges):.5f} "
            f"RANGE_MAX={np.max(ranges):.5f}"
        )

print()
print("=" * 78)
print("BEST DROP PER JOINT")
print("=" * 78)

for j, name in enumerate(joint_names):

    results = []

    for drop_ep in range(5):

        keep = [
            i for i in range(5)
            if i != drop_ep
        ]

        ranges = np.ptp(
            states[:, keep, j],
            axis=1,
        )

        results.append((
            float(np.mean(ranges)),
            float(np.max(ranges)),
            drop_ep,
        ))

    results.sort()

    best_mean, best_max, best_drop = results[0]

    all_ranges = np.ptp(
        states[:, :, j],
        axis=1,
    )

    all_mean = float(
        np.mean(all_ranges)
    )

    reduction = (
        100.0 *
        (all_mean - best_mean)
        / max(all_mean, 1e-12)
    )

    print(
        f"{name:<4} "
        f"BEST_DROP=episode_{best_drop} "
        f"ALL_MEAN={all_mean:.5f} "
        f"NEW_MEAN={best_mean:.5f} "
        f"NEW_MAX={best_max:.5f} "
        f"REDUCTION={reduction:.1f}%"
    )

print()
print(
    "R1A7_DEMO_LEAVE_ONE_OUT=PASS"
)
