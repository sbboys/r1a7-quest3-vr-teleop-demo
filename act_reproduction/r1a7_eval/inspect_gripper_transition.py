import h5py
import numpy as np

DATASET = "/data/R1A7/wrench_act_pilot_v1"

thresholds = [
    0.9,
    0.8,
    0.7,
    0.6,
    0.5,
    0.4,
    0.3,
    0.2,
]

print("=" * 78)
print("R1-A7 RIGHT GRIPPER TRANSITION INSPECTION")
print("=" * 78)

for ep in range(5):

    path = f"{DATASET}/episode_{ep}.hdf5"

    with h5py.File(path, "r") as f:
        action = f["action"][:]

    grip = action[:, 15]
    T = len(grip)

    print()
    print("=" * 78)
    print(f"episode_{ep}")
    print("=" * 78)

    print(f"T              = {T}")
    print(f"duration       = {T / 30.0:.3f} s")
    print(f"gripper min    = {grip.min():.6f}")
    print(f"gripper max    = {grip.max():.6f}")

    crossings = {}

    for th in thresholds:

        idx = np.flatnonzero(grip < th)

        if len(idx) == 0:
            crossings[th] = None
            print(
                f"first < {th:.1f}    = NOT_FOUND"
            )
        else:
            t = int(idx[0])
            crossings[th] = t

            print(
                f"first < {th:.1f}    = "
                f"{t:4d}  "
                f"{t / 30.0:7.3f} s  "
                f"{100.0 * t / T:6.2f}%"
            )

    t08 = crossings[0.8]
    t05 = crossings[0.5]
    t03 = crossings[0.3]

    print()

    if t08 is not None and t05 is not None:
        print(
            "0.8 -> 0.5 close time = "
            f"{(t05 - t08) / 30.0:.3f} s "
            f"({t05 - t08} frames)"
        )

    if t08 is not None and t03 is not None:
        print(
            "0.8 -> 0.3 close time = "
            f"{(t03 - t08) / 30.0:.3f} s "
            f"({t03 - t08} frames)"
        )

    if t05 is not None:

        lo = max(0, t05 - 30)
        hi = min(T, t05 + 31)

        print()
        print("samples around first < 0.5:")
        print(
            " ".join(
                f"{x:.3f}"
                for x in grip[lo:hi:5]
            )
        )

print()
print("R1A7_GRIPPER_TRANSITION_INSPECTION=PASS")
