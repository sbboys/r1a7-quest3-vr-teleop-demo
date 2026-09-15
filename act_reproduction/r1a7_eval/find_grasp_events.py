import h5py
import numpy as np

DATASET = "/data/R1A7/wrench_act_pilot_v1"

for ep in range(5):

    path = f"{DATASET}/episode_{ep}.hdf5"

    with h5py.File(path, "r") as f:
        action = f["action"][:]

    grip = action[:, 15]
    T = len(grip)

    # Search for the first sustained closing event.
    #
    # Require:
    #   previously mostly open
    #   current value < 0.5
    #   next 15 frames remain mostly below 0.6
    grasp_ts = None

    for t in range(15, T - 15):

        before = grip[t-15:t]
        after = grip[t:t+15]

        if (
            np.mean(before > 0.8) >= 0.8
            and grip[t] < 0.5
            and np.mean(after < 0.6) >= 0.8
        ):
            grasp_ts = t
            break

    print("=" * 72)
    print(f"episode_{ep}")

    print(f"T              = {T}")
    print(f"duration approx= {T/30.0:.3f} s")

    if grasp_ts is None:

        print("GRASP_EVENT=NOT_FOUND")

    else:

        print(f"grasp_ts       = {grasp_ts}")
        print(f"grasp_time     = {grasp_ts/30.0:.3f} s")
        print(
            f"episode_progress= "
            f"{100.0*grasp_ts/T:.2f}%"
        )

        lo = max(0, grasp_ts - 5)
        hi = min(T, grasp_ts + 6)

        print(
            "gripper around event =",
            " ".join(
                f"{x:.3f}"
                for x in grip[lo:hi]
            )
        )

print()
print("R1A7_GRASP_EVENT_DETECTION=PASS")
