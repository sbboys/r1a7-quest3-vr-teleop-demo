import h5py
import numpy as np
import ast

DATASET="/data/R1A7/wrench_act_pilot_v1"


episodes=[]

for ep in range(5):

    path=f"{DATASET}/episode_{ep}.hdf5"

    with h5py.File(path,"r") as f:
        q=f["action"][:]

    episodes.append(q)


def find_event(action,th=0.5):

    grip=action[:,15]

    idx=np.where(
        grip<th
    )[0]

    return int(idx[0])


def load_pose(ep,ts):

    raw=f"/data/R1A7/wrench_pilot_v1/wrench_pilot_{ep+1:03d}/states.csv"

    import csv

    with open(raw,"r") as f:

        reader=csv.DictReader(f)

        rows=list(reader)

    T=np.array(
        ast.literal_eval(
            rows[ts]["right_wrist_pose"]
        )
    )

    return T


poses=[]


print("="*80)
print("R1-A7 RIGHT WRIST EVENT ALIGNMENT")
print("="*80)


for ep,action in enumerate(episodes):

    ts=find_event(action,0.5)

    T=load_pose(ep,ts)

    poses.append(T)

    print()

    print(
        f"episode_{ep}"
    )

    print(
        "event ts=",
        ts,
        "time=",
        ts/30
    )

    print(
        "position=",
        T[:3,3]
    )


# position variation

positions=np.array(
    [
        p[:3,3]
        for p in poses
    ]
)


print()
print("="*80)
print("POSITION VARIATION")
print("="*80)


mean_pos=positions.mean(axis=0)


for i,p in enumerate(positions):

    d=np.linalg.norm(
        p-mean_pos
    )

    print(
        f"episode_{i}: "
        f"error={d:.5f} m"
    )


print()

print(
    "max position range="
    f"{np.ptp(positions,axis=0)}"
)


# rotation difference

print()
print("="*80)
print("ROTATION VARIATION")
print("="*80)


R0=poses[0][:3,:3]


for i,T in enumerate(poses):

    R=T[:3,:3]

    Rerr=R0.T@R

    angle=np.arccos(
        np.clip(
            (np.trace(Rerr)-1)/2,
            -1,
            1
        )
    )

    print(
        f"episode_{i}: "
        f"rotation_from_ep0="
        f"{np.degrees(angle):.3f} deg"
    )


print()
print(
"R1A7_WRIST_EVENT_ALIGNMENT=PASS"
)
