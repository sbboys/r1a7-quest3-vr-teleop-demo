import os
import sys
import pickle
from pathlib import Path

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.argv = [
    "r1a7_safety_scan",
    "--ckpt_dir", "/tmp/r1a7_unused",
    "--policy_class", "ACT",
    "--task_name", "r1a7_wrench_pilot_v1",
    "--seed", "0",
    "--num_epochs", "1",
]

from policy import ACTPolicy


ckpt_dir = os.environ["CKPT_DIR"]
episode_path = os.environ["EPISODE"]

camera_names = [
    "top",
    "left_wrist",
    "right_wrist",
]

arm_indices = list(range(7)) + list(range(8, 15))

joint_names = [
    "L_J1", "L_J2", "L_J3", "L_J4", "L_J5", "L_J6", "L_J7",
    "R_J1", "R_J2", "R_J3", "R_J4", "R_J5", "R_J6", "R_J7",
]

policy_config = {
    "lr": 1e-5,
    "num_queries": 100,
    "kl_weight": 10,
    "hidden_dim": 512,
    "dim_feedforward": 3200,
    "lr_backbone": 1e-5,
    "backbone": "resnet18",
    "enc_layers": 4,
    "dec_layers": 7,
    "nheads": 8,
    "camera_names": camera_names,
    "state_dim": 16,
}

with open(
    os.path.join(ckpt_dir, "dataset_stats.pkl"),
    "rb",
) as f:
    stats = pickle.load(f)

qpos_mean = np.asarray(stats["qpos_mean"], dtype=np.float32)
qpos_std = np.asarray(stats["qpos_std"], dtype=np.float32)
action_mean = np.asarray(stats["action_mean"], dtype=np.float32)
action_std = np.asarray(stats["action_std"], dtype=np.float32)

print("Building ACT...")

policy = ACTPolicy(policy_config)

ckpt_name = os.environ.get(
    "CKPT_NAME",
    "policy_best.ckpt",
)

ckpt_path = os.path.join(
    ckpt_dir,
    ckpt_name,
)

print("checkpoint =", ckpt_path)

state_dict = torch.load(
    ckpt_path,
    map_location="cpu",
)

policy.load_state_dict(
    state_dict,
    strict=True,
)

policy.cuda()
policy.eval()

print("POLICY_LOAD=PASS")

records = []

with h5py.File(episode_path, "r") as f:

    qpos_all = f["observations/qpos"][:]
    action_all = f["action"][:]

    T = len(qpos_all)

    # Scan every 10 frames, plus the final frame.
    timestamps = list(range(0, T, 10))

    if timestamps[-1] != T - 1:
        timestamps.append(T - 1)

    print("episode =", episode_path)
    print("T =", T)
    print("scan points =", len(timestamps))

    for count, ts in enumerate(timestamps, 1):

        qpos = qpos_all[ts]

        qpos_norm = (
            qpos - qpos_mean
        ) / qpos_std

        images = []

        for cam in camera_names:

            img = f[
                f"observations/images/{cam}"
            ][ts]

            img = np.transpose(img, (2, 0, 1))
            images.append(img)

        image = (
            np.stack(images, axis=0)
            .astype(np.float32)
            / 255.0
        )

        qpos_tensor = (
            torch.from_numpy(qpos_norm)
            .float()
            .unsqueeze(0)
            .cuda()
        )

        image_tensor = (
            torch.from_numpy(image)
            .float()
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            pred_norm = policy(
                qpos_tensor,
                image_tensor,
            )[0].cpu().numpy()

        pred = (
            pred_norm
            * action_std[None, :]
            + action_mean[None, :]
        )

        gt_start = max(0, ts - 1)

        current_arm = qpos[arm_indices]
        pred_first = pred[0, arm_indices]
        gt_first = action_all[gt_start, arm_indices]

        act_current = np.abs(
            pred_first - current_arm
        )

        gt_current = np.abs(
            gt_first - current_arm
        )

        act_gt = np.abs(
            pred_first - gt_first
        )

        records.append({
            "ts": ts,
            "act_current": act_current,
            "gt_current": gt_current,
            "act_gt": act_gt,
        })

        if count % 20 == 0 or count == len(timestamps):
            print(
                f"scan {count}/{len(timestamps)} "
                f"ts={ts}"
            )


# ------------------------------------------------------------
# Aggregate
# ------------------------------------------------------------

act_current_all = np.stack([
    r["act_current"] for r in records
])

gt_current_all = np.stack([
    r["gt_current"] for r in records
])

act_gt_all = np.stack([
    r["act_gt"] for r in records
])

flat_act_current = act_current_all.reshape(-1)
flat_gt_current = gt_current_all.reshape(-1)
flat_act_gt = act_gt_all.reshape(-1)

max_pos = np.unravel_index(
    np.argmax(act_gt_all),
    act_gt_all.shape,
)

max_record_idx, max_joint_idx = max_pos
max_ts = records[max_record_idx]["ts"]

print()
print("=" * 72)
print("GLOBAL SAFETY SUMMARY")
print("=" * 72)

def report(name, x):
    print(
        f"{name:<20} "
        f"MEAN={np.mean(x):.6f} "
        f"P95={np.percentile(x,95):.6f} "
        f"P99={np.percentile(x,99):.6f} "
        f"MAX={np.max(x):.6f} rad"
    )

report(
    "GT->current",
    flat_gt_current,
)

report(
    "ACT->current",
    flat_act_current,
)

report(
    "ACT->GT",
    flat_act_gt,
)

print()
print("Worst ACT->GT:")
print("  timestamp =", max_ts)
print("  arm-index =", max_joint_idx)
print("  joint     =", joint_names[max_joint_idx])
print(
    "  error rad =",
    f"{act_gt_all[max_record_idx, max_joint_idx]:.6f}",
)
print(
    "  error deg =",
    f"{np.degrees(act_gt_all[max_record_idx, max_joint_idx]):.3f}",
)

print()
print("=" * 72)
print("PER-JOINT ACT->GT")
print("=" * 72)

for j, name in enumerate(joint_names):

    x = act_gt_all[:, j]

    print(
        f"{j:2d} {name:<4} "
        f"MEAN={np.mean(x):.5f} "
        f"P95={np.percentile(x,95):.5f} "
        f"P99={np.percentile(x,99):.5f} "
        f"MAX={np.max(x):.5f}"
    )

print()
print("R1A7_ACT_DENSE_SAFETY_SCAN=PASS")
