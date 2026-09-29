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


# Keep ACT repo imports satisfied.
sys.argv = [
    "r1a7_v2_offline_gripper_scan",
    "--ckpt_dir", "/tmp/r1a7_unused",
    "--policy_class", "ACT",
    "--task_name", "r1a7_wrench_trigger_v2",
    "--seed", "0",
    "--num_epochs", "1",
]


from policy import ACTPolicy


CKPT_DIR = (
    "/data/R1A7/checkpoints/"
    "r1a7_wrench_trigger_v2_500ep"
)

CKPT_NAME = "policy_best.ckpt"

EPISODE_PATH = (
    "/data/R1A7/wrench_act_pilot_v2/"
    "episode_3.hdf5"
)

CAMERA_NAMES = [
    "top",
    "left_wrist",
    "right_wrist",
]


POLICY_CONFIG = {
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
    "camera_names": CAMERA_NAMES,
    "state_dim": 16,
}


SCAN_STEPS = [
    300,
    350,
    380,
    400,
    410,
    420,
    422,
    430,
    440,
    450,
    460,
    480,
    500,
    550,
    600,
    700,
    900,
    1100,
    1300,
]

CHUNK_OFFSETS = [
    0,
    10,
    25,
    50,
    75,
    99,
]


print("=" * 90)
print("R1-A7 ACT V2 OFFLINE GRIPPER SCAN")
print("=" * 90)

print("checkpoint =", os.path.join(CKPT_DIR, CKPT_NAME))
print("episode    =", EPISODE_PATH)
print()


# ---------------------------------------------------------------------
# Load statistics
# ---------------------------------------------------------------------

stats_path = os.path.join(
    CKPT_DIR,
    "dataset_stats.pkl",
)

with open(stats_path, "rb") as f:
    stats = pickle.load(f)


qpos_mean = np.asarray(
    stats["qpos_mean"],
    dtype=np.float32,
)

qpos_std = np.asarray(
    stats["qpos_std"],
    dtype=np.float32,
)

action_mean = np.asarray(
    stats["action_mean"],
    dtype=np.float32,
)

action_std = np.asarray(
    stats["action_std"],
    dtype=np.float32,
)


assert qpos_mean.shape == (16,)
assert qpos_std.shape == (16,)
assert action_mean.shape == (16,)
assert action_std.shape == (16,)


# ---------------------------------------------------------------------
# Build policy
# ---------------------------------------------------------------------

print("Building ACT...")

policy = ACTPolicy(
    POLICY_CONFIG
)

ckpt_path = os.path.join(
    CKPT_DIR,
    CKPT_NAME,
)

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
print()


# ---------------------------------------------------------------------
# Load validation episode
# ---------------------------------------------------------------------

root = h5py.File(
    EPISODE_PATH,
    "r",
)

qpos_all = np.asarray(
    root["/observations/qpos"][:],
    dtype=np.float32,
)

action_all = np.asarray(
    root["/action"][:],
    dtype=np.float32,
)

T = action_all.shape[0]

assert qpos_all.shape == (T, 16)
assert action_all.shape == (T, 16)


print("T =", T)
print("qpos shape  =", qpos_all.shape)
print("action shape=", action_all.shape)

print()

for camera_name in CAMERA_NAMES:
    key = f"/observations/images/{camera_name}"

    if key not in root:
        raise RuntimeError(
            f"missing camera dataset: {key}"
        )

    print(
        camera_name,
        "shape=",
        root[key].shape,
        "dtype=",
        root[key].dtype,
    )

print()
print("IMAGE_DATASETS=PASS")
print()


# ---------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------

def infer_step(t):
    qpos = qpos_all[t]

    qpos_norm = (
        qpos - qpos_mean
    ) / qpos_std

    images = []

    for role in CAMERA_NAMES:
        # HDF5 ACT dataset already stores RGB.
        img_rgb = np.asarray(
            root[
                f"/observations/images/{role}"
            ][t]
        )

        if (
            img_rgb.ndim != 3
            or img_rgb.shape[2] != 3
        ):
            raise RuntimeError(
                f"{role}: invalid image shape "
                f"{img_rgb.shape}"
            )

        img_chw = np.transpose(
            img_rgb,
            (2, 0, 1),
        )

        images.append(
            img_chw
        )

    image = (
        np.stack(
            images,
            axis=0,
        )
        .astype(np.float32)
        / 255.0
    )

    qpos_tensor = (
        torch.from_numpy(
            qpos_norm
        )
        .float()
        .unsqueeze(0)
        .cuda()
    )

    image_tensor = (
        torch.from_numpy(
            image.copy()
        )
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
    ).astype(np.float32)

    if pred.shape != (100, 16):
        raise RuntimeError(
            f"unexpected output shape: "
            f"{pred.shape}"
        )

    if not np.isfinite(pred).all():
        raise RuntimeError(
            "non-finite ACT output"
        )

    return pred


# ---------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------

print("=" * 90)
print("RIGHT GRIPPER: 1=open, 0=close")
print("=" * 90)

all_rows = []

for t in SCAN_STEPS:

    if t >= T:
        continue

    pred = infer_step(t)

    gt_now = float(
        action_all[t, 15]
    )

    qpos_now = float(
        qpos_all[t, 15]
    )

    grip_chunk = pred[:, 15]

    print()
    print("-" * 90)

    print(
        f"OBS_STEP={t:4d} "
        f"GT_NOW={gt_now:6.3f} "
        f"QPOS_NOW={qpos_now:6.3f}"
    )

    print(
        "PRED_CHUNK "
        f"min={grip_chunk.min():.3f} "
        f"max={grip_chunk.max():.3f} "
        f"mean={grip_chunk.mean():.3f}"
    )

    print(
        "offset | future_step | GT_future | PRED"
    )

    for k in CHUNK_OFFSETS:

        future_t = min(
            t + k,
            T - 1,
        )

        gt_future = float(
            action_all[
                future_t,
                15
            ]
        )

        pred_grip = float(
            grip_chunk[k]
        )

        print(
            f"{k:6d} | "
            f"{future_t:11d} | "
            f"{gt_future:9.3f} | "
            f"{pred_grip:6.3f}"
        )

        all_rows.append(
            (
                t,
                k,
                future_t,
                gt_future,
                pred_grip,
            )
        )


# ---------------------------------------------------------------------
# Focused grasp-transition summary
# ---------------------------------------------------------------------

print()
print("=" * 90)
print("GRASP TRANSITION SUMMARY")
print("=" * 90)

focus_steps = [
    350,
    400,
    420,
    422,
    430,
    440,
    450,
    460,
    480,
    500,
    550,
    600,
]

print(
    "step | GT_now | qpos | "
    "pred_s0 | pred_s10 | pred_s25 | "
    "pred_s50 | pred_s75 | pred_s99"
)

for t in focus_steps:

    if t >= T:
        continue

    pred = infer_step(t)

    g = pred[:, 15]

    print(
        f"{t:4d} | "
        f"{action_all[t,15]:6.3f} | "
        f"{qpos_all[t,15]:5.3f} | "
        f"{g[0]:7.3f} | "
        f"{g[10]:8.3f} | "
        f"{g[25]:8.3f} | "
        f"{g[50]:8.3f} | "
        f"{g[75]:8.3f} | "
        f"{g[99]:8.3f}"
    )


# ---------------------------------------------------------------------
# Simple diagnostic
# ---------------------------------------------------------------------

print()
print("=" * 90)
print("SIMPLE GRIPPER DIAGNOSTIC")
print("=" * 90)


pre_steps = [
    300,
    350,
    380,
    400,
]

post_steps = [
    500,
    550,
    600,
    700,
]


pre_pred = []
post_pred = []


for t in pre_steps:
    if t < T:
        pred = infer_step(t)
        pre_pred.append(
            float(pred[0, 15])
        )


for t in post_steps:
    if t < T:
        pred = infer_step(t)
        post_pred.append(
            float(pred[0, 15])
        )


pre_mean = float(
    np.mean(pre_pred)
)

post_mean = float(
    np.mean(post_pred)
)


print(
    f"pre-grasp pred_s0 mean  = "
    f"{pre_mean:.4f}"
)

print(
    f"post-grasp pred_s0 mean = "
    f"{post_mean:.4f}"
)

print(
    f"delta                    = "
    f"{pre_mean - post_mean:.4f}"
)


# This is only a diagnostic, not a final task-success criterion.
if (
    pre_mean > 0.50
    and post_mean < 0.50
):
    print()
    print(
        "V2_GRIPPER_CLOSE_LEARNED=PASS"
    )
else:
    print()
    print(
        "V2_GRIPPER_CLOSE_LEARNED=CHECK"
    )



print()
print("=" * 90)
print("DENSE GRIPPER TRANSITION SCAN")
print("=" * 90)

dense_steps = list(range(380, 521, 5))

rows = []

for t in dense_steps:
    pred = infer_step(t)

    gt = float(action_all[t, 15])
    q = float(qpos_all[t, 15])
    p = float(pred[0, 15])

    rows.append((t, gt, q, p))

    print(
        f"step={t:4d} "
        f"GT={gt:6.3f} "
        f"QPOS={q:6.3f} "
        f"PRED_S0={p:7.3f}"
    )

gt_arr = np.asarray([r[1] for r in rows])
pred_arr = np.asarray([r[3] for r in rows])

mae = float(np.mean(np.abs(gt_arr - pred_arr)))

# Count large upward reversals while GT is generally closing.
reversals = []

for i in range(1, len(rows)):
    dp = rows[i][3] - rows[i - 1][3]

    if dp > 0.20:
        reversals.append(
            (
                rows[i - 1][0],
                rows[i][0],
                rows[i - 1][3],
                rows[i][3],
                dp,
            )
        )

print()
print(f"DENSE_GRIP_MAE = {mae:.4f}")
print(f"LARGE_UPWARD_REVERSALS = {len(reversals)}")

for r in reversals:
    print(
        f"REVERSAL {r[0]}->{r[1]} "
        f"{r[2]:.3f}->{r[3]:.3f} "
        f"delta={r[4]:.3f}"
    )

print()
print("R1A7_V2_DENSE_GRIPPER_SCAN=PASS")

root.close()

print()
print(
    "R1A7_V2_OFFLINE_SCAN=PASS"
)
