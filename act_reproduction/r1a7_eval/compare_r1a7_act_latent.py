import os
import sys
import pickle
from pathlib import Path

import h5py
import numpy as np
import torch
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.argv = [
    "r1a7_latent_compare",
    "--ckpt_dir", "/tmp/r1a7_unused",
    "--policy_class", "ACT",
    "--task_name", "r1a7_wrench_pilot_v1",
    "--seed", "0",
    "--num_epochs", "1",
]

from policy import ACTPolicy
import detr.models.detr_vae as detr_vae_module


ckpt_dir = os.environ["CKPT_DIR"]
episode_path = os.environ["EPISODE"]
ts = int(os.environ["TS"])

camera_names = [
    "top",
    "left_wrist",
    "right_wrist",
]

arm_indices = list(range(7)) + list(range(8, 15))

joint_names = [
    "L_J1", "L_J2", "L_J3", "L_J4",
    "L_J5", "L_J6", "L_J7",
    "R_J1", "R_J2", "R_J3", "R_J4",
    "R_J5", "R_J6", "R_J7",
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

# ------------------------------------------------------------
# Load dataset statistics
# ------------------------------------------------------------

with open(
    os.path.join(ckpt_dir, "dataset_stats.pkl"),
    "rb",
) as f:
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

# ------------------------------------------------------------
# Build policy
# ------------------------------------------------------------

print("=" * 72)
print("R1-A7 ACT PRIOR vs POSTERIOR-MEAN")
print("=" * 72)

policy = ACTPolicy(policy_config)

state_dict = torch.load(
    os.path.join(
        ckpt_dir,
        "policy_best.ckpt",
    ),
    map_location="cpu",
)

policy.load_state_dict(
    state_dict,
    strict=True,
)

policy.cuda()
policy.eval()

print("POLICY_LOAD=PASS")

# Same normalization used by ACTPolicy.__call__.
image_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

# ------------------------------------------------------------
# Read one real observation and its future action sequence
# ------------------------------------------------------------

with h5py.File(episode_path, "r") as f:

    qpos_all = f["observations/qpos"][:]
    action_all = f["action"][:]

    T = len(qpos_all)

    if ts < 0 or ts >= T:
        raise RuntimeError(
            f"timestamp {ts} outside episode length {T}"
        )

    gt_start = max(0, ts - 1)

    if gt_start + 100 > T:
        raise RuntimeError(
            "Need a full 100-step action chunk for this test. "
            f"T={T}, ts={ts}, gt_start={gt_start}"
        )

    qpos = qpos_all[ts]

    gt = action_all[
        gt_start:gt_start + 100
    ].astype(np.float32)

    images = []

    for cam in camera_names:

        img = f[
            f"observations/images/{cam}"
        ][ts]

        img = np.transpose(
            img,
            (2, 0, 1),
        )

        images.append(img)

# ------------------------------------------------------------
# Normalize exactly as training
# ------------------------------------------------------------

qpos_norm = (
    qpos.astype(np.float32)
    - qpos_mean
) / qpos_std

gt_norm = (
    gt
    - action_mean[None, :]
) / action_std[None, :]

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

actions_tensor = (
    torch.from_numpy(gt_norm)
    .float()
    .unsqueeze(0)
    .cuda()
)

is_pad_tensor = torch.zeros(
    (1, 100),
    dtype=torch.bool,
    device="cuda",
)

# ------------------------------------------------------------
# A. True deployment path: actions=None -> z = 0
# ------------------------------------------------------------

with torch.inference_mode():

    prior_norm = policy(
        qpos_tensor,
        image_tensor,
    )[0]

prior_norm = (
    prior_norm
    .detach()
    .cpu()
    .numpy()
)

prior = (
    prior_norm
    * action_std[None, :]
    + action_mean[None, :]
)

# ------------------------------------------------------------
# B. Teacher-forced posterior mean: z = mu
#
# DETRVAE normally samples:
#   z = mu + std * eps
#
# Temporarily replace reparametrize() with z = mu.
# No source file is modified.
# ------------------------------------------------------------

normalized_image_tensor = image_normalize(
    image_tensor.clone()
)

original_reparametrize = (
    detr_vae_module.reparametrize
)

def posterior_mean(mu, logvar):
    return mu

detr_vae_module.reparametrize = posterior_mean

try:

    with torch.inference_mode():

        posterior_norm, _, latent_info = (
            policy.model(
                qpos_tensor,
                normalized_image_tensor,
                None,
                actions_tensor,
                is_pad_tensor,
            )
        )

finally:

    detr_vae_module.reparametrize = (
        original_reparametrize
    )

mu, logvar = latent_info

posterior_norm = (
    posterior_norm[0]
    .detach()
    .cpu()
    .numpy()
)

posterior = (
    posterior_norm
    * action_std[None, :]
    + action_mean[None, :]
)

mu_np = (
    mu[0]
    .detach()
    .cpu()
    .numpy()
)

std_np = (
    torch.exp(logvar[0] / 2)
    .detach()
    .cpu()
    .numpy()
)

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def arm_metrics(pred, gt):

    err = np.abs(
        pred[:, arm_indices]
        - gt[:, arm_indices]
    )

    first = err[0]
    first10 = err[:10]

    return {
        "chunk_mean": float(err.mean()),
        "chunk_max": float(err.max()),
        "first_mean": float(first.mean()),
        "first_max": float(first.max()),
        "first_joint": int(np.argmax(first)),
        "first10_mean": float(first10.mean()),
    }


prior_m = arm_metrics(
    prior,
    gt,
)

posterior_m = arm_metrics(
    posterior,
    gt,
)

current_arm = qpos[arm_indices]

prior_current = np.abs(
    prior[0, arm_indices]
    - current_arm
)

posterior_current = np.abs(
    posterior[0, arm_indices]
    - current_arm
)

# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

print()
print("episode  =", episode_path)
print("T        =", T)
print("ts       =", ts)
print("gt_start =", gt_start)

print()
print("=" * 72)
print("LATENT POSTERIOR")
print("=" * 72)

print(
    "mu abs mean =",
    f"{np.mean(np.abs(mu_np)):.6f}",
)

print(
    "mu abs max  =",
    f"{np.max(np.abs(mu_np)):.6f}",
)

print(
    "mu L2       =",
    f"{np.linalg.norm(mu_np):.6f}",
)

print(
    "std mean    =",
    f"{np.mean(std_np):.6f}",
)

print(
    "std min/max =",
    f"{np.min(std_np):.6f} / "
    f"{np.max(std_np):.6f}",
)

print()
print("=" * 72)
print("PRIOR ZERO-LATENT")
print("=" * 72)

print(
    "chunk arm MAE      = "
    f"{prior_m['chunk_mean']:.6f}"
)

print(
    "chunk arm MAX      = "
    f"{prior_m['chunk_max']:.6f}"
)

print(
    "first-step MAE     = "
    f"{prior_m['first_mean']:.6f}"
)

print(
    "first-step MAX     = "
    f"{prior_m['first_max']:.6f} "
    f"({joint_names[prior_m['first_joint']]})"
)

print(
    "first-10 MAE       = "
    f"{prior_m['first10_mean']:.6f}"
)

print(
    "ACT->current MAX   = "
    f"{np.max(prior_current):.6f}"
)

print()
print("=" * 72)
print("POSTERIOR-MEAN (GT ACTION PROVIDED)")
print("=" * 72)

print(
    "chunk arm MAE      = "
    f"{posterior_m['chunk_mean']:.6f}"
)

print(
    "chunk arm MAX      = "
    f"{posterior_m['chunk_max']:.6f}"
)

print(
    "first-step MAE     = "
    f"{posterior_m['first_mean']:.6f}"
)

print(
    "first-step MAX     = "
    f"{posterior_m['first_max']:.6f} "
    f"({joint_names[posterior_m['first_joint']]})"
)

print(
    "first-10 MAE       = "
    f"{posterior_m['first10_mean']:.6f}"
)

print(
    "ACT->current MAX   = "
    f"{np.max(posterior_current):.6f}"
)

print()
print("=" * 72)
print("RIGHT ARM PER-JOINT FIRST-STEP ERROR")
print("=" * 72)

prior_first_err = np.abs(
    prior[0, arm_indices]
    - gt[0, arm_indices]
)

posterior_first_err = np.abs(
    posterior[0, arm_indices]
    - gt[0, arm_indices]
)

for j in range(7, 14):

    print(
        f"{joint_names[j]:<4} "
        f"prior={prior_first_err[j]:.6f} "
        f"posterior={posterior_first_err[j]:.6f}"
    )

print()
print("=" * 72)
print("INTERPRETATION RATIO")
print("=" * 72)

ratio = (
    posterior_m["first_mean"]
    / max(prior_m["first_mean"], 1e-12)
)

print(
    "posterior/prior first-step MAE = "
    f"{ratio:.6f}"
)

print()
print("R1A7_ACT_LATENT_COMPARE=PASS")
