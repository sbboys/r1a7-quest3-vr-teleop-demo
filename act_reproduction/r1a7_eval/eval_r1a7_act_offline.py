import os
import sys
import pickle
from pathlib import Path

# Add ACT repository root so this script can import policy.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import torch

sys.argv = [
    "r1a7_offline_inference",
    "--ckpt_dir", "/tmp/r1a7_offline_unused",
    "--policy_class", "ACT",
    "--task_name", "r1a7_wrench_pilot_v1",
    "--seed", "0",
    "--num_epochs", "1",
]

from policy import ACTPolicy


ckpt_dir = os.environ["CKPT_DIR"]
episode_path = os.environ["EPISODE"]

ckpt_path = os.path.join(ckpt_dir, "policy_best.ckpt")
stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")

camera_names = [
    "top",
    "left_wrist",
    "right_wrist",
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

print("=" * 72)
print("R1-A7 WRENCH ACT OFFLINE INFERENCE")
print("=" * 72)

with open(stats_path, "rb") as f:
    stats = pickle.load(f)

qpos_mean = np.asarray(stats["qpos_mean"], dtype=np.float32)
qpos_std = np.asarray(stats["qpos_std"], dtype=np.float32)
action_mean = np.asarray(stats["action_mean"], dtype=np.float32)
action_std = np.asarray(stats["action_std"], dtype=np.float32)

print("STATS_LOAD=PASS")

policy = ACTPolicy(policy_config)

state_dict = torch.load(
    ckpt_path,
    map_location="cpu",
)

status = policy.load_state_dict(
    state_dict,
    strict=True,
)

print("checkpoint =", ckpt_path)
print("load status =", status)

policy.cuda()
policy.eval()

print("POLICY_STRICT_LOAD=PASS")

with h5py.File(episode_path, "r") as f:

    qpos_all = f["observations/qpos"][:]
    action_all = f["action"][:]

    T = qpos_all.shape[0]

    test_ts = sorted(set([
        0,
        T // 4,
        T // 2,
        3 * T // 4,
        max(0, T - 100),
    ]))

    print()
    print("episode =", episode_path)
    print("T =", T)
    print("test timestamps =", test_ts)

    arm_indices = list(range(7)) + list(range(8, 15))
    gripper_indices = [7, 15]

    all_arm_mae = []
    all_gripper_mae = []
    all_first_arm_mae = []
    all_first10_arm_mae = []

    print()
    print("=" * 72)
    print("INFERENCE RESULTS")
    print("=" * 72)

    for ts in test_ts:

        qpos = qpos_all[ts]

        qpos_norm = (
            (qpos - qpos_mean)
            / qpos_std
        )

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

        image = np.stack(images, axis=0)
        image = image.astype(np.float32) / 255.0

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
            )

        pred_norm = (
            pred_norm[0]
            .detach()
            .cpu()
            .numpy()
        )

        assert pred_norm.shape == (100, 16)
        assert np.isfinite(pred_norm).all()

        pred = (
            pred_norm
            * action_std[None, :]
            + action_mean[None, :]
        )

        gt_start = max(0, ts - 1)

        valid = min(
            100,
            T - gt_start,
        )

        gt = action_all[
            gt_start:gt_start + valid
        ]

        pred_valid = pred[:valid]

        arm_err = np.abs(
            pred_valid[:, arm_indices]
            - gt[:, arm_indices]
        )

        gripper_err = np.abs(
            pred_valid[:, gripper_indices]
            - gt[:, gripper_indices]
        )

        arm_mae = float(arm_err.mean())
        arm_max = float(arm_err.max())

        gripper_mae = float(gripper_err.mean())
        gripper_max = float(gripper_err.max())

        first_arm_mae = float(
            arm_err[0].mean()
        )

        first10_arm_mae = float(
            arm_err[:min(10, valid)].mean()
        )

        # Safety-oriented quantity:
        # Difference between ACT's first predicted arm command
        # and the robot's CURRENT measured arm position.
        #
        # This is different from prediction-vs-demonstration error.
        current_arm_q = qpos[arm_indices]
        predicted_first_arm = pred_valid[0, arm_indices]
        gt_first_arm = gt[0, arm_indices]

        # ACT first command relative to current measured qpos.
        first_command_delta = np.abs(
            predicted_first_arm - current_arm_q
        )

        first_command_delta_mean = float(
            first_command_delta.mean()
        )

        first_command_delta_max = float(
            first_command_delta.max()
        )

        first_command_delta_joint = int(
            np.argmax(first_command_delta)
        )

        # Demonstration command relative to current measured qpos.
        # This tells us how much command/state difference is normal
        # in the recorded demonstration itself.
        gt_command_delta = np.abs(
            gt_first_arm - current_arm_q
        )

        gt_command_delta_mean = float(
            gt_command_delta.mean()
        )

        gt_command_delta_max = float(
            gt_command_delta.max()
        )

        gt_command_delta_joint = int(
            np.argmax(gt_command_delta)
        )

        # Direct first-step model error against demonstration.
        first_model_error = np.abs(
            predicted_first_arm - gt_first_arm
        )

        first_model_error_max = float(
            first_model_error.max()
        )

        first_model_error_joint = int(
            np.argmax(first_model_error)
        )

        all_arm_mae.append(arm_mae)
        all_gripper_mae.append(gripper_mae)
        all_first_arm_mae.append(first_arm_mae)
        all_first10_arm_mae.append(first10_arm_mae)

        print()
        print(f"[ts={ts}]")
        print(f"gt_start          = {gt_start}")
        print(f"valid_chunk       = {valid}")

        print(
            "arm error rad      = "
            f"MAE {arm_mae:.5f} / "
            f"MAX {arm_max:.5f}"
        )

        print(
            "first-step arm MAE = "
            f"{first_arm_mae:.5f} rad"
        )

        print(
            "first-10 arm MAE   = "
            f"{first10_arm_mae:.5f} rad"
        )

        print(
            "GT ->current delta = "
            f"MEAN {gt_command_delta_mean:.5f} / "
            f"MAX {gt_command_delta_max:.5f} rad / "
            f"arm-index {gt_command_delta_joint}"
        )

        print(
            "ACT->current delta = "
            f"MEAN {first_command_delta_mean:.5f} / "
            f"MAX {first_command_delta_max:.5f} rad / "
            f"arm-index {first_command_delta_joint}"
        )

        print(
            "ACT->GT first MAX  = "
            f"{first_model_error_max:.5f} rad / "
            f"arm-index {first_model_error_joint}"
        )

        print(
            "gripper error      = "
            f"MAE {gripper_mae:.5f} / "
            f"MAX {gripper_max:.5f}"
        )

        print(
            "pred right gripper = "
            f"{pred_valid[:,15].min():.4f} .. "
            f"{pred_valid[:,15].max():.4f}"
        )

        print(
            "gt   right gripper = "
            f"{gt[:,15].min():.4f} .. "
            f"{gt[:,15].max():.4f}"
        )

print()
print("=" * 72)
print("SUMMARY")
print("=" * 72)

print(
    "mean chunk arm MAE rad = "
    f"{np.mean(all_arm_mae):.6f}"
)

print(
    "mean first-step MAE    = "
    f"{np.mean(all_first_arm_mae):.6f}"
)

print(
    "mean first-10 MAE      = "
    f"{np.mean(all_first10_arm_mae):.6f}"
)

print(
    "mean gripper MAE       = "
    f"{np.mean(all_gripper_mae):.6f}"
)

print()
print("R1A7_ACT_OFFLINE_INFERENCE=PASS")
