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
    "r1a7_v2_camera_ablation",
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


with open(
    CKPT_DIR + "/dataset_stats.pkl",
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


policy = ACTPolicy(
    POLICY_CONFIG
)

state_dict = torch.load(
    CKPT_DIR + "/policy_best.ckpt",
    map_location="cpu",
)

policy.load_state_dict(
    state_dict,
    strict=True,
)

policy.cuda()
policy.eval()


root = h5py.File(
    EPISODE_PATH,
    "r",
)


FIXED_STEP = 440

fixed_qpos = np.asarray(
    root["/observations/qpos"][FIXED_STEP],
    dtype=np.float32,
)


qpos_norm = (
    fixed_qpos - qpos_mean
) / qpos_std


qpos_tensor = (
    torch.from_numpy(qpos_norm)
    .float()
    .unsqueeze(0)
    .cuda()
)


SWEEP_STEPS = [
    300,
    350,
    380,
    390,
    400,
    410,
    420,
    425,
    430,
    435,
    440,
    445,
    450,
    455,
    460,
    465,
    470,
    475,
    480,
    500,
]


def get_image(role, step):

    img = np.asarray(
        root[
            f"/observations/images/{role}"
        ][step]
    )

    return np.transpose(
        img,
        (2, 0, 1),
    )


def infer_with_images(images):

    image = (
        np.stack(images, axis=0)
        .astype(np.float32)
        / 255.0
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
    )

    return pred[:, 15]


print("=" * 88)
print("R1-A7 V2 CAMERA ABLATION")
print("=" * 88)

print(
    f"qpos fixed at step={FIXED_STEP}"
)

print(
    f"non-swept cameras fixed at step={FIXED_STEP}"
)


for swept_camera in CAMERA_NAMES:

    print()
    print("=" * 88)

    print(
        "SWEEP CAMERA:",
        swept_camera,
    )

    print("=" * 88)

    print(
        "image_step | GT | "
        "pred_s0 | pred_mean"
    )

    values = []

    for step in SWEEP_STEPS:

        images = []

        for role in CAMERA_NAMES:

            if role == swept_camera:
                img_step = step
            else:
                img_step = FIXED_STEP

            images.append(
                get_image(
                    role,
                    img_step,
                )
            )

        g = infer_with_images(
            images
        )

        gt = float(
            root["/action"][
                step,
                15
            ]
        )

        values.append(
            float(g[0])
        )

        print(
            f"{step:10d} | "
            f"{gt:5.3f} | "
            f"{g[0]:7.3f} | "
            f"{g.mean():9.3f}"
        )

    values = np.asarray(values)

    print()

    print(
        f"{swept_camera} "
        f"range="
        f"{values.min():.3f}.."
        f"{values.max():.3f}"
    )

    print(
        f"{swept_camera} "
        f"span="
        f"{values.max()-values.min():.3f}"
    )


root.close()

print()
print(
    "R1A7_V2_CAMERA_ABLATION=PASS"
)
