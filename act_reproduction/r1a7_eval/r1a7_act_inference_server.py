import os
import sys
import time
import socket
import struct
import pickle
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ACT repo imports expect command-line arguments to exist.
sys.argv = [
    "r1a7_act_inference_server",
    "--ckpt_dir", "/tmp/r1a7_unused",
    "--policy_class", "ACT",
    "--task_name", "r1a7_wrench_pilot_v1",
    "--seed", "0",
    "--num_epochs", "1",
]

from policy import ACTPolicy


CKPT_DIR = os.environ.get(
    "CKPT_DIR",
    "/data/R1A7/checkpoints/r1a7_wrench_pilot_v1_500ep",
)

CKPT_NAME = os.environ.get(
    "CKPT_NAME",
    "policy_epoch_300_seed_0.ckpt",
)

SOCKET_PATH = os.environ.get(
    "ACT_SOCKET_PATH",
    "/tmp/r1a7_act.sock",
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


def recv_exact(conn, n):
    chunks = []
    remaining = n

    while remaining:
        chunk = conn.recv(remaining)

        if not chunk:
            raise ConnectionError(
                "socket closed while receiving payload"
            )

        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def recv_object(conn):
    header = recv_exact(conn, 8)
    size = struct.unpack("!Q", header)[0]

    if size <= 0 or size > 64 * 1024 * 1024:
        raise RuntimeError(
            f"invalid IPC payload size: {size}"
        )

    payload = recv_exact(conn, size)
    return pickle.loads(payload)


def send_object(conn, obj):
    payload = pickle.dumps(
        obj,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

    conn.sendall(
        struct.pack("!Q", len(payload))
        + payload
    )


print("=" * 78)
print("R1-A7 ACT REALTIME INFERENCE SERVER")
print("=" * 78)

print("checkpoint_dir =", CKPT_DIR)
print("checkpoint_name =", CKPT_NAME)
print("socket =", SOCKET_PATH)


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


def infer(request):
    start = time.perf_counter()

    qpos = np.asarray(
        request["qpos"],
        dtype=np.float32,
    )

    if qpos.shape != (16,):
        raise RuntimeError(
            f"invalid qpos shape: {qpos.shape}"
        )

    images_bgr = request["images"]

    images = []

    for role in CAMERA_NAMES:
        img = np.asarray(
            images_bgr[role]
        )

        if (
            img.shape != (480, 640, 3)
            or img.dtype != np.uint8
        ):
            raise RuntimeError(
                f"{role}: invalid image "
                f"shape={img.shape} "
                f"dtype={img.dtype}"
            )

        # Camera Manager supplies BGR.
        # ACT dataset was stored as RGB.
        img_rgb = img[:, :, ::-1]

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


    qpos_norm = (
        qpos - qpos_mean
    ) / qpos_std


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
            f"unexpected ACT output shape: "
            f"{pred.shape}"
        )

    if not np.all(
        np.isfinite(pred)
    ):
        raise RuntimeError(
            "ACT produced non-finite output"
        )


    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000.0


    return {
        "ok": True,
        "action": pred,
        "latency_ms": latency_ms,
        "server_monotonic": time.monotonic(),
    }


if os.path.exists(
    SOCKET_PATH
):
    os.unlink(
        SOCKET_PATH
    )


server = socket.socket(
    socket.AF_UNIX,
    socket.SOCK_STREAM,
)

server.bind(
    SOCKET_PATH
)

server.listen(1)


print()
print(
    "ACT_INFERENCE_SERVER_READY"
)
print(
    "waiting for TV control process...",
    flush=True,
)


try:
    while True:
        conn, _ = server.accept()

        print(
            "ACT_CLIENT_CONNECTED",
            flush=True,
        )

        try:
            while True:
                request = recv_object(
                    conn
                )

                if (
                    isinstance(request, dict)
                    and request.get("command")
                    == "ping"
                ):
                    send_object(
                        conn,
                        {
                            "ok": True,
                            "command": "pong",
                        },
                    )
                    continue

                try:
                    response = infer(
                        request
                    )

                except Exception as exc:
                    response = {
                        "ok": False,
                        "error":
                            f"{type(exc).__name__}: "
                            f"{exc}",
                    }

                send_object(
                    conn,
                    response,
                )

        except (
            ConnectionError,
            BrokenPipeError,
            EOFError,
        ):
            print(
                "ACT_CLIENT_DISCONNECTED",
                flush=True,
            )

        finally:
            conn.close()

finally:
    server.close()

    if os.path.exists(
        SOCKET_PATH
    ):
        os.unlink(
            SOCKET_PATH
        )
