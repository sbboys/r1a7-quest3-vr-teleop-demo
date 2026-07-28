#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调整 R1-A7 在桌面场景中的根位姿和双臂初始操作姿态。

默认修改：
  /home/robot/unitree_sim_isaaclab/robots/a7.py

示例：
  python set_a7_tabletop_pose.py --z 0.95
  python set_a7_tabletop_pose.py --z 1.00 --x -0.15 --y 0.0

说明：
- --z 控制整个 A7 根节点高度；
- 脚本同时把双臂设为较适合桌面操作的前伸弯肘姿态；
- 每次运行都会先生成时间戳备份；
- 修改后自动执行 Python 语法检查。
"""

from __future__ import annotations

import argparse
import datetime
import re
import shutil
from pathlib import Path


DEFAULT_CFG = Path(
    "/home/robot/unitree_sim_isaaclab/robots/a7.py"
)


READY_POSE = {
    "left_shoulder_pitch_joint": -0.55,
    "left_shoulder_roll_joint": 0.35,
    "left_shoulder_yaw_joint": 0.00,
    "left_elbow_joint": 1.00,
    "left_wrist_roll_joint": 0.00,
    "left_wrist_pitch_joint": 0.00,
    "left_wrist_yaw_joint": 0.00,
    "right_shoulder_pitch_joint": -0.55,
    "right_shoulder_roll_joint": -0.35,
    "right_shoulder_yaw_joint": 0.00,
    "right_elbow_joint": 1.00,
    "right_wrist_roll_joint": 0.00,
    "right_wrist_pitch_joint": 0.00,
    "right_wrist_yaw_joint": 0.00,
}


def replace_root_position(
    text: str,
    x: float,
    y: float,
    z: float,
) -> str:
    pattern = re.compile(
        r"(?P<indent>[ \t]*)pos\s*=\s*\(\s*"
        r"[-+0-9.eE]+\s*,\s*"
        r"[-+0-9.eE]+\s*,\s*"
        r"[-+0-9.eE]+\s*\)\s*,"
    )
    match = pattern.search(text)
    if match is None:
        raise RuntimeError(
            "没有在 a7.py 中找到 InitialStateCfg 的 pos=(x, y, z)"
        )

    indent = match.group("indent")
    replacement = (
        f"{indent}pos=({x:.4f}, {y:.4f}, {z:.4f}),"
    )
    return text[:match.start()] + replacement + text[match.end():]


def replace_joint_value(
    text: str,
    joint_name: str,
    value: float,
) -> str:
    pattern = re.compile(
        rf'(?P<prefix>["\']{re.escape(joint_name)}["\']\s*:\s*)'
        r"[-+0-9.eE]+"
    )
    match = pattern.search(text)
    if match is None:
        raise RuntimeError(
            f"a7.py 中找不到关节初始值：{joint_name}"
        )
    return (
        text[:match.start()]
        + match.group("prefix")
        + f"{value:.4f}"
        + text[match.end():]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CFG,
    )
    parser.add_argument("--x", type=float, default=-0.15)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument(
        "--z",
        type=float,
        default=0.95,
        help="A7根节点世界坐标Z，建议先从0.95开始",
    )
    parser.add_argument(
        "--keep-arm-pose",
        action="store_true",
        help="只修改模型根位姿，不修改双臂初始关节角",
    )
    args = parser.parse_args()

    config = args.config.expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(config)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = config.with_name(
        config.name + f".tabletop_{timestamp}.bak"
    )
    shutil.copy2(config, backup)
    print(f"已备份：{backup}")

    text = config.read_text(encoding="utf-8")
    text = replace_root_position(
        text,
        x=args.x,
        y=args.y,
        z=args.z,
    )

    if not args.keep_arm_pose:
        for joint_name, value in READY_POSE.items():
            text = replace_joint_value(
                text,
                joint_name,
                value,
            )

    compile(text, str(config), "exec")
    config.write_text(text, encoding="utf-8")

    print(f"已修改：{config}")
    print(
        f"根位姿：x={args.x:.4f}, "
        f"y={args.y:.4f}, z={args.z:.4f}"
    )
    if args.keep_arm_pose:
        print("双臂初始关节角保持原值。")
    else:
        print("双臂已设置为前伸弯肘的桌面操作预备姿态。")
    print("Python语法检查：通过")


if __name__ == "__main__":
    main()
