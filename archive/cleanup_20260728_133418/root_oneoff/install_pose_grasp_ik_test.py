#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安装 PoseGrasp IK 测试 Action Provider。

使用：
  python install_pose_grasp_ik_test.py

前提：
  本脚本与 action_provider_pose_grasp.py 位于同一目录，
  或 action_provider_pose_grasp.py 已经放在工程 action_provider/ 下。
"""

from __future__ import annotations

import datetime
import py_compile
import re
import shutil
from pathlib import Path


REPO = Path("/home/robot/unitree_sim_isaaclab")
SIM_MAIN = REPO / "sim_main.py"
FACTORY = REPO / "action_provider" / "create_action_provider.py"
PROVIDER_DST = (
    REPO / "action_provider" / "action_provider_pose_grasp.py"
)
PROVIDER_SRC = (
    Path(__file__).resolve().parent
    / "action_provider_pose_grasp.py"
)


def backup(path: Path) -> Path:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = path.with_name(path.name + f".pose_grasp_{stamp}.bak")
    shutil.copy2(path, dst)
    print(f"backup: {dst}")
    return dst


def patch_sim_main() -> None:
    text = SIM_MAIN.read_text(encoding="utf-8")

    # 删除之前用于打印名称的临时检查块。
    text = re.sub(
        r"\n[ \t]*# >>> POSE_GRASP_INSPECTION_BEGIN.*?"
        r"# <<< POSE_GRASP_INSPECTION_END\n?",
        "\n",
        text,
        flags=re.DOTALL,
    )

    if '"pose_grasp"' not in text:
        pattern = re.compile(
            r'(parser\.add_argument\(\s*"--action_source".*?'
            r'choices\s*=\s*\[)(.*?)(\])',
            flags=re.DOTALL,
        )
        match = pattern.search(text)
        if not match:
            raise RuntimeError(
                "无法在 sim_main.py 中定位 --action_source choices"
            )

        choices_body = match.group(2).rstrip()
        separator = "" if choices_body.endswith(",") else ","
        replacement = (
            match.group(1)
            + choices_body
            + separator
            + ' "pose_grasp"'
            + match.group(3)
        )
        text = text[:match.start()] + replacement + text[match.end():]

    SIM_MAIN.write_text(text, encoding="utf-8")


def patch_factory() -> None:
    text = FACTORY.read_text(encoding="utf-8")

    import_line = (
        "from action_provider.action_provider_pose_grasp "
        "import PoseGraspActionProvider\n"
    )
    if import_line not in text:
        text = import_line + text

    if 'args.action_source == "pose_grasp"' not in text:
        match = re.search(
            r"(?P<indent>[ \t]*)else:\s*\n"
            r"(?P=indent)[ \t]+print\(f?\"unknown action source:",
            text,
        )
        if not match:
            raise RuntimeError(
                "无法在 create_action_provider.py 中定位最终 else 分支"
            )

        indent = match.group("indent")
        branch = (
            f'{indent}elif args.action_source == "pose_grasp":\n'
            f"{indent}    return PoseGraspActionProvider(\n"
            f"{indent}        env=env,\n"
            f"{indent}        args_cli=args,\n"
            f"{indent}    )\n"
        )
        text = text[:match.start()] + branch + text[match.start():]

    FACTORY.write_text(text, encoding="utf-8")


def main() -> None:
    for path in (SIM_MAIN, FACTORY):
        if not path.is_file():
            raise FileNotFoundError(path)

    backup(SIM_MAIN)
    backup(FACTORY)

    if PROVIDER_SRC.is_file():
        shutil.copy2(PROVIDER_SRC, PROVIDER_DST)
        print(f"installed provider: {PROVIDER_DST}")
    elif not PROVIDER_DST.is_file():
        raise FileNotFoundError(
            "找不到 action_provider_pose_grasp.py。"
            "请把它和本安装脚本放在同一目录。"
        )

    patch_sim_main()
    patch_factory()

    py_compile.compile(str(SIM_MAIN), doraise=True)
    py_compile.compile(str(FACTORY), doraise=True)
    py_compile.compile(str(PROVIDER_DST), doraise=True)

    print("\n安装完成，三个文件语法检查均通过：")
    print(f"  {SIM_MAIN}")
    print(f"  {FACTORY}")
    print(f"  {PROVIDER_DST}")


if __name__ == "__main__":
    main()
