#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复 /home/robot/unitree_sim_isaaclab/sim_main.py 的缩进错误。

策略：
1. 备份当前损坏文件；
2. 搜索工程内现有 sim_main.py 备份；
3. 选择最新且语法有效的备份；
4. 如果没有有效备份，则从 git HEAD 提取原始 sim_main.py；
5. 删除旧的 POSE_GRASP_INSPECTION 临时代码；
6. 只重新加入 --action_source pose_grasp；
7. 最终执行语法检查。

本脚本不修改 A7 机器人配置和 A7 任务配置。
"""

from __future__ import annotations

import datetime
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


REPO = Path("/home/robot/unitree_sim_isaaclab")
TARGET = REPO / "sim_main.py"

BEGIN_MARKER = "# >>> POSE_GRASP_INSPECTION_BEGIN"
END_MARKER = "# <<< POSE_GRASP_INSPECTION_END"


def is_valid_python(path: Path) -> tuple[bool, str]:
    try:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def remove_inspection_block(text: str) -> str:
    # 删除带标记的完整临时代码块。
    pattern = re.compile(
        r"\n?[ \t]*"
        + re.escape(BEGIN_MARKER)
        + r".*?"
        + re.escape(END_MARKER)
        + r"[ \t]*\n?",
        flags=re.DOTALL,
    )
    return pattern.sub("\n", text)


def add_pose_grasp_choice(text: str) -> str:
    if re.search(r'["\']pose_grasp["\']', text):
        return text

    # 只在 --action_source 的 add_argument 调用内部修改 choices。
    block_pattern = re.compile(
        r'parser\.add_argument\(\s*["\']--action_source["\']'
        r'.*?\n\s*\)',
        flags=re.DOTALL,
    )
    block_match = block_pattern.search(text)
    if block_match is None:
        raise RuntimeError(
            "无法定位 parser.add_argument('--action_source', ...)"
        )

    block = block_match.group(0)
    choices_pattern = re.compile(
        r"(choices\s*=\s*\[)(.*?)(\])",
        flags=re.DOTALL,
    )
    choices_match = choices_pattern.search(block)
    if choices_match is None:
        raise RuntimeError(
            "找到了 --action_source，但没有找到 choices=[...]"
        )

    body = choices_match.group(2).rstrip()
    comma = "" if body.endswith(",") else ","
    new_choices = (
        choices_match.group(1)
        + body
        + comma
        + '\n        "pose_grasp",'
        + choices_match.group(3)
    )
    new_block = (
        block[:choices_match.start()]
        + new_choices
        + block[choices_match.end():]
    )

    return (
        text[:block_match.start()]
        + new_block
        + text[block_match.end():]
    )


def collect_backup_candidates() -> list[Path]:
    patterns = [
        "sim_main.py.before_fix_*.bak",
        "sim_main.py.pose_grasp_*.bak",
        "sim_main.py.a7_*.bak",
        "sim_main.py.*.bak",
        "backup_pose_grasp/sim_main.py.bak",
    ]

    found: dict[str, Path] = {}
    for pattern in patterns:
        for path in REPO.glob(pattern):
            if path.is_file() and path.resolve() != TARGET.resolve():
                found[str(path.resolve())] = path

    return sorted(
        found.values(),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def get_git_head_candidate(temp_dir: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "show", "HEAD:sim_main.py"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    candidate = temp_dir / "sim_main_from_git_head.py"
    candidate.write_text(result.stdout, encoding="utf-8")
    return candidate


def main() -> None:
    if not TARGET.is_file():
        raise FileNotFoundError(f"找不到 {TARGET}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    broken_backup = REPO / f"sim_main.py.broken_{timestamp}.bak"
    shutil.copy2(TARGET, broken_backup)
    print(f"[1/7] 已备份当前文件：{broken_backup}")

    current_ok, current_error = is_valid_python(TARGET)
    print(f"[2/7] 当前文件语法有效：{current_ok}")
    if not current_ok:
        print(f"      {current_error}")

    candidates = collect_backup_candidates()
    selected: Path | None = None

    print("[3/7] 搜索可用备份：")
    for candidate in candidates:
        ok, error = is_valid_python(candidate)
        print(f"      {'OK ' if ok else 'BAD'} {candidate}")
        if not ok:
            print(f"          {error}")
        if ok and selected is None:
            selected = candidate

    with tempfile.TemporaryDirectory() as temp_name:
        temp_dir = Path(temp_name)

        if selected is None:
            git_candidate = get_git_head_candidate(temp_dir)
            if git_candidate is not None:
                ok, error = is_valid_python(git_candidate)
                print(
                    "[4/7] git HEAD 候选："
                    f"{'OK' if ok else 'BAD'}"
                )
                if not ok:
                    print(f"      {error}")
                if ok:
                    selected = git_candidate
            else:
                print("[4/7] 无法从 git HEAD 获取 sim_main.py")
        else:
            print("[4/7] 已找到有效本地备份，不需要使用 git HEAD")

        if selected is None:
            raise RuntimeError(
                "没有找到任何语法有效的 sim_main.py 备份，"
                "也无法从 git HEAD 恢复。"
            )

        print(f"[5/7] 采用恢复源：{selected}")
        clean_text = selected.read_text(encoding="utf-8")
        clean_text = remove_inspection_block(clean_text)
        clean_text = add_pose_grasp_choice(clean_text)

        # 在写回前先对内存中的结果做语法检查。
        compile(clean_text, str(TARGET), "exec")
        TARGET.write_text(clean_text, encoding="utf-8")

    ok, error = is_valid_python(TARGET)
    if not ok:
        shutil.copy2(broken_backup, TARGET)
        raise RuntimeError(
            "修复结果仍有语法错误，已恢复到执行前的损坏文件。\n"
            + error
        )

    print("[6/7] sim_main.py 最终语法检查通过")

    # 验证 pose_grasp 已存在。
    final_text = TARGET.read_text(encoding="utf-8")
    if not re.search(r'["\']pose_grasp["\']', final_text):
        raise RuntimeError("修复后未找到 pose_grasp action source")

    print("[7/7] pose_grasp action source 已保留")
    print("\n修复完成。下一步执行：")
    print(
        "python -m py_compile sim_main.py "
        "action_provider/create_action_provider.py "
        "action_provider/action_provider_pose_grasp.py"
    )


if __name__ == "__main__":
    main()
