#!/usr/bin/env python3
from pathlib import Path
import datetime
import py_compile
import shutil

repo = Path("/home/robot/unitree_sim_isaaclab")
target = repo / "sim_main.py"
clean = repo / "backup_pose_grasp" / "sim_main.py.bak"

stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
saved = repo / f"sim_main.py.before_fix_{stamp}.bak"
shutil.copy2(target, saved)
print("Saved current file:", saved)

def check(path):
    py_compile.compile(str(path), doraise=True)

try:
    check(target)
    print("Current sim_main.py syntax is already valid.")
except py_compile.PyCompileError as exc:
    print(exc)
    if not clean.exists():
        raise SystemExit(f"Clean backup not found: {clean}")
    shutil.copy2(clean, target)
    check(target)
    print("Restored clean backup:", clean)

text = target.read_text(encoding="utf-8")
begin = "# >>> POSE_GRASP_INSPECTION_BEGIN"
if begin in text:
    print("Inspection block already exists.")
else:
    lines = text.splitlines(keepends=True)
    matches = [i for i, line in enumerate(lines) if line.strip() == "env.reset()"]
    if len(matches) != 1:
        print("env.reset candidates:")
        for i, line in enumerate(lines, 1):
            if "env.reset" in line:
                print(i, line.rstrip())
        raise SystemExit("Could not uniquely locate env.reset().")

    i = matches[0]
    source_line = lines[i]
    indent = source_line[:len(source_line) - len(source_line.lstrip())]

    block = f"""
{indent}{begin}
{indent}robot_inspect = env.scene["robot"]
{indent}object_inspect = env.scene["object"]
{indent}print("\\n" + "=" * 90)
{indent}print("POSE GRASP SCENE INSPECTION")
{indent}print("=" * 90)
{indent}print("\\nROBOT JOINT NAMES:")
{indent}for inspect_index, inspect_name in enumerate(robot_inspect.joint_names):
{indent}    print(f"{{inspect_index:3d}}: {{inspect_name}}")
{indent}print("\\nROBOT BODY NAMES:")
{indent}for inspect_index, inspect_name in enumerate(robot_inspect.body_names):
{indent}    print(f"{{inspect_index:3d}}: {{inspect_name}}")
{indent}print("\\nRIGHT ARM / HAND BODY CANDIDATES:")
{indent}for inspect_index, inspect_name in enumerate(robot_inspect.body_names):
{indent}    inspect_lower = inspect_name.lower()
{indent}    if "right" in inspect_lower and any(
{indent}        inspect_key in inspect_lower
{indent}        for inspect_key in ("wrist", "hand", "palm")
{indent}    ):
{indent}        print(f"{{inspect_index:3d}}: {{inspect_name}}")
{indent}if hasattr(object_inspect.data, "root_pose_w"):
{indent}    inspect_object_pose = object_inspect.data.root_pose_w
{indent}else:
{indent}    inspect_object_pose = object_inspect.data.root_link_pose_w
{indent}print("\\nOBJECT WORLD POSE:")
{indent}print(inspect_object_pose)
{indent}print("\\nROBOT ROOT WORLD POSE:")
{indent}print(robot_inspect.data.root_pose_w)
{indent}print("=" * 90 + "\\n")
{indent}# <<< POSE_GRASP_INSPECTION_END
"""
    lines[i + 1:i + 1] = [block]
    target.write_text("".join(lines), encoding="utf-8")
    check(target)
    print("Inserted inspection block after env.reset().")

print("Final syntax check: OK")
print("Target:", target)
