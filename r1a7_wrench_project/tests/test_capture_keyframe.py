import csv
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "capture_keyframe.py"
spec = importlib.util.spec_from_file_location("capture_keyframe", SCRIPT)
capture_keyframe = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(capture_keyframe)


def test_selects_last_valid_episode_row(tmp_path):
    episode = tmp_path / "episode"
    episode.mkdir()
    with (episode / "states.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp_system",
                "fsm_state",
                "joint_position",
                "gripper_position",
                "communication_ok",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "timestamp_system": "10.0",
                "fsm_state": "waiting",
                "joint_position": "[]",
                "gripper_position": "{}",
                "communication_ok": "False",
            }
        )
        writer.writerow(
            {
                "timestamp_system": "11.0",
                "fsm_state": "handset_teleop",
                "joint_position": "[" + ",".join(str(i) for i in range(14)) + "]",
                "gripper_position": '{"left": 1.0, "right": 2.0}',
                "communication_ok": "True",
            }
        )

    rows = capture_keyframe._read_valid_rows(episode / "states.csv")
    block = capture_keyframe._block_from_state_row("PRE_GRASP", rows[-1], "test")

    keyframe = block["PRE_GRASP"]
    assert keyframe["left_joint_position"] == list(range(7))
    assert keyframe["right_joint_position"] == list(range(7, 14))
    assert keyframe["left_gripper_position"] == 1.0
    assert keyframe["right_gripper_position"] == 2.0


def test_write_keyframe_appends_without_removing_notes(tmp_path):
    output = tmp_path / "keyframes.yaml"
    output.write_text("keyframes: {}\nnotes:\n  - keep me\n", encoding="utf-8")

    capture_keyframe._write_keyframe(
        output,
        {
            "PRE_GRASP": {
                "timestamp_system": 11.0,
                "left_joint_position": [0.0] * 7,
                "right_joint_position": [1.0] * 7,
                "left_gripper_position": 2.0,
                "right_gripper_position": 3.0,
                "description": "test",
            }
        },
    )

    text = output.read_text(encoding="utf-8")
    assert "  PRE_GRASP:" in text
    assert "notes:\n  - keep me\n" in text
