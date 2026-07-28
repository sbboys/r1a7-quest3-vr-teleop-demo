#!/usr/bin/env python3
"""Keyboard teach-in and trajectory recorder for the R1-A7 right arm."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import select
import signal
import sys
import termios
import time
import tty
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


RIGHT_ARM_NAMES = [
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]

DEFAULT_LIMITS = [
    (-1.35, 1.45),
    (-0.85, 0.35),
    (-1.15, 1.15),
    (0.25, 1.55),
    (-1.20, 1.20),
    (-1.00, 1.00),
    (-1.20, 1.20),
]


@dataclass
class ArmState:
    q: List[float]
    dq: List[float]
    stamp: float


class R1A7RightArmTeach:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state: Optional[LowState_] = None
        self.first_state_time: Optional[float] = None
        self.done = False
        self.publisher = None
        self.subscriber = None

        self.arm_indices = self._parse_indices(args.right_arm_indices)
        if len(self.arm_indices) != 7:
            raise ValueError("--right_arm_indices must contain exactly 7 indices")
        self.weight_index = int(args.weight_index)
        self.joint_limits = self._parse_limits(args.joint_limits)

        self.command_q: Optional[List[float]] = None
        self.selected_joint = 0
        self.motion_name = args.motion
        self.segment_label = args.label
        self.recording = args.auto_record
        self.step_rad = float(args.step_rad)
        self.large_step_rad = float(args.large_step_rad)
        self.start_monotonic = time.monotonic()
        self.last_sample_time = 0.0
        self.last_print_time = 0.0
        self.raw_terminal = False
        self.old_terminal = None
        self.prompt_printed = False

        self.output_dir = self._make_output_dir(args.output_dir)
        self.samples_jsonl = None
        self.samples_csv = None
        self.csv_writer = None
        self.waypoints_jsonl = None
        self.events_jsonl = None
        self.sample_count = 0
        self.waypoint_count = 0

    @staticmethod
    def _parse_indices(text: str) -> List[int]:
        return [int(part.strip()) for part in text.split(",") if part.strip()]

    @staticmethod
    def _parse_limits(text: str) -> List[tuple[float, float]]:
        if not text:
            return list(DEFAULT_LIMITS)
        chunks = [chunk.strip() for chunk in text.split(",") if chunk.strip()]
        if len(chunks) != 7:
            raise ValueError("--joint_limits must contain 7 min:max chunks")
        limits = []
        for chunk in chunks:
            lo_s, hi_s = chunk.split(":", 1)
            limits.append((float(lo_s), float(hi_s)))
        return limits

    @staticmethod
    def _make_output_dir(root: str) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(root).expanduser().resolve() / stamp
        out.mkdir(parents=True, exist_ok=False)
        return out

    def init(self) -> None:
        ChannelFactoryInitialize(self.args.domain_id, self.args.interface)
        if self.args.enter_debug_mode:
            self._enter_debug_mode()

        self.subscriber = ChannelSubscriber(self.args.state_topic, LowState_)
        self.subscriber.Init(self._lowstate_handler, 10)
        if self.args.mode == "jog":
            self.publisher = ChannelPublisher(self.args.command_topic, LowCmd_)
            self.publisher.Init()

        self._open_record_files()
        print("[R1-A7 TEACH] DDS initialized")
        print("[R1-A7 TEACH] interface:", self.args.interface)
        print("[R1-A7 TEACH] state topic:", self.args.state_topic)
        print("[R1-A7 TEACH] command topic:", self.args.command_topic if self.publisher else "disabled")
        print("[R1-A7 TEACH] mode:", self.args.mode)
        print("[R1-A7 TEACH] output:", self.output_dir)
        print("[R1-A7 TEACH] label:", self.segment_label)
        self._print_help()
        if self.args.input_mode == "key":
            self._enable_raw_terminal()

    def _enter_debug_mode(self) -> None:
        msc = MotionSwitcherClient()
        msc.SetTimeout(2.0)
        msc.Init()
        status, result = msc.CheckMode()
        print(f"[R1-A7 TEACH] motion_switcher CheckMode: status={status} result={result}")
        try:
            while result and result.get("name"):
                print("[R1-A7 TEACH] releasing active mode:", result)
                msc.ReleaseMode()
                time.sleep(0.5)
                status, result = msc.CheckMode()
                print(f"[R1-A7 TEACH] motion_switcher CheckMode: status={status} result={result}")
        except Exception as exc:
            print(f"[R1-A7 TEACH] failed to enter debug mode: {exc}")

    def _open_record_files(self) -> None:
        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "mode": self.args.mode,
            "interface": self.args.interface,
            "state_topic": self.args.state_topic,
            "command_topic": self.args.command_topic if self.args.mode == "jog" else None,
            "right_arm_indices": self.arm_indices,
            "joint_names": RIGHT_ARM_NAMES,
            "joint_limits": self.joint_limits,
            "initial_label": self.segment_label,
            "initial_joint": RIGHT_ARM_NAMES[self.selected_joint],
            "initial_motion": self.motion_name,
            "always_sample": self.args.always_sample,
            "units": "radian",
        }
        (self.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        self.samples_jsonl = (self.output_dir / "samples.jsonl").open("a", encoding="utf-8")
        self.waypoints_jsonl = (self.output_dir / "waypoints.jsonl").open("a", encoding="utf-8")
        self.events_jsonl = (self.output_dir / "events.jsonl").open("a", encoding="utf-8")
        self.samples_csv = (self.output_dir / "samples.csv").open("a", newline="", encoding="utf-8")
        fieldnames = [
            "t",
            "recording",
            "label",
            "selected_joint",
            "selected_joint_index",
            "motion",
            *[f"q_{name}" for name in RIGHT_ARM_NAMES],
            *[f"cmd_{name}" for name in RIGHT_ARM_NAMES],
        ]
        self.csv_writer = csv.DictWriter(self.samples_csv, fieldnames=fieldnames)
        self.csv_writer.writeheader()

    def _lowstate_handler(self, msg: LowState_) -> None:
        self.low_state = msg
        if self.first_state_time is None:
            self.first_state_time = time.monotonic()

    def _read_arm(self) -> Optional[ArmState]:
        if self.low_state is None:
            return None
        motor_state = self.low_state.motor_state
        max_idx = max(max(self.arm_indices), self.weight_index)
        if len(motor_state) <= max_idx:
            raise RuntimeError(f"lowstate has {len(motor_state)} motors, requested index {max_idx}")
        q = [float(motor_state[i].q) for i in self.arm_indices]
        dq = [float(motor_state[i].dq) for i in self.arm_indices]
        return ArmState(q=q, dq=dq, stamp=time.monotonic())

    def _enable_raw_terminal(self) -> None:
        if not sys.stdin.isatty():
            return
        self.old_terminal = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self.raw_terminal = True

    def _restore_terminal(self) -> None:
        if self.raw_terminal and self.old_terminal is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_terminal)
            self.raw_terminal = False

    def _read_key(self) -> Optional[str]:
        if not sys.stdin.isatty():
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return None
        return sys.stdin.read(1)

    def _prompt_line(self, prompt: str) -> str:
        self._restore_terminal()
        try:
            return input(prompt).strip()
        finally:
            if self.args.input_mode == "key":
                self._enable_raw_terminal()

    def _print_help(self) -> None:
        print(
            "[R1-A7 TEACH] commands: joint <1-7> | motion <name> | segment <1-7> <motion> | "
            "start/stop | s optional waypoint | n <label> | p print | q quit"
        )
        if self.args.input_mode == "key":
            print("[R1-A7 TEACH] key mode: press one key directly; Enter is not needed and keys are not echoed.")
        else:
            print(
                "[R1-A7 TEACH] line mode examples: joint 4, motion bend_extend, "
                "segment 4 elbow_bend_extend, start, s, stop."
            )
        print("[R1-A7 TEACH] joints:")
        for i, name in enumerate(RIGHT_ARM_NAMES, start=1):
            lo, hi = self.joint_limits[i - 1]
            print(f"  {i}: {name}  limit=[{lo:+.2f},{hi:+.2f}]")

    def _print_line_prompt(self) -> None:
        if self.args.input_mode != "line" or self.prompt_printed:
            return
        print("[R1-A7 TEACH] input> ", end="", flush=True)
        self.prompt_printed = True

    def _init_low_cmd_stop(self) -> None:
        for motor in self.low_cmd.motor_cmd:
            motor.tau = 0.0
            motor.q = 0.0
            motor.dq = 0.0
            motor.kp = 0.0
            motor.kd = 0.0

    def _rate_limit(self, target: List[float], current: List[float], dt: float) -> List[float]:
        max_delta = max(0.0, self.args.max_speed_rad_s) * max(dt, 1e-3)
        out = []
        for cur, des in zip(current, target):
            delta = max(-max_delta, min(max_delta, des - cur))
            out.append(cur + delta)
        return out

    def _publish(self, state: ArmState, command: List[float]) -> None:
        if self.publisher is None:
            return
        self._init_low_cmd_stop()
        if hasattr(self.low_cmd, "mode_pr"):
            self.low_cmd.mode_pr = 0
        if hasattr(self.low_cmd, "mode_machine") and hasattr(self.low_state, "mode_machine"):
            self.low_cmd.mode_machine = self.low_state.mode_machine
        count = min(len(self.low_cmd.motor_cmd), len(self.low_state.motor_state))
        for i in range(count):
            motor = self.low_cmd.motor_cmd[i]
            motor.mode = 1
            motor.tau = 0.0
            motor.q = float(self.low_state.motor_state[i].q)
            motor.dq = 0.0
            motor.kp = self.args.hold_kp
            motor.kd = self.args.hold_kd

        for idx, q in zip(self.arm_indices, command):
            motor = self.low_cmd.motor_cmd[idx]
            motor.mode = 1
            motor.tau = 0.0
            motor.q = float(q)
            motor.dq = 0.0
            motor.kp = self.args.kp
            motor.kd = self.args.kd

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)

    def _clip_joint(self, joint_idx: int, value: float) -> float:
        lo, hi = self.joint_limits[joint_idx]
        return max(lo, min(hi, value))

    def _handle_key(self, key: str, state: ArmState) -> None:
        if key in "1234567":
            self.selected_joint = int(key) - 1
            self._record_event("select_joint", {"joint": RIGHT_ARM_NAMES[self.selected_joint]})
            print(f"\n[R1-A7 TEACH] selected joint={self.selected_joint + 1}:{RIGHT_ARM_NAMES[self.selected_joint]}")
            return
        if key in ("a", "d", "A", "D") and self.command_q is not None:
            step = self.large_step_rad if key in ("A", "D") else self.step_rad
            if key in ("a", "A"):
                step = -step
            j = self.selected_joint
            self.command_q[j] = self._clip_joint(j, self.command_q[j] + step)
            self._record_event("jog", {"joint": RIGHT_ARM_NAMES[j], "target": self.command_q[j]})
            return
        if key == "r":
            self.recording = not self.recording
            self._record_event(
                "recording",
                {
                    "enabled": self.recording,
                    "label": self.segment_label,
                    "joint": RIGHT_ARM_NAMES[self.selected_joint],
                    "motion": self.motion_name,
                },
            )
            print(
                f"\n[R1-A7 TEACH] recording={'ON' if self.recording else 'OFF'} "
                f"joint={self.selected_joint + 1}:{RIGHT_ARM_NAMES[self.selected_joint]} "
                f"motion={self.motion_name or 'unset'} label={self.segment_label} "
                f"samples={self.sample_count}"
            )
            if self.recording:
                print("[R1-A7 TEACH] move the arm through the full motion now; use stop after returning if needed.")
            return
        if key == " ":
            self._record_waypoint(state, manual=True)
            return
        if key == "n":
            label = self._prompt_line("\nLabel for next segment: ")
            if label:
                self.segment_label = label
                self._record_event("label", {"label": label})
            return
        if key == "h":
            self.command_q = list(state.q)
            self._record_event("hold_current", {"q": state.q})
            print("\n[R1-A7 TEACH] command reset to current robot q")
            return
        if key == "p":
            self._print_status(state, force=True)
            return
        if key in ("q", "\x03"):
            self.done = True

    def _read_line_command(self) -> Optional[str]:
        if not sys.stdin.isatty():
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return None
        self.prompt_printed = False
        return sys.stdin.readline().strip()

    def _handle_line_command(self, command: str, state: ArmState) -> None:
        if not command:
            self._print_line_prompt()
            return
        lower = command.lower()
        if lower in ("help", "?"):
            self._print_help()
            self._print_line_prompt()
            return
        if lower in ("status", "st"):
            print(
                f"[R1-A7 TEACH] label={self.segment_label} recording={self.recording} "
                f"motion={self.motion_name or 'unset'} "
                f"always_sample={self.args.always_sample} "
                f"step={self.step_rad:.3f} large={self.large_step_rad:.3f} "
                f"selected={self.selected_joint + 1}:{RIGHT_ARM_NAMES[self.selected_joint]}"
            )
            self._print_line_prompt()
            return
        if lower in ("start", "begin", "on"):
            if not self.recording:
                self._handle_key("r", state)
            else:
                print("[R1-A7 TEACH] recording already ON")
            self._print_line_prompt()
            return
        if lower == "starts":
            if not self.recording:
                self._handle_key("r", state)
            self._record_waypoint(state, manual=True)
            self._print_line_prompt()
            return
        if lower == "startstop":
            print(
                "[R1-A7 TEACH] received merged command 'startstop'. "
                "The motion before this was not marked recording=ON, "
                "but samples are still saved when always_sample=True."
            )
            if self.recording:
                self._handle_key("r", state)
            self._print_line_prompt()
            return
        if lower in ("stop", "end", "off"):
            if self.recording:
                self._handle_key("r", state)
            else:
                print("[R1-A7 TEACH] recording already OFF")
            self._print_line_prompt()
            return
        if lower == "stops":
            if self.recording:
                self._handle_key("r", state)
            self._record_waypoint(state, manual=True)
            self._print_line_prompt()
            return
        if lower.startswith(("joint ", "j ")):
            value = command.split(maxsplit=1)[1].strip()
            try:
                joint_index = int(value) - 1
            except ValueError:
                print("[R1-A7 TEACH] invalid joint. Example: joint 4")
            else:
                if 0 <= joint_index < len(RIGHT_ARM_NAMES):
                    self.selected_joint = joint_index
                    self._record_event("select_joint", {"joint": RIGHT_ARM_NAMES[self.selected_joint]})
                    print(f"[R1-A7 TEACH] selected joint={self.selected_joint + 1}:{RIGHT_ARM_NAMES[self.selected_joint]}")
                else:
                    print("[R1-A7 TEACH] joint must be 1-7")
            self._print_line_prompt()
            return
        if lower.startswith(("motion ", "m ")):
            self.motion_name = command.split(maxsplit=1)[1].strip()
            self._record_event(
                "motion",
                {"joint": RIGHT_ARM_NAMES[self.selected_joint], "motion": self.motion_name},
            )
            print(f"[R1-A7 TEACH] motion={self.motion_name}")
            self._print_line_prompt()
            return
        if lower.startswith(("segment ", "seg ")):
            parts = command.split(maxsplit=2)
            if len(parts) < 3:
                print("[R1-A7 TEACH] invalid segment. Example: segment 4 elbow_bend_extend")
                self._print_line_prompt()
                return
            try:
                joint_index = int(parts[1]) - 1
            except ValueError:
                print("[R1-A7 TEACH] segment joint must be 1-7")
                self._print_line_prompt()
                return
            if not 0 <= joint_index < len(RIGHT_ARM_NAMES):
                print("[R1-A7 TEACH] segment joint must be 1-7")
                self._print_line_prompt()
                return
            self.selected_joint = joint_index
            self.motion_name = parts[2].strip()
            self.segment_label = f"j{joint_index + 1}_{self.motion_name}"
            self._record_event(
                "segment",
                {
                    "label": self.segment_label,
                    "joint": RIGHT_ARM_NAMES[self.selected_joint],
                    "motion": self.motion_name,
                },
            )
            print(
                f"[R1-A7 TEACH] segment label={self.segment_label} "
                f"joint={self.selected_joint + 1}:{RIGHT_ARM_NAMES[self.selected_joint]} "
                f"motion={self.motion_name}"
            )
            self._print_line_prompt()
            return
        if lower.startswith("step "):
            try:
                self.step_rad = max(0.001, min(0.200, float(command.split(maxsplit=1)[1])))
            except ValueError:
                print("[R1-A7 TEACH] invalid step. Example: step 0.02")
            else:
                self._record_event("step_rad", {"step_rad": self.step_rad})
                print(f"[R1-A7 TEACH] step={self.step_rad:.3f} rad")
            self._print_line_prompt()
            return
        if lower.startswith("large "):
            try:
                self.large_step_rad = max(0.001, min(0.500, float(command.split(maxsplit=1)[1])))
            except ValueError:
                print("[R1-A7 TEACH] invalid large step. Example: large 0.08")
            else:
                self._record_event("large_step_rad", {"large_step_rad": self.large_step_rad})
                print(f"[R1-A7 TEACH] large={self.large_step_rad:.3f} rad")
            self._print_line_prompt()
            return
        if lower.startswith("n "):
            label = command[2:].strip()
            if label:
                self.segment_label = label
                self._record_event("label", {"label": label})
                print(f"[R1-A7 TEACH] label={label}")
            self._print_line_prompt()
            return
        if lower == "n":
            label = self._prompt_line("\nLabel for next segment: ")
            if label:
                self.segment_label = label
                self._record_event("label", {"label": label})
                print(f"[R1-A7 TEACH] label={label}")
            self._print_line_prompt()
            return
        if lower in ("s", "save", "space"):
            self._record_waypoint(state, manual=True)
            self._print_line_prompt()
            return
        if command in ("1", "2", "3", "4", "5", "6", "7", "a", "d", "A", "D", "r", "h", "p", "q"):
            self._handle_key(command, state)
            self._print_line_prompt()
            return
        if all(ch in "1234567adAD" for ch in command):
            for ch in command:
                self._handle_key(ch, state)
            selected = RIGHT_ARM_NAMES[self.selected_joint]
            target = self.command_q[self.selected_joint] if self.command_q is not None else state.q[self.selected_joint]
            print(f"[R1-A7 TEACH] selected={self.selected_joint + 1}:{selected} target={target:+.3f}")
            self._print_line_prompt()
            return
        print(f"[R1-A7 TEACH] unknown command: {command}. Type help for commands.")
        self._print_line_prompt()

    def _sample_common(self, state: ArmState) -> dict:
        t = state.stamp - self.start_monotonic
        command_q = self.command_q if self.command_q is not None else state.q
        return {
            "t": t,
            "recording": self.recording,
            "label": self.segment_label,
            "selected_joint_index": self.selected_joint + 1,
            "selected_joint": RIGHT_ARM_NAMES[self.selected_joint],
            "motion": self.motion_name,
            "q": dict(zip(RIGHT_ARM_NAMES, state.q)),
            "dq": dict(zip(RIGHT_ARM_NAMES, state.dq)),
            "command_q": dict(zip(RIGHT_ARM_NAMES, command_q)),
        }

    def _record_sample(self, state: ArmState) -> None:
        assert self.samples_jsonl is not None
        assert self.csv_writer is not None
        assert self.samples_csv is not None
        sample = self._sample_common(state)
        self.samples_jsonl.write(json.dumps(sample, ensure_ascii=False) + "\n")
        self.samples_jsonl.flush()
        row = {
            "t": sample["t"],
            "recording": self.recording,
            "label": self.segment_label,
            "selected_joint": sample["selected_joint"],
            "selected_joint_index": sample["selected_joint_index"],
            "motion": sample["motion"],
        }
        row.update({f"q_{k}": v for k, v in sample["q"].items()})
        row.update({f"cmd_{k}": v for k, v in sample["command_q"].items()})
        self.csv_writer.writerow(row)
        self.samples_csv.flush()
        self.sample_count += 1

    def _record_waypoint(self, state: ArmState, manual: bool) -> None:
        assert self.waypoints_jsonl is not None
        waypoint = self._sample_common(state)
        waypoint["manual"] = manual
        self.waypoints_jsonl.write(json.dumps(waypoint, ensure_ascii=False) + "\n")
        self.waypoints_jsonl.flush()
        self.waypoint_count += 1
        print(
            f"\n[R1-A7 TEACH] waypoint {self.waypoint_count} saved "
            f"joint={RIGHT_ARM_NAMES[self.selected_joint]} motion={self.motion_name or 'unset'} "
            f"label={self.segment_label}"
        )

    def _record_event(self, event: str, payload: dict) -> None:
        assert self.events_jsonl is not None
        item = {"t": time.monotonic() - self.start_monotonic, "event": event, **payload}
        self.events_jsonl.write(json.dumps(item, ensure_ascii=False) + "\n")
        self.events_jsonl.flush()

    def _print_status(self, state: ArmState, force: bool = False) -> None:
        if self.args.input_mode == "line" and not force:
            return
        now = time.monotonic()
        if not force and now - self.last_print_time < self.args.print_period:
            return
        self.last_print_time = now
        selected = RIGHT_ARM_NAMES[self.selected_joint]
        q_text = " ".join(f"{name}={value:+.3f}" for name, value in zip(RIGHT_ARM_NAMES, state.q))
        cmd = self.command_q if self.command_q is not None else state.q
        cmd_text = " ".join(f"{name}={value:+.3f}" for name, value in zip(RIGHT_ARM_NAMES, cmd))
        print(
            f"\n[R1-A7 TEACH] rec={'ON ' if self.recording else 'OFF'} label={self.segment_label} "
            f"motion={self.motion_name or 'unset'} selected={self.selected_joint + 1}:{selected} "
            f"samples={self.sample_count} waypoints={self.waypoint_count}"
        )
        print(f"[R1-A7 TEACH] q:   {q_text}")
        print(f"[R1-A7 TEACH] cmd: {cmd_text}")
        self.prompt_printed = False
        self._print_line_prompt()

    def _release(self) -> None:
        if self.publisher is not None:
            self._init_low_cmd_stop()
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.publisher.Write(self.low_cmd)
            print("\n[R1-A7 TEACH] released control")

    def _close_files(self) -> None:
        for handle in [self.samples_jsonl, self.waypoints_jsonl, self.events_jsonl, self.samples_csv]:
            if handle is not None:
                handle.close()

    def run(self) -> int:
        self.init()
        sample_dt = 1.0 / max(1.0, self.args.record_hz)
        last_loop = time.monotonic()
        print("[R1-A7 TEACH] waiting for rt/lowstate ...")
        try:
            while not self.done:
                state = self._read_arm()
                if state is None:
                    time.sleep(0.02)
                    continue
                if self.command_q is None:
                    self.command_q = list(state.q)
                    self._record_event("initial_q", {"q": state.q})
                    self._print_line_prompt()

                if self.args.input_mode == "key":
                    key = self._read_key()
                    if key is not None:
                        self._handle_key(key, state)
                else:
                    command = self._read_line_command()
                    if command is not None:
                        self._handle_line_command(command, state)

                now = time.monotonic()
                dt = now - last_loop
                last_loop = now
                if self.args.mode == "jog":
                    self.command_q = self._rate_limit(self.command_q, state.q, dt)
                    self._publish(state, self.command_q)

                if (self.recording or self.args.always_sample) and state.stamp - self.last_sample_time >= sample_dt:
                    self.last_sample_time = state.stamp
                    self._record_sample(state)

                self._print_status(state)
                time.sleep(max(0.0, 1.0 / max(1.0, self.args.hz)))
        finally:
            self._restore_terminal()
            self._release()
            self._close_files()
            print(f"[R1-A7 TEACH] saved output: {self.output_dir}")
            print(f"[R1-A7 TEACH] samples={self.sample_count} waypoints={self.waypoint_count}")

        if self.first_state_time is None:
            print("[R1-A7 TEACH] no lowstate received")
            return 2
        return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="R1-A7 right-arm 7-joint teach-in recorder")
    parser.add_argument("--interface", default="enx9c69d37d0967")
    parser.add_argument("--domain_id", type=int, default=0)
    parser.add_argument("--state_topic", default="rt/lowstate")
    parser.add_argument("--command_topic", default="rt/lowcmd")
    parser.add_argument("--mode", choices=["record", "jog"], default="record")
    parser.add_argument("--enter_debug_mode", action="store_true")
    parser.add_argument("--right_arm_indices", default="22,23,24,25,26,27,28")
    parser.add_argument("--weight_index", type=int, default=31)
    parser.add_argument("--kp", type=float, default=24.0)
    parser.add_argument("--kd", type=float, default=1.4)
    parser.add_argument("--hold_kp", type=float, default=16.0)
    parser.add_argument("--hold_kd", type=float, default=1.0)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--record_hz", type=float, default=20.0)
    parser.add_argument("--print_period", type=float, default=0.5)
    parser.add_argument("--max_speed_rad_s", type=float, default=0.20)
    parser.add_argument("--step_rad", type=float, default=0.030)
    parser.add_argument("--large_step_rad", type=float, default=0.100)
    parser.add_argument("--joint_limits", default="", help="7 chunks like min:max,min:max,... in radians")
    parser.add_argument("--output_dir", default="data/r1a7_teach")
    parser.add_argument("--label", default="unlabeled")
    parser.add_argument("--motion", default="")
    parser.add_argument("--auto_record", action="store_true")
    parser.add_argument("--always_sample", action="store_true", help="write samples even when recording flag is OFF")
    parser.add_argument("--input_mode", choices=["line", "key"], default="line")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.mode == "jog":
        print("WARNING: This will publish rt/lowcmd commands to the real R1-A7 right arm.")
        print("Keep the emergency stop ready and clear the arm workspace.")
        if not sys.stdin.isatty():
            print("[R1-A7 TEACH] jog mode requires an interactive terminal")
            return 2
        answer = input("Type ENABLE to continue: ").strip()
        if answer != "ENABLE":
            print("[R1-A7 TEACH] aborted")
            return 2

    node = R1A7RightArmTeach(args)

    def _handle_signal(_signum, _frame):
        node.done = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    return node.run()


if __name__ == "__main__":
    raise SystemExit(main())
