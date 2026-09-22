#!/usr/bin/env python3
"""Desktop Cartesian jog console for the verified R1-A7 stream controller.

The console intentionally delegates robot communication, IK, LowCmd safety
limits, and the ENABLE gate to r1a7_cartesian_stream_control.py.  Holding a
button sends the same line command repeatedly; releasing it stops sending.
"""

from __future__ import annotations

import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk


ROOT = Path(__file__).resolve().parent
STREAM = ROOT / "r1a7_cartesian_stream_control.py"


class JogConsole(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("R1-A7 Cartesian Jog Console")
        self.geometry("760x620")
        self.minsize(680, 540)
        self.process: subprocess.Popen[str] | None = None
        self.output_queue: queue.Queue[str] = queue.Queue()
        self.running = False
        self.held: set[str] = set()
        self.repeat_job: str | None = None
        self.status = tk.StringVar(value="未启动")
        self.command_status = tk.StringVar(value="等待启动")
        self.host_ip = tk.StringVar(value="192.168.123.223")
        self.interface = tk.StringVar(value="enp6s0")
        self.step_mm = tk.DoubleVar(value=30.0)
        self.repeat_ms = tk.IntVar(value=250)
        self.build_ui()
        self.after(100, self.drain_output)
        self.protocol("WM_DELETE_WINDOW", self.close)

    def build_ui(self) -> None:
        pad = {"padx": 10, "pady": 6}
        top = ttk.LabelFrame(self, text="连接参数")
        top.pack(fill="x", **pad)
        ttk.Label(top, text="网卡").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.interface, width=18).grid(row=0, column=1)
        ttk.Label(top, text="本机 IP").grid(row=0, column=2, sticky="w")
        ttk.Entry(top, textvariable=self.host_ip, width=18).grid(row=0, column=3)
        ttk.Button(top, text="启动控制进程", command=self.start).grid(row=0, column=4, padx=12)
        ttk.Button(top, text="ENABLE", command=self.enable).grid(row=0, column=5)
        ttk.Label(top, textvariable=self.status, foreground="#1b5e20").grid(row=1, column=0, columnspan=6, sticky="w")

        settings = ttk.LabelFrame(self, text="Jog 参数")
        settings.pack(fill="x", **pad)
        ttk.Label(settings, text="水平步长 mm").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(settings, from_=1, to=50, increment=1, textvariable=self.step_mm, width=8).grid(row=0, column=1)
        ttk.Label(settings, text="发送周期 ms").grid(row=0, column=2, sticky="w", padx=(20, 0))
        ttk.Spinbox(settings, from_=100, to=1000, increment=50, textvariable=self.repeat_ms, width=8).grid(row=0, column=3)
        ttk.Label(settings, text="W/S/A/D = X/Y，U/J = Z；松开按钮停止重复发送").grid(row=0, column=4, padx=16)

        motion = ttk.LabelFrame(self, text="右臂笛卡尔 Jog")
        motion.pack(fill="both", expand=False, **pad)
        for col in range(3):
            motion.columnconfigure(col, weight=1)
        buttons = [("前  +X", "w", 1, 1), ("左  +Y", "a", 0, 1), ("右  -Y", "d", 2, 1),
                   ("后  -X", "s", 1, 3), ("上  +Z", "u", 0, 4), ("下  -Z", "j", 2, 4)]
        for label, command, col, row in buttons:
            b = ttk.Button(motion, text=label, width=16)
            b.grid(row=row, column=col, padx=14, pady=10, sticky="ew")
            b.bind("<ButtonPress-1>", lambda _e, c=command: self.press(c))
            b.bind("<ButtonRelease-1>", lambda _e, c=command: self.release(c))

        actions = ttk.Frame(self)
        actions.pack(fill="x", **pad)
        ttk.Button(actions, text="回启动姿态 H", command=lambda: self.send("h")).pack(side="left", padx=5)
        ttk.Button(actions, text="诊断 前+上 I", command=lambda: self.send("i")).pack(side="left", padx=5)
        ttk.Button(actions, text="停止并退出 Q", command=self.stop_process).pack(side="right", padx=5)
        ttk.Label(self, textvariable=self.command_status).pack(anchor="w", padx=20)

        log_frame = ttk.LabelFrame(self, text="控制进程输出")
        log_frame.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(log_frame, height=12, state="disabled", wrap="none")
        self.log.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        scroll.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=scroll.set)

    def start(self) -> None:
        if self.process and self.process.poll() is None:
            return
        if not STREAM.exists():
            messagebox.showerror("文件不存在", str(STREAM))
            return
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        cmd = [sys.executable, "-u", str(STREAM), "--interface", self.interface.get(),
               "--domain-id", "0", "--host-ip", self.host_ip.get(), "--no-enable-gripper",
               "--jog-step-mm", str(self.step_mm.get()),
               "--jog-vertical-step-mm", str(max(1.0, self.step_mm.get() / 3.0))]
        try:
            self.process = subprocess.Popen(cmd, cwd=str(ROOT.parent.parent.parent), env=env,
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True, bufsize=1)
        except OSError as exc:
            messagebox.showerror("启动失败", str(exc))
            return
        self.running = True
        self.status.set("控制进程已启动，等待状态和 ENABLE")
        threading.Thread(target=self.read_output, daemon=True).start()

    def read_output(self) -> None:
        assert self.process and self.process.stdout
        for line in self.process.stdout:
            self.output_queue.put(line.rstrip())
        self.output_queue.put("[process exited]")

    def drain_output(self) -> None:
        while True:
            try:
                line = self.output_queue.get_nowait()
            except queue.Empty:
                break
            self.log.configure(state="normal")
            self.log.insert("end", line + "\n")
            self.log.see("end")
            self.log.configure(state="disabled")
            if "Type ENABLE" in line:
                self.status.set("已连接，等待 ENABLE")
            elif "enabled" in line.lower() or "lowcmd enabled" in line.lower():
                self.status.set("已 ENABLE，可进行 Jog")
        self.after(100, self.drain_output)

    def write(self, text: str) -> None:
        if not self.process or self.process.poll() is not None or not self.process.stdin:
            self.command_status.set("控制进程未运行")
            return
        try:
            self.process.stdin.write(text + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            self.command_status.set("控制进程已断开")

    def enable(self) -> None:
        self.write("ENABLE")
        self.command_status.set("已发送 ENABLE，等待 LowCmd 激活")

    def send(self, command: str) -> None:
        self.write(command)
        self.command_status.set(f"已发送 {command.upper()}")

    def press(self, command: str) -> None:
        if command not in self.held:
            self.held.add(command)
            self.send(command)
        if self.repeat_job is None:
            self.repeat_job = self.after(max(100, self.repeat_ms.get()), self.repeat_held)

    def release(self, command: str) -> None:
        self.held.discard(command)
        self.command_status.set("Jog 已停止" if not self.held else "持续 Jog: " + ",".join(sorted(self.held)).upper())

    def repeat_held(self) -> None:
        self.repeat_job = None
        for command in tuple(self.held):
            self.send(command)
        if self.held:
            self.repeat_job = self.after(max(100, self.repeat_ms.get()), self.repeat_held)

    def stop_process(self) -> None:
        self.held.clear()
        self.send("q")
        self.status.set("已发送退出")

    def close(self) -> None:
        if self.repeat_job:
            self.after_cancel(self.repeat_job)
            self.repeat_job = None
        self.held.clear()
        if self.process and self.process.poll() is None:
            self.write("q")
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.terminate()
        self.destroy()


if __name__ == "__main__":
    JogConsole().mainloop()
