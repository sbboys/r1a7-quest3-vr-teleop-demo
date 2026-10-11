# R1-A7 手臂控制台方向切换纠错记录

日期：2026-09-30

适用分支：`codex/r1a7-arm-console`

适用控制台：`console3d_hierarchical`，默认端口 `8098`

## 1. 结论

右臂从“上 +Z”切换到“前 +X”时出现的瞬时抖动，直接原因不是 `r1a7_cartesian_stream_control_experimental.py` 缺少连续切换能力，而是原网页在两个方向之间插入了 `jog:stop`。停止后重新 START 会重新建立 Cartesian 运动起点和输出状态，与终端已验证的“不停止直接 U→W”操作不等价。

## 2. 原错误命令序列

原网页长按方向按钮后，松手立即停止：

```text
jog:start:u
jog:keepalive:u
jog:stop
jog:start:w
```

真机日志对应：

```text
[CARTESIAN JOG] STOP UP +Z
[CARTESIAN JOG] START FORWARD +X
```

一次现场日志中，方向重新 START 的瞬间最大关节命令速度约为 `61.3 deg/s`，明显高于稳定连续阶段约 `2~5 deg/s`，并伴随明显的手臂抖动。

## 3. 文档对照

`R1-A7机器人手臂连续控制调试工作汇总_2026-09-19.docx` 记录的有效控制语义是：按一次方向键后持续运动，使用 SPACE/X 明确停止；运动中从 U 切换到 W 时只修改活动方向，保留目标、orientation reference、IK 状态和速度状态连续。

原 GitHub 文档只记录了“松手发送 jog:stop”，没有明确指出这种按住模式不能复现终端 U→W 连续切换测试。这是原操作说明的边界遗漏。

## 4. 修正方案

控制台增加两个显式模式：

- **按住点动**：保留原行为，按住运动，松手发送 `jog:stop`。
- **连续切换**：点击方向后持续运动；点击另一方向直接发送新的 `jog:start:<direction>`，不在方向之间插入 `jog:stop`。

停止按钮、空格、Esc、窗口失焦、页面隐藏和断开连接仍发送 `jog:stop`，不改变安全停止路径。

本次未修改：

- `transfer_control/experimental/r1a7_cartesian_stream_control_experimental.py`；
- IK 权重、重力补偿、末端速度和末端加速度；
- 30 Hz IK 与 250 Hz LowCmd 输出链。

## 5. 正确命令序列

连续切换模式执行“上 +Z”后切换“前 +X”：

```text
jog:start:u
jog:keepalive:u
jog:start:w
jog:keepalive:w
```

预期日志：

```text
[CARTESIAN JOG] START UP +Z
[CARTESIAN JOG] SWITCH UP +Z -> FORWARD +X
```

两个方向之间不应出现：

```text
[CARTESIAN JOG] STOP UP +Z
```

## 6. 验证结果与边界

2026-09-30 真机复测确认，使用连续切换模式后右臂方向切换恢复正常。该结果证明本次切换抖动来自网页命令时序。

这不代表控制脚本的所有 IK 问题均已解决。高位姿态下 `ik_to_goal_mm` 较大、向前时 Z 保持不足或末端姿态误差等现象，应作为独立的 IK 任务优先级问题继续分析，不能与网页 STOP/START 问题混为一谈。
