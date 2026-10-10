# R1-A7 本体运动平滑性基准测试（无视觉）

目标：先把视觉、Fusion、连续 Cartesian IK 全部排除，只判断 **LowCmd / PD / 机械传动** 与 **one-shot IK + 轨迹执行** 是否平稳。

## 基线原则

- 默认 `500 Hz` LowCmd（2 ms），对应 2026-09-22 宇树 R1-A7 低层示例。
- 14 个手臂关节使用宇树示例增益：肩/肘 `Kp=100, Kd=2`；三腕 `Kp=50, Kd=2`。
- `dq_target=0`，`tau_ff=0`。
- 轨迹采用五次 minimum-jerk：`10s^3 - 15s^4 + 6s^5`。
- 不使用视觉，不使用连续 IK。
- MotionSwitcher 状态无法确认时直接拒绝测试。
- 使用项目已有 `r1a7_lowcmd_guard.py`，拒绝第二个已知 `rt/lowcmd` 发布者。

## 文件放置

建议放到：

```text
r1a7_threading_project/transfer_control/diagnostics/
  r1a7_arm_smoothness_baseline.py
  analyze_r1a7_arm_smoothness.py
```

## 0. 只计算、不运动（推荐先做）

从项目根目录：

```bash
PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv/bin/python \
  r1a7_threading_project/transfer_control/diagnostics/r1a7_arm_smoothness_baseline.py \
  --interface enp6s0 \
  --test cartesian \
  --axis z \
  --step-mm 2 \
  --plan-only
```

检查打印的 `goal joint delta deg`。脚本默认拒绝最大关节变化超过 `8 deg` 的 one-shot IK 目标。

## 1. Test A：单关节 LowCmd 平滑基线

第一轮建议右腕 pitch，仅 `1 deg`，2.5 s 去、2.5 s 回：

```bash
PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv/bin/python \
  r1a7_threading_project/transfer_control/diagnostics/r1a7_arm_smoothness_baseline.py \
  --interface enp6s0 \
  --test joint \
  --joint R_WRIST_PITCH \
  --amplitude-deg 1.0 \
  --duration-s 2.5
```

只有终端输入 `RUNSMOOTH` 后才会发 LowCmd。

**判定：**
- 如果这一项已经明显抖动：暂时不要继续 IK。优先查 LowCmd 周期抖动、PD、MotionSwitcher、DDS、关节机械/摩擦。
- 如果这一项平滑：继续 Test B。

## 2. Test B：one-shot IK + minimum-jerk，+Z 2 mm

```bash
PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv/bin/python \
  r1a7_threading_project/transfer_control/diagnostics/r1a7_arm_smoothness_baseline.py \
  --interface enp6s0 \
  --test cartesian \
  --axis z \
  --step-mm 2 \
  --duration-s 3.0
```

这里 IK 只在开始前计算一次。执行阶段只有关节 minimum-jerk 轨迹，因此能把“实时 IK 冗余解变化”完全排除。

## 3. Test C：one-shot IK + minimum-jerk，+X 2 mm

```bash
PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv/bin/python \
  r1a7_threading_project/transfer_control/diagnostics/r1a7_arm_smoothness_baseline.py \
  --interface enp6s0 \
  --test cartesian \
  --axis x \
  --step-mm 2 \
  --duration-s 3.0
```

若 2 mm 平滑，再逐步到 3 mm、5 mm；不要一开始扩大行程。

## 4. 日志分析

每次运行自动写：

```text
/tmp/r1a7_arm_smoothness_*.csv
```

分析：

```bash
PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv/bin/python \
  r1a7_threading_project/transfer_control/diagnostics/analyze_r1a7_arm_smoothness.py \
  /tmp/r1a7_arm_smoothness_xxx.csv
```

输出包括：

- tracking error RMS
- dq RMS
- acceleration RMS
- jerk RMS
- command ATV
- actual ATV

## 结果判读

### A 抖，B/C不用继续

说明问题已经存在于：

```text
500 Hz smooth q_cmd -> LowCmd -> PD -> real joint
```

优先排查底层、PD、DDS、机械摩擦/间隙，而不是 IK/视觉。

### A 平滑，B/C 抖

说明问题与多关节协同/one-shot IK 目标或不同关节动态响应有关。重点看 `q_cmd` 平滑但哪些 `q_actual/dq_actual` 抖。

### A/B/C 都平滑，但当前 Continuous Cartesian 抖

这是最有价值的结果：基本把问题定位到当前的

```text
30 Hz continuous IK -> raw_stream_solution -> 250 Hz speed-only slew
```

下一步才修改现有 Cartesian 控制器：真正接入 joint acceleration limiter，再做 jerk limiter，并对 `q_IK_raw/q_cmd/q_actual` 做 A/B。

## 安全边界

1. 同一时刻只能有一个 `rt/lowcmd` 发布者。
2. MotionSwitcher 检查失败或仍有运动服务时不运行。
3. 实体急停必须在手边，双臂工作空间清空。
4. 第一轮只用 1 deg 单关节和 2 mm one-shot IK。
5. 任意异常立即 Ctrl+C / 急停；不要为了“看清抖动”提高速度或行程。
6. 先不启用 gravity feedforward；基线通过后再单独做 `0` vs `0.25` A/B。
