# R1-A7 相机遥操作精准度调参

## 1. 先用预览模式看输入是否稳定

```bash
cd /home/robot/unitree_sim_isaaclab
./scripts/run_r1a7_camera_real_preview.sh --show
```

保持手臂不动，观察日志：

- `in=(...)` 应接近 0。如果手不动仍明显变化，增大 `--input_deadband` 或降低 `--camera_filter_alpha`。
- `SAT=none` 最好。如果经常出现 `SAT=关节名`，说明目标被限位裁剪，应降低对应 gain 或扩大安全限位。
- `grip=...` 在张手时应接近 0，握拳时应接近 1。
- `dex1_cmd=none` 表示没有收到 Dex1_1 状态，夹爪不会执行命令。先启动 Dex1_1 服务，并确认 `DEX1_DDS_OK`。

真机脚本默认使用：

```text
--robot_reference_q current
```

也就是以当前机器人姿态作为人体零点。这样人体手臂不动时，机器人不会先回到仿真姿态。若改回固定仿真参考姿态，启动后机器人会主动移动到该参考附近。

## 2. 手臂幅度不准

整体幅度小：

```bash
./scripts/run_r1a7_camera_real_control.sh --show --amplitude_scale 1.2
```

整体幅度大：

```bash
./scripts/run_r1a7_camera_real_control.sh --show --amplitude_scale 0.8
```

单方向不准时分别调：

- 左右：`--roll_gain`、`--yaw_gain`
- 上下：`--pitch_gain`
- 前后：`--depth_gain`、`--pitch_depth_ratio`、`--elbow_depth_ratio`
- 肘部弯曲：`--skeleton_elbow_gain`

当前真机脚本默认使用 `--coordination_mode anatomic`。这个模式下：

- 肩关节主要跟随人体大臂方向，也就是肩到肘的姿态。
- 肘关节主要跟随人体肘部弯曲。
- 手腕位移只做辅助，不再和骨架目标简单相加。

当前 R1-A7 实测为：`right_shoulder_pitch` 正方向会让大臂向后摆。因此真机脚本默认设置：

```text
--shoulder_pitch_sign 1.0
--skeleton_lift_sign -1.0
--depth_pitch_sign -1.0
```

人体大臂上抬时，骨架抬臂特征会推动机器人肩 pitch 往负方向走，避免大臂向后摆。
人体手臂前伸时，深度项同样推动肩 pitch 往负方向走，形成向前伸展趋势。

如果整臂动作仍然像“小臂抢动作”，提高骨架权重：

```bash
./scripts/run_r1a7_camera_real_control.sh --show \
  --skeleton_blend 0.85 \
  --elbow_skeleton_blend 0.95 \
  --elbow_vertical_ratio 0.0 \
  --elbow_depth_ratio 0.0
```

如果动作太僵硬、手腕位置跟随不足，降低骨架权重：

```bash
./scripts/run_r1a7_camera_real_control.sh --show \
  --skeleton_blend 0.55 \
  --elbow_skeleton_blend 0.65
```

如果只有上下抬压、没有前伸趋势，重点看日志里的 `reach=...` 和 `SAT=right_elbow...`。

- `reach` 有明显正值但肩 pitch 不往负方向走：确认 `--depth_pitch_sign -1.0`。
- 经常出现 `SAT=right_elbow:...->+0.55`：肘关节被伸直到下限。当前脚本默认 `--elbow_min 0.55`，用于避免前伸时小臂被拉成过直状态。
- 经常出现 `SAT=right_elbow:+1.70->+1.55` 或类似内容：人体弯肘已经被识别到，但机器人肘部被上限裁剪。当前脚本把真机肘部上限提高到 `--elbow_max 1.75`，并给肘部单独设置了更快的 `--elbow_max_speed_rad_s`、`--elbow_direct_max_step`、`--elbow_max_command_lead`。
- 当前真机/预览脚本加载 `data/r1a7_teach/latest_profile.json` 时会启用 `--teach_elbow_limits`，用你记录的 `elbow_bend_extend` 示教范围自动确定肘部上下限。启动日志会显示 `teach elbow limits: ...` 和最终 `elbow config: min=... max=...`。
- 如果人体弯肘时机器人反而伸直，或者人体伸直时机器人反而弯肘，说明肘部识别方向反了。需要同时翻转 `--skeleton_elbow_sign` 和 `--teach_elbow_sign`，当前真机脚本默认使用 `--skeleton_elbow_sign 1.0 --teach_elbow_sign 1.0`。
- 当前脚本还使用 `--robot_reference_q current`，前伸会从真机当前弯曲姿态开始计算，避免一启动就把肘部拉到仿真参考值。
- 前伸幅度仍偏小：增大深度项，并让肘部深度项更明显。

```bash
./scripts/run_r1a7_camera_real_control.sh --show \
  --depth_gain 3.2 \
  --pitch_depth_ratio 0.85 \
  --elbow_depth_ratio -0.55 \
  --elbow_min 0.35
```

如果前伸时大臂方向正确，但肘部太早打直导致动作像一根直杆，应保守一些。当前脚本默认使用这个方向：

```bash
./scripts/run_r1a7_camera_real_control.sh --show \
  --elbow_depth_ratio -0.25 \
  --elbow_min 0.55 \
  --elbow_skeleton_blend 0.80
```

## 2.2 左右方向相反

当前脚本默认保留相机画面方向，不再加 `--mirror_input`。如果人手在相机画面里向左，机器人应向画面左侧对应方向运动。

如果真实效果仍然左右相反，再临时加回：

```bash
./scripts/run_r1a7_camera_real_control.sh --show --mirror_input
```

如果抬到一定高度后日志出现：

```text
target=no  human_delta_too_large
```

说明人体动作幅度超过了安全输入范围。当前脚本默认 `--max_human_delta_m 1.20`，仍不够时可以临时放宽：

```bash
./scripts/run_r1a7_camera_real_control.sh --show --max_human_delta_m 1.50
```

## 2.1 大臂不抬，只是小臂动

大臂抬起主要由 `right_shoulder_pitch` 控制。日志里看这两个值：

```text
current: right_shoulder_pitch=...
cmd:     right_shoulder_pitch=...
tgt:     right_shoulder_pitch=...
```

`tgt` 是相机映射后的目标，`cmd` 是经过速度和跟随余量限制后的实际命令。预览模式下机器人不运动，所以 `cmd` 可能被限制在当前关节附近；判断映射是否足够，应优先看 `tgt` 和 `SAT`。

如果人手上抬时 `tgt right_shoulder_pitch` 没有明显变化，说明肩关节输入太弱，应增大：

```bash
./scripts/run_r1a7_camera_real_control.sh --show \
  --pitch_gain 8.5 \
  --skeleton_lift_gain 2.2
```

如果人手上抬时 `cmd/tgt right_shoulder_pitch` 变化方向反了，切换方向。当前 R1-A7 实测为：`right_shoulder_pitch` 正方向会让大臂向后摆，因此相机控制默认使用：

```bash
./scripts/run_r1a7_camera_real_control.sh --show --shoulder_pitch_sign 1.0
```

如果大臂还没抬起来，日志已经出现：

```text
SAT=right_shoulder_pitch:...
```

说明肩关节被安全限位裁剪。确认机械空间安全后再小幅扩大：

```bash
./scripts/run_r1a7_camera_real_control.sh --show \
  --pitch_min -1.20 \
  --pitch_max 1.55
```

如果小臂抢动作，降低肘部耦合：

```bash
./scripts/run_r1a7_camera_real_control.sh --show \
  --elbow_vertical_ratio 0.0 \
  --skeleton_elbow_gain 0.25 \
  --skeleton_lift_elbow_gain 0.0
```

如果 `tgt` 已经明显上抬，但真机 `current right_shoulder_pitch` 跟不上，增加肩关节跟随速度和命令余量：

```bash
./scripts/run_r1a7_camera_real_control.sh --show \
  --shoulder_max_speed_rad_s 0.60 \
  --shoulder_direct_max_step 0.10 \
  --shoulder_max_command_lead 0.70 \
  --kp 28.0 \
  --kd 1.5
```

也可以先脱离相机，单独测试右肩 pitch 电机是否能被低层命令驱动：

```bash
./scripts/test_r1a7_right_shoulder_pitch_lowcmd.sh
```

这个脚本只对 `right_shoulder_pitch` 做小幅正弦测试，其他右臂关节保持当前状态。若这个测试里大臂仍不明显运动，问题不在相机映射，而在机器人模式、低层控制权限、肩关节力矩/保护或关节索引。

如果小幅正弦测试只有轻微上抬，继续做更明确的右肩上抬斜坡测试：

```bash
./scripts/test_r1a7_right_shoulder_pitch_lift_lowcmd.sh
```

当前 R1-A7 实测为：正 `right_shoulder_pitch` 会让大臂向后摆，所以该脚本默认发送负方向 `--lift_offset_deg -35.0`。

日志会打印：

```text
q:   当前关节角
cmd: 实际发布命令
tgt: 目标关节角
```

如果 `cmd/tgt` 明显上升，但 `q right_shoulder_pitch` 只轻微变化，说明肩关节本体跟随不足，应优先检查机器人模式、关节保护、力矩限制，或继续提高 `--kp`、`--max_speed_rad_s`。如果 `q` 能随 `cmd/tgt` 上升，说明大臂本体控制正常，再回到相机参数调节。

## 3. 手臂抖动或漂移

更稳但更慢：

```bash
./scripts/run_r1a7_camera_real_control.sh --show \
  --camera_filter_alpha 0.12 \
  --input_deadband 0.025 \
  --depth_deadband 0.040 \
  --skeleton_deadband 0.050
```

更灵敏但更容易抖：

```bash
./scripts/run_r1a7_camera_real_control.sh --show \
  --camera_filter_alpha 0.30 \
  --input_deadband 0.005 \
  --depth_deadband 0.010
```

## 4. 夹爪开合不准

现在二值夹爪使用滞回：

- `--dex1_grip_close_threshold`：握拳分数超过此值才闭合。
- `--dex1_grip_open_threshold`：张手分数低于此值才打开。

误夹紧：提高闭合阈值，例如：

```bash
./scripts/run_r1a7_camera_real_control.sh --show \
  --dex1_grip_close_threshold 0.78 \
  --dex1_grip_open_threshold 0.35
```

夹爪不容易夹紧：降低闭合阈值，例如：

```bash
./scripts/run_r1a7_camera_real_control.sh --show \
  --dex1_grip_close_threshold 0.58 \
  --dex1_grip_open_threshold 0.30
```

## 5. Dex1_1 行程不准

先单独测试行程：

```bash
./scripts/test_dex1_1_right_gripper.sh --open_q 5.40 --close_q 0.0
```

如果打开不够，增大 `--open_q`；如果闭合过紧或撞限位，增大 `--close_q`，例如 `--close_q 0.3`。
