# R1-A7 扳手夹取示教与自主抓取工作汇总（2026-08-25 至 2026-08-27）

## 1. 工作目标

本阶段目标是把 R1-A7 右臂扳手夹取任务从 Quest VR 人工示教推进到可复现的自主关键帧抓取流程。核心要求包括：

- 使用 Quest 手柄采集扳手夹取示教数据。
- 支持夹爪开合和夹持状态记录。
- 从多组示教数据中提取右臂抓取关键帧。
- 让机器人自主执行“先抬臂、高位接近、夹取、保持”的流程。
- 记录每个阶段的测试结果和失败原因，形成可继续迭代的 GitHub 版本。

## 2. VR 示教控制流程

### 2.1 手柄控制与录制

当前 VR 控制脚本为：

- `tools/r1a7_vr_dual_arm_g1ik_real.py`

本阶段围绕该脚本完成了以下工作：

- 保留 R1-A7 自身关节映射，不直接套用 G1 的关节映射。
- 采用 G1 官方风格的目标裁剪方式，避免直接积分导致目标跳变。
- 增加 X 按钮录制流程：按下 X 开始记录，再次按下或到达时长后结束。
- 录制内容包括 `rt/lowstate` 关节状态、命令位置、夹爪命令、左右扳机、按钮状态等。
- 录制根目录为 `r1a7_wrench_project/data/episodes`。
- 夹爪切换为 LowCmd 内控模式，并使用绝对开合范围。

### 2.2 A/B/X 按钮职责

当前建议操作约定：

- 右手 A：开启/关闭机器人手柄控制。
- 左手 X：开始/停止示教数据录制。
- 右手 B：退出 VR 控制并进入回位流程。

其中 B/Home 流程由启动脚本托管：

- `r1a7_wrench_project/scripts/start_r1a7_vr_b_home.sh`

该脚本负责：

- 启动前检查旧 VR 进程和 `8012` 端口。
- 先执行 `AUTO_HOME V2`，通过后再进入 VR。
- 每次 VR 退出后清理独立进程组，避免残留子进程继续占用端口或 LowCmd。
- B 退出后由外层脚本重新执行 verified AUTO_HOME。

## 3. 夹爪控制工作

本阶段夹爪问题主要包括：

- 进入控制后夹爪闭合或无法打开。
- 当前闭合位置被误认为 open 基准。
- 扳机持续压到底时，物体阻挡导致夹爪无法完全闭合，出现抖动或弹出。

已采用的策略：

- 使用 DEX1_1/内控夹爪的绝对电机位置范围，不再用启动瞬间位置作为 open 基准。
- R1-A7 内控夹爪使用 LowCmd/LowState 电机索引 `31,33`。
- 右夹爪 open 约为 `4.80`，闭合/夹持目标在当前流程中使用约 `0.58`。
- 将夹爪闭合从“边移动边闭合”拆成独立关键帧 `CLOSE_GRIPPER_R`，减少抓取时推开扳手的风险。

## 4. AUTO_HOME 与关键帧回位

本阶段建立了多版自动关键帧回位脚本，关键文件包括：

- `r1a7_wrench_project/scripts/auto_move_to_keyframe_v2_verified.py`
- `r1a7_wrench_project/scripts/auto_move_to_keyframe_v2_1.py`
- `r1a7_wrench_project/scripts/auto_move_to_keyframe_segmented_home_v1.py`

最终继续迭代的脚本为：

- `r1a7_wrench_project/scripts/auto_move_to_keyframe_v2_1.py`

关键功能：

- 读取 `rt/lowstate` 当前关节状态。
- 检查起始速度、关节限位、MotionSwitcher 状态。
- 执行 LowCmd 零位移接管测试。
- 支持 `--force-direct` 直接整臂轨迹。
- 支持 `--use-keyframe-gripper` 从关键帧读取夹爪目标。
- 新增 `--sequence`，可在同一个 LowCmd 控制会话中连续执行多个关键帧，最后统一释放控制权。

## 5. 示教数据提取

本阶段基于 8 组扳手夹取示教数据：

- `wrench_grasp_001`
- `wrench_grasp_002`
- `wrench_grasp_003`
- `wrench_grasp_004`
- `wrench_grasp_005`
- `wrench_grasp_006`
- `wrench_grasp_007`
- `wrench_grasp_008`

原始数据目录约 24GB，不适合直接上传 GitHub。本次上传脚本、配置和小体积结果，原始 episode 数据保留在本机。

数据提取脚本：

- `r1a7_wrench_project/scripts/extract_wrench_grasp_reference.py`

V1 生成结果：

- `r1a7_wrench_project/config/wrench_grasp_reference_v1.yaml`
- `r1a7_wrench_project/results/wrench_grasp_reference_v1_report.json`

V1 采用固定时间窗：

- `RIGHT_SAFE`: 1-3 s
- `PRE_GRASP_R`: 6-8 s
- `GRASP_R`: 15-17 s
- `LIFT_R`: 25-29 s

V1 问题：

- `PRE_GRASP_R` 时间窗过早，实际取到了较低的桌面接近段。
- 自主执行时看起来像沿桌面平移，而不是先抬臂后抓取。
- 原始 `LIFT_R` 对右肘和右肩要求过激，当前负载下不稳定。

## 6. V2 高位抓取关键帧

为解决 V1 低位接近问题，新增 V2 高位关键帧配置：

- `r1a7_wrench_project/config/wrench_grasp_reference_v2_high.yaml`

V2 不再只依赖固定时间窗，而是结合 `right_wrist_pose` 末端高度和夹爪状态抽取关键阶段：

- `ARM_UP_R`：右臂抬起，夹爪打开。
- `PRE_GRASP_HIGH_R`：高位预抓取。
- `GRASP_NEAR_R`：接近扳手，但夹爪保持打开。
- `CLOSE_GRIPPER_R`：在接近位单独闭合夹爪。
- `POST_GRASP_HOLD_R`：夹紧后保持/轻微后续姿态。

V2 执行入口：

- `r1a7_wrench_project/scripts/run_wrench_grasp_reference_v2.py`

推荐命令：

```bash
PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv/bin/python \
  r1a7_wrench_project/scripts/run_wrench_grasp_reference_v2.py \
  --execute --assume-yes --stage-threshold 0.12
```

## 7. 关键失败原因与修复

### 7.1 “只沿桌面移动”

现象：

- 机器人看起来没有按示教数据先抬臂，而是在桌面附近平移接近。

原因：

- V1 `PRE_GRASP_R` 固定时间窗取错阶段。
- V2 初版 wrapper 每个关键帧单独启动一个进程。
- 每段结束后 `auto_move_to_keyframe_v2_1.py` 会释放 LowCmd 增益。
- 阶段间等待时手臂受重力回落，下一段又从低位重新开始。

修复：

- 新增 V2 高位关键帧。
- 在 `auto_move_to_keyframe_v2_1.py` 中新增 `--sequence` 连续执行。
- wrapper 默认使用连续 LowCmd 会话，只有显式 `--legacy-subprocess-stages` 才使用旧逐段方式。

### 7.2 原始 LIFT_R 不稳定

现象：

- 原始 `LIFT_R` 执行时右肘和右肩残余误差大。

原因：

- 示教中位数抬起动作对当前负载/夹爪状态过激。
- 右肘目标从抓取状态大幅回收，当前实机难以稳定跟随。

处理：

- V1 临时加入 `LIFT_SAFE_R` 验证保守保持。
- V2 改用 `POST_GRASP_HOLD_R`，避免直接采用过激 `LIFT_R`。

## 8. 测试结果

### 8.1 V1 测试

- `RIGHT_SAFE -> PRE_GRASP_R -> GRASP_R -> LIFT_SAFE_R` 中，部分阶段可达。
- 从 `RIGHT_SAFE` 开始时右肘目标过低，实际误差较大。
- 从 `PRE_GRASP_R` 开始可完成抓取和保持，但路径不是理想的抬臂抓取。

### 8.2 V2 多进程逐段测试

结果：

- 5 个阶段均可达并 PASS。
- 但每段结束释放 LowCmd，阶段间会掉臂。
- 该模式只用于诊断关键帧可达性，不作为最终抓取执行方式。

### 8.3 V2 连续 LowCmd 测试

连续序列：

```text
ARM_UP_R -> PRE_GRASP_HIGH_R -> GRASP_NEAR_R -> CLOSE_GRIPPER_R -> POST_GRASP_HOLD_R
```

结果：

- `KEYFRAME_SEQUENCE_READY = PASS`
- 阶段 2 之后初始误差变成小范围，说明手臂没有阶段间回落。
- 机器人动作逻辑变为先抬臂、高位接近、夹爪闭合、夹持保持。

最近一次测试结果：

- `ARM_UP_R`: PASS，最大误差约 `0.0577 rad`
- `PRE_GRASP_HIGH_R`: PASS，最大误差约 `0.0529 rad`
- `GRASP_NEAR_R`: PASS，最大误差约 `0.0459 rad`
- `CLOSE_GRIPPER_R`: PASS，最大误差约 `0.0461 rad`
- `POST_GRASP_HOLD_R`: PASS，最大误差约 `0.0587 rad`

## 9. 当前推荐运行流程

### 9.1 VR 示教采集

使用：

```bash
PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv/bin/python \
  tools/r1a7_vr_dual_arm_g1ik_real.py \
  --duration 0 \
  --treat_motion_ready_as_fresh \
  --enable_gripper \
  --gripper_mode lowcmd
```

或使用带 B/Home 流程的启动脚本：

```bash
bash r1a7_wrench_project/scripts/start_r1a7_vr_b_home.sh
```

### 9.2 自主抓取回放

使用：

```bash
PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv/bin/python \
  r1a7_wrench_project/scripts/run_wrench_grasp_reference_v2.py \
  --execute --assume-yes --stage-threshold 0.12
```

### 9.3 只做预检

去掉 `--execute`：

```bash
PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv/bin/python \
  r1a7_wrench_project/scripts/run_wrench_grasp_reference_v2.py \
  --stage-threshold 0.12
```

## 10. 暂未最终采用的试验

以下内容已测试或部分实现，但不是当前推荐主路径：

- 直接套用 G1 官方控制器整体逻辑：控制灵活性和 R1-A7 关节/负载条件不完全匹配。
- shoulder pitch 额外前馈：测试中效果不明显，暂未作为主策略。
- Pinocchio/URDF 重力补偿：已预留代码入口，但当前自主抓取以关键帧和连续 LowCmd 保持为主。
- V1 原始 `LIFT_R`：当前负载下不稳定，暂不作为最终抓取末段。
- 多进程逐段关键帧执行：只用于诊断，不用于正式抓取。

## 11. 下一步建议

- 继续采集不同扳手初始位置和夹取姿态的数据，扩展 V2 关键帧库。
- 给 `POST_GRASP_HOLD_R` 增加更明确的离桌高度目标，形成稳定“夹起”动作。
- 加入视觉定位后，把 `GRASP_NEAR_R` 从固定关键帧改为根据扳手位姿偏移。
- 对右肩 pitch 和右肘进行 payload 条件下的误差统计，决定是否再引入前馈补偿。
- 将大体积 episode 数据转移到外部数据仓库或 Git LFS，而不是普通 Git。

## 12. 后续记录

- 2026-08-30 至 2026-09-01 的示教抓取、起始姿态对齐和 VR 双臂控制迁移记录见 `docs/r1a7_wrench_teaching_and_vr_transfer_summary_2026-09-01_zh.md`。
