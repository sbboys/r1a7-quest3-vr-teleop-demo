# R1-A7 扳手示教抓取与 VR 双臂控制迁移汇总（2026-08-30 至 2026-09-01）

## 1. 本阶段目标

本阶段围绕“通过采集数据实现机器人示教抓取扳手，并继续优化 VR 双臂遥操作”展开。核心目标包括：

- 整理前期 8 组扳手夹取示教数据对应的自主抓取流程。
- 修正自主回放中“手臂贴桌面移动、依靠腕部抵桌抬起”的问题。
- 补齐起始姿态对齐和高位抬臂关键帧。
- 迁移另一台电脑上的 R1-A7 VR 双臂控制算法到当前电脑。
- 在迁移版控制基础上补充 A 按钮控制开关、X 按钮录制、DEX1 夹爪内控开合。
- 将本阶段可复用脚本、配置和说明上传到 GitHub。

## 2. 示教数据与自主抓取进展

### 2.1 已使用数据

前期已完成 8 组扳手夹取示教数据采集：

- `wrench_grasp_001`
- `wrench_grasp_002`
- `wrench_grasp_003`
- `wrench_grasp_004`
- `wrench_grasp_005`
- `wrench_grasp_006`
- `wrench_grasp_007`
- `wrench_grasp_008`

后续又补充了第 9 至 12 组数据，用于分析起始姿态对齐问题：

- `wrench_grasp_009`
- `wrench_grasp_010`
- `wrench_grasp_011`
- `wrench_grasp_012`

原始 episode 数据体积较大，不直接上传 GitHub；本次上传脚本、关键帧配置、说明文档和小体积结果。

### 2.2 自主回放主要问题

测试中观察到：

- 机器人没有按示教轨迹先抬起右臂再接近扳手。
- 右臂会沿桌面方向移动，随后依靠腕部或夹爪抵住桌面产生被动抬起。
- 当前实机起始姿态与采集数据起始姿态不一致，导致第一段轨迹从错误姿态开始。
- 逐段执行关键帧时，每段释放 LowCmd 后手臂会受重力或负载影响回落。

判断原因：

- V1 固定时间窗抽取的 `PRE_GRASP_R` 过早，包含低位接近段。
- 初始姿态没有先对齐到示教时的右臂抬起姿态。
- 多进程逐段回放会在阶段间释放控制权。
- 当前肩肘负载条件下，过激的后续抬升关键帧不稳定。

## 3. 自主抓取脚本与配置更新

### 3.1 关键配置

当前继续使用的关键帧配置：

- `r1a7_wrench_project/config/wrench_grasp_reference_v2_high.yaml`

本阶段新增或调整的阶段：

- `TEACH_START_R`：使用第 9 至 12 组数据的前 3 秒活动状态，作为示教起始姿态对齐。
- `ELBOW_UP_R`：先肘部抬起，避免直接贴桌面平移。
- `ARM_UP_R`：高位右臂姿态。
- `PRE_GRASP_HIGH_R`：保持高位进行目标接近前准备。
- `GRASP_NEAR_R`：下降到接近扳手位置，夹爪保持打开。
- `CLOSE_GRIPPER_R`：在抓取位单独闭合夹爪。
- `POST_GRASP_LIFT_R`：夹取后抬起。
- `PLACE_DOWN_R`：放回目标位置。
- `OPEN_GRIPPER_R`：松开夹爪。

### 3.2 执行脚本

当前自主回放入口：

- `r1a7_wrench_project/scripts/run_wrench_grasp_reference_v2.py`

底层关键帧执行脚本：

- `r1a7_wrench_project/scripts/auto_move_to_keyframe_v2_1.py`

本阶段脚本更新：

- `run_wrench_grasp_reference_v2.py` 默认连续执行完整序列，避免阶段间掉臂。
- 支持 `--start-at` 从指定阶段继续执行。
- 自动传入 `--hold-left-current`，右臂示教回放时左臂保持当前测量姿态，不强行回到无关 HOME。
- `auto_move_to_keyframe_v2_1.py` 新增 `--hold-left-current`，在加载关键帧后用当前左臂姿态覆盖目标左臂关节。

推荐自主抓取演示命令：

```bash
PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv/bin/python \
  r1a7_wrench_project/scripts/run_wrench_grasp_reference_v2.py \
  --execute --assume-yes --stage-threshold 0.12
```

只做预检时去掉 `--execute`。

## 4. VR 双臂控制迁移

### 4.1 迁移来源

本阶段接入另一台电脑上的迁移包：

- `/home/robot/R1A7_VR_dual_arm_transfer_20260831_001`

原始说明已复制到仓库：

- `docs/r1a7_vr_dual_arm_transfer_readme_2026-09-01_zh.md`

关键迁移脚本已复制到仓库：

- `r1a7_wrench_project/transfer_control_20260831/tools/run_r1a7_unitree_official_vuer_real_lowcmd.sh`
- `r1a7_wrench_project/transfer_control_20260831/tools/run_r1a7_unitree_official_vuer_real_lowcmd_record.sh`
- `r1a7_wrench_project/transfer_control_20260831/tools/r1a7_unitree_official_vuer_real_lowcmd.py`
- `r1a7_wrench_project/transfer_control_20260831/tools/r1a7_unitree_official_vuer_real_lowcmd_record.py`

### 4.2 当前机器适配

迁移脚本在当前电脑上完成以下适配：

- Conda 路径改为 `/home/robot/miniconda3/bin/conda`。
- 固定使用 `tv` 环境。
- 增加 `PYTHONNOUSERSITE=1`，避免用户级 Python 包污染 conda 环境。
- 机器人 DDS 网口使用 `enp6s0`。
- Quest/路由侧 IP 使用当前固定地址 `192.168.1.103`。
- VR 浏览器入口：

```text
https://192.168.1.103:8012/?ws=wss://192.168.1.103:8012
```

### 4.3 控制功能

迁移版控制当前功能：

- Quest 进入 VR 后，右手 A 控制机器人双臂遥操作开启/关闭。
- A 关闭后保持当前命令姿态，不再继续跟随手柄。
- VR 退出或连接中断后，机器人不继续接受旧手柄目标。
- 左手 X 开始/停止 CSV 数据记录。
- DEX1 夹爪通过 LowCmd 内控电机索引 `31,33` 控制。
- 左右扳机控制夹爪开合：扳机释放为打开，扳机压下为闭合。

X 按钮录制默认保存位置：

```text
/home/robot/R1A7_VR_dual_arm_transfer_20260831_001/unitree_sdk2-main/datasets/r1a7_vr_button_records/
```

上传到仓库内的迁移脚本保留了这些逻辑，后续可以把它们正式合并到主 `tools/` 控制入口。

## 5. 当前状态

已完成：

- 示教抓取关键帧流程整理。
- V2 高位抓取序列更新。
- 起始姿态对齐阶段补齐。
- 左臂保持当前姿态逻辑补齐。
- 迁移版 VR 控制脚本在当前机器可启动。
- A 按钮开关控制、X 按钮录制、DEX1 夹爪内控开合已集成到迁移版脚本。

仍需继续验证：

- X 按钮录制生成的 episode 是否包含完整 arm/gripper/button 字段。
- 扳机控制夹爪的实际开合方向和绝对范围是否需要针对当前夹爪重新微调。
- 自主抓取完整序列是否能稳定实现“先抬肘、再高位接近、夹取、抬起、放下”。
- 是否将迁移版 `R1A7_ArmIK` 控制正式替换当前主仓库的旧 G1IK 控制入口。

## 6. 下一步建议

1. 使用迁移版 VR 控制采集 3 至 5 组新的扳手抓取数据，要求从“右臂抬起、肘部弯曲”的一致起始姿态开始。
2. 每组数据用 X 按钮开始/停止，确保夹爪扳机动作被记录。
3. 重新抽取关键帧，优先保留真实的高位起始、下降夹取、闭合、抬起、放下阶段。
4. 用 dry-run 检查关键帧可达性，再做低速 `--execute`。
5. 如果自主回放仍贴桌面移动，优先检查起始姿态和第一阶段 `TEACH_START_R/ELBOW_UP_R`，不要直接调大腕部动作。
