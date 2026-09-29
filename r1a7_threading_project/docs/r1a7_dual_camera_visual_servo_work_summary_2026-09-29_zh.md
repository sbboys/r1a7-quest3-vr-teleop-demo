# R1 A7 双相机视觉伺服开发工作汇总

更新时间：2026-09-29
适用项目：R1-A7 双臂机器人镊子夹取细线并进行穿孔
记录来源：2026-09-24 至 2026-09-28 实验记录，以及 2026-09-29 工作汇总

## 1. 文档目的

本记录把两份阶段文档中的实验过程、代码文件、运行命令、验证结果和后续风险统一整理，作为 GitHub 分支的交接入口。本文区分已经实测或接入的内容、仍在验证的内容和下一步计划，不把计划描述成最终验收结果。

本阶段只覆盖“外部相机 + 腕部 Eye-in-Hand 相机”的镊尖和小孔检测、跟踪、融合与视觉伺服，不迁移扳手拧螺母任务。

## 2. 系统架构

```text
外部相机粗定位 ─┐
                 ├─ 双相机融合 ─ JSON 速度 ─ Cartesian/IK ─ LowCmd ─ R1-A7 右臂
腕部相机精定位 ─┘
```

| 模块 | 主要职责 | 当前入口 |
|---|---|---|
| External | ArUco、镊尖粗定位、目标孔选择、X/Z 速度 | `vision_dual_camera/task6_threading/task6_external_ibvs_v4_aruco_edge.py` |
| Wrist | Orbbec 图像、SAM2/PCA、Tip RefinerV2、孔检测、Y/Z 速度 | `vision_dual_camera/task6_threading/task6_wrist_ibvs_controller_v2_tip.py` |
| Fusion | 读取两个 JSON、判断来源有效性、单源接管、双源失效保护、输出统一 XYZ 速度 | `vision_dual_camera/task6_threading/task6_dual_ibvs_fusion.py` |
| Robot control | Cartesian 速度转目标、IK、250 Hz LowCmd | `transfer_control/tools/r1a7_cartesian_stream_control.py` |

两个视觉节点分别输出：镊尖像素、孔中心像素、`tip_state`、`hole_state`、置信度、`du/dv` 和建议速度。Fusion 输出 `/tmp/r1a7_dual_ibvs_velocity.json`，机器人控制器读取该文件。视觉节点和机器人控制器不能与其他程序同时占用 `rt/lowcmd`。

## 3. 主要开发工作

### 3.1 外部相机镊尖稳定化

原问题是 ArUco/局部 Tip 结果偶发丢失或大幅跳变，错误像素会直接进入 IBVS。当前 V4 处理链保留 ArUco 粗定位、局部精修和 Edge fallback，并增加跳变门控：

- 正常门控：`EXT_TIP_NORMAL_GATE_PX=25`；
- 中等跳变进入 `RELOCK`，连续 `3` 帧空间一致后重新锁定；
- 大于约 `80 px` 的结果拒绝；
- 保留 `TRACK`、`RELOCK`、`CATASTROPHIC_REJECT` 状态。

阶段实测结论：外部相机 Tip 丢失和错误大跳变明显减少。另增加 `External Tip Hold`，短时无候选时使用最近可信位置和速度预测，避免单帧丢失直接令 External 失效。

### 3.2 腕部相机镊尖稳定化

腕部相机采用 SAM2 分割、PCA 方向估计和 `TweezerTipRefinerV2` 精修。针对 Tip 跳变和 Refiner 短时丢失，当前参数和逻辑包括：

- `TIP_EMA_ALPHA=0.4`；
- `TIP_MAX_JUMP_PX=8`；
- `TIP_RELOCK_FRAMES=3`；
- `TIP_RELOCK_CLUSTER_RADIUS_PX=5`；
- `TIP_HOLD_S=0.30 s`，在短时 `REFINER LOST` 时按最近 Tip 和速度模型预测输出。

阶段实测确认 SAM2 模型能够加载，Tip 可经过 `SAM2 mask -> PCA -> RefinerV2 -> Tip pixel` 输出。仍需在机械臂持续运动和靠近小孔时继续验证 Hold 是否覆盖所有失效分支。

### 3.3 腕部相机小孔检测升级

原模块主要依赖 `cv2.HoughCircles`。约 1 mm 小孔在图像中只有约 1 至 3 px，Hough 对小半径、弱边缘和透视椭圆不稳定。

当前 V2/V3 路线优先使用：

```text
Canny -> findContours -> 面积过滤 -> fitEllipse -> 圆度/椭圆比例过滤 -> 孔中心
```

检测失败时保留 Hough fallback 和 Hold 逻辑。第一份汇总记录了实机可以输出 `hole=OK` 和 `H=(u,v)`；但靠近目标并随机械臂运动时仍出现 `LOST`、轮廓跳变和 NaN 风险，因此当前状态为“已接入，继续稳定化”，不是最终验收完成。

### 3.4 外部相机小孔检测升级

外部相机原逻辑由鼠标点击固定孔中心，并不是真正自动检测。当前点击只提供粗略 ROI 中心，随后使用 Canny、轮廓和椭圆拟合精修；失败时回退到点击点，保证控制链不中断。

当前约束：ROI 从约 `±60 px` 收紧到 `±30 px`，椭圆比例阈值提高到 `0.65`，候选中心必须位于点击点约 `25 px` 内。回退状态显示 `EXT TARGET LOCK FALLBACK`，后续需要降低该状态的 `hole_confidence`，避免把人工点击回退误认为高置信自动检测。

### 3.5 Dual Fusion 鲁棒融合

Fusion 不再只依据 `enabled + timestamp` 判断视觉源有效性，还要求 Tip 和 Hole 置信度不低于 `0.5`。当前策略：

| 模式 | 条件 | 输出策略 |
|---|---|---|
| `DUAL` | External、Wrist 均有效 | External 提供 X，Wrist 提供 Y，Z 按置信度动态融合 |
| `EXTERNAL_ONLY` | 仅 External 有效 | 控制 X/Z，Y 置零并限速 |
| `WRIST_ONLY` | 仅 Wrist 有效 | 控制 Y/Z，X 置零并限速 |
| `BLIND_GRACE` | 双源短时失效且不在近目标区 | 低速衰减保持最近安全速度，Y 置零 |
| `STOP` | 双源失效超时或近目标区不允许盲动 | 速度归零，`ready=False` |

Z 方向冲突时优先置信度较高的一侧。外部相机和腕部相机视角不同，不能直接比较两个 `hole_pixel` 或 `tip_pixel` 是否一致；目前只比较健康状态、置信度和共同控制轴 Z 的方向。

### 3.6 Cartesian 视觉控制停止条件

实验发现机器人接近目标孔时会停止，但两个相机尚未完全重合且 Fusion 仍有速度输出。原因包括：

- 通用 Cartesian 的低速 snap 逻辑会把小于约 `0.5 mm/s` 的速度清零；
- 旧的 `IBVS_MAX_COMMAND_LAG_M` 约束会在目标跟随过程中提前停止。

当前控制脚本的视觉分支增加 `not ibvs_velocity_active` 条件，使普通 Cartesian 仍保留原停止逻辑，而 IBVS 速度不被该 snap 误清零；视觉命令滞后约束按阶段记录设置为不限制。该修改需要后续在无负载、小行程、低速条件下复核安全性。

## 4. 环境与文件操作记录

### 4.1 版本核验

```bash
cd ~/unitree_sim_isaaclab_threading/r1a7_threading_project
WRIST=vision_dual_camera/task6_threading/task6_wrist_ibvs_controller_v2_tip.py
stat "$WRIST"
sha256sum "$WRIST"
grep -nE 'Sam2PcaTweezerDetector|TweezerTipRefinerV2|TIP_EMA_ALPHA|TIP_MAX_JUMP_PX' "$WRIST"
```

### 4.2 统一运行环境

三路视觉程序统一使用 `tv_sam2` 环境，避免系统 Python 与 Conda Python 的 OpenCV/ArUco API 不一致：

```bash
PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv_sam2/bin/python <script>
```

当前记录确认系统 Python OpenCV 4.5.4 与 `tv_sam2` OpenCV 4.10.0 的 ArUco API 不同；后续不要混用解释器。

### 4.3 三路启动

```bash
cd ~/unitree_sim_isaaclab_threading/r1a7_threading_project

# External
PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv_sam2/bin/python \
  vision_dual_camera/task6_threading/task6_external_ibvs_v4_aruco_edge.py

# Wrist
PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv_sam2/bin/python \
  vision_dual_camera/task6_threading/task6_wrist_ibvs_controller_v2_tip.py

# Fusion
PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv_sam2/bin/python \
  vision_dual_camera/task6_threading/task6_dual_ibvs_fusion.py
```

检查输出：

```bash
cat /tmp/r1a7_ibvs_velocity.json
cat /tmp/r1a7_wrist_ibvs_velocity.json
cat /tmp/r1a7_dual_ibvs_velocity.json
```

### 4.4 语法和文件完整性检查

```bash
PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv_sam2/bin/python -m py_compile \
  vision_dual_camera/task6_threading/task6_external_ibvs_v4_aruco_edge.py \
  vision_dual_camera/task6_threading/task6_wrist_ibvs_controller_v2_tip.py \
  vision_dual_camera/task6_threading/task6_dual_ibvs_fusion.py \
  transfer_control/tools/r1a7_cartesian_stream_control.py
```

本次归档使用 `stat`、`sha256sum`、`grep`、`cat`、`py_compile` 等只读或语法检查操作；没有在归档过程中运行真机运动，也没有启动视觉节点占用相机或 `rt/lowcmd`。

## 5. 当前结果与剩余风险

已经完成或已接入：

1. External/Wrist 双相机并行数据流；
2. 腕部 SAM2/PCA/RefinerV2 Tip 跟踪；
3. External Tip 跳变保护和短时 Hold；
4. Wrist Tip 跳变门控、RELOCK 和 Hold；
5. Wrist/External Hole 椭圆拟合优先路径；
6. Fusion 置信度有效性、单源接管、双源 Grace/STOP；
7. Cartesian 视觉分支的低速停止问题定位和代码接入；
8. 当前活动脚本、标定文件和原始 Word 记录的 GitHub 归档。

仍需验证：

1. Wrist Hole 的 `circles is None` 和 `best is None` 是否全部进入预测保持；
2. 机械臂运动时 Hole 是否继续出现大跳变或 NaN；
3. External fallback、HOLD、RELOCK 是否应映射为不同置信度；
4. 视觉误差很小时的最小速度、deadband 和低速收敛是否产生振荡；
5. DUAL、EXTERNAL_ONLY、WRIST_ONLY、BLIND_GRACE、STOP 的现场切换；
6. 在完成小范围视觉引导验收前，不进入完整穿孔和高速插入。

## 6. 后续建议顺序

1. 不接机器人运动，先运行 Wrist 并验证 Hole Hold、NaN 防护和状态输出。
2. 运行 External + Wrist + Fusion，检查模式切换、置信度和动态 Z 权重。
3. 采用低速、小行程测试，确认视觉速度不会被 Cartesian snap 清零。
4. 完成 Tip bias、Eye-in-Hand 和固定相机坐标一致性复核。
5. 增加接近目标 deadband、低误差降速和插入阶段的接触/超时保护。
6. 通过单孔重复实验后，再开展多孔穿孔成功率统计。

## 7. 安全边界

- 视觉节点不能与其他控制程序同时占用 `rt/lowcmd`。
- 第一次联调必须关闭机器人运动或使用极小 Cartesian 速度。
- 双相机同时失效、目标接近区检测失效、孔状态不可信时必须停止，而不是盲目继续。
- SAM2 checkpoint、录制数据和现场日志保留在本机，不通过 Git 提交。
