# R1 A7 视觉伺服与低速控制工作接续记录

更新时间：2026-10-10

适用项目：R1-A7 双臂机器人镊子夹取细线与目标孔对准

对应分支：`codex/r1a7-visual-servo-control-2026-10-10`

## 1. 文档定位

本记录把 2026-09-29 之后的低速控制、右臂 W/U 笛卡尔运动和 Wrist 视觉跟踪工作接到既有双相机视觉伺服归档上。父分支是 `codex/r1a7-dual-camera-visual-servo-2026-09-29`，因此此前的标定、External/Wrist/Fusion 架构、原始记录和操作说明都保留在 Git 历史中。

本次归档遵循两条边界：

- 原始 Word 文档作为实验记录保存，但其中的命令和建议不自动视为当前操作指令。
- 只上传源码、说明和原始记录；现场日志、`/tmp` trace、模型权重、录制数据、缓存和备份文件继续留在本机。

## 2. 工作演进

| 阶段 | 主要工作 | 阶段结果 |
|---|---|---|
| 2026-09-29 | External、Wrist、Fusion 双相机视觉框架；Tip/Hole 状态和置信度；单源接管与 Blind Grace | 双相机链路形成，Wrist Hole 的丢失和错误重捕获仍需继续解决 |
| 2026-10-07 | 500 Hz LowCmd、腕/肘单关节微动、LowState 量化、Z 向 Jacobian 条件分析 | 将超低速问题收敛到 stick-slip/dead-zone 与不利构型两个层面 |
| 2026-10-08 | 右肩 pitch fully-buffered、gravity FF、Frozen-q torque ramp、方向不对称和 hysteresis | 排除上位机轨迹和发送抖动为主因，确认执行层存在方向和状态依赖 |
| 2026-10-09 | DQIK、IK 100 Hz/LowCmd 500 Hz、Kd20/30、POSVEL、gravity/load 分层、W/U 对照 | W 的典型 stick-slip 基本解决；U 的早期低速转为 IK/构型与任务定义问题 |
| 2026-10-10 | Wrist Hole/Tip 性能、身份门控、Template 重捕获、预测安全化 | 10 mm/s 视觉伺服由用户确认通过；15/25 mm/s 尚未在修复后重新资格验证 |

## 3. 当前已知良好基线

### 3.1 机器人控制

- 控制器：`transfer_control/experimental/r1a7_cartesian_stream_control_kd30_posvel_wu_dqik_u010_v1.py`
- IK：100 Hz。
- LowCmd：500 Hz。
- 右肩 pitch：`Kp=100`，`Kd=30`。
- W 的 DQIK 上限：`0.06 rad/s`。
- U 的 DQIK 上限：`0.10 rad/s`。
- 肘部动态偏置：末端每抬高约 10 mm，elbow reference 约减小 2 度。
- 重力前馈保持当前实现和 2.8 Nm 限幅；尚未形成适用于全工作空间的最终姿态相关补偿。
- 当前明确通过的视觉引导速度：10 mm/s。

### 3.2 视觉链路

- External：`vision_dual_camera/task6_threading/task6_external_ibvs_v4_aruco_edge.py`
- Wrist：`vision_dual_camera/task6_threading/task6_wrist_ibvs_controller_v2_tip.py`
- Fusion：`vision_dual_camera/task6_threading/task6_dual_ibvs_fusion.py`
- 启动与日志：`run_task6_visual_stack.sh`

Wrist 最终保留的关键逻辑：

- Tip 重检测每 2 帧执行一次，Hole 每帧执行，软件循环由约 14 Hz 提升到约 24 至 27 Hz。
- Hole 全局重捕获不再采用 Hough-only，而是 Template、Geometry 和多帧确认组合。
- Local Template Gate、Initial/Recent 双模板和短时运动连续性共同限制错误身份跳转。
- Hole 预测只用于下一帧 ROI，不作为真实观测进入 IBVS。
- Tip relock 增加 anchor gate；长时 prediction 会过期；prediction 不再写回 `tip_ema`。
- Motion-aware gate 只用短时预测作为 jump reference，避免放宽静态 gate 后引入错误特征。

## 4. 控制问题的分层结论

### 4.1 W 向前低速运动

原始 10 mm/s W 运动存在“停住、误差积累、突然释放、再次粘滞”。DQIK 把非零 `dq_ref` 写入 LowCmd 后有所改善；仅将主要负载关节的 Kd 从 2 提高到 30，是目前最有效的改动。POSVEL 可以修正少量 raw dq 虚高事件，但作用小于 Kd30。

固定 friction FF、长期保持 anti-stiction、45→10 mm/s 启动 boost 和单纯提高频率都没有成为最终方案。最新多段 W/U trace 中未再观察到典型 LOW 与 FAST 反复切换。

### 4.2 长程前伸 backlog

短程 stick-slip 改善后，长程前伸暴露出另一个慢时间尺度问题：姿态负载持续增加，gravity FF 触及限幅后，P 位置误差被迫承担额外负载。分层 trace 显示 IK raw 到 stream、stream 到 LowCmd 的积压很小，主要误差发生在 LowCmd 到 actual。

后续应记录 `gravity_raw_model`、`gravity_scaled_before_clip` 和 `gravity_sent_after_clip`，再决定放宽限幅、校准模型/负载，或采用姿态相关的平滑 scaling。固定全工作空间 scale=1.0 在 home 附近存在过补证据。

### 4.3 U 抬起

U 早期没有正确 `dq_ref`，后来已将 DQIK 扩展到 W/U。提高 U 上限到 0.10 rad/s 解除了速度参考饱和，但没有解决起始构型下的慢速。

离线分层表明：早期 command 为 10 mm/s 时，raw IK 只能生成约 1 mm/s；stream、LowCmd 和 actual 基本跟随该结果。肘部是主要正向抬升来源，wrist pitch 的大幅运动则是 IK 为保持世界姿态主动分配的结果。

若动作语义是“肘部主导抬臂、腕部相对前臂基本不转”，后续应以 U 起始 wrist joint posture 为参考，优先验证 `position_only=True`，并按需要放宽严格的世界姿态或 TCP XY 直线约束，而不是继续提高 Kd、gravity FF 或 wrist Kd。

## 5. 与此前工作的连接

此前入口：

- `docs/r1a7_dual_camera_visual_servo_work_summary_2026-09-29_zh.md`：9 月 29 日双相机视觉伺服归档。
- `docs/r1a7_dual_camera_visual_servo_upload_manifest_2026-09-29_zh.md`：上一阶段上传边界。
- `docs/r1a7_project_snapshot_2026-09-22_zh.md`：完整细线穿孔项目索引。

本阶段新增入口：

- 本文：10 月 7 日至 10 月 10 日的控制与视觉接续说明。
- `docs/r1a7_visual_servo_control_upload_manifest_2026-10-10_zh.md`：本次源码、文档和排除项清单。
- `docs/source_records/2026-10-07_to_2026-10-10/`：7 份原始 Word 工作记录。

## 6. 后续工作顺序

1. 使用当前 Known-Good 版本重复验证 10 mm/s，保持机械臂稳定控制器不变。
2. 逐级重新资格验证 15 mm/s 和 25 mm/s；早期失败发生在视觉修复前，不能沿用旧结论。
3. 对 U 进行约 3 s 的 `position_only` 和 wrist posture A/B，分别记录 command、raw IK、sent、actual、condition 和 lag。
4. 增加 gravity clip 前后的 trace，分开验收短程低速连续性与长程姿态负载。
5. 现场继续验证 DUAL、EXTERNAL_ONLY、WRIST_ONLY、BLIND_GRACE 和 STOP 的故障切换。

## 7. 安全边界

- 同一时间只允许一个 `rt/lowcmd` publisher。
- 实机测试前确认 MotionSwitcher 无活动服务、急停可用且工作区清空。
- 先静态视觉检查，再做低速、小行程闭环；出现身份跳变、lag 快速增长或 condition 恶化立即停止。
- 不把 HOLD/prediction 当成长期真实观测，不把一次异常好结果直接用于参数定型。
- 本归档没有在整理过程中启动机器人、相机或控制节点。
