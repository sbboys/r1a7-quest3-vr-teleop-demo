# 截至2026-11-09完整研发工作记录 V2.0 全文

[原件](R1-A7_细线抓取与穿孔_完整研发工作记录_截至2026-11-09_V2.0.docx)；SHA-256：`c7f10261c29c0b7f9a34b98f70d9c2c3d616a6bd97dc95385259c6a92a3df611`。

全文保留原历史状态、命令和附录；它们不是本次执行授权。当前文件核对与SOP单列，原件不改写。

Unitree R1-A7  双相机细线抓取与穿孔

完整研发工作记录与真机操作归档

从三相机 U-Net 到自动开合、Jaw Corridor 与插孔阶段

| 项目字段 | 归档信息 |
| --- | --- |
| 归档版本 | V2.0 · 截至 2026 年 11 月 9 日 |
| 平台 | Unitree R1-A7 + 右臂镊子夹线 + External/Right Wrist/Left Wrist 三相机 |
| 项目根目录 | /home/robot/unitree_sim_isaaclab_threading/r1a7_threading_project |
| 基础文档 | 2026-11-02 技术路线汇总；2026-11-03 右腕张开镊子与自动抓取工作记录 |
| 本次新增 | 孔位/插孔工作、Fusion/Gate 修复、External-near Z 优先、Jaw Corridor、闭合状态机 |
| 文档用途 | 研发交接、SOP 复现、实验数据追溯、下一阶段插孔调试 |

文档口径：保留原有历史归档；按对话中已展示的命令、源码片段、终端日志和用户测试反馈判定证据等级。不能以“语法通过”代替“物理抓取成功”，不能以“进入 INSERTION_TRACKING”代替“穿孔成功”。

阅读指南与当前状态

本版将截至 11 月 3 日的训练、分割、TIP 语义与 Auto-Grasp 工作保留为历史基线，同时按照 11 月 4—9 日的开发演进补充插孔、Fusion/Gate、Z 轴优先控制、夹持通道判定及真机测试。最后完整保留 11 月 2 日和 11 月 3 日原始工作记录的文字与表格作为追溯附录。

| 类别 | 截至 11 月 9 日结论 | 证据层级 |
| --- | --- | --- |
| 夹取：视觉模型 | External/Left=旧 V2；Right Wrist=V3 的双模型策略保留；历史独立 test 有数值 | 11/03 历史实测 |
| 夹取：Z 分工 | 近距离 External 主导 Z，Wrist 保留 Y；用户报告效果正常 | 11/09 用户真机反馈 |
| 夹取：Jaw Corridor | 张开镊臂通道判断和自动闭合版本已由用户反馈正常执行 | 11/09 用户真机反馈（未提供完整现场日志） |
| Gate | 350px 图像误差硬阻断已移除；py_compile 通过 | 11/07 终端回执 |
| 插孔：流程 | 完成/部署 GRASP→WAIT_HOLE_TARGET→INSERTION_TRACKING 的流程设计与代码检查 | 源码片段/历史对话；整链动作待逐项验证 |
| 孔位 U-Net | 用户确认涉及采集、标注、训练及孔位识别；现有材料无完整训练表与 checkpoint 清单 | 本轮陈述；量化数据待补录 |
| 成功插孔 | 暂无穿过实体孔并验证出线的可核实结果 | 未验证 |

目录

第一篇  综合技术路线与版本状态

第二篇  11 月 4—9 日新增工作详录

第三篇  当前真机 SOP 与检查命令

第四篇  风险、问题闭环与下一步研发计划

附录 A  2026-11-02 原始技术路线工作记录（逐项转录）

附录 B  2026-11-03 原始右腕张开镊子工作记录（逐项转录）

附录 C  路径、参数、证据与回滚速查

第一篇  综合技术路线与版本状态

1. 研发目标、时间轴与资料口径

任务从三相机识别细线与镊子出发，经由 TIP→GRASP 视觉伺服完成右臂镊子靠近、自动张开、闭合抓取；之后切换为检测指定目标孔、选择同一物理孔，将细线自由端 FREE 导向 HOLE，并最终尝试插入。

| 时段 | 主要工作与实际判断 |
| --- | --- |
| 10 月下旬—11/02 | 三相机 U-Net、骨架/Temporal、FREE 端点身份、GRASP 目标、Fusion 的 Z 冲突、Gate 对冲突状态的放行；形成 11/02 工作总结。 |
| 11/03 | 右腕张开镊子 64 张采集/标注，V3 mixed replay 训练和 9 张 test；V2/V3 双模型部署；中点 TIP 与 Close Gate 方案。 |
| 11/03—11/07 | 引入夹取后选孔、三相机孔目标确认、插孔跟踪；孔 U-Net 采集/标注/训练在用户描述范围内，训练日志细节待补录。 |
| 11/07 | 定位 Gate STOP：External error 394.2px 超过 350px；备份并移除硬阈值，编译通过。 |
| 11/09 | Fusion 增加 External-near Z 优先（45/65px 滞回），并添加 Jaw Corridor 与 Controller 闭合锁存；用户报告四终端完整运行且功能正常。 |

2. 系统总体架构与数据路径

External USB ZOOM ----> U-Net V2 --> wire/TIP/GRASP --> External guidance JSON --+Right Wrist Orbbec --> U-Net V3 --> live wire/open jaw TIP --> Wrist JSON ---+--> Fusion --> GateLeft Wrist Orbbec ---> U-Net V2 --> grasp monitor / hole selection -------+                 |                                                  /tmp/r1a7_dual_ibvs_velocity.json                                                            |                                          Cartesian streaming / IK / rt/lowcmd                                                            |                                               Unitree R1-A7 right arm + tweezerPhase: GRASP --> CLOSE_SETTLE --> WAIT_HOLE_TARGET --> INSERTION_TRACKING

External 主要用于 Base X/Z 与近距离 Z 最终控制；Right Wrist 主要负责 Base Y，并在远距离参与 Z 融合；Left Wrist 在抓取中多用于观察，在选孔阶段需要与 External、Right Wrist 确认同一实体孔。Fusion 生成建议速度，Gate 决定是否输出，Controller 是真机 LowCmd 唯一执行入口。

| JSON | 生产者 / 核心字段 |
| --- | --- |
| /tmp/r1a7_external_wire_grasp_v1.json | Visual；timestamp, guidance_valid, wire_locked, du_px, dv_px, proposed_vx/z |
| /tmp/r1a7_wrist_wire_grasp_guidance.json | Visual；du_px, dv_px, proposed_vy/z；新版本增加 jaw_* |
| /tmp/r1a7_wire_grasp_fusion_shadow_v1.json | Fusion；fusion_mode, external_valid, wrist_valid, proposed_vx/vy/vz, grasp_ext_near, z_control_source |
| /tmp/r1a7_dual_ibvs_velocity.json | Gate；gate_state, enabled, input_age_s, vx/vy/vz, external_error_px, wrist_error_px |
| /tmp/r1a7_auto_grasp_close_settle_v1.json | Controller；state, active, reason, timestamp；Gate CLOSE_SETTLE_HOLD 依据 |
| /tmp/r1a7_insertion_target_state_v1.json | Visual；phase, target_ready/confirmed；用于插孔和抓取策略的分期 |
| /tmp/r1a7_insertion_perception_v1.json | Visual；插孔目标感知记录；具体 schema 以现场 JSON 为准 |

3. 三相机感知与训练成果基线

11/02 基线：三相机 wire/tool U-Net V2（mean foreground Dice 0.7502），骨架长路径与 Temporal 稳定性；Right Wrist 细线识别相对薄弱。11/03 专项改进：采集 Right Wrist 张开镊子 64 帧（47 train / 8 val / 9 test），按时间块而非邻帧随机划分，并保留 000043、000046 两个遮挡 hard cases。

| 模型评估项 | 旧 V2 | 右腕新 V3 | 证据 |
| --- | --- | --- | --- |
| 独立测试 Mean FG Dice | 0.509407 | 0.728989 | 11/03 文档；9 张独立 test |
| 独立测试 Wire Dice | 0.449626 | 0.566445 | 11/03 文档 |
| 独立测试 Tool Dice | 0.569188 | 0.891532 | 11/03 文档 |
| 旧域 val Mean FG Dice | 0.750200 | 0.7524 | 11/03 文档，回放防遗忘 |
| 右腕新域 val Mean FG Dice | 0.477433 | 0.7728 | 11/03 文档，epoch 90 best |

V3 mixed replay 使用旧 35 个样本 × 3 相机 = 105 项、新 47 个训练样本按 new_repeat=3 扩展为 141 项，每 epoch 246 项；batch 2、crop 512、stride 384、AdamW、lr 2e-5、weight decay 1e-4、0.5 weighted CE + 0.5 foreground Dice loss。最佳权重路径：/home/robot/r1a7_datasets/wire_unet/right_wrist_open_001/runs/v3_mixed_replay/best.pt。

2026-11-03 的“张开双尖端中点 TIP”当时仍是方案；到了本轮用户上传的现场源码中，RightWristTweezerTipMidpointV1 已有两臂识别、tip_pair_yx、tip_pair_separation_px、OPEN 前端中点及 CLOSED 回退分支，说明这一阶段的代码部署状态后来发生了变化。

4. 版本地位与证据等级

| 等级 | 释义 | 记录要求 |
| --- | --- | --- |
| A / 真机回执 | 用户明确反馈实际运行的功能正常或给出控制器日志 | 可标注“用户反馈已验证”，但区分客观量化成功率 |
| B / 源码 + 静态检查 | 给出 grep、补丁标记、py_compile 通过 | 不能上升为实机效果 |
| C / 历史文档记录 | 11/02 和 11/03 文档含实测数据、命令与判断 | 保留当时上下文与时间 |
| D / 方案或待补录 | 只有设计方案或口头范围，没有运行结果 | 标注待执行或待补实测 |

第二篇  11 月 4—9 日新增工作详录

5. 插孔任务阶段化改造

夹取阶段成功执行 CLOSE 后，不再沿用 GRASP 目标。新状态链为 GRASP → WAIT_HOLE_TARGET → INSERTION_TRACKING。V3 视觉代码片段已经显示：当 close_completed_signal() 成立后，程序打印 GRASP → WAIT_HOLE_TARGET，释放 External/Left 原细线锁，停止旧抓取引导并加载孔位检测分支。

GRASP:  External + Right Wrist TIP -> GRASP  auto open -> align -> CLOSE_SETTLE -> CLOSE_ONCE -> WAIT_CONTACT  close_completed_signal() --> WAIT_HOLE_TARGETWAIT_HOLE_TARGET:  freeze arm guidance; keep tweezer closed  release External and Left wire locks  show candidate holes; select same physical hole in 3 camera views  confirm target --> INSERTION_TRACKINGINSERTION_TRACKING:  External: FREE -> HOLE produces X/Z  Right Wrist: FREE -> HOLE produces Y/Z  Left Wrist: additional same-hole confirmation/monitor  Fusion/Gate/Controller resume permitted insertion trend

选孔必须匹配三个视角中的同一实体孔，而不是选“各相机屏幕中距离最近的一个孔”。记录中的操作为 External、Right Wrist、Left Wrist 依次选择目标，并在确认动作后切到 INSERTION_TRACKING。具体鼠标/键盘触发按当时运行 V3 终端提示为准，避免键位版本不一致。

插孔阶段的 FREE 控制点需要与抓取阶段 GRASP 区分。V3 insertion_guidance() 依据 FREE 和所选 HOLE 的像素差计算误差；可用于运动的 FREE 来源限定为 MEASURED 和 TIP_RELATIVE_PRED，DISPLAY_ONLY 只用于可视化，不应成为控制有效来源。该差异在用户粘贴的 V3 源码段中可见。

6. 孔位图像采集、标注与 U-Net 训练

用户明确要求将 11 月 3—9 日的“识别孔位、采集、标注、U-Net 训练”纳入总档案。已知 V3 视觉包含孔位检测分支、目标孔候选显示和目标确认；对话可见资料尚未提供这轮孔位数据集的样本数、split、标注 class、训练脚本名称、checkpoint 路径、best epoch、Dice/IoU、独立 test 指标，因此不以右腕张开镊子数据集 64 张或 V3 的 0.8915 Tool Dice 冒充孔位训练数据。

| 孔位研发事项 | 目前可以确认的内容 | 需补录的可核实项目 |
| --- | --- | --- |
| 图像采集 | 开展了插孔/孔位相关工作；V3 使用三相机观测 | 采集脚本、日期、每相机张数、曝光、分辨率、原始目录 |
| 标注 | 用户明确提到孔位标注工作 | class 定义、掩膜格式、QA、train/val/test 清单、遮挡样本 |
| U-Net 训练 | 用户明确提到孔位 U-Net 训练；运行 V3 有孔位模型分支 | trainer 路径、checkpoint SHA256、epoch、超参、train/val 曲线、独立测试 |
| 目标选择 | 等待孔选择、同一物理孔三视角确认 | 点击与确认回执、目标跨视角身份校验结果 |
| 视觉伺服 | FREE→HOLE，External X/Z + Wrist Y/Z | 目标误差趋势、相机丢孔、完成/失败条件 |

后续获取孔位训练证据时，优先补充真实训练终端最后 100 行、checkpoint 文件列表与 SHA256、标注 QA 报告及孔位 U-Net 的独立测试汇总。只有这些证据到位，才能在下一版文档填入具体数字。

7. Fusion 与 Gate 停止问题：350 px 误差硬门控

测试时 Fusion 为 DUAL、双相机均有效、心跳新鲜，但 Gate=STOP、enabled=False。现场一帧 JSON 的关键值为 input_age_s=0.0394358635、external_valid=True、wrist_valid=True、external_error_px=394.2004566、wrist_error_px=279.6926885；门限常量 MAX_EXTERNAL_ERROR_PX=350.0、MAX_WRIST_ERROR_PX=350.0。故该帧在旧 DUAL 条件下由 External 超阈值阻断。

# 原抓取 DUAL 规则（局部）perception_ok = (    source_fresh and external_valid and wrist_valid    and ext_error <= MAX_EXTERNAL_ERROR_PX    and wrist_error <= MAX_WRIST_ERROR_PX)# 修改后 DUAL 规则（局部）perception_ok = bool(source_fresh and external_valid and wrist_valid)

用户 2026-11-07 17:29:54 执行 Gate 备份，标记为 bak_before_remove_image_error_gate_20261107_172954；补丁输出 external/wrist threshold checks found=3/3，并回执 IMAGE_ERROR_HARD_GATE_DISABLED_OK、GATE_IMAGE_ERROR_PATCH_SYNTAX_OK。随后 grep 确认 DUAL、DUAL_Z_CONFLICT、WRIST_ONLY、EXTERNAL_ONLY 已不再用 350 px 作为硬 STOP 条件。

该处理属于“为趋势测试放宽感知误差安全门槛”，不是证明任意高像素误差均正确。Gate 仍需确认输入新鲜、有效标志与速度有限；高误差误识别导致误动的风险转移至上游感知和其他安全约束，后续应补充接近区域、方向合理性及幅度保护。

8. BLIND_GRACE：短时视觉丢失行为

Fusion 设置 BLIND_GRACE_TIME_S=0.40、BLIND_GRACE_SCALE=0.40、BLIND_GRACE_MAX_SPEED_MM_S=2.0，使短时间丢帧时按最近有效趋势低速过渡。Gate 后续增加 BLIND_GRACE 模式允许条件：source_fresh 即可在该模式下 perception_ok=True。其目的是避免一两帧缺失直接 STOP；但 BLIND_GRACE 是无新视觉修正的预测性运动，不能等同于有目标追踪的安全运动。

BLIND_GRACE_TIME_S = 0.40BLIND_GRACE_SCALE = 0.40BLIND_GRACE_MAX_SPEED_MM_S = 2.0# Gate 片段elif fusion_mode == 'BLIND_GRACE':    perception_ok = bool(source_fresh)

预期状态转换 DUAL → BLIND_GRACE → DUAL；持续视觉缺失仍应 STOP。异常情况下必须优先验证真实运行 Gate JSON（mode / age / enabled / v）及急停，而不能只看 Fusion 的模式文本。

9. External-near：近距离 Z 轴主导权切换

现场发现两台相机的 GRASP 投影不一致：External 镊尖接近固定夹取点时，Right Wrist 的在线抓取点仍可能受遮挡和重新识别而变化。原正常 DUAL 的 vz=0.35*ExternalZ+0.65*WristZ，腕部仍有 65% 上下作用，致使靠近阶段在 Z 方向反复修正，难以稳定闭合。

改动只发生在抓取阶段 Fusion：进入 External-near 条件后，X=External、Y=Right Wrist、Z=External；腕部 Z 权重变 0。新状态使用 45 px 进入、65 px 退出的滞回门限，避免阈值附近的反复切换。进入 INSERTION_TRACKING 后计划恢复原先插孔 Z 融合。

# 抓取阶段远处vx = ext_x; vy = wrist_yvz = 0.35 * ext_z + 0.65 * wrist_z  # 同向分支# External near 进入 <= 45 px; 退出 >= 65 pxif grasp_ext_near and not insertion_active:    vz = ext_z                          # wrist Z weight = 0# 插孔恢复非 near 的原始 Z 逻辑

用户 2026-11-09 16:40:56 备份 Fusion 至 bak_before_grasp_ext_z_priority_20261109_164056，执行补丁输出 GRASP_EXTERNAL_NEAR_Z_PRIORITY_PATCH_OK，随后回执 FUSION_EXT_Z_PRIORITY_SYNTAX_OK。grep 显示参数 45.0/65.0、grasp_ext_near 锁存、z_control_source 和权重诊断字段均已存在。后续用户确认真机执行功能正常。

需要区分：该策略近距离减少的是 Wrist 对 Base Z 的参与；它没有证明 Wrist 的 GRASP 检测不会晃动，也没有保证 External 自身抓取点一定准确。新版本把后一个问题继续下沉到 Jaw Corridor 解决。

10. 张开镊子的 TIP 与 Jaw Corridor

Right Wrist TweezerTipMidpointV1 中已有两个 connected component 的大小筛选、每片镊臂 PCA/contour 前沿尖端估计、双尖端间距和中点 TIP。源码常量包括 OPEN_MIN_COMPONENT_PIXELS=80、OPEN_SECOND_COMPONENT_RATIO=0.20、OPEN_TIP_SEPARATION_MIN_PX=8.0。可用字段 tip_pair_yx、tip_pair_separation_px；单臂或间距小的模式回退原 Tracker V2。

近距离遮挡导致单一 GRASP 点来回跳动，即使腕部只控制 Y，原 Controller 的 CLOSE 条件仍要求 abs(wm[eu])<=0.10 且 wm[jitter]<=0.035，容易反复退出。新的 Jaw Corridor 根据已识别的双尖端间距先在张开状态收集 12 个有效样本并取中位数作为固定 gap，启用 80% 通道（两侧各留 10% 余量）。此后将目标点投影到两镊尖连线方向，只需 lateral_error 落在通道半宽内。

# 背景几何（示意；实际执行以现场代码为准）fixed_gap_px = median(valid_open_gap_samples[:12])usable_half_width_px = 0.5 * fixed_gap_px * 0.80jaw_lateral_error_px = dot(grasp_yx - pair_mid_yx, pair_axis_yx)jaw_corridor_ok = abs(jaw_lateral_error_px) <= usable_half_width_pxif jaw_corridor_ok:    jaw_corridor_latched = True

V3 补丁新增的 wrist JSON 建议字段：jaw_tip_mode、jaw_pair_yx、jaw_pair_separation_px、jaw_calibration_samples、jaw_calibration_ready、jaw_fixed_gap_px、jaw_corridor_half_px、jaw_lateral_error_px、jaw_corridor_ok、jaw_corridor_latched。用户表示此版本能够正常执行。

关键边界：两镊尖连线上的投影进入通道并不单独证明细线在镊子纵深方向、夹持高度或物理接触处于合适位置。仍须依赖 External X/Z close、目标有效性、工具真实张开位置及现场操作员判读。对多次抓取，进程级 one-way latch 应有明确 reset/episode 边界，避免上一轮 True 影响下一次。

11. Controller 的 CLOSE_SETTLE 锁存化

旧 Controller OPEN_TRACKING 时计算 ext_close（External eu/ev/jitter）和 wrist_close（Right Wrist eu/jitter），进入 CLOSE_SETTLE 后又连续重检 close_ready。视觉遮挡产生瞬时 False 会触发 CLOSE_SETTLE EXIT reason=NOT_READY，因此 0.40 s 持续窗口很难完成。

新方案把 Wrist CLOSE 改为使用 jaw_corridor_latched；当 External close 与 Wrist corridor 同时满足，进入 CLOSE_SETTLE，Gate 输出停臂保持，Controller 不再因腕部后续目标点抖动退出；等待 close_hold_s=0.40 s 后发一次 CLOSE_ONCE 并进入 WAIT_CONTACT，再依据 contact_hold 或夹爪位置到位进入 DONE。

# 现行意图（状态机示意）close_ready = ext_close and wrist.get('jaw_corridor_latched', False)if state == 'OPEN_TRACKING' and close_ready:    state = 'CLOSE_SETTLE'    publish_close_settle(active=True, reason='ENTER')# CLOSE_SETTLE 期间不再反复审查 wrist Eu/Jitter# 仅在启用、已 OPEN、未 CLOSE 及完成 hold 等条件下发出一次 CLOSE# WAIT_CONTACT 以后仍需保持夹爪，等待 DONE

11/03 原版仍把 Right Wrist eu/jitter 当 CLOSE 必要条件、ev 已拟取消；11/09 本轮改为 Corridor latch + settle 单向锁存。用户后续明确回复“现在正常执行了并且功能正常”，可以登记为真机功能确认，但对最终夹住细线的重复率、夹持力及实际插孔成功率仍缺独立统计。

第三篇  当前真机 SOP 与检查命令

12. 运行前安全检查

真机由 rt/lowcmd 驱动。先清空右臂运动区域、确认急停和人工监护，停止旧 Controller；不要根据 pgrep 命中两个 PID 就推断必然有两个发布者：multiprocessing 父子进程可能继承相同命令行，应以 PPID/进程树/LowCmd guard 确认。不要手工删除 /tmp/r1a7_vr_g1ik_real.lock。

cd /home/robot/unitree_sim_isaaclab_threading/r1a7_threading_projectpgrep -af 'live_grasp_to_insertion_visual_v3.py|wire_grasp_fusion_shadow_v1.py|wire_grasp_control_gate_v1.py|r1a7_cartesian_stream_control_kd30_posvel_wu_dqik_u010_auto_grasp_direct_v1.py'# 如看到疑似重复 Controller，先核实父子关系（按真实 PID 替换）：ps -o pid,ppid,pgid,stat,lstart,cmd -p PID1,PID2# 首选在原终端 Ctrl+C 正常退出。

启动前依次查看修改标记及静态语法：以下检查不启动电机。

cd /home/robot/unitree_sim_isaaclab_threading/r1a7_threading_projectV3=vision_dual_camera/task6_threading/dataset_tools/wire_unet/live_grasp_to_insertion_visual_v3.pyFUSION=vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/wire_grasp_fusion_shadow_v1.pyGATE=vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/wire_grasp_control_gate_v1.pyCTRL=transfer_control/experimental/r1a7_cartesian_stream_control_kd30_posvel_wu_dqik_u010_auto_grasp_direct_v1.pygrep -n 'WRIST_JAW_CORRIDOR_V1' "$V3"grep -n 'GRASP_EXTERNAL_NEAR_Z_PRIORITY_V1' "$FUSION"grep -n 'DISABLE_IMAGE_ERROR_HARD_GATE_TREND_V1' "$GATE"grep -n 'JAW_CORRIDOR_CLOSE_LATCH_V1' "$CTRL"PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv_sam2/bin/python -m py_compile "$V3" "$FUSION" "$GATE"PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv/bin/python -m py_compile "$CTRL"

13. 终端 1：Visual V3

cd /home/robot/unitree_sim_isaaclab_threading/r1a7_threading_projectPYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv_sam2/bin/python -u vision_dual_camera/task6_threading/dataset_tools/wire_unet/live_grasp_to_insertion_visual_v3.py

启动后完成 External 与 Right Wrist 的 FREE/GRASP 初始化和校准。External 固定锁定选线，Right Wrist 实时识别。确认 [GRASP] EXT valid=True WRIST valid=True，再启动 Fusion。三相机选孔在自动 CLOSE 完成之后进行。

14. 终端 2：Fusion

cd /home/robot/unitree_sim_isaaclab_threading/r1a7_threading_projectPYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv_sam2/bin/python -u vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/wire_grasp_fusion_shadow_v1.py

抓取远距离预期日志 Z=DUAL；External 误差进入 45px 后预期 Z=EXT_ONLY；升到 65px 退出。插孔确认后应核实 insertion_tracking_active=True 并恢复插孔控制逻辑，不能依赖终端打印而忽略 JSON。

15. 终端 3：Gate

cd /home/robot/unitree_sim_isaaclab_threading/r1a7_threading_projectPYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv_sam2/bin/python -u vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/wire_grasp_control_gate_v1.py --enable-output

在无 Controller 的 Shadow/只读阶段，确认 Fusion DUAL 且 Gate 输出 enabled=True 和非零建议；视觉失效必须检查 Gate 是否按状态停机。在 CLOSE_SETTLE_HOLD 阶段 v=(0,0,0)、enabled=True 可能是预期保持；Gate STOP 且 enabled=False 为另一种状态，不可混淆。

16. 终端 4：Controller（最后启动，唯一 LowCmd 发布者）

下面保存的是本轮已给出的低速趋势测试启动模板：cartesian-jog-speed-mm-s=3。用户随后反馈功能正常，但未提供本次逐条的最终命令行校验日志；换机器或复位后先确认 --help 接受所有参数，域、IP、网卡与现机一致。

cd /home/robot/unitree_sim_isaaclab_threading/r1a7_threading_projectmkdir -p r1a7_logs/auto_graspSTAMP=$(date +%Y%m%d_%H%M%S)R1A7_AUTO_GRASP_ENABLE=1 R1A7_AUTO_GRASP_PREOPEN_NORM_MAX=0.70 R1A7_AUTO_GRASP_CLOSE_AXIS_NORM_MAX=0.10 PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv/bin/python -u transfer_control/experimental/r1a7_cartesian_stream_control_kd30_posvel_wu_dqik_u010_auto_grasp_direct_v1.py --interface enp6s0 --domain-id 0 --host-ip 192.168.123.223 --ik-frequency 30 --publish-frequency 250 --max-joint-speed 2.0 --cartesian-jog-speed-mm-s 3 --cartesian-jog-accel-mm-s2 20 --enable-gripper --gripper-initial-mode current --no-right-gripper-open-cap-current --no-record-d435i --log-file "r1a7_logs/auto_grasp/controller_${STAMP}.log"

程序提示 Type ENABLE 后，只在机器人工作空间安全、急停准备就绪、唯一发布者确认成功时输入 ENABLE。系统上电、镜头位置变化、相机标定变化或新版本补丁时，应先不闭合/不接触进行空载低速方向验证。

17. 终端 5：状态监控与验收

python3 - <<'PY'import json,timefrom pathlib import Pathitems=[ ('EXT','/tmp/r1a7_external_wire_grasp_v1.json'), ('WRIST','/tmp/r1a7_wrist_wire_grasp_guidance.json'), ('FUSION','/tmp/r1a7_wire_grasp_fusion_shadow_v1.json'), ('GATE','/tmp/r1a7_dual_ibvs_velocity.json'), ('PHASE','/tmp/r1a7_insertion_target_state_v1.json'), ('CLOSE','/tmp/r1a7_auto_grasp_close_settle_v1.json')]for name,path in items: print('\n====',name,'====') try:  d=json.loads(Path(path).read_text())  ts=float(d.get('timestamp',0))  print('age_s=',round(time.time()-ts,3))  for k in ('phase','guidance_valid','fusion_mode','gate_state',    'external_valid','wrist_valid','external_error_px','grasp_ext_near',    'z_control_source','jaw_fixed_gap_px','jaw_corridor_ok',    'jaw_corridor_latched','insertion_tracking_active','enabled',    'proposed_vx_mm_s','proposed_vy_mm_s','proposed_vz_mm_s',    'vx_mm_s','vy_mm_s','vz_mm_s','state','reason'):   if k in d: print(k,'=',d[k]) except Exception as e: print('ERROR',repr(e))PY

正常抓取日志关注 [WRIST JAW] OPEN GAP CALIBRATED、CORRIDOR LATCHED、CLOSE_SETTLE ENTER、CLOSE_ONCE、GRASP COMPLETE；不能把这些字符串单独当成“线确实被夹紧”的物理证据。插孔阶段需补充孔目标误差、连续位移、退出 STOP、细线是否进入并穿出孔的直接观测。

18. 正常停止、异常和回滚

正常退出：优先在 Controller 原终端 Q/程序支持的安全退出或 Ctrl+C；确认 R1-A7 lowcmd publisher stopped 后，再停止 Gate、Fusion、Visual。

异常运动：优先物理急停；不要依赖在其他终端先停止 Fusion 或 Gate 来确保所有控制命令立刻消失。

源码回滚：先停止所有相关进程；从 .bak_before_* 或 SUCCESS_SNAPSHOT 对应时间戳恢复；语法检查后按 Visual→Fusion→Gate→Controller 重新起。

不可手动删除 LowCmd guard 锁文件。相同 CLI 的两个 PID 必须先检查父子关系，不要盲目同时 kill。

第四篇  风险、问题闭环与后续研发

19. 问题—修复—证据矩阵

| 问题 | 原因 / 已采取措施 | 截至当前状态 |
| --- | --- | --- |
| OPEN 后迟迟不 CLOSE | V2 对张开镊子 mask/TIP 不稳；新增右腕专项数据训练 V3、Midpoint TIP | 模型量化改善；最终功能用户反馈正常 |
| 近距离上下反复微调 | External 与 Wrist 抓取点不一致；近时 Wrist Z 权重归零 | Fusion 源码+语法+用户反馈 |
| Wrist GRASP 点遮挡抖动 | 按两镊臂间距构成 Jaw Corridor；首次进入后 latch | Controller 逻辑与功能用户反馈 |
| CLOSE_SETTLE 反复 EXIT | 原状态每帧重新检查 noisy wrist；改为阶段锁存 | 已提供补丁，用户反馈功能正常 |
| Gate 在 DUAL 仍 STOP | External error=394.2px，超过350；移除硬阈值 | 备份+执行+py_compile 通过 |
| 短暂掉帧即停 | Fusion BLIND_GRACE 0.4s/0.4x/2mm/s | 代码逻辑记录；需进一步长期安全验证 |
| 目标孔漂移或错选 | 多相机同一孔人工确认、孔位 U-Net 和阶段选择 | 完整成功率尚缺 |
| 插孔接触与穿出 | 当前主要是视觉趋势与阶段联动 | 未证实真实穿孔 |
| HOME 非预期触发 | 11/03 历史记录 OPEN_TRACKING 中发生 HOME/jitter 突增 | 需复测，不能宣称已彻底修复 |

20. 现版本必须保留的安全边界

只允许一个独立 LowCmd publisher；Controller 父子进程不等于两个发布者，需核查。

视觉 valid、时间戳、孔目标有效和阶段状态是不同条件，不可只看一种标志。

去掉 350px 硬门槛后，必须监视误识别和错误方向；BLIND_GRACE 不能无限延长。

Jaw Corridor 的锁存需要单次任务生命周期管理；重启视觉进程是目前历史记录中最明确的复位边界。

CLOSE_SETTLE 中不再重审 Wrist 目标点，但碰撞、停止、感知总体失效等保护不能因此被无意绕过。

抓取成功 ≠ 夹持力可靠；GRASP COMPLETE ≠ 一定夹住细线；INSERTION_TRACKING ≠ 穿孔完成。

旧的 11/02、11/03“当前状态”只代表当日；新方案不能直接覆盖历史结论。

21. 建议下一阶段里程碑

| 优先级 | 任务 | 可量化验收要求 |
| --- | --- | --- |
| P0 | 冻结已工作的抓取四脚本并做 SHA256+配置快照 | 保存 git diff、成功日志、命令行和备份时间戳 |
| P0 | 检查 jaw latch 多轮复位和异常取消 | 多轮重复抓取不继承上一轮 True；异常后不会误闭合 |
| P1 | 补齐 11/03—09 孔位 U-Net 的采集/标注/训练实测台账 | 样本数、split、类别、best.pt、hash、Dice/IoU、独立测试 |
| P1 | 三相机同孔目标确认与误差追踪 | 记录同孔选择、刷新、遮挡、相机掉线与超时 |
| P1 | 插孔近距离控制和停臂恢复 | 确认不会永久卡 CLOSE_SETTLE_HOLD；无过期速度执行 |
| P2 | 实际入孔、出线确认和接触保护 | 记录独立任务样本 N、成功/失败、最大误差、接触事件 |
| P2 | 发布可复现实验包 | README+SOP+参数表+checkpoint manifest+日志 |

附录 A  2026-11-02 原始技术路线工作记录

以下为 11 月 2 日原始文档的逐项内容转录。文中的“当前”“待验证”均指当日状态，不能覆盖本版 11 月 9 日的综合结论。为排版统一，表格与文字已重新设置样式，不改动事实口径。

R1-A7 细线夹取视觉引导

当前聊天对话操作、技术路线、实验结果与待办事项汇总

工作区：/home/robot/unitree_sim_isaaclab_threading/r1a7_threading_project

整理日期：2026 年 11 月 2 日

依据：当前聊天记录及上传的 2026-10-21、2026-10-30 阶段文档

记录口径：本文区分终端已执行、代码检查已确认、根据日志推断和待验证事项。未收到用户执行回执的补丁，不记为已完成。

一、项目目标与当前结论

本阶段目标：保留 2026-10-30 后形成的三相机 U-Net 细线/镊子实时识别，不回退旧 V69 或 External candidate 感知链；把新视觉生成的 External 与 Right Wrist 引导误差送入 Fusion、Control Gate，再由 threading 工作区的 Cartesian/IK/LowCmd 控制器驱动 R1-A7 右臂镊子靠近 GRASP。当前只做靠近，不自动闭合夹爪。

| 模块 | 当前状态 | 证据与边界 |
| --- | --- | --- |
| U-Net 训练 | 阶段性完成 | V2 resume best epoch 72；验证集 mean foreground Dice 0.7502。 |
| 三相机实时识别 | 已运行 | External、Right Wrist、Left Wrist 实时取流与推理；单路统计约 13–15 FPS。 |
| 骨架/连通路径 | 已实现并评估 | threshold 0.86、主簇 bridge 24px、component max-gap 50px、longest geodesic path。 |
| Temporal V1 | 已完成一轮验证 | 左腕 RAW 跳变 28.80% 降至 TRACKED 0.27%；右腕 11.14% 降至 2.17%。 |
| FREE/GRASP 几何 | 已接入 | ManualFreeTracker 输出 FREE/FIXED 端点、GRASP 区域和 grasp_point_yx。 |
| 当前控制目标 | 保留 GRASP | 用户最终决定不改成 FREE；两相机仍按 TIP→GRASP 计算误差。 |
| Fusion/Gate Z 冲突 | 源码补丁已执行并 grep 验证 | 冲突时 vz=External vz；Gate 增加 DUAL_Z_CONFLICT 有效性分支；尚缺运行时验收。 |
| 真机闭环 | 已发生真实运动，未通过验收 | 出现过 +Z 抬臂，也出现过 DUAL_Z_CONFLICT→STOP；新策略尚未重新真机验收。 |
| 右夹爪 O/V/C | 补丁待执行/待验证 | 已提供按键补丁方案，未收到实际执行回执。 |
| 黑屏问题 | 未找到 OOM/Xid 证据 | GNOME Shell 日志异常更值得排查；资源快照显示 RAM/VRAM 余量充足。 |

当前最终数据链

三相机 U-Net wire/tool 分割 + skeleton/path + Temporal tracking   ├─ External TIP→GRASP → /tmp/r1a7_external_wire_grasp_v1.json   ├─ Right Wrist TIP→GRASP → /tmp/r1a7_wrist_wire_grasp_guidance.json   └─ Left Wrist：观察/锁线视角                  ↓wire_grasp_fusion_shadow_v1.py→ /tmp/r1a7_wire_grasp_fusion_shadow_v1.json                  ↓wire_grasp_control_gate_v1.py --enable-output→ /tmp/r1a7_dual_ibvs_velocity.json                  ↓threading Cartesian controller → IK → rt/lowcmd → R1-A7

关键边界：live_visual_guidance_shadow_v1.py 名称中含 shadow，但它是当前三相机视觉前端；是否真实驱动机器人由后端 Gate 的 --enable-output 和 LowCmd 控制器共同决定。

二、工作区与控制器版本差异

用户明确指定只使用 `/home/robot/unitree_sim_isaaclab_threading/r1a7_threading_project`，不使用 `/home/robot/unitree_sim_isaaclab_console`。

| 项目 | threading（当前指定） | console（不使用） |
| --- | --- | --- |
| 文件 SHA256（当时检查） | a5deaa317d4abdc4be0b81e60c7f3d88b6bf30d45036960d58aa043cc24a84bc | 58746681e6d373a8bc322bc67369b82d617435d1f856c1b3a10cff1e2c99b559 |
| IBVS CLI | 不支持 --ibvs-command-path 等参数 | 支持 --ibvs-command-path、--ibvs-max-age-s、--ibvs-max-speed-mm-s 等 |
| JSON 路径 | 本地代码内部固定读取 /tmp/r1a7_dual_ibvs_velocity.json（按本地 diff/检查） | 曾配置 /tmp/r1a7_console_ibvs_velocity.json |
| 使用决定 | 唯一指定版本 | 不使用 |

曾发生的错误：将 console 版本的 --ibvs-* 参数加到 threading 控制器，argparse 报 unrecognized arguments。后续 threading 控制器命令不再添加这些参数。

LowCmd 互斥与冲突

threading 控制器通过 `transfer_control/tools/r1a7_lowcmd_guard.py` 获取 `/tmp/r1a7_vr_g1ik_real.lock`。曾检测到 PID 11052、11081 为 console 工作区控制器，其中 PID 11052 持有 rt/lowcmd 锁，导致 threading 控制器启动失败。

pgrep -af 'r1a7_cartesian_stream_control_kd30_posvel_wu_dqik_u010_v1.py'flock -n /tmp/r1a7_vr_g1ik_real.lock -c 'echo LOWCMD_LOCK_FREE'

锁文件原则：不要 rm 锁文件。应从原终端/服务正常停止旧控制器，确认进程退出后再启动 threading 版本。

三、U-Net 数据集、训练与离线评估

数据集与训练

| 项目 | 记录值 |
| --- | --- |
| 数据集 | /home/robot/r1a7_datasets/wire_unet/pilot_001 |
| 规模/划分 | 50组×3相机=150张；train 105 masks，val 21 masks，test 24 masks。 |
| 类别 | 0 background；1 wire；2 tool；255 ignore（如适用）。 |
| 训练裁剪 | 512×512；batch size 2；U-Net 约 7,763,107 参数。 |
| Overfit | 4样本、150 epochs；best mean foreground Dice=0.943611；仅证明小样本可拟合。 |
| V1 train/val | 30 epochs；best epoch 27；mean FG Dice=0.656527。 |
| V2 resume | 从 V1 best epoch 27 继续 50 epochs；best epoch 72；mean FG Dice=0.750200。 |

V1 与 V2 指标

| 指标 | V1 | V2 resume best |
| --- | --- | --- |
| 最佳 epoch | 27 | 72 |
| mean foreground Dice | 0.656527 | 0.750200 |
| overall wire Dice | — | 0.5658 |
| overall tool Dice | — | 0.9346 |
| wire IoU | — | 0.3945 |
| wire Precision / Recall | — | 0.4336 / 0.8138 |
| tool Precision / Recall | — | 0.9373 / 0.9319 |

V2 各相机验证集 wire Dice：External 0.6315、Right Wrist 0.4849、Left Wrist 0.6015；tool Dice 分别为 0.9458、0.9343、0.9218。Right Wrist wire 是主要薄弱视角。

阈值、容差、路径和端点评估

| 测试 | 结果 | 说明 |
| --- | --- | --- |
| Threshold sweep | 阈值0.86；wire Dice 0.5744；P 0.4735；R 0.7300；mean FG 0.7545 | 当时实时阈值，不保证所有环境最优。 |
| Tolerance | ±1px F1 0.9384；±2px 0.9758；±3px 0.9788；±4px 0.9805 | 容差指标更符合细线标注宽度差，但不是控制精度。 |
| Main cluster | bridge radius 24px；exact F1 0.5799；±2px F1 0.9835；retained 0.9843 | 改善局部连通/容差结果。 |
| Connected Path V3 | 验证集21/21 connected；path-to-GT P95 median 1px；p95 2.177px；max 2.96px | 路径贴近 GT 的统计较好，但端点 proxy 有离群。 |
| Endpoint/Tangent V4 val | 端点 median 1.825px、p95 18.169px、max 42.72px；切线 median 2.50°、p95 9.92°、max25.20° | 验证集仍有困难样本。 |
| Endpoint/Tangent V4 test | 24 masks；端点 median 2.532px、p95 15.655px、max350.077px；切线 median0°、p95 6.79°、max31.16° | right_wrist 000041 出现350px端点异常；test已用于诊断。 |

right_wrist 000041 FREE span 概率诊断：79px 中 77px 被判 background、2px 判 wire；mean P(background)=0.9639、P(wire)=0.0348。说明该段在模型输出中几乎被判背景。用户从图像指出多张图存在绿色自由端可见区域，提示模型对部分可见自由端仍有漏检。

阶段结论：U-Net 路线可行；tool 分割明显好于 wire。细线容差指标较好，但端点、右腕视角、反光/遮挡和运动帧完整性仍需加强。

四、三相机实时检测、骨架与 Temporal 稳定性

实时路径技术流程

三相机 RGB→ U-Net logits（background / wire / tool）→ wire threshold 0.86→ keep_main_cluster（bridge=24px）→ skeletonize→ connect_skeleton_components（max_gap=50px）→ longest_geodesic_path→ RAW path→ Temporal Path Tracker V1→ TRACKED path

设备：External USB ZOOM 1280×720；Right Wrist Orbbec CPCBC53000C5 1280×800；Left Wrist Orbbec CPCBC530007Z 1280×800。相机流配置30 FPS，日志单路推理中位数约13–15 FPS，不代表三路硬件同步或端到端30 FPS。

三轮运动测试结果

| 轮次/会话 | 相机 | 帧数 | 跳变>35% | 其他指标 |
| --- | --- | --- | --- | --- |
| 14:02 motion_stability_20261030_140255 | External | 556 | 0% | wire/tool/path/connect 100%；FPS 15.36 |
| 同上 | Right Wrist | 556 | 7.73% | connect 99.46%；P95 0.4845；FPS14.41 |
| 同上 | Left Wrist | 556 | 3.06% | tool 99.28%；P95 0.0452；FPS14.09 |
| 14:20 motion_stability_20261030_142023 | External | 313 | 0.64% | path/connect 100%；FPS14.23 |
| 同上 | Right Wrist | 313 | 17.57% | P95 0.7899；anomaly17.57%；FPS13.74 |
| 同上 | Left Wrist | 313 | 15.65% | P95 1.0527；anomaly15.97%；FPS13.93 |
| 14:41 temporal_v1_20261030_144103 | External | 368 | RAW0%；TRACKED0% | support median0.9787；FPS14.81 |
| 同上 | Right Wrist | 368 | RAW11.14%；TRACKED2.17% | TRACK88.86%；RECOVER4.08%；HOLD4.89%；RESET2.17%；FPS14.44 |
| 同上 | Left Wrist | 368 | RAW28.80%；TRACKED0.27% | TRACK71.20%；RECOVER25.54%；HOLD2.99%；RESET0.27%；FPS13.15 |

颜色含义与根因

| 颜色/状态 | 含义 | 边界 |
| --- | --- | --- |
| 粉红 | 当前帧 U-Net wire 像素 | 当前模型输出。 |
| 黄色 | 当前帧 RAW skeleton/path | 对当前帧漏检敏感，可能闪烁/只剩一段。 |
| 红色 | Temporal tracked path | 历史关联/RECOVER/短时HOLD后输出；连续不代表每帧都有新观测。 |
| 蓝色 | tool/镊子分割 | 工具区域和尖端精定位是两项不同任务。 |

Left Wrist 曾出现完整线/半线切换。431→432 路径长度约309.2→148.99px；483→484约280.46→148.99px。无损 Pwire 诊断发现缺失走廊概率很低，说明部分异常是 U-Net 当前帧将线段判为背景，不只是 skeleton 连接问题。Temporal V1 能减少跳变，但不能凭空恢复模型长期漏检的线段。

# Temporal V1 启动cd /home/robot/unitree_sim_isaaclab_threading/r1a7_threading_projectPYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv_sam2/bin/python -u \vision_dual_camera/task6_threading/dataset_tools/wire_unet/live_wire_temporal_validation_v1.py # R 开始/停止记录；Q/ESC 退出# radius=7px; jump trigger=0.35; support min=0.65; hold max=2 frames

五、FREE 身份、GRASP 几何与当前控制目标

ManualFreeTracker 位于 `live_manual_free_grasp_validation_v1.py`。人工点击确定哪端是 FREE，路径据此定向，另一端记为 FIXED。geometry 包含 `free_endpoint_yx`、`fixed_endpoint_yx`、`free_region`、`grasp_region`、`grasp_point_yx`、`grasp_tangent_xy`。现有注释/显示约定：GRASP region 从 FREE 向内约20–60px，代表点约40px。

最终用户决定：不将 FREE 直接作为机器人控制目标，继续使用原方案的 GRASP 夹取点。FREE 仍用于身份确认、路径定向和 GRASP 几何生成。

# 当前 Guidance 仍是 TIP -> GRASPgrasp_yx = geometry.get("grasp_point_yx")du = grasp_u - tip_udv = grasp_v - tip_v # TIP tracker 参考点也仍是 grasp_point_yx

当前视觉脚本 `live_visual_guidance_shadow_v1.py` 通过 `visual.v2.v1.manual` 使用 ManualFreeTracker。External 与 Left Wrist 在人工 FREE 初始化后锁定线参考；Right Wrist 线参考持续实时更新；三个视角镊尖检测保持实时。External/Right Wrist 输出各自 TIP 到 GRASP 的像素误差。

六、Fusion、Gate 与真机运动问题

JSON 接口

| JSON 文件 | 生产者 | 用途 |
| --- | --- | --- |
| /tmp/r1a7_external_wire_grasp_v1.json | 新视觉前端 | External guidance_valid、du/dv、proposed_vx/vz、grasp_target_uv。 |
| /tmp/r1a7_wrist_wire_grasp_guidance.json | 新视觉前端 | Wrist guidance_valid、du/dv、proposed_vy/vz、grasp_target_uv。 |
| /tmp/r1a7_wire_grasp_fusion_shadow_v1.json | Fusion | external_valid、wrist_valid、fusion_mode、proposed_vx/vy/vz。 |
| /tmp/r1a7_dual_ibvs_velocity.json | Control Gate | gate_state、enabled、vx/vy/vz、validity/error 字段；threading controller读取。 |

原始规则与暂停原因

正常 DUAL：vx = External vxvy = Right Wrist vyvz = 0.35 * External vz + 0.65 * Wrist vz 原始 Z 冲突：两路 proposed_vz 方向相反→ vz = 0→ fusion_mode = DUAL_Z_CONFLICT→ Gate 不接受该模式→ gate_state=STOP, enabled=False, v=(0,0,0)

用户指出两台相机中的 GRASP 投影位置不一致，导致两路 Z 方向冲突。诊断时也曾有一组 JSON：EXT du=87,dv=-166，Wrist du=-3,dv=-267，Fusion DUAL，Gate DUAL_READY、enabled=True，v≈(5.71,0,8.21)。该非零快照与另一个暂停时刻不是同一时刻，不能混为一谈。

已执行并检查的两个补丁

| 文件 | 修改 | 当前证据 |
| --- | --- | --- |
| wire_grasp_fusion_shadow_v1.py | 备份 `.bak_20261102`；Z 冲突分支改为 `vz = ez`，mode 仍为 `DUAL_Z_CONFLICT`。 | 用户 grep 已确认 `vz = ez`。 |
| wire_grasp_control_gate_v1.py | 备份 `.bak_20261102`；新增 DUAL_Z_CONFLICT 分支，要求 source_fresh、两路 valid、误差均未超阈值时 perception_ok=True。 | 用户 grep 已确认分支存在。 |

尚未完成：两处补丁已做源码检查，但没有修改后实时 JSON 和真机日志证明新策略已经生效。不要把“代码已改”写成“真机已验收”。

threading 控制器与 Gate

# Gate 直接输出（用户要求不先跑 Shadow Gate）cd /home/robot/unitree_sim_isaaclab_threading/r1a7_threading_projectPYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv_sam2/bin/python -u \vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/wire_grasp_control_gate_v1.py \--enable-output # threading 控制器（不加 --ibvs-* 参数）PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv/bin/python -u \transfer_control/experimental/r1a7_cartesian_stream_control_kd30_posvel_wu_dqik_u010_v1.py \--interface enp6s0 --domain-id 0 --host-ip 192.168.123.223 \--ik-frequency 30 --publish-frequency 250 --max-joint-speed 2.0 \--cartesian-jog-speed-mm-s 10 --cartesian-jog-accel-mm-s2 20 \--no-record-d435i

历史 `--host-ip 192.168.123.223` 曾与日志中的 `enp6s0 (192.168.123.164)` 同时出现。重新启动前需按本机控制器参数语义和机器人 DDS 网络设置核对，不应仅凭网卡本机 IP 自动替换。

控制器提示 `Type ENABLE ...` 后输入 ENABLE。threading 版本内部读取 `/tmp/r1a7_dual_ibvs_velocity.json`；不要传 console 版本的 `--ibvs-command-path`、`--ibvs-max-age-s` 等参数。

真机运动现象

| 现象 | 日志证据 | 判断 |
| --- | --- | --- |
| 持续抬臂 | 收到约 vx=+4.16、vy=+2.19、vz=+8.82mm/s；合速度接近10mm/s；末端 Z 从约-192.1变到-144.7mm。 | 上游给出的视觉速度明确要求 Base +Z；优先核对像素误差、相机映射和目标，不先改 IK。 |
| 有像素误差但暂停 | DUAL_Z_CONFLICT、Gate STOP、enabled=False、速度全零。 | 原 Gate 拒绝冲突模式导致整机停；不是仅因 External 对准。 |

新策略目标：正常时 X=External、Y=Wrist、Z=0.35EXT+0.65Wrist；Z 冲突时 X=External、Y=Wrist、Z=External。冲突只改变 Z 来源，不再冻结 X/Y。该策略仍需运行时验证。

七、右夹爪 O/V/C 单键控制

控制器已有 `--enable-gripper`（默认 True）、`--gripper-initial-mode current/open`（默认 current）、右夹爪闭合目标范围 -0.20 至 -0.05rad；内部命令解析器支持 `gripper:right:open/hold/close`。但 ENABLE 后切换 raw-key 模式，原始白名单不支持输入完整文本命令。

| 按键 | 动作 | 映射 |
| --- | --- | --- |
| O/o | 右夹爪打开 | gripper:right:open |
| V/v | 右夹爪保持当前测量位置 | gripper:right:hold |
| C/c | 右夹爪闭合 | gripper:right:close |

此前已给出控制器备份和 O/V/C 补丁：将 o/v/c 加入 raw-key whitelist；将按键映射到对应命令；更新启动提示。

当前状态：用户尚未贴出 O/V/C 补丁执行结果、grep/py_compile 回执或真机按键回执，因此仍是待执行/待验证。`--enable-gripper` 仅启用夹爪 LowCmd，不代表 O/V/C 已经可用，也不代表自动闭爪。

# 启动时显式启用夹爪（在 O/V/C 补丁完成后）--enable-gripper --gripper-initial-mode current # ENABLE 后计划按键O = OPENV = HOLDC = CLOSE

八、电脑卡死/黑屏排查

| 检查 | 结果 | 结论 |
| --- | --- | --- |
| OOM | 精确 OOM 查询无输出；最初 grep 只匹配到 USB ZOOM Camera（字符串误匹配）。 | 没有典型 OOM killer 证据。 |
| NVIDIA | 580.178.04 驱动正常加载；没有 NVRM: Xid/GPU fallen off bus。 | 没有典型 NVIDIA GPU Xid 证据。 |
| r8169 XID 641 | 来自 Realtek RTL8125B 网卡驱动。 | 不是 NVIDIA GPU Xid。 |
| GNOME Shell | 出现大量 Can't update stage views actor / needs an allocation，及 TypeError: this.actor is null。 | 更支持 GNOME Shell/Mutter/X11 图形刷新异常，但不能证明唯一根因。 |
| 资源快照 | RAM 31GiB、空闲后仅用2.7GiB；swap 15.3GiB用0；GPU显存12.3GiB用460MiB、47°C。运行时截图RAM约3.9GiB、GPU约1.0GiB。 | 快照未显示资源耗尽。 |

当前优先怀疑 GUI/显示栈、OpenCV 多窗口刷新与 GNOME Shell/Mutter 交互；三相机视觉是主要负载，但尚未证明是唯一根因。headless/单窗口低刷新率模式和长时间资源曲线尚未确认完成。

九、常用启动与检测命令

启动前检查

cd /home/robot/unitree_sim_isaaclab_threading/r1a7_threading_projectpgrep -af 'r1a7_cartesian_stream_control_kd30_posvel_wu_dqik_u010_v1.py' || truepgrep -af 'live_visual_guidance_shadow_v1|wire_grasp_fusion_shadow_v1|wire_grasp_control_gate_v1' || trueflock -n /tmp/r1a7_vr_g1ik_real.lock -c 'echo LOWCMD_LOCK_FREE'

终端 1：新视觉

PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv_sam2/bin/python -u \vision_dual_camera/task6_threading/dataset_tools/wire_unet/live_visual_guidance_shadow_v1.py

终端 2：Fusion

PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv_sam2/bin/python -u \vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/wire_grasp_fusion_shadow_v1.py

终端 3：Gate 直接输出

PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv_sam2/bin/python -u \vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/wire_grasp_control_gate_v1.py \--enable-output

检测终端：四级 JSON

/home/robot/miniconda3/envs/tv_sam2/bin/python - <<'PY'import json, timefrom pathlib import Pathfiles = { "EXT":"/tmp/r1a7_external_wire_grasp_v1.json", "WRIST":"/tmp/r1a7_wrist_wire_grasp_guidance.json", "FUSION":"/tmp/r1a7_wire_grasp_fusion_shadow_v1.json", "GATE":"/tmp/r1a7_dual_ibvs_velocity.json",}for name, fn in files.items(): print("\\n====", name, "====") try: d=json.loads(Path(fn).read_text()) except Exception as e:  print("READ ERROR:",e); continue print("age_s =",round(time.time()-float(d.get("timestamp",0)),3)) for k in ("guidance_valid","external_valid","wrist_valid","du_px","dv_px",           "fusion_mode","gate_state","enabled","proposed_vx_mm_s",           "proposed_vy_mm_s","proposed_vz_mm_s","vx_mm_s","vy_mm_s","vz_mm_s"):  if k in d: print(k,"=",d[k])PY

检测终端：控制器日志

tail -f /tmp/r1a7_official_real_lowcmd.log \| grep --line-buffered -E 'IBVS INPUT|VIRTUAL VR|TRACKING SPLIT|Stop requested|publisher stopped'

运行时 RAM/GPU

watch -n 1 'free -hnvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheaderps -eo pid,rss,%mem,%cpu,cmd --sort=-rss | grep -E "live_visual_guidance|wire_grasp|r1a7_cartesian" | grep -v grep'

十、问题、错误与处理记录

| 问题 | 原因/证据 | 处理/状态 |
| --- | --- | --- |
| IBVS 参数报错 | 把 console 参数加到 threading 控制器。 | 已确认版本差异；后续不加 --ibvs-*。 |
| LowCmd 锁冲突 | console PID 11052/11081 持锁。 | 必须先正常停止旧进程；启动前复查。 |
| 持续向上运动 | 上游视觉速度 +Z 接近限速。 | 需要核验方向映射/目标误差；不盲目翻符号。 |
| 有误差但暂停 | DUAL_Z_CONFLICT 被旧 Gate 判 STOP。 | Fusion/Gate 源码已改，运行时待验收。 |
| FREE 与 GRASP 目标选择 | 曾讨论 TIP→FREE，后用户明确恢复 GRASP。 | 最终保留 TIP→GRASP。 |
| O/V/C | raw-key 模式不能直接输入文本命令。 | 已提供补丁，尚无执行回执。 |
| 桌面黑屏 | 无 OOM/Xid，GNOME Shell 日志异常。 | 优先 GUI/headless 对照；根因未最终确认。 |

十一、当前完成清单与下一步

已完成/有证据

U-Net V1/V2 训练、阈值扫描、验证集指标和容差评估。

骨架连接、主簇、连通路径、端点/切线评估脚本已运行。

三相机实时分割、镊子检测与三轮运动稳定性验证。

Temporal V1 对 RAW 跳变的抑制效果已量化。

新视觉前端已输出 External/Right Wrist guidance JSON；ManualFreeTracker 字段已查明。

Fusion 的 vz=External Z 补丁与 Gate 的 DUAL_Z_CONFLICT 分支已执行并 grep 验证。

最终视觉目标选择为 GRASP；未完成 TIP→FREE 改造。

待完成/待验证

新 Fusion/Gate 策略的实时 JSON 验证与真机受控运动验收。

External/Wrist 的 du/dv 到 Base X/Y/Z 符号、方向和工作范围核验。

threading 控制器 IBVS JSON freshness/旧 enabled=true 文件风险复核。

O/V/C 右夹爪快捷键补丁执行、语法检查和空载动作验证。

三相机视觉 headless/单窗口对照，继续定位 GNOME 黑屏。

右腕 wire 困难视角、运动遮挡和端点/GRASP 几何精度的独立 GT 测试。

自动闭爪、穿孔和穿出确认均不属于已完成内容。

推荐执行顺序

检查 git status、Fusion/Gate grep、Guidance 的 grasp_point_yx、控制器 O/V/C whitelist；保留本地未提交修改。

机器人不运动时运行视觉+Fusion+Gate，分别采样 DUAL 和 DUAL_Z_CONFLICT 的 JSON。

确认冲突时 vx=External、vy=Wrist、vz=External，Gate 不再 STOP；核对方向符号。

在急停可触达、工作空间清空、LowCmd 唯一 publisher、数据新鲜的条件下做短时受控真机测试，同步记录 JSON 和机器人日志。

完成 O/V/C 补丁后先空载单独测试，再用于人工夹取；不自动闭爪。

做 headless/单窗口对照并保存 RAM/VRAM/GNOME 日志。

真机停止顺序：正常停止时先在机器人控制器终端按 Q 退出 LowCmd publisher，确认进程退出后再停止 Gate、Fusion、视觉。不要先停止 Gate 而让没有 freshness 检查的控制器继续读取最后一个 enabled=true 命令。异常运动时使用现场急停。

十二、关键文件与路径索引

| 用途 | 路径 |
| --- | --- |
| 新视觉前端 | vision_dual_camera/task6_threading/dataset_tools/wire_unet/live_visual_guidance_shadow_v1.py |
| ManualFreeTracker | vision_dual_camera/task6_threading/dataset_tools/wire_unet/live_manual_free_grasp_validation_v1.py |
| Fusion | vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/wire_grasp_fusion_shadow_v1.py |
| Control Gate | vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/wire_grasp_control_gate_v1.py |
| threading controller | transfer_control/experimental/r1a7_cartesian_stream_control_kd30_posvel_wu_dqik_u010_v1.py |
| LowCmd guard | transfer_control/tools/r1a7_lowcmd_guard.py |
| U-Net 权重 | /home/robot/r1a7_datasets/wire_unet/pilot_001/runs/unet_train_val_v2_resume/best.pt |
| Gate 输出 | /tmp/r1a7_dual_ibvs_velocity.json |
| 机器人日志 | /tmp/r1a7_official_real_lowcmd.log |
| Temporal 会话 | /home/robot/r1a7_datasets/wire_live_validation/temporal_v1_20261030_144103 |

十三、资料来源与说明

本文依据当前聊天中用户粘贴的终端输出、JSON 检查结果、截图描述和明确决策；并参考用户上传的《2026.10.21 R1-A7 双相机夹线视觉引导》及此前《R1-A7 三相机细线与镊子实时识别：对话与实验全记录（2026-10-30）》。

本次生成文档时没有连接机器人或重新读取机器人硬盘；所有路径、PID、IP 和指标均为对话中出现的记录，不保证后续运行时仍然相同。本文是工程记录，不是安全认证报告，也不代表自主夹取或穿孔任务已完成。

附录 B  2026-11-03 右腕张开镊子工作记录

以下为 11 月 3 日原始文档的逐项内容转录。11 月 3 日的“方案已设计、待部署”在 11 月 9 日的源码片段与用户反馈中可能升级为“已部署/用户反馈正常”，请以各章节日期辨别。

R1-A7 右腕张开镊子视觉与自动抓取

工作记录与操作归档

数据采集 · 标注 · U-Net 增量训练 · 双模型部署 · TIP 语义调整 · Auto-Grasp CLOSE Gate

| 项目 | 双臂人形机器人 / Unitree R1-A7 细线抓取与穿线 |
| --- | --- |
| 工作日期 | 2026-11-03 |
| 代码工程 | /home/robot/unitree_sim_isaaclab_threading/r1a7_threading_project |
| 数据根目录 | /home/robot/r1a7_datasets/wire_unet |
| 文档用途 | 阶段工作归档、代码变更追踪、复现实验、后续继续施工 |

版本：Worklog V1.0

0. 文档说明与当前状态

本文件对本次对话中围绕 R1-A7 右腕相机、张开镊子视觉识别、U-Net 增量训练和自动抓取闭合逻辑所完成的工作进行归档。内容按“问题定位 -> 数据 -> 标注 -> 训练 -> 离线验证 -> 实时部署 -> TIP 语义 -> CLOSE Gate”顺序整理，并区分“已执行验证”和“尚待执行/确认”的操作。

核心结论  原自动 OPEN 成功但 CLOSE 不触发的主要视觉根因，是 V2 只在闭合镊子外观上训练，张开后 Right Wrist tool mask / TIP 失真。V3 通过 Right-Wrist OPEN 数据增量训练后，在独立 test 上 tool Dice 从 0.5692 提升到 0.8915；随后代码改为 External/Left 继续 V2、Right Wrist 单独 V3。当前下一阶段重点是张开后的双尖端中点 TIP 以及 CLOSE Gate 对 Right Wrist 垂直误差的处理。

| 工作项 | 状态 | 说明 |
| --- | --- | --- |
| Right Wrist OPEN 数据采集 | 已完成 | 64 张 clean Orbbec RGB 图像 |
| 64 张标注与 QA | 已完成 | 64/64 valid；000043/000046 保留为真实遮挡 hard cases |
| V2 checkpoint 冻结 | 已完成 | SHA256 与原 V2 best 完全一致 |
| V3 mixed replay 训练 | 已完成 | best epoch=90；新域显著提升，旧域无退化 |
| 9 张独立 test | 已完成 | mean Dice 0.5094 -> 0.7290 |
| Right Wrist 单独使用 V3 | 代码已完成 | 双模型 patch + py_compile 通过 |
| 张开后双尖端中点 TIP | 方案已设计/待确认部署 | closed 用原 TIP；open 用两尖端中点 |
| 取消 Right Wrist ev 的 CLOSE gate | 待执行/待验证 | 备份已做；精确 patch 已给出 |
| OPEN_TRACKING 中 HOME 触发问题 | 待排查 | 曾导致 jitter 突增并破坏稳定窗口 |

0.1 本次工作范围

分析真机 auto-grasp 日志，定位 OPEN 后 Right Wrist 视觉退化与 CLOSE 条件不满足问题。

采集 Right Wrist 张开/过渡/遮挡镊子数据，建立独立增量数据集。

恢复历史标注规则与标注工具，完成 64 张图像标注和 QA。

冻结旧 V2 checkpoint，编写 V3 mixed replay trainer，进行增量训练。

在未参与训练的 9 张 test 上进行 V2/V3 对比。

修改 live vision 为双模型结构：External/Left 用 V2，Right Wrist 用 V3。

设计 Right Wrist 张开后双尖端中点 TIP；闭合时保持原尖端 TIP。

根据实际遮挡工况，提出取消 Right Wrist 垂直 ev 作为 CLOSE 判据、其他判据不变。

0.2 状态定义

| 状态 | 含义 |
| --- | --- |
| 已完成 | 已执行并有终端输出或指标验证。 |
| 代码已完成 | 代码修改与语法检查已完成；是否经过完整真机闭环另行说明。 |
| 待执行/待验证 | 已给出精确操作，但本次对话未收到最终执行结果。 |
| 待排查 | 已观察到现象，但尚未修改。 |

1. 问题背景与初始故障

自动抓取控制器的目标流程为：右夹爪初始物理闭合 -> 接近细线 -> 满足预开条件后发一次 OPEN -> OPEN 后继续视觉对准 TIP 与 GRASP -> 稳定满足 CLOSE 条件后发一次 CLOSE -> 接触/位置到位后进入 DONE。初始闭合状态不额外发送 CLOSE。

1.1 初始现象

真机测试中自动 OPEN 可以正常执行，右夹爪实际位置约 +0.1575 rad，目标 +0.1585 rad。

状态能够进入 OPEN_TRACKING，但长时间保持 open=1、close=0。

旧 V2 的 Right Wrist U-Net 主要学习闭合镊子外观；镊子张开后 tool segmentation 与 TIP 位置明显不稳定。

当 OPEN 后 TIP/GRASP 误差与 jitter 不满足 CLOSE Gate，控制器不会下发 CLOSE。

1.2 原自动抓取关键条件

| 阶段 | 关键条件 | 默认值/行为 |
| --- | --- | --- |
| CLOSED_MONITOR -> OPEN | 双相机有效；归一化距离进入预开区；jitter 小；持续稳定 | preopen norm <= 0.70；jitter <= 0.10；hold 0.35 s |
| WAIT_OPEN -> OPEN_TRACKING | 右夹爪测量值接近 OPEN 目标 | open tolerance 0.020 rad |
| OPEN_TRACKING -> CLOSE | External 和 Wrist 每轴归一化误差 + jitter + 稳定保持 | axis <= 0.10；jitter <= 0.035；hold 0.40 s |
| WAIT_CONTACT -> DONE | 接触保持或闭合位置到位 | close q=-0.0500；位置容差 0.015 rad |

1.3 原控制器与夹爪常量

关键参数

GRIPPER_OPEN_Q  = [4.86,  +0.1585]GRIPPER_CLOSE_Q = [-0.08, -0.0500]R1A7_AUTO_GRASP_ENABLE=1R1A7_AUTO_GRASP_PREOPEN_NORM_MAX=0.70R1A7_AUTO_GRASP_CLOSE_AXIS_NORM_MAX=0.10

原则  当时没有直接放宽 CLOSE 阈值，而是优先修复 Right Wrist OPEN 后的视觉模型。原因是阈值放宽可能让错误/跳变的 TIP 通过安全门槛。

2. 现有视觉与控制链路

2.1 三相机与 U-Net 前端

| 相机 | 设备/分辨率 | 当前职责 |
| --- | --- | --- |
| External USB ZOOM | 1280×720 | External wire / GRASP；X、Z 方向信息 |
| Right Wrist Orbbec | CPCBC53000C5；1280×800 | Right Wrist live wire/tool/TIP；Y、Z 信息 |
| Left Wrist Orbbec | CPCBC530007Z；1280×800 | 辅助固定线参考；Left visual |

实时视觉主程序：

vision_dual_camera/task6_threading/dataset_tools/wire_unet/live_visual_guidance_shadow_v1.py

其视觉链路复用 live_hybrid_locked_wire_live_tweezer_tip_v3，并实际使用 visual.v2.TweezerTipTrackerV2。External/Left 的线参考在人工 FREE 标定后锁定，Right Wrist 的线几何保持 live。

2.2 图像误差到机器人方向映射

| 视觉误差 | 机器人方向 |
| --- | --- |
| External +U | +X |
| External +V | -Z |
| Right Wrist +U | -Y |
| Right Wrist +V | -Z |

2.3 JSON 与融合输出

/tmp/r1a7_external_wire_grasp_v1.json/tmp/r1a7_wrist_wire_grasp_guidance.json/tmp/r1a7_wire_grasp_fusion_shadow_v1.json/tmp/r1a7_dual_ibvs_velocity.json

3. Right Wrist OPEN 数据采集

旧的 OpenCV/V4L2 采集方式会选到错误 video node，出现绿/紫色、噪声或深度格式画面。Right Wrist Gemini 336L 的多个 /dev/video 节点分别暴露 Z16、GREY/NV12、BA81/YV12 等格式，因此最终改为与正式视觉一致的 pyorbbecsdk Pipeline + COLOR_SENSOR。

3.1 采集脚本与设备

vision_dual_camera/task6_threading/dataset_tools/wire_unet/capture_right_wrist_open_orbbec_v2.pyRight Wrist serial: CPCBC53000C5Requested COLOR: 1280x800, RGB, 30 fps

3.2 数据集结果

| 项目 | 结果 |
| --- | --- |
| 数据集根目录 | /home/robot/r1a7_datasets/wire_unet/right_wrist_open_001 |
| 原始图像目录 | images_raw/session_20261103_112013 |
| metadata | metadata/session_20261103_112013.csv |
| 采集数量 | 64 张 |
| 内容 | OPEN / transition / closed / occlusion 混合，重点覆盖 Right Wrist OPEN |

3.3 采集时间组与最终 split

根据连续时间间隔，64 张样本形成 6 个主要 acquisition block。最终不采用随机相邻帧划分，而按采集时间块分配。

| Group | ID 范围 | 数量 | 用途 |
| --- | --- | --- | --- |
| G01 | 000001-000005 | 5 | test |
| G02 | 000006-000025 | 20 | train |
| G03 | 000026-000046 | 21 | train |
| G04 | 000047-000054 | 8 | val |
| G05 | 000055-000060 | 6 | train |
| G06 | 000061-000064 | 4 | test |

最终新数据 split  train=47，val=8，test=9；test 在 V3 训练期间保留不用。

4. 标注流程、规则与 QA

4.1 历史标注语义

mask：单通道 PNG、uint8、与源图同尺寸。

0=background，1=visible wire，2=tool/tweezer，255=真正无法可靠判断的区域。

只标注真实可见的 wire；遮挡后不可见的线段不得补画或跨遮挡连接。

tool/tweezer 按可见轮廓标注为 class 2。

张开镊子时，两片可见镊臂分别属于 class 2；两片之间的空气间隙仍是 background，不填充凸包。

wire 在工具前方真实可见时按可见像素标注；被工具遮挡的部分不标 wire。

mask 不进行双线性/双三次插值缩放。

4.2 Right-Wrist-only 标注工具

历史 annotator 会对每个 sample_id 循环 external/right_wrist/left_wrist。为了新数据集只包含 Right Wrist，复制 annotator 并仅将 CAMERAS 改为 ("right_wrist",)，不修改原工具。

vision_dual_camera/task6_threading/dataset_tools/annotate_wire_unet_v2_right_wrist_open.pyCAMERAS = ("right_wrist",)

4.3 完成情况与 QA

| 指标 | 结果 |
| --- | --- |
| expected/images/masks/valid | 64 / 64 / 64 / 64 |
| wire pixels | min 274；median 397.0；max 560 |
| tool pixels | min 1878；median 2815.0；max 3487 |
| ignore=255 | 本批 64 张未使用 |
| QA contact sheets | 4 页，覆盖全部 64 张 |

QA 将 000043 与 000046 标记为 tool pixel 数较低的 review candidate。人工检查确认：二者属于真实遮挡、仅一片镊臂可见，并无漏标，wire 标注正确，因此作为 hard cases 保留，不重标。

5. V2 模型冻结与 V3 增量训练设计

5.1 V2 checkpoint 冻结

原 V2 best:/home/robot/r1a7_datasets/wire_unet/pilot_001/runs/unet_train_val_v2_resume/best.pt冻结副本:/home/robot/r1a7_datasets/wire_unet/right_wrist_open_001/runs/v3_right_wrist_open/init_v2_best.ptSHA256:cb7ae5f4664bb5f1a558beeb941a49847cc1694f4d343024391973d4052c1687

原 V2 与冻结副本 SHA256 完全一致，旧模型不会被覆盖。

5.2 为什么不能直接用原 V2 trainer 训练 64 张新图

原 TrainDataset 会把每个 sample_id 强制展开为 external/right_wrist/left_wrist 三个 camera item；原 validate 也按三相机遍历。因此 Right-Wrist-only 新数据不能直接作为原 dataset 根目录使用，否则会要求不存在的 External/Left 图像。

for sid in sample_ids:    for camera in ("external", "right_wrist", "left_wrist"):        self.items.append((sid, camera))

5.3 V3 mixed replay trainer

vision_dual_camera/task6_threading/dataset_tools/wire_unet/train_wire_unet_v3_right_wrist_open.py

V3 不改变原 V1/V2 的 crop 与 loss 语义，而是新增 MixedTrainDataset：旧 pilot train 按三相机 replay，新 Right Wrist train 以重复采样方式增强。

| 项目 | 设置 |
| --- | --- |
| 旧 train replay | 35 个旧 sample × 3 camera = 105 images |
| 新 train unique | 47 images |
| 新数据重复权重 | new_repeat=3 -> 141 items |
| 每 epoch 总 item | 246 |
| batch size | 2 |
| crop / val stride | 512 / 384 |
| learning rate | 2e-5 |
| weight decay | 1e-4 |
| optimizer | AdamW；V3 保留模型权重但重置 optimizer moments |
| loss | 0.5 × weighted CE + 0.5 × foreground Dice loss |
| best selection | 0.4 × old_val + 0.6 × new_val，且 old_val 不得低于 baseline-0.02 |

实际训练命令

PYTHONNOUSERSITE=1 \/home/robot/miniconda3/envs/tv_sam2/bin/python -u \vision_dual_camera/task6_threading/dataset_tools/wire_unet/train_wire_unet_v3_right_wrist_open.py \--old-dataset /home/robot/r1a7_datasets/wire_unet/pilot_001 \--new-dataset /home/robot/r1a7_datasets/wire_unet/right_wrist_open_001 \--resume /home/robot/r1a7_datasets/wire_unet/right_wrist_open_001/runs/v3_right_wrist_open/init_v2_best.pt \--epochs 20 --batch-size 2 --crop-size 512 --val-stride 384 \--lr 2e-5 --weight-decay 1e-4 --new-repeat 3

6. V3 训练结果与独立测试

6.1 Baseline 与 best epoch

| 指标 | V2 baseline | V3 best (epoch 90) | 变化 |
| --- | --- | --- | --- |
| Old val mean FG Dice | 0.750200 | 0.7524 | +0.0022（无遗忘） |
| New Right Wrist val mean FG Dice | 0.477433 | 0.7728 | +0.2954 |
| Combined score | 0.586540 | 0.764621 | +0.178081 |
| Old guard floor | 0.730200 | 满足 | 全程 old_guard=1 |

训练曲线在 epoch 90 达到最佳 combined score；epoch 91-92 没有继续提升，因此不再追加 epoch，保留 epoch 90 的 best.pt。

/home/robot/r1a7_datasets/wire_unet/right_wrist_open_001/runs/v3_mixed_replay/best.pt

6.2 9 张独立 test：V2 vs V3

| ID | V2 mean | V3 mean | Delta | Tool V2 -> V3 |
| --- | --- | --- | --- | --- |
| 000001 | 0.4826 | 0.6575 | +0.1750 | 0.6423 -> 0.9010 |
| 000002 | 0.5300 | 0.6738 | +0.1439 | 0.6916 -> 0.9005 |
| 000003 | 0.6293 | 0.7500 | +0.1207 | 0.7475 -> 0.9107 |
| 000004 | 0.5722 | 0.7082 | +0.1360 | 0.7186 -> 0.9001 |
| 000005 | 0.5650 | 0.7074 | +0.1423 | 0.7402 -> 0.9045 |
| 000061 | 0.2924 | 0.7238 | +0.4314 | 0.0956 -> 0.8320 |
| 000062 | 0.2789 | 0.7569 | +0.4779 | 0.0174 -> 0.8430 |
| 000063 | 0.2473 | 0.7026 | +0.4553 | 0.0451 -> 0.8465 |
| 000064 | 0.5817 | 0.7981 | +0.2164 | 0.7095 -> 0.9479 |

| Overall test | V2 | V3 | Delta |
| --- | --- | --- | --- |
| Wire Dice | 0.449626 | 0.566445 | +0.116819 |
| Tool Dice | 0.569188 | 0.891532 | +0.322344 |
| Mean FG Dice | 0.509407 | 0.728989 | +0.219582 |

结论  V3 已经解决 Right Wrist OPEN/遮挡外观下最严重的 tool segmentation 退化，同时 wire 也改善。000061-000063 的 tool Dice 从接近失效（0.017-0.096）恢复到约 0.83-0.85。

7. 实时视觉双模型部署

7.1 原实现

live_visual_guidance_shadow_v1.py 原先仅调用一次 base.load_model()，同一个 model 通过 base.run_one(model, device, frame) 用于 External、Right Wrist、Left Wrist。base 实际模块为 live_wire_unet_three_camera.py，load_model() 无 checkpoint 参数并使用固定 CHECKPOINT。

7.2 部署策略

最小风险策略  External 和 Left Wrist 继续使用旧 V2；只有 Right Wrist 使用 V3。这样修复问题域，不改变已经稳定的 External/Left 行为。

External     -> V2Left Wrist   -> V2Right Wrist  -> V3 (v3_mixed_replay/best.pt)

7.3 已执行备份

vision_dual_camera/task6_threading/dataset_tools/wire_unet/live_visual_guidance_shadow_v1.py.bak_before_right_wrist_v3_20261103_165708

7.4 已执行代码修改

在 live_visual_guidance_shadow_v1.py 中：

新增 import torch。

新增 RIGHT_WRIST_V3_CHECKPOINT 常量。

新增 load_model_from_checkpoint(checkpoint, device) helper，使用 base.UNet 加载指定 checkpoint。

保留原 model, device = base.load_model() 作为 V2。

新增 model_right_wrist 加载 V3。

三相机循环增加 active_model：right_wrist 使用 model_right_wrist，其余使用 model。

推理模型选择核心

active_model = (    model_right_wrist    if camera == "right_wrist"    else model)detections[camera], fps_values[camera] = base.run_one(    active_model, device, frames[camera])

修改后已执行 py_compile，输出 LIVE_DUAL_MODEL_SYNTAX_OK。

8. Right Wrist TIP 语义变更：闭合单尖端 / 张开双尖端中点

在 V3 解决 tool mask 后，新的几何问题出现：镊子完全张开后存在两个物理尖端，因此 Right Wrist 控制 TIP 不应再绑定其中任意一个单尖端，而应使用两个尖端的中点；夹爪闭合后仍保持原来的单尖端定义。

8.1 现有 TweezerTipTrackerV2 行为

update(tool_mask, grasp_yx) 直接接收 tool mask 与 GRASP。

先统计 tool_pixels。

保留最大 connected component，以避免小分割噪声成为假前沿。

对 tool 主区域做 PCA，GRASP 决定 PCA 轴前向。

TIP 取极前沿 contour band。

这一逻辑对闭合镊子有效，但张开后两片镊臂可能形成两个独立/近独立结构；只保留最大 component 会丢失另一尖端。

8.2 目标 TIP 定义

CLOSED:    tip_yx = 原 TweezerTipTrackerV2 单尖端OPEN:    tip_1 = 第一片镊臂尖端    tip_2 = 第二片镊臂尖端    tip_yx = ((y1+y2)/2, (x1+x2)/2)

8.3 设计方案（尚待确认是否已部署）

已设计新增文件 live_right_wrist_tweezer_tip_midpoint_v1.py，并计划只替换 Right Wrist 的 tip tracker；External/Left 继续使用原 TweezerTipTrackerV2。拟增加诊断字段：tip_mode、tip_pair_yx、tip_pair_separation_px、blade1_pixels、blade2_pixels。

| 模式 | 触发 | 输出 |
| --- | --- | --- |
| SINGLE_CLOSED | 单显著 component 或双尖端间距很小 | 原单 TIP |
| SINGLE_FALLBACK | 双尖端提取失败 | 回退原 V2 TIP |
| OPEN_MIDPOINT | 两个有效尖端且间距超过阈值 | 双尖端中点 TIP |

注意  本次对话没有收到该 midpoint tracker 的创建/py_compile/运行回传，因此应视为“方案已设计，部署状态待确认”，不能当作已完成代码。

8.4 当前观察到的 TIP/GRASP 垂直跳变

实时视觉日志中 Right Wrist 的 du 已接近 0，但 dv 经常在约 +20~+32 px 与 -15~-30 px 之间切换。该现象说明 OPEN 状态下垂直定位仍可能受遮挡、GRASP 点偏移或 TIP 模式切换影响。

9. Auto-Grasp CLOSE Gate：取消 Right Wrist 垂直 ev 判据

9.1 现场日志分析

控制日志显示 OPEN_TRACKING 中 Right Wrist 水平方向 eu 已经很小，但 ev 长时间远高于 0.10。例如：WRIST en=0.556, eu=-0.026, ev=+0.555, jit=0.014。此时 eu 与 jitter 合格，但 ev 单独阻止 CLOSE。

同时，实际工况中夹爪靠近细线后会遮挡 GRASP 点，Right Wrist 垂直方向 GRASP 识别可能偏移到其他位置，因此 Right Wrist ev 不再适合作为闭合必要条件。

9.2 当前源码中的 CLOSE 条件

ext_close = (    abs(em["eu"]) <= self.close_axis_norm_max    and abs(em["ev"]) <= self.close_axis_norm_max    and em["jitter"] <= self.close_jitter_max)wrist_close = (    abs(wm["eu"]) <= self.close_axis_norm_max    and abs(wm["ev"]) <= self.close_axis_norm_max    and wm["jitter"] <= self.close_jitter_max)

9.3 用户确认的新规则

变更要求  只取消 Right Wrist 的垂直 ev CLOSE 判定；External 的 eu/ev/jitter、Right Wrist 的 eu/jitter、0.40 s 稳定保持、OPEN 状态及其他逻辑全部保持不变。

修改后的 wrist_close

wrist_close = (    abs(wm["eu"]) <= self.close_axis_norm_max    and wm["jitter"] <= self.close_jitter_max)

| 条件 | 修改后是否参与 CLOSE |
| --- | --- |
| External &#124;eu&#124; <= 0.10 | 是 |
| External &#124;ev&#124; <= 0.10 | 是 |
| External jitter <= 0.035 | 是 |
| Right Wrist &#124;eu&#124; <= 0.10 | 是 |
| Right Wrist &#124;ev&#124; <= 0.10 | 否（取消） |
| Right Wrist jitter <= 0.035 | 是 |
| 连续稳定约 0.40 s | 是 |

9.4 备份与当前执行状态

已执行控制器备份命令，文件名模式为：

r1a7_cartesian_stream_control_kd30_posvel_wu_dqik_u010_auto_grasp_direct_v1.py.bak_before_remove_wrist_ev_close_gate_<timestamp>

之后已精确定位源码行：wrist_close 中包含 wm["eu"]、wm["ev"]、wm["jitter"] 三项。删除 wm["ev"] 的精确 patch 已给出，但本次对话尚未收到 patch 执行和 py_compile 回传，因此文档将其状态登记为“待执行/待验证”。

9.5 为什么不能误读 WRIST en

即使取消 Right Wrist ev CLOSE 判定，日志中的 WRIST en 仍可能很大，因为 en=hypot(eu,ev)。修改后不能用 en<0.10 判断是否允许 CLOSE；Right Wrist CLOSE 只看 |eu| 和 jitter。

10. 其他控制层变更与风险点

10.1 手动 O/C 恢复

此前 auto-grasp enabled 时曾阻断手动 Right OPEN/CLOSE。已将该额外阻断逻辑去掉，使手动 O/C 继续走原有 gripper_command 分支；自动 open_issued / close_issued latch 不因手动动作而重置。

行为说明  如果 auto 状态仍为 OPEN_TRACKING 且 close_issued=False，手动 C 可以物理闭合，但之后若自动视觉条件满足，控制器仍可能再下发一次自身的 CLOSE 目标。

10.2 HOME 触发问题

多次日志中在 OPEN_TRACKING 后出现 [STREAM START] HOME。HOME 运动会使视觉误差突然变化，Right Wrist jitter 曾从约 0.01-0.05 突增到 0.35 左右，从而必然打断 0.40 s CLOSE 稳定窗口。该问题与 Right Wrist ev gate 是独立问题。

当前状态  HOME 触发原因尚未完成源码检查和修改。后续应单独定位 HOME 触发条件，避免 OPEN_TRACKING 稳定对准阶段被非预期 HOME 打断。

10.3 真实机器人安全约束

LowCmd publisher 只允许一个实例。

锁文件 /tmp/r1a7_vr_g1ik_real.lock 不手工删除。

真机测试前检查 pgrep -af "r1a7_cartesian_stream_control|lowcmd|g1ik"。

停止顺序：Controller(Q) -> Gate -> Fusion -> Vision；不要在 controller 仍运行时先停 Gate。

紧急情况使用物理 E-stop。

11. 文件新增、修改、备份、删除登记

| 类型 | 文件/对象 | 内容 | 状态 |
| --- | --- | --- | --- |
| 新增/使用 | capture_right_wrist_open_orbbec_v2.py | Right Wrist Orbbec RGB 数据采集 | 已执行 |
| 新增 | annotate_wire_unet_v2_right_wrist_open.py | Right-Wrist-only 标注工具；原 annotator 不改 | 已执行 |
| 新增 | train_wire_unet_v3_right_wrist_open.py | 旧三相机 replay + 新 Right Wrist 高权重 mixed training | 已执行 |
| 新增 | eval_v2_v3_right_wrist_open_test.py | 9 张 test 上 V2/V3 对比与可视化 | 已执行 |
| 修改 | live_visual_guidance_shadow_v1.py | 双模型：External/Left=V2，Right Wrist=V3 | 已执行+语法通过 |
| 备份 | live_visual_guidance_shadow_v1.py.bak_before_right_wrist_v3_20261103_165708 | 双模型 patch 前备份 | 已执行 |
| 计划新增 | live_right_wrist_tweezer_tip_midpoint_v1.py | 张开双尖端中点 TIP；闭合回退原 TIP | 待确认部署 |
| 备份 | auto_grasp_direct_v1.py.bak_before_remove_wrist_ev_close_gate_<timestamp> | 删除 wrist ev gate 前备份 | 已执行 |
| 计划修改 | auto_grasp_direct_v1.py | 从 wrist_close 删除 wm["ev"] 条件 | 待执行/待验证 |
| 逻辑删除 | Right Wrist CLOSE 的 ev 判据 | 只删除判定条件，不删除文件 | 待执行/待验证 |
| 源文件删除 | 无 | 本次未执行源文件删除 | 无 |

12. 关键路径与命令汇总

12.1 视觉 / 融合 / Gate

# VisionPYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv_sam2/bin/python -u \vision_dual_camera/task6_threading/dataset_tools/wire_unet/live_visual_guidance_shadow_v1.py# FusionPYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv_sam2/bin/python -u \vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/wire_grasp_fusion_shadow_v1.py# GatePYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv_sam2/bin/python -u \vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/wire_grasp_control_gate_v1.py --enable-output

12.2 Auto-Grasp Controller

R1A7_AUTO_GRASP_ENABLE=1 \R1A7_AUTO_GRASP_PREOPEN_NORM_MAX=0.70 \R1A7_AUTO_GRASP_CLOSE_AXIS_NORM_MAX=0.10 \PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv/bin/python -u \transfer_control/experimental/r1a7_cartesian_stream_control_kd30_posvel_wu_dqik_u010_auto_grasp_direct_v1.py \--interface enp6s0 --domain-id 0 --host-ip 192.168.123.223 \--ik-frequency 30 --publish-frequency 250 --max-joint-speed 2.0 \--cartesian-jog-speed-mm-s 10 --cartesian-jog-accel-mm-s2 20 \--gripper-initial-mode current --no-right-gripper-open-cap-current --no-record-d435i

12.3 语法检查模板

PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv_sam2/bin/python -m py_compile <vision_script.py>PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv/bin/python -m py_compile <controller.py>

13. 推荐的下一步操作流程

后续建议严格按单变量验证推进，避免同时改 TIP、CLOSE Gate、jitter/hold 阈值和 HOME 行为，导致无法判断哪项修改真正有效。

执行并确认 auto_grasp_direct_v1.py 中只删除 Right Wrist wm["ev"] CLOSE 条件；运行 py_compile。

保持 close_axis_norm_max=0.10、close_jitter_max=0.035、close_hold_s=0.40，不做其他阈值调整。

确认 live vision 启动时同时打印旧 V2 checkpoint 与 Right Wrist V3 checkpoint。

先在自动抓取关闭状态验证 CLOSED/OPEN Right Wrist 的 tool mask、TIP 和 guidance_valid。

若启用双尖端 midpoint tracker，日志必须额外记录 tip_mode / tip_pair_separation_px，确认 OPEN 时确实进入 OPEN_MIDPOINT。

重新启动 Fusion、Gate、Auto-Grasp Controller，观察 OPEN_TRACKING。

重点看：External eu/ev/jitter、Right Wrist eu/jitter、close_ready 连续 0.40 s、CLOSE_ONCE、WAIT_CONTACT、DONE。

如果仍不 CLOSE，优先判断 jitter 或 HOME 是否打断窗口，而不是重新加入/放宽 Right Wrist ev。

如果出现非预期 HOME，独立检查 HOME 触发代码并修复，避免把视觉问题与运动状态机问题混在一起。

13.1 新 CLOSE Gate 的预期日志判断

例如：EXT   eu=+0.023 ev=-0.035 jit=0.016   -> 合格WRIST eu=-0.026 ev=+0.555 jit=0.014   -> 修改后合格（ev 被忽略）若持续 >= 0.40 s：[AUTO GRASP DIRECT] CLOSE_ONCE target=-0.05000state=WAIT_CONTACT, close=1

13.2 仍会阻止 CLOSE 的情况

External eu 或 ev 超过 0.10。

External jitter > 0.035。

Right Wrist eu > 0.10。

Right Wrist jitter > 0.035。

上述条件未连续保持 0.40 s。

视觉 invalid 或 OPEN 后视觉丢失达到 abort 条件。

HOME/其他运动使视觉误差或 jitter 突变。

14. 阶段结论

本阶段已经从“自动 OPEN 后视觉失效、长期无法 CLOSE”的问题，推进到“Right Wrist OPEN segmentation 已显著修复，实时系统已具备 Right-Wrist-only V3 模型能力”。V3 在独立测试集上的提升证明模型训练方向正确。

当前阻塞点已经从“U-Net 看不见张开镊子”转移到更具体的几何与状态机问题：一是张开后两个物理尖端应如何定义单一控制 TIP；二是夹具遮挡 GRASP 点后，Right Wrist 垂直误差 ev 不应再作为 CLOSE 的必要判据；三是 OPEN_TRACKING 中非预期 HOME 可能破坏稳定窗口。

当前推荐基线  模型层保持 External/Left=V2、Right Wrist=V3；控制层只取消 Right Wrist ev CLOSE gate，其余阈值不动；之后单独验证双尖端 midpoint TIP 与 HOME 行为。

附录 A. 关键模型与数据路径

| 对象 | 路径 |
| --- | --- |
| 旧数据集 | /home/robot/r1a7_datasets/wire_unet/pilot_001 |
| 新数据集 | /home/robot/r1a7_datasets/wire_unet/right_wrist_open_001 |
| V2 best | pilot_001/runs/unet_train_val_v2_resume/best.pt |
| V3 init | right_wrist_open_001/runs/v3_right_wrist_open/init_v2_best.pt |
| V3 best | right_wrist_open_001/runs/v3_mixed_replay/best.pt |
| 新 split | right_wrist_open_001/splits_right_wrist_open_grouped_v1.json |
| test 输出 | right_wrist_open_001/eval_v2_v3_test |

附录 B. 变更回滚要点

双模型视觉回滚：将 live_visual_guidance_shadow_v1.py 恢复为 .bak_before_right_wrist_v3_20261103_165708。

CLOSE Gate 回滚：使用 .bak_before_remove_wrist_ev_close_gate_<timestamp> 恢复控制器。

V2/V3 checkpoint 均独立保存；不要覆盖 pilot_001 的 V2 best。

不要手工删除 LowCmd 锁文件；真机回滚时先停止 controller 再替换代码。

附录 C  路径、版本、参数与回滚速查

C.1 当前运行入口

| 环节 | 脚本（相对项目根目录） |
| --- | --- |
| Visual V3 | vision_dual_camera/task6_threading/dataset_tools/wire_unet/live_grasp_to_insertion_visual_v3.py |
| Right Wrist Midpoint tracker | vision_dual_camera/task6_threading/dataset_tools/wire_unet/live_right_wrist_tweezer_tip_midpoint_v1.py |
| Fusion | vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/wire_grasp_fusion_shadow_v1.py |
| Gate | vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/wire_grasp_control_gate_v1.py |
| Controller | transfer_control/experimental/r1a7_cartesian_stream_control_kd30_posvel_wu_dqik_u010_auto_grasp_direct_v1.py |
| LowCmd guard | transfer_control/tools/r1a7_lowcmd_guard.py |

C.2 关键参数速查

| 参数 | 当前记录 | 作用与边界 |
| --- | --- | --- |
| GRASP_EXT_NEAR_ENTER_PX / EXIT_PX | 45 / 65 px | Fusion 抓取近距离滞回；插孔阶段需退出 |
| 远距离 Z 融合 | 0.35 EXT + 0.65 Wrist | 同向分支；冲突优先 EXT |
| 近距离 Z 融合 | 1.0 EXT + 0.0 Wrist | 只在 GRASP 且 grasp_ext_near |
| BLIND_GRACE | 0.40 s / 0.40 / 2.0mm/s | 短时掉帧；非有视觉修正的安全保证 |
| Gate MAX_INPUT_AGE_S | 0.20 s | 以现场版本复核 |
| External/Wrist error thresholds | 常量 350px 保留，硬门槛已移除 | 不等于可以忽略大误差风险 |
| Jaw calibration | 12 样本、gap 中位数 | 要求清楚区分 OPEN 和伪双连通域 |
| Jaw corridor | 0.80 × open gap | 投影进入通道需有 External X/Z 近距门槛 |
| CLOSE_SETTLE | 0.40 s | 进入后冻结、只发一次 CLOSE |
| Right gripper q | OPEN +0.1585 / CLOSE -0.0500 | 保留 11/03 已知值 |
| Controller trend-test cap | 3 mm/s | 本轮启动命令建议值，非性能上限 |

C.3 备份与验证回执索引

| 日期/环节 | 文件备份或检查 | 证据 |
| --- | --- | --- |
| 11/03 视觉双模型 | live_visual_guidance_shadow_v1.py.bak_before_right_wrist_v3_20261103_165708 | 历史文档：双模型语法通过 |
| 11/07 Gate | wire_grasp_control_gate_v1.py.bak_before_remove_image_error_gate_20261107_172954 | IMAGE_ERROR_HARD_GATE_DISABLED_OK / py_compile 通过 |
| 11/09 Fusion | wire_grasp_fusion_shadow_v1.py.bak_before_grasp_ext_z_priority_20261109_164056 | PATCH_OK / py_compile 通过 |
| 11/09 V3/Controller Jaw | bak_before_jaw_corridor_<timestamp>（命令已提供） | 用户报告功能正常；具体备份文件名待现场确认 |
| 11/09 成功快照 | SUCCESS_GRASP_JAW_CORRIDOR_<timestamp>（命令已提供） | 未见执行回执，需确认快照真实存在 |

C.4 归档资料与限制

源材料：① 《R1-A7 当前对话操作与技术路线汇总》（2026-11-02）；② 《R1-A7 右腕张开镊子视觉与自动抓取工作记录》（2026-11-03）；③ 本聊天中 2026-11-07 与 2026-11-09 终端输出、源码 grep/py_compile、JSON 数值及用户对“功能正常”的报告；④ 用户提供的 V3/RightWristTip/Controller 检查日志。

本次编写工作文档没有访问真实机器人、没有训练模型，也没有修改机器人磁盘中的任何源码或数据。孔位数据集的精确训练数字、跨相机孔位 GT、真实插孔成功率等尚待使用原始实验日志补录。完整原始归档留在附录 A/B，作为未来对话交接时避免丢失的历史证据。
