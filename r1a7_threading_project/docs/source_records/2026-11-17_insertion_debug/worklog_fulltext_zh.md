# 20261117 原文全文转录

原件字节不变；以下按OOXML段落顺序展开表格，保留段内换行。原始代码排版以DOCX为准。

R1-A7  双固定相机细线操作

抓取与插孔研发调试
全流程工作汇总

视觉检测、夹爪自动抓取、融合与 Gate、XY 优先插孔控制、FREE 端点稳定跟踪



文档属性

内容

截止日期

2026 年 11 月 17 日

适用项目

R1-A7 双臂机器人 / 细线抓取与插孔

记录范围

本次项目对话中的操作、终端反馈、补丁、备份，以及提供的 18:42 trace/full-chain

主要用途

实验复盘、版本交接、错误定位、下一轮调试前检查



当前结论  抓取侧的原始 Fusion + 原始 Gate 曾恢复并获得手臂/夹爪可动作的实测反馈；插孔控制已演进到 Shaper V6/V7 并编译通过，但最终 FREE 物理身份仍有误判，不能认定插孔闭环已经完成验证。



文档分为：现状总览、系统与程序、抓取优化、插孔控制演进、真实数据复盘、版本与备份、终端 SOP、待办和风险。

状态标记：✓ 终端/数据可证；◐ 有日志或截图但缺少完整源码/闭环验证；○ 方案或拟议，尚未见本次终端确认。



一、现状概览与重要结论

研究目标：将 R1-A7 使用镊子夹持细线后，通过 External 与 Left Wrist 两路固定相机观测细线自由端 FREE 和已确认孔中心 HOLE，先优化 X/Y 图像对齐，再按对齐程度渐进允许 Z 位移；Z 运动期间继续校正 X/Y，避免细线碰撞板面或端点跟踪错误。

模块

目前可确认的状态

后续处理

抓取链路

✓ 原始 wire_grasp_control_gate_v1.py 能让手臂与夹爪产生动作；Fusion 输出路径曾恢复为原始 shadow 文件。

保留经过验证的 Gate 与夹爪状态机，不继续试错式改动。

视觉三相机

✓ 三路均实时采集；插孔权重只给 External、Left，Right 继续监测。

区分画面 FREE-LIVE 与实际控制 FREE，记录同周期对应关系。

XY 优先插孔

✓ Shaper V6 的连续控制补丁、V7 的去 DONE 闭锁补丁终端编译通过。

重新确认完整本地源码、负 Z 符号、各安全停止条件。

端点防跳

✓ 视觉 V2（14 px）和 Shaper V3（4 帧）曾打补丁编译；○ V5 锁一次策略是否实际应用没有最终确证。

先修自由端的物理身份，而不是仅收紧阈值。

插孔完成

◐ 看到接近孔但仍有 LEFT FREE 识别偏移；无最终成功穿孔验证。

建立真正 FREE / HOLE 标签和连续质量评估，先离线/无运动回放。



关键架构约束  “最开始的 Gate”指 wire_grasp_control_gate_v1.py；不是 wire_grasp_control_gate_direct_insertion_v1.py。不要让 Fusion 和 Shaper 同时写同一个 Gate 输入 JSON，否则会发生 writer 竞争。



二、系统拓扑与代码目录

2.1 两种数据流不能混用

A. 已验证的原始抓取拓扑（无 Shaper writer）
Visual → wire_grasp_fusion_shadow_v1.py
       → /tmp/r1a7_wire_grasp_fusion_shadow_v1.json
       → wire_grasp_control_gate_v1.py
       → /tmp/r1a7_dual_ibvs_velocity.json → Controller

B. 本轮插孔主动整形拓扑（唯一 Gate-facing writer）
Visual → Fusion → /tmp/r1a7_wire_grasp_fusion_preinsert_v1.json
                → wire_insertion_xy_priority_shaper_v1.py
                  (--enable-output)
                → /tmp/r1a7_wire_grasp_fusion_shadow_v1.json
                → 原始 Gate → Controller



B 拓扑的 GRASP_PASS 使 Shaper 在抓取阶段透传上游 Fusion 对象；在 INSERTION_TRACKING 才使用插孔控制。A/B 切换时必须核验 Fusion 输出路径、Shaper SRC 路径、Gate 输入路径以及各文件唯一写入者。

2.2 环境、代码路径

ROOT=/home/robot/unitree_sim_isaaclab_threading/r1a7_threading_project
PY=/home/robot/miniconda3/envs/tv_sam2/bin/python
WIRE=$ROOT/vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire
VIS=$ROOT/vision_dual_camera/task6_threading/dataset_tools/wire_unet/live_grasp_to_insertion_visual_v3.py
FUSION=$WIRE/wire_grasp_fusion_shadow_v1.py
SHAPER=$WIRE/wire_insertion_xy_priority_shaper_v1.py
GATE=$WIRE/wire_grasp_control_gate_v1.py
CTRL=$ROOT/transfer_control/experimental/r1a7_cartesian_stream_control_kd30_posvel_wu_dqik_u010_auto_grasp_direct_v1.py



程序

作用

当前工作原则

live_grasp_to_insertion_visual_v3.py

三相机视频、线段/镊子观测、孔选择/确认、FREE 输出；阶段 GRASP/WAIT_HOLE_TARGET/INSERTION_TRACKING。

抓取相机 External+Right；插孔相机 External+Left。

wire_grasp_fusion_shadow_v1.py

组合抓取指令；插孔时使用相应感知输出，向中间 JSON 发布。

主动 Shaper 拓扑下改写至 preinsert，避免覆盖 Gate 输入。

wire_insertion_xy_priority_shaper_v1.py

GRASP_PASS；插孔 XY 对齐、Z 放权、FREE 身份约束。

2026-11-17 最后可证标记 V6+V7；本地实际 SHA 待归档。

wire_grasp_control_gate_v1.py

按照融合 mode/有效性把 proposed 转成实际控制速度。

原始 Gate 文件尽量冻结。

Controller 直控脚本

消费 Gate JSON，发关节/笛卡尔 lowcmd；夹爪闭合状态机，trace/full-chain 记录。

不因 Shaper 控制问题改动夹爪状态机。



2.3 阶段与相机角色

阶段

External

Right Wrist

Left Wrist

阶段输出

GRASP

夹取 X + 部分 Z

夹取 Y + 部分 Z、镊口

实时画面/辅助

OPEN_TRACKING → WAIT_CONTACT 等由 Controller 处理

WAIT_HOLE_TARGET

选孔、确认

监视与记录

选同一物理孔、确认

冻结 selected hole_uv，等待双视角目标确认

INSERTION_TRACKING

FREE/HOLE 的 X 投影

持续监视/记录但无插孔控制权

FREE/HOLE 的 Y 与 Z 相关投影

XY 先调整；XY 近后 Z 放权；XY 在下降期间持续修正



三、视觉性能优化与坐标标定

3.1 视觉处理链的已完成优化

早期三相机串行读图、逐相机串行 U-Net TILE512 / stride384 / threshold≈0.86，18 tiles；主要慢在 GPU 推理、几何/聚类后处理与阻塞读帧。后续采用最新帧读取线程、ThreadPool 几何并行、torch.compile(mode="reduce-overhead")。

指标/阶段

早期或基线

后续测得

说明

RAW 相机采集 p50

约 23.2 ms

维持实时三路

没有将降分辨率作为首要措施

整条推理链 p50

约 168.6~173.4 ms

约 100~114 ms

理论吞吐从约 5.9 FPS 提至约 9~10 FPS

U-Net 纯网络

约 69.18 ms

torch.compile 约 50.16 ms

约减少 27.5%；具体性能与 GPU/环境相关

几何聚类处理

约 76.9 ms

线程并行约 32.7 ms

避免串行每相机阻塞

batch 推理

增大 batch 后更慢

tile batch=1 更合适

当前 RTX 4070 SUPER 工况



数据时延 metadata 已包含或曾设计：source_frame_seq、source_frame_monotonic_s、source_frame_age_ms_at_publish、camera_set_skew_ms、vision_wire_total_ms、visual_cycle_id。插孔阶段建议用 FREE 的真实测量时间戳，而不是已冻结 HOLE 的时间戳做 freshness。

3.2 左相机标定与公式边界

Left 图像相对运动 Jacobian（单位按既有标定记录）：
  Δu = 39.7407 ΔX + 15.5967 ΔY - 31.7103 ΔZ
  Δv = 52.9663 ΔX + 12.3808 ΔY - 39.5067 ΔZ

固定 X、求解 Y/Z 时的逆关系：
  ΔY ≈ +0.176705 Δu - 0.141834 Δv       (mm)
  ΔZ ≈ +0.055377 Δu - 0.069761 Δv       (mm)

历史 Y 速度控制律（K≈0.25/s）：
  vy = +0.009833 * du + 0.007806 * dv  (mm/s)



重要勘误  历史 Direct Gate 曾将“Y 速度律”误作“物理 dY”用于阶段阈值；应区分位移（mm）与速度（mm/s）。后来 V2 像素分轴策略“External du→X，Left du→Y，Left dv→Z”是本轮实用近似，不等价于完整耦合标定，必须用实际运动数据检查符号和交叉耦合。



已记录相机约定：RGB→BGR、不旋转镜像；External 图像 +U 主要对应 Base+X、+V 主要对应 Base−Z；Right Wrist +U 主要对应 Base−Y、+V 主要对应 Base−Z。Left 采用该相机自己的 2D Jacobian；顶部下插的 Base−Z 方向须通过有保护的低速试验再确认。

四、夹取阶段：改动、回退与验证

4.1 夹取链路操作

启用三路相机 Visual，捕获/锁定线 FREE 与镊口；原抓取阶段继续使用 External + Right Wrist。

原 Fusion 依据两路 guidance 做速度融合，处理 COMPOSITE、Z 冲突、靠近抓取点等。

原 Gate 读取 Fusion 的 /tmp/r1a7_wire_grasp_fusion_shadow_v1.json，依赖 mode/有效性/新鲜度计算 ready 与输出。

Controller 消费 /tmp/r1a7_dual_ibvs_velocity.json，并管理 OPEN_TRACKING、WAIT_CONTACT、CLOSE_SETTLE、CLOSED_MONITOR 等夹爪过程。

夹爪闭合后，视觉进入 WAIT_HOLE_TARGET，由用户在 External 与 Left 选孔/确认，才进入插孔阶段。

4.2 Direct Gate 尝试为何被放弃

曾尝试 wire_grasp_control_gate_direct_insertion_v1.py，把抓取和插孔控制都直接放入新 Gate。之后出现抓取 close 不触发、RAW Fusion 字段兼容错误、GRASP_DUAL_NOT_READY/输出归零。V2/V3 分别补偿 DUAL mode/ready 兼容，但还存在无法保证一致的问题。用户明确要求“不是 Direct Gate，是最开始 GATE”，因此 Direct Gate 路线已停止。

16:48 真实动作日志并不能说明 Gate 全程禁用：Controller 曾收到 enabled=1 并运动；夹爪仍在 OPEN_TRACKING，WRIST converging=0，因此闭合请求未触发。另有 HOME 轨迹在抓取控制期间出现，需要查看实际 Controller 状态/触发源，而不是推断夹爪执行器坏。

4.3 原始 Gate/Fusion 回退确认

# 先在安全状态下检查文件名，不依赖跨终端 shell 变量
FUSION=$WIRE/wire_grasp_fusion_shadow_v1.py
GATE=$WIRE/wire_grasp_control_gate_v1.py

grep -nE 'TOPDOWN_SUPERVISOR_RAW_OUTPUT_V1|fusion_raw_v1.json|fusion_shadow_v1.json' "$FUSION"
PYTHONNOUSERSITE=1 "$PY" -m py_compile "$FUSION" "$GATE"
# 原 Fusion 回退验证时仅发现:
# 19: "/tmp/r1a7_wire_grasp_fusion_shadow_v1.json"



当次用户实际用完整绝对路径启动 wire_grasp_control_gate_v1.py --enable-output，反馈“可以控制手臂以及夹爪运动”；结合 Fusion grep 没有 RAW 标记，说明**原始抓取链在那次工况下恢复功能**。这不是“未来所有插孔动作都已验证”的证据。

终端变量事故  曾因换终端后 $PY / $FUSION / $GATE 未设置，Python 报错 “can't find __main__ module in .../r1a7_threading_project/”。原因是变量为空，不是脚本损坏。此后操作推荐使用绝对路径或每个终端先重新定义环境。



4.4 夹爪代码不应被插孔补丁侵入

Controller 历史闭合目标 GRIPPER_CLOSE_Q = [-0.08, -0.0500]；遇到 OPEN_TRACKING open=1 close=0，先检查收敛触发、门控状态和 HOME/close_settle，而不是先调力矩。抓取闭合受外部/腕部共同收敛条件影响，左相机插孔 FREE 跟踪不应修改这套触发。

五、插孔阶段的操作流程与设计演化

5.1 目标选择与视觉感知阶段

完成抓取，收到夹爪 close settle 等反馈，Visual 从 GRASP 进入 WAIT_HOLE_TARGET。

Hole U-Net 在 External 与 Left 识别孔候选；用户在两个画面各选择同一个物理孔，右键或确认键进入 INSERTION_TRACKING。

插孔开始后，固定相机的选定 hole_uv 冻结；不再让 Hole U-Net/孔 tracker 不断更新目标身份，仅持续刷新细线 FREE 的测量。

r1a7_insertion_target_state_v1.json 主要承担 phase/target_ready/target_confirmed；r1a7_insertion_perception_v1.json 承担 cameras.external/left_wrist 中的 free_uv/hole_uv 等观测。

Shaper 依据真实控制观测计算 External du、Left du/dv，先 XY 逼近；满足 XY 条件后逐步开放 Z。

在下降期间逐帧修正 XY；FREE 失效、输入过期、相机绑定错误时需失效保护，而非基于旧观测盲动。

5.2 Shaper V1 → V7 演进表

版本 / 阶段

主要动作与缘由

结果 / 状态

V1：XY 优先原型

GRASP_PASS 透传；APPROACH_XY → APPROACH_3D → FINAL_INSERT → INSERTION_DONE；依据 XY score 给 Z 动态权重。

◐ 形成原型，但 JSON 输入误绑与终止状态问题。

V1 输入修复

从 target_state 误读相机点：ext=None/left=None；改为从 insertion_perception 读 cameras。

✓ 编译与日志阶段绑定恢复，之后出现真实相机路径。

V2：像素横向优先

历史 left_dY 受大 dv 影响，导致 Y 持续横移、Z 永远 0；改用 External du 和 Left du 分别做横向判据。

✓ 作为本轮控制策略使用；仍需标定耦合验证。

V3：FREE 固定视角锁

两路 FREE 初始连续 4 帧；>14px 跳变触发 relock/STOP。

✓ 19:21 编译通过；日志出现 FIXED_FREE_RELOCK_PENDING。

V4：视觉 Z 完成

曾经 Z 下降 1mm 就 INSERTION_DONE；改为 Left dv 判定；15mm 作为行程保护。

✓ 补丁编译过；仍反复出现里程保护提前归零。

V5：近孔防跳与控制锁（提议）

建议视觉 8px、控制 6px、禁止自动换身份；仅根据对话无法确认最终本地采用情况。

○ 以最终源代码 grep/sha 为准。

V6：连续 XYZ

移除 MAX_FINAL_DESCENT_MM / descent_mm；FINAL_INSERT 按 left_dv 连续控制，Z 死区只归零 Z，XY 仍闭环。

✓ 终端显示关键标记并编译通过；无全场景成功实测。

V7：去终止 DONE

状态机入口将旧 INSERTION_DONE 重定向 FINAL_INSERT，避免旧状态永久锁死。

✓ legacy transitions replaced=0；grep 974/987/1034/1149 及 compile=0。



5.3 关键像素控制律（V6 目标逻辑）

error convention: e = hole_uv - free_uv
ex = external_hole_u - external_free_u
ly = left_hole_u - left_free_u
lz = left_hole_v - left_free_v

vx = clip(Kx * ex, XY_limit)           # X deadband
vy = clip(Ky * ly, XY_limit)           # Y deadband
score = max(|ex|/X_coarse, |ly|/Y_coarse)
z_scale = f(score)                     # XY 越准确，Z 越可放权

APPROACH_XY:   vz = 0; vx/vy 持续闭环
APPROACH_3D:   vz = -APPROACH_Z_MAX * z_scale
FINAL_INSERT:  vz = -FINAL_Z_MAX * sat(|lz|/80) * z_scale
               |lz| <= 8 px 时 vz=0；XY 仍可纠偏
               XY 偏差超返回阈值时，只 hold Z；修正 XY



历史阈值有不同版本：V2 曾使用 X coarse≈30 px、Left U coarse≈15 px，fine≈10/6 px，return≈45/25 px，稳定约 5 cycle；V6 使用 FINAL_Z_DEADBAND_PX=8、FINAL_Z_FULL_SPEED_PX=80、FINAL_Z_UP_MAX=0.02。以上属于当时试验参数，不等于已完成安全/物理有效性验证。

六、18:42 全链路数据复盘（有原始 CSV 支撑）

原始数据：r1a7_cartesian_trace_20261117_184206.csv 与 r1a7_official_real_lowcmd_full_chain.csv。本报告使用两者共同的 sample_monotonic 与阶段信息进行分析；下表数字已重新根据上传文件计算，区别于仅依据屏幕观察的推断。

指标

18:42 实测值

解释

full-chain / trace 总行数

3,428 / 3,428

采集跨度约 114.25 s。

阶段分布

GRASP 699；WAIT_HOLE 275；INSERTION 2,454

插孔约 81.78 s。

Shaper 插孔状态

APPROACH_XY 2,452；其余 2 行字段缺失

没有进入 APPROACH_3D / FINAL_INSERT。

Gate Z 非零样本

0 / 2,454

这次整段插孔没有 Z 速度。

Gate XY 速度

X 中位 −0.6；Y 中位 +0.6 mm/s

长时间饱和横移。

Command XYZ 位移

(−57.65, +57.73, 0.00) mm

命令随 XY 大幅变化。

Actual XYZ 位移

(−32.70, +40.61, −0.12) mm

实际 Z 几乎未变。

Cartesian lag

p50≈33.51 mm，区间 30.28~39.21 mm

命令与实际位置差显著。

Left dY（旧判断量）

−46.90 ~ −9.85 mm

未满足旧 1.5mm 阈值，导致 XY-only 锁定。

Left du 接近情况

170 行满足 |du|≤10px

说明图像横向接近不代表旧 dY 判据通过。





图 1  按插孔阶段采样的 Gate X/Y/Z 速度：Z 全程为零。



图 2  实际位置相对于插孔开始时刻的位移。

原数据字段对照  full-chain 里有 gate_consumed_json（Gate 实际消费对象）和 fusion_json（含 Shaper 插孔诊断字段），不能误以为每个嵌套 JSON 都保留全部 Shaper 字段。推荐以 fusion_json.insertion_xy_priority 取得 ext_du / left_du / left_dv / score / z_scale。



6.1 典型“横向已近但仍推 Y”机制

例：External du≈−24.79 px；Left du≈−9.81 px；Left dv≈+211.22 px
旧 left_dy_mm = 0.176705*du − 0.141834*dv ≈ −31.69 mm
旧判据 |left_dy_mm| <= 1.5 mm 始终不满足
旧 Y 速度项 0.009833*du + 0.007806*dv 仍显著非零
结果：XY 长时间运动；score>=1，z_scale=0，Z 不启动。



V2 将 Left Y 对齐改为主要关注 Left du，减少 Left dv 在“允许开始 Z”判据中造成的死锁；但这种像素解耦的有效性仍需要做 X/Y/Z 小幅单轴激励进行物理检查。

七、故障记录与处理路径

现象 / 可见日志

原因定位

正确排查或处理

$FUSION / $GATE 执行时报找不到 __main__

跨终端变量未定义，Python 实际收到目录路径。

用绝对路径或重新 export 变量；ls -l、py_compile 先确认。

Gate STOP age=inf，EXT=NO WRIST=NO

Fusion 改写至 preinsert，但 Shadow Shaper 只写 preview，原 Gate 的 shadow 文件无人更新。

严格核对 writer→reader 拓扑；不要两 writer 同写。

INSERTION_CAMERA_BINDING_INVALID ext=None left=None

Shaper 从 target_state 而非 perception_state 找 free_uv/hole_uv。

分离状态 JSON 与感知 JSON；精确键 root.cameras.external/left_wrist。

近孔仍横移，Z 始终 0

旧 score 含 Left dY 与大幅 dv，coarse_ok 永远 false。

查看 full-chain 的 insertion_xy_priority；调整判据并用单轴标定检验。

INSERTION_DONE；leftV 仍 +165px

初期 FINAL_INSERT 按累计 1mm 判完成，错误停止。

使用 V4 后观察；再演化到 V6 连续 Z 控制。

FINAL_INSERT；leftV>0 但 cmd 全零

V4 历史 15mm guard 或 actual stale 可能触发；state 没标 STOP，外观混淆。

先查 insertion_shaper_reason，不应只看 state；禁止直接放大安全界限。

FIXED_FREE_RELOCK_PENDING，LEFT jump 39~54px

固定相机 FREE 候选跳变/跟踪错误，或固定小阈值误拒真实图像位移。

标注真实 FREE；区分 raw/display/control；验证预测残差与观测时延。

V6 仍偶见 INSERTION_DONE

旧状态/版本进程未重启或残留分支。

V7 编译后重启；grep 源码、看进程 cmdline 和新日志。

Left FREE 贴到镊子/折点而非物理末端

端点身份初始化/再检测错误，滤波只能稳定错误点。

优先引入抓持端 anchor + skeleton geodesic endpoint，再加时序跟踪。



八、当前使用的程序、关键代码与备份

8.1 代码状态分类（截止最后终端输出）

文件/组件

最后可见标记/功能

确认程度

VIS: live_grasp_to_insertion_visual_v3.py

FIXED_CAMERA_FREE_IDENTITY_V2；External/Left 14px raw 跳变门；grasp/孔选择保留。

✓ 用户打补丁，visual_compile=0。

SHAPER: wire_insertion_xy_priority_shaper_v1.py

FIXED_FREE_CONTROL_LOCK_V3；V6 连续 XYZ；V7 兼容重定向。

✓ 终端分别显示 compile=0 和 marker/关键行。

Shaper V5 近孔 lock-once

之前曾给出 8px/6px 与“锁一次”候选补丁。

○ 无最后终端 grep 确认，不能断言已运行。

原 Gate: wire_grasp_control_gate_v1.py

GRASP/DUAL mode 有效性门控；输入保持 /tmp/...fusion_shadow_v1.json。

✓ 曾实测抓取动作；本轮补丁未要求改变源码。

Fusion: wire_grasp_fusion_shadow_v1.py

历史原始 shadow 输出曾恢复；主动 Shaper 拓扑切为 preinsert 输出。

◐ 本地最终当前路径未重新上传核验。

Controller: ...auto_grasp_direct_v1.py

读取 Gate velocity；夹爪自动状态机；actual XYZ 发布；trace/full-chain。

✓ 历史实际日志与 CSV 支持，最终源码快照待归档。



无法在本次对话里直接读取机器人本地所有 `.py` 文件的现行完整内容，因此这里整理的是“已执行/已见关键代码、关键参数、路径与版本”。真正逐字节的当前源代码应从机器人工作目录打包并记录 SHA256，避免用聊天里的补丁片段替代唯一的本地事实。

8.2 终端中出现的确切备份名

文件

备份名称

用途

VIS

live_grasp_to_insertion_visual_v3.py.before_fixed_free_identity_20261117_192048

加入固定相机 14px 门限之前；用户终端明确显示。

SHAPER

wire_insertion_xy_priority_shaper_v1.py.before_fixed_free_control_lock_20261117_192107

加入 V3 FREE 身份锁之前；用户终端明确显示。

Fusion

wire_grasp_fusion_shadow_v1.py.before_topdown_raw_20261117_135406

回退 TOPDOWN_RAW 修改时使用的原始 Fusion 备份。

Direct Gate

wire_grasp_control_gate_direct_insertion_v1.py.before_grasp_compat_v2_20261117_164215

失败的 Direct Gate 分支前一版本；保留档案不运行。

Direct Gate

wire_grasp_control_gate_direct_insertion_v1.py.before_fusion_ready_fix_v3_20261117_164517

Direct Gate Fusion ready 修复前。

Controller

...auto_grasp_direct_v1.py.before_actual_state_publish_20261117_140614

增加实际 XYZ 发布前。



另有通配符形式的备份名（运行脚本按时间生成，当前绝对文件名未在对话中全部展开）：SHAPER.before_visual_z_fix_*、before_remove_insertion_done_*、before_continuous_visual_xyz_*、VIS.before_fixed_free_near_hole_v5_*、Fusion.before_active_xy_*、Fusion.before_shadow_fix_* 等。请使用第九章的 find 命令确认本地是否真实存在。

8.3 当前关键代码标记检查

# 仅检查，不改动文件：
PYTHONNOUSERSITE=1 "$PY" -m py_compile "$VIS" "$FUSION" "$SHAPER" "$GATE"
grep -nE 'FIXED_CAMERA_FREE_IDENTITY_V2|FREE_TRACK_MAX_JUMP_PX_BY_CAMERA' "$VIS"
grep -nE 'FIXED_FREE_CONTROL_LOCK_V3|FIXED_FREE_LOCK_ONCE_V5|CONTINUOUS_VISUAL_XYZ_CONTROL_V6|NO_TERMINAL_INSERTION_DONE_V7|FINAL_Z_DEADBAND_PX|MAX_FINAL_DESCENT_MM|INSERTION_DONE' "$SHAPER"
grep -nE 'fusion_preinsert_v1.json|fusion_shadow_v1.json' "$FUSION"
sha256sum "$VIS" "$FUSION" "$SHAPER" "$GATE" "$CTRL"



最后一次用户显示 V7 的 grep 行号：974 注释标记、987 legacy redirect→FINAL_INSERT、1034 FINAL_INSERT 转移、1149 遗留 elif state == INSERTION_DONE。V6 的 grep 显示 127/141/145/149/1012/1025/1077/1079/1087/1098/1119/1123 等行，且旧 MAX_FINAL_DESCENT_MM/descent_mm 未再出现。

九、标准终端 SOP：核验、归档、调试、恢复

9.1 操作前的安全停止与环境核验

所有源代码修改、writer 切换、首次接入新视觉 authority 前，先在 Controller 的交互终端正常停止真机低层控制，确认设备处于可接受的安全姿态、工作空间无障碍、具备急停方式。不得仅依赖 pkill 作为机器人机械制动。

cd /home/robot/unitree_sim_isaaclab_threading/r1a7_threading_project
ROOT="$PWD"; PY=/home/robot/miniconda3/envs/tv_sam2/bin/python
WIRE="$ROOT/vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire"
VIS="$ROOT/vision_dual_camera/task6_threading/dataset_tools/wire_unet/live_grasp_to_insertion_visual_v3.py"
FUSION="$WIRE/wire_grasp_fusion_shadow_v1.py"
SHAPER="$WIRE/wire_insertion_xy_priority_shaper_v1.py"
GATE="$WIRE/wire_grasp_control_gate_v1.py"
CTRL="$ROOT/transfer_control/experimental/r1a7_cartesian_stream_control_kd30_posvel_wu_dqik_u010_auto_grasp_direct_v1.py"

pgrep -af 'wire_grasp_fusion_shadow_v1|wire_insertion_xy_priority_shaper_v1|wire_grasp_control_gate_v1|r1a7_cartesian_stream' || true
ls -lh "$VIS" "$FUSION" "$SHAPER" "$GATE" "$CTRL"
PYTHONNOUSERSITE=1 "$PY" -m py_compile "$VIS" "$FUSION" "$SHAPER" "$GATE" "$CTRL"



9.2 将全部现行 Python 及备份归档（只读）

STAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE="$ROOT/experiment_snapshots/xy_insertion_$STAMP"
mkdir -p "$ARCHIVE/sources" "$ARCHIVE/metadata"
for F in "$VIS" "$FUSION" "$SHAPER" "$GATE" "$CTRL"; do
    test -f "$F" || { echo "MISSING $F"; continue; }
    cp -a "$F" "$ARCHIVE/sources/"
done
find "$ROOT" -type f   \( -name '*.py.before_*' -o -name '*.py.current_raw_before_*' \)   -print > "$ARCHIVE/metadata/backup_paths.txt"
sha256sum "$ARCHIVE"/sources/*.py > "$ARCHIVE/metadata/source_sha256.txt"
git status --short > "$ARCHIVE/metadata/git_status.txt" 2>&1 || true
printf '%s\n' "$ARCHIVE"



如需把历史备份文件本体也复制，建议另开归档任务按 backup_paths.txt 逐个复制，避免不加选择地将全部大日志、模型权重和数据集打包成巨型存档。GitHub 上的受控版本不能代替本地未提交的 V6/V7 源码。

9.3 检查 JSON 数据流与相机 FREE 字段（不写控制）

python3 - <<'PY'
import json,os,time
paths = [
 '/tmp/r1a7_insertion_target_state_v1.json',
 '/tmp/r1a7_insertion_perception_v1.json',
 '/tmp/r1a7_wire_grasp_fusion_preinsert_v1.json',
 '/tmp/r1a7_wire_grasp_fusion_shadow_v1.json',
 '/tmp/r1a7_dual_ibvs_velocity.json',
 '/tmp/r1a7_cartesian_actual_state_v1.json',
]
for p in paths:
    print('\n',p)
    if not os.path.isfile(p):print('  MISSING');continue
    try:
        d=json.load(open(p));print('  age=',round(time.time()-float(d.get('timestamp',0)),3))
        print('  phase=',d.get('phase'),'mode=',d.get('fusion_mode'))
        print('  state=',d.get('insertion_shaper_state'),'reason=',d.get('insertion_shaper_reason'))
        print('  proposed=',[d.get('proposed_'+a+'_mm_s') for a in 'xyz'])
        cams=d.get('cameras') or {}
        for cam in ['external','left_wrist']:
            x=cams.get(cam) or {}
            print(' ',cam,'free=',x.get('free_uv'),'hole=',x.get('hole_uv'))
    except Exception as e:print('  ERROR',repr(e))
PY



请注意：视觉进程退出时 target_state 会变成 phase=EXIT、ready=false、confirmed=false，不能用 EXIT JSON 诊断 INSERTION_TRACKING 时的真实字段。用于查错的日志必须在感知进程仍运行时采集。

9.4 新增模块的验证顺序

完成第 9.1/9.2 源码归档，并确认本轮仅修改 VIS/SHAPER 所需段落。

用已记录的原视频或停机采集素材复放，确保 FREE 物理身份与跟踪点正确；不可只以 FREE-LIVE 十字“看起来不跳”当作通过。

Fusion/视觉启动但新 Shaper 先仅诊断输出（不改 Gate-facing writer）；检查镜头、free_uv、hole_uv、timestamp、innovation 和连续稳定性。

核查端点与机器人实际运动方向一致，分别验证 X/Y/Z 小命令对应的图像变化符号，以及 LEFT 的耦合标定残差。

校验输入过期/失锁/STOP 对真实 Gate 的效果、无控制输出异常，安全负责人批准后方可进行受控短距离闭环试验。

每次试验退出后立即存 trace、full-chain、视觉观测与程序 SHA；记录版本、时间、目标孔、失败原因。

不要直接照抄历史使能命令启动真机  对话中曾用 wire_insertion_xy_priority_shaper_v1.py --enable-output 与原 Gate --enable-output 进行真实测试；这属于历史执行记录，不代表当前版本已经满足安全复测条件。新的 FREE 物理端点识别仍有明确误判。



十、FREE 端点识别：当前根因与升级方案

10.1 为什么固定相机仍会漂移

相机静止并不意味着图像里的 FREE 静止：机械臂及柔性细线仍运动，近孔区域又有背景边缘、反光、细线与镊子重叠、局部遮挡。U-Net mask → skeleton → endpoint 若产生多个相似候选，FREE 可能被错误指向靠镊子/折点的候选；几像素跳变门和 EMA 无法恢复“物理哪一端”这个身份。

从最近提供的左相机截图：用户标记的黑点与绿色 FREE-LIVE 存在明显空间偏差。External 相对好，代码的主 tracker 类并非完全不同；视角、遮挡与初始端点候选歧义更可能是关键。

10.2 优先级最高：抓持端锚点 + skeleton 拓扑

使用现有 Wire U-Net 分割得到细线 mask；小连通域清理并抽取 skeleton（细线骨架）。

将检测到的镊尖/夹持位置投影到所属 skeleton 上，作为 grasp anchor。

沿 skeleton 连通图计算最短路径（geodesic）长度，候选 terminal degree≈1 点必须在同一细线连通体上。

选离 grasp anchor 拓扑路径最远且满足物理方向约束的 endpoint 作为 FREE；不接受单纯“离上一点近”的中段候选。

保留“目标不明/细线与工具 mask 粘连/骨架断裂”时的 invalid 状态；不得把假 FREE 填成 valid。

在两路固定视角分别保持自己的 FREE 物理身份，再用机械臂 actual ΔXYZ 与像素运动做跨周期一致性检查。

10.3 次级升级：TAPNext++ + 运动预测（候选实验）

当现有 U-Net + skeleton 可以准确初始化真正 FREE 后，可引入在线点跟踪器（如 TAPNext++ / Track-On-R）及局部支持点，在固定相机视频中持续跟踪已确认物理点。配合图像运动常速度滤波、机器人 actual ΔXYZ→Δu/Δv 预测、skeleton endpoint 几何门控。

FREE candidate (U-Net+skeleton) ── identity from grasp anchor ─┐
                                                             │
Video online tracker (FREE + support points) ─────────────────┤
                                                             ▼
Actual robot ΔXYZ → predicted Δuv ── innovation/residual gate
                                                             │
                              confidence + topology + freshness
                                                             ▼
                              FREE_control / FREE_valid
                                                             ▼
                              XY/Z Shaper (仅消费可信FREE)



TAPNext++、Track-On-R、多视角 TAP 等属于会话中讨论的研究候选；**本轮没有证据表明已下载模型、运行 A/B、配置权重或在机器人上正式使用**。建议先离线精度对照，再测时延/显存与物理安全接入。

10.4 必须同时采集的诊断字段

字段

用途

raw_free_uv / display_free_uv / control_free_uv

区别检测候选、可视化十字和真正驱动控制的点；避免把显示抖动误判为执行抖动。

free_source / measurement_timestamp / frame_seq

判断是测量、预测还是 hold；避免使用过期 point。

skeleton_distance_px / endpoint_degree / path_distance

判断 FREE 是否是沿细线物理末端。

prediction_uv / innovation_px / confidence

用运动数据预测位置，按残差拒绝错误候选，而非固定 6px 死门限。

ext_du / left_du / left_dv / xy_score / z_scale

在相机误差与允许控制速度间建立可解释对应。

Gate consumed vx/vy/vz、actual/command XYZ

验证 Controller 收到的真正速度及物理反应。



十一、下一轮研发与验收清单

工作包

建议实施顺序

可检验的通过标准

WP1：冻结当前事实

本地源代码快照、SHA、备份/运行时 topology，独立于聊天补丁。

编译全通过；唯一 writer；版本日志可复现。

WP2：标注 FREE Ground Truth

从 External/Left 视频取接近孔前后关键帧；记录黑点真实 FREE、镊尖与孔中心。

两路端点身份误检率/跳变率和误差分布可计算。

WP3：拓扑端点身份

grasp-anchor + skeleton geodesic；先离线替换候选，不改 Gate。

错误近镊端不会获得控制权限；断裂/遮挡能标为 invalid。

WP4：运动残差验证

实际 XYZ / 图像 Jacobian 与视觉 tracker 交叉核对。

符合真实 ΔXYZ 的观测被保留；明显离群点不传入速度控制。

WP5：XY/Z 验证

校准 X/Y/Z 引起的两相机运动方向；Z 连续控制，含退出阈值/滞回。

Z 在 XY 达阈值后开始；下降时 XY 仍修正；误差恶化仅暂停相应轴。

WP6：物理插孔验收

完成全部安全检查后，受控短程多次试验。

记录“达到孔位/真正进入孔/是否接触/是否过冲”，不能只以 2D 像素相近认定完成。



风险登记  取消 Shaper 的累计 Z 里程限制并不等于消除物理碰撞风险。必须保持 Controller 的速度、加速度、关节极限、工作空间及接触相关保护，并在近孔区域用外部碰撞/接触证据判定实际插入；单独的 leftV≈0 不能证明细线已穿孔。



附录 A：关键文件、日志和命名

/tmp/r1a7_insertion_target_state_v1.json
/tmp/r1a7_insertion_perception_v1.json
/tmp/r1a7_wire_grasp_fusion_preinsert_v1.json
/tmp/r1a7_wire_grasp_fusion_shadow_v1.json
/tmp/r1a7_wire_grasp_fusion_xy_priority_preview_v1.json
/tmp/r1a7_dual_ibvs_velocity.json
/tmp/r1a7_cartesian_actual_state_v1.json
/tmp/r1a7_official_real_lowcmd_full_chain.csv
/tmp/r1a7_cartesian_trace_20261117_184206.csv



18:42 CSV 分别有 96 列（full-chain）与 18 列（Cartesian trace）。trace 包括 IK/Jacobian/关节/actual_xyz_mm/command_xyz_mm 等；full-chain 才有视觉、Fusion、Gate 实际消费、Controller、夹爪与插孔阶段状态。

附录 B：历史 Console 提示的判读

GRASP_PASS            抓取侧透传（应保持原 Fusion 语义）
INSERTION_CAMERA_BINDING_INVALID
                      插孔相机 FREE/HOLE JSON 绑定失败
FIXED_FREE_RELOCK_PENDING
                      至少一路 FREE 身份未稳定；属于观测失效保护
APPROACH_XY           XY优先，Z暂不放行
APPROACH_3D           XY接近，Z动态放权
FINAL_INSERT          V6: 连续XY修正 + Z按Left dv与XY质量变化
INSERTION_DONE        早期终止状态；V7入口重定向仅是兼容保护
Z_PIXEL_DEADBAND      Z像素误差处于死区；按V6只应 hold Z
XY_REALIGN_Z_HOLD     横向误差偏大，先hold Z继续XY
ACTUAL_STATE_STALE / INPUT_STALE
                      状态过期，不可依旧观测驱动机器人



附录 C：版本核查后的记录模板

测试时间:
Git commit / branch:
VIS SHA256:
FUSION SHA256:
SHAPER SHA256:
GATE SHA256:
CTRL SHA256:
数据流 writer/reader: Fusion -> preinsert -> Shaper -> shadow -> Gate
三相机有效: External / Right / Left
phase 切换: GRASP -> WAIT_HOLE_TARGET -> INSERTION_TRACKING
实际端点 FREE（两相机）:
raw/display/control FREE 是否一致:
状态/原因/state/reason:
Gate consumed (vx,vy,vz):
机器人 actual ΔXYZ / command ΔXYZ:
有无异常触碰/误动作:
trace / full-chain / video / backup 路径:
结论与下一轮唯一变量:



上述模板应与正式实验数据一同保存。一次实验只变更一个关键控制量或一处 FREE identity 策略，便于将真实改善与版本差异对应起来。
