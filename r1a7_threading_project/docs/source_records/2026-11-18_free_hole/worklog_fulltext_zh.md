# 20261118 来源全文转录

原件字节不变；按OOXML段落顺序展开表格，保留段内换行，仅清理行末空白。

R1-A7  /  工具操作研究  /  技术归档

细线端点稳定识别、孔位重定位
与插孔阶段调试工作汇总

技术路线 · 具体实验与终端操作 · 源码修改记录 · 当前程序基线 · 下一轮验收

记录属性

内容

截止时间

2026 年 11 月 18 日（以已提供的终端/代码记录为准）

实验对象

宇树 R1-A7 双臂机器人；镊子夹持细线并引导进入孔位

控制阶段

GRASP → WAIT_HOLE_TARGET → INSERTION_TRACKING

主要内容

External / Left Wrist 视觉闭环，FREE 下半支身份保护，孔位确认/重定位，Shaper-Gate-Controller 调试

代码状态约定

已核对 = 本地终端输出/源码标记可证；已验证 = 存在编译或动作证据；待验证 = 只有方案/补丁指令，未见成功执行回执



核心结论

截至最后一条可核对终端输出：现行 Visual 已含下半支 FREE 身份约束，但仍发生 FREE-HOLD；Left Wrist 的孔在 INSERTION_TRACKING 阶段仍使用冻结目标。计划的“短时 tip 丢失容忍 + Left Hole 实时重识别”已给出 V3 补丁命令，但没有成功落盘与运行回执，不能写成已完成。



版本原则：保留原有抓取和夹爪闭合链路；插孔改动只在感知及分层控制中实施。现行速度参数已从 10/5 mm/s 试验上限恢复到安全备份基线。



1  研究目标与当前工作边界

任务是在镊子已夹住细线之后，用图像中的自由端 FREE 与目标孔 HOLE 构造视觉误差，使机械臂先完成平面内对齐，再在持续 XY 修正的条件下推进 Z 方向插入。当前工作重点不是改变夹爪闭合动作，而是使端点身份稳定、孔目标可追踪、控制速度可解释且可回滚。

工作线

已落实/当前状态

尚需解决

FREE 端点稳定

Skeleton 末端候选 + tip→FREE 物理下半支方向；反向半支拒绝；EMA / reacquire

出现 FREE-HOLD、端点轻微偏移；当前帧真实测量可能被拒绝

HOLE 孔位识别

WAIT_HOLE 阶段 Hole U-Net + 已选孔跟踪 + EMA；确认后冻结

左腕视角轻微变化造成旧像素坐标失配；INSERTION 需要受约束重新定位

插孔速度闭环

External du→X，Left du→Y，Left dv→Z；Shaper 分阶段限幅、Gate 兜底

低速/高速实验后回归原限幅；需确认控制实际速度及最小速度策略

真机验证

实际 Controller 停止与记录保存；原抓取 Gate 曾有夹爪/手臂动作反馈

尚未形成可靠穿孔成功证据；不应直接以视觉对齐宣布插入成功



2  插孔阶段完整技术路线

2.1 多相机分工与阶段状态

阶段

External

Right Wrist

Left Wrist

GRASP

抓取视觉主观测/误差

抓取侧观测、镊口辅助

实时采集与辅助记录

WAIT_HOLE_TARGET

目标孔候选与选孔确认

监控/数据

目标孔候选与选孔确认

INSERTION

FREE→HOLE 的 u 误差控制 X

仅监控/记录，不进入插孔控制

FREE→HOLE 的 u、v 误差用于 Y、Z



插孔控制相机集合固定为 INSERTION_CONTROL_CAMERAS=("external","left_wrist")。Right Wrist 在插孔阶段保留实时画面，但不进入控制权重。相机图像数据沿用 RGB→BGR、无左右翻转/旋转的既有约定。

2.2 当前实际数据流与唯一写入者

三相机画面 / Wire U-Net / Hole U-Net / FREE tracker
           |
           v
 live_grasp_to_insertion_visual_v3.py  (Visual)
           |
           v
 wire_grasp_fusion_shadow_v1.py  (Fusion)
 /tmp/r1a7_wire_grasp_fusion_preinsert_v1.json
           |
           v
 wire_insertion_xy_priority_shaper_v1.py  (Shaper)
 /tmp/r1a7_wire_grasp_fusion_shadow_v1.json
           |
           v
 wire_grasp_control_gate_v1.py  (Gate)
 /tmp/r1a7_dual_ibvs_velocity.json
           |
           v
 r1a7_cartesian_stream_control_*_auto_grasp_direct_v1.py
           |
           v
 R1-A7 lowcmd / 夹爪状态机 / full-chain 与 trace 记录



历史上还存在 Visual→Fusion→Gate→Controller 的纯抓取拓扑。当前带插孔 Shaper 的结构必须确保 Fusion 写 preinsert、Shaper 独占写 shadow、Gate 只读 shadow；不可让 Fusion 和 Shaper 同时覆盖 Gate 输入 JSON。Shaper 在 GRASP 阶段使用 GRASP_PASS 透传，以保持原抓取语义。

2.3 视觉误差到三轴速度的经验解耦

e_ext_u   = external.hole_u - external.free_u
 e_left_u  = left.hole_u     - left.free_u
 e_left_v  = left.hole_v     - left.free_v

 X <- e_ext_u       (External du)
 Y <- e_left_u      (Left Wrist du)
 Z <- e_left_v      (Left Wrist dv / depth)

 Kx = 0.020; Ky = 0.025; 横向死区均 4 px
 X/Y 先对齐；APPROACH_3D 随 score 放行 Z；FINAL_INSERT 持续 XY 修正。



这是一套实测优化后的“像素分轴”近似，并非完整相机雅可比逆解。Left 原标定保留为诊断：Δu≈39.7407ΔX+15.5967ΔY−31.7103ΔZ；Δv≈52.9663ΔX+12.3808ΔY−39.5067ΔZ。历史使用 left_dY≈0.176705du−0.141834dv 作为近孔判据，因 dv 项较大导致 Y 持续误推进，后来改由 Left du 直接判断横向对齐。

3  细线 FREE 端点稳定识别：当前算法与具体修改

3.1 基础图像处理与原有故障

Wire U-Net 输出细线分割掩膜，继而形成 wire_path / Skeleton；grasp.manual.ordered_path_from_mask(path) 给出路径两端点。早期根据与上一帧 filtered_uv 或 reference_uv 的欧氏距离选择“最近端点”，遇到反光、遮挡、细线弯曲、路径断裂或镊口附近上半支更靠近参考点时，会发生身份切换。绿色 FREE 标记可能锁到上半支，或进入 HOLD 后停在历史位置。

原算法已有中位数历史窗、EMA、小跳变常规更新、大跳变连续候选 reacquire、identity_margin 判别与 measurement_timestamp。单纯缩小/放宽跳变阈值不能根治物理端点身份错误。

3.2 已落地改动：INSERTION_LOWER_BRANCH_IDENTITY_V1

物理身份优先原则

镊子夹持位置将细线分为上半支（固定侧）和下半支（真正 FREE）。插孔阶段一旦用可信 tip→FREE 初始化 free_branch_dir，候选端点必须通过相对 tip 的同半支投影检查，反向上半支不得进入正常更新或 REACQUIRE。



源码对象 / 参数

当前设置或行为

目的

InsertionDirectFreeTracker.update(path_mask, reference_uv, now, tip_uv=None)

输入当前 wire_path，计算两个 Skeleton 端点；在物理分支合法候选中选择连续目标

把真实图像端点纳入插孔控制观测

free_branch_dir

可信初帧由 tip→FREE 初始化；之后仅随同半支被接受观测慢速更新

防止 FREE/FIXED 180° 翻转

MIN_PROJECTION_PX = 6.0

端点沿下半支方向的投影必须达到门限

拒绝上半支

MIN_LENGTH_PX = 12.0

初始化时 tip 与 FREE 要有足够距离

防止短矢量方向不可靠

DIR_ALPHA = 0.08；UPDATE_MIN_DOT = 0.25

仅可信同半支观测更新方向，并低权重融合

允许缓慢旋转但维持身份

BRANCH_TIP_LOST / OPPOSITE_BRANCH_REJECTED

tip 无效或候选不在下半支时调用 _temporary_loss()

有保护的失效输出，不误接管控制



3.3 现场复现：FREE-HOLD / 端点中断与轻微偏移



图 1  视觉画面出现 FREE-HOLD，标记保持旧位置



图 2  09:08 左腕画面：线端继续延伸，FREE 标记未随动



屏幕显示 FREE-HOLD 并不等于当前图像没有白色细线；它表示直接端点跟踪器没有接受当前帧为可信测量。_temporary_loss() 在短时窗口可返回 MEASURED_HOLD（旧 filtered_uv），之后可能只作 DISPLAY_ONLY。仅凭截图无法确定具体触发分支；需要结合 direct_state、direct_jump_px、direct_endpoints_uv 与 tip_uv 判断。

典型状态

含义

排查重点

NO_RAW_PATH / NO_ORDERED_PATH

掩膜为空，或路径排序不能得到有效端点

U-Net、Skeleton 是否断裂或被遮挡

NEED_BRANCH_TIP / BRANCH_TIP_LOST

初始化或后续分支保护所需 tip 无效

tip 检测连续性、是否可有限时缓存

OPPOSITE_BRANCH_REJECTED

两个候选均不满足下半支投影

tip 与分支方向是否漂移，真实 FREE 是否被错误排除

IDENTITY_AMBIGUOUS

两端点对参考位置的距离过于接近

原最近邻身份规则是否应在已锁定分支后退居次要

REACQUIRE

候选跳变超过正常门限，等待多帧一致

jump_px、连续候选聚类与帧率



3.4 已提出但未确认应用：FREE continuous tracking V3

V3 补丁指令拟新增 last_valid_tip_uv、last_valid_tip_time 与 INSERTION_FREE_LAST_TIP_HOLD_S=0.75；tip 短时丢失时用最近可信 tip 锚点继续做下半支投影判断，超过容忍时间仍保留 BRANCH_TIP_LOST 安全退出。另把 legacy identity_margin 判断限定于 free_branch_dir 尚未建立时，减少已锁分支被最近邻歧义规则反复冻结。

实施状态必须区分

终端已核对：V1 已存在于 Visual；V2 及先前补丁由于字符串锚点/固定缩进验证失败，均在写盘前退出，visual_compile=0 且没有新 marker。最后给出的 V3 修正版尚未收到 PATCHED、哈希和运行输出，不能认定已落地。V3 也尚未实现显式的 FREE 速度预测器；“短时 tip 复用”不应误写成完整运动预测。



4  目标孔 HOLE：原有识别、偏移原因与实时重定位路线

4.1 现行原始流程：选孔并冻结

WAIT_HOLE_TARGET
  External / Left Wrist: Hole U-Net 推理 + 孔候选提取
  用户点击物理孔目标；目标锁定/跟踪 + EMA
  External 与 Left Wrist 目标均确认后进入 INSERTION

INSERTION_TRACKING（截至已证基线）
  target_uv["external"]   = 已确认的静态孔坐标
  target_uv["left_wrist"] = 已确认的静态孔坐标
  Hole U-Net 不再运行；FREE 持续更新



直接代码证据：Hole inference 条件是 phase == PHASE_WAIT_HOLE；进入 INSERTION 后，原代码只依据 target_confirmed、target_selected、target_uv 是否存在保持 target_valid。新旧孔坐标无法在插孔阶段比较。

4.2 左腕孔位偏移：两类机理与现有证据边界

候选机理

画面表现

如何鉴别

相机/腕部轻微位姿变化

静止物理孔在 Left Wrist 图像中的中心连续移动，而被冻结的 marker 不动

额外记录图像特征/实时 hole_uv 与 anchor_uv；若缓慢连续则更支持视角几何变化

孔检测不稳或误选候选

检测中心突跳、多个候选相互切换，可能与反光/遮挡相伴

观察 raw center、置信度、形状/面积、目标距离与候选数

旧目标坐标被永久冻结

即便真实孔图像位置变化，目标十字仍保留确认时像素

核查 INSERTION 阶段 Hole U-Net 是否运行及 target_uv 是否更新



2026-11-18 09:02:37 的 Controller trace 只能证明控制及末端实际运动发生过，不含相机外参或逐帧 hole_raw_uv，因此不能仅据轨迹定量归因“相机机械晃动”与“识别漂移”。

4.3 计划：Left Wrist HOLE-LIVE、External HOLE-ANCHOR

Left Wrist:  hole_anchor_uv（初次确认）
         ↓
当前 Hole U-Net 候选 → 同一物理孔的局部匹配
         ↓
空间 ROI / jump gate / 多帧一致性检查（后续增强）
         ↓
EMA → hole_live_uv → target_uv["left_wrist"]
         ↓
实时左腕 FREE-HOLE 差值控制 Y / Z

External: 保留原选定孔像素作为固定参考
Right Wrist: 仅做监控



实际 V3 补丁内容仅拟把 Hole inference 的阶段条件扩展为 WAIT_HOLE 与 INSERTION，并在插孔时跳过非 left_wrist 相机，复用现有孔候选匹配及 EMA 更新 target_uv。独立 hole_anchor_uv、实时 drift_du/dv、严格的 ROI/跳变连续确认、失锁后 Z 停止等，是进一步工程化要求；在最后已见补丁中尚未完整实现，不能列为已落地功能。

5  插孔 Shaper/Gate 调试与速度版本回归

5.1 插孔状态机与 Z 放权

状态

主要逻辑

关键保护

APPROACH_XY

External du 控 X、Left du 控 Y；先不放 Z

观测可信、误差门限、分轴速度限制

APPROACH_3D

横向误差收敛后以 score 分级允许 Z

score 档位调节 z_scale；XYZ 持续观测

FINAL_INSERT

继续 XY 校正；Left dv 控 Z 的慢速下降/微抬

Left dv 死区只归零 Z；横向失配时暂停 Z

GRASP_PASS

抓取阶段透传原 Fusion、夹爪控制不由 Shaper 接管

避免侵入原 OPEN_TRACKING/WAIT_CONTACT



V6 移除了以累计 Z 位移强行结束的终止机制；V7 将遗留 INSERTION_DONE 分支转回 FINAL_INSERT。取消软件里程结束不等于可以无碰撞无限下降，必须继续依赖物理空间、接触/电流、速度及操作者停机保护。

5.2 已执行速度上限实验与回退

版本 / 时间

操作内容

结果 / 风险

原安全基线

Shaper 多状态轴向上限为 0.60/0.35/0.20/0.08、Z 0.10/0.05 mm/s；Gate 插孔总矢量上限 1.0，其他阶段 10.0

原参数已重新恢复

10 mm/s 探索

曾将 Shaper 六处速度上限上调到 10；Gate 插孔上限临时放到 10；Controller 进行真实动作测试

主观反馈末段速度过快；出现速度档位跳升；不应视为定速 10 mm/s

5 mm/s 探索

将 Shaper 六处改为 5；Gate 曾临时用常量 5.0

常量 Gate 误限制了非插孔抓取阶段；后续要求使用 conditional

02:04 备份恢复

从 pre_lower_branch_free_20261118_005624 恢复 Shaper 与 Gate 两个文件

双方 compile=0；哈希恢复历史基线；Visual 下半支补丁未被回退



当前安全基线参数

值（mm/s）

含义

APPROACH_XY_MAX / MID_XY_MAX

0.60 / 0.35

横向接近/中间阶段轴向限幅

FINE_XY_MAX / INSERT_XY_MAX

0.20 / 0.08

近孔与最终插入横向轴向限幅

APPROACH_Z_MAX / FINAL_Z_MAX

0.10 / 0.05

插孔接近/最终下降的 Z 限幅

Gate INSERTION / OTHER

1.0 / 10.0

速度向量范数上限；并非期望固定速度



速度由视觉误差经增益与状态机生成，限幅只是最大值。Controller 存在 IBVS_MIN_SPEED_MM_S = 1.0 的历史源码值，可能对小幅非零速度产生上调作用；是否在具体测试路径触发，应结合 Gate consumed JSON 与 Controller trace 联合核查，而非仅依据主观体感判断。

5.3 回退前后最重要的兼容性要求

恢复旧速度时只整体恢复 Shaper/Gate，两者的哈希与编译结果已有证据；Visual 不恢复，保留 lower-branch patch。抓取阶段夹爪 CLOSE_Q、OPEN_TRACKING/WAIT_CONTACT、夹爪力矩与 Controller 原程序在本轮插孔速度实验中未改；Controller 哈希与历史安全基线相同。

6  实验记录、输出证据与问题判定

6.1 2026-11-18 09:02:37 真机运行

终端反馈： [AUTO GRASP DIRECT] enabled=1 state=WAIT_CONTACT open=1 close=1 | EXT NA | WRIST NA；操作者发出 QUIT 后 Controller/lowcmd 正常停止，full-chain 保存 rows=11621 dropped=0 error=None，trace 保存 5400 个样本。这里的 open=1/close=1 是日志字段，不能单独证明夹爪已完成闭合，也不能证明视觉 FREE 已正确。

来源 / 指标

从 CSV 核对的结果

解释边界

09:02:37 trace

5,400 行；时长约 183.04 s；平均采样约 29.5 Hz

Controller 层采样，非视觉检测 FPS

cartesian_speed_mm_s

5,294 / 5,400 行记录为 1.0 mm/s；最大 1.0

仅表明 trace 中记录的控制速度参数/量级

actual_xyz_mm 首末点

净变化约 (−2.76, −80.94, −36.09) mm；净位移约 88.66 mm

不能区分腕部相机相对目标的像素漂移来源

cartesian_lag_mm

中位约 26.05 mm

该指标属于 Controller 几何/追踪诊断，不能直接当作插孔误差



6.2 2026-11-18 01:37 速度探索

另一份 01:37:09 trace 共 3,049 行，时间跨度约 104.69 s；cartesian_speed_mm_s 最大记录到 10 mm/s，P50 约 1.26 mm/s。操作者报告“插孔较快、末段突然加速”；该现象与较高上限、Z 缩放档位以及 Controller 下游速度行为均可能有关，但仅凭此 CSV 不能唯一确定是哪一层造成，应结合逐帧 Shaper/Gate 日志判定。

6.3 失效归因矩阵

现场现象

优先查看

不应直接采取的动作

FREE-HOLD 固定

direct_state；raw/endpoints/filtered；tip；branch direction；measurement age

直接放开所有 branch 门或无限延长 HOLD

FREE 跳到上半支

free_branch_dir、tip 位置、projection、Skeleton 端点拓扑

只靠“哪个端点更近”重新锁定

左腕孔标记漂移

target_uv 是否冻结；真实孔像素是否变化；孔候选 raw_uv

未经身份确认就全图切换孔目标

近孔时突然变快

z_scale；Shaper clip；Gate 速度范数；Controller min/max/实际 XYZ

把速度上限当作固定运行速度

夹爪不闭合或 WAIT_CONTACT

Controller close request/状态、Grasp Gate 兼容、Fusion 模式

修改插孔视觉代码前先误改夹爪 torque



7  当前使用到的源码、状态与可复现基线

下表是最后已上传终端 SHA256 的可证快照。GitHub 分支只能代表历史仓库状态；实际运行以本地文件和哈希为准。

程序 / 文件名

本轮用途与当前状态

SHA256（前 16 位）

live_grasp_to_insertion_visual_v3.py

三相机视觉；FREE 下半支 V1 已落地；HOLE 仍冻结；V3 待成功落盘

b9a12787fe2cc877

wire_grasp_fusion_shadow_v1.py

Fusion 上游汇总，插孔拓扑写 preinsert

dc21dd5599633b44

wire_insertion_xy_priority_shaper_v1.py

GRASP_PASS、XY 优先/Z 放权；六处原速上限已恢复

dfe8f2828673923f

wire_grasp_control_gate_v1.py

原始 Gate；插孔 1 / 其他 10 的条件速度上限

b31f3056a23775ed

r1a7_cartesian_stream_control_kd30_posvel_wu_dqik_u010_auto_grasp_direct_v1.py

Controller 关节/笛卡尔指令、夹爪状态、诊断 CSV；本轮未修改

0c22da1185880aa6



项目根目录：/home/robot/unitree_sim_isaaclab_threading/r1a7_threading_project；Python 环境：/home/robot/miniconda3/envs/tv_sam2/bin/python。Visual 位于 vision_dual_camera/task6_threading/dataset_tools/wire_unet/；Fusion/Shaper/Gate 位于 vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/；Controller 位于 transfer_control/experimental/。

7.1 对应完整 SHA256

VIS     b9a12787fe2cc877a419647b88971fc653fc59bacba8d6ef78e52354dd5ba911
FUSION  dc21dd5599633b445114e58f7959c81ff3dda8c38eaca0e84f65b54e0e6446f4
SHAPER  dfe8f2828673923f402494bd61e20193632bf189d22138ffcd8752961e00b5a3
GATE    b31f3056a23775ed4fb8ef0acd9b94b8ff87ad838b9982c26a78ca88cdaaae85
CTRL    0c22da1185880aa681e3147599c03bc5b9d408da42176e6418ebc0a34195fcce



7.2 已知备份与版本恢复点

备份 / 归档

操作时机

作用

/home/robot/r1a7_backups/pre_lower_branch_free_20261118_005624/

2026-11-18 00:56

包含 Visual/Tracker/Fusion/Shaper/Gate/Controller 备份及 tar，Shaper/Gate 速度回归来源

/home/robot/r1a7_backups/pre_free_hole_live_20261118_095120/

2026-11-18 09:51

当前 Visual b9a127... 版的完整副本，联合补丁前安全点

Visual.before_free_hole_live_v2_20261118_100436

2026-11-18 10:04

V2 再次尝试前的本地源文件副本

/tmp/r1a7_before_free_hole_live.sha256

09:51 前后终端记录

五个关键程序的原始校验清单



7.3 需要观测的 JSON 及字段

JSON 文件

关键观测内容

/tmp/r1a7_insertion_perception_v1.json

left_wrist: free_uv / free_source / direct_state / direct_raw_uv / direct_filtered_uv / tip_uv / hole_uv / hole_valid / free_branch_dir_uv

/tmp/r1a7_insertion_target_state_v1.json

phase / target_ready / target_confirmed / selected_control_cameras

/tmp/r1a7_wire_grasp_fusion_preinsert_v1.json

Fusion 原输出，供 Shaper 输入

/tmp/r1a7_wire_grasp_fusion_shadow_v1.json

Shaper 输出，供 Gate 输入

/tmp/r1a7_dual_ibvs_velocity.json

Gate 输出，Controller 消费的实际请求速度



8  实际操作过程与终端执行 / 失败记录

阶段 / 记录

执行操作

可证结果

准备与现场安全备份

copy Visual / Fusion / Shaper / Gate / CTRL、保存 sha256 和 source tar

产生 pre_lower_branch_free_005624，备份 tar 可校验

FREE 身份保护落地

将 lower-branch direction/投影判别加入 Visual direct tracker

09:51 导出的当前源码存在 INSERTION_LOWER_BRANCH_IDENTITY_V1、BRANCH_TIP_LOST 等

速度 10→5→原始

Shaper 六处 cap、Gate 速度条件反复调试

Gate 曾出现 IndentationError；恢复安全 Gate 后编译通过；最终恢复原 SHA

真机低速验证

Gate/Shaper + Controller 的历史真实运行、停止并保存 CSV

09:02 trace 5400、full-chain 11621，仍有 FREE-HOLD 与孔位旧坐标问题

FREE+HOLE 联合补丁 V1

整段字符串替换 Hole inference 条件

F Hole inference condition not found；在 p.write_text 前退出，文件未变

联合补丁 V2

按行定位，但最终静态验证写死固定缩进

Final validation failed for token: camera...；文件未变，visual_compile=0，新 marker 不存在

联合补丁 V3

改为按行定位 + 语义检查 + 内存 compile 后才写盘

已提供命令；截至本次记录未收到成功执行结果



上述两次补丁失败时屏幕混入的“PYint(...)”属于长段终端粘贴回显碎片；有 traceback 的实际失败点是 F 锚点或 V2 最终 token 检查。因为文件写入位于脚本末尾，失败时不应把源码判断为已经更改；最终以 sha256sum 与 grep marker 为准。

8.1 只读核验与当前状态确认命令

cd /home/robot/unitree_sim_isaaclab_threading/r1a7_threading_project
PY=/home/robot/miniconda3/envs/tv_sam2/bin/python
VIS=vision_dual_camera/task6_threading/dataset_tools/wire_unet/live_grasp_to_insertion_visual_v3.py

sha256sum "$VIS"
PYTHONNOUSERSITE=1 "$PY" -m py_compile "$VIS"; echo "visual_compile=$?"
grep -nE 'INSERTION_LOWER_BRANCH_IDENTITY_V1|BRANCH_TIP_LOST|OPPOSITE_BRANCH_REJECTED|FREE_CONTINUOUS_HOLE_LIVE_RELOCALIZATION_V3' "$VIS"
grep -n -A55 -B5 '# Hole inference' "$VIS" | head -100



请在同一个终端重新定义 PY 与 VIS。历史出现过新终端变量为空，导致“找不到命令”或 Python 将项目目录当成模块执行；这不是程序文件损坏。

8.2 只读查看 Left Wrist 的 FREE/HOLE 状态

python3 - <<'PY'
import json
p='/tmp/r1a7_insertion_perception_v1.json'
d=json.load(open(p))
wanted={'free_uv','free_source','tip_uv','hole_uv','hole_valid',
 'direct_state','direct_jump_px','direct_raw_uv','direct_filtered_uv',
 'direct_endpoints_uv','free_branch_locked','free_branch_dir_uv'}
def visit(x,path='root'):
 if isinstance(x,dict):
  hits={k:v for k,v in x.items() if k in wanted}
  if hits: print(path, hits)
  for k,v in x.items():visit(v,path+'.'+str(k))
 elif isinstance(x,list):
  for i,v in enumerate(x):visit(v,f'{path}[{i}]')
visit(d)
PY



8.3 防误操作的回退与核验

# 首先确保真机输出进程已安全停止。
cd /home/robot/unitree_sim_isaaclab_threading/r1a7_threading_project
VIS=vision_dual_camera/task6_threading/dataset_tools/wire_unet/live_grasp_to_insertion_visual_v3.py
SAFE=/home/robot/r1a7_backups/pre_free_hole_live_20261118_095120

STAMP=$(date +%Y%m%d_%H%M%S)
cp -a "$VIS" "$VIS.before_manual_rollback_${STAMP}"
cp -a "$SAFE/live_grasp_to_insertion_visual_v3.py" "$VIS"
/home/robot/miniconda3/envs/tv_sam2/bin/python -m py_compile "$VIS"
sha256sum "$VIS"



预期 Visual 回滚哈希为 b9a12787fe2cc877…；回滚只针对 Visual，不覆盖 Shaper/Gate 的已恢复速度基线。不要在进程正在产生真实机器人运动时直接替换正在执行的控制脚本。

9  下一轮验证计划与完成判据

阶段

执行范围

验收与证据

A 纯感知 FREE

只运行 Visual，不使能 Gate/Shaper 主动输出；记录每帧 direct_state 与真实末端

下半支持续跟随；短遮挡后恢复；异常候选不会跨到上半支；测量/hold 比例有统计

B 左腕 HOLE

插孔模式但控制输出关闭；分别记录 hole_anchor/raw/live 与时间戳

小幅相机/视角变动时 marker 同向连续更新；错孔/突然跳变被拒绝

C 静止噪声

相机静止、工具静止，采集连续孔与 FREE 像素

给出均值/标准差/p95 与极端跳变；定量设置 ROI / jump gate

D 失锁恢复

遮挡 tip、细线和孔，恢复后观察状态切换

短时 dropout 不接管异常值；长期丢失禁止 Z 继续下探

E 控制影子运行

运行 Fusion/Shaper/Gate 的 shadow/无运动记录

确认三轴误差方向、限速、z_scale、measurement_age，不发生过期数据更新

F 真机低速

通过感知与影子测试之后再开放运动

从接近到最终插孔有连续图像/命令/实际数据；接触/穿孔通过独立证据确认



当前必须补齐的功能缺口

V3 补丁本身不包含独立 hole_anchor/live 双轨诊断、严格 ROI + 连续多帧孔位认证、孔失锁时的 Z 硬停止，也不提供 FREE 真正的 velocity prediction。这些应在下一次受控修改中明确实现、运行记录及验收，而不是将文档中的目标架构视为当前可运行功能。



附录 A  关键代码标记与定位参考

代码位置（基线附近行号）

当前可证内容

Visual 1334、1385、1630、1773–1935

下半支常量、InsertionDirectFreeTracker.update、branch reject 及 BRANCH_TIP_LOST

Visual 4715–4767

固定孔阶段说明、Hole inference 仅 WAIT_HOLE 执行

Visual 4891–4968

raw_hole_uv → EMA filtered_hole_uv → target_uv[camera]

Visual 5520–5591

INSERTION 阶段 FREE 直接跟踪调用

Visual 5703–5779

External 与 Left Wrist 的 insertion_guidance 以及 target_uv 用法

Shaper 64/68/119–125

EXT_X_KP、LEFT_U_KP 与恢复后的六处速度上限

Gate 469–490

insertion_tracking_active ? 1.0 : 10.0 的范数限幅



附录 B  来源及证据等级

本报告结合：（1）项目历史文档《R1-A7_细线抓取与插孔研发调试全流程汇总_20261117.docx》；（2）2026-11-18 终端粘贴记录《粘贴的文本 (1).txt》（包含源码片段、编译输出、SHA、备份路径和插孔 Hole/FREE 数据结构）；（3）09:08 左腕截图及 FREE-HOLD 早期画面；（4）Controller trace 文件 r1a7_cartesian_trace_20261118_013709.csv、r1a7_cartesian_trace_20261118_090237.csv；（5）操作者在本项目内的操作反馈与现场描述。

证据判定原则：源码 marker + SHA + 编译可以证明“文件存在/语法正确”，不能证明“闭环识别稳定或穿孔成功”；Controller trace 可以证明运动与指令轨迹，不等于相机图像的实时识别质量；单张截图只能证明某个时刻的显示失配，不能独立锁定故障 direct_state。

截至最后可核对结果，继续工作的最合适基线为：Visual b9a12787…（下半支 V1，未确认 V3），Fusion dc21dd55…，Shaper dfe8f282…，Gate b31f3056…，Controller 0c22da11…。任何下一次修改应重新记录完整 sha256、补丁前备份、编译结果以及视觉无运动验证结果。

附录 C  下一轮试验记录模板（可直接填写）

建议每次纯视觉/影子控制/真机实验均保留一份独立记录，避免仅凭某张截图或某次体感推断算法效果。每次测试先记录版本，再记录“检测→融合→控制→实际运动”的同步时间序列。

记录字段

填写内容 / 验收要求

实验编号 / 时段 / 环境

日期时间、相机是否固定、光照/反光条件、细线姿态、孔位置、是否允许真实运动

源码版本

Visual/Fusion/Shaper/Gate/CTRL SHA256；Git 分支仅作补充，不代替本地哈希

FREE 测量质量

N 帧、MEASURED 比例、MEASURED_HOLD 比例、最长连续 HOLD、跨上半支误锁次数

孔位测量质量

anchor_uv、raw/live_uv、漂移标准差和 p95、最大跳变、重定位失败次数

时序新鲜度

source_frame_seq、measurement_timestamp、视觉循环间隔、控制 JSON 年龄

控制输出

ext_du/left_du/left_dv、xy_score/z_scale、Shaper 三轴请求、Gate 实际矢量限幅

物理运动/安全结果

Controller actual_xyz、command_xyz、cartesian_speed、是否触碰、是否穿孔、是否人工停止
