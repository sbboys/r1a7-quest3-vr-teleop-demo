# R1-A7双臂机器人_抓取插孔技术路线与实验工作记录_20261113 全文

[原件](R1-A7双臂机器人_抓取插孔技术路线与实验工作记录_20261113.docx)；SHA-256：`841157cc56aae89f1e80685e7fc96a29314bb2083974f13d23cc50b887dd01e6`。

保留历史状态、正文、表格和附注；历史指令不是本次执行请求。当前核对报告单独记录差异。

R1-A7 双臂人形机器人

细线抓取与插孔技术路线、程序变更及实验工作记录

对话阶段工作整理版 | 截止 2026-11-13

| 文档用途：汇总本聊天中讨论、执行、编译、测试、训练和归档的技术工作，作为后续实验和算法优化的追溯依据。 |
| --- |

| 证据边界：只把有终端输出或用户明确确认的内容记为已完成；仅给出过命令/方案而没有执行回显的内容，标为待确认。聊天记录不是完整 Git 审计记录。 |
| --- |

文档状态标记与阅读说明

| 标记 | 含义 |
| --- | --- |
| 已执行/有输出 | 用户提供了成功输出、日志或明确完成确认。 |
| 已修改/编译通过 | 代码补丁标记、grep 或 py_compile 有成功证据；不等同于实机验收。 |
| 已训练/已评估 | 有训练日志、指标、checkpoint 或哈希记录。 |
| 建议/待确认 | 对话中给出方案或命令，但未看到执行成功回显。 |
| 风险/待验证 | 出现退化、异常或尚未完成现场闭环。 |

目录

1 项目目标与总体技术路线

2 工作时间线

3 抓取运动控制与夹爪闭合

4 插孔选孔与控制相机策略

5 插孔全过程自动数据记录

6 同步数据筛选、去重和 split

7 Wire/Tweezer/Board/Hole 联合标注

8 Wire V4 训练与评估

9 Hole V2 训练与评估

10 运行时接入、孔跟踪和 FREE-HOLD 故障

11 /data/threading 实验数据中心

12 程序新增、修改、备份、删除和归档记录

13 当前状态、风险和未闭环事项

14 下一阶段标准实验流程

附录：路径、哈希、split IDs

1. 项目目标与总体技术路线

本阶段目标是提升细线（Wire）、镊子（Tweezer）、孔板（Board）和孔位（Hole）的识别率、稳定性与定位精度，并把抓取、选孔、插孔的视觉—融合—门控—控制链路转化为可重复、可记录、可分析的实验流程。

1.1 总体链路

External / Right Wrist / Left Wrist RGB

                 |

       +---------+---------+

       |                   |

   Wire U-Net          Hole U-Net

 Wire / Tweezer       Board / Hole

       |                   |

       +---------+---------+

                 |

 live_grasp_to_insertion_visual_v3.py

                 |

        +--------+--------+

        |                 |

      抓取阶段           插孔阶段

  Wire/Tweezer识别     Hole/FREE端点跟踪

        |                 |

        +--------+--------+

                 |

       wire_grasp_fusion_shadow_v1.py

                 |

       wire_grasp_control_gate_v1.py

                 |

       Cartesian Stream Controller

                 |

             Robot / LowCmd

1.2 两套模型分开训练

| 模型 | 输入 | 类别 | 候选版本 |
| --- | --- | --- | --- |
| Wire U-Net | 三相机 RGB | 0 background / 1 wire / 2 tweezer | V4 epoch 110 |
| Hole U-Net | 三相机 RGB | 0 background / 1 board / 2 hole | V2 epoch 52 |

本轮不是共享 Encoder 的双 Head 网络。Wire 与 Hole 保持两套独立模型、mask 和 checkpoint；训练数据采用相同的 35 个同步时间点和同一 train/val/test 分组。

1.3 已确定的原则

正常抓取阶段保持多轴速度合成；只有最终夹取调整区域才使用必要的对齐/补偿门控，不把全程改成分方向串行运动。

插孔选孔不再要求三个相机全部选中；由配置的控制相机集合判定，其他相机可仅显示/监视。

插孔阶段自动采集三相机图像和同步状态，后续筛选最有价值的困难帧。

35 个同步 group 按 sample ID 划分，禁止同一时刻的三相机图像跨 train/val/test。

增量训练保留旧域 replay 和 guard；reserved test 不参与训练和 checkpoint 选择。

实验中固定版本、保存哈希和日志；一次只改变一个主要因素。

2. 工作时间线

| 时间 | 阶段 | 工作和结果 |
| --- | --- | --- |
| 2026-11-10 | 抓取控制 | 检查夹爪不闭合和手臂停止；增加/检查 sequential alignment、最终夹取 gate、final-adjust 区域和 composite motion 恢复相关逻辑。 |
| 2026-11-10 | 选孔逻辑 | 取消必须三个相机都选孔才允许进入插孔的限制，加入控制相机集合统计。 |
| 2026-11-10 | 数据记录器 | 新增 recorder，自测 7 records / 21 images / 7 events，dropped_queue=0；集成 V3 并编译通过。 |
| 2026-11-10 | 数据筛选 | 从 197 条记录筛出 35 个同步时间点，即 105 张三相机图像。 |
| 2026-11-11 | 联合标注 | 建立联合标注器，完成 105 个视图的 Wire 与 Hole 标注，审计并冻结归档。 |
| 2026-11-11 | 训练准备 | 固定 25/5/5 group split，建立 training_runtime symlink，检查 105 组文件齐全。 |
| 2026-11-11 | Wire V4 | 旧域 replay + insertion 三相机训练；best epoch 110。 |
| 2026-11-11 | Hole V2 | 旧 Hole replay + insertion 三相机训练；best epoch 52。 |
| 2026-11-11 | 模型测试 | Wire V3/V4 与 Hole V1/V2 在旧域及 insertion reserved test 对比。 |
| 2026-11-12 | 实验管理 | 创建 /data/threading，建立 trial01、源码快照、模型 SHA256 并验证。 |
| 2026-11-12 | 采集路径 | 确认 recorder 支持 R1A7_INSERTION_DATASET_ROOT 环境变量，可覆盖默认保存根目录。 |

3. 抓取运动控制与夹爪闭合

3.1 发现的问题

夹爪未闭合，手臂接近目标后停止。

日志曾显示 EXT valid=True、WRIST valid=True，但 proposed=(0,0,0) mm/s；External error 出现约 14.6 px、15.3 px，另一次约 36.4–38.1 px。

Fusion JSON 曾出现 grasp_sequence_stage=WRIST_ALIGN、grasp_ext_near=True、grasp_wrist_sequence_ready=False，三轴 proposed velocity 全为 0。

用户明确要求恢复速度合成方式：最终夹取前的 External 对齐、Wrist 横向确认和前向补偿是最后调整，不应把此前整个运动过程分阶段、分方向串行化。

3.2 代码修改记录

| 文件 | 标记/修改 | 验证证据 |
| --- | --- | --- |
| wire_grasp_fusion_shadow_v1.py | GRASP_SEQUENTIAL_ALIGNMENT_V3_CONSTANTS / STATE / GEOMETRY / FINAL / JSON；EXT_ALIGN、WRIST_ALIGN、FORWARD_COMP 状态。 | grep 多次命中；py_compile 成功。 |
| wire_grasp_fusion_shadow_v1.py | FINAL_ADJUST_ONLY_AUTHORITY_V1、FINAL_ADJUST_LATCH_RESET_V1、COMPOSITE_STAGE_DIAGNOSTIC_V1。 | grep 命中；FUSION_FINAL_COMPILE_OK。 |
| wire_grasp_fusion_shadow_v1.py | GRASP_SEQ_EXT_REACQUIRE_PX=16.0、GRASP_SEQ_EXT_REACQUIRE_V1、WRIST_LATERAL_ONLY_READY_V1、FINAL_NEAR_TIP_DISTANCE_GATE_V1。 | grep 行号 177/479/537/1113；FUSION_COMPILE_OK。 |
| r1a7_cartesian_stream_control...auto_grasp_direct_v1.py | GRASP_SEQUENTIAL_CLOSE_GATE_V1：CLOSE 需满足 Fusion 序列与最终 Wrist 几何检查。 | CTRL_PY_COMPILE_OK。 |

3.3 目标运动逻辑

正常抓取区域：

    保持 External / Wrist 多轴合成速度



最终夹取调整区域：

    External 精确对齐

    Wrist 左右间距与夹爪走廊确认

    既有约 5 mm 前向补偿

    最终闭合许可

| 未闭环：相关代码通过语法检查，但对话中没有最终一次完整抓取成功、夹爪实际闭合并确认夹住细线的验收记录。 |
| --- |

4. 插孔选孔与控制相机策略

原程序要求三个相机全部选孔且有效才能进入插孔。因遮挡时部分相机无法选择，用户要求取消该硬门槛。

修改 live_grasp_to_insertion_visual_v3.py，增加 OPTIONAL_CONTROL_CAMERA_HOLE_SELECTION_V1 和 INSERTION_CONTROL_CAMERAS。

右键确认、Enter 确认、target_ready handshake 与 JSON 输出改为统计控制相机选择情况。

新增 selected_control_cameras、valid_control_cameras、control_target_count、degraded_single_view 字段。

py_compile 成功，grep 命中相关代码标记。

[INSERTION_TRACKING] target=E:1/1 R:1/0 L:0/0

FREE E=MEASURED R=MEASURED L=DISPLAY_ONLY



[HOLE SELECT] external: no hole near click d=315.1px

[HOLE SELECT] external: no hole near click d=670.5px

日志说明 External 有 selected/valid，Right Wrist 已 selected 但 valid=0，Left Wrist 为 display-only；点击位置与检测候选的距离门槛仍可能拒绝选孔。

4.1 重新选孔功能

已分析：首次选孔后 target_selected[camera]=True，后续只追踪该孔，直接再点击其他孔不一定会切换。提出的键位方案为 1=External 重选、2=Right Wrist 重选、3=Left Wrist 重选、N=全部重选、Enter=确认。对话中没有补丁成功输出，因此此功能尚未确认落地。重选应限制在 WAIT_HOLE_TARGET 且机器人停止时。

5. 插孔全过程自动数据记录

5.1 新增文件

vision_dual_camera/task6_threading/dataset_tools/wire_unet/insertion_dataset_recorder_v1.py

记录器按 WAIT_HOLE_TARGET / INSERTION_TRACKING 阶段自动开启，普通帧 5 Hz、事件帧 10 Hz，jpeg_quality=95、max_queue=128。

| 自测项 | 结果 |
| --- | --- |
| records | 7 |
| images | 21（三相机） |
| events | 7 |
| event counts | FUSION_STOP=7；TARGET_SELECTED_CHANGE=2；TARGET_VALID_CHANGE=2；GATE_DISABLED=4；PHASE_CHANGE=1 |
| dropped_queue | 0 |
| self test | SELF_TEST_OK |

5.2 V3 集成

live_grasp_to_insertion_visual_v3.py 增加 INSERTION_DATASET_RECORDER_INTEGRATION_V1。

初始化 InsertionDatasetRecorder，normal_hz=5、event_hz=10、jpeg_quality=95、max_queue=128。

传入三相机图像、target selected/valid、FREE source、Fusion/Gate 状态。

REC 与 V3 py_compile 成功，grep 命中初始化和 record 调用。

5.3 首次真实采集

/home/robot/r1a7_datasets/wire_unet/insertion_runtime_001/raw_sessions/20261110_152101_491

| 统计 | 结果 |
| --- | --- |
| 记录数 | 197 |
| 初始难例分析 | Wire 60 / Hole 60 |
| 后续同步组 | 35 groups |
| 同步图像 | 105 images |

首次数据保存在旧目录。后续确认 recorder 的 DEFAULT_ROOT 可由 R1A7_INSERTION_DATASET_ROOT 覆盖；V3 初始化时未显式传 root，因此可在启动进程时设置环境变量。

R1A7_INSERTION_DATASET_ROOT=/data/threading/raw_sessions/<trial_id> \

PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 \

/home/robot/miniconda3/envs/tv_sam2/bin/python -u \

vision_dual_camera/task6_threading/dataset_tools/wire_unet/live_grasp_to_insertion_visual_v3.py

| 待验证：尚无一次新 session 成功写入 /data/threading/raw_sessions 的终端输出。 |
| --- |

6. 同步数据筛选、去重与 split

6.1 独立图片去重阶段

先建立 insertion_hard_sample_analyzer_v1.py，对 197 条记录分别选择 Wire/Hole hard samples。insertion_annotation_dedup_v1.py 首次运行 contact sheet 时遇到 KeyError: 'source'，只生成 external 40 张部分结果，summary 不完整；修复后再次执行成功，独立图片方案共选 100 张（External 40、Right Wrist 35、Left Wrist 25）。

6.2 改为同步 group

用户明确取消 100 张独立图片，改为约 35 个同步时间点、约 105 张三相机图片。insertion_sync_group_dedup_v1.py 经多次 anchor mismatch 调整后，最终运行成功。

| 参数/结果 | 值 |
| --- | --- |
| total groups | 35 |
| wait groups | 10 |
| normal fraction | 0.20 |
| min time gap | 0.75 s |
| hash threshold | 8 |
| selected groups/images | 35 / 105 |
| phase counts | WAIT_HOLE_TARGET=10；INSERTION_TRACKING=25 |
| category counts | hard=35 |
| 实测最小时间间隔 | 0.811 s |

| 数据分布注意：最终 category_counts 为 hard=35；另一次诊断 stable_count=0。数据偏向困难样本，常规稳定场景覆盖不足，后续应补采正常样本、不同光照和不同视角。 |
| --- |

6.3 固定 split

使用 seed=44，按 sample ID 分组切分 25/5/5。相同 ID 的 External、Right Wrist、Left Wrist 必须在同一 split。

| Split | Sample IDs | Groups / Images |
| --- | --- | --- |
| Train | 000002 000003 000005 000006 000007 000009 000011 000014 000015 000016 000017 000018 000019 000020 000021 000022 000023 000024 000025 000026 000028 000029 000031 000032 000035 | 25 / 75 |
| Val | 000001 000004 000010 000030 000033 | 5 / 15 |
| Test | 000008 000012 000013 000027 000034 | 5 / 15 |

Split 文件：training_runtime/splits_insertion_joint_grouped_seed44_v1.json。对话中报告 SHA256：79954f512838a8df0ce0c6cf9eb8c52a9817741ed6ec474d6df1ffdbe7bd59bc。

6.4 训练兼容 symlink

training_runtime/images   -> images_raw/session_20261110_152101_sync35

training_runtime/masks    -> masks_wire

training_runtime/masks_v3 -> masks_hole

完整性检查：train=75、val=15、test=15、missing=0，输出 INSERTION_TRAIN_RUNTIME_OK。

7. Wire / Tweezer / Board / Hole 联合标注

用户要求同一张 RGB 同时标注 Wire/Tweezer 与 Board/Hole，不再把两套任务分成不同图片。最初 annotate_insertion_joint_v1.py 不存在，随后以 hole_unet/annotate_hole_board_unet_v3.py 为基底复制并扩展为联合标注器。

| 键位 | 功能 |
| --- | --- |
| 1 / 2 / 3 / 4 | Wire / Tweezer / Board / Hole |
| Enter | 确认当前标注 |
| C / Esc | 取消当前标注 |
| U | 撤销上一步 |
| Z / X | 放大 / 缩小 |
| 方向键 | 移动画面 |
| S | 同时保存 Wire + Hole mask |
| N / P | 下一张 / 上一张 |
| Q | 退出 |

用户确认全部标注完成。

| 审计项 | 结果 |
| --- | --- |
| expected_views / missing_files | 105 / 0 |
| bad_shapes / bad_values | 0 / 0 |
| wire_nonempty / hole_nonempty | 105 / 104 |
| geometry_records / geometry_missing | 105 / 0 |
| Wire pixel values | 0: 103,587,660；1: 135,831；2: 212,509 |
| Hole pixel values | 0: 99,673,677；1: 4,106,933；2: 155,390 |

| 需复核：Hole nonempty=104/105；有一张图没有 Hole 前景，需查看是否确实没有可见孔或存在漏标。 |
| --- |

mask_previews_hole 目录被用户打开时为空。分析认为标注器只保存 masks_hole，没有自动生成 preview。随后给出 Hole preview 和 Wire preview 的生成脚本，但没有用户成功执行回显，因此预览图是否已生成尚未确认。

7.1 标注冻结

insertion_joint_001/frozen/

  insertion_joint_annotation_complete_20261111_165043.tar.gz

第一次把 tar 放在 selection 子目录，出现归档过程中读取文件变化提示；随后改为 frozen 目录，tar_rc=0，文件约 22 MB。SHA256：9ad042afbb39543ec8deaa1c5d80512f8335fb2bbe6e9c18b237b82540334321。

8. Wire V4 增量训练与评估

8.1 训练器和数据

现有 Wire V3 trainer 的 new domain 只硬编码 Right Wrist，不能直接用于三相机 insertion。新增 train_wire_unet_v4_insertion_joint.py，加载 V3 checkpoint 权重、重置优化器并混合旧域 replay 与 insertion 三相机数据。

| 来源 | 训练项数 | Repeat |
| --- | --- | --- |
| pilot_001（三相机） | 105 | 1 |
| right_wrist_open_001（Right Wrist） | 47 | 1 |
| insertion_joint（三相机 train 75） | 225 | 3 |
| 总计 | 377 | — |

起始 checkpoint：right_wrist_open_001/runs/v3_mixed_replay/best.pt，epoch 90。训练 20 epochs，batch=2、crop=512、val stride=384、lr=1e-5。

8.2 训练指标

| 指标 | Baseline | Best epoch 110 |
| --- | --- | --- |
| Pilot val mean FG | 0.752381 | 0.7382（Δ约 -0.0142） |
| Open val mean FG | 0.772780 | 0.7485（Δ约 -0.0243） |
| Insertion val mean FG | 0.606505 | 0.8146（Δ约 +0.2081） |
| Combined score | 0.677249 | 0.782795 |

epoch 105、110 同时通过 pilot/open guard；最终 best epoch=110，训练返回码 0，INSERTION TEST USED=0。

8.3 Reserved test：V3 vs V4

| 测试集 | V3 | V4 | Δ |
| --- | --- | --- | --- |
| Pilot mean FG | 0.756371 | 0.744155 | -0.012216 |
| Right-wrist-open | 0.728989 | 0.734124 | +0.005135 |
| Insertion mean FG | 0.623970 | 0.849347 | +0.225377 |

| Insertion 相机 | V3 | V4 | Δ |
| --- | --- | --- | --- |
| External | 0.622687 | 0.842285 | +0.219599 |
| Right Wrist | 0.660550 | 0.837460 | +0.176910 |
| Left Wrist | 0.519603 | 0.870559 | +0.350956 |

Pilot per-camera：External 0.815847→0.814057；Right Wrist 0.706253→0.686464；Left Wrist 0.757492→0.744531。Open test Right Wrist 0.728989→0.734124。

8.4 冻结

frozen/models/wire_v4_insertion_replay_seed44_epoch110.pt

约 89 MB。SHA256：31dcc17b4353be524af6bd75613ecf8802745ad0a2f07fdc1bb19d0587be88d7。

| 结论：Wire V4 在 insertion reserved test 三相机显著提升，旧域指标在本轮 guard 范围内，标记为 ACCEPTED 候选。该 test 已经被查看，不再用于反复调参。 |
| --- |

9. Hole V2 增量训练与评估

Hole V1 原训练器无 resume 参数。新增 train_hole_board_unet_v2_insertion_replay.py，加载 Hole V1 的 model_state_dict，重置优化器，混合旧 Hole 与 insertion 三相机数据。旧 checkpoint epoch=38，类别权重 [0.20,1.00,5.00]。

| 项目 | 结果 |
| --- | --- |
| Old train unique / New train unique | 72 / 75 |
| Old repeat / New repeat | 1 / 3 |
| Mixed train items | 297 |
| Insertion val / reserved test | 15 / 15 images |
| Baseline old / insertion score | 0.871138 / 0.685642 |
| Baseline combined / old floor | 0.750566 / 0.841138 |
| Best epoch / score | 52 / 0.811673 |

Best epoch 52：old score 约 0.8593（比 baseline 约 -0.0118），insertion val score 约 0.7860（约 +0.1004），old_guard=1，训练返回码 0，NEW TEST USED=0。

9.1 Reserved test：V1 vs V2

| 指标 | V1 | V2 | Δ |
| --- | --- | --- | --- |
| Old Board Dice | 0.938840 | 0.905769 | -0.033071 |
| Old Hole Dice | 0.810572 | 0.811509 | +0.000936 |
| Old mean FG | 0.874706 | 0.858639 | -0.016067 |
| Old score | 0.829813 | 0.825648 | -0.004165 |
| Insertion Board Dice | 0.851124 | 0.940644 | +0.089520 |
| Insertion Hole Dice | 0.627552 | 0.770904 | +0.143352 |
| Insertion mean FG | 0.739338 | 0.855774 | +0.116436 |
| Insertion score | 0.661087 | 0.796365 | +0.135277 |

| 数据域/相机 | Board V1→V2 | Hole V1→V2 | Mean FG V1→V2 |
| --- | --- | --- | --- |
| Old External | 0.9312→0.8156 | 0.8097→0.8004 | 0.8705→0.8080 |
| Old Right Wrist | 0.9424→0.9210 | 0.8156→0.8221 | 0.8790→0.8715 |
| Old Left Wrist | 0.9334→0.9339 | 0.7969→0.7911 | 0.8651→0.8625 |
| Insertion External | 0.8321→0.9622 | 0.6894→0.7948 | 0.7607→0.8785 |
| Insertion Right Wrist | 0.8734→0.9085 | 0.5657→0.7440 | 0.7195→0.8262 |
| Insertion Left Wrist | 0.8484→0.9530 | 0.6099→0.7639 | 0.7291→0.8585 |

运行时 hole center 提取使用 mask==2 的连通域，因此 Board class 1 不直接参与 hole center 计算。但 Old External Board Dice 明显下降，仍应在实时显示与旧场景中复核。

frozen/models/hole_v2_insertion_replay_seed44_epoch52.pt

约 89 MB。SHA256：467c57b3e74f9b885013bc8c0a77a3881fad53283d0325b2e0ba3f450d0a6b52。

| 结论：Hole V2 insertion Hole Dice 三相机均提升；旧域 overall score 下降较小，但 External Board 有显著退化。标记为 ACCEPTED AS CANDIDATE，部署前需视觉-only复核。 |
| --- |

10. 运行时模型接入、孔位跟踪和 FREE-HOLD

10.1 模型接入

检查发现正式主程序 live_grasp_to_insertion_visual_v3.py 中 Wire 原加载为 External/Left 使用 base model，Right Wrist 使用 RIGHT_WRIST_V3_CHECKPOINT；Hole checkpoint 原指向 Hole V1，且在插孔阶段 lazy-load。用户要求直接接入 Wire V4/Hole V2。对话给出了备份正式 V3、修改 checkpoint、三路 Wire 改为 V4、py_compile 的补丁，但没有看到后续成功回显。因此正式运行时是否已实际换成 V4/V2，必须检查实际文件和启动日志。

10.2 腕部相机选定孔偏移

源码分析发现 target_uv 本来会在成功检测后被 EMA 更新，并用于下一帧局部搜索；问题是同一个平滑 target 同时承担搜索中心与控制目标，腕部相机移动时容易因 EMA 滞后。分析时参数为 ROI=50 px、max jump=25 px、EMA alpha=0.15、hold=0.80 s。

提出 DYNAMIC_SELECTED_HOLE_TRACK_V1：raw hole center 用于下一帧 ROI 搜索；EMA target 用于 IBVS 控制；External/Right Wrist/Left Wrist 分别设置 ROI、jump、EMA。该补丁命令已提出，但没有执行成功的终端证据，状态为待确认。

10.3 FREE-HOLD 与有像素误差但手臂不动

用户提供的画面出现 FREE-HOLD。分析认为该标签表示当前 FREE 点可能使用上次有效测量的 hold 值，而不是 fresh measurement。插孔分工被描述为 External FREE→HOLE 负责 X/Z，Right Wrist FREE→HOLE 负责 Y/Z；图像二维误差不等于机械臂按屏幕箭头直接移动。

必须同步检查 free_source、hole_uv、du/dv、guidance valid、Fusion vx/vy/vz、Gate enabled/STOP、Controller 实际 Cartesian 输出。由于未提供同一时刻完整 JSON 三件套，根因尚未确认。

10.4 重新选孔

当前选中后 target_selected=True，追踪锁定该孔。建议在 WAIT_HOLE_TARGET 增加单相机和全部重选键；进入 INSERTION_TRACKING 后不应无保护切换目标。补丁执行状态未确认。

11. /data/threading 实验数据中心

用户要求将新的测试数据统一放在文件管理器中的 /data/threading 文件夹。已创建 experiments、raw_sessions、analysis、reports、archives。/data 分区约 1.4 TB，总已用约 139 GB，可用约 1.2 TB；touch 写入测试返回 THREADING_DATA_ROOT_OK。

/data/threading/

  experiments/

    20261112_194018_trial01/

      logs/

      state/

      source_snapshot/

      analysis/

      events/

      results/

      start_time.txt

      git_commit.txt

      git_status.txt

      source_sha256.txt

      model_sha256.txt

  raw_sessions/

  analysis/

  reports/

  archives/

  current_trial.txt

用户曾在 /home/robot/r1a7_experiments/threading 建立同名 trial01，后将源码快照、git 信息、模型和源码 hash 整合到 /data/threading/experiments/20261112_194018_trial01。current_trial.txt 已指向该路径。

source_sha256.txt 中四个源码文件校验成功；model_sha256.txt 中 Wire V4 和 Hole V2 checkpoint 校验成功。

采集器已确认支持 R1A7_INSERTION_DATASET_ROOT 环境变量。推荐新 session root：/data/threading/raw_sessions/20261112_194018_trial01；但还没有新 session 成功写入该路径的输出。

12. 程序新增、修改、备份、删除和归档记录

12.1 新增程序

| 文件 | 用途 | 状态 |
| --- | --- | --- |
| insertion_dataset_recorder_v1.py | 插孔阶段自动图像/状态记录 | 自测和 V3 集成编译通过 |
| insertion_hard_sample_analyzer_v1.py | Wire/Hole hard sample 分析 | py_compile 与 session 分析成功 |
| insertion_annotation_dedup_v1.py | 独立图片去重 | 修复 KeyError 后成功选出 100 张；后被同步 group 方案替代 |
| insertion_sync_group_dedup_v1.py | 同步三相机 group 去重 | 最终选出 35 groups / 105 images |
| annotate_insertion_joint_v1.py | 联合 Wire/Hole 标注 | 用户确认完成全部标注 |
| train_wire_unet_v4_insertion_joint.py | Wire V4 mixed replay | 训练和评估成功 |
| train_hole_board_unet_v2_insertion_replay.py | Hole V2 mixed replay | 训练和评估成功 |

12.2 修改的既有程序

| 文件 | 修改 | 证据/状态 |
| --- | --- | --- |
| wire_grasp_fusion_shadow_v1.py | 抓取顺序对齐、final-adjust、latch reset、composite diagnostic、reacquire/wrist/near-tip gate | 多次 grep 与 py_compile 成功 |
| r1a7_cartesian_stream_control...auto_grasp_direct_v1.py | GRASP_SEQUENTIAL_CLOSE_GATE_V1 | py_compile 成功 |
| live_grasp_to_insertion_visual_v3.py | 可选控制相机选孔、recorder 集成；后续模型替换/动态跟踪方案 | 前两项编译通过；后续补丁执行未确认 |
| insertion_sync_group_dedup_v1.py | 视觉难例重评分与时间去重调整 | 最终选组成功 |
| annotate_insertion_joint_v1.py | Hole 功能、键位、缩放和平移交互 | 用户确认标注完成 |

12.3 备份

Fusion 文件生成过 before_grasp_sequence_v3、before_fix_final_adjust_latch 等时间戳备份。

同步选择器生成过 before_visual_only_rescore、rescore_v2、rescore_v3 等备份。

联合标注器在同名文件存在时执行过时间戳备份逻辑。

正式 V3 模型替换/动态追踪前的备份命令曾提供，但未见成功回显。

12.4 删除与移动

没有证据表明正式视觉、Fusion、Gate、Controller、训练器、checkpoint 或冻结 GT 被删除。

rm -rf /tmp/r1a7_insertion_recorder_selftest：只清理临时自测目录。

rm -f training_runtime/images、masks、masks_v3：只清理兼容 symlink 后重新创建，不删除原始 RGB 或 GT mask。

临时 /data/threading/experiments/20261112_194522_trial01 按命令移动/整合到正式 trial01。

未确认删除旧 /home/robot/r1a7_experiments/threading，可能仍保留副本。

12.5 归档

首次将 tar.gz 放在 selection 目录时，tar 提示文件在读取过程中发生变化；随后把归档放到 frozen 目录，tar_rc=0，约 22 MB。

insertion_joint_001/frozen/insertion_joint_annotation_complete_20261111_165043.tar.gz

SHA256：9ad042afbb39543ec8deaa1c5d80512f8335fb2bbe6e9c18b237b82540334321。

13. 当前基线、风险和未闭环事项

| 模块 | 当前记录 | 待确认 |
| --- | --- | --- |
| Wire 模型 | V4 epoch110，insertion test mean FG 0.849347，已冻结 | 正式主程序是否实际加载 V4；视觉-only实时验证 |
| Hole 模型 | V2 epoch52，insertion Hole Dice 0.770904，已冻结 | 正式主程序是否加载 V2；复核 Old External Board |
| 抓取控制 | composite 运动原则和 final close gate 已写入/编译 | 实机是否正常合成运动、成功闭合 |
| 选孔 | optional control camera 逻辑已编译 | 单/双相机选孔与重新选孔现场测试 |
| Hole tracking | raw search center 与 EMA 分离方案已提出 | 确认补丁是否实际应用并视觉验证 |
| FREE-HOLD | 图像出现 FREE-HOLD，根因未确认 | 同步采集 perception/target/Fusion/Gate/Controller |
| Recorder | 自测、集成成功，支持环境变量切 root | 新 session 写入 /data 路径验证 |
| Trial 01 | 目录、源码快照、模型哈希已建立并校验 | 四路日志、状态 JSONL、raw session 和结果报告尚待完成 |

BASELINE_RUNTIME_V1（拟定基线，首次完整 trial 验收后正式确认）

Wire:   wire_v4_insertion_replay_seed44_epoch110.pt

Hole:   hole_v2_insertion_replay_seed44_epoch52.pt

Visual: live_grasp_to_insertion_visual_v3.py

Fusion: wire_grasp_fusion_shadow_v1.py

Gate:   wire_grasp_control_gate_v1.py

Control:r1a7_cartesian_stream_control_kd30_posvel_wu_dqik_u010_auto_grasp_direct_v1.py

14. 下一阶段标准实验流程

14.1 Trial 开始前

确认工作区、机械臂初始状态、夹爪状态和急停；检查只有一个 LowCmd publisher。

确认 Visual/Fusion/Gate/Controller 的源码快照和 SHA256。

确认 Wire V4 与 Hole V2 checkpoint SHA256。

设置 R1A7_INSERTION_DATASET_ROOT 指向 /data/threading/raw_sessions/<trial_id>。

确认 logs/state/events/results 目录可写。

先做视觉和状态检查；只有满足现有安全条件时才启动真实控制。

14.2 统一事件名称

EXPERIMENT_START

GRASP_START

GRASP_NEAR_WIRE

GRASP_CLOSE

GRASP_CONFIRMED

HOLE_SELECTION_START

HOLE_SELECTED

INSERTION_START

FAILURE_FREE_HOLD

FAILURE_HOLE_LOST

FAILURE_NO_MOTION_WITH_PIXEL_ERROR

FAILURE_WRONG_DIRECTION

INSERTION_CONTACT

SUCCESS

FAILURE

EXPERIMENT_STOP

14.3 每次实验应保存

| 文件/目录 | 内容 |
| --- | --- |
| logs/visual.log | Wire/Hole/FREE source、target、阶段变化、感知错误 |
| logs/fusion.log | 轴来源、合成速度、阶段和停止原因 |
| logs/gate.log | enabled/STOP、valid/stale/age、拒绝原因 |
| logs/controller.log | 收到的命令、Cartesian/IK/关节限制和输出 |
| state/runtime_state.jsonl | 高频状态快照 |
| raw_sessions/<trial>/ | 三相机同步图像、records/events/summary |
| events/events.log | 人工标记时间点 |
| results/result.json | 成功/失败、阶段、物理结果、备注 |
| analysis/ | 误差/速度/valid/Gate/source/位置跟踪曲线 |

14.4 故障归因顺序

视觉测量

  -> guidance error / proposed velocity

  -> Fusion 轴来源与合成速度

  -> Gate enabled / STOP reason

  -> Controller received command

  -> IK / joint limits / actual motion

  -> 物理抓取或插孔结果

例如：视觉有误差但 proposed velocity=0，先查 perception/guidance；Fusion 非零但 Gate=0，查 Gate；Gate 非零但 Controller 无响应，查 Controller/IK。每次只修改一个层级。

14.5 第一批实验

建议在不改模型、速度、ROI、Fusion 权重的情况下先完成至少 5 次完整 trial。每次记录抓取成功、选孔成功、插孔成功、各相机 Hole/FREE 有效率、FREE-HOLD 比例、Fusion STOP、Gate STOP 和 Controller 响应。安全异常出现时立即停止，不为凑次数继续运行。首批实验完成后再做失败模式统计和单变量优化。

附录 A：关键路径与 SHA256

| 对象 | 路径 / 哈希 |
| --- | --- |
| 项目根目录 | /home/robot/unitree_sim_isaaclab_threading/r1a7_threading_project |
| 联合数据集 | /home/robot/r1a7_datasets/wire_unet/insertion_joint_001 |
| 同步 RGB | images_raw/session_20261110_152101_sync35 |
| Wire masks / Hole masks | masks_wire/ 与 masks_hole/（各含三个相机目录） |
| split 文件 | training_runtime/splits_insertion_joint_grouped_seed44_v1.json |
| split SHA256 | 79954f512838a8df0ce0c6cf9eb8c52a9817741ed6ec474d6df1ffdbe7bd59bc |
| 标注冻结包 SHA256 | 9ad042afbb39543ec8deaa1c5d80512f8335fb2bbe6e9c18b237b82540334321 |
| Wire V4 SHA256 | 31dcc17b4353be524af6bd75613ecf8802745ad0a2f07fdc1bb19d0587be88d7 |
| Hole V2 SHA256 | 467c57b3e74f9b885013bc8c0a77a3881fad53283d0325b2e0ba3f450d0a6b52 |
| Trial 01 | /data/threading/experiments/20261112_194018_trial01 |
| Trial 01 raw root（计划） | /data/threading/raw_sessions/20261112_194018_trial01 |

附录 B：当前待办清单

核验正式 V3 是否加载 Wire V4 三相机与 Hole V2，并保留启动日志。

核验 DYNAMIC_SELECTED_HOLE_TRACK_V1 是否实际应用；必要时先备份、精确补丁、编译和视觉-only验证。

实现并验证 WAIT_HOLE_TARGET 阶段安全重选孔。

用同一时间戳采集 FREE-HOLD 时的 perception、target state、Fusion、Gate、Controller 数据。

验证 recorder 将新 session 写入 /data/threading/raw_sessions/trial01。

建立并验证 runtime_state.jsonl 记录器，统计读取错误和时间戳。

检查 Hole nonempty=104/105 的样本，并生成 Wire/Hole mask preview。

完成至少 5 次 trial 后，生成自动分析图表和失败模式 Pareto。

| 最终结论：本阶段已完成同步三相机插孔数据采集、联合标注、数据审计与冻结归档，以及 Wire V4 / Hole V2 增量训练和保留测试。下一阶段从“训练新模型”转向“运行时模型接入核验、全链路实验记录、失败归因和单变量优化”。 |
| --- |
