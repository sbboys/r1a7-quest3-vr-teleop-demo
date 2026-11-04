# 2026-11-03 右腕张开镊子视觉与自动抓取原始记录全文

[原始DOCX](2026.11.3R1-A7_右腕张开镊子视觉与自动抓取工作记录_2026-11-03.docx)；SHA-256：`85e9b358cbedd96e499bf17912404a4ae74e29c932915b6e174bef5449bcecb0`。

历史操作、补丁与后续建议是来源记录，不是本次运行指令。当前文件核对状态见归档报告；Word原件保持不变。

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

| 核心结论  原自动 OPEN 成功但 CLOSE 不触发的主要视觉根因，是 V2 只在闭合镊子外观上训练，张开后 Right Wrist tool mask / TIP 失真。V3 通过 Right-Wrist OPEN 数据增量训练后，在独立 test 上 tool Dice 从 0.5692 提升到 0.8915；随后代码改为 External/Left 继续 V2、Right Wrist 单独 V3。当前下一阶段重点是张开后的双尖端中点 TIP 以及 CLOSE Gate 对 Right Wrist 垂直误差的处理。 |
| --- |

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

| GRIPPER_OPEN_Q  = [4.86,  +0.1585]GRIPPER_CLOSE_Q = [-0.08, -0.0500]R1A7_AUTO_GRASP_ENABLE=1R1A7_AUTO_GRASP_PREOPEN_NORM_MAX=0.70R1A7_AUTO_GRASP_CLOSE_AXIS_NORM_MAX=0.10 |
| --- |

| 原则  当时没有直接放宽 CLOSE 阈值，而是优先修复 Right Wrist OPEN 后的视觉模型。原因是阈值放宽可能让错误/跳变的 TIP 通过安全门槛。 |
| --- |

2. 现有视觉与控制链路

2.1 三相机与 U-Net 前端

| 相机 | 设备/分辨率 | 当前职责 |
| --- | --- | --- |
| External USB ZOOM | 1280×720 | External wire / GRASP；X、Z 方向信息 |
| Right Wrist Orbbec | CPCBC53000C5；1280×800 | Right Wrist live wire/tool/TIP；Y、Z 信息 |
| Left Wrist Orbbec | CPCBC530007Z；1280×800 | 辅助固定线参考；Left visual |

实时视觉主程序：

| vision_dual_camera/task6_threading/dataset_tools/wire_unet/live_visual_guidance_shadow_v1.py |
| --- |

其视觉链路复用 live_hybrid_locked_wire_live_tweezer_tip_v3，并实际使用 visual.v2.TweezerTipTrackerV2。External/Left 的线参考在人工 FREE 标定后锁定，Right Wrist 的线几何保持 live。

2.2 图像误差到机器人方向映射

| 视觉误差 | 机器人方向 |
| --- | --- |
| External +U | +X |
| External +V | -Z |
| Right Wrist +U | -Y |
| Right Wrist +V | -Z |

2.3 JSON 与融合输出

| /tmp/r1a7_external_wire_grasp_v1.json/tmp/r1a7_wrist_wire_grasp_guidance.json/tmp/r1a7_wire_grasp_fusion_shadow_v1.json/tmp/r1a7_dual_ibvs_velocity.json |
| --- |

3. Right Wrist OPEN 数据采集

旧的 OpenCV/V4L2 采集方式会选到错误 video node，出现绿/紫色、噪声或深度格式画面。Right Wrist Gemini 336L 的多个 /dev/video 节点分别暴露 Z16、GREY/NV12、BA81/YV12 等格式，因此最终改为与正式视觉一致的 pyorbbecsdk Pipeline + COLOR_SENSOR。

3.1 采集脚本与设备

| vision_dual_camera/task6_threading/dataset_tools/wire_unet/capture_right_wrist_open_orbbec_v2.pyRight Wrist serial: CPCBC53000C5Requested COLOR: 1280x800, RGB, 30 fps |
| --- |

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

| 最终新数据 split  train=47，val=8，test=9；test 在 V3 训练期间保留不用。 |
| --- |

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

| vision_dual_camera/task6_threading/dataset_tools/annotate_wire_unet_v2_right_wrist_open.pyCAMERAS = ("right_wrist",) |
| --- |

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

| 原 V2 best:/home/robot/r1a7_datasets/wire_unet/pilot_001/runs/unet_train_val_v2_resume/best.pt冻结副本:/home/robot/r1a7_datasets/wire_unet/right_wrist_open_001/runs/v3_right_wrist_open/init_v2_best.ptSHA256:cb7ae5f4664bb5f1a558beeb941a49847cc1694f4d343024391973d4052c1687 |
| --- |

原 V2 与冻结副本 SHA256 完全一致，旧模型不会被覆盖。

5.2 为什么不能直接用原 V2 trainer 训练 64 张新图

原 TrainDataset 会把每个 sample_id 强制展开为 external/right_wrist/left_wrist 三个 camera item；原 validate 也按三相机遍历。因此 Right-Wrist-only 新数据不能直接作为原 dataset 根目录使用，否则会要求不存在的 External/Left 图像。

| for sid in sample_ids:    for camera in ("external", "right_wrist", "left_wrist"):        self.items.append((sid, camera)) |
| --- |

5.3 V3 mixed replay trainer

| vision_dual_camera/task6_threading/dataset_tools/wire_unet/train_wire_unet_v3_right_wrist_open.py |
| --- |

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

| PYTHONNOUSERSITE=1 \/home/robot/miniconda3/envs/tv_sam2/bin/python -u \vision_dual_camera/task6_threading/dataset_tools/wire_unet/train_wire_unet_v3_right_wrist_open.py \--old-dataset /home/robot/r1a7_datasets/wire_unet/pilot_001 \--new-dataset /home/robot/r1a7_datasets/wire_unet/right_wrist_open_001 \--resume /home/robot/r1a7_datasets/wire_unet/right_wrist_open_001/runs/v3_right_wrist_open/init_v2_best.pt \--epochs 20 --batch-size 2 --crop-size 512 --val-stride 384 \--lr 2e-5 --weight-decay 1e-4 --new-repeat 3 |
| --- |

6. V3 训练结果与独立测试

6.1 Baseline 与 best epoch

| 指标 | V2 baseline | V3 best (epoch 90) | 变化 |
| --- | --- | --- | --- |
| Old val mean FG Dice | 0.750200 | 0.7524 | +0.0022（无遗忘） |
| New Right Wrist val mean FG Dice | 0.477433 | 0.7728 | +0.2954 |
| Combined score | 0.586540 | 0.764621 | +0.178081 |
| Old guard floor | 0.730200 | 满足 | 全程 old_guard=1 |

训练曲线在 epoch 90 达到最佳 combined score；epoch 91-92 没有继续提升，因此不再追加 epoch，保留 epoch 90 的 best.pt。

| /home/robot/r1a7_datasets/wire_unet/right_wrist_open_001/runs/v3_mixed_replay/best.pt |
| --- |

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

| 结论  V3 已经解决 Right Wrist OPEN/遮挡外观下最严重的 tool segmentation 退化，同时 wire 也改善。000061-000063 的 tool Dice 从接近失效（0.017-0.096）恢复到约 0.83-0.85。 |
| --- |

7. 实时视觉双模型部署

7.1 原实现

live_visual_guidance_shadow_v1.py 原先仅调用一次 base.load_model()，同一个 model 通过 base.run_one(model, device, frame) 用于 External、Right Wrist、Left Wrist。base 实际模块为 live_wire_unet_three_camera.py，load_model() 无 checkpoint 参数并使用固定 CHECKPOINT。

7.2 部署策略

| 最小风险策略  External 和 Left Wrist 继续使用旧 V2；只有 Right Wrist 使用 V3。这样修复问题域，不改变已经稳定的 External/Left 行为。 |
| --- |

| External     -> V2Left Wrist   -> V2Right Wrist  -> V3 (v3_mixed_replay/best.pt) |
| --- |

7.3 已执行备份

| vision_dual_camera/task6_threading/dataset_tools/wire_unet/live_visual_guidance_shadow_v1.py.bak_before_right_wrist_v3_20261103_165708 |
| --- |

7.4 已执行代码修改

在 live_visual_guidance_shadow_v1.py 中：

新增 import torch。

新增 RIGHT_WRIST_V3_CHECKPOINT 常量。

新增 load_model_from_checkpoint(checkpoint, device) helper，使用 base.UNet 加载指定 checkpoint。

保留原 model, device = base.load_model() 作为 V2。

新增 model_right_wrist 加载 V3。

三相机循环增加 active_model：right_wrist 使用 model_right_wrist，其余使用 model。

推理模型选择核心

| active_model = (    model_right_wrist    if camera == "right_wrist"    else model)detections[camera], fps_values[camera] = base.run_one(    active_model, device, frames[camera]) |
| --- |

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

| CLOSED:    tip_yx = 原 TweezerTipTrackerV2 单尖端OPEN:    tip_1 = 第一片镊臂尖端    tip_2 = 第二片镊臂尖端    tip_yx = ((y1+y2)/2, (x1+x2)/2) |
| --- |

8.3 设计方案（尚待确认是否已部署）

已设计新增文件 live_right_wrist_tweezer_tip_midpoint_v1.py，并计划只替换 Right Wrist 的 tip tracker；External/Left 继续使用原 TweezerTipTrackerV2。拟增加诊断字段：tip_mode、tip_pair_yx、tip_pair_separation_px、blade1_pixels、blade2_pixels。

| 模式 | 触发 | 输出 |
| --- | --- | --- |
| SINGLE_CLOSED | 单显著 component 或双尖端间距很小 | 原单 TIP |
| SINGLE_FALLBACK | 双尖端提取失败 | 回退原 V2 TIP |
| OPEN_MIDPOINT | 两个有效尖端且间距超过阈值 | 双尖端中点 TIP |

| 注意  本次对话没有收到该 midpoint tracker 的创建/py_compile/运行回传，因此应视为“方案已设计，部署状态待确认”，不能当作已完成代码。 |
| --- |

8.4 当前观察到的 TIP/GRASP 垂直跳变

实时视觉日志中 Right Wrist 的 du 已接近 0，但 dv 经常在约 +20~+32 px 与 -15~-30 px 之间切换。该现象说明 OPEN 状态下垂直定位仍可能受遮挡、GRASP 点偏移或 TIP 模式切换影响。

9. Auto-Grasp CLOSE Gate：取消 Right Wrist 垂直 ev 判据

9.1 现场日志分析

控制日志显示 OPEN_TRACKING 中 Right Wrist 水平方向 eu 已经很小，但 ev 长时间远高于 0.10。例如：WRIST en=0.556, eu=-0.026, ev=+0.555, jit=0.014。此时 eu 与 jitter 合格，但 ev 单独阻止 CLOSE。

同时，实际工况中夹爪靠近细线后会遮挡 GRASP 点，Right Wrist 垂直方向 GRASP 识别可能偏移到其他位置，因此 Right Wrist ev 不再适合作为闭合必要条件。

9.2 当前源码中的 CLOSE 条件

| ext_close = (    abs(em["eu"]) <= self.close_axis_norm_max    and abs(em["ev"]) <= self.close_axis_norm_max    and em["jitter"] <= self.close_jitter_max)wrist_close = (    abs(wm["eu"]) <= self.close_axis_norm_max    and abs(wm["ev"]) <= self.close_axis_norm_max    and wm["jitter"] <= self.close_jitter_max) |
| --- |

9.3 用户确认的新规则

| 变更要求  只取消 Right Wrist 的垂直 ev CLOSE 判定；External 的 eu/ev/jitter、Right Wrist 的 eu/jitter、0.40 s 稳定保持、OPEN 状态及其他逻辑全部保持不变。 |
| --- |

修改后的 wrist_close

| wrist_close = (    abs(wm["eu"]) <= self.close_axis_norm_max    and wm["jitter"] <= self.close_jitter_max) |
| --- |

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

| r1a7_cartesian_stream_control_kd30_posvel_wu_dqik_u010_auto_grasp_direct_v1.py.bak_before_remove_wrist_ev_close_gate_<timestamp> |
| --- |

之后已精确定位源码行：wrist_close 中包含 wm["eu"]、wm["ev"]、wm["jitter"] 三项。删除 wm["ev"] 的精确 patch 已给出，但本次对话尚未收到 patch 执行和 py_compile 回传，因此文档将其状态登记为“待执行/待验证”。

9.5 为什么不能误读 WRIST en

即使取消 Right Wrist ev CLOSE 判定，日志中的 WRIST en 仍可能很大，因为 en=hypot(eu,ev)。修改后不能用 en<0.10 判断是否允许 CLOSE；Right Wrist CLOSE 只看 |eu| 和 jitter。

10. 其他控制层变更与风险点

10.1 手动 O/C 恢复

此前 auto-grasp enabled 时曾阻断手动 Right OPEN/CLOSE。已将该额外阻断逻辑去掉，使手动 O/C 继续走原有 gripper_command 分支；自动 open_issued / close_issued latch 不因手动动作而重置。

| 行为说明  如果 auto 状态仍为 OPEN_TRACKING 且 close_issued=False，手动 C 可以物理闭合，但之后若自动视觉条件满足，控制器仍可能再下发一次自身的 CLOSE 目标。 |
| --- |

10.2 HOME 触发问题

多次日志中在 OPEN_TRACKING 后出现 [STREAM START] HOME。HOME 运动会使视觉误差突然变化，Right Wrist jitter 曾从约 0.01-0.05 突增到 0.35 左右，从而必然打断 0.40 s CLOSE 稳定窗口。该问题与 Right Wrist ev gate 是独立问题。

| 当前状态  HOME 触发原因尚未完成源码检查和修改。后续应单独定位 HOME 触发条件，避免 OPEN_TRACKING 稳定对准阶段被非预期 HOME 打断。 |
| --- |

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

| # VisionPYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv_sam2/bin/python -u \vision_dual_camera/task6_threading/dataset_tools/wire_unet/live_visual_guidance_shadow_v1.py# FusionPYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv_sam2/bin/python -u \vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/wire_grasp_fusion_shadow_v1.py# GatePYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv_sam2/bin/python -u \vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/wire_grasp_control_gate_v1.py --enable-output |
| --- |

12.2 Auto-Grasp Controller

| R1A7_AUTO_GRASP_ENABLE=1 \R1A7_AUTO_GRASP_PREOPEN_NORM_MAX=0.70 \R1A7_AUTO_GRASP_CLOSE_AXIS_NORM_MAX=0.10 \PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv/bin/python -u \transfer_control/experimental/r1a7_cartesian_stream_control_kd30_posvel_wu_dqik_u010_auto_grasp_direct_v1.py \--interface enp6s0 --domain-id 0 --host-ip 192.168.123.223 \--ik-frequency 30 --publish-frequency 250 --max-joint-speed 2.0 \--cartesian-jog-speed-mm-s 10 --cartesian-jog-accel-mm-s2 20 \--gripper-initial-mode current --no-right-gripper-open-cap-current --no-record-d435i |
| --- |

12.3 语法检查模板

| PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv_sam2/bin/python -m py_compile <vision_script.py>PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv/bin/python -m py_compile <controller.py> |
| --- |

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

| 例如：EXT   eu=+0.023 ev=-0.035 jit=0.016   -> 合格WRIST eu=-0.026 ev=+0.555 jit=0.014   -> 修改后合格（ev 被忽略）若持续 >= 0.40 s：[AUTO GRASP DIRECT] CLOSE_ONCE target=-0.05000state=WAIT_CONTACT, close=1 |
| --- |

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

| 当前推荐基线  模型层保持 External/Left=V2、Right Wrist=V3；控制层只取消 Right Wrist ev CLOSE gate，其余阈值不动；之后单独验证双尖端 midpoint TIP 与 HOME 行为。 |
| --- |

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
