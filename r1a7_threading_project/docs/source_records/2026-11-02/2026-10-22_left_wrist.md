# 2026-10-22_left_wrist 原始记录全文转录

来源：[原始 DOCX](../2026-10-20_to_2026-10-22/2026.10.22_左腕相机工作记录.docx)。SHA-256：`b149bf6b1cf82632dde0f6fd9ee78cfd71b1ba6420d3dee2d9bcd8d85a5fbda2`。

历史记录中的命令、建议和待办仅供追溯，不是本次操作授权。全文转录保留表格与嵌入图片；原始版式以 DOCX 为准。

左腕 007Z 复用外部相机细线与镊尖识别工作记录

工作范围：识别流程核对、左腕迁移、镊尖标定与故障诊断

整理日期：2026 年 10 月 22 日。本文件仅整理本次对话中与左腕 007Z 镊尖识别、细线识别流程迁移、外部相机识别逻辑分析以及左腕镊尖标定有关的工作。

## 1. 工作目标与当前结论

目标是在左腕 Orbbec 007Z 相机上复用外部相机程序的镊尖候选生成和细线检测流程，同时保留左腕相机自己的 RGB 取流、相机内参和左腕专用码标—镊尖偏移标定。

截至本次最后一次源码核验，左腕脚本的修改命令返回 PATCH OK，Python 语法编译通过，旧 SAM2 掩膜门控相关关键词检查无输出。源码层面的替换已完成；但成功修改后的全新运行日志尚未在最后一次补丁之后提供，因此实际运行效果仍待重新启动验证。

## 2. 外部相机原程序与实际检测模块

| 项目 | 对话中确认的内容 |
| --- | --- |
| 外部相机主程序 | vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/external_wire_grasp_guidance_v1.py |
| 细线检测器 | stable_wire_detector.py 中的 WireLocalDetectorV65 |
| 镊尖候选算法 | task6_external_ibvs_v4_aruco_edge.py 中的 V4 镊尖候选生成逻辑 |
| 细线初始化 | 人工点击真实细线自由端，作为检测器 seed tip |
| 逐帧检测 | 每帧调用 wire.detect(frame, active_roi, ...) |
| ROI 管理 | roi_from_tip、roi_from_path、expand_roi |
| 外部相机设备 | 原程序使用 /dev/video16；该设备取流设置不迁移到左腕 |


人工点击自由端是初始化检测器的种子点，不表示把整条细线永久固定为一组像素坐标。原程序仍然逐帧检测当前图像，根据检测结果更新 ROI；候选丢失时会扩大搜索范围。

## 3. 外部细线识别流程与参数

## 3.1 检测器参数

| 参数 | 外部程序原值 | 用途说明 |
| --- | --- | --- |
| local_length_px | 200.0 | 细线局部检测长度参数 |
| grasp_offset_px | 60.0 | 抓取目标偏移参数 |
| grasp_band_start_px | 40.0 | 抓取段起始参数 |
| grasp_band_end_px | 110.0 | 抓取段结束参数 |


早期左腕脚本使用 180 / 55 / 35 / 90，与外部原程序不一致。最后一次成功补丁输出确认，左腕两个 WireLocalDetectorV65 初始化位置都改为 200 / 60 / 40 / 110。

## 3.2 细线检测调用与掩膜门控

外部程序实际采用的调用形式为：

wire_result = wire.detect(    frame,    active_roi,    tweezer_exclusion_mask=None,    tweezer_tip=tip_pixel,)

此前左腕程序额外初始化 InsertionTipTracker，并要求 SAM2 镊子排除掩膜有效后才执行细线检测；掩膜无效时输出 TWEEZER_MASK_REQUIRED。这会直接阻断细线检测。最后一次补丁移除了该跟踪器依赖、掩膜有效性门控和相关状态字段，恢复无掩膜调用，并将左腕镊尖坐标作为 tweezer_tip 传入。

无掩膜调用不再依赖 SAM2 掩膜是否可用，但镊子边缘可能被检测为细线候选，因此仍需用真实画面确认中心线与自由端落在细线上。

## 3.3 ROI 与丢失恢复

| 阶段 | 外部程序行为 |
| --- | --- |
| 初始 ROI | roi_from_tip(seed, frame.shape)，默认 half_w=180、half_h=180，典型尺寸 360×360 px（边界处会裁剪）。 |
| 检测成功 | 通过 roi_from_path(centerline, frame.shape) 根据当前中心线更新搜索区域。 |
| 检测丢失 | 围绕最近可信 anchor_roi 调用 expand_roi；扩张量随 miss_count 增加，最多额外扩张 60 px。 |
| 重新初始化 | 重新点击自由端后重建检测器、设置 seed，并重置 ROI 与 miss_count。 |


因此外部程序采用“人工初始化 + 每帧局部检测 + 动态 ROI + 丢失恢复”，并非初始化后停止检测。

## 4. 镊尖识别算法与门控调整

## 4.1 候选生成流程

程序检测 ArUco ID 20，计算码标姿态，再结合码标到物理镊尖的三维偏移生成镊尖像素候选。V4 返回 tip_uv、branch_id、quality 等信息。V4 候选生成与最终 ExternalTipGate 输出是两个独立阶段：候选生成成功不等于最终跟踪一定有效。

诊断期间出现过 CANDIDATE_OK（质量分数约 90）但最终 tip_gate_status=REJECT / NO_TRACK 的情况，说明部分失败发生在候选门控阶段；同时也出现 MARKER_FOUND ... candidates=0，说明 V4 候选生成本身仍可能间歇性失败。

## 4.2 取消严格跳变门控

为避免初始锁定因候选短暂缺失而反复重置，ExternalTipGate.update() 后来改为直接接受有效、有限的 V4 tip_uv 测量，不再要求连续三帧确认，不再以空间跳变或分支切换拒绝候选，也不使用 Kalman 预测值覆盖当前测量。若 V4 没有候选，程序仍返回无有效测量，不伪造坐标。

| 日志表现 | 说明 |
| --- | --- |
| INIT_WAIT 1/3 / NO_TRACK | 旧流程处于初始等待阶段，尚未建立跟踪。 |
| tweezer_tip=(700, 524), tip_gate_valid=True | 门控调整后成功输出当前 V4 候选。 |
| guidance=True、wire_valid=True、tip_valid=True、grasp_valid=True | 曾经有多帧同时获得有效镊尖、细线自由端和抓取目标。 |
| 后续 v4_tip 出现大幅坐标跳变 | 取消跳变门控后，异常候选也可能被直接接受，需要验证静止时的候选稳定性。 |


取消门控改善了候选输出的可用性，但不会自动提高 V4 候选的几何准确度。若码标姿态或局部精修生成错误候选，直接接受也会输出错误坐标。当前应先做纯视觉验证。

## 5. 左腕 007Z 镊尖标定工作

## 5.1 标定目的

码标中心不是物理镊尖。标定的目标是得到 P_tip_marker，即真实镊尖相对于 ArUco 标记坐标系的三维偏移。程序结合码标姿态、相机内参和该偏移，将真实镊尖投影到左腕图像的像素坐标。

左腕的投影计算使用左腕相机内参和左腕图像坐标。码标—镊尖三维偏移的使用前提是工具安装方式与坐标定义一致，并需要通过左腕图像样本独立验证。

## 5.2 标定过程

使用脚本 vision_dual_camera/task7_wire_threading/left_wrist/calibrate_left_wrist_007z_marker_tip.py。

相机锁定 CPCBC530007Z，图像 1280×800、30 FPS；采样时要求 ID 20 码标和真实物理镊尖同时可见。

在不同工具/腕部视角下人工点击真实镊尖；按 N 采集下一组样本，U 撤销上一组，Q 拟合并保存候选。

首次只采集 7 组样本，少于至少 8 组的要求，程序中止且没有保存候选。

后续采集 14 组样本。原始拟合中第 5 组误差约 62.99 px，明显偏离其他样本；另存了排除第 5 组后的候选文件。

## 5.3 当前左腕候选标定

| 项目 | 当前记录值 |
| --- | --- |
| 标定文件 | calibration/left_wrist_007z_marker_tip_calibration_candidate_no_sample5.json |
| 相机序列号 | CPCBC530007Z |
| tip_marker_m | [0.0001639940, -0.0868431440, -0.0258313063] m |
| tip_marker_mm | [+0.164, -86.843, -25.831] mm |
| coordinate_rms | 约 2.792 px |
| point_rms | 约 3.948 px |
| median point error | 约 3.833 px |
| maximum point error | 约 5.381 px |
| 状态 | CANDIDATE_REQUIRES_VALIDATION，尚未独立验收 |


排除第 5 组后的候选文件另存，没有覆盖原始候选文件。当前左腕主程序读取 no_sample5 候选。拟合误差是重投影拟合指标，不能单独证明真实镊尖在所有姿态下都准确。

## 5.4 左腕相机内参

K = [[607.7092, 0.0, 640.3098],     [0.0, 607.6644, 404.9614],     [0.0, 0.0, 1.0]]dist = [-0.029733, 0.033434, -0.000169, -0.000187, -0.011704]

当前内参来源记录为 vision_dual_camera/handeye/collect_handeye_samples.py。

## 6. 左腕程序迁移与最后核验状态

| 检查项 | 最后一次源码核验结果 |
| --- | --- |
| 目标程序 | vision_dual_camera/task7_wire_threading/left_wrist/left_wrist_007z_wire_perception_v1.py |
| 相机取流 | 保留 Orbbec SDK、序列号 CPCBC530007Z、RGB 1280×800@30 FPS。 |
| 镊尖候选 | 调用外部 V4 get_v4_tip，但传入左腕 K、dist 和左腕 P_tip_marker。 |
| 细线检测器 | 调用 ext.WireLocalDetectorV65；外部模块来自 stable_wire_detector.py。 |
| 检测参数 | 两个初始化位置均为 200 / 60 / 40 / 110。 |
| SAM2 门控 | 补丁成功移除；旧依赖关键词检查无输出。 |
| 抓取目标 | target 从 wire_result.get('grasp_target_uv') 读取。 |
| 动态 ROI | 恢复 ext.roi_from_path 与 ext.expand_roi。 |
| 语法 | python -m py_compile 通过。 |
| 运行时验收 | 最后一次成功补丁后的全新启动日志尚未提供；需要重新启动确认实际行为。 |
| 运动控制 | 程序保持纯视觉感知，不发送机器人运动指令。 |


## 7. 后续验证步骤

重新启动左腕程序，确认启动日志不再出现 InsertionTipTracker、SAM2 model loading 或 TWEEZER_MASK_REQUIRED。

确认相机序列号为 CPCBC530007Z，且日志显示加载左腕专用标定文件。

人工点击真实细线自由端，检查绿色中心线、红色自由端和抓取目标是否与真实细线一致。

让镊子和细线静止，连续观察 5–10 秒，记录 tip、wire_tip_uv、grasp_target_uv 和有效状态。

再缓慢移动镊子，观察 V4 候选是否异常跳变。取消跳变门控后，异常候选可能被直接接受，因此该步骤只做视觉测试。

只有镊尖和细线输出经过独立验证后，才考虑后续控制接入；当前程序不发送运动命令。

## 8. 关键文件清单

| 用途 | 路径 |
| --- | --- |
| 左腕主程序 | vision_dual_camera/task7_wire_threading/left_wrist/left_wrist_007z_wire_perception_v1.py |
| 左腕镊尖标定工具 | vision_dual_camera/task7_wire_threading/left_wrist/calibrate_left_wrist_007z_marker_tip.py |
| 左腕候选标定 | calibration/left_wrist_007z_marker_tip_calibration_candidate_no_sample5.json |
| 外部相机识别源程序 | vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/external_wire_grasp_guidance_v1.py |
| 外部细线稳定检测器 | vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/stable_wire_detector.py |
| V4 镊尖候选模块 | vision_dual_camera/task6_threading/task6_external_ibvs_v4_aruco_edge.py |
| 左腕运行状态 JSON | /tmp/r1a7_left_wrist_007z_external_migration/left_wrist_007z_perception.json |
| 左腕运行日志 | /tmp/r1a7_left_wrist_007z_external_migration/logs/left_wrist_007z_perception.log |


记录说明：本文依据本次对话中出现的源码片段、终端输出和标定数值整理。“源码修改成功”与“实际运行通过”分开记录；未提供新运行日志的事项不视为已完成运行验收。
