# R1 A7 双相机视觉伺服 GitHub 上传清单

上传日期：2026-09-29
发布分支：`codex/r1a7-dual-camera-visual-servo-2026-09-29`
发布基线：`codex/r1a7-threading-project-complete-2026-09-22`

## 1. 本次新增和更新

### 文档

- `docs/r1a7_dual_camera_visual_servo_work_summary_2026-09-29_zh.md`
  - 双相机架构、阶段工作、实验结果、问题分析、运行命令、风险和下一步。
- `docs/source_records/R1-A7双相机视觉伺服开发工作整理汇总.docx`
  - 用户提供的 2026-09-29 阶段汇总原文。
- `docs/source_records/R1_A7双相机视觉伺服开发全过程实验操作记录_20260924-20260928.docx`
  - 用户提供的 2026-09-24 至 2026-09-28 实验操作原文。

### 当前活动视觉代码

- `vision_dual_camera/task6_threading/task6_external_ibvs_v4_aruco_edge.py`
- `vision_dual_camera/task6_threading/task6_wrist_ibvs_controller_v2_tip.py`
- `vision_dual_camera/task6_threading/task6_dual_ibvs_fusion.py`
- `transfer_control/tools/r1a7_cartesian_stream_control.py`
- `vision_dual_camera/task6_threading/perception/tweezer_tip/sam2_pca_detector.py`
- `vision_dual_camera/task6_threading/perception/tweezer_tip/tip_refiner_v2.py`
- `vision_dual_camera/task6_threading/perception/tweezer_tip/test_external_aruco_local_tip_refine_v2.py`
- `vision_dual_camera/task6_threading/perception/tweezer_tip/test_external_aruco_local_tip_refine_v3_hybrid.py`
- `vision_dual_camera/task6_threading/perception/tweezer_tip/test_external_aruco_local_tip_refine_v4_fusion.py`

### 标定和参数

- `calibration/tweezer_marker_tip_calibration.json`
- `calibration/tweezer_marker_tip_calibration_refined.json`
- `calibration/tweezer_marker_tip_calibration_median30.json`
- `vision_dual_camera/task6_threading/wrist_tip_calibration.json`

### 审查补充文件

- `aruco_id20_5x5_50.png`
- `vision_dual_camera/task6_threading/perception/tweezer_tip/test_wrist_tip_tracking_v2.py`
- `vision_dual_camera/task6_threading/perception/tweezer_tip/tip_temporal_filter.py`
- `vision_dual_camera/task6_threading/perception/tweezer_tip/calibrate_aruco_marker_to_tip.py`
- `vision_dual_camera/task6_threading/perception/tweezer_tip/calibrate_aruco_marker_to_tip_median30.py`
- `vision_dual_camera/task6_threading/perception/tweezer_tip/refine_marker_tip_pixel.py`
- `vision_dual_camera/task6_threading/test_external_fk_tip_projection.py`

详细审查结果见 `docs/r1a7_dual_camera_visual_servo_upload_audit_2026-09-29_zh.md`。

## 2. 不上传内容

- `third_party/sam2/checkpoints/*.pt`：模型 checkpoint 体积大，且属于外部模型文件；文档只保留本机路径和使用说明。
- `third_party/sam2/`：完整第三方源码、示例数据和其内部 Git 历史不纳入本次快照；运行时应在 `tv_sam2` 环境中安装 SAM2，或按本机路径配置。
- `perception/**/output/`：实验图片、CSV、临时 JSON 和调试输出。
- `__pycache__`、`.pyc`、`*.log`、`Log/`：运行生成物。
- `*.bak*`、`*.backup*`、`*before*`、`*legacy*`：历史副本，避免误用旧版本。
- `data/`、视频和录制数据：体积大且可能含现场数据；只在文档中记录路径和结果摘要。

## 3. 环境要求

推荐复用本机已有环境：

```text
/home/robot/miniconda3/envs/tv_sam2/bin/python
```

主要依赖：Python、OpenCV 4.10 系列、NumPy、PyTorch、SAM2、`pyorbbecsdk`、Unitree DDS/控制依赖。不要用系统 Python 替代 `tv_sam2`，否则可能遇到 `ArucoDetector` API 不兼容。

SAM2 checkpoint 需要在本机准备：

```text
r1a7_threading_project/third_party/sam2/checkpoints/sam2.1_hiera_small.pt
```

该文件未上传。复现实验时应先确认 checkpoint 存在，并在文档记录的本机路径下运行。

## 4. 复现前检查

```bash
cd ~/unitree_sim_isaaclab_threading/r1a7_threading_project
test -f third_party/sam2/checkpoints/sam2.1_hiera_small.pt
sha256sum vision_dual_camera/task6_threading/task6_wrist_ibvs_controller_v2_tip.py
python -m py_compile \
  vision_dual_camera/task6_threading/task6_external_ibvs_v4_aruco_edge.py \
  vision_dual_camera/task6_threading/task6_wrist_ibvs_controller_v2_tip.py \
  vision_dual_camera/task6_threading/task6_dual_ibvs_fusion.py \
  transfer_control/tools/r1a7_cartesian_stream_control.py
```

## 5. 发布边界

本次提交是“当前工作状态归档”，不是重新开发功能。代码按本地工作区当前活动版本复制；未对控制算法做额外修改。真机运行前必须再次检查相机设备、输出 JSON、速度上限、`rt/lowcmd` 控制锁和急停路径。
