# 左腕奥比中光 007Z 采集与细线识别

本目录是 2026-10-22 左腕相机工作的独立实现。它使用 `pyorbbecsdk` 按序列号
`CPCBC530007Z` 采集 `1280x800@30` RGB，深度流关闭。

## 程序

- `calibrate_left_wrist_007z_marker_tip.py`：左腕 Marker ID 20 到镊尖的候选标定。
- `left_wrist_007z_wire_perception_v1.py`：迁移外部相机细线/镊尖逻辑的主程序。
- `left_wrist_007z_hole_tracker.py`：左腕孔跟踪实验程序。

从项目根目录运行识别：

```bash
cd /path/to/repository/r1a7_threading_project
PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv_sam2/bin/python \
  vision_dual_camera/task7_wire_threading/left_wrist/left_wrist_007z_wire_perception_v1.py
```

## 当前边界

- 主程序读取
  `calibration/left_wrist_007z_marker_tip_calibration_candidate_no_sample5.json`；
- 该候选含 13 个样本，坐标 RMS 约 `2.792 px`、点 RMS 约 `3.948 px`；
- 相机内参仍来自 `collect_handeye_samples.py` 中的右腕标定，必须重新确认左腕内参来源；
- 标定尚未独立验证；
- 程序固定写出 `motion_enabled=false` 和 `motion_command_sent=false`；
- 状态中的 `grasp_guidance_valid` 保持 `false`，因此不能直接接入机械臂执行。

文档只记录了源码补丁和 `py_compile` 通过，没有补丁后的新实机运行日志。本目录当前只能标记为
“代码已同步、运行待复测”，不能标记为左腕识别或抓线真机通过。
