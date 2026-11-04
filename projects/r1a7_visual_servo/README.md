# R1-A7 双相机视觉伺服

## 目标

完成相机内参、外参和手眼标定，检测细线、孔和目标位姿，并为后续视觉闭环控制提供输入。

## 代码位置

- 单相机视觉：`r1a7_threading_project/vision/`
- 双相机视觉：`r1a7_threading_project/vision_dual_camera/`
- 标定结果：`r1a7_threading_project/calibration/`
- 相机配置：`r1a7_threading_project/config/cameras.yaml`
- 阶段记录：`docs/r1a7_camera_*.md`
- 当前启动栈：`r1a7_threading_project/run_task6_visual_stack.sh`
- 2026-10-10 交接：`r1a7_threading_project/docs/r1a7_visual_servo_control_handoff_2026-10-10_zh.md`
- 2026-10-18 实验与控制台更新：`r1a7_threading_project/docs/r1a7_experiment_console_update_2026-10-18_zh.md`
- 双相机夹线研究：`r1a7_threading_project/vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/`
- 左腕 007Z：`r1a7_threading_project/vision_dual_camera/task7_wire_threading/left_wrist/`
- 2026-10-26 衔接记录：`r1a7_threading_project/docs/r1a7_wire_grasp_left_wrist_sync_2026-10-26_zh.md`
- 2026-10-27 指标协议：`r1a7_threading_project/docs/r1a7_wire_threading_metrics_protocol_2026-10-27_zh.md`

## 当前状态

2026-10-10 的 Wrist Hole/Tip 跟踪、视觉状态机和低速控制衔接工作已进入统一工作线。控制台执行门采用分层上限：双目 `10 mm/s`、单目降级 `2 mm/s`、短时盲行 `0.5 mm/s`；上限是安全裁剪，不是独立 TCP 实测速度。记录确认的穿孔成功基线为 10 mm/s；15/25 mm/s 缺少成功/失败标签，仍需在修正后的状态机上重新做真机资格验证。

标定 JSON 和少量复核结果可以提交；原始标定图像和视频应按数据清单管理。

双相机夹线栈与上述穿孔视觉执行门是不同研究入口：前者当前默认 Shadow，速度文件上限
为 10 mm/s，未完成真机抓线；左腕 007Z 也保持感知-only。不得混用两条工作线的验证结论。

视觉评价不能只报告像素误差或 valid 帧比例。正式研究还需独立真值、端到端延迟、
实际末端误差和穿孔试次成功率；协议中的 `0.05/0.10 mm` 等均为待验证候选门槛。


## 2026-11-02 三相机 U-Net 研究补充

- [核对报告](../../r1a7_threading_project/docs/r1a7_four_document_github_audit_2026-11-02_zh.md)
- [U-Net实验模块](../../r1a7_threading_project/vision_dual_camera/task6_threading/dataset_tools/wire_unet/README.md)
- [原始记录与控制差异快照](../../r1a7_threading_project/docs/source_records/2026-11-02/README.md)

三相机研究与原控制台默认入口分别管理。11/02文档记载发生过运动但未验收；本次未运行硬件，Fusion/Gate/控制器补丁仅归档。

## 2026-11-04 Right Wrist OPEN

[归档报告](../../r1a7_threading_project/docs/r1a7_right_wrist_open_autograsp_archive_2026-11-04_zh.md)保存11/03流程、技术方案、采集标注训练评估与操作记录。当前研究前端采用External/Left V2、Right Wrist V3及mask形态midpoint；控制器只归档，不替换现用控制台。
