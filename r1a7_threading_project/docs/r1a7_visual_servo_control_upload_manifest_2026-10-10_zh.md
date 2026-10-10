# R1 A7 视觉伺服与低速控制上传清单

更新时间：2026-10-10

父分支：`codex/r1a7-dual-camera-visual-servo-2026-09-29`

父提交：`621e53d512f2318f77fffc11f673bc657dc72f56`

目标分支：`codex/r1a7-visual-servo-control-2026-10-10`

## 1. 上传范围

本分支是 2026-09-29 双相机视觉伺服归档的续接快照，上传以下内容：

- External、Wrist、Fusion 的 2026-10-10 活动版本。
- 三路视觉统一启动和日志 sidecar 脚本。
- 2026-10-07 低速平滑诊断脚本中的正式实验版本。
- 2026-10-08 右肩 pitch fully-buffered、gravity、Frozen-q 和 breakaway 诊断脚本。
- 2026-10-09 DQIK、anti-stiction、VELDEF、Kd20/30、POSVEL 和 W/U 实验版本。
- 7 份原始 Word 工作记录。
- GitHub 可直接阅读的中文接续说明和本清单。

## 2. 关键活动文件

### 2.1 视觉

- `run_task6_visual_stack.sh`
- `vision_dual_camera/task6_threading/task6_external_ibvs_v4_aruco_edge.py`
- `vision_dual_camera/task6_threading/task6_wrist_ibvs_controller_v2_tip.py`
- `vision_dual_camera/task6_threading/task6_dual_ibvs_fusion.py`

### 2.2 当前控制基线

- `transfer_control/experimental/r1a7_cartesian_stream_control_kd30_posvel_wu_dqik_u010_v1.py`

### 2.3 诊断与实验链

- `transfer_control/diagnostics/` 中无备份后缀的低速基线、肘部 governor、rate-matched、continuous、dither、Frozen-q 和分析脚本。
- `transfer_control/tools/r1a7_right_shoulder_pitch_*.py` 中与 10 月 8 日文档对应的 fully-buffered、gravity、Frozen-q、预定位和 AUTO_BREAKAWAY 脚本。
- `transfer_control/experimental/` 中与工作记录对应的 DQIK、AST V2/V3、VELDEF、Kd20/30、POSVEL 和 W/U/U-task 版本。

## 3. 原始记录与 SHA256

目录：`docs/source_records/2026-10-07_to_2026-10-10/`

| 文件 | SHA256 |
|---|---|
| `R1-A7_全对话工作内容汇总_20261009.docx` | `f6c4b21f6aaea054de7796809783b3a8e3588e80489bacb82f9f450c42fbc823` |
| `R1-A7_双臂人形机器人_低速平滑控制工作总结_2026-10-07.docx` | `fd0381a28f539c979e6593981cd176d924b08a69a9a1c0f6dac44a8573fcc4d8` |
| `R1-A7_右臂低速笛卡尔卡滞问题_工作总结_20261009.docx` | `5291c6d16b2d54ed2ac432fc37302b0bf16981d87bea26c371521779d55e601c` |
| `R1-A7_右臂低速运动卡顿诊断与控制优化阶段总结_2026-10-08.docx` | `19deb77fa6ffd506ec6ecbc703100e8e7b3558c4cab27e39ad4041f8e85d37b5` |
| `R1-A7_右臂抬起卡顿诊断与优化工作汇总_20261009.docx` | `15ed59a88fef8236c783706968c507dc87ff7f210a1a6b8ca89d8cdc69dac50e` |
| `R1-A7双相机视觉伺服开发工作整理汇总.docx` | `67ce059b2040482bb922a0a4fd14726c89dbf5d4087cc6a05118328ceea3de37` |
| `R1-A7双臂人形机器人视觉伺服调试与操作工作记录.docx` | `27c8719a4ebe32b28275a4c0ad309a8948ce22ac0ad63f704a69771d6347d32f` |

## 4. 明确排除

- `/tmp/r1a7_cartesian_trace_*.csv` 和 `/tmp/r1a7_visual_test_*` 现场日志。
- `r1a7_logs/`、运行时 JSON、缓存、`__pycache__` 和临时输出。
- `.bak`、`.backup`、`PRE_*` 和其它逐轮备份文件。
- SAM2 仓库、checkpoint、模型权重和外部 Conda 环境。
- 相机录制、训练数据、ACT 数据集和 checkpoint。
- 与本次视觉/控制工作无关的控制台前端修改和本地零散文件。

这些内容不适合直接进入 GitHub。若后续需要复现实验，应在本机按原始 Word 中的 trace 索引查找，或另行建立带路径、大小和哈希的数据清单。

## 5. 验证要求

归档提交前执行：

```bash
python -m py_compile <本清单中的 Python 文件>
bash -n r1a7_threading_project/run_task6_visual_stack.sh
git diff --check
git status --short
```

这些检查只验证语法、脚本结构和 Git 边界，不代表已重新运行真实机器人、相机或视觉伺服。
