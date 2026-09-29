# R1 A7 双相机视觉伺服 GitHub 上传审查

审查日期：2026-09-29

## 1. 审查范围

本次将本机 `/home/robot/unitree_sim_isaaclab_threading/r1a7_threading_project` 与远程分支 `codex/r1a7-dual-camera-visual-servo-2026-09-29` 逐项对照，并重点核对两份阶段文档中出现的运行入口、标定文件、实验辅助程序、结果记录和原始资料。

## 2. 审查结论

### 已经上传的核心内容

- External V4、Wrist V2、Dual Fusion 和 Cartesian 控制入口；
- Tip 精修依赖和当前 Wrist Tip 标定结果；
- External 镊尖标记到 Tip 的三份标定结果；
- 手眼、固定相机、TCP/P_tip 等前一阶段冻结标定资料；
- 两份 2026-09-24 至 2026-09-28 的原始 Word 记录；
- 2026-09-29 工作汇总和上传清单。

### 本次确认遗漏并已补充

以下文件属于复现实验或标定链路，不是机器人运行时的必需入口，但保留它们可以让后续人员复现实验过程：

| 文件 | 用途 |
|---|---|
| `aruco_id20_5x5_50.png` | External 相机使用的 ArUco 5x5_50、ID 20 标记图 |
| `perception/tweezer_tip/test_wrist_tip_tracking_v2.py` | SAM2 + PCA + RefinerV2 腕部 Tip 独立跟踪实验入口 |
| `perception/tweezer_tip/tip_temporal_filter.py` | Wrist Tip 跟踪实验使用的时间滤波器 |
| `perception/tweezer_tip/calibrate_aruco_marker_to_tip.py` | ArUco 标记到镊尖初始标定 |
| `perception/tweezer_tip/calibrate_aruco_marker_to_tip_median30.py` | 30 帧中值镊尖标定 |
| `perception/tweezer_tip/refine_marker_tip_pixel.py` | 标定结果精修 |
| `task6_threading/test_external_fk_tip_projection.py` | 固定相机、手眼/TCP 与外部 Tip 投影的诊断验证 |

## 3. 有意不上传的内容

以下内容已检查，属于明确的排除项，不是漏传：

- `PRE_*`、`*.bak*`、`*_backup*`、`*before*` 和 `*legacy*` 历史版本；
- `perception/**/output/` 调试图片、CSV、临时 JSON；
- `Log/`、`*.log`、`__pycache__`、`.pyc`；
- `data/`、视频和现场录制文件；
- `third_party/sam2/checkpoints/*.pt` 模型权重；
- 完整 `third_party/sam2` 源码和其示例数据。

这些内容已经在上传清单中注明。SAM2 checkpoint 仍需在本机 `tv_sam2` 环境中准备，不能仅凭 GitHub 分支直接启动腕部 SAM2 跟踪。

## 4. 仍需注意的文件名差异

实验记录中曾写过 `task6_wrist_tip_tracking_v2.py`，本地实际复现实验入口位于：

```text
vision_dual_camera/task6_threading/perception/tweezer_tip/test_wrist_tip_tracking_v2.py
```

后续运行时应使用实际路径，避免因文档中的旧文件名造成“文件不存在”的误判。

## 5. 审查后的复现顺序

1. 先准备 `tv_sam2` 环境和本机 SAM2 checkpoint。
2. 用 `aruco_id20_5x5_50.png` 打印或显示 ID20 标记，并确认外部相机内参路径。
3. 需要重新标定时，依次使用初始标定、中值标定和精修脚本。
4. 先运行 `test_wrist_tip_tracking_v2.py` 验证 Tip，再运行 Wrist IBVS。
5. 外部相机的坐标链异常时，使用 `test_external_fk_tip_projection.py` 做诊断，不直接进入穿孔动作。
