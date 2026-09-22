# R1-A7 细线镊子穿孔项目完整快照

更新时间：2026-09-22
项目目录：`r1a7_threading_project`
快照分支：`codex/r1a7-threading-project-complete-2026-09-22`

## 1. 快照目标

本分支保存 R1-A7 细线镊子穿孔项目截至 2026-09-22 的可复现工程内容，包括：

- 镊子与夹爪绑定、夹爪控制和阶段性实验记录；
- 右臂 Cartesian、IK、LowCmd 和本地 3D 控制台；
- 头部/外部相机与腕部 Eye-in-Hand 相机视觉引导；
- 右腕手眼标定、TCP/P_tip 标定和固定相机标定资料；
- 细线、镊尖、孔检测与穿孔状态机的实施方案；
- 研究方案、PPT 资源、运行说明和五份 Word 原始工作记录。

## 2. 模块索引

### 2.1 控制与控制台

- `transfer_control/tools/`：真机 Cartesian、视觉接收和 VR/LowCmd 控制入口。
- `transfer_control/experimental/`：网页整行命令兼容控制副本及历史分层 IK 实验入口。
- `console3d/`：8097 端口的完整 3D 手臂控制台。
- `console3d_hierarchical/`：8098 端口的当前稳定基线控制台。
- `console3d/model/`：R1-A7、Dex1-1 URDF/STL 模型。

控制链统一为：

```text
网页/视觉速度建议 → Cartesian 目标 → IK → 250 Hz LowCmd → R1-A7
```

同一时间只能有一个 `rt/lowcmd` 发布者。网页控制台、视觉控制和终端控制不能同时 ENABLE。

### 2.2 镊子、夹爪与任务配置

- `scripts/`：夹爪状态读取、方向探测、夹爪/镊子标定和闭合测试。
- `config/`：相机、夹爪镊子标定和任务阶段配置。
- `docs/gripper_tweezer_binding_notes_zh.md`：镊子绑定和开度记录说明。

### 2.3 视觉引导

- `vision/`：头部相机工具跟踪、局部 Jacobian、Homography 和单相机视觉伺服。
- `vision_dual_camera/fixed_camera/`：固定外部相机内参/外参和工作平面标定。
- `vision_dual_camera/handeye/`：右腕 Eye-in-Hand 手眼采样、求解、LOO 和实时验证。
- `vision_dual_camera/task6_threading/`：外部相机 ArUco、腕部孔/镊尖跟踪、双相机速度融合。
- `tool_calibration/tcp_pivot/`：TCP/P_tip Pivot 采样、求解、冻结和实时验证。

当前双相机职责：

- 外部相机：大视野粗定位，主要生成 X/Z 速度；
- 腕部相机：镊尖与孔的局部精定位，主要生成 Y/Z 速度；
- 融合节点：合成为单一 XYZ Cartesian 速度文件，再交给唯一机器人控制器。

### 2.4 研究资料和原始记录

- `docs/threading_research_plan_2026-09-11_zh.md`：研究路线和论文复现规划。
- `docs/R1A7_镊子夹取细线与自主穿孔研究方案汇报_2026-09-11*.pptx`：阶段汇报 PPT。
- `docs/ppt_assets/`：汇报图片资源。
- `docs/source_records/`：从桌面归档的 Word 原始工作记录。

## 3. 标定工作结果

### 3.1 右腕 Eye-in-Hand 手眼标定

- 20 组 ChArUco 样本，35 个角点均稳定检测。
- Daniilidis 作为当前冻结结果。
- 离线 Position RMS：约 `3.680 mm`。
- LOO Position RMS：约 `4.086 mm`。
- Camera→Base 实时验证 Position RMS：约 `2.709 mm`。

### 3.2 TCP/P_tip Pivot 标定

- `P_tip_ee ≈ [104.805, 1.606, 105.459] mm`。
- 20 组样本，LOO 最大 P_tip 变化约 `0.851 mm`。
- 6 姿态实时验证 Position RMS 约 `4.284 mm`。
- 当前主要误差集中在 Base-Y 方向，不能直接等同于柔性细线线尖位置。

### 3.3 外部固定相机

固定相机 Pixel→Board→Base 平面标定路线已经确定，现有结果文件和采集资料已归档；外部相机标定仍需按文档说明完成独立验收，不能把规划结果当作已完成实验结果。

## 4. 视觉检测和穿孔操作路线

1. `SEARCH_LINE`：RGB/ROI 中分割细线，骨架化并提取自由端点。
2. `ALIGN_GRIPPER`：将镊尖对准线端，使用小步长闭环修正。
3. `GRASP`：闭合镊子并通过线端随动、夹爪状态或轻微抬升确认夹取。
4. `SEARCH_HOLE`：检测目标孔中心、尺寸和方向。
5. `ALIGN_HOLE`：将线端移动到孔前安全位置并完成 XY/姿态对准。
6. `INSERT`：沿孔轴低速插入，结合视觉、力矩/力传感和超时保护。
7. `RECOVER`：检测丢失、卡阻或置信度下降时回退并重新定位。

第一版建议先采用 SAM2/几何方法跑通闭环：

- 细线：Mask → Skeleton → Endpoint；
- 镊子：Mask → PCA/轮廓求尖端；
- 孔：ROI/轮廓 → 圆或椭圆拟合；
- 后续瓶颈明确后，再替换为 DLO Perceiver 或 Tip/Tail Heatmap CNN。

## 5. 当前边界

- Cartesian 连续控制和双相机速度接入链路已建立，但高位抬升后向前的 IK XYZ 优先级问题仍需单独解决。
- 单相机 R→T 视觉闭环已验证；双相机已具备基本穿孔对准能力。
- 当前固定 TCP 只适合刚性工具工作点。夹持柔性细线后，最终控制目标必须升级为实时 `P_hole - P_line-tip`。
- 最后几毫米不能依赖一次开环坐标换算，需要近距离 RGB 视觉闭环、低速插入和接触/柔顺保护。

## 6. 快照排除项

本分支不上传以下内容：

- `__pycache__`、`.pyc`、浏览器缓存和 Codex 临时目录；
- `r1a7_threading_project/Log/` 运行日志；
- 带 `before_*`、`baseline_*`、`backup`、`legacy` 等名称的历史脚本副本；
- 训练 checkpoint、视频和外部 conda/SDK 环境。

这些文件不影响源码复现，仍保留在本机工作区；活动代码、冻结标定结果、校准样本和研究资料保留在快照中。
