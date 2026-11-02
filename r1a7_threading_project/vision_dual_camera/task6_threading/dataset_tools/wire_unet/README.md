# 三相机 U-Net 细线/工具研究模块

研究源码快照：2026-11-02，来自 threading 工作区。25个 Python 文件首次纳入 GitHub，本次只做静态检查，未运行训练、相机或控制。

## 阅读入口与模块关系

- 训练：`overfit_wire_unet.py`（U-Net定义与小样本拟合）、`train_wire_unet_v1.py`、`train_wire_unet_v2.py`、`train_wire_unet_v2_resume.py`。
- 离线评估：`eval_wire_*.py`；困难帧：`diagnose_*.py`、`analyze_wire_components_v2.py`。
- 三相机：`live_wire_unet_three_camera.py`；稳定性：`live_wire_stability_validation_three_camera.py`；时序：`live_wire_temporal_validation_v1.py`。
- FREE/GRASP：`live_free_grasp_validation_v1.py`、`live_manual_free_grasp_validation_v1.py`、`live_hybrid_locked_free_grasp_v1.py`。
- 镊尖依赖：`live_hybrid_locked_wire_live_tweezer_tip_v1.py` → v2 → v3；它们保留为被导入模块，并非三个可任意互换的最终入口。
- 后期视觉前端：`live_visual_guidance_shadow_v1.py` → v3 → v2 → v1 → base/temporal/manual。保持此目录结构。

## 环境与数据

文档记录 `tv_sam2`、Python3.10.20、torch2.3.1+cu121、OpenCV4.11.0、pyorbbecsdk2.1.1；实际部署还需检查模块的 NumPy/SciPy 等导入及对应环境，不把本次语法检查视为依赖安装验证。

脚本仍保留历史绝对路径 `/home/robot/r1a7_datasets/wire_unet/pilot_001` 和 checkpoint 位置；未改写已用参数。权重未随Git提交，见项目 `docs/source_records/2026-11-02/manifest.json`。

External=USB ZOOM 20685204b12d5283；Right Wrist=CPCBC53000C5；Left Wrist=CPCBC530007Z。以上是历史设备身份；节点和首帧需实时核对。阈值0.86、bridge24px、max-gap50px、tile512/stride384是实验配置，不是跨场景合格保证。

## 输出与验证边界

`live_visual_guidance_shadow_v1.py` 仍写共享 `/tmp/r1a7_external_wire_grasp_v1.json` 和 `/tmp/r1a7_wrist_wire_grasp_guidance.json`。名称含shadow不代表可无条件安全启动：若后台Gate已开放输出且控制器运行，新文件可能进入真实动作链。运行前必须确认消费者与输出状态。

本模块不是8098控制台默认感知替换；融合/控制差异仅归档在 `docs/source_records/2026-11-02/research_code_snapshot/`。不要把本分支原位Fusion/Gate的旧策略误认为11/02研究补丁已部署。

TIP→GRASP保留。FREE用于初始化和几何定向；不自动闭爪。Temporal HOLD不是新测量；长度跳变下降不等于尖端或工具定位精度达标。

完整核对、历史指标和待验收项见项目 `docs/r1a7_four_document_github_audit_2026-11-02_zh.md`。
