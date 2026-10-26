# R1-A7 双相机夹线视觉引导

本目录同步 2026-10-20 至 2026-10-22 的细线、镊尖、双相机融合与控制门研究源码。
它是研究入口，不代表真机抓线已经完成。

## 当前入口

- 右腕识别：`right_wrist_wire_tweezer_v69_nomask.py`
- 外部相机引导：`external_wire_grasp_guidance_v1.py`
- 外部相机候选版：`external_wire_grasp_guidance_v2_candidate.py`
- 腕部引导桥：`wrist_wire_grasp_guidance_bridge.py`
- 双相机融合：`wire_grasp_fusion_shadow_v1.py`
- 控制输出门：`wire_grasp_control_gate_v1.py`
- 稳定栈启动器：`run_dual_camera_wire_grasp_stack.sh`
- 候选栈启动器：`run_dual_camera_wire_grasp_stack_candidate.sh`

## 安全默认值

不带参数启动时为 Shadow：

```bash
cd /path/to/repository/r1a7_threading_project/vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire
./run_dual_camera_wire_grasp_stack.sh
```

只有显式增加 `--enable-motion`，控制门才可能在感知有效时写出 `enabled=true`：

```bash
./run_dual_camera_wire_grasp_stack.sh --enable-motion
```

该参数只开放 `/tmp/r1a7_dual_ibvs_velocity.json` 的输出许可；机器人侧驱动、控制权、
deadman、急停和唯一 LowCmd 发布者仍须单独确认。不得把“文件输出有效”写成“真机已经运动”。

控制门当前边界：

- 支持 `DUAL`、`WRIST_ONLY`、`EXTERNAL_ONLY`；
- `DUAL_Z_CONFLICT` 和其他状态停机；
- 输入最大年龄 `0.20 s`；
- 三维合速度上限 `10 mm/s`；
- 不自动闭合夹爪；
- 不启用盲行。

日志默认写入 `r1a7_threading_project/r1a7_logs/wire_grasp/`。可通过
`R1A7_WIRE_LOG_DIR` 改写，Python 可通过 `R1A7_WIRE_PYTHON` 改写。

## 标定边界

稳定外部引导当前选择
`calibration/tweezer_marker_tip_calibration_xminus85_zminus25_branchreproj_candidate.json`。
文件名中的 `candidate` 是状态的一部分：它只能作为实验候选，不能视为正式标定。

## 验证状态

- 2026-10-26：52 个 Python 源文件语法编译通过，两个启动脚本通过 Bash 语法检查。
- 文档记录过 Shadow 输出中的 `EXTERNAL_ONLY_READY` 和 `DUAL_READY`。
- 未在本次同步中启动相机、写入机器人控制或执行夹爪动作。
- 原研究目录的 `test_*.py` 是依赖本机图片和 `wire_init.json` 的临时探针，未纳入本目录；
  它们不能作为可复现单元测试。
