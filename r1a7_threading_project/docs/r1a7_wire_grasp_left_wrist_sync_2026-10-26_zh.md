# R1-A7 双相机夹线与左腕 007Z GitHub 衔接记录

日期：2026-10-26

来源：2026-10-20 至 2026-10-22 五份 Word 工作记录及本机研究目录

目标分支：`codex/r1a7-console-camera-suite-2026-10-11`

## 结论

同步前，当前 GitHub 分支没有双相机夹线研究目录、左腕 007Z 独立程序及相应候选标定。
本次已选择性补齐源码、小型标定结果和原始记录，同时保留以下事实边界：

- 双相机夹线仅有感知、Shadow 融合和受门控的速度文件输出，没有本次真机运动证据；
- 左腕 007Z 主程序是 RGB 感知程序，明确禁止发送运动命令；
- 夹爪自动闭合文档是工作方案，不是已实现功能；
- 所有带 `candidate` 的标定仍是候选，不能写成正式标定；
- 文档内的下一步操作仅作为历史记录，不作为本次同步指令。

## 已纳入的工作

### 双相机夹线

路径：
`vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/`

纳入 52 个算法/运行 Python 文件和两个启动脚本，包括：

- 右腕 V69 细线与镊尖感知；
- 外部相机 V1 和 V2 candidate 引导；
- W5 至 W65 检测器演进链和稳定包装器；
- 腕部状态桥、双相机 Shadow 融合和控制门；
- `DUAL`、`WRIST_ONLY`、`EXTERNAL_ONLY` 三种有效模式；
- `0.20 s` 新鲜度门和 `10 mm/s` 三维合速度上限；
- 便携式项目根路径和默认持久日志目录。

启动器默认是 Shadow。`--enable-motion` 只允许控制门写出有效速度文件，仍不代表机器人驱动
已经取得控制权，也不绕过机器人侧 deadman。

### 左腕 007Z

路径：
`vision_dual_camera/task7_wire_threading/left_wrist/`

纳入：

- `calibrate_left_wrist_007z_marker_tip.py`；
- `left_wrist_007z_wire_perception_v1.py`；
- `left_wrist_007z_hole_tracker.py`。

主程序按序列号 `CPCBC530007Z` 使用 Orbbec SDK 采集 `1280x800@30` RGB，关闭深度流。
当前候选标定为 13 个样本，坐标 RMS 约 `2.792 px`、点 RMS 约 `3.948 px`、中位误差约
`3.833 px`、最大误差约 `5.381 px`。

程序仍使用 `collect_handeye_samples.py` 中标记为右腕来源的内参，且状态输出显式保持
`grasp_guidance_valid=false`、`motion_enabled=false`、`motion_command_sent=false`。这是需要后续
复测和设计确认的安全边界，不在本次文档同步中擅自开放。

### 标定和原始记录

补入 11 个小型 JSON，包括左腕孔标定、左腕镊尖候选、外部镊尖多个候选及独立离线验证结果。
原始 Word 文件以规范化文件名保存于：
`docs/source_records/2026-10-20_to_2026-10-22/`。

## 未纳入和未宣称完成的内容

- 未导入 `.bak`、`__pycache__`、调试截图、临时输出和运行日志；
- 未导入 8 个依赖本机测试图片与 `wire_init.json` 的一次性 `test_*.py` 探针；
- 未实现“细线进入夹口后自动闭爪”的状态机；
- 未运行相机、机械臂、夹爪或 LowCmd；
- 未把历史 Shadow 日志当作真机执行成功证据。

## 本次验证

- 55 个新增 Python 文件（双相机 52、左腕 3）全部通过 `py_compile`；
- 两个 Bash 启动脚本通过 `bash -n`；
- 候选 JSON 全部通过 JSON 解析；
- 仓库既有控制台测试需在最终提交前再次运行；
- Word 内容已完成结构化提取。由于当前文档渲染运行时与系统 GLIBC/GLIBCXX 不兼容，
  本次未能完成 LibreOffice 页面渲染，因此不声称做过版面视觉验收。

## 下一步资格门

1. 左腕 007Z 先确认真实内参来源并做独立重投影验证；
2. 运行左腕主程序，保存补丁后的首个持久日志和代表帧；
3. 双相机先运行 Shadow，连续验证时间戳、模式切换和坐标方向；
4. 只在唯一 LowCmd 发布者、急停、起始姿态和驱动 deadman 都确认后，才考虑短时速度输出；
5. 自动闭爪必须先定义并验证 `wire_inside_jaws`，默认保持未实现。
