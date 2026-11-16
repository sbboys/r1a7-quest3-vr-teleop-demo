# 11月16日研发操作与排查步骤

本文件是来源文档与当前源码核对后的操作记录，不代表本次已执行实验。历史硬件启动模板见[11/09 SOP](r1a7_grasp_insertion_sop_2026-11-09_zh.md)，当前模型及控制资格已变化，不能不核对版本就照抄。源资料的完整操作过程见[四文档入口](source_records/2026-11-16_research/README.md)。

## 1 固定实际版本

研究根目录：`/home/robot/unitree_sim_isaaclab_threading/r1a7_threading_project`；集成归档目录不同。先核对Git状态、四脚本hash与manifest、模型hash和环境。研究输出链的Fusion/Gate/controller归档在 `research_code_snapshot/current/`，没有部署覆盖控制台默认文件。

```bash
cd /home/robot/unitree_sim_isaaclab_threading/r1a7_threading_project
sha256sum vision_dual_camera/task6_threading/dataset_tools/wire_unet/live_grasp_to_insertion_visual_v3.py
rg -n 'WIRE_V4_CHECKPOINT|HOLE_CHECKPOINT|HOLE_MIN_AREA|INSERTION_CONTROL_CAMERAS' vision_dual_camera/task6_threading/dataset_tools/wire_unet/live_grasp_to_insertion_visual_v3.py
sha256sum /data/threading/new_board_dataset/frozen/models/wire_v5_newboard_seed44_epoch123.pt /data/threading/new_board_dataset/frozen/models/hole_v3_newboard_seed44_epoch64.pt
```

这只核对路径/字节，不能证明实际模型加载。不要自动调用完整视觉脚本或import训练脚本来“验证语法”，部分模块启动时有相机/推理/输出副作用。

## 2 数据与标注

旧会话197 records/591 images→hard analyzer→sync group dedup→35组105图；独立100图方案是中间历史版本。正式split为seed44的25/5/5，training_runtime的images/masks/masks_v3是兼容路径，不能把symlink当作额外数据。

NEW_BOARD_01采集用 `hole_unet/manual_newboard_hole_capture_v1.py`：S保存同步组、R切scene、U撤销、P统计、Q退出；当前归档只确认21组。NEW_BOARD_02没有完成证据，不因命令模板存在而新建或声称完成。

联合标注用 `wire_unet/annotate_insertion_joint_v1.py`。当前工具提示L=Wire、T=Tweezer、B=Board、R=参考孔、H自动孔、E手工孔；K保存，N/J前后，Z/X缩放、WASD平移、Q退出。旧文档早期数字键/保存键方案与现版有差异，以源码/UI为准。两张mask分别保存；真实遮挡不补线，镊臂间空气不填tool。复核旧联合104/105孔前景样本；新板audit_v2按最新key覆盖重复保存历史，不删除日志来凑行数。

## 3 训练 评估和冻结

Wire V4、V5与Hole V2、V3使用独立trainer，分别加载上代权重并按各自replay策略训练。固定split和GT版本，记录命令、环境、seed、guard与最佳epoch；test不用于训练/选模，评估后也不能继续称未观察。

原JSON可复核V4 epoch110、Hole V2 epoch52、V5 epoch123、Hole V3 epoch64。Hole V3 External test Hole Dice=0；先稳定灯光/目标身份、比较viewer与完整Visual面积阈值，不用“只留最大孔”掩盖错孔。模型冻结后保存hash、指标及数据归档包hash；文件本身外置。

## 4 记录器与试验目录

`insertion_dataset_recorder_v1.py` 已接入V3并支持 `R1A7_INSERTION_DATASET_ROOT`。只在已经完成现场/消费者安全核对的视觉运行环境中设置此变量；设置变量本身不会创建成功采集证据。

```bash
# 将TRIAL_ID替换为本次试验名称，再设置记录根目录
export R1A7_INSERTION_DATASET_ROOT=/data/threading/raw_sessions/TRIAL_ID
```

上面的TRIAL_ID是待填写占位符，运行前换为实际trial目录，不可原样复制。每次记录源码/模型hash、命令行、阶段、同步JSON、四路日志、events和物理结果。确认新session真实写入后才记“/data采集通过”。

## 5 四终端与Gate授权核对

顺序为Visual→Fusion→Gate→Controller，最后一个会发布真实LowCmd。本次没有启动它们；真机仍需实时核验网卡/IP、唯一独立发布者、锁、起始状态、急停和工作空间。父子进程可能同命令，不按PID数量盲目kill、不删锁。

当前Gate默认不开放输出。DUAL_READY表示感知条件准备好，不代表CLI授权已经打开。先只读检查参数定义与实际启动命令；只有明确授权进行真机执行后才按既有流程使用输出参数。不能伪造enabled=True或close完成JSON。

```bash
cd /home/robot/unitree_sim_isaaclab_threading/r1a7_threading_project
rg -n -A 5 'enable-output|output_enabled =|perception_ok' vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/wire_grasp_control_gate_v1.py
```

同一时间读取External/Wrist guidance、Fusion、Gate、phase及controller日志。先找proposed是否非零，再找Gate授权与reason，再查controller/IK真实响应。截图DUAL_READY+enabled=False与缺输出参数相容，但归档没有当时完整cmdline，不断言唯一根因。

## 6 选孔 暂停与停止

现版允许控制视角的可选选孔，仍需确认选中的是真孔。单相机1/2/3/N重选和通用P/R暂停尚未确认可用，不能按建议键位操作机器人。CLOSE_SETTLE自动HOLD、Gate disabled、Ctrl+C退出是不同动作；VR已有暂停提示也不等于新增任意阶段键盘保持。

正常退出先安全结束Controller、确认LowCmd publisher stopped，再停Gate/Fusion/Visual；异常优先物理急停。原始控制速度文件的新鲜度/旧enabled指令与FREE-HOLD风险需要单独回归，不在带载运动中热替换文件。源码/模型回滚依据Git与before-change快照，先停止相关进程再做比较和验证。
