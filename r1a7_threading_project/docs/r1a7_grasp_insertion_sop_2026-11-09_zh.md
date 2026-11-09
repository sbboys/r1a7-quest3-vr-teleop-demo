# 夹取与插孔研究操作步骤

此SOP依据11/09文档及当前文件整理；本次只发布说明，没有执行以下运行命令。原始历史操作全部保存在[全文](source_records/2026-11-09_grasp_insertion/worklog_fulltext_zh.md)。

## 1 先确认版本和文件位置

研究运行根目录为 `/home/robot/unitree_sim_isaaclab_threading/r1a7_threading_project`。仓库保留的Fusion/Gate/controller研究版本在 `docs/source_records/2026-11-09_grasp_insertion/research_code_snapshot/current/`；V3视觉在原 `vision_dual_camera/.../wire_unet/` 路径。**不要从仓库正常控制台目录直接运行同名Gate并假定已采用11/09策略。** 部署/回滚是单独操作，不由归档完成。

核对manifest的SHA-256、四SUCCESS快照、wire V2/Right Wrist V3/孔模型best.pt及环境；模型不随Git提供。检查当前工作区Git状态，保留未提交文件。

## 2 采集 标注 训练的离线准备

三相机采集程序为 `capture_three_camera_dataset_v1.py`；正式孔位session由trainer默认指向 `hole_tube_001/images_raw/session_20261105_105208`，采集元数据作为来源记录。标注使用hole_unet的v3工具，类别0背景/1board/2hole/255ignore。保留真实可见孔，按分组split进行QA，旧标注v1/v2只作历史参考。

训练程序为 `hole_unet/train_hole_board_unet_v1.py`。源码支持 `--root`、`--session`、`--split`、`--run-dir`、`--epochs`、`--batch-size`、`--lr` 等；正式group为24/6/6。配置、history及SHA在结果目录。不要把补写的模板当成已经恢复的原始训练CLI；本次也未新增独立test结果。

## 3 真机前检查

只读核对当前网卡/IP/路由、急停与起始姿态、相机身份及首帧、唯一LowCmd控制权和锁。文档中的enp6s0/192.168.123.223是历史值，不是本次已确认的现场状态。两个同命令PID要检查PPID/进程树，不盲目kill父子进程；不删除锁文件。

先核对四脚本标记和语法：WRIST_JAW_CORRIDOR_V1、GRASP_EXTERNAL_NEAR_Z_PRIORITY_V1、DISABLE_IMAGE_ERROR_HARD_GATE_TREND_V1、JAW_CORRIDOR_CLOSE_LATCH_V1。编译不等于运行或硬件通过。

## 4 按顺序启动四个终端

每个终端先进入研究根目录。以下是原文档对应的研究运行命令；Gate/controller会进入真实动作链，只有独立完成现场检查后才可运行。

终端1：视觉。环境tv_sam2。

```bash
cd /home/robot/unitree_sim_isaaclab_threading/r1a7_threading_project
PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv_sam2/bin/python -u \
  vision_dual_camera/task6_threading/dataset_tools/wire_unet/live_grasp_to_insertion_visual_v3.py
```

完成FREE/GRASP初始化，确认External/Wrist有效、TIP与线真实对应。后续near Z与jaw诊断不能只看是否输出valid。

终端2：Fusion。

```bash
cd /home/robot/unitree_sim_isaaclab_threading/r1a7_threading_project
PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv_sam2/bin/python -u \
  vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/wire_grasp_fusion_shadow_v1.py
```

终端3：Gate。先无controller检查JSON；开放输出仍可能被其他后台消费者读取。

```bash
cd /home/robot/unitree_sim_isaaclab_threading/r1a7_threading_project
PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv_sam2/bin/python -u \
  vision_dual_camera/task6_threading/perception/wire/proposed/r1a7_wire/wire_grasp_control_gate_v1.py --enable-output
```

终端4：controller最后启动。保存文档3mm/s低速模板，不宣称这是用户最终逐条回执命令；网卡/IP应按当前情况确认。

```bash
cd /home/robot/unitree_sim_isaaclab_threading/r1a7_threading_project
mkdir -p r1a7_logs/auto_grasp
R1A7_AUTO_GRASP_ENABLE=1 \
R1A7_AUTO_GRASP_PREOPEN_NORM_MAX=0.70 \
R1A7_AUTO_GRASP_CLOSE_AXIS_NORM_MAX=0.10 \
PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv/bin/python -u \
  transfer_control/experimental/r1a7_cartesian_stream_control_kd30_posvel_wu_dqik_u010_auto_grasp_direct_v1.py \
  --interface enp6s0 --domain-id 0 --host-ip 192.168.123.223 \
  --ik-frequency 30 --publish-frequency 250 --max-joint-speed 2.0 \
  --cartesian-jog-speed-mm-s 3 --cartesian-jog-accel-mm-s2 20 \
  --enable-gripper --gripper-initial-mode current \
  --no-right-gripper-open-cap-current --no-record-d435i \
  --log-file "r1a7_logs/auto_grasp/controller_$(date +%Y%m%d_%H%M%S).log"
```

在现场条件确认后才回应ENABLE。不要将历史3mm/s模板视为已经完成方向与停止资格测试。

## 5 观察夹取和阶段切换

关注OPEN GAP CALIBRATED、CORRIDOR LATCHED、CLOSE_SETTLE、CLOSE_ONCE、WAIT_CONTACT、GRASP COMPLETE。Gate在CLOSE_SETTLE_HOLD时零速度且enabled=True可能是预期保持；STOP/disabled是另一状态。必须同时观察实际线是否夹住，不能只凭日志字符串宣布成功。

抓取完成进入WAIT_HOLE_TARGET后，三相机选择同一实体孔。当前V3键盘支持Enter/Space/E确认，前提为三路均有选择；R在等待/插孔阶段重选。确认后进入INSERTION_TRACKING；继续监测FREE来源、目标身份、误差、时间戳、Gate状态与真实位移。键位以部署文件为准；这些键可能重新允许动作，不是纯界面操作。

## 6 保存证据 停止和回滚

将命令行、四脚本hash、阶段/速度JSON、controller持久日志和独立物理观察按单次任务保存。记录失败、STOP、超时与重试，不把一轮内多次重试当独立成功样本。

正常结束先Q/程序安全退出停止controller，确认LowCmd publisher stopped后再停Gate、Fusion、Visual；异常优先物理急停。不要先停Gate而让controller继续读取旧enabled指令。回滚前先停相关程序，依据Git/备份恢复，重新做静态和现场检查；本次归档没有执行回滚、清锁或真机动作。
