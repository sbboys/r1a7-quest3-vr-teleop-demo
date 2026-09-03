# R1-A7 官方 VR 双臂遥操作测试流程

日期：2026-09-02

目标：在当前项目旁路测试宇树官方 R1-A7 VR 双臂遥操作能力，不使用本项目里的 IK、接触控制、夹爪保持或任何 `tools/r1a7_vr_dual_arm_g1ik_real.py` 逻辑。

## 当前结论

- 当前项目分支是 `r1a7-wrench-baseline`，远端 `demo` 指向 `git@github.com:sbboys/r1a7-quest3-vr-teleop-demo.git`。
- 旧目录 `/home/robot/xr_teleoperate` 是较早官方快照，未提供 `R1_A7` 启动选项。
- 已新增独立官方目录 `/home/robot/xr_teleoperate_r1_official`，HEAD 为 `845b25a`，官方代码包含 `--arm R1_A7`、`R1_A7_ArmIK`、`R1_A7_ArmController` 和 `assets/r1/r1_a7.urdf`。
- 静态导入检查通过，R1-A7 双臂电机索引为 `15,16,17,18,19,20,21,22,23,24,25,26,27,28`。
- 当前活体检查：机器人控制网 `192.168.123.223` 可达；官方默认图像服务器 `192.168.123.164` 不可达；8012 未启动。

## 绝对不要使用的本项目入口

本次测试不启动以下项目脚本：

```bash
python tools/r1a7_vr_dual_arm_g1ik_real.py
python tools/r1a7_camera_real_teleop.py
python r1a7_wrench_project/scripts/start_r1a7_vr_b_home.sh
```

这些是本项目历史适配/调参路径，不符合“全部采用官方教程”的约束。

## 官方代码准备

官方代码位于：

```bash
/home/robot/xr_teleoperate_r1_official
```

如需重新拉取：

```bash
cd /home/robot
git clone --depth 1 https://github.com/unitreerobotics/xr_teleoperate.git xr_teleoperate_r1_official
cd xr_teleoperate_r1_official
git submodule update --init --depth 1
```

依赖环境使用既有 `tv` conda 环境：

```bash
PYTHONNOUSERSITE=1 /home/robot/miniconda3/bin/conda run --no-capture-output -n tv \
python -m pip install -r /home/robot/xr_teleoperate_r1_official/requirements.txt
```

本机已有 HTTPS 证书：

```text
/home/robot/.config/xr_teleoperate/cert.pem
/home/robot/.config/xr_teleoperate/key.pem
```

## 启动前检查

确认没有旧遥操作进程和端口占用：

```bash
ss -lntp | grep -E '8012|60000|60001' || true
ps -ef | grep -E 'xr_teleoperate|r1a7_vr|quest|televuer|vuer|teleop|dex1' | grep -v grep || true
```

确认当前网络：

```bash
ip -br addr
ip route | grep -E '192\.168\.(1|123)|default' || true
ping -c 1 -W 1 192.168.123.223
ping -c 1 -W 1 192.168.123.164
```

当前已观察到：

```text
enp6s0           UP  192.168.123.223/24
enx9c69d37d0967  UP  192.168.1.103/24
```

因此默认参数为：

```text
NETWORK_INTERFACE=enp6s0
HOST_IP=192.168.1.103
IMG_SERVER_IP=192.168.123.164
```

如果 `192.168.123.164` 不通，官方程序可能回退使用本地 `cam_config_server.yaml` 并继续启动 Vuer；但如果需要第一视角图像、WebRTC 图像流或录制图像，必须先在机器人 PC2 或实际图像服务器上启动官方 teleimager 服务，或把 `IMG_SERVER_IP` 改为实际图像服务器地址。

## 官方 R1-A7 启动命令

建议使用包装脚本，它只进入官方目录并执行官方 `teleop_hand_and_arm.py`：

```bash
cd /home/robot/unitree_sim_isaaclab
NETWORK_INTERFACE=enp6s0 \
HOST_IP=192.168.1.103 \
IMG_SERVER_IP=192.168.123.164 \
INPUT_MODE=controller \
DISPLAY_MODE=pass-through \
EE=dex1 \
bash r1a7_wrench_project/scripts/start_official_r1a7_vr_teleop.sh
```

如果现场没有启动外置 Dex1 服务，先只测试官方 R1-A7 双臂：

```bash
cd /home/robot/unitree_sim_isaaclab
NETWORK_INTERFACE=enp6s0 \
HOST_IP=192.168.1.103 \
IMG_SERVER_IP=192.168.123.164 \
INPUT_MODE=controller \
DISPLAY_MODE=pass-through \
EE=none \
bash r1a7_wrench_project/scripts/start_official_r1a7_vr_teleop.sh
```

当前包装脚本默认启用两个现场增强功能：

```text
A_TOGGLE_CONTROL=1
ARM_REFERENCE_MODE=fixed_waist
```

含义：

- 进入 VR 后，第一次按右手柄 A 开始控制双臂；再按一次 A 暂停控制。
- 暂停时机器人保持暂停瞬间的双臂位置，直到再次按 A 恢复控制。
- `fixed_waist` 会忽略头显姿态对手柄坐标的重映射，适合 Quest 3 挂在脖子上、不戴在头上的测试方式。

挂脖测试建议：

- 启动程序前先把 Quest 3 挂在脖子上，正面尽量朝机器人前方或操作者前方，不要大角度歪斜。
- 左右手柄先放在自然、安全的初始位置。
- 进入 `Virtual Reality` 后不要在终端按 `r`。
- 第一次按右手柄 A 后，机器人开始跟随手柄。
- 需要暂停时再按一次右手柄 A，机器人保持暂停瞬间的双臂位置。
- 再次按右手柄 A 恢复跟随。

等终端显示官方程序已就绪后，在 Quest 3 浏览器打开：

```text
https://192.168.1.103:8012/?ws=wss://192.168.1.103:8012
```

进入 Vuer 后点击 `Virtual Reality`，允许浏览器权限，看到连接日志后，在终端按：

```text
不需要按 r；默认等待右手柄 A 开始控制。
```

退出时在同一终端按：

```text
q
```

## 阻塞条件

当前影响稳定真机 VR 测试的阻塞项是：

```text
1. 192.168.123.164 图像服务器不可达
2. 机器人控制网口 enp6s0 在测试中多次 Link is Down / Link is Up
```

官方主程序启动时会创建 `ImageClient` 并读取图像服务配置；当前版本会在连接不到图像服务器时尝试回退到本地配置。若使用 `DISPLAY_MODE=pass-through`，可以先做手柄/双臂遥操作连接测试；若要看机器人视角或记录图像，必须先恢复图像服务。

另外，官方脚本使用 `sshkeyboard` 读取终端按键；需要在真实交互终端中运行。不要用后台、无 TTY、IDE 非交互运行方式启动，否则会出现 `Inappropriate ioctl for device`，且无法按 `r`/`q` 控制状态切换。

## 2026-09-02 实测记录

本次测试只使用官方目录 `/home/robot/xr_teleoperate_r1_official`，没有启动本项目的 VR/IK/接触控制脚本。

### 成功项

- 官方 `teleop_hand_and_arm.py --arm=R1_A7` 可以启动。
- `R1_A7_ArmController` 可以订阅 `rt/lowstate`，日志出现 `Subscribe dds ok`。
- 官方启动阶段可以执行 `Enter debug mode: Success`。
- 官方启动阶段会执行 `head and waist returning to zero`，随后 `head and waist return to zero OK`。
- Quest 3 可以连接主机 `192.168.1.103:8012`，连接端 IP 观察为 `192.168.1.124`。
- 发送 `r` 后，官方程序进入 `start Tracking`。
- 一次有效测试中，2 秒内 R1-A7 双臂关节出现明显变化，最大变化约 `1.2 rad`，说明官方 R1-A7 双臂链路可以实际驱动机器人。
- 发送 `q` 后，官方程序进入 `ctrl_dual_arm_go_home start`，随后 `Image client has been closed` 和 `Finally, exiting program`，可以正常退出。

### 未完成项

- 未测试 Dex1 夹爪。原因是官方 `--ee=dex1` 会启动 `Dex1_1_Gripper_Controller` 并等待外置 Dex1 DDS 服务；现场该服务/topic 未启动，程序会卡在 `Waiting to subscribe dds`。
- 未测试机器人第一视角图像和图像录制。原因是官方默认图像服务器 `192.168.123.164:60000` 不可达，程序只能回退本地相机配置。

### 风险和异常

测试过程中出现大量 CycloneDDS 写入错误：

```text
ddsi_udp_conn_write to udp/192.168.123.161:<port> failed with retcode -1
```

同时内核日志记录到机器人控制网口 `enp6s0` 多次掉线：

```text
enp6s0: Link is Down
enp6s0: Link is Up - 1Gbps/Full - flow control rx/tx
```

因此，虽然官方双臂链路已经证明可以驱动机器人，但当前硬件网络链路不稳定，不建议继续长时间测试。继续测试前必须先排查 `enp6s0` 对应网线、交换机、机器人端口或网卡。

## 官方启动/归位行为解释

根据官方 `R1_A7_ArmController` 当前实现：

- 程序初始化时会创建 `rt/lowcmd` publisher 和 `rt/lowstate` subscriber。
- 收到 `rt/lowstate` 后，官方控制器读取当前全身电机位置，并把低层命令初始化为当前姿态。
- 初始化阶段会调用 `ctrl_head_and_waist_go_home()`，将头部和可用腰部关节在约 3 秒内缓慢回零。
- 双臂目标 `q_target` 默认是 14 个 0；发布线程启动后，会以约 250 Hz 将双臂向官方零位/home 目标控制。因此现场可能观察到双臂自动抬起或回到某个初始保持姿态。
- 在终端显示 `Press [r]` 时，程序还没有进入手柄跟随主循环，但官方低层保持线程已经运行。
- Quest 进入 `Virtual Reality` 本身不会单独触发控制状态切换；真正开始跟随是终端收到 `r` 后进入 `start Tracking` 主循环。
- 进入 `start Tracking` 后，程序通过 `TeleVuerWrapper.get_tele_data()` 获取 XR 手柄/手腕数据，再调用 `R1_A7_ArmIK.solve_ik()` 生成双臂目标。
- 按 `q` 退出时，官方程序会调用 `ctrl_dual_arm_go_home()`，将双臂目标设为 0，并等待接近零位后退出。

因此，现场看到的“启动后抬臂并保持”“进入 VR 后在某一姿态保持”“退出时回到 home/zero”都符合官方控制器逻辑；但如果同时出现 DDS 写失败或网口掉线，保持/归位动作可能延迟、中断或不可靠。
