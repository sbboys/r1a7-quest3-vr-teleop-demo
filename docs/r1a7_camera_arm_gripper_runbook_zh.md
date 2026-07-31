# R1-A7 Gemini 相机控制右臂与 Dex1_1 夹爪操作手册

本文档记录当前已经跑通的流程：电脑通过 Gemini 相机识别人体右臂动作，控制 R1-A7 真实机器人右臂和 Dex1_1 右夹爪。

## 0. 当前推荐启动命令

进入工程目录：

```bash
cd /home/robot/unitree_sim_isaaclab
```

一终端同时启动右臂和右夹爪控制：

```bash
CONTROL_CHANNEL=lowcmd ENABLE_DEX1=1 DEX1_SIDE=right \
./scripts/run_r1a7_camera_real_control_one_terminal.sh --show
```

只控制右臂、不控制夹爪：

```bash
CONTROL_CHANNEL=lowcmd ENABLE_DEX1=0 \
./scripts/run_r1a7_camera_real_control_one_terminal.sh --show
```

启动后如果终端提示输入确认，清空机器人周围空间，确认急停可用，再输入：

```text
ENABLE
```

## 1. 从另一台电脑接回本电脑后的通信建立

当机器人信号线、USB 扩展坞、夹爪串口或相机从另一台电脑接回本电脑后，先不要直接运行相机控制，按下面顺序恢复通信。

### 1.1 检查网卡与 IP

```bash
ip -br addr
```

当前已经跑通的本电脑机器人网卡是：

```text
enx9c69d37d0967  192.168.123.223/24
```

如果显示成 `192.168.123.223/32`，DDS 通信可能异常，需要改回 `/24`：

```bash
sudo nmcli connection modify unitree-r1a7-body \
  ipv4.method manual \
  ipv4.addresses 192.168.123.223/24 \
  ipv4.gateway '' \
  ipv4.never-default yes \
  ipv4.routes '192.168.123.0/24' \
  ipv6.method ignore

sudo nmcli connection down unitree-r1a7-body
sudo nmcli connection up unitree-r1a7-body
```

再次确认：

```bash
ip -br addr
ip route get 192.168.123.161
```

正常时 `ip route get 192.168.123.161` 应该走：

```text
dev enx9c69d37d0967 src 192.168.123.223
```

### 1.2 检查能否收到机器人低层状态

```bash
/home/robot/miniconda3/bin/conda run --no-capture-output -n isaaclab \
python -u tools/r1a7_right_arm_comm.py \
  --interface enx9c69d37d0967 \
  --domain_id 0 \
  --idl hg \
  --state_topic rt/lowstate \
  --mode monitor \
  --duration 5
```

正常现象：

- 终端显示 `DDS initialized`
- 能收到 `rt/lowstate`
- 能打印右臂 7 个关节状态

异常现象：

```text
no rt/lowstate received
```

处理：

1. 先确认网卡是 `192.168.123.223/24`，不是 `/32`。
2. 确认网线接的是机器人本体通信口。
3. 确认机器人处于可开发/可控制状态。
4. 使用 `CONTROL_CHANNEL=lowcmd`，不要用 `arm_sdk` 作为当前主控方式。

### 1.3 检查夹爪串口

```bash
ls -l /dev/ttyUSB* /dev/ttyCH343USB* /dev/ttyACM*
lsusb
```

正常时会看到多个 `/dev/ttyUSB*`。如果没有任何串口：

1. 重新插拔 USB 扩展坞。
2. 检查夹爪控制线是否接到当前电脑。
3. 检查扩展坞是否供电正常。
4. 再运行 `lsusb` 和 `ls -l /dev/ttyUSB*`。

## 2. 夹爪控制不了的解决流程

夹爪问题先分清三种情况：

- 没有串口：电脑没有识别到夹爪控制板。
- 没有 DDS 状态：Dex1 服务没启动或网络接口不对。
- 有 `dex1_cmd` 但夹爪不动：电机可能进入保护状态。

### 2.1 启动 Dex1_1 服务

单独启动服务：

```bash
cd /home/robot/unitree_sim_isaaclab
./scripts/run_dex1_1_service.sh
```

正常时会看到类似：

```text
Detected motors:
  - Motor ID: 0  Side: right  Port: /dev/ttyUSB0  cmdTopic: rt/dex1/right/cmd  stateTopic: rt/dex1/right/state
  - Motor ID: 1  Side: left   Port: /dev/ttyUSB3  cmdTopic: rt/dex1/left/cmd   stateTopic: rt/dex1/left/state
Dex1-1 Gripper Server started.
```

右夹爪必须检测到：

```text
Motor ID: 0  Side: right
```

### 2.2 检查右夹爪 DDS 状态

```bash
./scripts/check_dex1_1_body_gripper.sh
```

正常时会收到：

```text
side=right state_q=... cmd_q=...
```

如果提示：

```text
no state received
```

处理：

1. 确认 `run_dex1_1_service.sh` 正在运行。
2. 确认服务使用的网卡是 `enx9c69d37d0967`。
3. 确认右夹爪被识别为 `Motor ID: 0 Side: right`。

### 2.3 官方右夹爪开合测试

在运行相机控制前，先用官方测试确认右夹爪能动：

```bash
./scripts/cycle_dex1_1_right_gripper_official.sh
```

正常现象：

- 右夹爪实际开合。
- 终端里的 `R=` 数值会变化。

异常现象：

- `R=` 数值不变。
- 右夹爪实际不动。

这说明问题不在相机识别，而在 Dex1 服务、电机状态或硬件。

### 2.4 `merror=8` 时的处理

如果服务日志中出现：

```text
Motor 0 debug ... merror=8
```

并且同时出现：

```text
timeout=false msg_q=... cmd_q=... state_q=固定不变
```

含义是：相机程序或官方测试已经把命令发到 Dex1 服务，但右夹爪电机处在错误/保护状态，不执行动作。

处理步骤：

```bash
cd /home/robot/unitree_sim_isaaclab
./scripts/stop_dex1_1_service_bg.sh
pkill -f dex1_1_gripper_server
```

然后对机器人/夹爪电源做完整断电：

1. 停止所有夹爪服务。
2. 关闭机器人或夹爪供电。
3. 等待 10-15 秒。
4. 重新上电。
5. 重新插拔 USB 扩展坞或夹爪串口线。

再启动服务：

```bash
./scripts/run_dex1_1_service.sh
```

确认右夹爪：

```text
Motor 0 debug ... merror=0
```

只有 `merror=0` 后，才继续运行相机控制。

### 2.5 夹爪夹住后不能松开

先停止相机控制，然后执行：

```bash
cd /home/robot/unitree_sim_isaaclab
./scripts/open_dex1_1_right_gripper.sh
```

如果仍不能松开，检查日志：

```bash
tail -n 80 Log/dex1_1_service.log
```

如果右夹爪是 `merror=8`，按 2.4 的断电复位流程处理。

## 3. 手臂控制不了的解决流程

手臂控制问题优先检查 `rt/lowstate` 和控制通道。

### 3.1 检查低层状态

```bash
/home/robot/miniconda3/bin/conda run --no-capture-output -n isaaclab \
python -u tools/r1a7_right_arm_comm.py \
  --interface enx9c69d37d0967 \
  --domain_id 0 \
  --idl hg \
  --state_topic rt/lowstate \
  --mode monitor \
  --duration 5
```

如果提示：

```text
no rt/lowstate received
```

优先处理网络：

```bash
ip -br addr
ip route get 192.168.123.161
```

网卡必须是：

```text
enx9c69d37d0967  192.168.123.223/24
```

### 3.2 使用 lowcmd 小幅测试右臂

确认机器人周围安全，急停可用，再运行：

```bash
/home/robot/miniconda3/bin/conda run --no-capture-output -n isaaclab \
python -u tools/r1a7_right_arm_comm.py \
  --interface enx9c69d37d0967 \
  --domain_id 0 \
  --idl hg \
  --state_topic rt/lowstate \
  --command_topic rt/lowcmd \
  --debug_lowcmd \
  --enable_control \
  --mode test \
  --duration 8 \
  --test_amplitude_deg 3 \
  --kp 32 \
  --kd 1.8
```

如果提示确认，输入：

```text
ENABLE
```

正常现象：

- 右臂有小幅动作。
- 终端中 `q` 会跟随 `cmd` 变化。

如果这个测试能动，但相机控制不动，说明机器人通信和手臂控制正常，问题在相机输入、人体识别或相机控制参数。

如果这个测试也不动：

1. 确认机器人不是锁死或急停状态。
2. 确认机器人处于阻尼/可控制状态。
3. 确认 `CONTROL_CHANNEL=lowcmd`。
4. 确认 `rt/lowstate` 能收到。
5. 重新启动机器人本体控制状态后再测。

### 3.3 不要让腰部或底座参与

当前任务只控制右臂 7 个关节。运行时应保持：

```bash
LOWCMD_HOLD_INDICES=22,23,24,25,26,27,28
```

总入口脚本默认已经按右臂 7 个关节控制。如果运行过程中出现腰部或底座扭转，优先检查是否使用了错误脚本或错误控制通道。

## 4. 日常完整操作流程

### 4.1 上电后第一次运行

```bash
cd /home/robot/unitree_sim_isaaclab
ip -br addr
```

确认：

```text
enx9c69d37d0967  192.168.123.223/24
```

检查右臂状态：

```bash
/home/robot/miniconda3/bin/conda run --no-capture-output -n isaaclab \
python -u tools/r1a7_right_arm_comm.py \
  --interface enx9c69d37d0967 \
  --domain_id 0 \
  --idl hg \
  --state_topic rt/lowstate \
  --mode monitor \
  --duration 3
```

检查右夹爪：

```bash
./scripts/cycle_dex1_1_right_gripper_official.sh
```

确认右臂和夹爪都正常后，再启动相机控制：

```bash
CONTROL_CHANNEL=lowcmd ENABLE_DEX1=1 DEX1_SIDE=right \
./scripts/run_r1a7_camera_real_control_one_terminal.sh --show
```

### 4.2 相机控制运行时怎么看日志

正常日志应包含：

```text
control: ENABLED
Dex1: True right
target=yes
dex1_cmd=...
dex1_state=...
cmd: right_shoulder_pitch=... right_elbow=...
```

判断方法：

- `target=yes`：相机识别到人体目标。
- `cmd` 变化：右臂控制命令在变化。
- `dex1_cmd` 变化：夹爪控制命令在变化。
- `dex1_state` 跟随变化：夹爪实际状态在变化。

如果 `dex1_cmd` 变化但 `dex1_state` 不变，检查 Dex1 服务日志里的 `merror`。

### 4.3 退出

在运行终端按：

```text
Ctrl+C
```

总入口会自动停止后台 Dex1 服务。如果需要手动确认：

```bash
./scripts/status_dex1_1_service_bg.sh
```

需要强制停止：

```bash
./scripts/stop_dex1_1_service_bg.sh
```

## 5. 常用脚本说明

- `scripts/run_r1a7_camera_real_control_one_terminal.sh`：当前推荐总入口，一个终端启动相机、右臂、夹爪。
- `scripts/run_r1a7_camera_real_control.sh`：相机控制参数脚本。
- `scripts/run_dex1_1_service.sh`：前台启动 Dex1_1 夹爪服务。
- `scripts/start_dex1_1_service_bg.sh`：后台启动 Dex1_1 服务。
- `scripts/stop_dex1_1_service_bg.sh`：停止后台 Dex1_1 服务。
- `scripts/status_dex1_1_service_bg.sh`：查看 Dex1_1 服务状态和最近日志。
- `scripts/cycle_dex1_1_right_gripper_official.sh`：官方右夹爪开合测试。
- `scripts/open_dex1_1_right_gripper.sh`：尝试打开右夹爪。
- `tools/r1a7_right_arm_comm.py`：右臂 DDS 通信检查和小幅测试。
- `tools/r1a7_camera_real_teleop.py`：相机识别到右臂/夹爪命令的核心程序。

## 6. 常见现象与判断

### 6.1 右臂能动，夹爪不动

重点看：

```bash
tail -n 100 Log/dex1_1_service.log
```

如果有：

```text
Motor 0 debug ... merror=8
```

按 2.4 处理。

### 6.2 夹爪服务能启动，但只检测到左夹爪

现象：

```text
Detected motors:
  - Motor ID: 1 Side: left
```

没有：

```text
Motor ID: 0 Side: right
```

处理：

1. 重新插拔右夹爪串口线。
2. 重新插拔 USB 扩展坞。
3. 断电重启夹爪。
4. 再运行 `./scripts/run_dex1_1_service.sh`。

### 6.3 机器人手臂完全不动

先确认：

```bash
ip -br addr
```

再确认：

```bash
/home/robot/miniconda3/bin/conda run --no-capture-output -n isaaclab \
python -u tools/r1a7_right_arm_comm.py \
  --interface enx9c69d37d0967 \
  --domain_id 0 \
  --idl hg \
  --state_topic rt/lowstate \
  --mode monitor \
  --duration 5
```

如果收不到 `rt/lowstate`，先修网络和机器人状态，不要继续调相机。

### 6.4 相机画面有，但机器人不跟随

看相机控制日志：

- `target=no`：人体关键点没有识别稳定，需要调整人站位、光照、距离。
- `target=yes` 但 `cmd` 不变：输入变化太小或死区过大。
- `target=yes` 且 `cmd` 变化，但机器人不动：回到 3.1 和 3.2 检查右臂控制链路。

### 6.5 夹爪夹紧/松开阈值不合适

可以通过环境变量调整：

```bash
DEX1_GRIP_OPEN_THRESHOLD=0.55 DEX1_GRIP_CLOSE_THRESHOLD=0.75 \
CONTROL_CHANNEL=lowcmd ENABLE_DEX1=1 DEX1_SIDE=right \
./scripts/run_r1a7_camera_real_control_one_terminal.sh --show
```

如果夹爪误夹紧，提高 `DEX1_GRIP_CLOSE_THRESHOLD`。
如果夹爪不容易夹紧，降低 `DEX1_GRIP_CLOSE_THRESHOLD`。
如果夹爪不容易松开，提高张开动作稳定性，先确认 `merror=0`，再调整 `DEX1_GRIP_OPEN_THRESHOLD`。

## 7. 安全注意

1. 每次真实机器人控制前，确认机器人周围没有人和障碍物。
2. 急停必须在手边。
3. 第一次测试只做小幅动作。
4. 夹爪测试时不要把手放在夹爪内部。
5. `merror` 非 0 时不要反复发送夹爪命令，应先停服务并断电复位。

