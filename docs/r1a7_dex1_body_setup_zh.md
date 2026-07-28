# R1-A7 本体 Dex1_1 夹爪配置

当前拓扑：

- Dex1_1 夹爪串口板连接在机器人本体上。
- 相机连接在电脑上。
- 电脑上的相机识别程序只通过 DDS 发送 `rt/dex1/right/cmd`。
- 机器人本体必须运行 `dex1_1_gripper_server`，并发布 `rt/dex1/right/state`。
- R1-A 右臂仍然通过 `rt/lowstate` 读取状态，并通过 `rt/lowcmd` 写入低层控制命令。

注意：`dex1_1_gripper_server` 必须运行在能看到夹爪串口设备的主机上。运行前应能看到下面至少一种设备：

```bash
ls -l /dev/ttyUSB* /dev/ttyCH343USB* /dev/ttyACM*
```

如果这些设备都不存在，说明当前主机没有接到 Dex1_1 串口板；此时启动服务不会成功。

## 0. 整链路检查

电脑连接机器人网口后先运行：

```bash
cd /home/robot/unitree_sim_isaaclab
./scripts/check_r1a7_camera_arm_dex1_pipeline.sh
```

结果含义：

- `ARM_DDS_OK`：电脑已收到 R1-A 的 `rt/lowstate`。
- `DEX1_DDS_OK`：电脑已收到 Dex1_1 的 `rt/dex1/right/state`。
- `DEX1_DDS_FAIL`：夹爪服务未在能看到串口板的主机上运行，或服务绑定网卡错误。

只有同时出现 `ARM_DDS_OK` 和 `DEX1_DDS_OK`，才能继续做相机识别联动。

## 1. 电脑侧只读检测

在电脑项目目录运行：

```bash
cd /home/robot/unitree_sim_isaaclab
./scripts/check_dex1_1_body_gripper.sh
```

如果输出 `no state received`，说明电脑还没有收到本体侧 Dex1_1 状态。继续执行本体侧配置。

## 2. 本体侧启动 Dex1_1 服务

在机器人本体终端运行，先找到 192.168.123 网段的网卡：

```bash
ip -br addr
```

然后进入官方 Dex1_1 服务目录。实际目录按本体文件位置调整：

```bash
cd ~/dex1_1_service
sudo apt update
sudo apt install libserialport-dev
mkdir -p build
cmake -S . -B build
cmake --build build -j"$(nproc)"
```

启动服务，`eth0` 替换为本体上带 `192.168.123.*` 地址的网卡名：

```bash
sudo ./bin/dex1_1_gripper_server --network eth0
```

服务正常时会打印检测到的电机：

```text
Motor ID: 0  Side: Right  cmdTopic: rt/dex1/right/cmd  stateTopic: rt/dex1/right/state
```

右夹爪必须是 `Motor ID: 0`。

## 3. 需要校准时

手动闭合夹爪后，在本体上执行：

```bash
cd ~/dex1_1_service
sudo ./bin/dex1_1_gripper_server -c
```

提示右夹爪时输入 `s` 回车完成校准。

## 4. 电脑侧测试右夹爪

本体服务保持运行后，在电脑上执行：

```bash
cd /home/robot/unitree_sim_isaaclab
./scripts/check_r1a7_camera_arm_dex1_pipeline.sh
./scripts/check_dex1_1_body_gripper.sh
./scripts/test_dex1_1_right_gripper.sh
```

第一个脚本只读状态；第二个脚本会让右夹爪开合测试。

## 5. 相机识别联动右臂和右夹爪

确认第 4 步可用后运行：

```bash
cd /home/robot/unitree_sim_isaaclab
./scripts/run_r1a7_camera_real_control.sh --show
```

按提示输入 `ENABLE` 后开始控制。日志里：

- `dex1_cmd=none`：没有收到 `rt/dex1/right/state`，本体 Dex1_1 服务未在线或网卡错误。
- `dex1_cmd=数字`：电脑已经发布夹爪命令。
