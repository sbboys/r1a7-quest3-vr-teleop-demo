# R1-A7 VR 双臂遥操作迁移与 D435i 采集记录

日期：2026-09-02

## 当前结论

当前 R1-A7 的 VR 双臂控制实际运行入口来自迁移包：

```bash
/home/robot/R1A7_VR_dual_arm_transfer_20260831_001/unitree_sdk2-main/tools/r1a7_unitree_official_vuer_real_lowcmd.py
```

项目内已经同步保留一份归档副本：

```bash
r1a7_wrench_project/transfer_control_20260831/tools/r1a7_unitree_official_vuer_real_lowcmd.py
```

数据采集模块已经放在 `unitree_sim_isaaclab` 项目内：

```bash
r1a7_wrench_project/scripts/record_d435i_color_depth.py
```

X 按钮触发记录时，机器人状态和 D435i 普通视频会保存到：

```bash
r1a7_wrench_project/data/episodes/
```

## 网络和运行入口

当前 Quest/路由侧固定使用：

```text
192.168.1.103
```

Quest 浏览器地址：

```text
https://192.168.1.103:8012/?ws=wss://192.168.1.103:8012
```

机器人 DDS 网口：

```text
enp6s0
```

推荐启动命令：

```bash
bash /home/robot/R1A7_VR_dual_arm_transfer_20260831_001/unitree_sdk2-main/tools/run_r1a7_unitree_official_vuer_real_lowcmd.sh
```

项目内归档副本启动命令：

```bash
bash r1a7_wrench_project/transfer_control_20260831/tools/run_r1a7_unitree_official_vuer_real_lowcmd.sh
```

启动后需要在终端输入：

```text
ENABLE
```

然后进入 Quest 浏览器页面，点击 `Enter VR`。

## 控制逻辑

当前控制链路为：

```text
Quest / TeleVuer 手柄位姿
-> R1A7_ArmIK
-> R1-A7 14 维双臂关节目标
-> rt/lowcmd
-> R1-A7 真机双臂
```

操作约定：

- 右手 A：开启/关闭机器人控制。
- 左手 X：开始/停止数据记录。
- 扳机：控制 DEX1 夹爪开合。
- 退出 VR 或关闭控制后，机器人不继续跟随旧手柄数据。

## 夹爪抖动处理

针对夹爪夹持扳手后持续闭合导致腕部/夹爪抖动的问题，当前控制脚本保留了接触保持逻辑：

- 默认启用 `--gripper-contact-hold`。
- 当夹爪目标继续闭合但实际关节停止明显运动时，认为已经接触物体。
- 接触后锁定当前位置，并叠加很小的闭合偏置，避免继续硬顶目标闭合位置。
- 目标是保持柔性夹紧力，同时减少抖动和弹出扳手的风险。

当前默认参数：

```text
gripper_contact_error = 0.08
gripper_contact_stall_eps = 0.004
gripper_contact_stall_time = 0.25
gripper_contact_hold_bias = 0.035
```

## D435i 采集方式

之前使用 `rs-record` 生成 `.db3`，但该方式不适合快速确认视频画面质量，也曾出现文件存在但普通播放器不可直接查看的问题。

当前 X 记录改为显式打开 D435i 的 color + depth：

```text
d435i_color.mp4
d435i_depth_preview.mp4
d435i_frames.csv
```

其中：

- `d435i_color.mp4`：普通 RGB 视频，可直接打开查看。
- `d435i_depth_preview.mp4`：深度伪彩预览视频，可直接打开查看。
- `d435i_frames.csv`：每帧时间戳、帧号和相机序列号。
- `states.csv`：机器人状态、目标关节、手柄按钮、触发器和腕部目标位姿。

## 已修复的问题

### 相机录制没有画面

现象：

按 X 后生成了记录目录，但没有 `d435i_color.mp4` 或视频无有效画面。

原因：

VR 主控制进程运行在 `tv` 环境，并带有：

```text
PYTHONNOUSERSITE=1
```

D435i 子进程继承该环境变量后，即使用 `/usr/bin/python3` 启动，也无法导入用户目录中的 `pyrealsense2`，导致录制程序启动后立即报错：

```text
ModuleNotFoundError: No module named 'pyrealsense2'
```

修复：

启动 D435i 子进程时单独移除 `PYTHONNOUSERSITE`：

```python
camera_env = dict(os.environ)
camera_env.pop("PYTHONNOUSERSITE", None)
subprocess.Popen(..., env=camera_env)
```

修复后已验证：

```text
D435I_OPENCV_DONE frames=1783
```

## 最新有效样例数据

最新确认有画面的样例记录为：

```bash
r1a7_wrench_project/data/episodes/r1a7_vr_003/
```

文件：

```text
metadata.json
states.csv
d435i_color.mp4
d435i_depth_preview.mp4
d435i_frames.csv
d435i_record.log
```

该样例用于证明：

- X 按钮能同步触发机器人状态记录和 D435i 录制。
- D435i 彩色视频和深度预览均已正常写入。
- 录制脚本能够在主 VR 控制环境下正常调用。

## 后续建议

短期继续使用当前链路采集扳手示教数据。若后续要提高项目可维护性，建议把实际运行入口从外部迁移包切换到项目内归档副本：

```bash
r1a7_wrench_project/transfer_control_20260831/tools/run_r1a7_unitree_official_vuer_real_lowcmd.sh
```

这样 GitHub 上保存的脚本、采集模块和数据目录会完全对应，后续复现实验更清晰。
