# R1-A7 深度相机手眼标定与视觉控制阶段总结（2026-08-20）

## 当前硬件与网络

- 机器人：Unitree R1-A7
- 深度相机：Orbbec Gemini 336L
- 标定物：AprilTag 36h11 id=0，黑色编码区边长 `0.092 m`
- 机器人直连网口：`enp6s0`
- 机器人地址：`192.168.123.161`
- 本机机器人网口地址：`192.168.123.223/24`
- 当前有效低层控制链路：
  - 状态：`rt/lowstate`
  - 命令：`rt/lowcmd`

## 已完成的标定链路

本阶段目标是得到相机坐标系与机器人 `base` 坐标系之间的转换，并验证“相机识别二维码 -> 转换到 base -> 控制探针靠近目标点”的流程。

坐标链路为：

```text
T_base_camera = T_base_tag * T_tag_camera
```

其中：

- `T_tag_camera` 来自深度相机识别 AprilTag 后的反变换。
- `T_base_tag` 来自右臂探针触碰 AprilTag 四角后拟合得到。

## 探针 TCP

探针固定在右臂夹爪上，TCP 使用 `right_wrist_yaw_link` 为参考。

结果文件：

```text
data/r1a7_teach/20260819_212932/probe_tcp_result.json
```

当前采用的探针 TCP：

```text
probe_tcp_in_frame_m = [0.254042, 0.015301, 0.026988]
```

该结果与人工测量“探针尖端距离 right_wrist_yaw_link 约 26 cm”一致。

## 当前相机外参结果

最新相机 AprilTag 外参文件：

```text
calibration/orbbec_gemini336l_apriltag_extrinsic.json
```

最新 `T_base_camera` 文件：

```text
calibration/r1a7_base_camera_from_apriltag.json
```

当前 `T_base_camera`：

```text
[[ 0.998231  0.057098 -0.016554  0.407515]
 [ 0.045619 -0.557168  0.829146 -0.397914]
 [ 0.038119 -0.828434 -0.558788  0.227874]
 [ 0.000000  0.000000  0.000000  1.000000]]
```

当前 `T_base_tag`：

```text
[[ 0.004287  0.999990  0.000961  0.443626]
 [-0.997436  0.004345 -0.071435 -0.021844]
 [-0.071439 -0.000652  0.997445 -0.079512]
 [ 0.000000  0.000000  0.000000  1.000000]]
```

## 新增与修改的工具

### 相机与标定

- `scripts/calibrate_orbbec_apriltag.py`
  - 使用 Orbbec SDK 采集 RGB 图像。
  - 检测 AprilTag 36h11 id=0。
  - 输出 `T_tag_to_camera`、`T_camera_to_tag`、重投影误差和标定截图。

- `scripts/compute_r1a7_probe_tcp.py`
  - 使用多姿态固定点采集数据计算探针 TCP。

- `scripts/compute_r1a7_base_camera_from_tag.py`
  - 使用探针触碰到的四角点计算 `T_base_tag`。
  - 结合 `T_tag_camera` 合成 `T_base_camera`。

- `scripts/verify_r1a7_camera_base_loop.py`
  - 用相机再次观察 AprilTag，验证转换回 base 后是否接近固定的 `T_base_tag`。

### 视觉目标控制

- `scripts/r1a7_move_probe_to_apriltag_safe.py`
  - 读取当前 AprilTag 位姿。
  - 转换为 base 下目标点。
  - 使用 Pinocchio 对右臂探针 TCP 做 IK。
  - 默认目标点为二维码中心上方 `0.20 m`。
  - 支持 `--execute` 后通过 `rt/lowcmd` 发布右臂控制。
  - 增加安全检查：
    - `reference tag error`
    - 最大目标距离
    - 最大关节变化
    - IK 误差
  - 增加分段执行逻辑：
    - 每段最多推进 `0.05 m`
    - 每 `0.5 s` 基于最新 lowstate 重新规划
    - 执行后输出最终探针位置和目标误差

### 手臂与 VR 控制

- `tools/r1a7_right_arm_comm.py`
  - 增加 `--test_joint_index`
  - 可用于指定右臂某个关节做小幅测试，例如 `7` 表示 `right_wrist_yaw`

- `tools/r1a7_vr_dual_arm_g1ik_real.py`
  - 增加 `--fixed_hold_indices`
  - 默认固定腰部电机 `13` 的启动角度，避免 VR 控制时腰部跟随当前状态漂移。

## 控制链路问题与结论

测试过程中出现过“脚本显示发布目标，但机器人手臂没有反应”的问题。

排查结论：

- 网络 `enp6s0 -> 192.168.123.161` 正常。
- `rt/lowstate` 可读时，说明 DDS 状态链路正常。
- `body/loco RPC` 多次返回 `3104`，说明高层 body RPC 不稳定或不可用。
- `rt/arm_sdk` 在当前机器人状态下未稳定驱动手臂。
- 当前可用控制链路是：

```text
state_topic   = rt/lowstate
command_topic = rt/lowcmd
```

因此视觉控制脚本后续统一使用 `rt/lowcmd`，不再依赖 `arm_sdk`。

## 视觉控制实验结果

已验证：

- 相机能识别 AprilTag 并转换到 base。
- `reference tag error` 在重新标定后可回到毫米级。
- 右臂能根据视觉目标产生实际运动。

当前限制：

- 当右臂/探针靠近二维码上方时，会遮挡 AprilTag，导致相机无法继续识别。
- 控制目标从较远位置直接到二维码上方时，Z 方向抬升不足。
- 多次执行中误差从约 `0.46 m` 降到约 `0.19 m` 后进入平台区。

最新一次 50 秒执行结果：

```text
target probe position base: [ 0.418850, -0.024479,  0.119015]
final probe position base:  [ 0.385787, -0.083672, -0.054505]
final target error: 0.186295 m
```

主要剩余误差在 Z 方向，说明单纯延长时间不能完全解决，需要调整运动策略。

## 下一步建议

后续应改为“先视觉定位，再纯 base 坐标分阶段运动”，避免靠近后继续依赖被遮挡的 AprilTag：

```text
1. 视觉确定 T_base_tag 与最终目标点
2. 当前 XY 位置先抬高到安全高度
3. 在安全高度平移到 tag XY 上方
4. 再下降或微调到目标高度
```

同时建议：

- Z 抬升阶段提高 `kp`、`max-speed-rad-s` 和保持时间。
- 执行阶段不要依赖实时 AprilTag，因为靠近后二维码可能被遮挡。
- 在工作台布置多个 AprilTag，减少单个 tag 被遮挡导致视觉中断的风险。
