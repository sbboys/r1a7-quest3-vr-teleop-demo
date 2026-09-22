# R1-A7 手臂控制台（终端基线）

该控制台将网页操作映射到已验证的终端控制基线，不修改
`r1a7_cartesian_stream_control.py`、官方 `robot_arm_ik.py` 或低层控制逻辑。

后台固定使用：

- 原加权 IK，不注入分层 IK。
- 右臂重力补偿 `0.25`。
- 末端速度默认 `10 mm/s`，加速度默认 `30 mm/s²`。
- 不传 `--max-joint-speed`，使用控制器默认 `2.0 rad/s`。
- 夹爪暂时禁用，先保证手臂轨迹可与终端基线直接比较。

启动网页服务：

```bash
cd /home/robot/unitree_sim_isaaclab_threading
PYTHONNOUSERSITE=1 /home/robot/miniconda3/envs/tv/bin/python -u \
  r1a7_threading_project/console3d_hierarchical/server.py \
  --host 127.0.0.1 --port 8098
```

打开 `http://127.0.0.1:8098/`。

## 停止方式

1. 松开末端方向按钮：发送 `jog:stop`，平滑减速后保持当前位置。
2. 点击“停止运动”，或按空格键/`Esc`：发送 `jog:stop`，控制器继续在线并保持当前位置。
3. 点击“断开连接”：先停止运动，再发送退出命令并释放 `rt/lowcmd` 控制锁。

窗口失焦或页面切到后台时，控制台也会自动发送停止运动命令。
