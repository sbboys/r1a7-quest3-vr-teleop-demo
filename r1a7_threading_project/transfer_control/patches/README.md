# R1-A7 外部 IK 依赖补丁

控制台控制器从运行工作区加载：

`/home/robot/R1A7_VR_dual_arm_transfer_20260831_001/robot_dev/xr_teleoperate/teleop/robot_control/robot_arm_ik.py`

该文件不在本仓库内。为了让左、右臂使用相同的腕部/肘部软姿态约束，本目录保存
可审计补丁 `r1a7_arm_ik_posture_symmetry_20261016.patch`。补丁以 R1-A7 rebase
版本的 `robot_arm_ik.py` 为基准；本机基准文件 SHA256 为：

`18587cb8c459585f5a293f9857a3b405a3e94b2f6b490260f1d571066b370114`

## 应用方式

先确认目标文件与基准兼容，再应用；不要对未知版本强制打补丁：

```bash
cd /home/robot/R1A7_VR_dual_arm_transfer_20260831_001/robot_dev/xr_teleoperate
git apply --check /path/to/r1a7_arm_ik_posture_symmetry_20261016.patch
git apply /path/to/r1a7_arm_ik_posture_symmetry_20261016.patch
```

当前控制器启动时会检查 `R1A7_ArmIK.solve_ik()` 是否提供左臂对称参数。如果外部
依赖未打补丁，程序会在取得 LowCmd 控制权前报错退出，不会静默退回不对称控制。

此补丁只解决源码依赖可追溯性，不代表左臂已经通过真实机器人运动验收。
