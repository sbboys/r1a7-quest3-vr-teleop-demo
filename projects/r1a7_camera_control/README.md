# R1-A7 相机识别与手臂控制

## 目标

利用 Gemini、腕部 RGB 或固定相机获取图像/位姿，并驱动 R1-A7 手臂或夹爪完成相机引导控制。

## 代码位置

- 相机动作提供器：`camera_teleop/action_provider/`
- 当前相机控制脚本：`tools/r1a7_camera_real_teleop.py`
- Gemini 位姿输入：`action_provider/gemini_pose_source.py`
- 相机调试和安装：`docs/r1a7_camera_*.md`
- 早期兼容实现：`camera_teleop/`、`vr_teleop/`

早期相机识别代码保留用于追溯，不默认作为最新真机入口。

