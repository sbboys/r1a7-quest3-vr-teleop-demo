# R1-A7 手臂控制台

## 功能

提供 R1-A7 机器人三维模型、视角控制、笛卡尔方向控制、单关节控制、夹爪控制、连接参数和运行日志。

## 代码位置

- 主要控制台：`r1a7_threading_project/console3d/`
- 分层控制台实验：`r1a7_threading_project/console3d_hierarchical/`
- 3D 模型：`r1a7_threading_project/console3d/model/`
- 前端：`r1a7_threading_project/console3d/static/`
- 服务端：`r1a7_threading_project/console3d/server.py`

控制台是操作入口，不等于底层控制算法。修改底层手臂控制脚本前，应先确认项目文档的保护边界。

