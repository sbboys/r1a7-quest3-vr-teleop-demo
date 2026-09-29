# R1-A7 共享运行时

这些目录同时被多个项目使用，不归属于某一个实验：

- `dds/`：DDS 通信和状态/命令封装；
- `robots/`：机器人模型和硬件定义；
- `tasks/`：Isaac Lab 任务；
- `scripts/`：启动、停止、检查和校准脚本；
- `tools/`：通用工具、状态检查、IK 和低层接口；
- `calibration/`：共享标定文件；
- `requirements.txt`、`.gitmodules`：公共依赖。

共享运行时的修改必须同时检查细线、扳手、VR 和相机项目的影响。

