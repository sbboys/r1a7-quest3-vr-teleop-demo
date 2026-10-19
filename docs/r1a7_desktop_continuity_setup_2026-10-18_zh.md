# R1-A7 桌面连续工作入口整理记录（2026-10-18）

## 目标

解决长对话压缩或新建对话后重复扫描设备、误用旧脚本、重新选择工作区和偏离已验证
路线的问题。整理过程没有移动或删除原文件。

## 当前桌面入口

桌面新增：

`/home/robot/桌面/00_R1-A7_连续项目入口`

桌面根目录新增快捷方式：

`/home/robot/桌面/R1-A7_连续项目入口.desktop`

入口目录通过符号链接连接以下事实源：

| 名称 | 目标 | 定位 |
|---|---|---|
| `01_当前GitHub工作区` | `/home/robot/unitree_sim_isaaclab_console` | 唯一当前代码入口 |
| `02_新对话启动词` | `docs/NEW_CHAT_START.md` | 新对话/长对话恢复模板 |
| `03_当前状态` | `docs/CURRENT_STATE.md` | 已完成、未验证和下一步 |
| `04_验证矩阵` | `docs/VALIDATION_MATRIX.md` | 区分代码、零运动和实机验证 |
| `05_研究实验参考区` | `/home/robot/unitree_sim_isaaclab_threading` | 实验参考，不自动清理 |
| `06_原始工作记录` | `/home/robot/桌面/jiqiren/renwu` | Word 和阶段记录来源 |
| `07_历史桌面快照_2026-09-22` | 原桌面归档目录 | 仅用于追溯 |
| `09_关键决策` | `docs/DECISIONS.md` | 防止无证据回退参数/路线 |
| `10_AGENTS项目规则` | 根目录 `AGENTS.md` | 新会话自动工作规则 |

旧桌面副本内新增 `00_历史快照_不要作为当前代码入口.md`，明确它不是当前代码。

## 只读运行状态

仓库新增：

`scripts/r1a7_context_snapshot.sh`

桌面“更新 R1-A7 只读运行状态”快捷方式调用该脚本，将时间戳快照写入：

`/home/robot/桌面/00_R1-A7_连续项目入口/runtime/`

并更新 `最新运行状态.txt` 链接。快照包含 Git、网络、8098 端口、相关进程、控制锁
和视频设备稳定链接，不发送机器人命令。

2026-10-18 首次快照确认：8098 控制台运行中；两个同名控制器 PID 是父子关系，
LowCmd 锁由父进程持有。该观察只对生成时刻有效，不能当作以后实时状态。

## ChatGPT/Codex 项目设置

本地项目应把 `/home/robot/unitree_sim_isaaclab_console` 设为主文件夹，使新对话从该
目录开始并自动发现 `AGENTS.md`。需要跨目录核对时，可把
`/home/robot/unitree_sim_isaaclab_threading` 和原始工作记录目录作为辅助文件夹；
它们不取代主工作区。

## 使用边界

- 桌面入口解决的是“事实源和恢复流程”，不是取消硬件预检。
- 实时 IP、路由、进程、LowCmd 发布者、起始姿态和相机首帧必须在真机前重查。
- `runtime/` 快照是本地运行资料，不提交 GitHub。
- 阶段完成后更新 `CURRENT_STATE.md` 与 `VALIDATION_MATRIX.md`，再提交 GitHub。
