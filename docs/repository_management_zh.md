# R1-A7 GitHub 仓库审查与管理方案

## 审查结论

当前仓库不是单一产品仓库，而是 R1-A7 长周期研发工作台，包含：

1. Unitree Isaac Lab 基础仿真；
2. Quest 3 VR 遥操作；
3. Gemini、腕部 RGB 和固定相机调试；
4. 扳手抓取和示教迁移；
5. 细线、镊子和穿孔任务；
6. R1-A7 手臂 3D 控制台；
7. ACT 复现与 R1-A7 迁移；
8. 双相机标定、视觉检测和视觉伺服；
9. 实验记录、PPT、视频、标定数据和运行日志。

目前这些内容通过分支快照保存，项目边界没有完全体现在目录和 README 中。主要风险是：同一文件在多个快照中重复、实验代码和稳定入口混在一起、数据文件进入代码分支、不同分支使用不同目录体系。

## 分类原则

### 代码

可复现、会被运行或导入的 Python、C++、JavaScript、HTML、Shell、配置和机器人模型文件，放在项目目录或共享运行时目录。

### 文档

启动手册、故障排查、研究方案、实验报告、代码解析报告和操作记录，统一进入 `docs/` 或对应项目的 `docs/`。

### 结果

少量可复核的 JSON、CSV、标定结果可以保留；视频、原始图像、录制包、模型权重和 checkpoint 使用 manifest 保存。

### 历史内容

旧脚本、失败版本、备份文件和一次性探针放入 `archive/`，不进入稳定入口，不作为新代码复制来源。

## 建议的最终目录

现有路径暂不立即移动。稳定后可在重构分支中逐步调整为：

```text
projects/
  r1a7_threading/
  r1a7_act_reproduction/
  r1a7_arm_console/
  r1a7_vision_servo/
  r1a7_wrench/
shared/
  simulation/
  dds/
  robots/
  teleoperation/
docs/
  architecture/
  operations/
  experiments/
  source_records/
  archive/
datasets/
checkpoints/
third_party/
tools/
```

目录移动必须单独提交，并同步更新启动脚本、Python 导入路径和文档链接；在此之前使用索引进行“软分类”。

## 大文件处理

审查发现部分后期快照包含几十 MB 的视频、状态 CSV 和 PPTX。后续规则：

- 代码和小型配置直接提交；
- checkpoint、视频和大规模数据放在外部存储或 Git LFS；
- 仓库只提交 `datasets/README.md` 或 `checkpoints/README.md`；
- manifest 必须包含文件名、大小、SHA256、采集日期、设备、环境和获取方式；
- 不对旧历史立即执行 `filter-repo`，避免破坏现有提交链接。历史清理另开任务处理。

## 真机安全边界

任何会发布 `rt/lowcmd` 的脚本都必须在 README 中注明：

- 是否连接真实机器人；
- 使用的网卡和 IP；
- 是否需要停止其他控制程序；
- 启动、停止和急停方式；
- 是否会自动移动机械臂或夹爪。

仿真、只读状态、VR 遥操作和真机控制必须分别标注，不能只依靠文件名判断风险。

