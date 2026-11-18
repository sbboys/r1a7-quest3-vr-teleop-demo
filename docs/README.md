# 文档目录

## 当前治理入口

- [当前工作状态](CURRENT_STATE.md)
- [关键决策记录](DECISIONS.md)
- [验证矩阵](VALIDATION_MATRIX.md)
- [新对话与长对话恢复入口](NEW_CHAT_START.md)
- [2026-10-18 桌面连续工作入口整理记录](r1a7_desktop_continuity_setup_2026-10-18_zh.md)
- [项目总索引](../PROJECT_INDEX.md)
- [GitHub 分支目录表](../BRANCH_CATALOG.md)
- [2026-10-11 GitHub 全面整理报告](r1a7_github_reorganization_2026-10-11_zh.md)
- [2026-10-18 实验工作与控制台更新报告](../r1a7_threading_project/docs/r1a7_experiment_console_update_2026-10-18_zh.md)
- [2026-10-26 双相机夹线与左腕 007Z 衔接记录](../r1a7_threading_project/docs/r1a7_wire_grasp_left_wrist_sync_2026-10-26_zh.md)
- [2026-10-27 细线夹取与穿孔研究指标协议](../r1a7_threading_project/docs/r1a7_wire_threading_metrics_protocol_2026-10-27_zh.md)
- [2026-10-27 现有运动、感知与任务指标基线](../r1a7_threading_project/docs/r1a7_existing_motion_perception_task_metrics_2026-10-27_zh.md)
- [仓库管理与数据边界](repository_management_zh.md)
- [统一仓库来源矩阵](unified_source_matrix_zh.md)

## 逻辑分类

- `architecture/`：架构、模块和代码解析。
- `operations/`：运行、通信、相机、VR 和真机安全操作。
- `experiments/`：实验参数、现象、结果和结论。
- `source_records/`：Word、PPT 和聊天记录整理稿。
- `archive/`：历史文档和已结束项目说明。

历史文档暂时保留原路径，防止旧启动说明和链接失效。新增文档按上述逻辑分类；物理迁移必须放在独立重构提交中。

## 2026-11-02 连续研究补充

- [四文档核对与同步报告](../r1a7_threading_project/docs/r1a7_four_document_github_audit_2026-11-02_zh.md)
- [原件、全文、快照与哈希](../r1a7_threading_project/docs/source_records/2026-11-02/README.md)

- [2026-11-04 右腕张开镊子 V3 双模型与自动抓取归档](../r1a7_threading_project/docs/r1a7_right_wrist_open_autograsp_archive_2026-11-04_zh.md)

## 2026-11-09 研发归档补充

- [11/09研发核对](../r1a7_threading_project/docs/r1a7_grasp_insertion_archive_2026-11-09_zh.md) · [技术路线](../r1a7_threading_project/docs/r1a7_grasp_insertion_architecture_2026-11-09_zh.md) · [操作SOP](../r1a7_threading_project/docs/r1a7_grasp_insertion_sop_2026-11-09_zh.md)

## 2026-11-16 研发归档

[四文档核对](../r1a7_threading_project/docs/r1a7_four_document_research_audit_2026-11-16_zh.md) · [技术路线](../r1a7_threading_project/docs/r1a7_research_architecture_2026-11-16_zh.md) · [操作步骤](../r1a7_threading_project/docs/r1a7_research_sop_2026-11-16_zh.md)

## 2026-11-17 插孔研发调试归档

[11/17来源、代码与数据核对](../r1a7_threading_project/docs/r1a7_insertion_debug_archive_2026-11-17_zh.md)承接8daf540：新增XY Shaper V6/V7、External＋Left固定孔/FREE门及full-chain recorder；原Gate与11/16研究字节相同。输出链仅快照，未替换控制台。3428行trace已无损归档，匹配full-chain缺失，文档联合统计待原日志复核。原抓取动作属于历史反馈；FREE身份和最终实体插孔未验收。本次无设备/硬件运行。

## 2026-11-18 FREE与孔位研究归档

[11/18文档及源码核对](../r1a7_threading_project/docs/r1a7_free_hole_archive_2026-11-18_zh.md)承接7c31630。Visual已从文档b9a12787演进为c6f7b9e5，含0.75秒tip复用和Left插孔期重识孔；实际识别/运动未验收。四主输出程序哈希不变，速度基线回退可追溯；Controller的1mm/s最小合速度可能抬升低请求，待同时间日志核查。保留安全备份、失败Gate和精确diff；文档两轮trace原件缺失。输出链未部署控制台，本次无硬件运行。
