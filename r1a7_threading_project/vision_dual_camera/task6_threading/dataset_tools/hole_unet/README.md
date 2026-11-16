# 孔位U-Net研究模块

## 2026-11-09 夹取到选孔研究

孔位标注v1/v2/v3、QA/复核、三分类训练、三相机孔检测和插孔感知。正式配置及历史结果见项目docs，独立test尚未补齐。

技术与操作入口：项目 `docs/r1a7_grasp_insertion_archive_2026-11-09_zh.md`。这些程序可能写共享运动JSON或启动相机；不要通过import或随意启动验证语法。模型外置，现有控制台默认控制器没有替换。

## 2026-11-16 后续研究

新增recorder/筛选/联合标注与Wire V4/V5、Hole V2/V3/新板采集。当前Visual V3原变量名WIRE_V4_CHECKPOINT实际指Wire V5，Hole指V3；属于实验源码更新，不是控制台自动部署。模型/数据外置，candidate和输出链快照只归档。最新核对入口为项目 `docs/r1a7_four_document_research_audit_2026-11-16_zh.md`。
