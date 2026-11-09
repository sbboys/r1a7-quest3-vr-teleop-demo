# 孔位U-Net来源数据与历史结果

保存group split、config、40轮history、QA统计、四份采集metadata、v3标注几何/日志，以及 `history_summary.json` 补录摘要。训练36组×3相机，train/val/test=24/6/6组。history最大val_score对应epoch38，不能代替独立test。

原始训练图、masks_v3、QA图和best/last权重仅登记 [外置清单](../../source_records/2026-11-09_grasp_insertion/manifest.json)。不保证网络下载来源；同机路径不是远端备份。此次不读取checkpoint的pickle对象，不重新训练或推理。
