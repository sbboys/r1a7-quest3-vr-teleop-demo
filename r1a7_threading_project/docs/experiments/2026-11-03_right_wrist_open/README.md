# Right Wrist OPEN 训练和测试历史记录

- `runs/v3_mixed_replay/`：baseline、best与20轮history；best epoch90，历史73–92。
- `eval_v2_v3_test/metrics.json`：9图逐图及总体V2/V3指标。
- `splits_right_wrist_open_grouped_v1.json`：正式时间块47/8/9划分；sample-level文件仅保留工具来源。
- `qa_mask_stats.csv`：64图像素计数、尺寸与ignore；`metadata/`为正式session采集记录；`annotation_log.jsonl`为标注过程记录。
- `archive_verification.json`：本次不运行模型/设备的归档检查结果。

原始图像、mask、QA/test对比图和权重仅在 [清单](../../source_records/2026-11-03_right_wrist_open/manifest.json) 中登记本地路径、大小与SHA-256。指标是历史文件核对，不是本次重跑推理，也不是抓取成功率。
