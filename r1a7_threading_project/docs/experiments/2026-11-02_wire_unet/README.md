# U-Net 与三相机历史结果摘要

本目录只保存小型 JSON：训练/验证结果、固定划分、采集元数据、三次实验summary。来源路径和SHA-256见 `../../source_records/2026-11-02/manifest.json`。

- `training/`：V1与V2 resume历史结果。不是重新训练或独立测试。
- `dataset/`：pilot_001的采集身份和sample-level划分；同布置相关性限制继续保留。
- `live_validation/`：2026-10-30 14:02、14:20、14:41三轮summary；每路556/313/368帧。
- 模型、原始RGB、masks、Pwire、逐帧完整记录均未上传。本机路径不能替代异机下载来源。

指标解释、RAW/TRACKED非完全对称口径和缺少GT几何验收的边界，见 `../../r1a7_four_document_github_audit_2026-11-02_zh.md`。
