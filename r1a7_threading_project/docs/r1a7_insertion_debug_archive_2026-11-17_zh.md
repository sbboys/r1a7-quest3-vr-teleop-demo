# 2026年11月17日抓取与插孔研发归档核对

## 结论与连续性

本次从远端研发分支 `codex/r1a7-console-camera-suite-2026-10-11` 的 `8daf540484e38412a74b2bd2a733262b91dcab21` 开始核对。新DOCX不是旧文档的重复副本，承接11/16的新板模型与抓取状态机，新增插孔控制迭代、实际状态与全链路记录、FREE身份问题。原件、407段全文和2张原图均保留。全文按段落/表格顺序转录；仅去行末空白，代码段内换行已保留；DOCX未改写，未做整份Word版式渲染。

[来源入口与源码清单](source_records/2026-11-17_insertion_debug/README.md)；[精确哈希及变化分类](source_records/2026-11-17_insertion_debug/manifest.json)；[技术路线与操作步骤](r1a7_insertion_debug_sop_2026-11-17_zh.md)。

## 与11月16日归档的差异

| 对象 | 当前文件核对 | 处理 |
|---|---|---|
| Visual V3 | 修改；插孔改External＋Left，Right监视；固定孔目标；FREE 14px门 | 同步既有研究视觉路径，同时保存完整研究快照和diff |
| Fusion | 修改；当前写preinsert，不再直接写shadow | 快照及diff，不覆盖控制台原位Fusion |
| XY Shaper | 新增；含V3身份锁、V6连续XYZ、V7旧DONE重定向 | 新研究模块保存快照，未部署默认控制台 |
| 原Gate | 与11/16研究快照SHA完全相同 | 保存相同字节，确认“回到原Gate”不等于换Direct Gate |
| direct Controller | 修改；actual-state、full-chain、reanchor请求路径 | 快照及diff，保留原夹爪目标与状态机，不作真机复测 |
| FullChainAsyncRecorder | 新增；Controller真实导入依赖 | 与控制器按原相对目录一并保存 |
| Direct Gate / topdown supervisor | 当日失败/替代尝试 | historical_candidates，不作默认入口 |
| 时间戳备份 | 当日相关22份 | before_change按原路径保存，不清理研究区 |

保存32份来源文件，包括原件、当前源码、候选和备份以及正常路径视觉副本。新增/修改分类以manifest的source_changes为准；所有备份仅证明文件存在，不证明每份曾运行。无正式源文件删除；状态转移和累计下降限制的逻辑移除通过精确diff公开。历史每一次补丁并不一定都有独立备份，不能把端点diff等同完整逐次编辑日志。

## 文档待确认项已据本机补充

- 当前Fusion路径确认为 `/tmp/r1a7_wire_grasp_fusion_preinsert_v1.json`；Shaper active输出为shadow；默认不带 `--enable-output` 时只写preview。因此11/16的四终端直连模板不能直接用于这套当前研究文件，否则原Gate可能没有新鲜输入。
- Visual的 `INSERTION_CONTROL_CAMERAS` 为external、left_wrist。固定孔在插孔期保持确认的目标坐标；FREE持续更新。Wire V5 epoch123/Hole V3 epoch64配置承接前档；不新增权重加载或识孔可靠性结论。
- `FIXED_CAMERA_FREE_IDENTITY_V2` 与External/Left=14px存在；Shaper `FIXED_FREE_CONTROL_LOCK_V3` 为14px、5px簇、4次确认、EMA .25。循环确认次数不应自动当作4个独立新相机帧，独立frame去重需另验。
- V6实际存在且FINAL_INSERT中Z死区只归零Z，XY继续；left_dv正时负Z、负时可用最多.02mm/s正Z恢复。符号仍须现场标定；源码注释不是物理方向验收。
- V7在状态机入口把INSERTION_DONE重定向FINAL_INSERT；仍有遗留elif DONE，不能仅凭grep命中断言旧终止状态仍为正常可达路径。
- 未找到 `FIXED_FREE_LOCK_ONCE_V5`，不将8px/6px锁一次提议写成已实施。TAPNext++、Track-On-R与grasp-anchor/geodesic升级仍为候选，不添加实现或安装记录。
- 文档的推理性能表按历史报告保留；当前主脚本未找到torch.compile/ThreadPool实施标记，未保留当时独立benchmark，不能把性能数字当作本次已复测指标。

## 18:42数据及不可复算边界

本机18:42 Cartesian trace确有3428数据行、18列、跨度114.251626秒，已无损gzip归档并记录解压后原始哈希。整段trace包括抓取与准备，不能用整段XYZ差代替文档的插孔子段差。

文档称full-chain也是3428行、插孔2454行、Gate Z全零，并提供两张图；本次只作为来源结论保留。当前 `/tmp/r1a7_official_real_lowcmd_full_chain.csv` 已是另一时段（sample_monotonic起点92973.68，而trace起点78831.73），在搜索的研究日志、桌面资料、下载和当前工作区未找到18:42匹配副本。故未用不匹配日志拼接或伪造插孔区间，无法独立复算2454行、饱和速度中位及插孔位移。原图只证明文档所展示图像，不替代缺失原始CSV。

[trace摘要](experiments/2026-11-17_insertion_debug/trace_summary.json)保留整段统计与口径。权重、原视频和所有试验大日志未批量进入普通Git；18:42 trace已上载可下载本体，其余仅本地路径不算远端备份。

## 验证等级与后续

本次执行原件/快照SHA核验、不导入源码的compile、原文覆盖、图片/trace完整性、当前与旧快照diff、仓库检查。未运行相机、推理、Shaper、Gate、DDS或LowCmd，没有本次新硬件实验。

文档明确记录原始抓取链曾恢复手臂/夹爪动作，保留为历史操作员反馈；FREE仍可能指向折点/镊子附近，XY/Z像素解耦与实际耦合、取消累计Z限后的碰撞风险、时序/失效保护都未验收。没有最终物理插孔成功及重复率结论。

下一步依次为冻结完整同轮源码与日志、标注真实FREE、离线端点身份和运动残差验证、逐轴方向验证、同时间链路停止/失效检查，再进行独立授权的受控短程实验。不得把文档建议当成当前部署任务。
