# 四份研发文档与GitHub连续性核对 2026年11月16日

比较基线 `e5138d6e79084de9326110485eb690f522988d71`，目标仓库 `sbboys/r1a7-quest3-vr-teleop-demo`，研发分支 `codex/r1a7-console-camera-suite-2026-10-11`。本次按既有授权继续归档与快进合并；在独立工作区整理，不清理研究区、不更换现用控制台输出链。

## 核对结论与阅读入口

四份文档在研究路线上连贯，后两阶段新增内容尚未上传。11/03与11/09原件已在基线中逐字节相同，复用原件和全文，不重复上传；11/13与11/16原件、全文和一幅Gate截图新增归档。

- 11/03：右腕OPEN数据、Wire V3、midpoint及自动OPEN/CLOSE研究。
- 11/09：Jaw Corridor、near-Z、抓取到选孔与插孔、Hole V1。
- 11/13：197组原始记录→35同步组联合标注→Wire V4/Hole V2→试验数据中心。
- 11/16：NEW_BOARD_01的21组/63图、Hole V3、Wire V5、模型接入及Gate未授权/暂停问题。

[四份来源与全文](source_records/2026-11-16_research/README.md) · [技术路线和模型结果](r1a7_research_architecture_2026-11-16_zh.md) · [具体操作与排查步骤](r1a7_research_sop_2026-11-16_zh.md) · [小型历史证据](experiments/2026-11-16_research/README.md)。

原始Word仅作为证据，历史补丁、清理、训练和运动命令不构成本次执行请求。本次没有重新训练、推理、取流、启用Gate、DDS/LowCmd、机器人动作或修复控制逻辑；原Word未改写、未重新验收版式。

## 内容相同与缺失

| 文档 | GitHub基线 | 本次处理 |
|---|---|---|
| 2026.11.3右腕张开镊子记录 | SHA-256相同 | 复用 `2026-11-03_right_wrist_open` 原件/全文 |
| 截至2026-11-09 V2.0 | SHA-256相同 | 复用 `2026-11-09_grasp_insertion` 原件/全文 |
| 20261113技术路线与实验 | 缺失 | 新增原件、全文、关键源码与训练/数据证据 |
| 20261116完整对话整理 | 缺失 | 新增原件、全文、原截图及当前状态差异 |

完整路径、大小、哈希及外置资产见manifest。原文档保留当日状态；后续源码变化和数值纠正写在新报告，不静默改写来源。

## 当前源码比文档更进一步的项目

- 当前Visual V3的 `WIRE_V4_CHECKPOINT` 实际指向冻结Wire V5 epoch123（保留旧变量名）；`HOLE_CHECKPOINT`指向Hole V3 epoch64。模型替换前的时间戳备份确实存在并被保留。因此应从“只有替换方案”更新为“当前路径已写入、语法通过”；尚缺实际加载日志和闭环验收。Hole仍lazy-load，启动只看到Wire不能证明Hole已加载。
- `DYNAMIC_SELECTED_HOLE_TRACK_V1`不只是标记：选孔后保留raw中心，下一帧优先用latest_holes里的raw center搜索，EMA target用于控制。相机分别ROI/jump/EMA参数存在。源码实现不代替运动时目标身份/稳定性验收。
- Optional control camera逻辑已存在，控制相机为External和Right Wrist，Left用于监视。旧11/09“三相机必须全部选孔”是历史状态，不能再当作当前唯一准入规则。
- Recorder已接入，支持 `R1A7_INSERTION_DATASET_ROOT`。保留197 records/591 images、0 dropped_queue的历史summary；这不证明后续新session已经写入/data指定trial。
- 1/2/3/N重选、通用人工P/R暂停和单孔后处理并未据当前核对获得完整实施/验收证据；不补写这些功能，也不擅自将V4 candidate替代V3默认入口。已有VR暂停提示也不能冒充任意阶段的键盘保持功能。

## Gate False的静态补充

原截图可见DUAL_READY、来源OK、新鲜age和非零速度，同时enabled=False。当前Gate源码为 `output_enabled = bool(args.enable_output and perception_ok)`，CLI的 `--enable-output` 是默认关闭的store_true参数。11/16文档的启动示例省略该参数，这与截图现象相容，但没有当时实际命令行/同时间JSON，不能认定它就是现场唯一根因。本次不改为强制True，也不执行授权或运动。

## 模型证据与数字纠正

| 模型 | 候选epoch | 保留测试的关键量 | 限制 |
|---|---:|---|---|
| Wire V4 | 110 | insertion mean FG Dice 0.849347 | Pilot test轻微下降；test已查看 |
| Hole V2 | 52 | insertion Hole Dice 0.770904；score 0.796365 | 旧External Board明显退化 |
| Wire V5 | 123 | 新板mean FG Dice 0.902626；wire 0.868815；tool 0.936436 | 同session/scene_000，非跨场景证明 |
| Hole V3 | 64 | 新板score 0.375648；mean FG Dice 0.480038；External Hole Dice 0 | 不能作为稳定识孔已验收 |

11/16文档8.2将Hole V2旧test的0.829813→0.825648称为mean FG；原JSON显示它们是score。真实mean FG为0.874706→0.858639。此次明确纠正指标名称，原文档不改写。Wire/Hole两套独立三分类模型不是共享encoder双head；两张mask类别编号相同、语义不同。

新旧联合数据split为35组的25/5/5；NEW_BOARD_01为21组的15/3/3，Wire V5与Hole V3使用相同新板split。新板audit_v2为63视图、pass；旧联合资料的Hole nonempty104/105仍是待复核项。单session时间点分组避免同一时刻三相机跨split，但不能消除同场景近重复带来的泛化限制。

4份checkpoint与2个冻结GT包本次重新计算SHA-256，均与文档记录相同。历史训练/测试JSON与摘要按原字节保留，不读取checkpoint pickle或重跑推理。

## 源码 新增 修改 删除与备份

正常研究目录补入10个缺失程序（recorder、分析、去重、联合标注、采集、4个增量训练器），更新既有Visual V3，共11份源码；保留原相互导入路径。完整输出链当前四脚本另存到 `research_code_snapshot/current/`，相对e5138d6的精确diff可追溯。

单独保存V4 candidate及5份关键before-change源码备份，均不设为默认入口。原位Fusion、Gate、普通控制台controller保持旧集成版本；因此归档合并不等于已经将新研究控制策略部署进控制台。

本次没有删除正式源文件或清理symlink/原数据。文档中历史清理/tmp、重建training_runtime symlink和移动trial目录只按历史记录保留，没有在本次执行；无法把当前diff重建成每一次历史编辑记录。

## 数据边界与验证

上传59份来源文件（2个新原件、21份源码/快照/备份、36份小型数据/指标文件），另外复用两份旧原件；保存2份新全文、原截图、技术/SOP、差异及校验工具。504个选定RGB/mask和6份大模型/GT包只登记哈希、大小、路径与获取方式；完整原始会话图片、全部控制日志、模型与GT压缩包不进入普通Git历史。仅本地路径不能算异机下载或完整远端备份。

可复核命令：`python3 scripts/verify_r1a7_research_archive_20261116.py`；`R1A7_DIFF_BASE=e5138d6 bash scripts/repository_sanity_check.sh`；`git diff --check`。验证文件及Git索引完整性、全文覆盖、源码不导入编译、split、JSON指标和链接、原位控制代码不变。全文转录仅清理行末空白，原Word字节不变。CRLF来源CSV和采集脚本原有文件末尾空行按精确路径gitattributes保留源字节，其他空白检查继续保留。老归档校验器若针对已演进的同路径源码，应在对应历史commit工作区复核。

当前未解决：新模型真实加载回执、Hole V3错孔/闪烁、Gate实际启动授权、FREE-HOLD同时间链路归因、通用人工HOLD、最终物理夹线/入孔/出线及重复率。NEW_BOARD_02仍按文档规划管理，不宣称已采集训练。
