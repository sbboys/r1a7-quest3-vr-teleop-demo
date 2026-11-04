# 右腕张开镊子视觉与自动抓取工作归档

工作记录日期：2026-11-03；本次文件核对与归档日期：2026-11-04（Asia/Shanghai）。比较基线为 `c1f466acdac92cbc13d5f2cdbb1c92c10dc247ab`，承接四文档归档与合并成果。仓库为 `sbboys/r1a7-quest3-vr-teleop-demo`，研发目标分支为 `codex/r1a7-console-camera-suite-2026-10-11`。

## 结论与证据范围

11/03记录连续推进了11/02的三相机U-Net与TIP→GRASP研究：针对右腕张开镊子分割退化，新增Orbbec RGB采集、单右腕标注、旧域replay的V3训练与9图测试，再接入仅右腕V3的双模型视觉，以及张开双尖端中点TIP和直接自动抓取控制研究。

本文区分三个时点：原文档形成时的状态、本次读取的当前文件、运行或硬件验收。当前源码比文档末尾更进一步：midpoint模块已存在并被前端接入，Right Wrist `ev` CLOSE条件已从研究控制器删除。二者均不能因此认定真机验收完成。

本次只读取、复制、比较与静态核验文件，并发布归档；没有训练、推理、相机取流、Gate输出、IK、DDS/LowCmd或机器人动作。文档中的历史指令、计划和阈值不作为本次执行授权。

## 原始来源和完整操作过程

原始DOCX、全文、源码/结果哈希及保留的历史备份见[来源入口](source_records/2026-11-03_right_wrist_open/README.md)。全文保留正文、表格、历史启动命令、参数、键位、错误分析、回滚和后续建议；原件字节不改动。原件没有嵌入图片；本次没有重排或验收Word版式。

按文档及文件核对的工作流程：

1. **故障定位。** 文档记载自动OPEN后停留OPEN_TRACKING，右腕tool/TIP退化或误差/jitter不满足CLOSE。文档将V2缺少张开镊子外观作为主要视觉原因；不能据此排除所有状态机与几何问题。
2. **采集。** V4L2早期采集出现异常画面，正式采集改用 `capture_right_wrist_open_orbbec_v2.py`，锁定CPCBC53000C5、COLOR_SENSOR、1280×800 RGB@30。历史失败采集程序另作快照保留，不能误当推荐入口。
3. **标注和QA。** 新增Right-Wrist-only标注器，保留原三相机v2标注器作对照。类别0背景、1可见wire、2可见tool、255无法判断；两镊臂间空气不填tool、遮挡线不补画。当前64图/64mask，QA64行，ignore均为0。000043/000046按文档为人工确认的真实遮挡hard case；本次未重新独立审图。
4. **分组。** 六个时间块分配47 train / 8 val / 9 test；实际训练使用 `splits_right_wrist_open_grouped_v1.json`。另一个sample-level文件是标注工具读取的历史文件，不替代正式训练划分。
5. **冻结和训练。** 旧V2与init_v2_best哈希相同；V3使用旧105图replay和新47图×3=141项，每epoch246项，batch2、crop512、stride384、lr2e-5、AdamW、重置optimizer moments。旧/新验证分数按0.4/0.6加权，old_guard下限baseline−0.02。历史20轮覆盖epoch73–92，best90。
6. **评估。** V2/V3对同一9图test评估，保存总体及逐图指标。此次保留指标文件和脚本，未重新加载模型推理。
7. **双模型部署源码。** `live_visual_guidance_shadow_v1.py`保留External/Left的V2，仅Right Wrist使用V3；当前代码还引用了右腕midpoint模块。模块源码按原路径更新，训练权重继续外置。
8. **TIP语义。** 当前右腕模块使用两个显著tool连通分量、各自前沿尖端和中点；不足条件时回退原单尖端。它是图像结构启发式，不是通过LowState直接确认夹爪物理OPEN/CLOSED；单镊臂遮挡时可能回退。
9. **直接自动抓取。** 源码快照含CLOSED_MONITOR→WAIT_OPEN→OPEN_TRACKING→WAIT_CONTACT/DONE等阶段。当前CLOSE条件保留External eu/ev/jitter、Right Wrist eu/jitter及0.40s窗口，不再检查Right Wrist ev。手动O/C仍走gripper_command路径，自动latch独立，可能随后再次发出自动CLOSE。
10. **后续与停止。** 文档中的HOME打断窗口、遮挡下TIP/GRASP正确性、CLOSE/接触/DONE、数据新鲜度和控制退出仍需验证。历史停止顺序Controller→Gate→Fusion→Vision仅作记录；部署前必须按当前文件与工位核验，不从归档直接执行。

## 文档状态与当前文件的对照

| 工作项 | 文档形成时 | 2026-11-04当前文件核对 | 验收边界 |
|---|---|---|---|
| V3训练与9图测试 | 已完成 | 历史JSON数值与文档相符，best90 | 没有重新训练/推理 |
| 双模型 | 已改+语法通过 | V3常量、加载函数和Right Wrist模型选择存在 | 未重启实时视觉 |
| 张开双尖端中点TIP | 设计/待确认部署 | 文件存在，前端仅为Right Wrist构造新tracker | 静态存在不等于OPEN识别正确或已运行 |
| 取消Right Wrist ev CLOSE条件 | 待执行/待验证 | 当前wrist_close只有eu与jitter | 与前备份相比只删除相关8行，未重新真机验收 |
| HOME问题 | 待排查 | 本次没有修复或证明根因 | 保留未解决 |
| 自动闭爪 | 前期旧记录仅方案 | 新研究控制器已有自动OPEN/CLOSE状态机源码 | 文档记载OPEN发生过，缺完整CLOSE→DONE验收 |

旧状态文档的“自动闭爪未实现”只适用于原控制台集成线/旧日期，不能覆盖本次新增研究控制器。新源码存在也不能改写成控制台已集成自动抓取或真机成功。

## 重新核对的指标与限制

| 指标 | V2 | V3 |
|---|---:|---:|
| 新域test wire Dice | 0.449626 | 0.566445 |
| 新域test tool Dice | 0.569188 | 0.891532 |
| 新域test mean FG Dice | 0.509407 | 0.728989 |
| 旧域val mean FG Dice | 0.750200 | 0.752381 |
| 新域val mean FG Dice | 0.477433 | 0.772780 |

结果文件见[结果入口](experiments/2026-11-03_right_wrist_open/README.md)。V3整体old_val略升且20轮old_guard均通过，但并非所有旧类别/视角都不下降：旧域tool Dice由约0.934632降至0.926855。因此文档“旧域无退化”在本归档中限定为总体均值与guard，不作为逐类全面结论。

9图test与train/val按已保存时间块分离，当前test ID无交集；它仍来自同日小规模采集，不能据此证明新场景泛化、TIP几何误差、运动稳定性或自动抓取成功率。

## 新增 修改 删除和回滚

- 新增正常研究源码6个：Orbbec采集、V3训练、V2/V3评估、右腕midpoint、原v2标注器、单右腕标注器。
- 修改已有研究源码1个：实时视觉前端，保存当前完整文件；相对c1f466a及pre-V3备份的精确diff均保留。
- 另存4个源码档案：直接auto-grasp控制器、失败V4L2采集、pre-V3视觉备份、pre-remove-wrist-ev控制器备份。仅存 `research_code_snapshot/`，不部署原位控制器。
- 逻辑删除：研究控制器wrist_close的ev判据；对保存的pre-remove备份进行差异核对。没有在本次仓库提交中删除源文件，不把已保存版本差异当作全部历史变更。
- 原位Fusion、Gate、控制台controller均保持上一集成版本。两个控制依赖 `visual_servo_receiver.py`、`r1a7_lowcmd_guard.py` 与研究区字节相同；外部IK/SDK依赖未打包。
- 回滚来源有原备份与Git父提交；本次没有实际恢复或运行备份。

## 上传和外置数据边界

上传原件、全文、报告、必要源码、精确diff、两个历史源码备份，以及9份小型结果/划分/QA/采集/标注记录。保留原始编码和哈希。两份来源CSV使用CRLF，通过仅限这两条路径的 `.gitattributes` 保留原字节并识别CRLF换行，不改写源记录或关闭其他空白检查。

外置清单逐文件记录正式session的64张原RGB、64张整理images、64张masks、4张QA图和9张test对比图，共205个图像文件，以及V2、冻结init、V3三个checkpoint。权重约93MB/个；原始数据、图像、权重、全量机器人日志不进入本次普通Git提交。其他失败采集session和标注临时备份不作为正式训练集上传或打包。

V2和冻结init均为 `cb7ae5f4664bb5f1a558beeb941a49847cc1694f4d343024391973d4052c1687`；V3为 `2130e7c1d0ccd6b795f17e60dcb58105c00e8adb412b39573804fb48da472e30`。本地路径是定位线索，不是远端数据备份或异机下载地址；本次归档尚不等于可离线完整复现实验。

## 本次验证

可复核命令：`python3 scripts/verify_r1a7_right_wrist_archive_20261104.py`；`R1A7_DIFF_BASE=c1f466a bash scripts/repository_sanity_check.sh`；`git diff --check`。

验证来源哈希、Python语法（不导入运行）、全文覆盖、分组与样本数量、QA数值、历史指标、备份差异、本地Markdown链接和新增大文件边界。检查不验证依赖安装、模型推理、相机、动作或抓取成功。实际结果保存在本次结果目录的 `archive_verification.json`。
