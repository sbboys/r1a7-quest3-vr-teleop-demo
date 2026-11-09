# 截至11月9日夹取与穿孔研发归档核对

日期：2026-11-09（Asia/Shanghai）。仓库 `sbboys/r1a7-quest3-vr-teleop-demo`；目标研发分支 `codex/r1a7-console-camera-suite-2026-10-11`；比较基线 `4498710adde23d65a883fcc2815245de49fdf63d`。本次继续已授权的关键代码、技术路线、流程和具体步骤上传，使用独立工作区，不清理或改写原研究目录。

## 结论和入口

本次新增夹取到选孔/插孔的视觉代码、孔位采集标注训练QA代码，保存近距离External Z优先、Jaw Corridor、CLOSE_SETTLE锁存、Gate修改及四份同时间成功命名源码快照。技术与操作分别见[技术路线](r1a7_grasp_insertion_architecture_2026-11-09_zh.md)、[操作步骤](r1a7_grasp_insertion_sop_2026-11-09_zh.md)、[原件全文和清单](source_records/2026-11-09_grasp_insertion/README.md)。

用户给出的下载路径已不存在；在 `/home/robot/桌面/jiqiren/renwu/` 找到同名V2.0文档，SHA-256为 `c7f10261c29c0b7f9a34b98f70d9c2c3d616a6bd97dc95385259c6a92a3df611`。保留原件及全部正文/表格/附录转录，不重排原Word。原件没有嵌入图片，本次不宣称完成版式验收。

## 文档 当前文件与验证等级

| 事项 | 文档记载 | 本次当前文件核对 | 仍不能推断 |
|---|---|---|---|
| 四终端夹取和Jaw Corridor | 用户反馈正常，缺完整现场日志 | 当前四脚本及4份SUCCESS命名快照存在，全部逐字节相同 | 客观重复夹取成功率、夹持力或本次真机验收 |
| 孔位训练 | 数量/训练表/checkpoint待补录 | 找到split、config、40轮history、QA和2份checkpoint | 仅凭val不能填独立test或穿孔成功率 |
| External近距离Z优先 | 45/65px滞回 | 抓取阶段near锁存及诊断字段存在 | Wrist/External几何必然准确 |
| Jaw Corridor | 12样本、中间80%、首次进入锁存 | V3参数与字段存在，controller使用jaw_corridor_latched | 线处于正确纵深/高度或多轮reset已通过 |
| Gate 350px硬阻断删除 | 11/07补丁与语法回执 | 常量仍保留，但模式有效性分支不再使用误差阈值 | 任意大误差可信或放宽保护已充分验证 |
| BLIND_GRACE | 0.40s、比例0.40、2mm/s上限 | Fusion参数和Gate模式分支存在 | 它仍是短时无新视觉修正运动，不是新观测 |
| 插孔选孔确认 | 三相机同一实体孔 | E/Enter/Space确认、R重选、阶段JSON与FREE来源区分存在 | 三视角选择必然是同一物理孔或已穿出 |
| CLOSE_SETTLE退场 | 存在卡住风险 | 当前Gate含插孔确认后停用grasp hold的逻辑 | 状态机所有异常退出和旧JSON风险已解决 |

SUCCESS文件名是历史来源标记，文档的用户反馈作为间接证据保留；本次没有执行训练、推理、相机、Gate、DDS/LowCmd或机器人动作。没有把进入INSERTION_TRACKING写成穿孔成功。

## 本次补齐的孔位训练证据

数据根目录 `/home/robot/r1a7_datasets/wire_unet/hole_tube_001`。正式split为36组×3相机=108个图像项，train/val/test分别24/6/6组，即72/18/18图像项。保留原划分，不把右腕OPEN的64图与孔位数据混用。

config记录UNet GroupNorm、base32、背景/board/hole三分类、255忽略；crop512，采样hole/board/random=0.65/0.25/0.10；类别权重0.2/1/5；0.5 CE+0.5前景Dice；选模分数0.70 HoleDice+0.30 MeanFGDice。trainer默认batch2、lr3e-4、40轮，保存参数不等于当时完整命令行已被恢复。

history.csv记录epoch1–40；按val_score最大行得到epoch38：board Dice=0.958467、hole Dice=0.855727、mean FG Dice=0.907097、score=0.871138。该行三视角hole Dice为External0.847202、Right Wrist0.844343、Left Wrist0.867838。它们是历史验证集指标；本次未读取checkpoint的pickle对象，也未重新推理，不单独认定best.pt内部epoch或独立test精度。

实际结果和补录摘要见[孔位结果目录](experiments/2026-11-09_hole_unet/README.md)。没有发现独立孔位test指标文件；test划分存在与test已评估是两回事。标注/QA、checkpoint路径及哈希由清单保留；原始像素数据与模型外置。

## 关键代码 新增 修改 删除和回滚

- 按原相对路径纳入全部8个 `hole_unet/*.py`（标注v1/v2/v3、训练、三相机孔检测、插孔感知、QA/复核），以及grasp→insertion v1/v2/v3和三相机采集，共12份源码。V1/V2为演进参考，V3为本次文档的主要视觉入口。
- 原位 `live_visual_guidance_shadow_v1.py` 和依赖没有变化；V3通过它复用旧wire/tool双模型与midpoint。孔位模块通过相邻目录导入；不移动路径破坏依赖。
- Fusion、Gate和直接自动抓取controller放在 `research_code_snapshot/current/`，按原相对路径完整保存；**未覆盖正常控制台同路径文件**。从仓库默认入口直接启动并不会自动得到这套研究策略。
- 四份 `success_20261109_171103/` 快照与当前四脚本字节相同；保存四份关键pre-change备份（11/07误差门控、11/09 Z优先与两份Jaw前版本），并提供相对Git基线和备份的精确 `.diff.json`。
- 逻辑变更包括350px硬STOP条件移除、BLIND_GRACE放行、Wrist点误差判据替换为corridor latch、settle重检方式变化。原始常量和注释可能保留，不用关键词存在性替代真实分支检查。
- 本次不删除任何源文件；历史已删除但未保留的内容无法从当前文件完整重建。回滚以Git父提交及保留备份为依据，没有实际替换运行版本。

## 上传边界和检查

上传原件、全文、三个专门说明文档、上述源码/快照/备份、差异、孔位小型元数据与历史结果。原图/mask/QA图片和两个 `.pt` 只登记路径、大小、SHA-256及获取方式，不进入普通Git；先前wire V2/V3模型继续复用既有外置清单。未打包外部IK/SDK环境或完整机器人运行日志，不能称为异机完整离线复现包。

检查命令：`python3 scripts/verify_r1a7_grasp_insertion_archive_20261109.py`；`R1A7_DIFF_BASE=4498710 bash scripts/repository_sanity_check.sh`；`git diff --check`。检查来源哈希/索引跟踪、全文覆盖、源码语法、四快照一致、数据划分与history、相对链接、大文件边界和原位控制文件不变。原始CSV的CRLF通过精确路径gitattributes保留，其他空白检查不取消。

后续尚需：多轮jaw复位/异常取消、CLOSE_SETTLE退出、目标跨视角身份、模型独立test、FREE来源可靠性、停止距离和真实入孔/出线证据。静态归档不增加硬件验收等级。
