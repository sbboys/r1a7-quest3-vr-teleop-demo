# 夹取插孔 数据标注与增量训练技术路线

承接[11/09路线](r1a7_grasp_insertion_architecture_2026-11-09_zh.md)，11/16核对范围及证据见[报告](r1a7_four_document_research_audit_2026-11-16_zh.md)。

```text
三相机RGB -> Wire模型(wire/tool) + Hole模型(board/hole)
         -> Visual V3: TIP/GRASP、FREE/孔选择与跟踪、recorder
         -> Fusion: 复合运动；最终调整区EXT/Wrist/前探/close-ready
         -> Gate: 感知/新鲜度/显式授权 -> 速度JSON
         -> Cartesian / IK / 唯一LowCmd -> 机器人

插孔原始197同步组 -> hard分析/同步去重35组 -> 联合同图双mask
 -> group split25/5/5 -> Wire V4 + Hole V2 -> 保留测试 -> 冻结
新物理孔板21同步组 -> 联合标注63视图 -> split15/3/3
 -> Wire V5(旧域replay) + Hole V3(只用新图、V2初始化)
 -> 保留测试/真实视觉核验 -> 按阶段验证闭环
```

## 控制与阶段

正常抓取保持多轴合成；最终近尖端区域才由EXT对齐、Wrist横向/走廊与前探资格分级控制。近距Z优先、Jaw Corridor与close-settle机制保留；不能把“最终调整分级”扩展为全程逐轴串行。close前还读取Fusion的sequence stage/close-ready。

GRASP→WAIT_HOLE_TARGET→INSERTION_TRACKING继续存在。External和Right Wrist是控制相机，至少一个控制视角有效的逻辑与其他相机显示分别管理；Left-only不能冒充有效控制相机。孔raw搜索中心与EMA控制目标分离，HOLD或TIP-relative推断不算fresh测量。FREE-HOLD停动必须沿同时间的perception→Fusion→Gate→controller追踪，不能只凭屏幕误差归因。

## 数据和模型分层

Wire mask的1/2是细线/镊子，Hole mask的1/2是孔板/孔；同RGB双mask分别存放。旧联合35组和新板21组分别冻结group split。三相机同group不可跨集合；新板全scene_000，属pilot。

Wire V4由V3初始化，回放Pilot/Open/Insertion；V5由V4初始化，再加NEW_BOARD_01及四域guard。Hole V2回放旧板和Insertion；Hole V3加载V2权重，但训练数据没有旧板replay。因此两类模型旧域退化风险不同，不能用Wire指标证明孔位已可靠。

## 运行时与归档

当前Visual变量名WIRE_V4_CHECKPOINT实际指V5，HOLE_CHECKPOINT指V3；Hole在后续阶段lazy-load。默认HOLE_MIN_AREA=15，而viewer曾以1做诊断，不能只换模型就认为后处理相同。当前仅归档路径与静态实现，没有实际加载/运动验收。

recorder按阶段写同步图像、records和事件，root可由环境变量指定。`/data/threading`的trial/source/model hash应与每次运行绑定。库中保留小型摘要与标注/模型指标，原始数据和checkpoint通过外置清单管理；研究输出链位于档案current目录，默认控制台未被替换。
