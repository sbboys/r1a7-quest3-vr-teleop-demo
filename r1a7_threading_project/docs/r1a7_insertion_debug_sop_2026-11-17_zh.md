# 2026年11月17日插孔研究技术路线与操作步骤

本文件是归档与复核流程，历史使能命令见来源全文，不在归档任务中执行。

## 两套拓扑与版本选择

```text
历史原始抓取：Visual → Fusion(shadow) → 原Gate → Controller
当前研究插孔：Visual → Fusion(preinsert) → Shaper(active→shadow) → 原Gate → Controller
诊断预览：Fusion(preinsert) → Shaper(默认→preview)，不会补写Gate输入
```

GRASP仍为External＋Right；Shaper GRASP_PASS透传上游对象。抓取后WAIT_HOLE_TARGET在External/Left选择同一物理孔，确认进入INSERTION_TRACKING；固定孔目标、只更新FREE。Right保留监视。源码快照是完整研究链，不能混用控制台原位同名旧程序，不能同时开两个shadow writer。Direct Gate与topdown supervisor均为历史候选，禁止作为本文件默认入口。

XY采用External du→X、Left du→Y；coarse 30/15px、fine10/6px、return45/25px、稳定5周期。误差为hole-free；实际符号、耦合须独立验证。APPROACH_XY只XY，APPROACH_3D按XY质量放负Z，FINAL_INSERT继续XY并按Left dv放Z，8px死区只停Z；V7不再用DONE终止。没有由此证明实体已插入。

## 可直接执行的离线归档校验

在此仓库根目录运行，仅读取文件，不导入控制程序：

```bash
python3 scripts/verify_r1a7_insertion_archive_20261117.py
R1A7_DIFF_BASE=8daf540 bash scripts/repository_sanity_check.sh
```

manifest记录现行文件与备份的SHA。research_code_snapshot/current保留原相对层级，Controller依赖同层transfer_control/tools/full_chain_runtime_recorder.py；其余外部依赖承接仓库与既有模型清单。快照不是独立可执行发行包，不应在来源档案目录直接启动硬件。

## 下一轮具体操作

1. 保留现场控制程序，先正常结束控制器并确认停止；不靠删除锁或pkill处理控制权。源代码变更和writer切换另按实际现场授权进行。
2. 检查五主程序＋recorder的SHA及进程cmdline；记录Fusion→preinsert→Shaper→shadow→Gate唯一writer拓扑。每个终端重新定义变量或用绝对路径，避免空变量错误。
3. 用无运动回放标注External/Left真实FREE及孔，分别保存raw/display/control、frame seq、measurement timestamp、prediction/hold来源，不凭十字稳定性判通过。
4. 先验证身份锁和独立新帧计数，FREE失效/过期时的实际Gate输出；默认preview不等于整机无运动，后台消费者/旧JSON也须核验。
5. 实际动作前重新确认网卡路由、唯一LowCmd、锁、相机身份及首帧、起始姿态与急停；分别验证X/Y/Z符号和耦合，保留短距离上限及接触保护。
6. 同一轮保存trace/full-chain/视觉记录、启动命令、源码SHA与目标孔；退出后立刻复制出/tmp，避免后续同名覆盖。分别报告图像对齐、接触、入孔、出线及失败，不把像素阈值当完成判据。
7. 正常退出先控制器安全Q并确认LowCmd停止，再停止下游/上游诊断进程；异常优先物理急停。仅在停机后回退指定备份，重新核验拓扑。

尚未实施的grasp-anchor＋skeleton geodesic与在线点跟踪器方案见原件第十章；本次不改算法、不安装模型。
