# 11月18日FREE身份与孔位重定位研究归档

## 连续性与新增内容

已fetch核对远端研发分支 `codex/r1a7-console-camera-suite-2026-10-11`，基线为 `7c316301769957b1adb3b4b3562b215bf03a8497`。11/18文档是11/17插孔调试的连续后续，新增下半支FREE身份保护、FREE-HOLD排查、Left孔位重定位、10→5→原限幅的试验/回退过程。旧夹取链与XY Shaper技术路线属于继承内容，不重复宣称为新增功能。

[原件与源码入口](source_records/2026-11-18_free_hole/README.md)；[全文转录](source_records/2026-11-18_free_hole/worklog_fulltext_zh.md)；[逐文件SHA及变化分类](source_records/2026-11-18_free_hole/manifest.json)；[技术路线与步骤](r1a7_free_hole_sop_2026-11-18_zh.md)。原件保持字节不变，409段全文按OOXML顺序展开表格，段内换行保留，仅去行末空白；2张嵌图原字节保存并已查看，整份Word未渲染/验版。

## 本机源码比文档记录更新

| 程序 | 对文档/11月17日的核对 | 归档处理 |
|---|---|---|
| Visual V3 | 当前SHA `c6f7b9e5f36e1f4e1106c58368e123386e87f393da710c0444c88d8f71afb50e`，已越过文档b9a12787基线 | 更新原研究视觉路径并保存完整快照 |
| Fusion | dc21dd55，与文档及11/17相同，写preinsert | coherent current快照，控制台原位不替换 |
| Shaper | dfe8f282，与文档及11/17相同 | 原速度已回退；V6/V7保留 |
| 原Gate | b31f3056，与文档及11/17相同 | 插孔1/其他10mm/s矢量限幅 |
| Controller | 0c22da11，与文档及11/17相同 | 保留夹爪链与actual/full-chain接口 |
| Full-chain recorder | 与11/17相同 | 按Controller依赖的相对路径保存 |

文档写联合V3补丁尚待成功落盘，但当前Visual不仅有 `FREE_CONTINUOUS_HOLE_LIVE_RELOCALIZATION_V3` 标记，实际执行段也存在：

- `INSERTION_FREE_LAST_TIP_HOLD_S=0.75`；当前tip有效时刷新最近可信tip；已初始化分支且短时丢失时复用历史tip，长期丢失继续BRANCH_TIP_LOST。
- legacy identity_margin拒绝仅在free_branch_dir尚未初始化时生效，避免已锁分支被最近邻歧义反复冻结。
- Hole inference条件扩展到WAIT_HOLE和INSERTION；INSERTION中跳过非left_wrist视角。Left复用既有局部孔匹配、raw搜索中心和EMA更新target_uv；External保持确认的固定目标。

因此将V3从“文档未收落盘回执”补充为“当前源码存在、编译检查通过”，不补写运行成功或FREE-HOLD已解决。文档b9a12787原基线确实存在于09:51安全备份，另存文档基线→当前和11/17→当前两份精确diff，不改写原DOCX。

## FREE与HOLE技术边界

下半支V1依据tip→FREE方向做同半支投影，初始化长度12px、最小投影6px、方向EMA .08、更新dot .25。它不是完整skeleton geodesic身份方案；“下半支”依赖可信初始化，不能仅以画面上下替代物理身份。

短时tip复用不是FREE运动速度预测。当前独立hole_anchor/live双轨诊断、孔位独立连续多帧认证、严格按孔测量新鲜度硬停Z等尚缺完整证据。已有ROI/跳变/EMA不能直接等同这些新增验收条件。两张截图可见FREE-HOLD与显示失配，单帧不能唯一确定跟踪器的失效分支或相机机械晃动。

## 速度回退与控制链解释

当前Shaper限幅为XY .60/.35/.20/.08，Z .10/.05mm/s；Gate按insertion_tracking_active选择1.0或10.0mm/s合速度上限。两者哈希与00:56备份/11/17快照一致，支持已回退的文件结论。10和5的历史试验备份按原字节保留，不能把上限称为定速。

Controller内 `IBVS_MIN_SPEED_MM_S=1.0` 不仅是常量：接收路径在候选速度范数大于0且小于1时按方向整体放大到1mm/s。因此Shaper的低请求速度可能被下游抬升；是否在文档具体试验里触发，仍需同时间Gate consumed与Controller数据。这里保留兼容性风险，不修改控制器或宣称低速已经验收。

## 文件 修改 新增 删除与失败备份

保存34份来源文件：新原件、完整6文件研究链、正常路径视觉副本、当日相关备份、两个安全备份目录、候选lower_branch tracker及原有测试源码。Safety backup的source.sha256及小型tar均保留并验证tar内6文件，未抽取覆盖研究区。experimental_candidates只作为研究候选，不设置默认入口。

原位控制台Fusion、Gate、普通controller均与7c31630一致，未部署研究输出链。未删除正式程序或清理研究区；条件扩展与逻辑变化由diff追溯，不能据端点diff还原每次历史编辑。

两份Gate失败备份分别为broken_gate_cap10_013920、failed_cap10_v4_014039，实际不导入编译均在475行出现IndentationError。按expected_historical_compile_failures精确记录，不修复其历史字节、不纳入可执行入口。其余源码按manifest核验语法。

## 数据缺口与实验结论

文档记载09:02 trace 5400行/183.04s、full-chain11621行及01:37 trace3049行等；本次在/tmp、研究日志、桌面资料、下载及home文件名搜索中未找到对应CSV和原“粘贴的文本”附件。故所有相关运动统计保留为来源记录，不标成本次已复算；未拿其他时段日志替代，也不伪造CSV。

原始图像/权重/视频未批量上传，前档模型清单继续有效。09:02停止/动作反馈属于文档历史证据，不等于本次真机复测，WAIT_CONTACT、open=1/close=1或像素对准都不证明实体夹持/穿孔完成。11/17缺失full-chain也未由本次补齐。

## 本次校验与下一步

执行文件哈希、全文/图片完整性、当前/历史源码语法分类、tar与checksum匹配、diff可重建、链接及仓库检查；没有导入模型、运行相机、Shaper、Gate、DDS/LowCmd或硬件。

下一步依次为同步保存真实端点标注与direct_state序列、验证V3短丢失与孔重定位、补齐孔测量有效期/身份认证、联合核查下游速度抬升，最后才是独立授权的受控物理试验。归档本身不实施这些算法修改。
