# FREE与孔位重定位技术路线及复核步骤

本文件承接11/17插孔流程，记录当前源代码及下一轮验证口径。文档中的历史补丁、回退与使能命令不在本次执行。

## 当前技术路线

```text
Wire U-Net → Skeleton端点 → tip→FREE下半支投影约束
 → 连续性/EMA/reacquire → FREE测量或HOLD/失效
                          ↓
WAIT_HOLE：External＋Left选同一孔并确认
INSERTION：External固定目标；Left Hole U-Net局部重定位＋EMA
                          ↓
External du→X；Left du→Y；Left dv→Z
 → XY粗对齐 → 渐进放Z → FINAL_INSERT持续XYZ纠偏
 → Gate门控/矢量限幅 → Controller/DQIK/LowCmd
```

当前数据流为Visual→Fusion(preinsert)→Shaper(shadow)→原Gate(dual_ibvs_velocity)→Controller；Shaper默认preview不填补shadow输入，不代表其他进程不能读取已有控制JSON。每个输出路径必须核验唯一writer。保留原GRASP_PASS与夹爪状态机，不改夹爪扭矩。

## 本次归档的离线校验命令

在本仓库根目录运行，仅读文件、编译文本，不import相机/控制脚本：

```bash
python3 scripts/verify_r1a7_free_hole_archive_20261118.py
R1A7_DIFF_BASE=7c31630 bash scripts/repository_sanity_check.sh
```

文档b9a12787基线在safety_backups/pre_free_hole_live_20261118_095120中；current为c6f7b9e5版本。比较时使用manifest完整路径与哈希，不能仅凭同名Visual或V3标记推断已运行。失败Gate只存before_change，不作为回滚目标。

## 下一轮具体步骤与证据

1. 先记录五主程序及recorder SHA、实际启动命令与writer/reader拓扑；不要执行来源文档里的整体回退命令覆盖现行V3。每个终端重新定义变量或使用绝对路径。
2. 先在无运动且无后台控制消费者的条件下验证FREE身份：标注真实末端，保存raw/filtered/control、tip、branch_dir、direct_state、measurement_timestamp和frame_seq；计算测量/HOLD比例及跨分支误锁率。
3. 验证tip短丢失与长期丢失，确认0.75秒复用后的退出；它是历史tip复用，不是运动预测，也不能保证静止显示点仍是真实测量。
4. 验证Left孔raw/live与真实同一孔对应、External目标保持固定；记录局部搜索、跳变拒绝、EMA、测量年龄。补齐anchor/live诊断和失锁停Z前，不将当前简单重定位升级为孔身份可靠性验收。
5. 联合检查Shaper请求、Gate实际消费、Controller候选速度及actual XYZ。当前min-speed=1mm/s可能放大亚毫米请求，先定位再设计修改；禁止仅通过提高上限解决动作迟缓。
6. 保存整轮trace/full-chain/视觉序列到独立持久目录，退出即复制出/tmp；补齐01:37、09:02原CSV后才能复算文档统计。没有同时间链路时明确标缺失。
7. 感知、方向、限速与停止验证通过后，真机前重新核验网卡路由、唯一LowCmd、锁、相机首帧、起始姿态与急停。按低速短行程分步验收；正常先控制器安全退出并确认LowCmd停止，异常优先物理急停。

通过标准应分别报告端点身份、孔位匹配、失锁恢复、真实速度、接触与实体穿孔，不能用trace控制采样频率替代视觉FPS，也不能用夹爪状态代替夹持证据。
