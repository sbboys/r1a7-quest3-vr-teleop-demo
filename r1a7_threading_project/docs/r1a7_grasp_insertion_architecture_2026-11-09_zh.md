# 夹取到选孔与插孔的技术路线

截至2026-11-09的研究架构；[核对报告](r1a7_grasp_insertion_archive_2026-11-09_zh.md)记录证据等级与代码位置。历史穿孔基线和本轮新链路的验收结论分别管理。

## 视觉与动作的数据流

```text
External/Left RGB -> wire/tool U-Net V2
Right Wrist RGB  -> wire/tool U-Net V3 -> 双尖端中点TIP
                -> wire skeleton/path/temporal/FREE身份/GRASP
                -> External/Wrist guidance JSON
                -> Fusion -> Gate -> velocity JSON
                -> Cartesian/IK -> 唯一rt/lowcmd控制器 -> 手臂/夹爪

Controller close state -> Visual阶段切换
GRASP -> CLOSE_SETTLE -> WAIT_CONTACT/DONE
      -> WAIT_HOLE_TARGET -> 三视角选同一实体孔并确认
      -> INSERTION_TRACKING: FREE -> HOLE
```

## 抓取阶段

TIP→GRASP为抓取误差。External提供X/Z，Wrist提供Y/Z；远处同向Z按0.35/0.65融合，冲突沿用研究分支规则。External误差<=45px进入near，>=65px退出；near且非插孔阶段时Z完全来自External、Y继续Wrist。相机到机器人方向约定必须在实际姿态下复核。

张开镊子取两臂前沿中点；12个有效gap样本中位数冻结通道宽度，中心80%作为可用范围。目标进入横向通道后latch；controller使用External close条件与jaw_corridor_latched进入CLOSE_SETTLE，Gate停臂，稳定时间到后CLOSE_ONCE。通道是二维横向几何，不证明纵深、实际接触和可靠夹住。

## 插孔阶段

抓取完成信号触发WAIT_HOLE_TARGET；停止旧抓取引导并释放原wire锁。人工在External、Right Wrist、Left Wrist选同一物理孔，三路都有目标才允许确认。孔位三分类模型输出board/hole候选，孔身份与时序跟踪由后处理及人工选择管理。

确认后FREE→HOLE产生引导；MEASURED和TIP_RELATIVE_PRED可被当前源码作为控制来源，DISPLAY_ONLY只显示。TIP-relative预测仍是推断，需独立核验遮挡和几何漂移；不能称为当帧直接测量。当前Gate有插孔确认后禁止重新进入grasp hold的机制，实际完整时序未在本次运行。

## 接口与阶段职责

| JSON | 生产者 | 作用 |
|---|---|---|
| `/tmp/r1a7_external_wire_grasp_v1.json` | Visual | External误差/有效性/建议X,Z |
| `/tmp/r1a7_wrist_wire_grasp_guidance.json` | Visual | Wrist误差/建议Y,Z/jaw诊断与latch |
| `/tmp/r1a7_wire_grasp_fusion_shadow_v1.json` | Fusion | 融合模式/来源/建议速度/near状态 |
| `/tmp/r1a7_dual_ibvs_velocity.json` | Gate | enabled、gate_state、输出速度 |
| `/tmp/r1a7_auto_grasp_close_settle_v1.json` | Controller | close状态、保持理由和时间 |
| `/tmp/r1a7_insertion_target_state_v1.json` | Visual | 选孔、确认和阶段 |
| `/tmp/r1a7_insertion_perception_v1.json` | Visual | 插孔感知；schema以源码为准 |

Gate研究版删除350px硬门槛，但仍有来源新鲜度、有效性与有限速度检查。BLIND_GRACE为最长0.40s的旧趋势低速过渡（0.40缩放、2mm/s上限），不是有新观测的闭环。旧文件、latch及预测的生命周期仍需验证；以上结构不构成安全认证或穿孔成功证据。
