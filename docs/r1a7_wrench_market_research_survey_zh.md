# R1-A7 人形机器人扳手操作相关研究方案调研

## 调研结论

R1-A7 具体型号公开论文和成熟研究案例仍然较少。当前市面上更常见的做法，是把 R1-A7 视为 Unitree R1 系列固定式双臂/上半身操作平台，并参考 Unitree G1/H1、XR 遥操作、LeRobot/ACT/Diffusion Policy、视觉定位 + IK、接触丰富操作等路线。

对本项目最有价值的结论是：

```text
固定工装
-> Quest 遥操作采集
-> 只读 lowstate 记录
-> 关键帧和状态机 baseline
-> 对准/套入搜索策略
-> 示教统计或窄阶段 BC 修正
-> 后续加入 AprilTag/视觉
-> 最后再考虑双臂辅助
```

这条路线与市面主流趋势一致，但比直接上大模型、端到端学习或全自主双臂力控更符合当前 R1-A7 硬件条件和硕士课题周期。

## 1. 官方/厂商主推路线：开放二次开发 + 双臂桌面操作

Unitree 官方把 R1 系列定位为低成本、可二次开发的人形机器人平台。R1-A/R1-A-D 更偏向上半身双臂操作，适合科研教育、桌面操作、工业装配和 AI 开发。

公开资料中可确认的方向：

- R1 系列支持不同自由度配置。
- R1-A7 是 7 自由度双臂版本。
- 官方路线依赖 Unitree SDK、底层接口、双目视觉和可选末端执行器。
- 官方文档强调人形机器人结构复杂、功率大，开发时必须保持安全距离。

对本项目的启发：

- R1-A7 适合作为双臂桌面操作研究平台。
- 官方能力支持扳手工具操作研究，但不代表可以直接上全自主系统。
- 当前阶段应先做固定工装、低速动作、单臂优先、可记录可复现实验。

参考：

- Unitree R1 官方页：https://www.unitree.com/R1
- Unitree R1 开发文档：https://support.unitree.com/home/en/R1_developer
- 公开产品报道：https://finance.sina.com.cn/jjxw/2026-04-30/doc-inhwhvkc7285603.shtml

## 2. XR/Quest 遥操作 + 示教数据采集路线

这是目前 Unitree 人形机器人操作研究中最实际的路线之一。Unitree 官方开源的 `xr_teleoperate` 使用 XR 设备遥操作机器人，并记录数据用于 imitation learning。

典型研究方式：

1. 人通过 VR/XR/手柄遥操作机器人。
2. 同步记录关节、夹爪、图像、任务阶段和动作。
3. 从 demonstration 中训练 Behavior Cloning、ACT、Diffusion Policy 或其他策略。
4. 先做抓取、搬运、放置，再扩展到接触丰富操作。

对本项目的适配判断：

- 非常适合。
- 本项目已有 Quest 3 遥操作、`rt/lowstate` 记录和低层夹爪控制路径。
- 不应跳过遥操作采集；它应该成为扳手任务第一批真实数据来源。

参考：

- Unitree `xr_teleoperate`：https://github.com/unitreerobotics/xr_teleoperate

## 3. LeRobot / ACT / Diffusion Policy 模仿学习路线

Unitree 官方开源生态中已有 `unitree_IL_lerobot` 方向，支持 Diffusion Policy、ACT 等主流模仿学习算法，并适配部分 Unitree 硬件。虽然公开适配重点不一定是 R1-A7，但研究路线可以迁移。

典型流程：

1. 用遥操作采集 episode。
2. 标准化 observation/action。
3. 训练 ACT、Diffusion Policy 或 Behavior Cloning。
4. 部署到真机执行抓取、放置和简单操作。

对本项目的适配判断：

- 适合作为后续阶段，不适合作为第一阶段。
- 原因是当前相机链路还不稳定，自动真机控制也未完全开放。
- 推荐只学习窄阶段，例如 `PRE_NUT -> CONTACT_START` 的小范围对准修正。
- 不建议直接学习完整扳手任务。

参考：

- Unitree 开源页：https://www.unitree-robot.com/mobile/opensource/index.html
- Unitree G1 模仿学习传感器配置研究：https://arxiv.org/abs/2603.28422

## 4. 视觉定位 + IK + 经典控制路线

另一类主流路线是：

```text
目标检测 / 6D pose
-> 坐标变换
-> IK
-> 轨迹规划
-> 真机执行
```

Unitree G1 相关研究中，已有将 YOLO、SAM、FoundationPose、IK 和 Unitree SDK 串联起来进行抓取任务的方案。核心思想是通过视觉获得物体 6D pose，再用 IK 控制人形机器人抓取。

对本项目的适配判断：

- 有参考价值，但不是第一步。
- 扳手-螺母任务比普通抓取更依赖精确对准和接触状态。
- 当前相机还不稳定，应先用固定工装和人工标定替代视觉。
- 后续视觉恢复后，第一步应使用 AprilTag/ArUco，而不是直接上复杂 6D pose。

参考：

- Foundation-model-based Unitree G1 manipulation 相关研究：https://arxiv.org/abs/2604.17258

## 5. Foundation Model / VLM / LLM 分层规划路线

很多人形机器人研究会使用 VLM/LLM 做高层任务规划：

```text
语言指令
-> 分解任务步骤
-> 视觉识别对象
-> 调用低层抓取/移动/放置技能
```

这种路线适合长流程任务，例如整理桌面、多物体搬运、人机协作。

对本项目的适配判断：

- 不适合作为当前主线。
- 扳手任务的核心难点不是语言理解，而是抓稳扳手、对准螺母、套入和接触状态下旋转。
- VLM/LLM 只能在底层技能稳定后作为任务调度层。

## 6. Sim-to-Real / RL / Whole-body Loco-manipulation 路线

Unitree G1/H1 上有大量研究关注：

- 走路、跑步、摔倒恢复。
- whole-body manipulation。
- sim-to-real。
- residual learning。
- AMP/RL。
- 视觉 student policy。

对本项目的适配判断：

- 当前不适合。
- 这类路线需要大量仿真、训练资源、动力学建模和真机安全验证。
- R1-A7 当前更适合固定式双臂/上半身桌面工具操作，而不是 whole-body RL。

参考：

- Unitree G1 locomotion/recovery 相关研究：https://arxiv.org/abs/2605.18611
- Loco-manipulation 相关研究：https://arxiv.org/html/2510.05070v1
- Residual learning 相关研究：https://arxiv.org/html/2511.15200v1

## 7. 接触丰富操作 / 阻抗控制路线

扳手任务本质是 contact-rich manipulation。更先进的研究路线通常会使用：

- 力/力矩传感器。
- 触觉传感器。
- 阻抗控制。
- Admittance control。
- 接触状态估计。
- 局部搜索策略。

对本项目的适配判断：

- 当前不能默认 R1-A7 已具备可靠力控接口。
- 力反馈、接触阈值和稳定自动控制链路都需要后续验证。
- 因此应采用无力传感器替代方案。

推荐替代方案：

- 低速位置控制。
- 小步线性搜索。
- 小扰动搜索。
- 人工标注接触结果。
- 从关节误差、运动停滞或夹爪闭合阻滞间接判断接触。
- 失败后退出并重试。

这部分可以成为论文研究重点。

## 8. 方案对比

| 路线 | 市面做法 | 对 R1-A7 当前项目的适配性 | 结论 |
| --- | --- | --- | --- |
| 官方 SDK 二次开发 | 调底层接口，做双臂/视觉/末端控制 | 高 | 作为工程基础 |
| XR 遥操作 + 数据采集 | 人遥操作，记录 episode，用于示教学习 | 很高 | 优先执行 |
| ACT/DP 模仿学习 | 用示教数据训练策略 | 中 | 后续做窄阶段学习 |
| 视觉 + IK | 检测/6D pose 后轨迹执行 | 中 | 相机恢复后再做 |
| VLM/LLM 分层规划 | 高层任务分解，调用技能库 | 低 | 暂不作为主线 |
| RL/Sim-to-Real | 大规模仿真训练 whole-body 技能 | 低 | 当前不采用 |
| 接触/阻抗控制 | 力控、触觉、接触状态估计 | 中低 | 先用无力传感器替代方案 |

## 9. 对本项目最终方案的影响

市面方案调研后，本项目不应选择以下路线作为主线：

- 端到端视觉动作学习。
- 完整双臂自主工具使用。
- 大规模强化学习。
- Whole-body loco-manipulation。
- 复杂力控或阻抗控制。
- VLM/LLM 任务规划。

本项目应采用：

```text
固定工装 + 右臂优先 + Quest 遥操作采集
+ 只读 rt/lowstate 记录
+ 关键帧状态机 baseline
+ 对准/套入搜索策略
+ 示教统计或窄阶段 BC 修正
+ 后续 AprilTag/ArUco 视觉扩展
+ 双臂辅助作为扩展实验
```

这一路线的优点：

- 符合当前硬件。
- 能立即开始采集真实数据。
- 安全风险可控。
- 有明确 baseline。
- 有对比实验和消融实验。
- 即使学习或视觉模块失败，也能形成可写论文的实验结果。
