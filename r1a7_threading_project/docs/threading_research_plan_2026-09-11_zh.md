# R1-A7 镊子夹取细线与自主穿孔研究方案大纲

更新日期：2026-09-11

## 1 研究目标

本项目面向 Unitree R1-A7 双臂人形机器人，使用 Dex1-1 两指夹爪绑定镊子，使夹爪开合驱动镊子开合。目标是实现机器人自主识别细线，使用镊子夹取细线端部，再将细线端部引导并穿入孔内，任务形态类似人类穿针引线。

本项目不应被定义为普通抓取任务，而应定义为“工具介导的细长柔性物体精密操作与孔插入任务”。其核心难点包括细线视觉检测、线端定位、镊子工具标定、柔性线受力与形变、孔位识别、精密对准、低速插入、失败诊断和安全恢复。

## 2 文献检索结论

### 2.1 是否已有高度相关研究

已有非常接近的研究。最直接相关的是：

- Zhenjun Yu 等，Precise Robotic Needle-Threading with Tactile Perception and Reinforcement Learning，CoRL 2023。该工作提出 T-NT，把针穿线拆成 Tail-end Finding 和 Tail-end Insertion 两阶段，使用视觉触觉传感器定位线端，再用触觉引导强化学习完成线端插入针孔。论文页面显示其测试线径包含 0.2 mm、0.5 mm、1 mm、2 mm，孔隙尺寸包含 0.6 mm x 7.5 mm 等级。  
  资料：https://arxiv.org/abs/2311.02396  
  项目页：https://sites.google.com/view/tac-needlethreading

这篇论文和本项目最接近，但不能直接照搬，因为它依赖指尖视觉触觉传感器和 Unity 触觉仿真，而当前 R1-A7 使用 Dex1-1 夹爪加镊子，暂未安装触觉传感器。因此它适合作为任务分解、评价指标和后续触觉扩展的核心参考，而不是第一阶段直接复现路线。

### 2.2 相关研究方向

柔性物体与 DLO 操作综述指出，柔性物体相较刚体有无限维状态、复杂动力学、遮挡和感知建模困难；更合理的路线是把数据驱动方法和解析/规则方法结合，而不是完全端到端。

- A Survey on Robotic Manipulation of Deformable Objects: Recent Advances, Open Challenges and New Frontiers, arXiv 2023。  
  资料：https://arxiv.org/abs/2312.10419

- Challenges and Outlook in Robotic Manipulation of Deformable Objects, arXiv 2021。  
  资料：https://arxiv.org/abs/2105.01767

线缆和线束研究与本项目也相关，因为它们同属 DLO 或 stiff DLO 的抓取、引导、插入、布线问题。

- Sequential Manipulation of Deformable Linear Object Networks with Endpoint Pose Measurements using Adaptive Model Predictive Control, ICRA 2024。该工作强调在受限空间和有限可见性下，仅用端点位姿测量进行线束操作，并用自适应 MPC 完成安装任务。  
  资料：https://arxiv.org/abs/2402.10372

- Harnessing with Twisting: Single-Arm Deformable Linear Object Manipulation for Industrial Harnessing Task, IROS 2024。该工作用单臂、力/力矩传感器、MPC、路径规划和插入 primitives 做线束夹具插入。  
  资料：https://arxiv.org/abs/2410.10729

- Behavioral Cloning for Robotic Connector Assembly: An Empirical Study, arXiv 2026。该工作用远程操作采集最多 300 条成功示教，融合力/力矩和固定相机训练行为克隆模型，面向连接器插入达到较高成功率。  
  资料：https://arxiv.org/abs/2602.22100

视觉识别方面，T-NT 使用 Grounded-SAM 来分割针孔和线的触觉 imprint。当前项目没有触觉图像，但可以借鉴“先分割线端和孔，再交给控制/策略”的结构。可参考：

- Segment Anything, ICCV 2023。  
  资料：https://arxiv.org/abs/2304.02643

- Grounding DINO, ECCV 2024。  
  资料：https://arxiv.org/abs/2303.05499

- Grounded SAM, arXiv 2024。  
  资料：https://arxiv.org/abs/2401.14159

学习控制方面，已有工作表明 ACT 和 Diffusion Policy 都适合从示教学习多步动作，但本任务的最终插入阶段接触精度高，不能只依赖端到端模仿学习。

- Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware，ACT/ALOHA。  
  资料：https://arxiv.org/abs/2304.13705

- Diffusion Policy: Visuomotor Policy Learning via Action Diffusion。  
  资料：https://arxiv.org/abs/2303.04137

- LeRobot 提供 ACT、Diffusion 等策略实现，可作为后续快速复现和数据格式参考。  
  资料：https://github.com/huggingface/lerobot

## 3 本项目推荐技术路线

### 3.1 总体路线

推荐采用“工程基线 + 视觉定位 + 分阶段策略学习 + 插入规则控制”的混合路线：

```text
工具标定
  -> 细线和孔视觉识别
  -> VR 示教采集
  -> 分阶段自主策略
  -> 低速插入与失败恢复
  -> 后续触觉/力控增强
```

不要一开始做完全端到端“图像输入到完整穿孔动作”。当前更稳妥的系统应拆成 5 个阶段：

1. tweezer_binding：夹爪与镊子绑定，确定接触 baseline 和安全闭合范围。
2. wire_tail_detection：识别细线端部，输出线端像素、方向和可见性置信度。
3. tweezer_grasp_wire：镊子尖移动到线端附近并闭合，完成夹线。
4. align_wire_to_hole：夹住线后，将线端送到孔前对准。
5. insertion_servo：低速插入，失败则回退、微调、重试。

### 3.2 感知方案

第一阶段使用已有三相机：

- right_wrist：主视角，用于镊子尖、线端、夹取状态。
- top：工作台全局视角，用于孔位和线的大致走向。
- left_wrist：辅助视角，用于遮挡时确认线端和孔。

建议先用传统视觉建立可解释基线：

- 高对比背景。
- 线使用白/黑/彩色高对比材料。
- 孔板固定，孔边缘使用明显颜色或 AprilTag/标记辅助。
- 用 OpenCV 做细线 skeleton、端点检测、孔圆/椭圆检测。

后续再引入 Grounded-SAM/SAM/GroundingDINO 做分割，但不建议一开始把它放进实时闭环，因为细线很细，通用分割模型未必稳定。

### 3.3 控制方案

夹爪控制：

- 保持当前细线项目脚本中的 `--gripper-initial-mode current`。
- 保持 `--right-gripper-open-cap-current`，防止松开 trigger 后自动全开。
- 记录 `measured_gripper_q`、`sent_gripper_q`、`goal_gripper_q`、`right_trigger` 和 `gripper_contact_hold`。
- 对镊子任务定义新的相对闭合坐标：`baseline_q`、`close_delta_q`，不要直接复用扳手的 open/close 语义。

手臂控制：

- 粗移动阶段可继续复用 Quest/TeleVuer + R1A7_ArmIK。
- 自主阶段先使用关键点伺服：镊子尖像素点和线端像素点闭环对齐。
- 插入阶段采用低速笛卡尔推进，不建议直接用大幅关节目标。
- 每次失败保存相机帧、夹爪状态和动作日志。

### 3.4 学习方案

推荐分三步：

第一步：行为克隆或 ACT 复现 `grasp_wire_v1`，只做“夹线”。

- 输入：三路 RGB + 16D qpos + right gripper relative state。
- 输出：右臂动作 + 右夹爪相对闭合动作。
- 数据量：先采 30-50 条成功夹线示教。
- 目标：稳定夹住线端，不要求穿孔。

第二步：视觉伺服或规则控制复现 `align_wire_to_hole_v1`。

- 输入：线端关键点、孔中心关键点、镊子尖关键点。
- 输出：小步位姿修正。
- 目标：把线端送到孔前 2-5 mm。

第三步：插入阶段采用混合策略。

- 初始版本：规则控制 + 小范围搜索。
- 进阶版本：Diffusion Policy 或 RL 学习插入微动作。
- 后续增强：触觉传感器或力控，参考 T-NT。

## 4 现阶段最适合复现的论文

### 4.1 第一优先级：T-NT 针穿线

复现目标不是完整复现其硬件，而是复现其任务结构：

- Tail-end Finding -> 本项目改为视觉线端检测。
- Tail-end Insertion -> 本项目改为孔前对准 + 低速插入。
- Tactile observation -> 当前先用 right_wrist/top RGB 代替。
- RL insertion -> 当前先用规则控制，后续再考虑 RL。

可复现实验：

1. 固定线和孔板。
2. 人工把线端放在可见区域。
3. 视觉检测线端和孔。
4. 镊子夹住线端。
5. 移动到孔前。
6. 低速插入。

### 4.2 第二优先级：线束插入与 DLO 端点控制

可借鉴 ICRA 2024 端点测量 + MPC 思路，但先简化为：

- 只跟踪线端，不建整条线模型。
- 孔板固定，减少状态维度。
- 不做复杂线束网络，只做单根线端插入。

### 4.3 第三优先级：行为克隆连接器插入

可借鉴其数据采集思想：

- 远程操作采集成功示教。
- 记录视觉、机器人状态、动作、失败原因。
- 先做单阶段任务，再扩展到多阶段。

本项目可以把它转换为：

```text
teleop demos -> grasp_wire_v1 dataset -> BC/ACT baseline -> real robot evaluation
```

### 4.4 第四优先级：ACT 与 Diffusion Policy

ACT 适合你当前已有系统，因为 R1-A7 已经完成过三相机、16D、低层推理和 Temporal Aggregation 迁移。Diffusion Policy 适合作为第二条学习路线，尤其是插入动作存在多模态搜索时。

当前建议：

- 先复用 ACT 做夹线阶段 baseline。
- 再用 Diffusion Policy 做对比。
- 不要一开始用 Diffusion Policy 解决全流程穿孔。

## 5 推荐实验阶段

### 阶段 0：硬件和安全边界

目标：确认镊子绑定、夹爪相对闭合范围、线不会被夹断。

产物：

- baseline JSON。
- gripper close delta 表。
- 20 次开合循环记录。
- 夹线成功/失败短视频或 states.csv。

通过标准：

- 松开 trigger 不会自动全开。
- 夹爪能让镊子稳定闭合。
- 镊子不滑动、不明显变形。

### 阶段 1：视觉可见性和线端检测

目标：确认 right_wrist/top 能识别线端和孔。

实验：

- 静态拍摄线端和孔。
- 不同背景、光照、线径、孔径。
- OpenCV 端点检测。

通过标准：

- 线端检测误差小于 3-5 mm。
- 孔中心检测误差小于 2-3 mm。
- 镊子尖端可见或可由夹爪姿态估计。

### 阶段 2：VR 夹线示教

目标：采集 30-50 条只夹线成功示教。

记录字段：

- 三相机 RGB。
- 16D qpos。
- 右 trigger。
- measured/sent/goal gripper q。
- contact_hold。
- 成功标签和失败原因。

通过标准：

- 手动成功率大于 80%。
- 失败可归因：看不清、夹不住、线滑走、镊子偏、孔板动。

### 阶段 3：夹线策略复现

目标：训练并评估 `grasp_wire_v1`。

候选方法：

- ACT：复用现有 R1-A7 ACT 经验，最快形成 baseline。
- LeRobot ACT：若后续要统一数据格式，可迁移到 LeRobot。
- Diffusion Policy：作为第二模型对照。

通过标准：

- 10 次真机测试中至少 6 次稳定夹住线。
- 失败时必须保存图像、qpos、action 和 trigger。

### 阶段 4：线端对孔

目标：夹住线后把线端送到孔前。

推荐方法：

- 关键点检测 + 视觉伺服。
- 初始孔板固定，孔法向固定。
- 小步修正，不做大范围运动。

通过标准：

- 线端到孔中心误差小于孔半径的一半。
- 线端姿态大致沿孔法向。

### 阶段 5：低速穿孔

目标：完成线端插入孔内。

推荐方法：

- 低速直线推进。
- 小幅螺旋/网格搜索。
- 失败时回退 1-3 mm，重新对准。
- 后续加入触觉或力反馈。

通过标准：

- 线端进入孔内并穿过可见长度。
- 记录每次插入成功/卡住/滑线/弯线。

## 6 最终推荐研究路线

本项目当前最适合的路线不是直接复现 T-NT 的触觉 RL，而是：

```text
R1-A7 工程链稳定
  -> 镊子夹爪相对闭合标定
  -> 线端和孔的视觉识别
  -> VR 采集夹线示教
  -> ACT 复现夹线 baseline
  -> 视觉伺服完成线端对孔
  -> 规则控制完成低速插入
  -> 后续再引入 Diffusion Policy / RL / 触觉
```

这样做的理由：

1. 你已有 R1-A7、Quest/TeleVuer、LowCmd、三相机和 ACT 数据链，先复用可降低工程风险。
2. 细线穿孔的主要不确定性在视觉可见性和工具接触，不在模型结构。
3. T-NT 证明“线端寻找 + 插入”分阶段合理，但它依赖触觉传感器，当前不能直接搬到 Dex1-1 镊子系统。
4. 线束插入论文说明 DLO 任务适合端点控制、插入 primitive 和混合控制。
5. ACT/Diffusion Policy 适合做夹线和对孔策略，但完整穿孔阶段需要规则搜索或力/触觉闭环。

## 7 论文复现操作大纲

### 7.1 T-NT 结构复现

目标：复现任务分解和评价方式。

操作：

1. 阅读 T-NT 论文和项目页。
2. 建立本项目等价阶段：线端检测、线端夹取、孔前对准、插入。
3. 设计线径和孔径矩阵，例如从粗线/大孔开始，再减小到目标细线/小孔。
4. 记录每阶段成功率和误差。

### 7.2 ACT 夹线复现

目标：复用已有 R1-A7 ACT 经验训练 `grasp_wire_v1`。

操作：

1. 用 VR 采集 30-50 条夹线示教。
2. 转换为 HDF5，保持 16D qpos/action。
3. 夹爪 action 使用 trigger 意图，不使用 contact-held sent q 作为闭合标签。
4. 训练 ACT。
5. 真机只测试夹线，不进入穿孔。

### 7.3 Diffusion Policy 对照

目标：当 ACT 夹线 baseline 稳定后，做第二模型对照。

操作：

1. 将同一批数据转成 LeRobot 或 Diffusion Policy 可读格式。
2. 先离线可视化 action 和图像同步。
3. 训练 Diffusion Policy。
4. 与 ACT 在同一测试集上比较夹线成功率、失败类型和动作平滑性。

### 7.4 视觉伺服对孔复现

目标：实现不依赖大模型的可解释对孔。

操作：

1. 固定孔板。
2. 检测孔中心和线端。
3. 建立像素误差到腕部小步运动的映射。
4. 小步闭环移动到孔前。
5. 低速插入并记录失败。

## 8 当前最近一步

最近一步不是训练，而是完成以下工程闭环：

1. 启动细线专用 VR 控制。
2. 保证右夹爪松开 trigger 不会回到全开。
3. 左手 A 开始记录。
4. 右 trigger 慢速闭合镊子。
5. 保存 `states.csv`。
6. 检查 `measured_gripper_q`、`sent_gripper_q`、`goal_gripper_q` 和 `right_trigger` 是否正确记录。

只有这一步通过后，才进入视觉可见性和夹线示教采集。

