# R1-A7 右臂 7 关节示教记录

这个流程用于先人工调整真机右臂 7 个关节，记录你认为协调的动作轨迹。记录结果后续可用于相机遥操作的关节映射标定。

## 1. 人工拖动示教记录

这个模式不向机器人发送控制命令，只记录 `rt/lowstate` 中的真实右臂关节角。适合你选择关节和动作方式后，用手把机械臂调整到目标位置，并保存全过程数据。

记录脚本默认持续保存低状态采样。`start` 和 `stop` 的作用是把这段数据标记为正式动作段，样本里的 `recording` 字段会变为 `true`。即使误操作导致 `start` 没有生效，`samples.csv` 里仍会保留这期间的关节轨迹。

启动：

```bash
cd /home/robot/unitree_sim_isaaclab
./scripts/run_r1a7_right_arm_teach_record.sh
```

默认是命令行输入模式，输入命令后按回车：

- `joint 1` 到 `joint 7`：选择当前示教关节
- `motion 动作名`：设置当前运动方式
- `segment 关节编号 动作名`：同时设置关节、动作方式和数据段标签
- `start`：开始连续记录整段运动过程
- `stop`：停止连续记录
- `s`：可选，保存当前关键点；不输入 `s` 也会保存 `start` 到 `stop` 之间的完整轨迹
- `n 标签名`：输入新的动作标签
- `p`：打印当前 7 关节角
- `q`：退出并保存

示例：

```text
segment 4 elbow_bend_extend
start
手动调整机器人肘部弯曲再伸展，最后回到原始位置
s
stop
segment 1 shoulder_pitch_forward
start
手动调整大臂前伸再后缩，最后回到原始位置
s
stop
q
```

## 2. 键盘点动示教控制

如果不能手动拖动机械臂，才使用这个模式。它会发布 `rt/lowcmd` 控制右臂。先清空机械臂周围空间，准备急停。

```bash
cd /home/robot/unitree_sim_isaaclab
./scripts/run_r1a7_right_arm_teach_jog.sh
```

出现提示后输入：

```text
ENABLE
```

默认是命令行输入模式，输入命令后按回车：

- `1`：大臂前后，`right_shoulder_pitch`
- `2`：大臂偏转，`right_shoulder_roll`
- `3`：大臂扭转，`right_shoulder_yaw`
- `4`：肘部弯曲，`right_elbow`
- `5`：手腕转动，`right_wrist_roll`
- `6`：手腕上下，`right_wrist_pitch`
- `7`：手腕左右，`right_wrist_yaw`
- `a` / `d`：当前关节小幅减小或增大
- `A` / `D`：当前关节大幅减小或增大
- `step 0.02`：把小幅调整量设为 `0.02 rad`
- `large 0.08`：把大幅调整量设为 `0.08 rad`
- `status`：查看当前标签、记录状态、步长和选中的关节
- `h`：把命令目标重置为当前真机姿态
- `r`：开始或暂停连续记录
- `s`：保存当前关键点
- `n 标签名`：输入新的动作标签
- `p`：打印当前姿态
- `q`：退出并释放控制

如果想使用原来的单键实时模式，不需要回车，可以加：

```bash
./scripts/run_r1a7_right_arm_teach_jog.sh --input_mode key
```

## 3. 建议记录的标签

每做一个动作前按 `n` 输入标签，然后按 `r` 开始记录，动作结束后再按 `r` 停止。

建议标签：

- `lift_up`：整臂抬起
- `press_down`：整臂下压
- `shoulder_roll_out_in`：大臂向外和向躯干靠拢
- `shoulder_yaw_twist`：大臂扭转
- `elbow_bend_extend`：肘部弯曲和伸展
- `elbow_down`：肘部下压
- `wrist_pitch_up_down`：手腕上下
- `wrist_yaw_left_right`：手腕左右
- `wrist_roll_rotate`：手腕旋转
- `reach_forward_keep_bent`：保持肘部弯曲的前伸

推荐采集方式：

1. 优先使用 `run_r1a7_right_arm_teach_record.sh` 做人工拖动记录。
2. 每个动作前输入 `segment 关节编号 动作名`。
3. 输入 `start` 开始记录。
4. 手动调整机器人手臂完成完整往返运动，最后回到原始位置。
5. 过程中可以输入 `s` 保存关键姿态；不输入也会保存整段连续轨迹。
6. 动作结束后输入 `stop` 停止记录。

## 4. 数据位置

每次运行会生成一个独立目录：

```text
data/r1a7_teach/YYYYMMDD_HHMMSS/
```

里面包含：

- `metadata.json`：关节名称、索引、限位和运行配置
- `samples.jsonl`：连续轨迹样本
- `samples.csv`：同样的轨迹，便于表格查看
- `waypoints.jsonl`：按空格保存的关键点
- `events.jsonl`：标签切换、关节选择、点动事件

## 5. 用记录结果改相机控制

记录完成后先生成每个动作标签的关节统计：

```bash
cd /home/robot/unitree_sim_isaaclab
./scripts/summarize_r1a7_teach_data.sh
```

它会在最新记录目录里生成：

```text
label_summary.csv
```

再生成相机控制用的示教配置：

```bash
./scripts/build_r1a7_teach_profile.sh
```

它会生成：

```text
data/r1a7_teach/latest_profile.json
```

真实相机控制脚本会自动读取这个文件，并用 `TEACH_PROFILE_BLEND` 控制融合比例：

```bash
TEACH_PROFILE_BLEND=0.65 ./scripts/run_r1a7_camera_real_control.sh --show
```

重点看 `reach_forward_keep_bent` 这段数据。如果你希望前伸时肘部仍保持弯曲，把这段里的：

- `right_shoulder_pitch`
- `right_shoulder_roll`
- `right_shoulder_yaw`
- `right_elbow`

作为目标关系，再回写到相机控制的前伸映射参数中。
