# R1-A7 Wrench Operation Handover

## Current Status

- Branch: `r1a7-wrench-baseline`
- Current phase: Phase A complete, mock skeleton of Phase C started, read-only real lowstate recording verified.
- Real robot motion: not enabled.
- Default execution mode: `backend=mock`, `dry_run=true`, `enable_robot_motion=false`.
- Real movement for the next operation must be done through the existing handset/Quest teleop path, not through the new wrench FSM.

## Completed

- Read the wrench task document and converted it into the fixed-workcell, no-camera, right-arm-first baseline.
- Referenced the existing Quest 3 dual-arm and internal Dex1 gripper control path.
- Created `docs/current_system_inventory.md`.
- Added `scripts/collect_environment_info.sh`.
- Added `r1a7_wrench_project/` with:
  - robot abstraction interface
  - mock robot backend
  - blocked real R1-A7 adapter placeholder
  - safety monitor
  - quintic joint trajectory generator
  - data episode recorder
  - keyframe capture placeholder
  - no-motion wrench FSM skeleton
  - unit tests

## Confirmed Control References

- Existing real control entry: `tools/r1a7_vr_dual_arm_g1ik_real.py`
- Existing state topic: `rt/lowstate`
- Existing command topic: `rt/lowcmd`
- Existing left arm indices: `15,16,17,18,19,20,21`
- Existing right arm indices: `22,23,24,25,26,27,28`
- Existing internal gripper indices: `31,33`

## Current Commands

Collect environment:

```bash
./scripts/collect_environment_info.sh
```

Run unit tests:

```bash
PYTHONPATH=r1a7_wrench_project python3 -m pytest -q r1a7_wrench_project/tests
```

Run no-motion FSM:

```bash
python3 r1a7_wrench_project/scripts/run_wrench_fsm.py --backend mock
```

Record a mock episode:

```bash
python3 r1a7_wrench_project/scripts/record_teleoperation.py --duration 2.0
```

Preview keyframe capture shape:

```bash
python3 r1a7_wrench_project/scripts/capture_keyframe.py HOME
```

## Verification Results

- `PYTHONPATH=r1a7_wrench_project python3 -m pytest -q r1a7_wrench_project/tests`: passed, 4 tests.
- `python3 r1a7_wrench_project/scripts/run_wrench_fsm.py --backend mock`: reached `COMPLETE`.
- `./scripts/collect_environment_info.sh`: wrote `logs/environment_20260802_154434.txt`.
- `record_teleoperation.py` smoke test wrote `data/episodes/episode_smoke_wrench_mock`.
- `PYTHONNOUSERSITE=1 /home/robot/miniconda3/bin/conda run --no-capture-output -n tv python -u r1a7_wrench_project/scripts/record_real_lowstate.py --interface enx9c69d37d0967 --domain-id 0 --state-topic rt/lowstate --duration 2 --rate-hz 20 --episode-id episode_real_lowstate_smoke_20260802_tv`: passed, wrote 40 samples to `data/episodes/episode_real_lowstate_smoke_20260802_tv`.

## Known Gaps

- Real R1-A7 adapter is intentionally blocked.
- Keyframe capture currently uses mock only.
- No true `rt/lowstate` read-only wrapper has been added to the wrench package yet.
- No real motion command is implemented in the wrench package.
- Fixture frames are unset in `r1a7_wrench_project/config/frames.yaml`.
- Torque/current thresholds for contact and tightening remain unconfirmed.

## Next Step

Research plan and Codex review loop:

- `docs/r1a7_wrench_research_iterative_plan_zh.md`

Use the existing handset/Quest teleop path for robot movement and run the new read-only recorder in parallel. The recorder subscribes to `rt/lowstate`, maps the confirmed arm and gripper indices, and writes episodes while the teleop program runs unchanged. Do not add real motion publishing to the wrench package until read-only logs, keyframes, and joint mapping are verified.

Read-only real-state recording command:

```bash
PYTHONNOUSERSITE=1 /home/robot/miniconda3/bin/conda run --no-capture-output -n tv \
python -u r1a7_wrench_project/scripts/record_real_lowstate.py \
  --interface enx9c69d37d0967 \
  --domain-id 0 \
  --state-topic rt/lowstate \
  --duration 30
```

## Next Operator Procedure

Use two terminals.

Terminal 1 starts the existing Quest/handset teleop script. This is the only process that should publish robot motion commands.

Terminal 2 starts the read-only recorder above. It only subscribes to `rt/lowstate` and writes episode data.

For the first wrench data episode, perform only:

1. Move both arms slowly in free space.
2. Open and close both grippers once.
3. Move the right arm to the wrench pre-grasp area without touching the wrench.
4. Stop the recorder.
5. Inspect the episode CSV before attempting grasp data.

After this passes, record separate episodes for `PRE_GRASP`, `GRASP`, `LIFT`, `TRANSFER_SAFE`, and `PRE_NUT`.
