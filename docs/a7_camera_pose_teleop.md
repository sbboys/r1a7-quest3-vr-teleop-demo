# R1-A7 Camera Pose Teleoperation

This project path controls only the simulated R1-A7 upper-body arm, using an
RGBD camera and MediaPipe body/hand landmarks. It does not require Unitree DDS
or real-robot services.

## Control Pipeline

```text
Gemini / Orbbec RGBD camera
  -> MediaPipe shoulder, elbow, wrist, hand landmarks
  -> human arm features
       horizontal: wrist left/right in the camera image
       vertical: wrist up/down in the camera image
       reach: wrist depth toward/away from the camera
       skeleton lift/side/reach/bend: shoulder-elbow-wrist arm shape
  -> R1-A7 7-DoF arm target
       shoulder_pitch
       shoulder_roll
       shoulder_yaw
       elbow
       wrist_roll
       wrist_pitch
       wrist_yaw
  -> rate limit, command lead limit, joint limits, torso safety
  -> Isaac Lab joint-position action
```

The active A7 arm joint order follows Unitree-style 7-DoF arm ordering:

```text
shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
wrist_roll, wrist_pitch, wrist_yaw
```

## Recommended Launch

```bash
./scripts/run_a7_camera_pose.sh
```

The script accepts extra `sim_main.py` arguments at the end. For example:

```bash
./scripts/run_a7_camera_pose.sh --camera_pose_direct_depth_sign 1.0
```

## Direction Tuning

Use these first when a direction is inverted:

```text
--camera_pose_mirror_input
--camera_pose_direct_vertical_sign 1.0 or -1.0
--camera_pose_direct_depth_sign 1.0 or -1.0
```

Use these when the motion is too small or too large:

```text
--camera_pose_direct_pitch_gain
--camera_pose_direct_roll_gain
--camera_pose_direct_yaw_gain
--camera_pose_direct_depth_gain
--camera_pose_direct_elbow_gain
```

Use these for A7 skeleton retargeting:

```text
--camera_pose_direct_skeleton_lift_gain
--camera_pose_direct_skeleton_side_roll_gain
--camera_pose_direct_skeleton_side_yaw_gain
--camera_pose_direct_skeleton_reach_pitch_gain
--camera_pose_direct_skeleton_reach_yaw_gain
--camera_pose_direct_skeleton_elbow_gain
--camera_pose_direct_skeleton_lift_elbow_gain
```

## Debug Log

When `--camera_pose_debug` is enabled, the log includes:

```text
direct_in=(horizontal, vertical, rawY, mY, mX, mZ, reach)
skel=(lift, side, reach, elbow_bend)
pitch/roll/yaw/elbow cur/cmd/tgt
lost_age
```

Expected signs with the default script:

```text
rawY changes when the wrist moves up/down in the image.
mZ changes when the wrist moves toward/away from the camera.
reach = direct_depth_sign * mZ.
```

If `target=no` appears frequently, first improve visibility:

```text
Keep shoulder, elbow, and wrist in view.
Avoid hiding the wrist behind the torso.
Move far enough in depth; several millimeters is not enough for visible reach.
Use --camera_pose_min_visibility 0.03 to 0.08 for debugging only.
```

For stable use, keep visibility stricter, for example:

```text
--camera_pose_min_visibility 0.15
```

## Lost Target Behavior

The controller now has two phases after camera target loss:

```text
--camera_pose_lost_hold_s      hold last arm command
--camera_pose_lost_return_s    smoothly return toward A7 home posture
```

This avoids sudden reset when MediaPipe temporarily loses the wrist.
