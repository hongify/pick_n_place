# ROS 2 Bridge

The bridge connects SO101 arms and USB cameras to ROS 2.
Run `python3 scripts/ros2_bridge_so101.py` to select devices and calibration
files in the terminal. Calibration files are stored in `calibrate/`.
Teleop runs separately with `python3 scripts/so101_teleop.py`.

| Data | Topic |
| --- | --- |
| Leader state | `/so101/leader/joint_states` |
| Real follower state | `/so101/real/follower/joint_states` |
| Sim follower state | `/so101/sim/follower/joint_states` |
| Follower commands | `/so101/follower/commands` |
| Wrist image | `/so101/camera/wrist/image_raw` |
| Top image | `/so101/camera/top/image_raw` |

States and commands use `sensor_msgs/msg/JointState` in this order:
`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`.
Body joints use radians by default. The gripper maps LeRobot's `[0, 100]`
range to `[-0.1745330, 1.7453292]`. Use `--ros-unit lerobot` for raw units.

Scan cameras with `--scan-cameras`. Add `--visualize-cameras` to view frames.
In the runtime menu, `r` returns the follower to its saved rest position,
`c` toggles camera publishing, and `v` toggles the camera windows.
Entering `exit` returns the follower to rest and closes the bridge.

In Isaac Sim, set the joint command subscriber to `/so101/follower/commands`
and the state publisher to `/so101/sim/follower/joint_states`.
Match the joint names and order above.

2026.06.27
