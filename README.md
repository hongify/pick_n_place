# SO101 Pick and Place

Pick-and-place experiments with an SO101 arm, ROS 2, and SmolVLA.
The task is to pick up a candy and place it in a bowl.

A leader arm provides teleoperation input. Wrist and top cameras capture
observations, and ROS 2 records episodes as MCAP bags. The recordings can
be converted to LeRobot datasets for training. Teleop, replay, and policy
inference share the same command topic for the real arm and Isaac Sim.

## Getting started

Set up ROS 2 and LeRobot, then run the bridge to select your arms,
calibration files, and cameras:

```bash
python3 scripts/ros2_bridge_so101.py
```

Start teleoperation in another terminal:

```bash
python3 scripts/so101_teleop.py
```

Record an episode:

```bash
python3 scripts/record_episode.py
```

Run one command source at a time: teleop, replay, or inference.
Adjust device and model paths for your setup.

See [guide.md](guide.md) for replay, dataset conversion, and inference.
[ros2_bridge.md](ros2_bridge.md) lists the ROS topics and hardware settings.
[task.md](task.md) outlines the data collection plan.

2026.06.27
