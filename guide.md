# Usage

Load your ROS 2 environment and make LeRobot available in Python.
Run the commands below from the repository root. Change device paths
and checkpoint paths to match your setup.

Connect the arms:

```bash
python3 scripts/ros2_bridge_so101.py --no-ui --robot-mode both --leader-port /dev/ttyUSB0 --follower-port /dev/ttyUSB1
python3 scripts/so101_teleop.py
```

Record an episode, then replay it:

```bash
python3 scripts/record_episode.py
python3 scripts/play_episode.py --episode-dir data/raw/pick_candy_v1/episode_000001 --wait-for-subscriber
```

Convert recordings to a LeRobot dataset:

```bash
python3 scripts/convert_rosbag_to_lerobot.py --input data/raw/pick_candy_v1 --target real --overwrite --verify
```

Run a trained policy:

```bash
python3 scripts/inference.py --policy-path outputs/train/pick_candy_smolvla_v1/checkpoints/last/pretrained_model --target real --device cuda --fps 10
```

For Isaac Sim, press Play in the scene and use `--target sim` for
conversion and inference. Teleop, replay, and inference all publish to
`/so101/follower/commands`; run one command source at a time.

2026.06.27
