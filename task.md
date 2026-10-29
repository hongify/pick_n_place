# SO101 Pick and Place

Pick up a candy and place it in a bowl with an SO101 arm.
The setup uses a leader arm, a follower arm, and wrist and top cameras.

ROS 2 handles control and recording. Episodes are saved as MCAP bags,
converted to LeRobot datasets, and used to fine-tune SmolVLA.
The same command topic works with the real arm and Isaac Sim.

Start with a few successful recordings to check the data pipeline, then
collect more examples with different candy positions. Keep failed runs
for review and use successful runs for training.

See [guide.md](guide.md) for commands and [ros2_bridge.md](ros2_bridge.md)
for the hardware connection.

2026.06.27
