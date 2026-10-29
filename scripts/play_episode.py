#!/usr/bin/env python3



from __future__ import annotations

import argparse
import time
from pathlib import Path

import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions
from sensor_msgs.msg import CameraInfo, Image, JointState

from so101_ros_contract import (
    FOLLOWER_COMMAND_TOPIC,
    JOINT_ORDER,
    JointTransform,
    TOP_CAMERA_INFO_TOPIC,
    TOP_IMAGE_TOPIC,
    WRIST_CAMERA_INFO_TOPIC,
    WRIST_IMAGE_TOPIC,
    transform_joint_state,
)

SUPPORTED_TOPIC_TYPES = {
    "sensor_msgs/msg/JointState": JointState,
    "sensor_msgs/msg/Image": Image,
    "sensor_msgs/msg/CameraInfo": CameraInfo,
}

CAMERA_TOPICS = {
    WRIST_IMAGE_TOPIC,
    TOP_IMAGE_TOPIC,
    WRIST_CAMERA_INFO_TOPIC,
    TOP_CAMERA_INFO_TOPIC,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a recorded SO-101 episode with follower commands and camera topics."
    )
    parser.add_argument(
        "--episode-dir",
        required=True,
        help="Episode directory that contains metadata.yaml and the recorded bag prefix.",
    )
    parser.add_argument(
        "--bag-uri",
        default="",
        help="Override the rosbag2 URI. Defaults to <episode-dir>/episode.",
    )
    parser.add_argument(
        "--storage",
        default="mcap",
        choices={"mcap", "sqlite3"},
        help="rosbag2 storage backend used by the episode.",
    )
    parser.add_argument(
        "--source-topic",
        default=FOLLOWER_COMMAND_TOPIC,
        help="Topic to read from the bag.",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Playback speed multiplier. 1.0 matches the original timing.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Replay the bag repeatedly until interrupted.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print matched messages without publishing them.",
    )
    parser.add_argument(
        "--wait-for-subscriber",
        action="store_true",
        help="Wait until at least one subscriber is connected before playback starts.",
    )
    return parser.parse_args()


def resolve_bag_uri(args: argparse.Namespace) -> Path:
    episode_dir = Path(args.episode_dir).expanduser().resolve()
    if not episode_dir.is_dir():
        raise FileNotFoundError(f"episode dir not found: {episode_dir}")
    if args.bag_uri:
        return Path(args.bag_uri).expanduser().resolve()
    return episode_dir / "episode"


class LeaderEpisodePlayer(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("leader_episode_player")
        self.args = args
        self.command_publisher = self.create_publisher(JointState, FOLLOWER_COMMAND_TOPIC, QoSProfile(depth=10))
        self.transforms = [JointTransform(joint_name, joint_name) for joint_name in JOINT_ORDER]
        self.camera_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.bag_uri = str(resolve_bag_uri(args))
        self.reader = None
        self.message_count = 0
        self.command_count = 0
        self.topic_publishers: dict[str, object] = {}
        self.type_by_topic: dict[str, str] = {}
        self.message_class_by_topic: dict[str, type] = {}
        mapping = ", ".join(f"{item.source_name}->{item.target_name}" for item in self.transforms)
        self.get_logger().info(f"command target: {FOLLOWER_COMMAND_TOPIC}")
        self.get_logger().info(f"joint mapping: {mapping}")

    def open_reader(self) -> None:
        reader = SequentialReader()
        reader.open(
            StorageOptions(uri=self.bag_uri, storage_id=self.args.storage),
            ConverterOptions(input_serialization_format="", output_serialization_format=""),
        )
        topic_metadata = reader.get_all_topics_and_types()
        self.type_by_topic = {item.name: item.type for item in topic_metadata}
        if self.args.source_topic not in self.type_by_topic:
            raise ValueError(f"source topic not found in bag: {self.args.source_topic}")
        self.message_class_by_topic = {
            topic: SUPPORTED_TOPIC_TYPES[topic_type]
            for topic, topic_type in self.type_by_topic.items()
            if topic_type in SUPPORTED_TOPIC_TYPES
        }
        self.topic_publishers = {}
        for topic, topic_type in self.type_by_topic.items():
            if topic == self.args.source_topic and topic_type == "sensor_msgs/msg/JointState":
                continue
            if topic in CAMERA_TOPICS and topic_type in {
                "sensor_msgs/msg/Image",
                "sensor_msgs/msg/CameraInfo",
            }:
                message_class = SUPPORTED_TOPIC_TYPES[topic_type]
                self.topic_publishers[topic] = self.create_publisher(message_class, topic, self.camera_qos)
        reader.set_filter(StorageFilter(topics=list(self.type_by_topic)))
        self.reader = reader

    def wait_for_subscriber(self) -> None:
        if not self.args.wait_for_subscriber:
            return

        self.get_logger().info(f"waiting for subscriber on {FOLLOWER_COMMAND_TOPIC}")
        while rclpy.ok() and self.command_publisher.get_subscription_count() == 0:
            time.sleep(0.1)

    def iter_messages(self):
        if self.reader is None:
            raise RuntimeError("reader is not open")
        while self.reader.has_next():
            topic, data, timestamp = self.reader.read_next()
            topic_type = self.type_by_topic.get(topic, "")
            message_class = self.message_class_by_topic.get(topic)
            message = None
            if message_class is not None:
                message = deserialize_message(data, message_class)
            yield topic, topic_type, timestamp, message

    def play_once(self) -> None:
        self.open_reader()
        self.wait_for_subscriber()
        previous_timestamp_ns = None
        for topic, topic_type, timestamp_ns, message in self.iter_messages():
            if previous_timestamp_ns is not None:
                delta_s = max(0.0, (timestamp_ns - previous_timestamp_ns) / 1e9)
                time.sleep(delta_s / self.args.time_scale)
            previous_timestamp_ns = timestamp_ns

            if topic == self.args.source_topic and topic_type == "sensor_msgs/msg/JointState":
                if message is None:
                    continue
                target_msg = transform_joint_state(
                    self,
                    message,
                    self.transforms,
                )
                if self.args.dry_run:
                    self.get_logger().info(
                        f"dry-run command publish {self.command_count + 1}: "
                        f"{target_msg.position}"
                    )
                else:
                    self.command_publisher.publish(target_msg)
                self.command_count += 1
            elif topic in self.topic_publishers and message is not None and not self.args.dry_run:
                self.topic_publishers[topic].publish(message)
            self.message_count += 1

        self.get_logger().info(
            f"completed playback with {self.message_count} bag messages and {self.command_count} command messages"
        )

def main() -> int:
    args = parse_args()
    if args.time_scale <= 0.0:
        raise ValueError("--time-scale must be a positive finite number")

    rclpy.init()
    try:
        player = LeaderEpisodePlayer(args)
        try:
            while True:
                player.play_once()
                if not args.loop:
                    break
        finally:
            player.destroy_node()
    finally:
        try:
            rclpy.shutdown()
        except RCLError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
