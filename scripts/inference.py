#!/usr/bin/env python3



from __future__ import annotations

import argparse
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import rclpy
import torch
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image as ImageMsg
from sensor_msgs.msg import JointState

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference

from so101_ros_contract import (
    FOLLOWER_COMMAND_TOPIC,
    JOINT_ORDER,
    TOP_IMAGE_TOPIC,
    WRIST_IMAGE_TOPIC,
    copy_joint_state_msg,
    lerobot_to_ros_position,
    ros_to_lerobot_position,
    target_state_topic,
)


def joint_state_to_dataset_vector(msg: JointState, ros_unit: str) -> np.ndarray:
    position_by_name = dict(zip(msg.name, msg.position, strict=False))
    missing = [name for name in JOINT_ORDER if name not in position_by_name]
    if missing:
        raise ValueError(f"joint state is missing joints: {missing}")
    return np.asarray(
        [ros_to_lerobot_position(name, position_by_name[name], ros_unit) for name in JOINT_ORDER],
        dtype=np.float32,
    )


def action_vector_to_joint_state(node: Node, action: np.ndarray, ros_unit: str) -> JointState:
    values = np.asarray(action, dtype=np.float32).reshape(-1)
    if values.shape[0] < len(JOINT_ORDER):
        raise ValueError(f"policy action has {values.shape[0]} dims, expected at least {len(JOINT_ORDER)}")

    msg = JointState()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.name = list(JOINT_ORDER)
    msg.position = [
        lerobot_to_ros_position(joint_name, float(values[index]), ros_unit)
        for index, joint_name in enumerate(JOINT_ORDER)
    ]
    return msg


def image_msg_to_rgb_array(msg: ImageMsg) -> np.ndarray:
    encoding = msg.encoding.lower()
    height = int(msg.height)
    width = int(msg.width)
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if encoding in {"rgb8", "bgr8", "rgba8", "bgra8"}:
        channels = 4 if "a" in encoding else 3
        array = data.reshape(height, width, channels)
        if encoding == "bgr8":
            array = array[:, :, ::-1]
        elif encoding == "bgra8":
            array = array[:, :, [2, 1, 0]]
        elif encoding == "rgba8":
            array = array[:, :, :3]
        return np.ascontiguousarray(array)

    if encoding == "mono8":
        array = data.reshape(height, width)
        return np.repeat(array[:, :, None], 3, axis=2)

    raise ValueError(f"unsupported image encoding: {msg.encoding}")


def copy_image_msg(msg: ImageMsg) -> ImageMsg:
    copied = ImageMsg()
    copied.header = msg.header
    copied.height = msg.height
    copied.width = msg.width
    copied.encoding = msg.encoding
    copied.is_bigendian = msg.is_bigendian
    copied.step = msg.step
    copied.data = bytes(msg.data)
    return copied


class Ros2PolicyInference(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("so101_lerobot_policy_inference")
        self.args = args
        self.lock = threading.Lock()
        self.latest_top: ImageMsg | None = None
        self.latest_wrist: ImageMsg | None = None
        self.latest_state: JointState | None = None
        self.latest_top_time = 0.0
        self.latest_wrist_time = 0.0
        self.latest_state_time = 0.0
        self.inference_running = False
        self.last_warn_time = 0.0
        self.command_count = 0

        camera_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(ImageMsg, args.top_image_topic, self._top_callback, camera_qos)
        self.create_subscription(ImageMsg, args.wrist_image_topic, self._wrist_callback, camera_qos)
        self.create_subscription(JointState, args.state_topic, self._state_callback, QoSProfile(depth=10))
        self.command_pub = self.create_publisher(JointState, args.command_topic, QoSProfile(depth=10))

        self.policy, self.preprocessor, self.postprocessor, self.device, self.use_amp = self._load_policy()
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

        self.timer = self.create_timer(1.0 / args.fps, self._timer_callback)
        mode = "dry-run" if args.dry_run else "publishing"
        self.get_logger().info(
            f"Loaded policy from {args.policy_path}; {mode} {args.fps:g} Hz commands to {args.command_topic}"
        )

    def _load_policy(self):
        policy_path = Path(self.args.policy_path).expanduser()
        policy_id = str(policy_path.resolve()) if policy_path.exists() else str(policy_path)
        cli_overrides = ["--device", self.args.device] if self.args.device else []
        cfg = PreTrainedConfig.from_pretrained(policy_id, cli_overrides=cli_overrides)
        policy_cls = get_policy_class(cfg.type)
        policy = policy_cls.from_pretrained(policy_id, config=cfg)
        preprocessor, postprocessor = make_pre_post_processors(cfg, pretrained_path=policy_id)
        device = torch.device(cfg.device)
        return policy, preprocessor, postprocessor, device, bool(cfg.use_amp)

    def _top_callback(self, msg: ImageMsg) -> None:
        with self.lock:
            self.latest_top = copy_image_msg(msg)
            self.latest_top_time = time.monotonic()

    def _wrist_callback(self, msg: ImageMsg) -> None:
        with self.lock:
            self.latest_wrist = copy_image_msg(msg)
            self.latest_wrist_time = time.monotonic()

    def _state_callback(self, msg: JointState) -> None:
        with self.lock:
            self.latest_state = copy_joint_state_msg(msg)
            self.latest_state_time = time.monotonic()

    def _snapshot(self) -> tuple[ImageMsg, ImageMsg, JointState] | None:
        now = time.monotonic()
        with self.lock:
            top = self.latest_top
            wrist = self.latest_wrist
            state = self.latest_state
            ages = {
                "top": now - self.latest_top_time,
                "wrist": now - self.latest_wrist_time,
                "state": now - self.latest_state_time,
            }

        missing = [
            name
            for name, value in (("top", top), ("wrist", wrist), ("state", state))
            if value is None
        ]
        stale = [name for name, age in ages.items() if age > self.args.max_staleness_s]
        if missing or stale:
            if now - self.last_warn_time > 2.0:
                detail = []
                if missing:
                    detail.append(f"missing={missing}")
                if stale:
                    detail.append(f"stale={stale}")
                self.get_logger().warning("waiting for fresh observations: " + " ".join(detail))
                self.last_warn_time = now
            return None

        return top, wrist, state

    def _timer_callback(self) -> None:
        if self.inference_running:
            return
        snapshot = self._snapshot()
        if snapshot is None:
            return

        self.inference_running = True
        try:
            top_msg, wrist_msg, state_msg = snapshot
            observation = {
                self.args.top_feature: image_msg_to_rgb_array(top_msg),
                self.args.wrist_feature: image_msg_to_rgb_array(wrist_msg),
                "observation.state": joint_state_to_dataset_vector(state_msg, self.args.ros_unit),
            }
            action = self._predict_action(observation)
            command_msg = action_vector_to_joint_state(self, action, self.args.ros_unit)
            if self.args.dry_run:
                if self.command_count == 0 or self.command_count % max(1, int(self.args.fps)) == 0:
                    self.get_logger().info(f"dry-run action[{self.command_count}]: {command_msg.position}")
            else:
                self.command_pub.publish(command_msg)
            self.command_count += 1
        except Exception as exc:
            self.get_logger().error(f"inference step failed: {exc}")
        finally:
            self.inference_running = False

    def _predict_action(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        with (
            torch.inference_mode(),
            torch.autocast(device_type=self.device.type)
            if self.device.type == "cuda" and self.use_amp
            else nullcontext(),
        ):
            batch = prepare_observation_for_inference(
                observation=observation,
                device=self.device,
                task=self.args.task,
                robot_type=self.args.robot_type,
            )
            batch = self.preprocessor(batch)
            action = self.policy.select_action(batch)
            action = self.postprocessor(action)
        return action.squeeze(0).detach().to("cpu").numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subscribe to SO-101 ROS2 observations, run a LeRobot policy, and publish follower commands."
    )
    parser.add_argument("--policy-path", required=True, help="Local checkpoint/pretrained policy path or HF repo id.")
    parser.add_argument("--device", default="cuda", help="Torch device override, for example cuda or cpu.")
    parser.add_argument("--fps", type=float, default=10.0, help="Inference/publish rate.")
    parser.add_argument("--task", default="Pick up the candy and place it in the bowl.", help="Task text.")
    parser.add_argument("--robot-type", default="so101", help="Robot type passed to the LeRobot processor.")
    parser.add_argument("--ros-unit", choices={"rad", "lerobot"}, default="rad", help="Unit used on ROS topics.")
    parser.add_argument(
        "--target",
        choices={"real", "sim", "alias"},
        default="real",
        help="Observation target preset. real=/so101/real/follower/joint_states, sim=/so101/sim/follower/joint_states.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run inference but do not publish commands.")
    parser.add_argument(
        "--max-staleness-s",
        type=float,
        default=0.5,
        help="Maximum age for image/state messages before skipping inference.",
    )
    parser.add_argument("--top-image-topic", default=TOP_IMAGE_TOPIC)
    parser.add_argument("--wrist-image-topic", default=WRIST_IMAGE_TOPIC)
    parser.add_argument(
        "--state-topic",
        default="",
        help="Override observation state topic. Defaults from --target.",
    )
    parser.add_argument("--command-topic", default=FOLLOWER_COMMAND_TOPIC)
    parser.add_argument("--top-feature", default="observation.images.camera1")
    parser.add_argument("--wrist-feature", default="observation.images.camera2")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fps <= 0.0:
        raise ValueError("--fps must be positive")
    if not args.state_topic:
        args.state_topic = target_state_topic(args.target)

    rclpy.init()
    node: Ros2PolicyInference | None = None
    try:
        node = Ros2PolicyInference(args)
        rclpy.spin(node)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
