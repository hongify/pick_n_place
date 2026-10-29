#!/usr/bin/env python3



from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

from rclpy.node import Node
from sensor_msgs.msg import JointState


JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

GRIPPER_JOINT = "gripper"

LEADER_STATE_TOPIC = "/so101/leader/joint_states"
REAL_FOLLOWER_STATE_TOPIC = "/so101/real/follower/joint_states"
SIM_FOLLOWER_STATE_TOPIC = "/so101/sim/follower/joint_states"
FOLLOWER_STATE_TOPIC = "/so101/follower/joint_states"
FOLLOWER_COMMAND_TOPIC = "/so101/follower/commands"

WRIST_IMAGE_TOPIC = "/so101/camera/wrist/image_raw"
TOP_IMAGE_TOPIC = "/so101/camera/top/image_raw"
WRIST_CAMERA_INFO_TOPIC = "/so101/camera/wrist/camera_info"
TOP_CAMERA_INFO_TOPIC = "/so101/camera/top/camera_info"

TARGET_STATE_TOPICS = {
    "real": REAL_FOLLOWER_STATE_TOPIC,
    "sim": SIM_FOLLOWER_STATE_TOPIC,
    "alias": FOLLOWER_STATE_TOPIC,
}


GRIPPER_ROS_MIN_RAD = -0.1745330
GRIPPER_ROS_MAX_RAD = 1.7453292
GRIPPER_LEROBOT_MIN = 0.0
GRIPPER_LEROBOT_MAX = 100.0


@dataclass(frozen=True)
class JointTransform:
    source_name: str
    target_name: str
    scale: float = 1.0
    offset: float = 0.0
    lower: float | None = None
    upper: float | None = None


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def clamp_optional(value: float, lower: float | None, upper: float | None) -> float:
    if lower is not None and value < lower:
        value = lower
    if upper is not None and value > upper:
        value = upper
    return value


def lerobot_key(joint_name: str) -> str:
    return f"{joint_name}.pos"


def normalize_lerobot_dict(values: dict[str, float]) -> dict[str, float]:
    return {key.removesuffix(".pos"): float(value) for key, value in values.items()}


def gripper_lerobot_to_ros(value: float) -> float:
    value = clamp(value, GRIPPER_LEROBOT_MIN, GRIPPER_LEROBOT_MAX)
    ratio = (value - GRIPPER_LEROBOT_MIN) / (GRIPPER_LEROBOT_MAX - GRIPPER_LEROBOT_MIN)
    return GRIPPER_ROS_MIN_RAD + ratio * (GRIPPER_ROS_MAX_RAD - GRIPPER_ROS_MIN_RAD)


def gripper_ros_to_lerobot(value: float) -> float:
    value = clamp(value, GRIPPER_ROS_MIN_RAD, GRIPPER_ROS_MAX_RAD)
    ratio = (value - GRIPPER_ROS_MIN_RAD) / (GRIPPER_ROS_MAX_RAD - GRIPPER_ROS_MIN_RAD)
    return GRIPPER_LEROBOT_MIN + ratio * (GRIPPER_LEROBOT_MAX - GRIPPER_LEROBOT_MIN)


def lerobot_to_ros_position(joint_name: str, value: float, ros_unit: str) -> float:
    if ros_unit == "lerobot":
        return float(value)
    if joint_name == GRIPPER_JOINT:
        return gripper_lerobot_to_ros(value)
    return math.radians(value)


def ros_to_lerobot_position(joint_name: str, value: float, ros_unit: str) -> float:
    if ros_unit == "lerobot":
        return float(value)
    if joint_name == GRIPPER_JOINT:
        return gripper_ros_to_lerobot(value)
    return math.degrees(value)


def build_joint_state_msg(node: Node, values: dict[str, float], ros_unit: str) -> JointState:
    msg = JointState()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.name = list(JOINT_ORDER)
    msg.position = [
        lerobot_to_ros_position(joint_name, values[joint_name], ros_unit)
        for joint_name in JOINT_ORDER
    ]
    return msg


def command_msg_to_lerobot_action(msg: JointState, ros_unit: str) -> dict[str, float]:
    position_by_name = dict(zip(msg.name, msg.position, strict=False))
    missing = [joint_name for joint_name in JOINT_ORDER if joint_name not in position_by_name]
    if missing:
        raise ValueError(f"command is missing joints: {missing}")
    return {
        lerobot_key(joint_name): ros_to_lerobot_position(joint_name, position_by_name[joint_name], ros_unit)
        for joint_name in JOINT_ORDER
    }


def copy_joint_state_msg(msg: JointState) -> JointState:
    copied = JointState()
    copied.header = msg.header
    copied.name = list(msg.name)
    copied.position = list(msg.position)
    copied.velocity = list(msg.velocity)
    copied.effort = list(msg.effort)
    return copied


def require_joint_positions(msg: JointState, source_names: list[str]) -> dict[str, float]:
    position_by_name = dict(zip(msg.name, msg.position, strict=False))
    missing = [name for name in source_names if name not in position_by_name]
    if missing:
        raise ValueError(f"joint state is missing joints: {missing}")
    return {name: float(position_by_name[name]) for name in source_names}


def parse_csv(value: str, expected_len: int, label: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if len(items) != expected_len:
        raise argparse.ArgumentTypeError(f"{label} must contain {expected_len} comma-separated values")
    return items


def parse_float_csv(value: str, expected_len: int, label: str) -> list[float]:
    items = parse_csv(value, expected_len, label)
    try:
        return [float(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must contain only numbers") from exc


def parse_optional_float_csv(value: str, expected_len: int, label: str) -> list[float | None]:
    items = parse_csv(value, expected_len, label)
    parsed: list[float | None] = []
    for item in items:
        if item.lower() in {"none", "nan", "_"}:
            parsed.append(None)
        else:
            try:
                parsed.append(float(item))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"{label} must contain numbers or 'none'") from exc
    return parsed


def build_joint_transforms(
    target_joint_names: str,
    scale: str,
    offset: str,
    lower_limit: str,
    upper_limit: str,
    source_joint_names: str = ",".join(JOINT_ORDER),
) -> list[JointTransform]:
    source_names = parse_csv(source_joint_names, len(JOINT_ORDER), "--source-joint-names")
    target_names = parse_csv(target_joint_names, len(JOINT_ORDER), "--target-joint-names")
    scales = parse_float_csv(scale, len(JOINT_ORDER), "--scale")
    offsets = parse_float_csv(offset, len(JOINT_ORDER), "--offset")
    lower_limits = parse_optional_float_csv(lower_limit, len(JOINT_ORDER), "--lower-limit")
    upper_limits = parse_optional_float_csv(upper_limit, len(JOINT_ORDER), "--upper-limit")
    return [
        JointTransform(source_name, target_name, item_scale, item_offset, lower, upper)
        for source_name, target_name, item_scale, item_offset, lower, upper in zip(
            source_names,
            target_names,
            scales,
            offsets,
            lower_limits,
            upper_limits,
            strict=True,
        )
    ]


def transform_joint_state(
    node: Node,
    source: JointState,
    transforms: list[JointTransform],
    publish_empty_velocity_effort: bool = False,
    keep_source_header: bool = False,
) -> JointState:
    position_by_name = require_joint_positions(source, [item.source_name for item in transforms])
    msg = JointState()
    if keep_source_header:
        msg.header = source.header
    else:
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = source.header.frame_id
    msg.name = [item.target_name for item in transforms]
    msg.position = [
        clamp_optional(position_by_name[item.source_name] * item.scale + item.offset, item.lower, item.upper)
        for item in transforms
    ]
    if publish_empty_velocity_effort:
        msg.velocity = []
        msg.effort = []
    else:
        msg.velocity = [0.0] * len(msg.name)
        msg.effort = [0.0] * len(msg.name)
    return msg


def target_state_topic(target: str) -> str:
    if target not in TARGET_STATE_TOPICS:
        raise ValueError(f"unknown target: {target}")
    return TARGET_STATE_TOPICS[target]
