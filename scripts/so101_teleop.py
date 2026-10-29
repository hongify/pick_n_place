#!/usr/bin/env python3



from __future__ import annotations

import argparse
import math
import sys
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from so101_ros_contract import (
    FOLLOWER_COMMAND_TOPIC,
    GRIPPER_JOINT,
    JOINT_ORDER,
    LEADER_STATE_TOPIC,
    build_joint_transforms,
    transform_joint_state,
)


COMMAND_OFFSET_TOPIC = "/so101/follower/command_offsets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SO-101 leader joint states into the standard follower command topic."
    )
    parser.add_argument("--source-topic", default=LEADER_STATE_TOPIC, help="Leader JointState topic.")
    parser.add_argument("--command-topic", default=FOLLOWER_COMMAND_TOPIC, help="Standard SO-101 action topic.")
    parser.add_argument(
        "--target-joint-names",
        default=",".join(JOINT_ORDER),
        help="Comma-separated output joint names. Defaults to the standard SO-101 joint order.",
    )
    parser.add_argument("--scale", default="1,1,1,1,1,1", help="Comma-separated per-joint multiplier.")
    parser.add_argument("--offset", default="0,0,0,0,0,0", help="Comma-separated per-joint offset.")
    parser.add_argument(
        "--ros-unit",
        choices={"rad", "lerobot"},
        default="rad",
        help="Unit used by ROS JointState positions. Affects runtime offset step display/conversion.",
    )
    parser.add_argument("--command-offset-topic", default=COMMAND_OFFSET_TOPIC)
    parser.add_argument(
        "--offset-step-deg",
        type=float,
        default=5.0,
        help="Runtime command offset step for body joints. In rad mode this is converted to radians.",
    )
    parser.add_argument(
        "--max-offset-deg",
        type=float,
        default=45.0,
        help="Absolute runtime command offset limit for body joints. In rad mode this is converted to radians.",
    )
    parser.add_argument(
        "--gripper-offset-step",
        type=float,
        default=5.0,
        help="Runtime gripper offset step when --ros-unit lerobot is used.",
    )
    parser.add_argument(
        "--max-gripper-offset",
        type=float,
        default=45.0,
        help="Absolute runtime gripper offset limit when --ros-unit lerobot is used.",
    )
    parser.add_argument("--no-runtime-ui", action="store_true", help="Disable the runtime stdin offset menu.")
    parser.add_argument("--qos-depth", type=int, default=1, help="QoS queue depth for teleop state/command topics.")
    parser.add_argument(
        "--best-effort",
        action="store_true",
        help="Use BEST_EFFORT reliability for lower latency on local teleop topics.",
    )
    parser.add_argument(
        "--lower-limit",
        default="none,none,none,none,none,none",
        help="Comma-separated lower clamp values, or 'none'.",
    )
    parser.add_argument(
        "--upper-limit",
        default="none,none,none,none,none,none",
        help="Comma-separated upper clamp values, or 'none'.",
    )
    parser.add_argument(
        "--publish-empty-velocity-effort",
        action="store_true",
        help="Publish empty velocity/effort arrays. By default zero arrays matching the joint count are used.",
    )
    return parser.parse_args()


class RuntimeOffsets:
    def __init__(self) -> None:
        self.values = {joint_name: 0.0 for joint_name in JOINT_ORDER}
        self.lock = threading.Lock()


def build_command_offsets_msg(node: Node, offsets: dict[str, float]) -> JointState:
    msg = JointState()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.name = list(JOINT_ORDER)
    msg.position = [offsets[joint_name] for joint_name in JOINT_ORDER]
    return msg


def format_offset_value(joint_name: str, value: float, ros_unit: str) -> str:
    if ros_unit == "rad":
        return f"{math.degrees(value):+.1f} deg"
    if joint_name == GRIPPER_JOINT:
        return f"{value:+.1f} lerobot"
    return f"{value:+.1f} deg"


class SO101Teleop(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("so101_teleop")
        self.args = args
        self.runtime_offsets = RuntimeOffsets()
        self.shutdown_requested = threading.Event()
        self.transforms = build_joint_transforms(
            target_joint_names=args.target_joint_names,
            scale=args.scale,
            offset=args.offset,
            lower_limit=args.lower_limit,
            upper_limit=args.upper_limit,
        )
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=args.qos_depth,
            reliability=ReliabilityPolicy.BEST_EFFORT if args.best_effort else ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.publisher = self.create_publisher(JointState, args.command_topic, qos)
        self.offset_publisher = self.create_publisher(JointState, args.command_offset_topic, qos)
        self.subscription = self.create_subscription(JointState, args.source_topic, self._on_leader_state, qos)
        self.warned_missing: set[tuple[str, ...]] = set()

        mapping = ", ".join(f"{item.source_name}->{item.target_name}" for item in self.transforms)
        self.get_logger().info(f"subscribing {args.source_topic}; publishing {args.command_topic}")
        self.get_logger().info(f"publishing runtime command offsets: {args.command_offset_topic}")
        self.get_logger().info(f"joint mapping: {mapping}")

    def get_command_offsets(self) -> dict[str, float]:
        with self.runtime_offsets.lock:
            return dict(self.runtime_offsets.values)

    def _command_offset_step(self, joint_name: str) -> float:
        if self.args.ros_unit == "rad":
            return math.radians(self.args.offset_step_deg)
        if joint_name == GRIPPER_JOINT:
            return self.args.gripper_offset_step
        return self.args.offset_step_deg

    def _max_command_offset(self, joint_name: str) -> float:
        if self.args.ros_unit == "rad":
            return math.radians(self.args.max_offset_deg)
        if joint_name == GRIPPER_JOINT:
            return self.args.max_gripper_offset
        return self.args.max_offset_deg

    def set_command_offset(self, joint_name: str, value: float) -> float:
        if joint_name not in JOINT_ORDER:
            raise ValueError(f"unknown joint: {joint_name}")
        max_offset = self._max_command_offset(joint_name)
        value = max(-max_offset, min(max_offset, value))
        with self.runtime_offsets.lock:
            self.runtime_offsets.values[joint_name] = value
        self.publish_command_offsets()
        return value

    def add_command_offset_step(self, joint_name: str, direction: int) -> float:
        if direction not in {-1, 1}:
            raise ValueError("direction must be -1 or 1")
        delta = self._command_offset_step(joint_name) * direction
        with self.runtime_offsets.lock:
            current = self.runtime_offsets.values[joint_name]
        return self.set_command_offset(joint_name, current + delta)

    def reset_command_offsets(self, joint_name: str | None = None) -> None:
        with self.runtime_offsets.lock:
            if joint_name is None:
                for name in JOINT_ORDER:
                    self.runtime_offsets.values[name] = 0.0
            else:
                if joint_name not in JOINT_ORDER:
                    raise ValueError(f"unknown joint: {joint_name}")
                self.runtime_offsets.values[joint_name] = 0.0
        self.publish_command_offsets()

    def apply_runtime_offsets(self, msg: JointState) -> JointState:
        offsets = self.get_command_offsets()
        for index, joint_name in enumerate(msg.name):
            if index < len(msg.position) and joint_name in offsets:
                msg.position[index] += offsets[joint_name]
        return msg

    def publish_command_offsets(self) -> None:
        self.offset_publisher.publish(build_command_offsets_msg(self, self.get_command_offsets()))

    def _on_leader_state(self, msg: JointState) -> None:
        try:
            command = transform_joint_state(
                self,
                msg,
                self.transforms,
                publish_empty_velocity_effort=self.args.publish_empty_velocity_effort,
            )
        except ValueError as exc:
            missing_key = tuple(str(exc).split(": ", maxsplit=1)[-1:])
            if missing_key not in self.warned_missing:
                self.warned_missing.add(missing_key)
                self.get_logger().warning(str(exc))
            return
        command = self.apply_runtime_offsets(command)
        self.publisher.publish(command)
        self.publish_command_offsets()


def print_command_offsets(node: SO101Teleop) -> None:
    offsets = node.get_command_offsets()
    print("\nCurrent command offsets:")
    for index, joint_name in enumerate(JOINT_ORDER, start=1):
        value = format_offset_value(joint_name, offsets[joint_name], node.args.ros_unit)
        print(f"  {index}. {joint_name:<14} {value}")


def prompt_runtime_joint_choice() -> str:
    print("\nSelect joint for command offset:")
    for index, joint_name in enumerate(JOINT_ORDER, start=1):
        print(f"  {index}. {joint_name}")
    print("  r. reset all offsets")
    print("  b. back")
    print("  exit. stop teleop")

    while True:
        value = input("Joint: ").strip().lower()
        if value == "exit":
            return "__exit__"
        if value in {"b", "back", "q", "quit"}:
            return ""
        if value in {"r", "reset"}:
            return "__reset_all__"
        if value.isdigit():
            index = int(value) - 1
            if 0 <= index < len(JOINT_ORDER):
                return JOINT_ORDER[index]
        if value in JOINT_ORDER:
            return value
        print(f"Choose 1..{len(JOINT_ORDER)}, r, or b.")


def run_command_offset_menu(node: SO101Teleop) -> bool:
    while True:
        print_command_offsets(node)
        joint_name = prompt_runtime_joint_choice()
        if not joint_name:
            return True
        if joint_name == "__exit__":
            return False
        if joint_name == "__reset_all__":
            node.reset_command_offsets()
            print("Reset all command offsets.")
            continue

        while True:
            current = node.get_command_offsets()[joint_name]
            step = node._command_offset_step(joint_name)
            current_text = format_offset_value(joint_name, current, node.args.ros_unit)
            step_text = format_offset_value(joint_name, step, node.args.ros_unit)
            print(f"\n{joint_name} offset: {current_text} (step {step_text})")
            value = input("Enter +, -, 0 reset, b back: ").strip().lower()
            if value == "exit":
                return False
            if value in {"b", "back", "q", "quit"}:
                break
            if value == "+":
                updated = node.add_command_offset_step(joint_name, 1)
            elif value == "-":
                updated = node.add_command_offset_step(joint_name, -1)
            elif value in {"0", "reset", "r"}:
                node.reset_command_offsets(joint_name)
                updated = 0.0
            else:
                print("Choose +, -, 0, or b.")
                continue
            updated_text = format_offset_value(joint_name, updated, node.args.ros_unit)
            print(f"Updated {joint_name}: {updated_text}")


def runtime_console_loop(node: SO101Teleop) -> None:
    print("\nRuntime UI: enter 'o' to set command offsets while teleop is running.")
    print("Enter 'exit' to stop teleop.")
    while rclpy.ok():
        try:
            value = input("Teleop menu [o offsets, exit, h help]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            node.shutdown_requested.set()
            return
        if value in {"o", "offset", "offsets"}:
            keep_running = run_command_offset_menu(node)
            if not keep_running:
                node.shutdown_requested.set()
                return
        elif value == "exit":
            node.shutdown_requested.set()
            return
        elif value in {"h", "help", "?"}:
            print("Commands: o = set command offsets, exit = stop teleop, h = help.")
        elif value:
            print("Unknown command. Enter 'h' for help.")


def main() -> int:
    args = parse_args()
    if args.qos_depth <= 0:
        raise ValueError("--qos-depth must be positive")
    if args.offset_step_deg <= 0:
        raise ValueError("--offset-step-deg must be positive")
    if args.max_offset_deg < 0:
        raise ValueError("--max-offset-deg must be non-negative")
    if args.gripper_offset_step <= 0:
        raise ValueError("--gripper-offset-step must be positive")
    if args.max_gripper_offset < 0:
        raise ValueError("--max-gripper-offset must be non-negative")

    rclpy.init()
    node = SO101Teleop(args)
    runtime_thread = None
    if not args.no_runtime_ui and sys.stdin.isatty():
        runtime_thread = threading.Thread(target=runtime_console_loop, args=(node,), daemon=True)
        runtime_thread.start()
    try:
        while rclpy.ok() and not node.shutdown_requested.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if runtime_thread is not None:
            runtime_thread.join(timeout=0.5)


if __name__ == "__main__":
    raise SystemExit(main())
