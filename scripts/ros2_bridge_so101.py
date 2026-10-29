#!/usr/bin/env python3

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from lerobot.robots import make_robot_from_config, so_follower
from lerobot.teleoperators import make_teleoperator_from_config, so_leader
from lerobot.utils.robot_utils import precise_sleep

from so101_ros_contract import (
    FOLLOWER_COMMAND_TOPIC,
    FOLLOWER_STATE_TOPIC,
    GRIPPER_JOINT,
    JOINT_ORDER,
    LEADER_STATE_TOPIC,
    REAL_FOLLOWER_STATE_TOPIC,
    TOP_CAMERA_INFO_TOPIC,
    TOP_IMAGE_TOPIC,
    WRIST_CAMERA_INFO_TOPIC,
    WRIST_IMAGE_TOPIC,
    build_joint_state_msg,
    clamp,
    command_msg_to_lerobot_action,
    lerobot_key,
    normalize_lerobot_dict,
)
PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION_DIR = PROJECT_DIR / "calibrate"
UI_THEME = "auto"
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_BLUE = "\033[34m"
ANSI_YELLOW = "\033[33m"
ANSI_CYAN = "\033[36m"
ANSI_GREEN = "\033[32m"
ANSI_RED = "\033[31m"


@dataclass
class LatestCommand:
    action: dict[str, float] | None = None
    stamp_monotonic: float = 0.0
    lock: object = field(default_factory=threading.Lock)


@dataclass
class CameraCandidate:
    device: str
    ok: bool
    width: int = 0
    height: int = 0
    fps: float = 0.0
    fourcc: str = ""
    mean: float = 0.0
    std: float = 0.0
    reason: str = ""


def import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for camera scan/publish. Install python3-opencv or opencv-python.") from exc
    return cv2


def video_device_sort_key(device: str) -> int:
    return int(device.removeprefix("/dev/video"))


def default_video_devices() -> list[str]:
    devices = [path for path in glob("/dev/video*") if path.removeprefix("/dev/video").isdigit()]
    devices = [path for path in devices if 0 <= video_device_sort_key(path) <= 7]
    return sorted(devices, key=video_device_sort_key)


def serial_device_sort_key(device: str) -> tuple[str, int]:
    for prefix in ("/dev/ttyUSB", "/dev/ttyACM"):
        if device.startswith(prefix) and device.removeprefix(prefix).isdigit():
            return prefix, int(device.removeprefix(prefix))
    return device, 0


def default_serial_devices() -> list[str]:
    devices = glob("/dev/ttyUSB*") + glob("/dev/ttyACM*")
    devices = [path for path in devices if path.startswith(("/dev/ttyUSB", "/dev/ttyACM"))]
    return sorted(devices, key=serial_device_sort_key)


def fourcc_to_string(value: float) -> str:
    value = int(value)
    return "".join(chr((value >> (8 * index)) & 0xFF) for index in range(4)).strip()


def configure_camera_capture(cv2, cap, width: int, height: int, fps: float, fourcc: str) -> None:
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)


def probe_camera(device: str, width: int, height: int, fps: float, fourcc: str, samples: int = 5) -> CameraCandidate:
    cv2 = import_cv2()
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return CameraCandidate(device=device, ok=False, reason="open failed")

    configure_camera_capture(cv2, cap, width, height, fps, fourcc)

    frame = None
    for _ in range(samples):
        ok, candidate = cap.read()
        if ok and candidate is not None and candidate.size > 0:
            frame = candidate
            break

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
    actual_fourcc = fourcc_to_string(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()

    if frame is None:
        return CameraCandidate(
            device=device,
            ok=False,
            width=actual_width,
            height=actual_height,
            fps=actual_fps,
            fourcc=actual_fourcc,
            reason="no frame",
        )

    mean = float(frame.mean())
    std = float(frame.std())
    if frame.ndim != 3 or frame.shape[2] != 3:
        return CameraCandidate(
            device=device,
            ok=False,
            width=frame.shape[1],
            height=frame.shape[0],
            fps=actual_fps,
            fourcc=actual_fourcc,
            mean=mean,
            std=std,
            reason=f"not 3-channel image: shape={frame.shape}",
        )
    if std < 1.0:
        return CameraCandidate(
            device=device,
            ok=False,
            width=frame.shape[1],
            height=frame.shape[0],
            fps=actual_fps,
            fourcc=actual_fourcc,
            mean=mean,
            std=std,
            reason="blank/near-constant frame",
        )

    return CameraCandidate(
        device=device,
        ok=True,
        width=frame.shape[1],
        height=frame.shape[0],
        fps=actual_fps,
        fourcc=actual_fourcc,
        mean=mean,
        std=std,
        reason="rgb frame readable",
    )


def scan_cameras(width: int, height: int, fps: float, fourcc: str) -> list[CameraCandidate]:
    return [probe_camera(device, width, height, fps, fourcc) for device in default_video_devices()]


def set_ui_theme(theme: str) -> None:
    global UI_THEME
    UI_THEME = theme


def terminal_supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def resolved_ui_theme() -> str:
    if UI_THEME == "none" or not terminal_supports_color():
        return "none"
    if UI_THEME in {"dark", "light"}:
        return UI_THEME

    colorfgbg = os.environ.get("COLORFGBG", "")
    if colorfgbg:
        try:
            background = int(colorfgbg.split(";")[-1])
            return "light" if background >= 7 else "dark"
        except ValueError:
            pass
    return "dark"


def ui_text(text: str, role: str = "accent") -> str:
    theme = resolved_ui_theme()
    if theme == "none":
        return text
    if role == "title":
        color = ANSI_YELLOW if theme == "dark" else ANSI_BLUE
        return f"{ANSI_BOLD}{color}{text}{ANSI_RESET}"
    if role == "prompt":
        color = ANSI_CYAN if theme == "dark" else ANSI_BLUE
        return f"{color}{text}{ANSI_RESET}"
    if role == "success":
        return f"{ANSI_GREEN}{text}{ANSI_RESET}"
    if role == "warn":
        return f"{ANSI_YELLOW}{text}{ANSI_RESET}"
    if role == "error":
        return f"{ANSI_RED}{text}{ANSI_RESET}"
    color = ANSI_YELLOW if theme == "dark" else ANSI_BLUE
    return f"{color}{text}{ANSI_RESET}"


def print_camera_scan(candidates: list[CameraCandidate]) -> None:
    if not candidates:
        print("No /dev/video0..7 devices found.")
        return
    print_section("Camera scan")
    for index, candidate in enumerate(candidates):
        status = ui_text("OK ", "success") if candidate.ok else ui_text("BAD", "error")
        detail = (
            f"{candidate.width}x{candidate.height} fps={candidate.fps:.1f} "
            f"fourcc={candidate.fourcc or '-'} mean={candidate.mean:.1f} std={candidate.std:.1f}"
        )
        print(f"  [{index}] {status} {candidate.device:<11} {detail} {candidate.reason}")


def print_section(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(ui_text(title, "title"))
    print("=" * 72)


def prompt_choice(prompt: str, choices: set[str], default: str) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        value = input(ui_text(f"{prompt}{suffix}: ", "prompt")).strip()
        if not value and default:
            return default
        if value in choices:
            return value
        print(f"Choose one of: {', '.join(sorted(choices))}")


def prompt_numbered_choice(prompt: str, options: list[tuple[str, str]], default_index: int = 0) -> str:
    print(f"\n{ui_text(prompt, 'title')}")
    for index, (label, _) in enumerate(options, start=1):
        default_marker = " *" if index - 1 == default_index else ""
        print(f"  {index}. {label}{default_marker}")

    while True:
        value = input(ui_text(f"Select [default {default_index + 1}]: ", "prompt")).strip()
        if not value:
            return options[default_index][1]
        if value.isdigit():
            selected = int(value) - 1
            if 0 <= selected < len(options):
                return options[selected][1]
        print(f"Choose 1..{len(options)}")


def wait_for_reconnected_serial_device(role: str, before_devices: list[str], timeout_s: float = 30.0) -> str:
    print(f"\n{ui_text(f'Auto detect {role} serial device', 'title')}")
    print("  This detects the device even if it comes back with the same /dev/ttyUSB number.")
    print("  Unplug the target USB serial device, then press ENTER.")
    input(ui_text("  Press ENTER after unplugging: ", "prompt"))

    before = set(before_devices)
    unplugged_devices = set(default_serial_devices())
    missing_devices = sorted(before - unplugged_devices, key=serial_device_sort_key)
    if not missing_devices:
        print(ui_text("  No device disappeared. Check the cable and try auto detect again.", "warn"))
        return ""

    print(ui_text("  disappeared:", "warn"))
    for device in missing_devices:
        print(f"    {device}")

    print("  Plug the same USB serial device back in, then press ENTER.")
    input(ui_text("  Press ENTER after reconnecting: ", "prompt"))

    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        current_devices = set(default_serial_devices())
        returned_devices = sorted(current_devices & set(missing_devices), key=serial_device_sort_key)
        new_devices = sorted(current_devices - unplugged_devices, key=serial_device_sort_key)
        candidates = returned_devices or new_devices

        if len(candidates) == 1:
            print(f"  detected: {ui_text(candidates[0], 'accent')}")
            return candidates[0]
        if len(candidates) > 1:
            options = [(device, device) for device in candidates]
            return prompt_numbered_choice(f"Multiple new devices detected for {role}", options, 0)
        time.sleep(0.5)

    print(ui_text("  no reconnected serial device detected", "warn"))
    return ""


def prompt_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = "[default: yes]" if default else "[default: no]"
    while True:
        value = input(ui_text(f"{prompt}? (yes/no) {suffix}: ", "prompt")).strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Choose y or n.")


def prompt_text(prompt: str, default: str) -> str:
    value = input(ui_text(f"{prompt} [{default}]: ", "prompt")).strip()
    return value if value else default


def prompt_serial_device(role: str, devices: list[str]) -> str:
    options = [(device, device) for device in devices]
    options.append(("auto detect (reconnect)", "__auto_detect__"))
    options.append(("none (do not connect)", ""))
    default_index = 0 if devices else len(options) - 1
    selected = prompt_numbered_choice(f"Select {role} serial device", options, default_index)
    if selected == "__auto_detect__":
        return wait_for_reconnected_serial_device(role, devices)
    return selected


def calibration_path(args: argparse.Namespace, role: str) -> Path:
    device_id = args.leader_id if role == "leader" else args.follower_id
    return Path(args.calibration_dir).expanduser().resolve() / f"{device_id}.json"


def ensure_serial_device_permission(device: str) -> None:
    if not device:
        return
    if os.access(device, os.R_OK | os.W_OK):
        return

    print(f"\nSerial permission missing: {device}")
    print(f"  trying chmod 666 {device}")
    try:
        os.chmod(device, 0o666)
    except PermissionError:
        print(f"  permission denied; running sudo chmod 666 {device}")
        subprocess.run(["sudo", "chmod", "666", device], check=True)

    if not os.access(device, os.R_OK | os.W_OK):
        raise PermissionError(f"Could not get read/write permission for {device}")


def ensure_selected_serial_permissions(args: argparse.Namespace) -> None:
    for device in sorted({args.leader_port, args.follower_port} - {""}, key=serial_device_sort_key):
        ensure_serial_device_permission(device)


def choose_calibration(args: argparse.Namespace, role: str, interactive: bool) -> bool:
    path = calibration_path(args, role)
    print(f"\n{ui_text(f'{role.capitalize()} calibration file', 'title')}:")
    print(f"  {ui_text(str(path), 'accent')}")

    if not path.is_file():
        print(ui_text(f"  missing {role} calibration file; calibration is required", "warn"))
        return True

    if not interactive:
        print(f"  using existing {role} calibration file")
        return False

    use_existing = prompt_yes_no(f"Use existing {role} calibration file", True)
    if use_existing:
        print(f"  existing {role} calibration will be loaded automatically")
        return False

    print(ui_text(f"  starting new {role} calibration", "warn"))
    return True


def apply_calibration_policy(args: argparse.Namespace, interactive: bool) -> None:
    if args.leader_port and not args.calibrate_leader:
        args.calibrate_leader = choose_calibration(args, "leader", interactive)
    if args.follower_port and not args.calibrate_follower:
        args.calibrate_follower = choose_calibration(args, "follower", interactive)


def prompt_camera_device(role: str, candidates: list[CameraCandidate], default_index: int | None) -> str:
    options = []
    for candidate in candidates:
        if not candidate.ok:
            continue
        label = (
            f"{candidate.device} "
            f"({candidate.width}x{candidate.height}, mean={candidate.mean:.1f}, std={candidate.std:.1f})"
        )
        options.append((label, candidate.device))
    options.append(("none (do not publish)", ""))
    default_device = candidates[default_index].device if default_index is not None else ""
    default_option_index = next(
        (index for index, (_, device) in enumerate(options) if device == default_device),
        len(options) - 1,
    )
    return prompt_numbered_choice(f"Select {role} camera", options, default_option_index)


def run_cli_ui(args: argparse.Namespace) -> argparse.Namespace:
    print_section("SO-101 ROS2 hardware bridge setup")
    print("Calibration directory:")
    print(f"  {ui_text(str(Path(args.calibration_dir).expanduser().resolve()), 'accent')}")

    print_section("Serial device selection")
    serial_devices = default_serial_devices()
    if serial_devices:
        print("Detected serial devices:")
        for device in serial_devices:
            print(f"  {device}")
    else:
        print("No /dev/ttyUSB* or /dev/ttyACM* serial devices found.")

    args.leader_port = prompt_serial_device("leader", serial_devices)
    remaining_serial_devices = [device for device in serial_devices if device != args.leader_port]
    args.follower_port = prompt_serial_device("follower", remaining_serial_devices)

    if args.leader_port and args.follower_port:
        args.robot_mode = "both"
    elif args.leader_port:
        args.robot_mode = "leader"
    elif args.follower_port:
        args.robot_mode = "follower"
    else:
        args.robot_mode = "none"

    print("\nSelected robot mode:")
    print(f"  {ui_text(args.robot_mode, 'accent')}")
    if args.leader_port:
        print(f"  leader:   {ui_text(args.leader_port, 'accent')}")
    if args.follower_port:
        print(f"  follower: {ui_text(args.follower_port, 'accent')}")

    print_section("Serial permission")
    ensure_selected_serial_permissions(args)

    print_section("Calibration")
    apply_calibration_policy(args, interactive=True)
    run_selected_calibrations(args)
    args.calibrate_leader = False
    args.calibrate_follower = False

    print_section("Camera selection")
    candidates = scan_cameras(args.camera_width, args.camera_height, args.camera_fps, args.camera_fourcc)
    print_camera_scan(candidates)
    ok_indices = [index for index, candidate in enumerate(candidates) if candidate.ok]
    args.wrist_camera = prompt_camera_device("wrist", candidates, ok_indices[0] if ok_indices else None)
    args.top_camera = prompt_camera_device("top", candidates, ok_indices[1] if len(ok_indices) > 1 else None)
    if args.wrist_camera or args.top_camera:
        args.visualize_cameras = prompt_yes_no("Show camera visualization window", args.visualize_cameras)

    return args


class SO101Ros2Bridge(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("so101_ros2_bridge")
        self.args = args
        self.latest_command = LatestCommand()
        self.camera_publish_enabled = bool(args.wrist_camera or args.top_camera)
        self.camera_visualization_enabled = bool(args.visualize_cameras)
        self.rest_return_requested = threading.Event()
        self.suppress_commands_until_monotonic = 0.0
        self.shutdown_requested = threading.Event()

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.leader_pub = self.create_publisher(JointState, args.leader_state_topic, qos)
        self.follower_pub = self.create_publisher(JointState, args.follower_state_topic, qos)
        self.follower_alias_pub = None
        if args.publish_follower_state_alias:
            self.follower_alias_pub = self.create_publisher(JointState, args.follower_state_alias_topic, qos)
        self.command_sub = None
        if args.robot_mode in {"follower", "both"}:
            self.command_sub = self.create_subscription(
                JointState,
                args.command_topic,
                self._on_command,
                command_qos,
            )
    def _on_command(self, msg: JointState) -> None:
        if time.monotonic() < self.suppress_commands_until_monotonic:
            return
        try:
            action = command_msg_to_lerobot_action(msg, self.args.ros_unit)
            stamp_monotonic = time.monotonic()
            with self.latest_command.lock:
                self.latest_command.action = action
                self.latest_command.stamp_monotonic = stamp_monotonic
        except ValueError as exc:
            self.get_logger().warning(str(exc))

    def latest_command_snapshot(self) -> tuple[dict[str, float] | None, float]:
        with self.latest_command.lock:
            if self.latest_command.action is None:
                return None, self.latest_command.stamp_monotonic
            return dict(self.latest_command.action), self.latest_command.stamp_monotonic

    def request_shutdown(self) -> None:
        self.shutdown_requested.set()

    def request_rest_return(self) -> bool:
        self.rest_return_requested.set()
        self.get_logger().info("Rest return requested.")
        return True

    def consume_rest_return_request(self) -> bool:
        if not self.rest_return_requested.is_set():
            return False
        self.rest_return_requested.clear()
        return True

    def suppress_commands_for(self, duration_s: float) -> None:
        with self.latest_command.lock:
            self.latest_command.action = None
        self.suppress_commands_until_monotonic = time.monotonic() + max(duration_s, 0.0)

    def toggle_camera_publish(self) -> bool:
        if not self.args.wrist_camera and not self.args.top_camera:
            self.get_logger().warning("Camera publish toggle ignored because no cameras are selected.")
            self.camera_publish_enabled = False
            return self.camera_publish_enabled
        self.camera_publish_enabled = not self.camera_publish_enabled
        state = "enabled" if self.camera_publish_enabled else "disabled"
        self.get_logger().info(f"Camera publish {state}.")
        return self.camera_publish_enabled

    def toggle_camera_visualization(self) -> bool:
        self.camera_visualization_enabled = not self.camera_visualization_enabled
        state = "enabled" if self.camera_visualization_enabled else "disabled"
        self.get_logger().info(f"Camera visualization {state}.")
        return self.camera_visualization_enabled

def make_leader(args: argparse.Namespace):
    if args.robot_mode not in {"leader", "both"} or not args.leader_port:
        return None

    cfg = so_leader.SO101LeaderConfig(
        port=args.leader_port,
        id=args.leader_id,
        calibration_dir=Path(args.calibration_dir).expanduser().resolve(),
        use_degrees=True,
    )
    teleop = make_teleoperator_from_config(cfg)
    teleop.connect(calibrate=args.auto_calibrate_if_needed)
    return teleop


def make_follower(args: argparse.Namespace):
    if args.robot_mode not in {"follower", "both"} or not args.follower_port:
        return None

    cfg = so_follower.SO101FollowerConfig(
        port=args.follower_port,
        id=args.follower_id,
        calibration_dir=Path(args.calibration_dir).expanduser().resolve(),
        use_degrees=True,
        max_relative_target=args.max_relative_target,
        disable_torque_on_disconnect=not args.keep_torque_on_disconnect,
    )
    robot = make_robot_from_config(cfg)
    robot.connect(calibrate=args.auto_calibrate_if_needed)
    return robot


def rest_position_path(args: argparse.Namespace) -> Path:
    if args.rest_position_file:
        return Path(args.rest_position_file).expanduser().resolve()
    return Path(args.calibration_dir).expanduser().resolve() / "follower_rest_position.json"


def lerobot_values_to_action(values: dict[str, float]) -> dict[str, float]:
    return {lerobot_key(joint_name): values[joint_name] for joint_name in JOINT_ORDER}


def rest_step_size(args: argparse.Namespace, joint_name: str) -> float:
    if joint_name == "shoulder_lift":
        return args.rest_shoulder_lift_step_deg
    if joint_name == GRIPPER_JOINT:
        return args.rest_gripper_step
    return args.rest_joint_step_deg


def rest_tolerance(args: argparse.Namespace, joint_name: str) -> float:
    if joint_name == GRIPPER_JOINT:
        return args.rest_gripper_tolerance
    return args.rest_joint_tolerance_deg


def record_follower_rest_position(args: argparse.Namespace, follower) -> dict[str, float] | None:
    if follower is None:
        return None

    values = normalize_lerobot_dict(follower.get_observation())
    action = lerobot_values_to_action(values)
    path = rest_position_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "Follower rest position captured before SO-101 ROS2 bridge publish/teleop loop starts.",
        "created_unix_time": time.time(),
        "follower_id": args.follower_id,
        "joint_order": JOINT_ORDER,
        "lerobot_position": values,
        "lerobot_action": action,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved follower rest position: {ui_text(str(path), 'accent')}")
    return action


def return_follower_to_rest(
    args: argparse.Namespace,
    follower,
    rest_action: dict[str, float] | None,
    *,
    force: bool = False,
) -> None:
    if follower is None or rest_action is None or (args.no_rest_on_exit and not force):
        return

    print(ui_text("Returning follower arm to recorded rest position one joint at a time.", "warn"))
    period_s = 1.0 / max(args.fps, 1.0)

    try:
        hold_values = normalize_lerobot_dict(follower.get_observation())
        for joint_name in JOINT_ORDER:
            rest_key = lerobot_key(joint_name)
            rest_value = rest_action[rest_key]
            step = rest_step_size(args, joint_name)
            tolerance = rest_tolerance(args, joint_name)
            deadline = time.monotonic() + args.rest_joint_timeout

            print(f"  aligning {joint_name} -> {rest_value:.3f} (tol {tolerance:.3f}, step {step:.3f})")
            while True:
                current_values = normalize_lerobot_dict(follower.get_observation())
                current_value = current_values[joint_name]
                error = rest_value - current_value
                if abs(error) <= tolerance:
                    break
                if time.monotonic() >= deadline:
                    print(ui_text(f"  {joint_name} did not reach rest tolerance before timeout; continuing.", "warn"))
                    break

                next_value = current_value + clamp(error, -step, step)
                command_values = dict(hold_values)
                command_values[joint_name] = next_value
                if args.debug:
                    print(
                        f"    {joint_name}: current={current_value:.3f}, "
                        f"target={next_value:.3f}, rest={rest_value:.3f}, error={error:.3f}"
                    )
                follower.send_action(lerobot_values_to_action(command_values))
                precise_sleep(period_s)
            hold_values[joint_name] = normalize_lerobot_dict(follower.get_observation())[joint_name]
    except KeyboardInterrupt:
        print(ui_text("Rest return interrupted. Disconnecting follower.", "warn"))


def calibrate_leader(args: argparse.Namespace) -> None:
    if not args.leader_port:
        return
    cfg = so_leader.SO101LeaderConfig(
        port=args.leader_port,
        id=args.leader_id,
        calibration_dir=Path(args.calibration_dir).expanduser().resolve(),
        use_degrees=True,
    )
    teleop = make_teleoperator_from_config(cfg)
    teleop.connect(calibrate=False)
    try:
        teleop.calibrate()
    finally:
        teleop.disconnect()


def calibrate_follower(args: argparse.Namespace) -> None:
    if not args.follower_port:
        return
    cfg = so_follower.SO101FollowerConfig(
        port=args.follower_port,
        id=args.follower_id,
        calibration_dir=Path(args.calibration_dir).expanduser().resolve(),
        use_degrees=True,
        max_relative_target=args.max_relative_target,
        disable_torque_on_disconnect=not args.keep_torque_on_disconnect,
    )
    robot = make_robot_from_config(cfg)
    robot.connect(calibrate=False)
    try:
        robot.calibrate()
    finally:
        robot.disconnect()


def run_selected_calibrations(args: argparse.Namespace) -> None:
    if args.calibrate_leader:
        while True:
            print("\nStarting leader calibration.")
            try:
                calibrate_leader(args)
                break
            except Exception as exc:
                print(f"\nLeader calibration failed:\n  {exc}")
                choice = prompt_choice("Choose retry, skip, or abort", {"retry", "skip", "abort"}, "retry")
                if choice == "retry":
                    continue
                if choice == "skip":
                    break
                raise
    if args.calibrate_follower:
        while True:
            print("\nStarting follower calibration.")
            try:
                calibrate_follower(args)
                break
            except Exception as exc:
                print(f"\nFollower calibration failed:\n  {exc}")
                choice = prompt_choice("Choose retry, skip, or abort", {"retry", "skip", "abort"}, "retry")
                if choice == "retry":
                    continue
                if choice == "skip":
                    break
                raise


def runtime_console_loop(node: SO101Ros2Bridge, stop_event: threading.Event) -> None:
    print(f"\n{ui_text('Runtime UI', 'title')}: bridge hardware controls.")
    print("Enter 'r' to return the follower arm to rest position.")
    print("Enter 'c' to toggle camera publish on/off while the bridge is running.")
    print("Enter 'v' to toggle camera visualization on/off while the bridge is running.")
    print("Enter 'exit' to return the follower arm to rest position and stop the bridge.")
    while not stop_event.is_set():
        try:
            value = input(ui_text("Bridge menu [r rest, c camera publish, v view, exit, h help]: ", "prompt")).strip().lower()
        except EOFError:
            return
        except KeyboardInterrupt:
            return

        if value in {"r", "rest", "return", "return rest"}:
            requested = node.request_rest_return()
            if requested:
                print(ui_text("Returning follower to rest position.", "warn"))
        elif value in {"c", "camera", "cameras", "publish", "camera publish"}:
            enabled = node.toggle_camera_publish()
            print(ui_text(f"Camera publish is now {'enabled' if enabled else 'disabled'}.", "warn"))
        elif value in {"v", "view", "visualize", "visualization"}:
            enabled = node.toggle_camera_visualization()
            print(ui_text(f"Camera visualization is now {'enabled' if enabled else 'disabled'}.", "warn"))
        elif value == "exit":
            print(ui_text("Exit requested. Stopping bridge after current loop.", "warn"))
            node.request_shutdown()
            return
        elif value in {"h", "help", "?"}:
            print("Commands: r = return to rest, c = toggle camera publish, v = toggle camera visualization, exit = return follower to rest and stop, h = help.")
        elif value:
            print("Unknown command. Enter 'h' for help.")


def start_runtime_console_if_enabled(args: argparse.Namespace, node: SO101Ros2Bridge) -> tuple[threading.Event, threading.Thread | None]:
    stop_event = threading.Event()
    if args.no_runtime_ui:
        return stop_event, None
    if not sys.stdin.isatty():
        node.get_logger().info("Runtime UI disabled because stdin is not interactive.")
        return stop_event, None

    thread = threading.Thread(target=runtime_console_loop, args=(node, stop_event), daemon=True)
    thread.start()
    return stop_event, thread


def spin_ros_callbacks(node: SO101Ros2Bridge) -> None:
    while rclpy.ok() and not node.shutdown_requested.is_set():
        rclpy.spin_once(node, timeout_sec=0.001)


def start_camera_bridge_process(args: argparse.Namespace) -> subprocess.Popen | None:
    if not args.wrist_camera and not args.top_camera:
        return None

    command = [
        sys.executable,
        str(PROJECT_DIR / "scripts" / "ros2_bridge_camera.py"),
        "--camera-width",
        str(args.camera_width),
        "--camera-height",
        str(args.camera_height),
        "--camera-fps",
        str(args.camera_fps),
        "--camera-fourcc",
        args.camera_fourcc,
        "--wrist-image-topic",
        args.wrist_image_topic,
        "--top-image-topic",
        args.top_image_topic,
        "--wrist-camera-info-topic",
        args.wrist_camera_info_topic,
        "--top-camera-info-topic",
        args.top_camera_info_topic,
    ]
    if args.wrist_camera:
        command.extend(["--wrist-camera", args.wrist_camera])
    if args.top_camera:
        command.extend(["--top-camera", args.top_camera])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    def lower_camera_priority() -> None:
        try:
            os.nice(10)
        except OSError:
            pass

    return subprocess.Popen(command, env=env, preexec_fn=lower_camera_priority)


def camera_publish_desired(args: argparse.Namespace, node: SO101Ros2Bridge) -> bool:
    return bool((args.wrist_camera or args.top_camera) and node.camera_publish_enabled)


def start_camera_viewer_processes(args: argparse.Namespace, node: Node) -> list[subprocess.Popen]:
    if not args.visualize_cameras:
        return []
    viewer_script = PROJECT_DIR / "scripts" / "ros2_view_camera.py"
    if not viewer_script.exists():
        node.get_logger().warning("Camera viewer disabled: ros2_view_camera.py not found.")
        return []

    command = [
        sys.executable,
        str(viewer_script),
        "--mode",
        "windows",
        "--fps",
        str(args.visualize_fps),
        "--cell-width",
        str(args.visualize_cell_width),
        "--cell-height",
        str(args.visualize_cell_height),
    ]
    if args.wrist_camera:
        command.extend(["--wrist-topic", args.wrist_image_topic])
    if args.top_camera:
        command.extend(["--top-topic", args.top_image_topic])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        process = subprocess.Popen(command, env=env)
    except OSError as exc:
        node.get_logger().warning(f"Failed to start camera viewer: {exc}")
        return []
    node.get_logger().info(f"Started camera viewer pid={process.pid}")
    return [process]


def stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3.0)


def stop_processes(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        stop_process(process)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish real SO-101 leader/follower and USB cameras as ROS2 topics."
    )
    parser.add_argument("--ui", action="store_true", help="Open an interactive CLI setup UI.")
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Skip the interactive CLI setup UI and use command-line arguments directly.",
    )
    parser.add_argument(
        "--ui-theme",
        choices=["auto", "dark", "light", "none"],
        default="auto",
        help="CLI color theme. Dark uses yellow accents, light uses blue accents.",
    )
    parser.add_argument(
        "--robot-mode",
        choices=["leader", "follower", "both", "none"],
        default="both",
        help="Which SO-101 serial hardware to connect.",
    )
    parser.add_argument("--leader-port", default="", help="Leader serial port. Empty disables leader.")
    parser.add_argument("--follower-port", default="", help="Follower serial port. Empty disables follower.")
    parser.add_argument("--leader-id", default="leader_calibrate")
    parser.add_argument("--follower-id", default="follower_calibrate")
    parser.add_argument(
        "--calibration-dir",
        default=str(DEFAULT_CALIBRATION_DIR),
        help="Directory for LeRobot calibration JSON files.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--ros-unit",
        choices=["rad", "lerobot"],
        default="rad",
        help="rad publishes body joints in radians and maps gripper 0..100 to the Isaac gripper rad range. "
        "lerobot publishes raw LeRobot units: body degrees, gripper 0..100.",
    )
    parser.add_argument("--leader-state-topic", default=LEADER_STATE_TOPIC)
    parser.add_argument("--follower-state-topic", default=REAL_FOLLOWER_STATE_TOPIC)
    parser.add_argument("--follower-state-alias-topic", default=FOLLOWER_STATE_TOPIC)
    parser.add_argument(
        "--publish-follower-state-alias",
        action="store_true",
        help="Also publish real follower state on /so101/follower/joint_states for legacy consumers.",
    )
    parser.add_argument("--command-topic", default=FOLLOWER_COMMAND_TOPIC)
    parser.add_argument("--wrist-camera", default="", help="Video device for wrist camera, e.g. /dev/video0.")
    parser.add_argument("--top-camera", default="", help="Video device for top camera, e.g. /dev/video2.")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=10.0)
    parser.add_argument(
        "--camera-fourcc",
        default="MJPG",
        help="Requested camera pixel format FOURCC. Use an empty string to let OpenCV/V4L2 choose.",
    )
    parser.add_argument("--visualize-cameras", action="store_true", help="Show selected cameras in GUI viewer windows.")
    parser.add_argument("--visualize-fps", type=float, default=10.0)
    parser.add_argument("--visualize-cell-width", type=int, default=480)
    parser.add_argument("--visualize-cell-height", type=int, default=360)
    parser.add_argument("--wrist-image-topic", default=WRIST_IMAGE_TOPIC)
    parser.add_argument("--top-image-topic", default=TOP_IMAGE_TOPIC)
    parser.add_argument("--wrist-camera-info-topic", default=WRIST_CAMERA_INFO_TOPIC)
    parser.add_argument("--top-camera-info-topic", default=TOP_CAMERA_INFO_TOPIC)
    parser.add_argument(
        "--scan-cameras",
        action="store_true",
        help="Scan /dev/video0..7 for devices that return real 3-channel frames, print results, then exit.",
    )
    parser.add_argument(
        "--command-timeout",
        type=float,
        default=0.5,
        help="Seconds before an external command is considered stale.",
    )
    parser.add_argument("--no-runtime-ui", action="store_true", help="Disable the runtime stdin menu.")
    parser.add_argument(
        "--rest-position-file",
        default="",
        help="Path to save the follower rest position captured before publishing starts. Default: calibration_dir/follower_rest_position.json.",
    )
    parser.add_argument(
        "--rest-joint-step-deg",
        type=float,
        default=8.0,
        help="Maximum body joint movement per control tick when returning to rest, in LeRobot degrees.",
    )
    parser.add_argument(
        "--rest-shoulder-lift-step-deg",
        type=float,
        default=15.0,
        help="Maximum shoulder_lift movement per control tick when returning to rest, in LeRobot degrees.",
    )
    parser.add_argument(
        "--rest-joint-tolerance-deg",
        type=float,
        default=5.0,
        help="Body joint tolerance for moving to the next joint during rest return, in LeRobot degrees.",
    )
    parser.add_argument(
        "--rest-gripper-step",
        type=float,
        default=5.0,
        help="Maximum gripper movement per control tick when returning to rest, in LeRobot gripper units.",
    )
    parser.add_argument(
        "--rest-gripper-tolerance",
        type=float,
        default=5.0,
        help="Gripper tolerance for moving past gripper during rest return, in LeRobot gripper units.",
    )
    parser.add_argument(
        "--rest-joint-timeout",
        type=float,
        default=5.0,
        help="Maximum seconds to spend aligning one joint before continuing to the next joint.",
    )
    parser.add_argument(
        "--no-rest-on-exit",
        action="store_true",
        help="Do not send the follower arm back to the recorded rest position on exit.",
    )
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=-1.0,
        help="LeRobot safety clamp in LeRobot units. Default -1 disables it.",
    )
    parser.add_argument("--calibrate-leader", action="store_true", help="Force leader calibration before bridge start.")
    parser.add_argument("--calibrate-follower", action="store_true", help="Force follower calibration before bridge start.")
    parser.add_argument(
        "--auto-calibrate-if-needed",
        action="store_true",
        help="Let LeRobot prompt for calibration if no valid calibration exists while connecting.",
    )
    parser.add_argument(
        "--keep-torque-on-disconnect",
        action="store_true",
        help="Leave follower torque enabled when the process exits.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.max_relative_target < 0:
        args.max_relative_target = None
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.camera_fps <= 0:
        raise ValueError("--camera-fps must be positive")
    if args.camera_width <= 0 or args.camera_height <= 0:
        raise ValueError("--camera-width and --camera-height must be positive")
    if args.visualize_fps <= 0:
        raise ValueError("--visualize-fps must be positive")
    if args.visualize_cell_width <= 0 or args.visualize_cell_height <= 0:
        raise ValueError("--visualize-cell-width and --visualize-cell-height must be positive")
    args.camera_fourcc = args.camera_fourcc.upper()
    if args.camera_fourcc and len(args.camera_fourcc) != 4:
        raise ValueError("--camera-fourcc must be exactly 4 characters, for example MJPG or YUYV")
    if args.rest_joint_step_deg <= 0:
        raise ValueError("--rest-joint-step-deg must be positive")
    if args.rest_shoulder_lift_step_deg <= 0:
        raise ValueError("--rest-shoulder-lift-step-deg must be positive")
    if args.rest_joint_tolerance_deg < 0:
        raise ValueError("--rest-joint-tolerance-deg must be non-negative")
    if args.rest_gripper_step <= 0:
        raise ValueError("--rest-gripper-step must be positive")
    if args.rest_gripper_tolerance < 0:
        raise ValueError("--rest-gripper-tolerance must be non-negative")
    if args.rest_joint_timeout <= 0:
        raise ValueError("--rest-joint-timeout must be positive")
    return args


def main() -> None:
    args = parse_args()
    set_ui_theme(args.ui_theme)

    if args.scan_cameras:
        print_camera_scan(scan_cameras(args.camera_width, args.camera_height, args.camera_fps, args.camera_fourcc))
        return

    use_interactive_ui = args.ui or not args.no_ui
    if use_interactive_ui:
        args = run_cli_ui(args)
    else:
        ensure_selected_serial_permissions(args)
        apply_calibration_policy(args, interactive=False)

    run_selected_calibrations(args)

    rclpy.init()
    node = SO101Ros2Bridge(args)
    leader = None
    follower = None
    follower_rest_action = None
    runtime_stop_event = None
    ros_spin_thread = None
    camera_process = None
    camera_viewer_processes = []

    try:
        runtime_stop_event, _ = start_runtime_console_if_enabled(args, node)
        ros_spin_thread = threading.Thread(target=spin_ros_callbacks, args=(node,), daemon=True)
        ros_spin_thread.start()
        leader = make_leader(args)
        follower = make_follower(args)
        follower_rest_action = record_follower_rest_position(args, follower)

        node.get_logger().info(f"JOINT_ORDER: {JOINT_ORDER}")
        node.get_logger().info(f"robot_mode={args.robot_mode}")
        if leader is not None:
            node.get_logger().info(f"Publishing leader state: {args.leader_state_topic}")
        if follower is not None:
            node.get_logger().info(f"Publishing follower state: {args.follower_state_topic}")
            if node.follower_alias_pub is not None:
                node.get_logger().info(f"Publishing follower state alias: {args.follower_state_alias_topic}")
            node.get_logger().info(f"Subscribing commands: {args.command_topic}")
        node.get_logger().info(
            f"fps={args.fps}, ros_unit={args.ros_unit}, camera_publish={node.camera_publish_enabled}"
        )

        if camera_publish_desired(args, node):
            camera_process = start_camera_bridge_process(args)
            if camera_process is not None:
                node.get_logger().info(f"Started camera bridge process pid={camera_process.pid}")
        if node.camera_visualization_enabled:
            camera_viewer_processes = start_camera_viewer_processes(args, node)

        while rclpy.ok() and not node.shutdown_requested.is_set():
            loop_start = time.perf_counter()

            if leader is not None:
                leader_values = normalize_lerobot_dict(leader.get_action())
                leader_msg = build_joint_state_msg(node, leader_values, args.ros_unit)
                node.leader_pub.publish(leader_msg)

            if follower is not None:
                follower_values = normalize_lerobot_dict(follower.get_observation())
                follower_msg = build_joint_state_msg(node, follower_values, args.ros_unit)
                node.follower_pub.publish(follower_msg)
                if node.follower_alias_pub is not None:
                    node.follower_alias_pub.publish(follower_msg)

                if node.consume_rest_return_request():
                    node.suppress_commands_for(args.command_timeout)
                    return_follower_to_rest(args, follower, follower_rest_action, force=True)
                    node.suppress_commands_for(args.command_timeout)
                else:
                    latest_action, latest_stamp = node.latest_command_snapshot()
                    command_age = time.monotonic() - latest_stamp
                    if latest_action is not None and command_age <= args.command_timeout:
                        sent = follower.send_action(latest_action)
                        if args.debug:
                            compact = {key: round(value, 3) for key, value in sent.items()}
                            node.get_logger().info(f"sent action: {compact}")
            elif node.consume_rest_return_request():
                node.get_logger().warning("Rest return skipped because follower is not connected.")

            should_publish_cameras = camera_publish_desired(args, node)
            if should_publish_cameras and camera_process is None:
                camera_process = start_camera_bridge_process(args)
                if camera_process is not None:
                    node.get_logger().info(f"Started camera bridge process pid={camera_process.pid}")
            elif not should_publish_cameras and camera_process is not None:
                stop_process(camera_process)
                camera_process = None

            if node.camera_visualization_enabled and not camera_viewer_processes and (args.wrist_camera or args.top_camera):
                camera_viewer_processes = start_camera_viewer_processes(args, node)
            elif not node.camera_visualization_enabled:
                stop_processes(camera_viewer_processes)
                camera_viewer_processes = []

            precise_sleep(max(1.0 / args.fps - (time.perf_counter() - loop_start), 0.0))

    except KeyboardInterrupt:
        pass
    finally:
        node.request_shutdown()
        if runtime_stop_event is not None:
            runtime_stop_event.set()
        if ros_spin_thread is not None:
            ros_spin_thread.join(timeout=1.0)
        stop_processes(camera_viewer_processes)
        stop_process(camera_process)
        if follower is not None:
            try:
                return_follower_to_rest(args, follower, follower_rest_action)
            finally:
                follower.disconnect()
        if leader is not None:
            leader.disconnect()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
