#!/usr/bin/env python3



from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import shlex
import time
import sys
import threading
from dataclasses import dataclass, asdict
from pathlib import Path

from so101_ros_contract import (
    FOLLOWER_COMMAND_TOPIC,
    FOLLOWER_STATE_TOPIC,
    LEADER_STATE_TOPIC,
    REAL_FOLLOWER_STATE_TOPIC,
    SIM_FOLLOWER_STATE_TOPIC,
    TOP_CAMERA_INFO_TOPIC,
    TOP_IMAGE_TOPIC,
    WRIST_CAMERA_INFO_TOPIC,
    WRIST_IMAGE_TOPIC,
)


DEFAULT_TOPICS = [
    WRIST_IMAGE_TOPIC,
    TOP_IMAGE_TOPIC,
    WRIST_CAMERA_INFO_TOPIC,
    TOP_CAMERA_INFO_TOPIC,
    REAL_FOLLOWER_STATE_TOPIC,
    SIM_FOLLOWER_STATE_TOPIC,
    FOLLOWER_STATE_TOPIC,
    LEADER_STATE_TOPIC,
    FOLLOWER_COMMAND_TOPIC,
    "/episode/event",
]
PROJECT_DIR = Path(__file__).resolve().parents[1]
ROS_DISTRO = os.environ.get("ROS_DISTRO", "humble")
ROS_SETUP = Path(f"/opt/ros/{ROS_DISTRO}/setup.bash")


@dataclass
class EpisodeMetadata:
    episode_id: str
    task: str
    robot: str
    result: str | None
    failure_type: str | None
    notes: str
    storage: str
    bag_prefix: str
    created_unix_time: float
    started_unix_time: float | None
    ended_unix_time: float | None
    bag_return_code: int | None
    topics: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record one SO-101 episode to rosbag2/MCAP.")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_DIR / "data/raw/pick_candy_v1"),
        help="Directory that contains per-episode folders.",
    )
    parser.add_argument(
        "--episode-id",
        default="",
        help="Episode directory name, for example episode_000001. Auto-allocates if omitted.",
    )
    parser.add_argument(
        "--task",
        default="Pick up the candy and place it in the bowl.",
        help="Task instruction to store in metadata.",
    )
    parser.add_argument("--robot", default="so101", help="Robot name to store in metadata.")
    parser.add_argument(
        "--result",
        default=None,
        choices={"success", "failure"},
        help="Episode result to store in metadata.",
    )
    parser.add_argument(
        "--failure-type",
        default=None,
        help="Failure type to store in metadata, for example miss_grasp.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Free-form notes to store in metadata.",
    )
    parser.add_argument(
        "--topic",
        action="append",
        dest="topics",
        help="Additional topic to record. Can be repeated. Defaults to the project schema.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command and metadata path without starting rosbag recording.",
    )
    return parser.parse_args()


def next_episode_id(output_root: Path) -> str:
    pattern = re.compile(r"^episode_(\d{6})$")
    max_index = 0
    if output_root.exists():
        for path in output_root.iterdir():
            if not path.is_dir():
                continue
            match = pattern.match(path.name)
            if match:
                max_index = max(max_index, int(match.group(1)))
    return f"episode_{max_index + 1:06d}"


def yaml_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    text = str(value)
    if text == "":
        return '""'
    if re.fullmatch(r"[A-Za-z0-9_./:-]+", text):
        return text
    return json.dumps(text)


def yaml_dump(data: dict[str, object], indent: int = 0) -> str:
    lines: list[str] = []
    prefix = " " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(yaml_dump(value, indent + 2))
        elif isinstance(value, list):
            if not value:
                lines.append(f"{prefix}{key}: []")
            else:
                lines.append(f"{prefix}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(f"{prefix}  -")
                        lines.append(yaml_dump(item, indent + 4))
                    else:
                        lines.append(f"{prefix}  - {yaml_scalar(item)}")
        else:
            lines.append(f"{prefix}{key}: {yaml_scalar(value)}")
    return "\n".join(lines)


def build_metadata(args: argparse.Namespace, episode_id: str, bag_prefix: Path, topics: list[str]) -> EpisodeMetadata:
    now = time.time()
    return EpisodeMetadata(
        episode_id=episode_id,
        task=args.task,
        robot=args.robot,
        result=args.result,
        failure_type=args.failure_type,
        notes=args.notes,
        storage="mcap",
        bag_prefix=str(bag_prefix),
        created_unix_time=now,
        started_unix_time=None,
        ended_unix_time=None,
        bag_return_code=None,
        topics=topics,
    )


def metadata_to_yaml(metadata: EpisodeMetadata) -> str:
    return yaml_dump(asdict(metadata)) + "\n"


def write_metadata(path: Path, metadata: EpisodeMetadata) -> None:
    path.write_text(metadata_to_yaml(metadata), encoding="utf-8")


def prompt_choice(prompt: str, choices: set[str], default: str | None = None) -> str:
    while True:
        suffix = f" [default: {default}]" if default else ""
        raw = input(f"{prompt}{suffix}: ").strip().lower()
        if not raw and default is not None:
            return default
        if raw in choices:
            return raw
        print(f"Invalid choice. Expected one of: {', '.join(sorted(choices))}")


def prompt_optional_text(prompt: str) -> str | None:
    raw = input(f"{prompt} (optional, blank for none): ").strip()
    return raw or None


def start_enter_to_stop_watcher(bag_process: subprocess.Popen[bytes]) -> threading.Thread | None:
    if not sys.stdin.isatty():
        return None

    def wait_for_enter() -> None:
        try:
            input("Press Enter to stop recording.")
        except EOFError:
            return
        if bag_process.poll() is None:
            try:
                bag_process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass

    thread = threading.Thread(target=wait_for_enter, daemon=True)
    thread.start()
    return thread


def resolve_output_root(output_root: str) -> Path:
    path = Path(output_root).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_DIR / path).resolve()


def main() -> int:
    args = parse_args()

    output_root = resolve_output_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    episode_id = args.episode_id or next_episode_id(output_root)
    episode_dir = output_root / episode_id
    episode_dir.mkdir(parents=True, exist_ok=False)

    bag_prefix = episode_dir / "episode"
    metadata_path = episode_dir / "metadata.yaml"
    topics = list(dict.fromkeys((args.topics or DEFAULT_TOPICS)))

    metadata = build_metadata(args, episode_id, bag_prefix, topics)
    metadata.storage = "mcap"
    metadata.started_unix_time = time.time()
    write_metadata(metadata_path, metadata)

    record_command = [
        "source",
        shlex.quote(str(ROS_SETUP)),
        "&&",
        "ros2",
        "bag",
        "record",
        "--storage",
        "mcap",
        "-o",
        shlex.quote(str(bag_prefix)),
        *[shlex.quote(topic) for topic in topics],
    ]
    command = ["bash", "-lc", " ".join(record_command)]

    print(f"episode_id: {episode_id}")
    print(f"output_dir: {episode_dir}")
    print(f"metadata: {metadata_path}")
    print("command:")
    print("  " + " ".join(command))

    if args.dry_run:
        return 0

    if not ROS_SETUP.is_file():
        raise FileNotFoundError(f"ROS setup not found: {ROS_SETUP}")

    bag_process = subprocess.Popen(command)
    interrupted = False
    stop_watcher = start_enter_to_stop_watcher(bag_process)

    def handle_signal(signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        if bag_process.poll() is None:
            try:
                bag_process.send_signal(signum)
            except ProcessLookupError:
                pass

    previous_sigint = signal.signal(signal.SIGINT, handle_signal)
    previous_sigterm = signal.signal(signal.SIGTERM, handle_signal)

    try:
        return_code = bag_process.wait()
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)

    metadata.ended_unix_time = time.time()
    metadata.bag_return_code = return_code

    if stop_watcher is not None:
        stop_watcher.join(timeout=0.1)

    if metadata.result is None and sys.stdin.isatty():
        metadata.result = prompt_choice("Result (s/f, then Enter)", {"s", "f"})
        metadata.result = "success" if metadata.result == "s" else "failure"
        if metadata.result == "failure" and metadata.failure_type is None:
            metadata.failure_type = prompt_optional_text("Failure type (then Enter)")
    elif metadata.result is None:
        metadata.result = "failure" if interrupted else "success"
        if metadata.result == "failure" and metadata.failure_type is None and interrupted:
            metadata.failure_type = "interrupted"

    write_metadata(metadata_path, metadata)

    return return_code if return_code is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
