#!/usr/bin/env python3



from __future__ import annotations

import argparse
import contextlib
import json
import re
import shutil
import tempfile
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions
from sensor_msgs.msg import Image as ImageMsg
from sensor_msgs.msg import JointState

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from so101_ros_contract import (
    FOLLOWER_COMMAND_TOPIC,
    JOINT_ORDER,
    TOP_IMAGE_TOPIC,
    WRIST_IMAGE_TOPIC,
    ros_to_lerobot_position,
    target_state_topic,
)


WristTopic = WRIST_IMAGE_TOPIC
TopTopic = TOP_IMAGE_TOPIC
StateTopic = target_state_topic("real")
ActionTopic = FOLLOWER_COMMAND_TOPIC

DEFAULT_REQUIRED_TOPICS = [WristTopic, TopTopic, StateTopic, ActionTopic]


@dataclass
class TopicSeries:
    stamps_ns: list[int]
    messages: list[object]


@dataclass
class EpisodeSpec:
    episode_dir: Path
    bag_uri: Path
    metadata: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SO-101 rosbag2/MCAP episodes into a LeRobot dataset."
    )
    parser.add_argument(
        "--input",
        action="append",
        dest="inputs",
        help=(
            "Episode directory or a parent directory containing episode_* folders. "
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="",
        help=(
            "Local output directory for the LeRobot dataset. "
            "If omitted, uses data/lerobot/<raw-folder-name>."
        ),
    )
    parser.add_argument(
        "--repo-id",
        default="",
        help="LeRobot repo id used in metadata. Defaults to the output directory name.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target dataset FPS.",
    )
    parser.add_argument(
        "--task",
        default="",
        help="Task instruction override. If omitted, the episode metadata task is used.",
    )
    parser.add_argument(
        "--ros-unit",
        choices={"rad", "lerobot"},
        default="rad",
        help="Unit stored in the dataset for joint state/action values.",
    )
    parser.add_argument(
        "--target",
        choices={"real", "sim", "alias"},
        default="real",
        help="State topic preset to convert. real=/so101/real/follower/joint_states, sim=/so101/sim/follower/joint_states.",
    )
    parser.add_argument(
        "--state-topic",
        default="",
        help="Override state topic. Defaults from --target.",
    )
    parser.add_argument(
        "--action-topic",
        default=FOLLOWER_COMMAND_TOPIC,
        help="Action topic to convert.",
    )
    parser.add_argument(
        "--max-nearest-gap-s",
        type=float,
        default=0.25,
        help="Maximum allowed gap for nearest-neighbor matching.",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Also convert episodes whose metadata result is not success.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory before writing the dataset.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Reload the generated dataset at the end for a quick validation pass.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the generated dataset to the Hugging Face Hub after conversion.",
    )
    return parser.parse_args()


def parse_metadata(path: Path) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if not path.is_file():
        return metadata

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith(" "):
            continue
        match = re.match(r"^([A-Za-z0-9_/-]+):\s*(.*)$", line)
        if not match:
            continue
        key, raw_value = match.groups()
        raw_value = raw_value.strip()
        if raw_value in {"", "null", "~"}:
            metadata[key] = None
        elif raw_value in {"true", "false"}:
            metadata[key] = raw_value == "true"
        elif (raw_value.startswith('"') and raw_value.endswith('"')) or (
            raw_value.startswith("'") and raw_value.endswith("'")
        ):
            metadata[key] = raw_value[1:-1]
        else:
            try:
                metadata[key] = int(raw_value)
            except ValueError:
                try:
                    metadata[key] = float(raw_value)
                except ValueError:
                    metadata[key] = raw_value
    return metadata


def resolve_output_root(output_root: str) -> Path:
    path = Path(output_root).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def default_output_root_from_episode_specs(episode_specs: list[EpisodeSpec]) -> Path:
    if not episode_specs:
        raise ValueError("cannot infer output root without episodes")
    raw_root_name = episode_specs[0].episode_dir.parent.name
    return Path("data/lerobot") / raw_root_name


def episode_bag_uri(episode_dir: Path) -> Path:
    bag_uri = episode_dir / "episode"
    if bag_uri.exists():
        return bag_uri
    mcap_files = sorted(episode_dir.glob("*.mcap"))
    if len(mcap_files) == 1:
        return mcap_files[0].with_suffix("")
    raise FileNotFoundError(f"cannot find bag URI under {episode_dir}")


def discover_episodes(inputs: list[str] | None) -> list[EpisodeSpec]:
    if not inputs:
        raise ValueError("at least one --input is required")

    episode_dirs: list[Path] = []
    for raw_input in inputs:
        path = Path(raw_input).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"input path does not exist: {path}")
        if path.is_file():
            episode_dirs.append(path.parent)
            continue

        metadata = path / "metadata.yaml"
        bag_prefix = path / "episode"
        if metadata.exists() or bag_prefix.exists() or any(path.glob("*.mcap")):
            episode_dirs.append(path)
            continue

        candidates = [
            child
            for child in sorted(path.iterdir())
            if child.is_dir() and (child / "metadata.yaml").exists()
        ]
        if not candidates:
            raise ValueError(f"no episode directories found under {path}")
        episode_dirs.extend(candidates)

    seen = set()
    specs: list[EpisodeSpec] = []
    for episode_dir in episode_dirs:
        episode_dir = episode_dir.resolve()
        if episode_dir in seen:
            continue
        seen.add(episode_dir)
        metadata = parse_metadata(episode_dir / "metadata.yaml")
        specs.append(EpisodeSpec(episode_dir=episode_dir, bag_uri=episode_bag_uri(episode_dir), metadata=metadata))
    return sorted(specs, key=lambda spec: spec.episode_dir.name)


def stamp_to_ns(msg: object) -> int:
    stamp = msg.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def joint_state_to_vector(msg: JointState, ros_unit: str) -> np.ndarray:
    position_by_name = dict(zip(msg.name, msg.position, strict=False))
    missing = [name for name in JOINT_ORDER if name not in position_by_name]
    if missing:
        raise ValueError(f"joint state is missing joints: {missing}")
    return np.asarray(
        [ros_to_lerobot_position(name, position_by_name[name], ros_unit) for name in JOINT_ORDER],
        dtype=np.float32,
    )


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


def resize_rgb_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_width, target_height = size
    if image.shape[0] == target_height and image.shape[1] == target_width:
        return image
    pil = Image.fromarray(image)
    resampling = getattr(Image, "Resampling", Image)
    resized = pil.resize((target_width, target_height), resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def get_message_series(messages: list[tuple[int, object]]) -> TopicSeries:
    messages = sorted(messages, key=lambda item: item[0])
    stamps = [stamp for stamp, _ in messages]
    payloads = [msg for _, msg in messages]
    return TopicSeries(stamps_ns=stamps, messages=payloads)


def nearest_message(series: TopicSeries, target_ns: int, max_gap_ns: int) -> object:
    stamps = series.stamps_ns
    if not stamps:
        raise ValueError("cannot sample from an empty series")

    index = bisect_left(stamps, target_ns)
    candidates: list[int] = []
    if index < len(stamps):
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)

    best_index = min(candidates, key=lambda idx: abs(stamps[idx] - target_ns))
    best_gap = abs(stamps[best_index] - target_ns)
    if best_gap > max_gap_ns:
        raise ValueError(f"nearest sample too far away: gap_ns={best_gap}, target_ns={target_ns}")
    return series.messages[best_index]


def read_bag_topics(bag_uri: Path, required_topics: set[str]) -> dict[str, TopicSeries]:
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag_uri), storage_id="mcap"),
        ConverterOptions(input_serialization_format="", output_serialization_format=""),
    )

    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    missing = sorted(required_topics - set(topic_types))
    if missing:
        raise ValueError(f"bag is missing required topics: {missing}")

    reader.set_filter(StorageFilter(topics=sorted(required_topics)))

    raw_messages: dict[str, list[tuple[int, object]]] = {topic: [] for topic in required_topics}
    while reader.has_next():
        topic, data, _timestamp = reader.read_next()
        if topic not in required_topics:
            continue
        msg_type = topic_types[topic]
        if msg_type == "sensor_msgs/msg/Image":
            msg = deserialize_message(data, ImageMsg)
        elif msg_type == "sensor_msgs/msg/JointState":
            msg = deserialize_message(data, JointState)
        else:
            continue
        raw_messages[topic].append((stamp_to_ns(msg), msg))

    return {topic: get_message_series(messages) for topic, messages in raw_messages.items()}


def infer_image_shape(series: TopicSeries) -> tuple[int, int, int]:
    if not series.messages:
        raise ValueError("empty image series")
    msg = series.messages[0]
    array = image_msg_to_rgb_array(msg)
    height, width, _channels = array.shape
    return (3, height, width)


def build_features(image_shapes: dict[str, tuple[int, int, int]]) -> dict[str, dict]:
    return {
        "observation.images.camera1": {
            "dtype": "image",
            "shape": image_shapes["camera1"],
            "names": ["height", "width", "channels"],
        },
        "observation.images.camera2": {
            "dtype": "image",
            "shape": image_shapes["camera2"],
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(JOINT_ORDER),),
            "names": list(JOINT_ORDER),
        },
        "action": {
            "dtype": "float32",
            "shape": (len(JOINT_ORDER),),
            "names": list(JOINT_ORDER),
        },
    }


def collect_episode_times(series_map: dict[str, TopicSeries]) -> tuple[int, int]:
    start_ns = max(series.stamps_ns[0] for series in series_map.values())
    end_ns = min(series.stamps_ns[-1] for series in series_map.values())
    if end_ns <= start_ns:
        raise ValueError("episode time range is empty")
    return start_ns, end_ns


def build_episode_buffer(
    dataset: LeRobotDataset,
    series_map: dict[str, TopicSeries],
    start_ns: int,
    end_ns: int,
    ros_unit: str,
    task: str,
    max_nearest_gap_s: float,
    temp_dir: Path,
) -> dict:
    step_ns = int(round(1_000_000_000 / dataset.fps))
    max_gap_ns = int(round(max_nearest_gap_s * 1_000_000_000))

    top_shape = dataset.features["observation.images.camera1"]["shape"]
    wrist_shape = dataset.features["observation.images.camera2"]["shape"]
    top_size = (top_shape[2], top_shape[1])
    wrist_size = (wrist_shape[2], wrist_shape[1])

    episode_buffer = dataset.create_episode_buffer()
    episode_buffer["task"] = []
    frame_index = 0

    wrist_series = series_map[WristTopic]
    top_series = series_map[TopTopic]
    state_series = series_map[StateTopic]
    action_series = series_map[ActionTopic]

    current_ns = start_ns
    while current_ns <= end_ns:
        wrist_msg = nearest_message(wrist_series, current_ns, max_gap_ns)
        top_msg = nearest_message(top_series, current_ns, max_gap_ns)
        state_msg = nearest_message(state_series, current_ns, max_gap_ns)
        action_msg = nearest_message(action_series, current_ns, max_gap_ns)

        wrist_img = resize_rgb_image(image_msg_to_rgb_array(wrist_msg), wrist_size)
        top_img = resize_rgb_image(image_msg_to_rgb_array(top_msg), top_size)
        wrist_path = temp_dir / "wrist" / f"frame_{frame_index:06d}.png"
        top_path = temp_dir / "top" / f"frame_{frame_index:06d}.png"
        wrist_path.parent.mkdir(parents=True, exist_ok=True)
        top_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(wrist_img).save(wrist_path)
        Image.fromarray(top_img).save(top_path)

        rel_ts_s = (current_ns - start_ns) / 1e9
        episode_buffer["timestamp"].append(rel_ts_s)
        episode_buffer["frame_index"].append(frame_index)
        episode_buffer["task"].append(task)
        episode_buffer["observation.images.camera2"].append(str(wrist_path))
        episode_buffer["observation.images.camera1"].append(str(top_path))
        episode_buffer["observation.state"].append(joint_state_to_vector(state_msg, ros_unit))
        episode_buffer["action"].append(joint_state_to_vector(action_msg, ros_unit))

        frame_index += 1
        current_ns += step_ns

    if frame_index == 0:
        raise ValueError("no frames were generated for this episode")

    episode_buffer["size"] = frame_index
    return episode_buffer


def ensure_clean_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if any(output_root.iterdir()) and not overwrite:
            raise FileExistsError(
                f"output directory already exists and is not empty: {output_root}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_root)


def convert_episode(
    dataset: LeRobotDataset,
    spec: EpisodeSpec,
    ros_unit: str,
    task_override: str,
    max_nearest_gap_s: float,
    include_failed: bool,
) -> bool:
    result = spec.metadata.get("result")
    if not include_failed and result != "success":
        print(f"skip {spec.episode_dir.name}: result={result}")
        return False

    task = task_override or str(spec.metadata.get("task") or "")
    if not task:
        raise ValueError(f"task is missing for episode: {spec.episode_dir}")

    series_map = read_bag_topics(spec.bag_uri, set(DEFAULT_REQUIRED_TOPICS))
    start_ns, end_ns = collect_episode_times(series_map)

    with tempfile.TemporaryDirectory(prefix=f"{spec.episode_dir.name}_", dir=dataset.root) as tmp:
        temp_dir = Path(tmp)
        episode_buffer = build_episode_buffer(
            dataset=dataset,
            series_map=series_map,
            start_ns=start_ns,
            end_ns=end_ns,
            ros_unit=ros_unit,
            task=task,
            max_nearest_gap_s=max_nearest_gap_s,
            temp_dir=temp_dir,
        )
        frame_count = episode_buffer["size"]
        dataset.save_episode(episode_buffer)

    print(
        f"converted {spec.episode_dir.name}: "
        f"frames={frame_count}, task={task!r}, result={result}"
    )
    return True


def main() -> int:
    global ActionTopic, DEFAULT_REQUIRED_TOPICS, StateTopic

    args = parse_args()
    StateTopic = args.state_topic or target_state_topic(args.target)
    ActionTopic = args.action_topic
    DEFAULT_REQUIRED_TOPICS = [WristTopic, TopTopic, StateTopic, ActionTopic]

    episode_specs = discover_episodes(args.inputs)
    if not episode_specs:
        raise ValueError("no episodes discovered")

    output_root = resolve_output_root(args.output_root) if args.output_root else resolve_output_root(
        str(default_output_root_from_episode_specs(episode_specs))
    )
    ensure_clean_output_root(output_root, args.overwrite)

    repo_id = args.repo_id or output_root.name
    task_name = args.task.strip()

    rclpy.init()
    dataset: LeRobotDataset | None = None
    try:
        image_shapes: dict[str, tuple[int, int, int]] = {}
        for spec in episode_specs:
            series_map = read_bag_topics(spec.bag_uri, set(DEFAULT_REQUIRED_TOPICS))
            if "camera2" not in image_shapes:
                image_shapes["camera2"] = infer_image_shape(series_map[WristTopic])
            if "camera1" not in image_shapes:
                image_shapes["camera1"] = infer_image_shape(series_map[TopTopic])
            break

        if len(image_shapes) != 2:
            raise RuntimeError("failed to infer image shapes from the first episode")

        features = build_features(image_shapes)
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=args.fps,
            features=features,
            root=output_root,
            robot_type="so101",
            use_videos=False,
        )

        converted = 0
        skipped = 0
        for spec in episode_specs:
            try:
                if convert_episode(
                    dataset=dataset,
                    spec=spec,
                    ros_unit=args.ros_unit,
                    task_override=task_name,
                    max_nearest_gap_s=args.max_nearest_gap_s,
                    include_failed=args.include_failed,
                ):
                    converted += 1
                else:
                    skipped += 1
            except Exception as exc:
                skipped += 1
                print(f"failed {spec.episode_dir.name}: {exc}")
                raise

        dataset.finalize()

        summary = {
            "repo_id": repo_id,
            "output_root": str(output_root),
            "fps": args.fps,
            "ros_unit": args.ros_unit,
            "converted_episodes": converted,
            "skipped_episodes": skipped,
            "episode_dirs": [str(spec.episode_dir) for spec in episode_specs],
        }
        (output_root / "conversion_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        if args.verify:
            verified = LeRobotDataset(repo_id=repo_id, root=output_root, download_videos=False)
            print(
                f"verified dataset: episodes={verified.num_episodes}, frames={verified.num_frames}, "
                f"features={list(verified.features)}"
            )

        if args.push_to_hub:
            dataset.push_to_hub()

        print(f"done: converted={converted}, skipped={skipped}, output={output_root}")
        return 0
    finally:
        if dataset is not None:
            with contextlib.suppress(Exception):
                dataset.finalize()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
