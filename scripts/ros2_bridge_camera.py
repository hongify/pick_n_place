#!/usr/bin/env python3

import argparse
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image


@dataclass
class CameraSpec:
    name: str
    device: str
    image_topic: str
    camera_info_topic: str


@dataclass
class CameraStream:
    spec: CameraSpec
    capture: object
    image_pub: object
    info_pub: object
    frame_id: str
    lock: object = field(default_factory=threading.Lock)
    latest_frame: object | None = None
    latest_stamp_monotonic: float = 0.0
    last_read_monotonic: float = 0.0
    last_publish_monotonic: float = 0.0
    last_warn_monotonic: float = 0.0
    running: bool = True
    thread: object | None = None


def import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for camera publish. Install python3-opencv or opencv-python.") from exc
    return cv2


def opencv_visualization_unavailable_reason(cv2) -> str:
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return "no DISPLAY/WAYLAND_DISPLAY is set"

    try:
        build_info = cv2.getBuildInformation()
    except cv2.error:
        return ""

    for line in build_info.splitlines():
        key, separator, value = line.strip().partition(":")
        if separator and key == "GUI" and value.strip().upper() in {"", "NONE", "NO"}:
            return "OpenCV was built without HighGUI window support"
    return ""


def fourcc_to_string(value: float) -> str:
    value = int(value)
    return "".join(chr((value >> (8 * index)) & 0xFF) for index in range(4)).strip()


def configure_camera_capture(cv2, cap, width: int, height: int, fps: float, fourcc: str) -> None:
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)


def bgr_frame_to_image_msg(node: Node, frame, frame_id: str) -> Image:
    cv2 = import_cv2()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if not rgb.flags["C_CONTIGUOUS"]:
        rgb = rgb.copy()

    msg = Image()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = frame_id
    msg.height = int(rgb.shape[0])
    msg.width = int(rgb.shape[1])
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = int(rgb.shape[1] * 3)
    msg.data = rgb.tobytes()
    return msg


def image_msg_to_camera_info_msg(image_msg: Image) -> CameraInfo:
    msg = CameraInfo()
    msg.header = image_msg.header
    msg.height = image_msg.height
    msg.width = image_msg.width
    return msg


class CameraRos2Bridge(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("so101_camera_ros2_bridge")
        self.args = args
        self.camera_streams: list[CameraStream] = []
        self.shutdown_requested = threading.Event()
        self.visualization_enabled = args.visualize
        self.last_visualize_monotonic = 0.0

    def open_cameras(self) -> None:
        camera_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        specs = []
        if self.args.wrist_camera:
            specs.append(
                CameraSpec(
                    name="wrist",
                    device=self.args.wrist_camera,
                    image_topic=self.args.wrist_image_topic,
                    camera_info_topic=self.args.wrist_camera_info_topic,
                )
            )
        if self.args.top_camera:
            specs.append(
                CameraSpec(
                    name="top",
                    device=self.args.top_camera,
                    image_topic=self.args.top_image_topic,
                    camera_info_topic=self.args.top_camera_info_topic,
                )
            )

        if not specs:
            self.get_logger().info("No cameras selected.")
            return

        cv2 = import_cv2()
        if self.visualization_enabled:
            unavailable_reason = opencv_visualization_unavailable_reason(cv2)
            if unavailable_reason:
                self.get_logger().warning(f"Camera visualization disabled: {unavailable_reason}")
                self.visualization_enabled = False

        open_errors = []
        for spec in specs:
            capture = cv2.VideoCapture(spec.device, cv2.CAP_V4L2)
            if not capture.isOpened():
                open_errors.append(f"{spec.name}={spec.device}")
                self.get_logger().warning(f"Failed to open {spec.name} camera: {spec.device}")
                capture.release()
                continue
            configure_camera_capture(
                cv2,
                capture,
                self.args.camera_width,
                self.args.camera_height,
                self.args.camera_fps,
                self.args.camera_fourcc,
            )
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            stream = CameraStream(
                spec=spec,
                capture=capture,
                image_pub=self.create_publisher(Image, spec.image_topic, camera_qos),
                info_pub=self.create_publisher(CameraInfo, spec.camera_info_topic, camera_qos),
                frame_id=f"so101_camera_{spec.name}",
            )
            stream.thread = threading.Thread(target=self._camera_read_loop, args=(stream,), daemon=True)
            stream.thread.start()
            self.camera_streams.append(stream)
            actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
            actual_fourcc = fourcc_to_string(capture.get(cv2.CAP_PROP_FOURCC))
            self.get_logger().info(
                f"Publishing {spec.name} camera {spec.device} -> {spec.image_topic} "
                f"({actual_width}x{actual_height} fps={actual_fps:.1f} fourcc={actual_fourcc or '-'})"
            )

        if specs and not self.camera_streams:
            failed = ", ".join(open_errors)
            raise RuntimeError(f"Failed to open any selected camera: {failed}")

    def _camera_read_loop(self, stream: CameraStream) -> None:
        read_period_s = 1.0 / max(self.args.camera_fps, 1.0)
        while stream.running:
            now = time.monotonic()
            if now - stream.last_read_monotonic < read_period_s:
                time.sleep(min(read_period_s, 0.01))
                continue
            ok, frame = stream.capture.read()
            stream.last_read_monotonic = time.monotonic()
            if ok and frame is not None:
                with stream.lock:
                    stream.latest_frame = frame
                    stream.latest_stamp_monotonic = stream.last_read_monotonic
            else:
                time.sleep(0.05)

    def publish_cameras(self) -> None:
        publish_period_s = 1.0 / max(self.args.camera_fps, 1.0)
        for stream in self.camera_streams:
            now = time.monotonic()
            if now - stream.last_publish_monotonic < publish_period_s:
                continue
            with stream.lock:
                frame = None if stream.latest_frame is None else stream.latest_frame.copy()
                frame_age = time.monotonic() - stream.latest_stamp_monotonic
            if frame is None:
                if now - stream.last_warn_monotonic > 2.0:
                    self.get_logger().warning(f"No frame yet from {stream.spec.name} camera {stream.spec.device}")
                    stream.last_warn_monotonic = now
                continue
            if frame_age > 1.0:
                if now - stream.last_warn_monotonic > 2.0:
                    self.get_logger().warning(
                        f"Using stale frame from {stream.spec.name} camera {stream.spec.device}: {frame_age:.1f}s old"
                    )
                    stream.last_warn_monotonic = now
                continue
            image_msg = bgr_frame_to_image_msg(self, frame, stream.frame_id)
            stream.image_pub.publish(image_msg)
            stream.info_pub.publish(image_msg_to_camera_info_msg(image_msg))
            stream.last_publish_monotonic = now

    def visualize_cameras(self) -> None:
        if not self.visualization_enabled:
            return

        visualize_period_s = 1.0 / max(self.args.visualize_fps, 1.0)
        now = time.monotonic()
        if now - self.last_visualize_monotonic < visualize_period_s:
            return

        cv2 = import_cv2()
        try:
            if self.args.visualize_mode == "windows":
                open_windows = 0
                displayed_frames = 0
                for stream in self.camera_streams:
                    with stream.lock:
                        frame = None if stream.latest_frame is None else stream.latest_frame.copy()
                    if frame is None:
                        continue
                    displayed_frames += 1
                    window_name = camera_window_name(stream)
                    cv2.imshow(window_name, frame)
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
                        open_windows += 1
                if displayed_frames == 0:
                    return
            else:
                frames = []
                labels = []
                for stream in self.camera_streams:
                    with stream.lock:
                        frame = None if stream.latest_frame is None else stream.latest_frame.copy()
                    frames.append(frame)
                    labels.append(f"{stream.spec.name}: {stream.spec.device}")

                if not frames:
                    return

                mosaic = build_camera_mosaic(
                    cv2,
                    frames,
                    labels,
                    self.args.visualize_cell_width,
                    self.args.visualize_cell_height,
                )
                cv2.imshow("SO101 cameras", mosaic)
                open_windows = 1 if cv2.getWindowProperty("SO101 cameras", cv2.WND_PROP_VISIBLE) >= 1 else 0
            key = cv2.waitKey(1) & 0xFF
        except cv2.error as exc:
            self.get_logger().warning(f"Camera visualization disabled: {exc}")
            self.visualization_enabled = False
            return
        if key in {27, ord("q")}:
            self.get_logger().info("Camera visualization closed by user.")
            self.visualization_enabled = False
            cv2.destroyAllWindows()
        elif open_windows == 0:
            self.get_logger().info("Camera visualization windows closed.")
            self.visualization_enabled = False
        self.last_visualize_monotonic = now

    def close_cameras(self) -> None:
        for stream in self.camera_streams:
            stream.running = False
        for stream in self.camera_streams:
            if stream.thread is not None:
                stream.thread.join(timeout=1.0)
            stream.capture.release()
        self.camera_streams.clear()
        if self.visualization_enabled:
            cv2 = import_cv2()
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


def camera_window_name(stream: CameraStream) -> str:
    return f"SO101 {stream.spec.name} camera {stream.spec.device}"


def fit_frame_to_cell(cv2, frame, width: int, height: int):
    if frame is None:
        return np.zeros((height, width, 3), dtype=np.uint8)

    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, int(source_width * scale))
    resized_height = max(1, int(source_height * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    cell = np.zeros((height, width, 3), dtype=np.uint8)
    y = (height - resized_height) // 2
    x = (width - resized_width) // 2
    cell[y : y + resized_height, x : x + resized_width] = resized
    return cell


def build_camera_mosaic(cv2, frames: list, labels: list[str], cell_width: int, cell_height: int):
    cells = []
    for frame, label in zip(frames, labels, strict=False):
        cell = fit_frame_to_cell(cv2, frame, cell_width, cell_height)
        cv2.putText(
            cell,
            label,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cells.append(cell)

    if len(cells) == 1:
        return cells[0]

    cols = 2
    rows = []
    for start in range(0, len(cells), cols):
        row_cells = cells[start : start + cols]
        while len(row_cells) < cols:
            row_cells.append(np.zeros((cell_height, cell_width, 3), dtype=np.uint8))
        rows.append(cv2.hconcat(row_cells))
    return cv2.vconcat(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish SO101 USB cameras to ROS2 image topics.")
    parser.add_argument("--wrist-camera", default="")
    parser.add_argument("--top-camera", default="")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=10.0)
    parser.add_argument("--camera-fourcc", default="MJPG")
    parser.add_argument("--wrist-image-topic", default="/so101/camera/wrist/image_raw")
    parser.add_argument("--top-image-topic", default="/so101/camera/top/image_raw")
    parser.add_argument("--wrist-camera-info-topic", default="/so101/camera/wrist/camera_info")
    parser.add_argument("--top-camera-info-topic", default="/so101/camera/top/camera_info")
    parser.add_argument("--visualize", action="store_true", help="Show selected cameras in OpenCV windows.")
    parser.add_argument("--visualize-mode", choices=["windows", "mosaic"], default="windows")
    parser.add_argument("--visualize-fps", type=float, default=10.0)
    parser.add_argument("--visualize-cell-width", type=int, default=480)
    parser.add_argument("--visualize-cell-height", type=int, default=360)
    args = parser.parse_args()

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
    return args


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = CameraRos2Bridge(args)

    def request_shutdown(signum, _frame):
        node.get_logger().info(f"Received signal {signum}; stopping camera bridge.")
        node.shutdown_requested.set()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    try:
        node.open_cameras()
        while rclpy.ok() and not node.shutdown_requested.is_set():
            node.publish_cameras()
            node.visualize_cameras()
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(0.005)
    finally:
        node.close_cameras()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
