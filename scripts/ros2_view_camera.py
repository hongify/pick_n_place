#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


def import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for camera visualization. Install python3-opencv or opencv-python.") from exc
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


def image_msg_to_bgr(cv2, msg: Image):
    if msg.encoding.lower() not in {"rgb8", "bgr8"}:
        raise ValueError(f"unsupported encoding: {msg.encoding}")
    frame = np.frombuffer(msg.data, dtype=np.uint8)
    frame = frame.reshape((msg.height, msg.width, 3))
    if msg.encoding.lower() == "rgb8":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame


class CameraViewer(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("so101_camera_viewer")
        self.args = args
        self.cv2 = import_cv2()
        unavailable_reason = opencv_visualization_unavailable_reason(self.cv2)
        if unavailable_reason:
            raise RuntimeError(f"Camera visualization unavailable: {unavailable_reason}")

        self.shutdown_requested = threading.Event()
        self.latest = {}
        self.camera_names = []
        self.lock = threading.Lock()
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )


        self._subscriptions = []
        if args.wrist_topic:
            self.camera_names.append("wrist")
            self._subscriptions.append(
                self.create_subscription(Image, args.wrist_topic, lambda msg: self._on_image("wrist", msg), qos)
            )
        if args.top_topic:
            self.camera_names.append("top")
            self._subscriptions.append(
                self.create_subscription(Image, args.top_topic, lambda msg: self._on_image("top", msg), qos)
            )

        self.last_visualize = 0.0
        self.stale_black_seconds = max(1.0, 3.0 / max(self.args.fps, 1.0))

    def _on_image(self, name: str, msg: Image) -> None:
        with self.lock:
            self.latest[name] = (msg, time.monotonic())

    def _black_frame(self, height: int | None = None, width: int | None = None):
        height = height if height is not None else self.args.cell_height
        width = width if width is not None else self.args.cell_width
        return np.zeros((height, width, 3), dtype=np.uint8)

    def visualize(self) -> None:
        period_s = 1.0 / max(self.args.fps, 1.0)
        now = time.monotonic()
        if now - self.last_visualize < period_s:
            return

        with self.lock:
            frames = dict(self.latest)

        if self.args.mode == "windows":
            for name in self.camera_names:
                entry = frames.get(name)
                if entry is None:
                    frame = self._black_frame()
                else:
                    msg, stamp = entry
                    if now - stamp > self.stale_black_seconds:
                        frame = self._black_frame(msg.height, msg.width)
                    else:
                        frame = image_msg_to_bgr(self.cv2, msg)
                self.cv2.imshow(f"SO101 {name}", frame)
        else:
            tiles = []
            for name in self.camera_names:
                entry = frames.get(name)
                if entry is None:
                    frame = self._black_frame()
                else:
                    msg, stamp = entry
                    if now - stamp > self.stale_black_seconds:
                        frame = self._black_frame(self.args.cell_height, self.args.cell_width)
                    else:
                        frame = image_msg_to_bgr(self.cv2, msg)
                resized = self.cv2.resize(frame, (self.args.cell_width, self.args.cell_height))
                tiles.append(resized)
            if len(tiles) == 1:
                mosaic = tiles[0]
            else:
                mosaic = self.cv2.hconcat(tiles)
            self.cv2.imshow("SO101 cameras", mosaic)

        self.cv2.waitKey(1)
        self.last_visualize = now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SO-101 camera topics without owning the camera publish bridge.")
    parser.add_argument("--wrist-topic", default="/so101/camera/wrist/image_raw")
    parser.add_argument("--top-topic", default="/so101/camera/top/image_raw")
    parser.add_argument("--mode", choices=["windows", "mosaic"], default="windows")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--cell-width", type=int, default=480)
    parser.add_argument("--cell-height", type=int, default=360)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.cell_width <= 0 or args.cell_height <= 0:
        raise ValueError("--cell-width and --cell-height must be positive")

    rclpy.init()
    viewer = CameraViewer(args)

    def handle_signal(_signum, _frame):
        viewer.shutdown_requested.set()

    previous_sigint = signal.signal(signal.SIGINT, handle_signal)
    previous_sigterm = signal.signal(signal.SIGTERM, handle_signal)
    try:
        while rclpy.ok() and not viewer.shutdown_requested.is_set():
            rclpy.spin_once(viewer, timeout_sec=0.0)
            viewer.visualize()
            time.sleep(0.01)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        viewer.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
