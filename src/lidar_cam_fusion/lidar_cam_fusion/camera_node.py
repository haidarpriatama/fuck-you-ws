#!/usr/bin/env python3
"""
Node Pengambil Gambar Kamera (USB V4L2 / Logitech MX Brio)
==========================================================
Membaca streaming video dari perangkat /dev/video* dan mempublikasikannya
ke topik ROS 2 sensor_msgs/msg/Image.
"""

import time
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class UsbCameraNode(Node):
    def __init__(self):
        super().__init__('camera_publisher_node')

        # Parameter
        self.declare_parameter('video_device', '/dev/video2')  # Jalur atau index kamera (/dev/video2 atau 2)
        self.declare_parameter('image_width', 1280)
        self.declare_parameter('image_height', 720)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('frame_id', 'camera_optical_frame')

        raw_device = self.get_parameter('video_device').get_parameter_value().string_value
        # Konversi ke int jika string adalah digit murni (mis. "2")
        self.video_device = int(raw_device) if raw_device.isdigit() else raw_device

        self.width = self.get_parameter('image_width').get_parameter_value().integer_value
        self.height = self.get_parameter('image_height').get_parameter_value().integer_value
        self.fps = self.get_parameter('fps').get_parameter_value().double_value
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.pub_image = self.create_publisher(Image, self.image_topic, 10)

        # Inisialisasi VideoCapture dengan MJPG codec
        self.cap = None
        self.open_camera()

        timer_period = 1.0 / max(self.fps, 1.0)
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(
            f"Kamera aktif | Device: {self.video_device} | Resolusi: {self.width}x{self.height} @ {self.fps}fps | "
            f"Topik: {self.image_topic}"
        )

    def open_camera(self):
        """Membuka device kamera dan mengatur resolusi serta codec MJPG."""
        self.get_logger().info(f"Membuka kamera pada {self.video_device}...")
        self.cap = cv2.VideoCapture(self.video_device)
        if not self.cap.isOpened():
            self.get_logger().error(f"Gagal membuka kamera pada device: {self.video_device}!")
            return False

        # Set MJPG codec untuk framerate tinggi tanpa overhead USB bandwidth
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        return True

    def timer_callback(self):
        if not rclpy.ok():
            return
        if self.cap is None or not self.cap.isOpened():
            if not self.open_camera():
                return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.get_logger().warn("Gagal membaca frame kamera!", throttle_duration_sec=3.0)
            return

        if not rclpy.ok():
            return

        now = self.get_clock().now()
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = self.frame_id

        try:
            self.pub_image.publish(msg)
        except Exception:
            pass

    def destroy_node(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.get_logger().info("Device kamera dilepaskan.")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = UsbCameraNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
