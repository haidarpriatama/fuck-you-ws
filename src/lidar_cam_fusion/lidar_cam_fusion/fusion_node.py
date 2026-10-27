#!/usr/bin/env python3
"""
Node Fusi Sensor Kamera + LiDAR Livox Mid-360 (Package ROS2 Humble)
===================================================================
Robot ABU Robocon 2027

Fitur Utama:
- Deteksi 2D target (kubus kardus merah 35x35x35 cm) dengan YOLO11 (Ultralytics) atau fallback HSV.
- Akumulasi titik LiDAR Livox (livox_ros_driver2/msg/CustomMsg) dengan buffer thread-safe.
- Proyeksi 3D ke 2D (cv2.projectPoints) dengan konvensi:
      p_cam = R_cl @ p_lidar + t_cl
- Seleksi titik di dalam bounding box yang diperkecil (shrink factor).
- 1D depth clustering (gap threshold) untuk mengisolasi kluster terdekat (permukaan kubus)
  dari objek/dinding latar belakang.
- Filter ukuran (min_pts, max_extent).
- Publikasi posisi 3D ke /fusion/object_point (PointStamped).
- Publikasi marker 3D kubus 35x35x35 cm ke /fusion/object_marker (visualization_msgs/msg/Marker) untuk RViz2.
- Publikasi pointcloud 3D cluster ke /fusion/cluster_cloud dan ROI ke /fusion/roi_cloud (PointCloud2) untuk RViz2.
- Publikasi gambar overlay visual ke /fusion/debug (sensor_msgs/msg/Image).
- Live preview langsung di layar dengan jendela OpenCV GUI (show_preview, default: False agar fokus di RViz).
- Transformasi koordinat opsional ke frame base_link via TF2 jika enable_tf diaktifkan.

Batasan:
- Akumulasi titik (window_s) mengasumsikan sensor diam/statis. Saat robot bergerak, diperlukan deskewing odometri.
- Model YOLO Ultralytics berlisensi AGPL-3.0.
"""

import collections
import math
import os
import sys
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ROS 2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Header

try:
    import sensor_msgs_py.point_cloud2 as pc2
    HAS_PC2 = True
except ImportError:
    HAS_PC2 = False

try:
    from cv_bridge import CvBridge
except ImportError:
    CvBridge = None

try:
    from livox_ros_driver2.msg import CustomMsg
except ImportError:
    CustomMsg = None

# TF2 imports
try:
    import tf2_ros
    import tf2_geometry_msgs
    HAS_TF2 = True
except ImportError:
    HAS_TF2 = False

# Ultralytics YOLO import
try:
    import torch
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


class LidarCamFusionNode(Node):
    def __init__(self):
        super().__init__('lidar_cam_fusion_node')

        # -------------------------------------------------------------
        # Deklarasi Parameter ROS 2
        # -------------------------------------------------------------
        self.declare_parameter('detector', 'yolo')
        self.declare_parameter('model', '/home/haidar/ros2_ws/models/best11n.pt')
        self.declare_parameter('target_class', 'R')
        self.declare_parameter('conf', 0.50)

        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('lidar_topic', '/livox/lidar')
        self.declare_parameter('cam_frame', 'camera_optical_frame')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('enable_tf', False)

        # Default: False agar tidak memunculkan window popup baru, visualisasi fokus di RViz
        self.declare_parameter('show_preview', False)

        # Waktu akumulasi titik (dinaikkan ke 0.45s agar kerapatan scan Livox Mid-360 cukup rapat)
        self.declare_parameter('window_s', 0.45)
        self.declare_parameter('time_offset', 0.0)

        # Filter Bbox & Clustering
        self.declare_parameter('shrink', 0.08)  # Penyusutan 8% dari tepi box
        self.declare_parameter('gap', 0.25)     # Gap clustering 25 cm
        self.declare_parameter('min_pts', 3)    # Toleransi titik minimum (default 3)
        self.declare_parameter('max_extent', 0.50)
        self.declare_parameter('min_range', 0.35)  # Blind zone Livox Mid-360 & sasis robot

        # Parameter Sudut Kemiringan LiDAR ke Bawah (Pitch Angle dalam derajat)
        # Positif (+) jika LiDAR dimiringkan menunduk ke bawah
        self.declare_parameter('lidar_pitch_deg', 0.0)

        # =============================================================
        # PARAMETER KALIBRASI
        # =============================================================
        # Matriks K kamera Logitech MX Brio 1280x720
        default_K = [
            775.1078, 0.0, 617.6829,
            0.0, 770.2547, 340.3842,
            0.0, 0.0, 1.0
        ]
        self.declare_parameter('K', default_K)

        default_D = [0.1026, -0.1739, -0.0117, -0.0025, 0.0]
        self.declare_parameter('D', default_D)

        # Rotasi Ekstrinsik Nominal (Kamera menghadap KANAN LiDAR):
        # Livox: X maju, Y kiri, Z atas
        # Kamera: X kanan (-X_l), Y bawah (-Z_l), Z maju (-Y_l)
        default_R_cl = [
            -1.0,  0.0,  0.0,
             0.0,  0.0, -1.0,
             0.0, -1.0,  0.0
        ]
        self.declare_parameter('R_cl', default_R_cl)

        # Translasi t_cl [tx, ty, tz] meter (posisi kamera relatif terhadap LiDAR)
        default_t_cl = [0.0, 0.02, 0.05]
        self.declare_parameter('t_cl', default_t_cl)

        # -------------------------------------------------------------
        # Baca Parameter
        # -------------------------------------------------------------
        self.load_parameters()

        # Inisialisasi CV Bridge
        if CvBridge is None:
            self.get_logger().error("cv_bridge tidak ditemukan!")
        self.bridge = CvBridge()

        # Inisialisasi Detektor
        self.init_detector()

        # Inisialisasi TF2 (Opsional)
        self.tf_buffer = None
        self.tf_listener = None
        if self.enable_tf:
            if HAS_TF2:
                self.tf_buffer = tf2_ros.Buffer()
                self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
                self.get_logger().info(f"TF2 diaktifkan -> target frame: {self.base_frame}")
            else:
                self.get_logger().warn("Pustaka tf2_ros tidak tersedia. TF dinonaktifkan.")
                self.enable_tf = False

        # Buffer Titik LiDAR (x, y, z, timestamp)
        self.lidar_lock = threading.Lock()
        self.lidar_buffer = collections.deque(maxlen=100000)

        # Publishers
        self.pub_object_point = self.create_publisher(PointStamped, '/fusion/object_point', 10)
        self.pub_marker = self.create_publisher(Marker, '/fusion/object_marker', 10)
        self.pub_debug = self.create_publisher(Image, '/fusion/debug', 10)
        self.pub_cluster_cloud = self.create_publisher(PointCloud2, '/fusion/cluster_cloud', 10)
        self.pub_roi_cloud = self.create_publisher(PointCloud2, '/fusion/roi_cloud', 10)

        # Subscribers
        lidar_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        if CustomMsg is not None:
            self.sub_lidar = self.create_subscription(
                CustomMsg,
                self.lidar_topic,
                self.lidar_callback,
                lidar_qos
            )
            self.get_logger().info(f"Subscribed LiDAR: {self.lidar_topic} (CustomMsg)")
        else:
            self.get_logger().error("livox_ros_driver2/msg/CustomMsg tidak dapat diimpor!")

        self.sub_image = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )
        self.get_logger().info(f"Subscribed Camera: {self.image_topic}")

        self.gui_warned = False

    def load_parameters(self):
        """Memuat parameter ROS dan menghitung matriks rotasi."""
        self.detector_type = self.get_parameter('detector').get_parameter_value().string_value.lower()
        self.model_path = self.get_parameter('model').get_parameter_value().string_value
        self.target_class = self.get_parameter('target_class').get_parameter_value().string_value
        self.conf = self.get_parameter('conf').get_parameter_value().double_value

        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.lidar_topic = self.get_parameter('lidar_topic').get_parameter_value().string_value
        self.cam_frame = self.get_parameter('cam_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.enable_tf = self.get_parameter('enable_tf').get_parameter_value().bool_value

        self.show_preview = self.get_parameter('show_preview').get_parameter_value().bool_value

        self.window_s = self.get_parameter('window_s').get_parameter_value().double_value
        self.time_offset = self.get_parameter('time_offset').get_parameter_value().double_value
        self.shrink = self.get_parameter('shrink').get_parameter_value().double_value
        self.gap = self.get_parameter('gap').get_parameter_value().double_value
        self.min_pts = self.get_parameter('min_pts').get_parameter_value().integer_value
        self.max_extent = self.get_parameter('max_extent').get_parameter_value().double_value
        self.min_range = self.get_parameter('min_range').get_parameter_value().double_value
        self.lidar_pitch_deg = self.get_parameter('lidar_pitch_deg').get_parameter_value().double_value

        # Matriks K (3x3)
        raw_K = self.get_parameter('K').get_parameter_value().double_array_value
        if len(raw_K) != 9:
            self.get_logger().error(f"Parameter K harus 9 elemen, terbaca {len(raw_K)}")
            raw_K = [775.0, 0.0, 640.0, 0.0, 770.0, 360.0, 0.0, 0.0, 1.0]
        self.K = np.array(raw_K, dtype=np.float64).reshape((3, 3))

        # Distorsi D
        raw_D = self.get_parameter('D').get_parameter_value().double_array_value
        self.D = np.array(raw_D, dtype=np.float64)

        # Rotasi R_cl (3x3)
        raw_R = self.get_parameter('R_cl').get_parameter_value().double_array_value
        if len(raw_R) != 9:
            self.get_logger().error(f"Parameter R_cl harus 9 elemen, terbaca {len(raw_R)}")
            raw_R = [-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0]
        R_mat = np.array(raw_R, dtype=np.float64).reshape((3, 3))

        # Jika ada penyesuaian sudut miring (pitch tilt) bracket:
        # Rotasi pitch terhadap sumbu X frame kamera (menggeser titik naik/turun vertikal pada gambar)
        if abs(self.lidar_pitch_deg) > 1e-4:
            th = math.radians(self.lidar_pitch_deg)
            R_pitch_cam = np.array([
                [1.0, 0.0,           0.0],
                [0.0, math.cos(th), -math.sin(th)],
                [0.0, math.sin(th),  math.cos(th)]
            ], dtype=np.float64)
            self.R_cl = R_pitch_cam @ R_mat
            self.get_logger().info(f"Kompensasi kemiringan bracket (pitch): {self.lidar_pitch_deg} derajat diterapkan.")
        else:
            self.R_cl = R_mat

        # Translasi t_cl (3x1)
        raw_t = self.get_parameter('t_cl').get_parameter_value().double_array_value
        if len(raw_t) != 3:
            self.get_logger().error(f"Parameter t_cl harus 3 elemen, terbaca {len(raw_t)}")
            raw_t = [0.0, 0.0, 0.0]
        self.t_cl = np.array(raw_t, dtype=np.float64).reshape((3, 1))

        self.get_logger().info(
            f"Konfigurasi aktif | Detektor: {self.detector_type} | Target: '{self.target_class}' | "
            f"Window: {self.window_s}s | Shrink: {self.shrink} | MinPts: {self.min_pts} | Pitch: {self.lidar_pitch_deg} deg"
        )

    def init_detector(self):
        """Inisialisasi YOLO atau fallback HSV."""
        self.yolo_model = None
        self.device = 'cpu'

        if self.detector_type == 'yolo':
            if not HAS_YOLO:
                self.get_logger().warn("Ultralytics YOLO tidak terpasang! Mengalihkan ke fallback HSV.")
                self.detector_type = 'hsv'
                return

            try:
                if torch.cuda.is_available():
                    self.device = 'cuda:0'
                    gpu_name = torch.cuda.get_device_name(0)
                    self.get_logger().info(f"GPU: {gpu_name} ({self.device}) aktif untuk inferensi YOLO.")
                else:
                    self.device = 'cpu'
                    self.get_logger().warn("GPU CUDA tidak ditemukan. Menggunakan CPU.")

                self.get_logger().info(f"Memuat model YOLO: {self.model_path}")
                self.yolo_model = YOLO(self.model_path)
                
                names = getattr(self.yolo_model, 'names', {})
                self.get_logger().info(f"Kelas model terdeteksi: {names}")

                # Warmup inference
                dummy = np.zeros((320, 320, 3), dtype=np.uint8)
                _ = self.yolo_model(dummy, device=self.device, verbose=False)
                self.get_logger().info("Warmup model inferensi selesai.")
            except Exception as e:
                self.get_logger().error(f"Gagal memuat YOLO ({e}). Beralih ke fallback HSV.")
                self.detector_type = 'hsv'

    def lidar_callback(self, msg: CustomMsg):
        """Buffer titik Livox CustomMsg thread-safe."""
        base_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pts_arr = []
        for p in msg.points:
            pt_time = base_sec + p.offset_time * 1e-9
            pts_arr.append((p.x, p.y, p.z, pt_time))

        if not pts_arr:
            return

        with self.lidar_lock:
            self.lidar_buffer.extend(pts_arr)
            cutoff = base_sec - max(self.window_s * 3.0, 1.5)
            while self.lidar_buffer and self.lidar_buffer[0][3] < cutoff:
                self.lidar_buffer.popleft()

    def image_callback(self, msg: Image):
        """Pipeline utama fusi sensor."""
        if self.bridge is None:
            return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Error konversi Image: {e}")
            return

        img_h, img_w = cv_img.shape[:2]
        img_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # 1. Deteksi 2D
        bboxes, latency_ms = self.detect_objects(cv_img)
        if latency_ms >= 0:
            self.get_logger().info(
                f"[{self.detector_type.upper()}] Latensi: {latency_ms:.2f} ms | Target terdeteksi: {len(bboxes)}",
                throttle_duration_sec=2.0
            )

        # 2. Ambil titik LiDAR dari buffer
        t_target = img_time + self.time_offset
        t_start = t_target - self.window_s
        pts_lidar = self.get_lidar_points_in_window(t_start, t_target)

        # 3. Proyeksi titik LiDAR ke bidang gambar
        pts_cam, proj_uv, valid_mask = self.project_lidar_to_camera(pts_lidar, img_w, img_h)

        # Publikasikan pointcloud ROI dalam frame kamera jika ada
        if HAS_PC2 and pts_cam.shape[0] > 0:
            self.publish_cloud(self.pub_roi_cloud, pts_cam, msg.header.stamp)

        debug_img = cv_img.copy()

        # Render titik LiDAR yang masuk FOV kamera (biru muda)
        if proj_uv is not None and len(proj_uv) > 0:
            for pt_uv in proj_uv:
                u, v = int(round(pt_uv[0])), int(round(pt_uv[1]))
                if 0 <= u < img_w and 0 <= v < img_h:
                    cv2.circle(debug_img, (u, v), 1, (220, 180, 100), -1)

        # 4. Filter titik & depth clustering dalam bounding box
        detected_objects_3d = []
        cluster_all_pts = []

        for bbox_data in bboxes:
            box = bbox_data['box']
            label = bbox_data['label']
            score = bbox_data['score']

            shrunk_box = self.compute_shrunk_box(box, self.shrink, img_w, img_h)

            x1, y1, x2, y2 = [int(v) for v in box]
            sx1, sy1, sx2, sy2 = [int(v) for v in shrunk_box]
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(debug_img, (sx1, sy1), (sx2, sy2), (0, 255, 255), 1)

            cluster_cam_pts, cluster_uv, raw_in_box_count = self.extract_cluster_in_box(
                pts_cam, proj_uv, shrunk_box, self.gap, self.min_pts, self.max_extent, self.min_range
            )

            if cluster_cam_pts is not None and len(cluster_cam_pts) >= self.min_pts:
                obj_x = float(np.median(cluster_cam_pts[:, 0]))
                obj_y = float(np.median(cluster_cam_pts[:, 1]))
                obj_z = float(np.median(cluster_cam_pts[:, 2]))
                dist = math.sqrt(obj_x**2 + obj_y**2 + obj_z**2)

                detected_objects_3d.append({
                    'pos_cam': (obj_x, obj_y, obj_z),
                    'distance': dist,
                    'pts_count': len(cluster_cam_pts),
                    'label': label,
                    'score': score
                })
                cluster_all_pts.append(cluster_cam_pts)

                # Highlight titik kluster objek (hijau terang)
                for c_uv in cluster_uv:
                    u, v = int(round(c_uv[0])), int(round(c_uv[1]))
                    if 0 <= u < img_w and 0 <= v < img_h:
                        cv2.circle(debug_img, (u, v), 3, (0, 255, 0), -1)

                info_text = f"{label} {score:.2f} | d={dist:.2f}m (x={obj_x:.2f}, y={obj_y:.2f}, z={obj_z:.2f}) [{len(cluster_cam_pts)}pts]"
                cv2.putText(debug_img, info_text, (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                info_text = f"{label} {score:.2f} | NO LIDAR CLUSTER ({raw_in_box_count} pts)"
                cv2.putText(debug_img, info_text, (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
                self.get_logger().info(
                    f"[{label}] Target terdeteksi di gambar ({raw_in_box_count} titik LiDAR valid dalam box), "
                    f"tetapi kluster tidak lolos kriteria fisik kubus (min_pts={self.min_pts}, min_range={self.min_range}m).",
                    throttle_duration_sec=3.0
                )

        # 5. Publikasikan target 3D & Marker RViz2
        if detected_objects_3d:
            detected_objects_3d.sort(key=lambda item: item['distance'])
            best_obj = detected_objects_3d[0]

            self.publish_object_point(
                best_obj['pos_cam'],
                msg.header.stamp,
                best_obj['distance'],
                best_obj['label']
            )
            self.publish_cube_marker(
                best_obj['pos_cam'],
                msg.header.stamp,
                best_obj['label']
            )

        if HAS_PC2 and cluster_all_pts:
            merged_cluster_pts = np.vstack(cluster_all_pts)
            self.publish_cloud(self.pub_cluster_cloud, merged_cluster_pts, msg.header.stamp)

        # 6. Publikasi topik gambar /fusion/debug
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
            debug_msg.header = msg.header
            self.pub_debug.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f"Gagal publish debug image: {e}")

        # 7. Live Preview OpenCV GUI jika diaktifkan (default: False)
        if self.show_preview:
            try:
                cv2.imshow("Fusion Preview - Kamera & LiDAR (Robocon 2027)", debug_img)
                cv2.waitKey(1)
            except Exception as e:
                if not self.gui_warned:
                    self.get_logger().warn(
                        f"Tidak dapat menampilkan jendela GUI OpenCV ({e}). "
                        "Jendela dinonaktifkan secara otomatis. Pantau visual melalui RViz2."
                    )
                    self.gui_warned = True

    def detect_objects(self, img_bgr: np.ndarray) -> Tuple[List[dict], float]:
        """Deteksi 2D dengan YOLO atau fallback HSV."""
        results = []
        latency_ms = -1.0

        if self.detector_type == 'yolo' and self.yolo_model is not None:
            t0 = time.perf_counter()
            pred = self.yolo_model(img_bgr, device=self.device, conf=self.conf, verbose=False)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            if pred and len(pred) > 0:
                boxes_obj = pred[0].boxes
                if boxes_obj is not None:
                    names = self.yolo_model.names
                    for box in boxes_obj:
                        cls_id = int(box.cls[0].item())
                        cls_name = names.get(cls_id, str(cls_id))
                        score = float(box.conf[0].item())

                        if self.target_class != "" and cls_name != self.target_class:
                            continue

                        xyxy = box.xyxy[0].cpu().numpy().tolist()
                        results.append({
                            'box': xyxy,
                            'label': cls_name,
                            'score': score
                        })
            return results, latency_ms

        # Fallback HSV
        t0 = time.perf_counter()
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 100, 50]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 100, 50]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 400:
                x, y, w, h = cv2.boundingRect(cnt)
                results.append({
                    'box': [float(x), float(y), float(x + w), float(y + h)],
                    'label': 'Red_HSV',
                    'score': 1.0
                })
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return results, latency_ms

    def get_lidar_points_in_window(self, t_start: float, t_end: float) -> np.ndarray:
        """Mengambil titik LiDAR dalam rentang waktu buffer."""
        with self.lidar_lock:
            if not self.lidar_buffer:
                return np.empty((0, 3), dtype=np.float64)

            valid_pts = [
                (pt[0], pt[1], pt[2])
                for pt in self.lidar_buffer
                if t_start <= pt[3] <= t_end
            ]

            if len(valid_pts) == 0 and len(self.lidar_buffer) > 0:
                latest_time = self.lidar_buffer[-1][3]
                valid_pts = [
                    (pt[0], pt[1], pt[2])
                    for pt in self.lidar_buffer
                    if pt[3] >= (latest_time - self.window_s)
                ]

        if not valid_pts:
            return np.empty((0, 3), dtype=np.float64)
        return np.array(valid_pts, dtype=np.float64)

    def project_lidar_to_camera(self, pts_lidar: np.ndarray, img_w: int, img_h: int):
        """Proyeksi titik LiDAR ke koordinat kamera dan piksel gambar."""
        if pts_lidar.shape[0] == 0:
            return np.empty((0, 3)), np.empty((0, 2)), np.zeros(0, dtype=bool)

        pts_cam = (self.R_cl @ pts_lidar.T + self.t_cl).T
        in_front = pts_cam[:, 2] > 0.1
        if not np.any(in_front):
            return np.empty((0, 3)), np.empty((0, 2)), np.zeros(0, dtype=bool)

        pts_cam_valid = pts_cam[in_front]
        rvec = np.zeros(3, dtype=np.float64)
        tvec = np.zeros(3, dtype=np.float64)
        proj, _ = cv2.projectPoints(pts_cam_valid, rvec, tvec, self.K, self.D)
        proj_uv = proj.reshape(-1, 2)
        return pts_cam_valid, proj_uv, in_front

    @staticmethod
    def compute_shrunk_box(box: List[float], shrink: float, img_w: int, img_h: int) -> List[float]:
        """Penyusutan bounding box 2D."""
        x1, y1, x2, y2 = box
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        sx1 = max(0.0, x1 + shrink * bw)
        sy1 = max(0.0, y1 + shrink * bh)
        sx2 = min(float(img_w - 1), x2 - shrink * bw)
        sy2 = min(float(img_h - 1), y2 - shrink * bh)
        return [sx1, sy1, sx2, sy2]

    @staticmethod
    def extract_cluster_in_box(
        pts_cam: np.ndarray,
        proj_uv: np.ndarray,
        shrunk_box: List[float],
        gap: float = 0.25,
        min_pts: int = 3,
        max_extent: float = 0.50,
        min_range: float = 0.35
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """
        Ekstraksi dan klasterisasi titik 3D dalam 2D bounding box berdasarkan metodologi:
        - Darshan Bhanushali (LiDAR-Camera Fusion for 3D Object Detection)
        - Jeremy Cohen / Fritz AI (LiDAR and Camera Fusion in Self-Driving Cars)
        
        Langkah:
        1. ROI BBox Filtering & Blind Spot Rejection (Z < min_range).
        2. 1D Depth Clustering dengan batas toleransi selisih jarak (gap).
        3. Eliminasi kluster noise (< min_pts).
        4. Seleksi kluster target foreground terdekat yang valid.
        5. Pemangkasan outlier lantai/dinding (statistical/median trimming) sesuai batas fisik kubus (max_extent).
        """
        if pts_cam.shape[0] == 0 or proj_uv.shape[0] == 0:
            return None, None, 0

        sx1, sy1, sx2, sy2 = shrunk_box
        inside_mask = (
            (proj_uv[:, 0] >= sx1) & (proj_uv[:, 0] <= sx2) &
            (proj_uv[:, 1] >= sy1) & (proj_uv[:, 1] <= sy2) &
            (pts_cam[:, 2] >= min_range)  # Filter blind zone / chassis noise
        )

        pts_in_box = pts_cam[inside_mask]
        uv_in_box = proj_uv[inside_mask]
        raw_count = int(pts_in_box.shape[0])

        if raw_count < min_pts:
            return None, None, raw_count

        depths = pts_in_box[:, 2]
        sorted_indices = np.argsort(depths)
        sorted_depths = depths[sorted_indices]

        # 1D Depth clustering
        clusters_idx = []
        current_cluster = [sorted_indices[0]]

        for i in range(1, len(sorted_depths)):
            if sorted_depths[i] - sorted_depths[i - 1] <= gap:
                current_cluster.append(sorted_indices[i])
            else:
                clusters_idx.append(current_cluster)
                current_cluster = [sorted_indices[i]]
        clusters_idx.append(current_cluster)

        # Filter hanya kluster yang memiliki titik >= min_pts (menolak noise sporadis)
        valid_clusters = [c for c in clusters_idx if len(c) >= min_pts]
        if not valid_clusters:
            return None, None, raw_count

        # Urutkan kluster valid berdasarkan jarak kedalaman median terdekat (foreground target)
        valid_clusters.sort(key=lambda c: np.median(depths[c]))
        best_cluster_indices = valid_clusters[0]

        selected_pts = pts_in_box[best_cluster_indices]
        selected_uv = uv_in_box[best_cluster_indices]

        # Pemangkasan outlier (Statistical / Median Trimming):
        # Mencegah kegagalan jika titik lantai/rumput bersambung dengan bagian bawah kubus
        med_x = float(np.median(selected_pts[:, 0]))
        med_y = float(np.median(selected_pts[:, 1]))
        med_z = float(np.median(selected_pts[:, 2]))

        half_extent = max(max_extent * 0.5, 0.20)
        inlier_mask = (
            (np.abs(selected_pts[:, 2] - med_z) <= half_extent) &
            (np.abs(selected_pts[:, 0] - med_x) <= half_extent) &
            (np.abs(selected_pts[:, 1] - med_y) <= half_extent)
        )

        inlier_pts = selected_pts[inlier_mask]
        inlier_uv = selected_uv[inlier_mask]

        if len(inlier_pts) < min_pts:
            return None, None, raw_count

        return inlier_pts, inlier_uv, raw_count

    def publish_object_point(self, pos_cam: Tuple[float, float, float], stamp, distance: float, label: str):
        """Publikasi PointStamped ke /fusion/object_point."""
        pt_msg = PointStamped()
        pt_msg.header.stamp = stamp
        pt_msg.header.frame_id = self.cam_frame
        pt_msg.point.x = float(pos_cam[0])
        pt_msg.point.y = float(pos_cam[1])
        pt_msg.point.z = float(pos_cam[2])

        if self.enable_tf and self.tf_buffer is not None:
            try:
                transformed_msg = self.tf_buffer.transform(
                    pt_msg,
                    self.base_frame,
                    timeout=rclpy.duration.Duration(seconds=0.05)
                )
                self.pub_object_point.publish(transformed_msg)
                self.get_logger().info(
                    f"[{label}] Jarak: {distance:.3f}m | Frame '{self.base_frame}': "
                    f"({transformed_msg.point.x:.3f}, {transformed_msg.point.y:.3f}, {transformed_msg.point.z:.3f})",
                    throttle_duration_sec=0.5
                )
                return
            except Exception as e:
                self.get_logger().warn(
                    f"TF gagal ({e}). Publikasi dalam '{self.cam_frame}'.",
                    throttle_duration_sec=3.0
                )

        self.pub_object_point.publish(pt_msg)
        self.get_logger().info(
            f"[{label}] Jarak: {distance:.3f}m | Frame '{self.cam_frame}': "
            f"({pos_cam[0]:.3f}, {pos_cam[1]:.3f}, {pos_cam[2]:.3f})",
            throttle_duration_sec=0.5
        )

    def publish_cube_marker(self, pos_cam: Tuple[float, float, float], stamp, label: str):
        """
        Publikasikan Marker 3D kubus 35x35x35 cm ke /fusion/object_marker untuk visualisasi di RViz2.
        """
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.cam_frame
        marker.ns = "detected_cube_35cm"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # Posisi pusat kubus
        marker.pose.position.x = float(pos_cam[0])
        marker.pose.position.y = float(pos_cam[1])
        marker.pose.position.z = float(pos_cam[2]) + 0.175
        marker.pose.orientation.w = 1.0

        # Dimensi fisik: 35x35x35 cm (0.35 x 0.35 x 0.35 m)
        marker.scale.x = 0.35
        marker.scale.y = 0.35
        marker.scale.z = 0.35

        # Warna merah semi-transparan
        marker.color.r = 1.0
        marker.color.g = 0.1
        marker.color.b = 0.1
        marker.color.a = 0.7

        marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
        self.pub_marker.publish(marker)

    def publish_cloud(self, publisher, pts_cam: np.ndarray, stamp):
        """Publikasi PointCloud2 dalam frame kamera optik."""
        if not HAS_PC2 or pts_cam.shape[0] == 0:
            return

        header = Header()
        header.stamp = stamp
        header.frame_id = self.cam_frame

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        pts_f32 = pts_cam[:, :3].astype(np.float32)
        cloud_msg = pc2.create_cloud(header, fields, pts_f32)
        publisher.publish(cloud_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LidarCamFusionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
