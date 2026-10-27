#!/usr/bin/env python3
"""
Test Detector 4-Corner Kubus Kardus Merah (Tanpa YOLO / Deep Learning)
====================================================================
Robot ABU Robocon 2027

Deskripsi:
Program mandiri (standalone) untuk menguji deteksi 4 sudut (Top-Left, Top-Right,
Bottom-Right, Bottom-Left) dari kubus kardus merah 35x35x35 cm menggunakan
metode Classical Computer Vision (HSV Color Segmentation + Morphological Closing
untuk menangani logo putih + Polygon/Rotated Bounding Box Corner Extraction).

Fitur:
1. Bisa dijalankan langsung dari USB Webcam (/dev/video2) TANPA perlu ROS 2.
2. Bisa juga dijalankan via ROS 2 Topic (/image_raw) jika diinginkan.
3. Mendukung pengujian pada file gambar statis (--image path/to/image.png).
4. Window GUI interaktif dengan Slider / Trackbar HSV untuk kalibrasi instan di berbagai kondisi cahaya.
5. Deteksi 4 sudut presisi (TL, TR, BR, BL), titik tengah (Centroid), orientasi rotasi, dan dimensi piksel.

Cara Menjalankan:
  # 1. Menggunakan Kamera USB langsung (Logitech MX Brio):
  ~/yolo_venv/bin/python test_corner_detector.py --device /dev/video2

  # 2. Menggunakan ROS 2 Topic /image_raw:
  source /opt/ros/humble/setup.bash
  ~/yolo_venv/bin/python test_corner_detector.py --ros

  # 3. Menguji pada file gambar:
  ~/yolo_venv/bin/python test_corner_detector.py --image path/ke/gambar.jpg
"""

import argparse
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Optional ROS2 import
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False


def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    """
    Mengurutkan 4 titik sudut secara konsisten:
    Index 0: Top-Left (TL)
    Index 1: Top-Right (TR)
    Index 2: Bottom-Right (BR)
    Index 3: Bottom-Left (BL)
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL memiliki jumlah (x + y) terkecil
    rect[2] = pts[np.argmax(s)]  # BR memiliki jumlah (x + y) terbesar

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR memiliki selisih (y - x) terkecil
    rect[3] = pts[np.argmax(diff)]  # BL memiliki selisih (y - x) terbesar
    return rect


class BoxCornerDetector:
    def __init__(
        self,
        h_low1: int = 0,
        h_high1: int = 12,
        h_low2: int = 160,
        h_high2: int = 180,
        s_low: int = 80,
        s_high: int = 255,
        v_low: int = 50,
        v_high: int = 255,
        min_area: int = 2500,
        close_ksize: int = 25
    ):
        self.h_low1 = h_low1
        self.h_high1 = h_high1
        self.h_low2 = h_low2
        self.h_high2 = h_high2
        self.s_low = s_low
        self.s_high = s_high
        self.v_low = v_low
        self.v_high = v_high
        self.min_area = min_area
        self.close_ksize = close_ksize

    def detect(self, img_bgr: np.ndarray) -> Tuple[List[dict], np.ndarray]:
        """
        Mendeteksi kubus kardus merah dan mengekstrak 4 sudutnya.
        
        Returns:
            results: List of dict {
                'corners': np.ndarray (4, 2) [TL, TR, BR, BL],
                'center': (cx, cy),
                'box_xyxy': [x1, y1, x2, y2],
                'area': float,
                'width': float,
                'height': float,
                'angle': float
            }
            mask_clean: Masker biner hasil segmentasi warna merah setelah penutupan morfologi
        """
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # 1. Masking Merah 2-Range (merah berada di ujung 0-15 dan 160-180)
        mask1 = cv2.inRange(hsv, np.array([self.h_low1, self.s_low, self.v_low]),
                                 np.array([self.h_high1, self.s_high, self.v_high]))
        mask2 = cv2.inRange(hsv, np.array([self.h_low2, self.s_low, self.v_low]),
                                 np.array([self.h_high2, self.s_high, self.v_high]))
        mask = cv2.bitwise_or(mask1, mask2)

        # 2. Morfologi: Closing besar untuk menyambungkan permukaan kubus yang terhalang logo putih ("天")
        k_close = max(3, self.close_ksize | 1)  # pastikan ganjil
        kernel_c = cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close))
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_c)

        # Opening kecil untuk menghilangkan noise semut di latar belakang
        kernel_o = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_o)

        # 3. Cari Kontur
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            # A. Rotated Bounding Box (Oriented MinAreaRect)
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (bw, bh), angle = rect
            box_raw = cv2.boxPoints(rect)  # 4 sudut float32

            # B. Polygon Approximation dari Convex Hull (Quad fitting)
            hull = cv2.convexHull(cnt)
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.035 * peri, True)

            # Jika approx berhasil menghasilkan 4 sudut yang mendekati trapesium/persegi
            if len(approx) == 4:
                corners = approx.reshape(4, 2).astype(np.float32)
            else:
                corners = box_raw.astype(np.float32)

            # Urutkan 4 sudut [TL, TR, BR, BL]
            ordered_corners = order_points_clockwise(corners)

            # Bounding box sumbu lurus (xyxy)
            x_min = float(np.min(ordered_corners[:, 0]))
            x_max = float(np.max(ordered_corners[:, 0]))
            y_min = float(np.min(ordered_corners[:, 1]))
            y_max = float(np.max(ordered_corners[:, 1]))

            # Hitung lebar & tinggi bidang kardus
            edge_w = np.linalg.norm(ordered_corners[1] - ordered_corners[0])
            edge_h = np.linalg.norm(ordered_corners[3] - ordered_corners[0])

            results.append({
                'corners': ordered_corners,
                'center': (float(cx), float(cy)),
                'box_xyxy': [x_min, y_min, x_max, y_max],
                'area': float(area),
                'width': float(edge_w),
                'height': float(edge_h),
                'angle': float(angle)
            })

        # Urutkan dari kardus terbesar
        results.sort(key=lambda x: x['area'], reverse=True)
        return results, mask_clean


def draw_corner_visualizations(
    img_draw: np.ndarray,
    detections: List[dict],
    fps: float = 0.0,
    show_labels: bool = True
) -> np.ndarray:
    """Menggambar visualisasi sudut, garis tepi, centroid, dan teks koordinat."""
    # Warna sudut dalam BGR
    # TL: Cyan, TR: Kuning, BR: Magenta, BL: Hijau Terang
    corner_colors = [
        (255, 255, 0),    # 0: Top-Left (Cyan)
        (0, 255, 255),    # 1: Top-Right (Kuning)
        (255, 0, 255),    # 2: Bottom-Right (Magenta)
        (0, 255, 0)       # 3: Bottom-Left (Hijau)
    ]
    corner_names = ["TL", "TR", "BR", "BL"]

    for i, det in enumerate(detections):
        corners = det['corners'].astype(np.int32)
        cx, cy = int(det['center'][0]), int(det['center'][1])

        # 1. Garis poligon sisi kubus
        cv2.polylines(img_draw, [corners], isClosed=True, color=(0, 200, 255), thickness=2, lineType=cv2.LINE_AA)

        # 2. Garis silang diagonal untuk visualisasi perspektif
        cv2.line(img_draw, tuple(corners[0]), tuple(corners[2]), (80, 120, 255), 1, cv2.LINE_AA)
        cv2.line(img_draw, tuple(corners[1]), tuple(corners[3]), (80, 120, 255), 1, cv2.LINE_AA)

        # 3. Titik Centroid
        cv2.circle(img_draw, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(
            img_draw,
            f"Center ({cx}, {cy})",
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

        # 4. Gambar 4 Sudut (Corner Points)
        for idx in range(4):
            pt = tuple(corners[idx])
            col = corner_colors[idx]
            name = corner_names[idx]

            # Lingkaran luar + titik inti
            cv2.circle(img_draw, pt, 7, (0, 0, 0), -1)
            cv2.circle(img_draw, pt, 5, col, -1)

            if show_labels:
                lbl = f"{name}: ({pt[0]},{pt[1]})"
                # Offset teks agar tidak tumpang tindih
                off_x = 10 if idx in [1, 2] else -75
                off_y = -10 if idx in [0, 1] else 20
                cv2.putText(
                    img_draw,
                    lbl,
                    (pt[0] + off_x, pt[1] + off_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    col,
                    1,
                    cv2.LINE_AA
                )

        # Info Bounding Dimensi
        x1, y1, x2, y2 = [int(v) for v in det['box_xyxy']]
        info_header = f"Kardus #{i+1} | Area: {det['area']:.0f}px | W: {det['width']:.1f}px | H: {det['height']:.1f}px"
        cv2.putText(
            img_draw,
            info_header,
            (x1, max(y1 - 15, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    # Status FPS
    fps_text = f"FPS: {fps:.1f} | Objek Terdeteksi: {len(detections)}"
    cv2.putText(img_draw, fps_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    # Petunjuk tombol
    hint = "Tekan 'q'/'ESC': Keluar | 't': Toggle Sliders | 's': Simpan Foto"
    cv2.putText(img_draw, hint, (20, img_draw.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return img_draw


def run_standalone_camera(device: str, detector: BoxCornerDetector):
    """Menjalankan deteksi langsung dari USB Camera (/dev/video*)."""
    # Cek apakah device berupa integer atau string
    if device.isdigit():
        cap_arg = int(device)
    else:
        cap_arg = device

    print(f"Membuka kamera: {device}...")
    cap = cv2.VideoCapture(cap_arg, cv2.CAP_V4L2)

    # Set resolusi 1280x720 MJPG
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka kamera pada device: {device}!")
        print("Periksa index kamera dengan: v4l2-ctl --list-devices")
        return

    win_main = "Deteksi 4-Corner Kubus (ABU Robocon 2027)"
    win_ctrl = "Kalibrasi Slider HSV"
    cv2.namedWindow(win_main, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_main, 1024, 576)

    cv2.namedWindow(win_ctrl, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_ctrl, 400, 300)

    # Callback dummy untuk trackbar
    def nothing(x): pass

    # Trackbar untuk kalibrasi dinamis
    cv2.createTrackbar("H_Low1", win_ctrl, detector.h_low1, 180, nothing)
    cv2.createTrackbar("H_High1", win_ctrl, detector.h_high1, 180, nothing)
    cv2.createTrackbar("H_Low2", win_ctrl, detector.h_low2, 180, nothing)
    cv2.createTrackbar("H_High2", win_ctrl, detector.h_high2, 180, nothing)
    cv2.createTrackbar("S_Low", win_ctrl, detector.s_low, 255, nothing)
    cv2.createTrackbar("V_Low", win_ctrl, detector.v_low, 255, nothing)
    cv2.createTrackbar("MinArea/100", win_ctrl, int(detector.min_area / 100), 200, nothing)
    cv2.createTrackbar("CloseKernel", win_ctrl, detector.close_ksize, 51, nothing)

    t_prev = time.time()
    fps = 0.0

    print("Kamera aktif. Silakan arahkan kamera ke kubus kardus merah.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Gagal membaca frame dari kamera.")
            time.sleep(0.05)
            continue

        # Baca nilai terbaru dari trackbar
        detector.h_low1 = cv2.getTrackbarPos("H_Low1", win_ctrl)
        detector.h_high1 = cv2.getTrackbarPos("H_High1", win_ctrl)
        detector.h_low2 = cv2.getTrackbarPos("H_Low2", win_ctrl)
        detector.h_high2 = cv2.getTrackbarPos("H_High2", win_ctrl)
        detector.s_low = cv2.getTrackbarPos("S_Low", win_ctrl)
        detector.v_low = cv2.getTrackbarPos("V_Low", win_ctrl)
        detector.min_area = max(100, cv2.getTrackbarPos("MinArea/100", win_ctrl) * 100)
        detector.close_ksize = max(3, cv2.getTrackbarPos("CloseKernel", win_ctrl))

        # Deteksi
        t_now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-4, t_now - t_prev))
        t_prev = t_now

        dets, mask_clean = detector.detect(frame)
        view = frame.copy()
        draw_corner_visualizations(view, dets, fps=fps)

        # Picture-in-picture masker biner di pojok kanan atas
        mask_bgr = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)
        pip_h, pip_w = 180, 320
        mask_small = cv2.resize(mask_bgr, (pip_w, pip_h))
        view[10:10 + pip_h, view.shape[1] - pip_w - 10:view.shape[1] - 10] = mask_small
        cv2.rectangle(view, (view.shape[1] - pip_w - 10, 10), (view.shape[1] - 10, 10 + pip_h), (0, 255, 255), 1)
        cv2.putText(view, "Masker Biner (HSV)", (view.shape[1] - pip_w - 5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        cv2.imshow(win_main, view)
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            break
        elif key == ord('s'):
            filename = f"corner_capture_{int(time.time())}.jpg"
            cv2.imwrite(filename, view)
            print(f"Tangkapan layar tersimpan: {filename}")

    cap.release()
    cv2.destroyAllWindows()


def run_image_test(image_path: str, detector: BoxCornerDetector):
    """Menguji deteksi pada satu file gambar."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Gagal memuat file gambar: {image_path}")
        return

    t0 = time.perf_counter()
    dets, mask_clean = detector.detect(frame)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\nHasil Pengujian Sudut ({image_path}):")
    print(f"- Latensi Deteksi: {latency_ms:.2f} ms")
    print(f"- Objek Ditemukan: {len(dets)}")

    for i, d in enumerate(dets):
        print(f"\n[Kubus #{i+1}]")
        print(f"  Pusat (Centroid): ({d['center'][0]:.1f}, {d['center'][1]:.1f})")
        print(f"  Bounding Box: {d['box_xyxy']}")
        print(f"  Dimensi: Lebar={d['width']:.1f}px, Tinggi={d['height']:.1f}px | Rasio: {d['width']/max(1e-3, d['height']):.2f}")
        print("  4 Sudut Terurut (Clockwise):")
        labels = ["Top-Left (TL)", "Top-Right (TR)", "Bottom-Right (BR)", "Bottom-Left (BL)"]
        for lbl, pt in zip(labels, d['corners']):
            print(f"    - {lbl:16s}: ({pt[0]:.1f}, {pt[1]:.1f})")

    out_view = frame.copy()
    draw_corner_visualizations(out_view, dets, fps=1000.0 / max(1e-3, latency_ms))

    save_path = "corner_detection_result.jpg"
    cv2.imwrite(save_path, out_view)
    print(f"\nHasil visualisasi disimpan ke: {save_path}")

    # Tampilkan jika di desktop GUI
    try:
        cv2.imshow("Hasil Deteksi 4-Corner", out_view)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        pass


def run_ros2_node(topic: str, detector: BoxCornerDetector):
    """Menjalankan deteksi melalui ROS 2 subscription."""
    if not HAS_ROS2:
        print("[ERROR] ROS 2 (rclpy/sensor_msgs) tidak terdeteksi di environment ini!")
        return

    rclpy.init()
    node = Node('test_corner_detector_node')
    bridge = CvBridge()

    win_name = "ROS2 Corner Detector (/image_raw)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1024, 576)

    def img_callback(msg):
        try:
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            node.get_logger().error(f"CvBridge error: {e}")
            return

        dets, mask = detector.detect(cv_img)
        view = cv_img.copy()
        draw_corner_visualizations(view, dets)

        for d in dets:
            c = d['corners'].astype(int)
            node.get_logger().info(
                f"[Corner Detector] Kubus terdeteksi di ({d['center'][0]:.1f}, {d['center'][1]:.1f}) | "
                f"TL:{tuple(c[0])} TR:{tuple(c[1])} BR:{tuple(c[2])} BL:{tuple(c[3])}",
                throttle_duration_sec=1.0
            )

        cv2.imshow(win_name, view)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

    node.create_subscription(Image, topic, img_callback, 10)
    print(f"Subscribed ke topik ROS 2: {topic}. Menunggu data gambar...")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Detector 4-Corner Kubus Kardus Merah Robocon 2027")
    parser.add_argument("--device", type=str, default="/dev/video2",
                        help="Jalur device kamera (/dev/video2 atau index integer)")
    parser.add_argument("--image", type=str, default=None,
                        help="Uji coba pada file gambar statis")
    parser.add_argument("--ros", action="store_true",
                        help="Gunakan topik gambar ROS 2 (/image_raw)")
    parser.add_argument("--topic", type=str, default="/image_raw",
                        help="Topik ROS 2 jika --ros diaktifkan")

    args = parser.parse_args()

    # Inisialisasi detector dengan preset kardus merah
    detector = BoxCornerDetector(
        h_low1=0, h_high1=12,
        h_low2=160, h_high2=180,
        s_low=80, s_high=255,
        v_low=50, v_high=255,
        min_area=2500,
        close_ksize=25
    )

    if args.image is not None:
        run_image_test(args.image, detector)
    elif args.ros:
        run_ros2_node(args.topic, detector)
    else:
        run_standalone_camera(args.device, detector)


if __name__ == '__main__':
    main()
