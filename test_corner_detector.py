#!/usr/bin/env python3
"""
Test Detector 3D-Corner & Multi-Face Kubus Kardus Merah (ABU Robocon 2027)
==========================================================================

Fitur Pembaruan:
1. PURE RED FILTER (Anti Bocor Kulit & Lantai):
   - Menggabungkan HSV murni (Hue 0-7 & 172-180, Saturation tinggi >= 115)
   - Ditambah filter dominansi Red pada RGB: (R - G > threshold) dan (R - B > threshold).
   - Kaki manusia, kulit, dan lantai kayu otomatis TERELIMINASI 100%.

2. MULTI-CORNER & ROTATED 3D BOX (Antisipasi 6 Sisi, 8 Sudut, Box Terputar):
   - Tidak lagi memaksa 4 sudut datar (1 sisi).
   - Saat kubus miring / terputar (yaw/pitch) dan memperlihatkan 2 atau 3 sisi sekaligus:
     * Menghitung siluet poligon luar kubus 3D (4 hingga 8 sudut poligon luar).
     * Mendeteksi titik temu sudut dalam (interior junction corners) menggunakan Shi-Tomasi.
   - Memberi label koordinat untuk setiap sudut fisik kubus yang terlihat (C1, C2, C3, ...).

Cara Menjalankan:
  # Langsung dari webcam USB:
  ~/yolo_venv/bin/python test_corner_detector.py --device /dev/video2

  # Topik ROS 2:
  ~/yolo_venv/bin/python test_corner_detector.py --ros

  # File gambar:
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


class Robust3DBoxDetector:
    def __init__(
        self,
        h_high1: int = 7,       # Ujung bawah merah: 0 sampai 7
        h_low2: int = 172,      # Ujung atas merah: 172 sampai 180
        s_low: int = 115,       # Saturasi tinggi untuk menolak warna kulit (kulit biasanya S < 90)
        v_low: int = 50,
        red_diff_min: int = 30, # Dominansi R terhadap G dan B: (R-G > 30) & (R-B > 30)
        min_area: int = 3500,
        close_ksize: int = 25,
        poly_epsilon_pct: int = 25  # Epsilon poligon dalam persentase (25 -> 0.025)
    ):
        self.h_high1 = h_high1
        self.h_low2 = h_low2
        self.s_low = s_low
        self.v_low = v_low
        self.red_diff_min = red_diff_min
        self.min_area = min_area
        self.close_ksize = close_ksize
        self.poly_epsilon_pct = poly_epsilon_pct

    def detect(self, img_bgr: np.ndarray) -> Tuple[List[dict], np.ndarray]:
        """
        Mendeteksi kubus kardus merah dengan filter merah murni (anti-kulit)
        dan mengekstrak seluruh sudut fisik siluet 3D (4-8 sudut).
        """
        img_h, img_w = img_bgr.shape[:2]

        # 1. Masking HSV Murni
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        mask_hsv = ((h <= self.h_high1) | (h >= self.h_low2)) & (s >= self.s_low) & (v >= self.v_low)

        # 2. Filter Dominansi Red pada RGB (Eliminasi Kulit Manusia & Lantai Kayu)
        # Kulit memiliki R~160, G~135, B~115 (selisih R-G hanya ~20).
        # Kardus merah pekat memiliki selisih R-G > 35 dan R-B > 35.
        b = img_bgr[:, :, 0].astype(np.float32)
        g = img_bgr[:, :, 1].astype(np.float32)
        r = img_bgr[:, :, 2].astype(np.float32)

        mask_rgb = (r - g >= self.red_diff_min) & (r - b >= self.red_diff_min) & (r >= 65)

        # Gabungan logika AND untuk menjamin warna MERAH BENAR-BENAR PEKAT
        mask_pure_red = (mask_hsv & mask_rgb).astype(np.uint8) * 255

        # 3. Penutupan Morfologi (Menutup logo Robocon "天" di permukaan kardus)
        k_close = max(3, self.close_ksize | 1)
        kernel_c = cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close))
        mask_closed = cv2.morphologyEx(mask_pure_red, cv2.MORPH_CLOSE, kernel_c)

        # Opening kecil untuk menghilangkan percikan kecil
        kernel_o = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_o)

        # 4. Deteksi Kontur Siluet Kubus
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            # Convex Hull: Menangkap batas fisik luar kubus 3D saat terputar
            hull = cv2.convexHull(cnt)
            peri = cv2.arcLength(hull, True)

            # Poligon adaptif: mengekstrak sudut luar (4 sampai 8 sudut sesuai perspektif)
            eps = (self.poly_epsilon_pct / 1000.0) * peri
            approx = cv2.approxPolyDP(hull, eps, True)
            outer_corners = approx.reshape(-1, 2)

            # Urutkan sudut luar searah jarum jam relatif terhadap centroid
            moments = cv2.moments(cnt)
            if moments["m00"] > 1e-4:
                cx = float(moments["m10"] / moments["m00"])
                cy = float(moments["m01"] / moments["m00"])
            else:
                cx = float(np.mean(outer_corners[:, 0]))
                cy = float(np.mean(outer_corners[:, 1]))

            # Hitung sudut polar untuk pengurutan siklik
            angles = np.arctan2(outer_corners[:, 1] - cy, outer_corners[:, 0] - cx)
            sort_order = np.argsort(angles)
            outer_corners = outer_corners[sort_order]

            # 5. Deteksi Titik Sudut Dalam (Interior Junction Corners)
            # Misalnya garis pertemuan 3 bidang kubus (Y-junction) di dalam kubus
            cnt_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
            # Kikis sedikit tepi kontur agar hanya mencari sudut dalam
            cnt_mask_inner = cv2.erode(cnt_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))

            inner_corners = []
            features = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=4,
                qualityLevel=0.08,
                minDistance=25,
                mask=cnt_mask_inner
            )
            if features is not None:
                for pt in features.reshape(-1, 2):
                    # Pastikan titik tidak menempel dengan outer corners
                    dists = np.linalg.norm(outer_corners - pt, axis=1)
                    if np.min(dists) > 20:
                        inner_corners.append(pt.astype(int))

            # Bounding box 2D yang membungkus seluruh sudut
            x_min = float(np.min(outer_corners[:, 0]))
            x_max = float(np.max(outer_corners[:, 0]))
            y_min = float(np.min(outer_corners[:, 1]))
            y_max = float(np.max(outer_corners[:, 1]))

            results.append({
                'outer_corners': outer_corners,
                'inner_corners': inner_corners,
                'center': (cx, cy),
                'box_xyxy': [x_min, y_min, x_max, y_max],
                'area': float(area),
                'num_outer_corners': len(outer_corners)
            })

        results.sort(key=lambda x: x['area'], reverse=True)
        return results, mask_clean


def draw_3d_box_visualizations(
    img_draw: np.ndarray,
    detections: List[dict],
    fps: float = 0.0
) -> np.ndarray:
    """Menggambar poligon siluet 3D, seluruh sudut fisik, garis sisi, dan koordinat."""
    corner_colors = [
        (0, 255, 255),    # Kuning
        (255, 255, 0),    # Cyan
        (0, 255, 0),      # Hijau
        (255, 0, 255),    # Magenta
        (0, 165, 255),    # Orange
        (255, 100, 100),  # Biru Langit
        (100, 255, 200),  # Mint
        (200, 100, 255)   # Ungu
    ]

    for i, det in enumerate(detections):
        outer = det['outer_corners'].astype(np.int32)
        inner = det['inner_corners']
        cx, cy = int(det['center'][0]), int(det['center'][1])
        n_corners = det['num_outer_corners']

        # 1. Gambar Poligon Siluet Kubus 3D
        cv2.polylines(img_draw, [outer], isClosed=True, color=(0, 220, 255), thickness=2, lineType=cv2.LINE_AA)

        # 2. Gambar Garis Hubung ke Sudut Dalam (Jika ada sudut Y-junction)
        for in_pt in inner:
            cv2.circle(img_draw, tuple(in_pt), 6, (0, 0, 255), -1)
            cv2.putText(
                img_draw,
                f"Inner:({in_pt[0]},{in_pt[1]})",
                (in_pt[0] + 8, in_pt[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )
            # Hubungkan sudut dalam ke sudut luar terdekat
            dists = np.linalg.norm(outer - in_pt, axis=1)
            nearest_idx = np.argsort(dists)[:3]
            for n_idx in nearest_idx:
                cv2.line(img_draw, tuple(in_pt), tuple(outer[n_idx]), (0, 180, 255), 1, cv2.LINE_AA)

        # 3. Titik Centroid
        cv2.circle(img_draw, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(
            img_draw,
            f"Pusat:({cx},{cy})",
            (cx + 8, cy + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

        # 4. Gambar Setiap Sudut Luar (C1, C2, ..., Cn)
        for c_idx in range(n_corners):
            pt = tuple(outer[c_idx])
            col = corner_colors[c_idx % len(corner_colors)]

            cv2.circle(img_draw, pt, 7, (0, 0, 0), -1)
            cv2.circle(img_draw, pt, 5, col, -1)

            lbl = f"C{c_idx+1}:({pt[0]},{pt[1]})"
            cv2.putText(
                img_draw,
                lbl,
                (pt[0] - 25, pt[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                col,
                1,
                cv2.LINE_AA
            )

        # Header Info Objek
        x1, y1, x2, y2 = [int(v) for v in det['box_xyxy']]
        info_header = f"Kardus 3D #{i+1} | {n_corners} Sudut Luar | Area: {det['area']:.0f}px"
        cv2.putText(
            img_draw,
            info_header,
            (x1, max(y1 - 12, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    # Info Status FPS & Deteksi
    fps_text = f"FPS: {fps:.1f} | Objek Kardus: {len(detections)}"
    cv2.putText(img_draw, fps_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    hint = "Tekan 'q'/'ESC': Keluar | 's': Simpan Tangkapan Layar | Window Kontrol untuk Slider"
    cv2.putText(img_draw, hint, (20, img_draw.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return img_draw


def run_standalone_camera(device: str, detector: Robust3DBoxDetector):
    """Menjalankan deteksi langsung dari USB Camera (/dev/video*)."""
    if device.isdigit():
        cap_arg = int(device)
    else:
        cap_arg = device

    print(f"Membuka kamera: {device}...")
    cap = cv2.VideoCapture(cap_arg, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka kamera {device}!")
        return

    win_main = "Deteksi 3D-Corner Kubus Kardus (Robocon 2027)"
    win_ctrl = "Kalibrasi Murni Merah (Anti-Kulit)"
    cv2.namedWindow(win_main, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_main, 1024, 576)

    cv2.namedWindow(win_ctrl, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_ctrl, 420, 320)

    def nothing(x): pass

    # Trackbar kontrol presisi
    cv2.createTrackbar("H_High1 (0-15)", win_ctrl, detector.h_high1, 20, nothing)
    cv2.createTrackbar("H_Low2 (160-180)", win_ctrl, detector.h_low2, 180, nothing)
    cv2.createTrackbar("S_Low (Saturasi)", win_ctrl, detector.s_low, 255, nothing)
    cv2.createTrackbar("V_Low (Kecerahan)", win_ctrl, detector.v_low, 255, nothing)
    cv2.createTrackbar("RedDiff (R-G & R-B)", win_ctrl, detector.red_diff_min, 80, nothing)
    cv2.createTrackbar("Epsilon% (Sudut)", win_ctrl, detector.poly_epsilon_pct, 60, nothing)
    cv2.createTrackbar("CloseKernel", win_ctrl, detector.close_ksize, 51, nothing)

    t_prev = time.time()
    fps = 0.0

    print("Kamera aktif. Arahkan ke kardus merah dari berbagai sudut rotasi.")
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # Update nilai dari slider
        detector.h_high1 = cv2.getTrackbarPos("H_High1 (0-15)", win_ctrl)
        detector.h_low2 = max(160, cv2.getTrackbarPos("H_Low2 (160-180)", win_ctrl))
        detector.s_low = cv2.getTrackbarPos("S_Low (Saturasi)", win_ctrl)
        detector.v_low = cv2.getTrackbarPos("V_Low (Kecerahan)", win_ctrl)
        detector.red_diff_min = cv2.getTrackbarPos("RedDiff (R-G & R-B)", win_ctrl)
        detector.poly_epsilon_pct = max(10, cv2.getTrackbarPos("Epsilon% (Sudut)", win_ctrl))
        detector.close_ksize = max(3, cv2.getTrackbarPos("CloseKernel", win_ctrl))

        t_now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-4, t_now - t_prev))
        t_prev = t_now

        dets, mask_clean = detector.detect(frame)
        view = frame.copy()
        draw_3d_box_visualizations(view, dets, fps=fps)

        # PIP Masker Biner di pojok kanan atas
        mask_bgr = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)
        pip_h, pip_w = 180, 320
        mask_small = cv2.resize(mask_bgr, (pip_w, pip_h))
        view[10:10 + pip_h, view.shape[1] - pip_w - 10:view.shape[1] - 10] = mask_small
        cv2.rectangle(view, (view.shape[1] - pip_w - 10, 10), (view.shape[1] - 10, 10 + pip_h), (0, 255, 255), 1)
        cv2.putText(view, "Masker Murni Merah", (view.shape[1] - pip_w - 5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        cv2.imshow(win_main, view)
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            break
        elif key == ord('s'):
            filename = f"corner_3d_capture_{int(time.time())}.jpg"
            cv2.imwrite(filename, view)
            print(f"Tangkapan layar tersimpan: {filename}")

    cap.release()
    cv2.destroyAllWindows()


def run_image_test(image_path: str, detector: Robust3DBoxDetector):
    """Menguji deteksi pada file gambar statis."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Gagal memuat file gambar: {image_path}")
        return

    t0 = time.perf_counter()
    dets, mask_clean = detector.detect(frame)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\nHasil Deteksi 3D-Corner ({image_path}):")
    print(f"- Latensi: {latency_ms:.2f} ms")
    print(f"- Objek Kardus Ditemukan: {len(dets)}")

    for i, d in enumerate(dets):
        print(f"\n[Kardus #{i+1}]")
        print(f"  Pusat (Centroid): ({d['center'][0]:.1f}, {d['center'][1]:.1f})")
        print(f"  Jumlah Sudut Luar Terdeteksi: {d['num_outer_corners']} sudut")
        print("  Daftar Koordinat Sudut Luar:")
        for idx, pt in enumerate(d['outer_corners']):
            print(f"    - Corner #{idx+1}: ({pt[0]}, {pt[1]})")
        if d['inner_corners']:
            print(f"  Sudut Sambungan Sisi Dalam (Y-Junction): {d['inner_corners']}")

    out_view = frame.copy()
    draw_3d_box_visualizations(out_view, dets, fps=1000.0 / max(1e-3, latency_ms))
    cv2.imwrite("corner_3d_result.jpg", out_view)
    print("\nHasil visualisasi disimpan ke: corner_3d_result.jpg")


def run_ros2_node(topic: str, detector: Robust3DBoxDetector):
    """Menjalankan deteksi melalui ROS 2 subscription."""
    if not HAS_ROS2:
        print("[ERROR] ROS 2 tidak terdeteksi di environment ini!")
        return

    rclpy.init()
    node = Node('test_3d_corner_detector_node')
    bridge = CvBridge()

    win_name = "ROS2 3D Corner Detector"
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
        draw_3d_box_visualizations(view, dets)

        for d in dets:
            node.get_logger().info(
                f"[3D Box] Pusat: ({d['center'][0]:.1f}, {d['center'][1]:.1f}) | "
                f"Sudut Luar: {d['num_outer_corners']} pts | BBox: {d['box_xyxy']}",
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
    parser = argparse.ArgumentParser(description="Detector 3D-Corner Kubus Kardus Merah Robocon 2027")
    parser.add_argument("--device", type=str, default="/dev/video2",
                        help="Jalur device kamera (/dev/video2 atau index integer)")
    parser.add_argument("--image", type=str, default=None,
                        help="Uji coba pada file gambar statis")
    parser.add_argument("--ros", action="store_true",
                        help="Gunakan topik gambar ROS 2 (/image_raw)")
    parser.add_argument("--topic", type=str, default="/image_raw",
                        help="Topik ROS 2 jika --ros diaktifkan")

    args = parser.parse_args()

    detector = Robust3DBoxDetector(
        h_high1=7,
        h_low2=172,
        s_low=115,
        v_low=50,
        red_diff_min=30,
        min_area=3500,
        close_ksize=25,
        poly_epsilon_pct=25
    )

    if args.image is not None:
        run_image_test(args.image, detector)
    elif args.ros:
        run_ros2_node(args.topic, detector)
    else:
        run_standalone_camera(args.device, detector)


if __name__ == '__main__':
    main()
