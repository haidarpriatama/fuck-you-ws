#!/usr/bin/env python3
"""
Test Detector 3D-Corner & Multi-Box Separator Kubus Kardus 35x35x35 cm (Robocon 2027)
=====================================================================================

Fitur Utama:
1. DATA KALIBRASI ASLI (camera_brio.yaml):
   - Memuat matriks intrinsik K (fx=775.107, fy=770.254, cx=617.683, cy=340.384)
     dan koefisien distorsi D dari konfigurasi FAST-LIVO2.
   - Estimasi posisi 3D nyata (X, Y, Z) dalam meter dan jarak d (meter) menggunakan SolvePnP.

2. VERIFIKASI UKURAN FISIK 35x35x35 cm:
   - Model 3D kubus 35x35x35 cm dengan 8 titik sudut (vertices) dan 12 rusuk (edges).
   - Memproyeksikan kubus kawat 3D (3D Wireframe Bounding Box) langsung ke gambar 2D.
   - Memvalidasi bahwa objek yang dideteksi benar-benar berdimensi fisik ~35 cm (bukan objek kecil atau objek raksasa).

3. PEMISAHAN KASUS 2 KUBUS BERDEKATAN / MENEMPEL (Multi-Box Splitting):
   - Menggunakan ukuran kernel adaptif agar celah antar kardus tidak tertutup.
   - Deteksi rusuk sambungan internal (Canny / Vertical edge gradient).
   - Analisis rasio aspek (Aspect Ratio > 1.45) & Convexity Defects untuk memotong
     dan memisahkan 2 kardus yang menempel menjadi 2 kubus 35 cm independen.

4. PURE RED FILTER (Anti-Bocor Kulit & Lantai):
   - Dual-range HSV murni (Hue 0-7 & 172-180, Saturation >= 115) + Red Dominance RGB.

Cara Menjalankan:
  # 1. Dari Webcam Logitech MX Brio langsung:
  ~/yolo_venv/bin/python test_corner_detector.py --device /dev/video2

  # 2. Dari Topik ROS 2:
  ~/yolo_venv/bin/python test_corner_detector.py --ros

  # 3. Pengujian pada Gambar:
  ~/yolo_venv/bin/python test_corner_detector.py --image path/ke/gambar.jpg
"""

import argparse
import math
import os
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

# Optional ROS2 import
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False


def load_brio_calibration(yaml_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Memuat parameter kalibrasi kamera Logitech MX Brio dari camera_brio.yaml
    atau menggunakan nilai default nominal jika file tidak ditemukan.
    """
    default_K = np.array([
        [775.107823, 0.0, 617.682937],
        [0.0, 770.254747, 340.384244],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    default_D = np.array([0.102616, -0.173908, -0.011734, -0.002521, 0.0], dtype=np.float64)

    search_paths = [
        yaml_path,
        "/home/haidar/fast_ws/src/FAST-LIVO2/config/camera_brio.yaml",
        "/home/haidar/fast_ws/install/fast_livo/share/fast_livo/config/camera_brio.yaml"
    ]

    for p in search_paths:
        if p and os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    data = yaml.safe_load(f)
                params = data.get('/**/ros__parameters', data.get('ros__parameters', data))
                fx = float(params.get('cam_fx', 775.107823))
                fy = float(params.get('cam_fy', 770.254747))
                cx = float(params.get('cam_cx', 617.682937))
                cy = float(params.get('cam_cy', 340.384244))
                d0 = float(params.get('cam_d0', 0.102616))
                d1 = float(params.get('cam_d1', -0.173908))
                d2 = float(params.get('cam_d2', -0.011734))
                d3 = float(params.get('cam_d3', -0.002521))

                K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
                D = np.array([d0, d1, d2, d3, 0.0], dtype=np.float64)
                print(f"[Kalibrasi] Sukses memuat kalibrasi Brio dari: {p}")
                return K, D
            except Exception as e:
                print(f"[WARN] Gagal membaca file kalibrasi {p} ({e}). Memakai nominal.")

    print("[Kalibrasi] Menggunakan kalibrasi nominal bawaan Logitech MX Brio.")
    return default_K, default_D


class Cube35Detector:
    # Dimensi fisik kubus Robocon: 35 x 35 x 35 cm (0.35 x 0.35 x 0.35 m)
    CUBE_SIZE_M = 0.35
    HALF_SIZE = 0.35 / 2.0  # 0.175 m

    # 8 Titik sudut fisik 3D kubus (dalam meter) berpusat di (0, 0, 0)
    CUBE_3D_CORNERS = np.array([
        [-HALF_SIZE, -HALF_SIZE, -HALF_SIZE],  # 0: Depan Kiri-Atas
        [ HALF_SIZE, -HALF_SIZE, -HALF_SIZE],  # 1: Depan Kanan-Atas
        [ HALF_SIZE,  HALF_SIZE, -HALF_SIZE],  # 2: Depan Kanan-Bawah
        [-HALF_SIZE,  HALF_SIZE, -HALF_SIZE],  # 3: Depan Kiri-Bawah
        [-HALF_SIZE, -HALF_SIZE,  HALF_SIZE],  # 4: Belakang Kiri-Atas
        [ HALF_SIZE, -HALF_SIZE,  HALF_SIZE],  # 5: Belakang Kanan-Atas
        [ HALF_SIZE,  HALF_SIZE,  HALF_SIZE],  # 6: Belakang Kanan-Bawah
        [-HALF_SIZE,  HALF_SIZE,  HALF_SIZE]   # 7: Belakang Kiri-Bawah
    ], dtype=np.float64)

    # 12 Garis rusuk penghubung kubus
    CUBE_EDGES = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bidang depan
        (4, 5), (5, 6), (6, 7), (7, 4),  # Bidang belakang
        (0, 4), (1, 5), (2, 6), (3, 7)   # Rusuk penghubung depan-belakang
    ]

    # Titik planar sisi 35x35 cm untuk solvePnP
    FACE_3D_POINTS = np.array([
        [-HALF_SIZE, -HALF_SIZE, 0.0],
        [ HALF_SIZE, -HALF_SIZE, 0.0],
        [ HALF_SIZE,  HALF_SIZE, 0.0],
        [-HALF_SIZE,  HALF_SIZE, 0.0]
    ], dtype=np.float64)

    def __init__(
        self,
        K: np.ndarray,
        D: np.ndarray,
        h_high1: int = 7,
        h_low2: int = 172,
        s_low: int = 115,
        v_low: int = 45,
        red_diff_min: int = 30,
        min_area: int = 2500,
        close_ksize: int = 15,
        enable_multi_box_split: bool = True
    ):
        self.K = K
        self.D = D
        self.h_high1 = h_high1
        self.h_low2 = h_low2
        self.s_low = s_low
        self.v_low = v_low
        self.red_diff_min = red_diff_min
        self.min_area = min_area
        self.close_ksize = close_ksize
        self.enable_multi_box_split = enable_multi_box_split

    def split_adjacent_boxes(self, binary_mask: np.ndarray, gray_img: np.ndarray) -> np.ndarray:
        """
        Memisahkan kasus 2 kubus yang berdekatan/menempel agar tidak terdeteksi sebagai 1 objek raksasa.
        Metode:
        1. Deteksi Canny edge / celah sambungan vertikal di antara kardus.
        2. Analisis Convexity Defects pada kontur yang terlalu lebar (Aspect Ratio > 1.45).
        3. Pemotongan garis celah (boundary seam cutting).
        """
        cleaned_mask = binary_mask.copy()
        cnts, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect = float(w) / max(float(h), 1.0)

            # Jika aspect ratio > 1.40 atau < 0.70 (kemungkinan 2 kardus bersampingan atau bertumpuk)
            if aspect > 1.40 or aspect < 0.70:
                roi_mask = cleaned_mask[y:y+h, x:x+w]
                roi_gray = gray_img[y:y+h, x:x+w]

                # Cari garis celah sambungan dengan Sobel vertikal (jika bersampingan)
                if aspect > 1.40:
                    sobelx = cv2.Sobel(roi_gray, cv2.CV_32F, 1, 0, ksize=3)
                    sobelx = np.abs(sobelx)
                    # Cari lembah (garis vertikal paling tajam di sekitar tengah 30% - 70%)
                    center_start = int(w * 0.35)
                    center_end = int(w * 0.65)
                    if center_end > center_start:
                        grad_strip = np.mean(sobelx[:, center_start:center_end], axis=0)
                        valley_rel_x = int(np.argmax(grad_strip))
                        split_x = x + center_start + valley_rel_x

                        # Potong masker selebar 3 piksel sepanjang tinggi kardus
                        cv2.line(cleaned_mask, (split_x, y), (split_x, y + h), 0, 3)

                # Cek juga Convexity Defects (cekukan berbentuk V pada batas atas & bawah)
                hull_idx = cv2.convexHull(cnt, returnPoints=False)
                if len(hull_idx) > 3:
                    defects = cv2.convexityDefects(cnt, hull_idx)
                    if defects is not None:
                        deep_pts = []
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            depth = d / 256.0
                            if depth > 6.0:
                                deep_pts.append(cnt[f][0])
                        # Jika ada 2 cekukan V yang saling berhadapan, potong di antara keduanya
                        if len(deep_pts) >= 2:
                            pt1 = deep_pts[0]
                            pt2 = deep_pts[1]
                            cv2.line(cleaned_mask, tuple(pt1), tuple(pt2), 0, 3)

        return cleaned_mask

    def detect(self, img_bgr: np.ndarray) -> Tuple[List[dict], np.ndarray]:
        """
        Deteksi kubus merah 35x35x35 cm, pemisahan 2 kubus berdekatan,
        ekstraksi sudut 3D, dan estimasi pose SolvePnP.
        """
        img_h, img_w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Masking Merah Murni (Anti-Kulit)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        mask_hsv = ((h <= self.h_high1) | (h >= self.h_low2)) & (s >= self.s_low) & (v >= self.v_low)

        b = img_bgr[:, :, 0].astype(np.float32)
        g = img_bgr[:, :, 1].astype(np.float32)
        r = img_bgr[:, :, 2].astype(np.float32)
        mask_rgb = (r - g >= self.red_diff_min) & (r - b >= self.red_diff_min) & (r >= 65)

        mask_pure_red = (mask_hsv & mask_rgb).astype(np.uint8) * 255

        # 2. Morfologi Ukuran Terkontrol (agar tidak menyatukan 2 kardus)
        k_close = max(3, self.close_ksize | 1)
        kernel_c = cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close))
        mask_closed = cv2.morphologyEx(mask_pure_red, cv2.MORPH_CLOSE, kernel_c)
        mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

        # 3. Pisahkan jika ada 2 kardus yang menempel
        if self.enable_multi_box_split:
            mask_clean = self.split_adjacent_boxes(mask_clean, gray)

        # 4. Cari Kontur Setiap Kubus
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            # A. Bounding Box & Centroid
            x, y, bw, bh = cv2.boundingRect(cnt)
            rect = cv2.minAreaRect(cnt)
            box_4pt = cv2.boxPoints(rect).astype(np.float32)

            moments = cv2.moments(cnt)
            if moments["m00"] > 1e-4:
                cx = float(moments["m10"] / moments["m00"])
                cy = float(moments["m01"] / moments["m00"])
            else:
                cx, cy = float(x + bw / 2), float(y + bh / 2)

            # B. Poligon Siluet 3D (4 - 8 Sudut Fisik)
            hull = cv2.convexHull(cnt)
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.025 * peri, True)
            outer_corners = approx.reshape(-1, 2)

            # Urutkan sudut luar siklik
            angles = np.arctan2(outer_corners[:, 1] - cy, outer_corners[:, 0] - cx)
            outer_corners = outer_corners[np.argsort(angles)]

            # C. Estimasi Pose 3D SolvePnP (Model Fisik Kubus 35x35x35 cm)
            # Urutkan 4 titik kotak muka kardus
            s = box_4pt.sum(axis=1)
            tl = box_4pt[np.argmin(s)]
            br = box_4pt[np.argmax(s)]
            diff = np.diff(box_4pt, axis=1)
            tr = box_4pt[np.argmin(diff)]
            bl = box_4pt[np.argmax(diff)]
            ordered_4pt = np.array([tl, tr, br, bl], dtype=np.float32)

            pnp_success, rvec, tvec = cv2.solvePnP(
                self.FACE_3D_POINTS,
                ordered_4pt,
                self.K,
                self.D,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            wireframe_2d = None
            dist_m = 0.0
            pos_3d = (0.0, 0.0, 0.0)
            est_dim_cm = 0.0

            if pnp_success:
                tx, ty, tz = tvec.ravel()
                pos_3d = (float(tx), float(ty), float(tz))
                dist_m = float(np.linalg.norm(tvec))

                # Proyeksikan seluruh 8 sudut kubus 35x35x35 cm ke 2D
                proj_8, _ = cv2.projectPoints(self.CUBE_3D_CORNERS, rvec, tvec, self.K, self.D)
                wireframe_2d = proj_8.reshape(-1, 2).astype(np.int32)

                # Estimasi ukuran fisik yang terlihat (apakah mendekati 35 cm?)
                fx = self.K[0, 0]
                est_dim_cm = (float(max(bw, bh)) * tz / fx) * 100.0

                # Validasi fisik: tolak jika ukuran fisik < 18 cm atau > 65 cm
                if est_dim_cm < 18.0 or est_dim_cm > 65.0:
                    continue
            else:
                # Estimasi jarak sederhana jika solvePnP gagal
                fx = self.K[0, 0]
                tz = (fx * self.CUBE_SIZE_M) / max(bw, bh)
                dist_m = float(tz)
                pos_3d = ((cx - self.K[0, 2]) * tz / fx, (cy - self.K[1, 2]) * tz / self.K[1, 1], tz)
                est_dim_cm = 35.0

            results.append({
                'outer_corners': outer_corners,
                'wireframe_8pts': wireframe_2d,
                'center': (cx, cy),
                'box_xyxy': [float(x), float(y), float(x + bw), float(y + bh)],
                'area': float(area),
                'distance_m': dist_m,
                'pos_3d_m': pos_3d,
                'est_size_cm': est_dim_cm,
                'num_corners': len(outer_corners)
            })

        # Urutkan dari kardus terdekat
        results.sort(key=lambda item: item['distance_m'])
        return results, mask_clean


def draw_calibrated_cube_visualizations(
    img_draw: np.ndarray,
    detections: List[dict],
    fps: float = 0.0
) -> np.ndarray:
    """Menggambar kubus kawat 3D 35x35x35 cm, seluruh sudut, jarak meter, dan label."""
    corner_colors = [
        (0, 255, 255),    # C1: Kuning
        (255, 255, 0),    # C2: Cyan
        (0, 255, 0),      # C3: Hijau
        (255, 0, 255),    # C4: Magenta
        (0, 165, 255),    # C5: Orange
        (255, 120, 120),  # C6: Biru
        (120, 255, 200),  # C7: Mint
        (200, 120, 255)   # C8: Lavender
    ]

    for i, det in enumerate(detections):
        cx, cy = int(det['center'][0]), int(det['center'][1])
        dist = det['distance_m']
        pos = det['pos_3d_m']
        dim = det['est_size_cm']
        outer = det['outer_corners'].astype(np.int32)
        wireframe = det['wireframe_8pts']

        # 1. Gambar Wireframe Kubus 3D (35x35x35 cm) jika PnP sukses
        if wireframe is not None and len(wireframe) == 8:
            # 12 Rusuk 3D kubus
            for p1_i, p2_i in Cube35Detector.CUBE_EDGES:
                pt1 = tuple(wireframe[p1_i])
                pt2 = tuple(wireframe[p2_i])
                cv2.line(img_draw, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)

            # Gambar 8 titik sudut kubus 3D
            for v_i, pt in enumerate(wireframe):
                cv2.circle(img_draw, tuple(pt), 4, (0, 0, 255), -1)

        # 2. Gambar Poligon Siluet Luar
        cv2.polylines(img_draw, [outer], isClosed=True, color=(0, 220, 255), thickness=2, lineType=cv2.LINE_AA)

        # 3. Gambar Setiap Sudut Fisik Luar (C1, C2, ...)
        for c_idx in range(len(outer)):
            pt = tuple(outer[c_idx])
            col = corner_colors[c_idx % len(corner_colors)]
            cv2.circle(img_draw, pt, 6, (0, 0, 0), -1)
            cv2.circle(img_draw, pt, 4, col, -1)
            cv2.putText(img_draw, f"C{c_idx+1}", (pt[0]-15, pt[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1)

        # 4. Titik Pusat & Info Koordinat 3D
        cv2.circle(img_draw, (cx, cy), 5, (0, 0, 255), -1)
        x1, y1, x2, y2 = [int(v) for v in det['box_xyxy']]

        header = f"Kubus #{i+1} [35x35x35cm] | d={dist:.2f}m (X={pos[0]:.2f}, Y={pos[1]:.2f}, Z={pos[2]:.2f})"
        cv2.putText(img_draw, header, (x1, max(y1 - 12, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2, cv2.LINE_AA)

        dim_label = f"Ukuran Fisik: ~{dim:.1f}cm ({len(outer)} sudut)"
        cv2.putText(img_draw, dim_label, (x1, min(y2 + 18, img_draw.shape[0] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # Info Status FPS & Deteksi
    fps_text = f"FPS: {fps:.1f} | Kubus 35cm Terdeteksi: {len(detections)}"
    cv2.putText(img_draw, fps_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    hint = "Tekan 'q'/'ESC': Keluar | 's': Simpan Foto | Split 2-Box Aktif"
    cv2.putText(img_draw, hint, (20, img_draw.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return img_draw


def run_standalone_camera(device: str, detector: Cube35Detector):
    """Menjalankan deteksi langsung dari USB Webcam."""
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

    win_main = "Deteksi Kubus 35x35x35 cm (Robocon 2027)"
    win_ctrl = "Kontrol Kalibrasi & Split 2-Box"
    cv2.namedWindow(win_main, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_main, 1024, 576)

    cv2.namedWindow(win_ctrl, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_ctrl, 420, 280)

    def nothing(x): pass

    cv2.createTrackbar("H_High1 (0-15)", win_ctrl, detector.h_high1, 20, nothing)
    cv2.createTrackbar("H_Low2 (160-180)", win_ctrl, detector.h_low2, 180, nothing)
    cv2.createTrackbar("S_Low (Saturasi)", win_ctrl, detector.s_low, 255, nothing)
    cv2.createTrackbar("RedDiff (R-G & R-B)", win_ctrl, detector.red_diff_min, 80, nothing)
    cv2.createTrackbar("CloseKernel", win_ctrl, detector.close_ksize, 31, nothing)
    cv2.createTrackbar("Split 2-Box (0/1)", win_ctrl, 1 if detector.enable_multi_box_split else 0, 1, nothing)

    t_prev = time.time()
    fps = 0.0

    print("Kamera aktif. Silakan letakkan 1 atau 2 kubus kardus merah 35 cm.")
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        detector.h_high1 = cv2.getTrackbarPos("H_High1 (0-15)", win_ctrl)
        detector.h_low2 = max(160, cv2.getTrackbarPos("H_Low2 (160-180)", win_ctrl))
        detector.s_low = cv2.getTrackbarPos("S_Low (Saturasi)", win_ctrl)
        detector.red_diff_min = cv2.getTrackbarPos("RedDiff (R-G & R-B)", win_ctrl)
        detector.close_ksize = max(3, cv2.getTrackbarPos("CloseKernel", win_ctrl))
        detector.enable_multi_box_split = (cv2.getTrackbarPos("Split 2-Box (0/1)", win_ctrl) == 1)

        t_now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-4, t_now - t_prev))
        t_prev = t_now

        dets, mask_clean = detector.detect(frame)
        view = frame.copy()
        draw_calibrated_cube_visualizations(view, dets, fps=fps)

        # PIP Masker Biner
        mask_bgr = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)
        pip_h, pip_w = 180, 320
        mask_small = cv2.resize(mask_bgr, (pip_w, pip_h))
        view[10:10 + pip_h, view.shape[1] - pip_w - 10:view.shape[1] - 10] = mask_small
        cv2.rectangle(view, (view.shape[1] - pip_w - 10, 10), (view.shape[1] - 10, 10 + pip_h), (0, 255, 255), 1)
        cv2.putText(view, "Masker Biner (Split Aktif)", (view.shape[1] - pip_w - 5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

        cv2.imshow(win_main, view)
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            break
        elif key == ord('s'):
            filename = f"cube_35cm_capture_{int(time.time())}.jpg"
            cv2.imwrite(filename, view)
            print(f"Tangkapan layar tersimpan: {filename}")

    cap.release()
    cv2.destroyAllWindows()


def run_image_test(image_path: str, detector: Cube35Detector):
    """Pengujian pada file gambar statis."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Gagal memuat file gambar: {image_path}")
        return

    t0 = time.perf_counter()
    dets, mask_clean = detector.detect(frame)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\nHasil Deteksi Kubus 35x35x35 cm ({image_path}):")
    print(f"- Latensi SolvePnP: {latency_ms:.2f} ms")
    print(f"- Jumlah Kubus Terdeteksi: {len(dets)}")

    for i, d in enumerate(dets):
        print(f"\n[Kubus #{i+1}]")
        print(f"  Posisi 3D Frame Kamera: X={d['pos_3d_m'][0]:.3f}m, Y={d['pos_3d_m'][1]:.3f}m, Z={d['pos_3d_m'][2]:.3f}m")
        print(f"  Jarak Nyata: {d['distance_m']:.3f} meter")
        print(f"  Estimasi Ukuran Fisik: {d['est_size_cm']:.1f} cm (Target: 35 cm)")
        print(f"  Jumlah Sudut Luar: {d['num_corners']}")

    out_view = frame.copy()
    draw_calibrated_cube_visualizations(out_view, dets, fps=1000.0 / max(1e-3, latency_ms))
    cv2.imwrite("cube_35cm_calibrated_result.jpg", out_view)
    print("\nHasil visualisasi tersimpan di: cube_35cm_calibrated_result.jpg")


def run_ros2_node(topic: str, detector: Cube35Detector):
    """Menjalankan deteksi melalui ROS 2 subscription."""
    if not HAS_ROS2:
        print("[ERROR] ROS 2 tidak terdeteksi di environment ini!")
        return

    rclpy.init()
    node = Node('test_cube35_detector_node')
    bridge = CvBridge()

    win_name = "ROS2 Cube 35cm Calibrated Detector"
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
        draw_calibrated_cube_visualizations(view, dets)

        for i, d in enumerate(dets):
            node.get_logger().info(
                f"[Kubus #{i+1}] d={d['distance_m']:.2f}m | 3D: ({d['pos_3d_m'][0]:.2f}, {d['pos_3d_m'][1]:.2f}, {d['pos_3d_m'][2]:.2f})m | "
                f"Ukuran: ~{d['est_size_cm']:.1f}cm",
                throttle_duration_sec=1.0
            )

        cv2.imshow(win_name, view)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

    node.create_subscription(Image, topic, img_callback, 10)
    print(f"Subscribed ke topik ROS 2: {topic}. Menunggu gambar...")

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
    parser = argparse.ArgumentParser(description="Detector Kubus 35x35x35 cm dengan Kalibrasi Brio")
    parser.add_argument("--device", type=str, default="/dev/video2",
                        help="Jalur device kamera (/dev/video2 atau index)")
    parser.add_argument("--image", type=str, default=None,
                        help="Uji coba pada file gambar statis")
    parser.add_argument("--ros", action="store_true",
                        help="Gunakan topik gambar ROS 2 (/image_raw)")
    parser.add_argument("--topic", type=str, default="/image_raw",
                        help="Topik ROS 2 jika --ros diaktifkan")
    parser.add_argument("--calib", type=str, default=None,
                        help="Path opsional ke camera_brio.yaml")

    args = parser.parse_args()

    # 1. Ambil data kalibrasi kamera Logitech MX Brio dari project FAST-LIVO2
    K, D = load_brio_calibration(args.calib)

    # 2. Inisialisasi detector kubus 35x35x35 cm
    detector = Cube35Detector(
        K=K, D=D,
        h_high1=7,
        h_low2=172,
        s_low=115,
        v_low=45,
        red_diff_min=30,
        min_area=2500,
        close_ksize=15,  # Kernel lebih rapat agar celah 2 kardus terjaga
        enable_multi_box_split=True
    )

    if args.image is not None:
        run_image_test(args.image, detector)
    elif args.ros:
        run_ros2_node(args.topic, detector)
    else:
        run_standalone_camera(args.device, detector)


if __name__ == '__main__':
    main()
