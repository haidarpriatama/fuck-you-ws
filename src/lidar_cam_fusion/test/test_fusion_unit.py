#!/usr/bin/env python3
"""
Unit Test untuk lidar_cam_fusion.fusion_node (Data Sintetis)
============================================================
Menguji:
1. Proyeksi titik LiDAR ke kamera dengan konvensi p_cam = R_cl @ p_lidar + t_cl
2. Logika penyusutan bounding box (shrink)
3. 1D Depth clustering & filter ukuran (memisahkan kubus 35 cm dari dinding di belakangnya)
4. Detektor YOLO & fallback HSV
"""

import sys
import unittest
import numpy as np
import cv2

from lidar_cam_fusion.fusion_node import LidarCamFusionNode


class TestFusionAlgorithms(unittest.TestCase):
    def setUp(self):
        self.fx = 775.0
        self.fy = 770.0
        self.cx = 640.0
        self.cy = 360.0
        self.K = np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self.D = np.zeros(5, dtype=np.float64)

        # LiDAR frame: X maju, Y kiri, Z atas
        # Camera optical frame: X kanan, Y bawah, Z maju
        self.R_cl = np.array([
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0],
            [1.0,  0.0,  0.0]
        ], dtype=np.float64)
        self.t_cl = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)

    def test_bounding_box_shrink(self):
        """Memverifikasi penyusutan bounding box."""
        box = [100.0, 100.0, 300.0, 300.0]
        shrink = 0.20
        img_w, img_h = 1280, 720
        shrunk = LidarCamFusionNode.compute_shrunk_box(box, shrink, img_w, img_h)

        expected = [140.0, 140.0, 260.0, 260.0]
        for val, exp in zip(shrunk, expected):
            self.assertAlmostEqual(val, exp, delta=1e-3)

    def test_lidar_projection(self):
        """Memverifikasi proyeksi titik 3D ke piksel 2D kamera."""
        p_lidar = np.array([[2.0, 0.0, 0.0]], dtype=np.float64)
        p_cam = (self.R_cl @ p_lidar.T + self.t_cl).T

        self.assertAlmostEqual(p_cam[0, 0], 0.0)
        self.assertAlmostEqual(p_cam[0, 1], 0.0)
        self.assertAlmostEqual(p_cam[0, 2], 2.0)

        proj, _ = cv2.projectPoints(p_cam, np.zeros(3), np.zeros(3), self.K, self.D)
        uv = proj.reshape(-1, 2)[0]

        self.assertAlmostEqual(uv[0], self.cx, delta=1e-2)
        self.assertAlmostEqual(uv[1], self.cy, delta=1e-2)

    def test_depth_clustering_cube_and_wall(self):
        """
        Sintesis titik awan kubus 35 cm di depan (Z=2.0m) dan dinding di belakang (Z=4.5m).
        Memverifikasi bahwa algoritma 1D depth clustering hanya mengambil titik kubus terdekat.
        """
        np.random.seed(42)

        # 1. Permukaan kubus 35 cm pada jarak 2.0 meter
        n_cube = 50
        cube_x = np.random.uniform(-0.15, 0.15, n_cube)
        cube_y = np.random.uniform(-0.15, 0.15, n_cube)
        cube_z = np.random.uniform(2.00, 2.30, n_cube)
        cube_cam = np.stack([cube_x, cube_y, cube_z], axis=1)

        # 2. Dinding di belakang kubus pada jarak 4.5 meter
        n_wall = 40
        wall_x = np.random.uniform(-0.25, 0.25, n_wall)
        wall_y = np.random.uniform(-0.25, 0.25, n_wall)
        wall_z = np.random.uniform(4.40, 4.60, n_wall)
        wall_cam = np.stack([wall_x, wall_y, wall_z], axis=1)

        all_pts_cam = np.vstack([cube_cam, wall_cam])

        proj, _ = cv2.projectPoints(all_pts_cam, np.zeros(3), np.zeros(3), self.K, self.D)
        all_uv = proj.reshape(-1, 2)

        shrunk_box = [570.0, 290.0, 710.0, 430.0]

        selected_pts, selected_uv, raw_count = LidarCamFusionNode.extract_cluster_in_box(
            all_pts_cam, all_uv, shrunk_box, gap=0.30, min_pts=5, max_extent=0.50
        )

        self.assertIsNotNone(selected_pts, "Kluster objek harus terdeteksi!")
        self.assertGreaterEqual(len(selected_pts), 5)

        mean_z = np.mean(selected_pts[:, 2])
        self.assertTrue(1.95 <= mean_z <= 2.35, f"Kedalaman rata-rata harus kubus (2.0-2.35m), didapat: {mean_z:.2f}m")
        self.assertTrue(np.all(selected_pts[:, 2] < 3.0), "Tidak boleh ada titik dinding (Z > 4m) yang masuk kluster!")

        spans = np.max(selected_pts, axis=0) - np.min(selected_pts, axis=0)
        self.assertLessEqual(np.max(spans), 0.50)

    def test_rejection_when_extent_too_large(self):
        """Memverifikasi bahwa cluster ditolak jika sebaran titiknya melebihi ukuran kubus."""
        pts_large = np.array([
            [-0.45, 0.0, 2.0],
            [ 0.00, 0.0, 2.0],
            [ 0.45, 0.0, 2.0],
            [-0.20, 0.1, 2.0],
            [ 0.20, -0.1, 2.0]
        ], dtype=np.float64)

        proj, _ = cv2.projectPoints(pts_large, np.zeros(3), np.zeros(3), self.K, self.D)
        uv = proj.reshape(-1, 2)
        shrunk_box = [0.0, 0.0, 1280.0, 720.0]

        selected_pts, _, _ = LidarCamFusionNode.extract_cluster_in_box(
            pts_large, uv, shrunk_box, gap=0.30, min_pts=5, max_extent=0.50
        )
        self.assertIsNone(selected_pts, "Kluster harus ditolak karena max_extent > 0.50m")


    def test_noise_and_ground_rejection(self):
        """
        Skenario nyata yang sering menyebabkan 'NO LIDAR CLUSTER' di lapangan:
        - Ada titik noise sasis dekat (Z=0.15m, 2 titik < min_pts)
        - Ada 50 titik kubus kardus di Z=1.1m
        - Ada 10 titik lantai yang menyambung di bagian bawah kubus
        Algoritma baru (metode Darshan / Fritz AI) harus:
        1. Menolak noise sasis (Z < min_range atau kluster < min_pts)
        2. Memilih kluster kubus yang padat
        3. Memangkas titik lantai berlebih sehingga kubus tetap terdeteksi!
        """
        np.random.seed(123)

        # 1. Noise sasis dekat (Z < min_range atau titik sedikit)
        noise_pts = np.array([
            [0.05, 0.10, 0.15],
            [0.06, 0.12, 0.16]
        ])

        # 2. Kubus 35 cm di Z=1.10m
        n_cube = 50
        cube_x = np.random.uniform(-0.10, 0.10, n_cube)
        cube_y = np.random.uniform(-0.10, 0.10, n_cube)
        cube_z = np.random.uniform(1.05, 1.15, n_cube)
        cube_pts = np.stack([cube_x, cube_y, cube_z], axis=1)

        # 3. Titik lantai yang melebar ke bawah
        n_ground = 10
        ground_x = np.random.uniform(-0.10, 0.10, n_ground)
        ground_y = np.random.uniform(0.30, 0.60, n_ground)
        ground_z = np.random.uniform(1.05, 1.15, n_ground)
        ground_pts = np.stack([ground_x, ground_y, ground_z], axis=1)

        all_pts = np.vstack([noise_pts, cube_pts, ground_pts])
        proj, _ = cv2.projectPoints(all_pts, np.zeros(3), np.zeros(3), self.K, self.D)
        all_uv = proj.reshape(-1, 2)

        # Box mencakup area proyeksi
        shrunk_box = [0.0, 0.0, 1280.0, 720.0]

        selected_pts, selected_uv, raw_count = LidarCamFusionNode.extract_cluster_in_box(
            all_pts, all_uv, shrunk_box, gap=0.25, min_pts=3, max_extent=0.50, min_range=0.35
        )

        self.assertIsNotNone(selected_pts, "Kluster kubus harus berhasil diekstrak meskipun ada noise dan lantai!")
        mean_z = np.mean(selected_pts[:, 2])
        self.assertTrue(1.05 <= mean_z <= 1.15, f"Kedalaman kluster harus di kubus (~1.10m), didapat: {mean_z:.2f}m")
        # Pastikan noise dekat Z < 0.35m TIDAK masuk
        self.assertTrue(np.all(selected_pts[:, 2] >= 0.35), "Noise dekat tidak boleh masuk!")

    def test_lidar_fov_clipping(self):
        """Memverifikasi bahwa titik LiDAR di luar FOV kamera dipotong secara tepat."""
        class DummyNode:
            def __init__(self, K, D, R_cl, t_cl, min_range):
                self.K = K
                self.D = D
                self.R_cl = R_cl
                self.t_cl = t_cl
                self.min_range = min_range
            project_lidar_to_camera = LidarCamFusionNode.project_lidar_to_camera

        dummy = DummyNode(self.K, self.D, self.R_cl, self.t_cl, min_range=0.35)

        # Titik 1: Di depan pas di tengah (Z_c = 2.0m, tepat di tengah gambar [640, 360])
        p1 = [2.0, 0.0, 0.0]

        # Titik 2: Di belakang robot (Z_c = -2.0m -> harus dieliminasi)
        p2 = [-2.0, 0.0, 0.0]

        # Titik 3: Jauh di samping kiri (di luar HFOV -> harus dieliminasi)
        p3 = [1.0, 3.0, 0.0]

        pts_lidar = np.array([p1, p2, p3], dtype=np.float64)
        pts_cam_fov, proj_uv_fov, in_fov, pts_lidar_fov = dummy.project_lidar_to_camera(pts_lidar, 1280, 720)

        self.assertEqual(len(pts_cam_fov), 1, "Hanya 1 titik (tengah) yang boleh lolos FOV kamera!")
        self.assertAlmostEqual(proj_uv_fov[0][0], self.cx, delta=1.0)
        self.assertAlmostEqual(proj_uv_fov[0][1], self.cy, delta=1.0)
        self.assertTrue(0 <= proj_uv_fov[0][0] < 1280)
        self.assertTrue(0 <= proj_uv_fov[0][1] < 720)


if __name__ == '__main__':
    unittest.main()
