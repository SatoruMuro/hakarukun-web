import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from colab.volume_pipeline import (
    BoardSpec,
    MaskQuality,
    PoseFrame,
    VolumeBounds,
    assess_mask_quality,
    camera_orbit_coverage_degrees,
    carve_visual_hull,
    create_charuco_board,
    make_voxel_grid,
    occupancy_diagnostics,
    occupancy_to_mesh,
    parse_frame_indices,
    render_charuco_board,
    select_pose_frames,
    save_volume_outputs,
)


def look_at_pose(camera_center, target):
    camera_center = np.asarray(camera_center, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - camera_center
    forward /= np.linalg.norm(forward)
    world_down = np.asarray([0.0, 1.0, 0.0])
    right = np.cross(world_down, forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack((right, down, forward))
    translation = -rotation @ camera_center
    rvec, _ = cv2.Rodrigues(rotation)
    return rvec, translation.reshape(3, 1)


class VolumePipelineTest(unittest.TestCase):
    def test_mask_quality_rejects_board_sized_leak(self):
        roi = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(roi, (20, 20), (180, 180), 255, -1)
        good_mask = np.zeros_like(roi)
        cv2.ellipse(good_mask, (100, 100), (35, 50), 0, 0, 360, 255, -1)
        leaked_mask = roi.copy()

        good = assess_mask_quality(good_mask, roi, (100, 100), 1)
        leaked = assess_mask_quality(leaked_mask, roi, (100, 100), 2)

        self.assertTrue(good.accepted)
        self.assertFalse(leaked.accepted)
        self.assertIn("mask-too-large", leaked.reasons)
        self.assertIn("touches-search-border", leaked.reasons)

    def test_manual_and_automatic_frame_selection(self):
        def frame(index, accepted=True):
            return PoseFrame(
                image_path=f"frame-{index}.png",
                rvec=np.zeros((3, 1)),
                tvec=np.zeros((3, 1)),
                charuco_corner_count=20,
                frame_index=index,
                mask_quality=MaskQuality(index, accepted, [], 0.2, 0.0, 0.0, 0.0),
            )

        frames = [frame(1), frame(2, accepted=False), frame(3), frame(4)]
        accepted, rejected = select_pose_frames(frames, "3-4")
        self.assertEqual([item.frame_index for item in accepted], [1])
        self.assertEqual([item.frame_index for item in rejected], [2, 3, 4])
        self.assertEqual(parse_frame_indices("1, 4-6, 9"), {1, 4, 5, 6, 9})

    def test_rectangular_prism_diagnostics_report_height(self):
        voxel_size = 0.002
        occupancy = np.zeros((12, 32, 48), dtype=bool)
        occupancy[1:11, 1:31, 1:47] = True
        axes = (
            np.arange(48, dtype=np.float64) * voxel_size,
            np.arange(32, dtype=np.float64) * voxel_size,
            -np.arange(12, 0, -1, dtype=np.float64) * voxel_size,
        )
        diagnostics = occupancy_diagnostics(occupancy, axes, voxel_size)
        self.assertAlmostEqual(diagnostics.length_mm, 92.0, places=1)
        self.assertAlmostEqual(diagnostics.width_mm, 60.0, places=1)
        self.assertAlmostEqual(diagnostics.height_mm, 20.0, places=1)
        self.assertAlmostEqual(diagnostics.max_cross_section_cm2, 55.2, places=1)
        self.assertFalse(diagnostics.touches_height_limit)
        self.assertFalse(diagnostics.touches_horizontal_limit)

    def test_orbit_coverage_detects_nearly_full_circle(self):
        bounds = VolumeBounds(0.02, 0.16, 0.04, 0.21, 0.12)
        center = np.asarray([0.09, 0.125, -0.05])
        frames = []
        for index, angle in enumerate(np.linspace(0, 2 * np.pi, 16, endpoint=False), start=1):
            camera = center + np.asarray([0.3 * np.cos(angle), 0.3 * np.sin(angle), -0.2])
            rvec, tvec = look_at_pose(camera, center)
            frames.append(PoseFrame("", rvec, tvec, 20, frame_index=index))
        coverage = camera_orbit_coverage_degrees(frames, bounds)
        self.assertIsNotNone(coverage)
        self.assertGreater(coverage, 330)

    def test_saved_result_includes_dimensions_and_cross_sections(self):
        voxel_size = 0.004
        occupancy = np.zeros((7, 12, 16), dtype=bool)
        occupancy[1:6, 1:11, 1:15] = True
        axes = (
            np.arange(16, dtype=np.float64) * voxel_size,
            np.arange(12, dtype=np.float64) * voxel_size,
            -np.arange(7, 0, -1, dtype=np.float64) * voxel_size,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = save_volume_outputs(
                occupancy,
                axes,
                voxel_size,
                directory,
                usable_frames=16,
                minimum_views=8,
                support_ratio=0.88,
            )
            saved = json.loads((Path(directory) / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result.dimensions.height_mm, 20.0)
            self.assertEqual(saved["dimensions"]["height_mm"], 20.0)
            self.assertTrue((Path(directory) / "cross_sections.csv").exists())
            self.assertTrue((Path(directory) / "mask_quality.json").exists())

    def test_mesh_coordinates_are_centered_on_voxels(self):
        occupancy = np.ones((1, 1, 1), dtype=bool)
        axes = (
            np.asarray([0.10]),
            np.asarray([0.20]),
            np.asarray([-0.03]),
        )
        mesh = occupancy_to_mesh(occupancy, axes, 0.01)
        np.testing.assert_allclose(
            mesh.bounds,
            np.asarray([[0.095, 0.195, -0.035], [0.105, 0.205, -0.025]]),
            atol=1e-8,
        )

    def test_generated_board_is_detectable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = render_charuco_board(Path(directory) / "board.png")
            image = cv2.imread(str(path))
            _, board = create_charuco_board(BoardSpec())
            detector = cv2.aruco.CharucoDetector(board)
            corners, ids, _, _ = detector.detectBoard(image)
            self.assertIsNotNone(ids)
            self.assertGreaterEqual(len(ids), 40)
            self.assertEqual(corners.shape[0], ids.shape[0])

    def test_visual_hull_recovers_synthetic_ellipsoid(self):
        bounds = VolumeBounds(0.02, 0.16, 0.04, 0.21, 0.12)
        voxel_size = 0.006
        points, shape, _ = make_voxel_grid(bounds, voxel_size)
        center = np.asarray([0.09, 0.125, -0.055], dtype=np.float32)
        radii = np.asarray([0.04, 0.05, 0.05], dtype=np.float32)
        truth = np.sum(((points - center) / radii) ** 2, axis=1) <= 1.0

        camera_matrix = np.asarray(
            [[360.0, 0.0, 200.0], [0.0, 360.0, 200.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        distortion = np.zeros((5, 1), dtype=np.float64)
        pose_frames = []
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            target = center.astype(np.float64)
            for index, angle in enumerate(np.linspace(0, 2 * np.pi, 16, endpoint=False)):
                camera_center = [
                    center[0] + 0.32 * np.cos(angle),
                    center[1] + 0.32 * np.sin(angle),
                    -0.24,
                ]
                rvec, tvec = look_at_pose(camera_center, target)
                projected, _ = cv2.projectPoints(
                    points[truth], rvec, tvec, camera_matrix, distortion
                )
                uv = np.rint(projected.reshape(-1, 2)).astype(np.int32)
                mask = np.zeros((400, 400), dtype=np.uint8)
                valid = (
                    (uv[:, 0] >= 0)
                    & (uv[:, 0] < 400)
                    & (uv[:, 1] >= 0)
                    & (uv[:, 1] < 400)
                )
                mask[uv[valid, 1], uv[valid, 0]] = 255
                mask = cv2.morphologyEx(
                    mask,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                )
                mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
                image_path = directory / f"image_{index:02d}.png"
                mask_path = directory / f"mask_{index:02d}.png"
                cv2.imwrite(str(image_path), np.zeros((400, 400, 3), np.uint8))
                cv2.imwrite(str(mask_path), mask)
                pose_frames.append(
                    PoseFrame(
                        image_path=str(image_path),
                        rvec=rvec,
                        tvec=tvec,
                        charuco_corner_count=30,
                        mask_path=str(mask_path),
                    )
                )

            occupancy, _, _, _ = carve_visual_hull(
                pose_frames,
                camera_matrix,
                distortion,
                bounds,
                voxel_size_m=voxel_size,
                support_ratio=0.9,
                minimum_views=12,
            )
            estimated = int(np.count_nonzero(occupancy))
            true_count = int(np.count_nonzero(truth))
            self.assertGreater(estimated, true_count * 0.75)
            self.assertLess(estimated, true_count * 1.65)


if __name__ == "__main__":
    unittest.main()
