import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from colab.volume_pipeline import (
    BoardSpec,
    PoseFrame,
    VolumeBounds,
    carve_visual_hull,
    create_charuco_board,
    make_voxel_grid,
    occupancy_to_mesh,
    render_charuco_board,
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
