"""Core pipeline for the experimental Volume Hakarukun Colab notebook.

The implementation estimates a visual hull from silhouettes observed around a
metric ChArUco board.  It is intentionally conservative: concavities that are
not visible in silhouettes remain filled and can overestimate the true volume.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class BoardSpec:
    squares_x: int = 7
    squares_y: int = 10
    square_length_m: float = 0.025
    marker_length_m: float = 0.018
    dictionary_id: int = cv2.aruco.DICT_4X4_100

    @property
    def width_m(self) -> float:
        return self.squares_x * self.square_length_m

    @property
    def height_m(self) -> float:
        return self.squares_y * self.square_length_m


@dataclass
class PoseFrame:
    image_path: str
    rvec: np.ndarray
    tvec: np.ndarray
    charuco_corner_count: int
    mask_path: str | None = None


@dataclass(frozen=True)
class VolumeBounds:
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    height_m: float

    def validate(self, board_spec: BoardSpec) -> None:
        if not (0 <= self.x_min_m < self.x_max_m <= board_spec.width_m):
            raise ValueError("X bounds must fit inside the ChArUco board.")
        if not (0 <= self.y_min_m < self.y_max_m <= board_spec.height_m):
            raise ValueError("Y bounds must fit inside the ChArUco board.")
        if self.height_m <= 0:
            raise ValueError("Maximum object height must be positive.")


@dataclass
class VolumeResult:
    voxel_volume_cm3: float
    mesh_volume_cm3: float | None
    occupied_voxels: int
    voxel_size_mm: float
    usable_frames: int
    minimum_views: int
    support_ratio: float
    mesh_watertight: bool | None
    warning: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def create_charuco_board(spec: BoardSpec):
    dictionary = cv2.aruco.getPredefinedDictionary(spec.dictionary_id)
    board = cv2.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y),
        spec.square_length_m,
        spec.marker_length_m,
        dictionary,
    )
    return dictionary, board


def render_charuco_board(
    output_path: str | Path,
    spec: BoardSpec = BoardSpec(),
    pixels_per_square: int = 160,
) -> Path:
    """Render the exact board image without adding a scale-changing margin."""
    _, board = create_charuco_board(spec)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    size = (spec.squares_x * pixels_per_square, spec.squares_y * pixels_per_square)
    image = board.generateImage(size, marginSize=0, borderBits=1)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Could not write board image to {output_path}")
    return output_path


def extract_video_frames(
    video_path: str | Path,
    output_dir: str | Path,
    target_frames: int = 36,
    max_side: int = 1280,
    minimum_sharpness: float = 35.0,
) -> list[Path]:
    """Extract evenly spaced, reasonably sharp frames from a video."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < 2:
        capture.release()
        raise ValueError("The uploaded video has too few frames.")

    candidate_count = min(frame_count, max(target_frames * 3, target_frames))
    candidate_indices = np.unique(
        np.linspace(0, frame_count - 1, candidate_count, dtype=np.int64)
    )
    candidates: list[tuple[int, float, np.ndarray]] = []
    for frame_index in candidate_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            continue
        height, width = frame.shape[:2]
        scale = min(1.0, max_side / max(height, width))
        if scale < 1:
            frame = cv2.resize(
                frame,
                (round(width * scale), round(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharpness >= minimum_sharpness:
            candidates.append((int(frame_index), sharpness, frame))
    capture.release()

    if len(candidates) < 8:
        raise ValueError(
            "Too few sharp frames were found. Record again with slower movement and brighter light."
        )

    # Keep temporal coverage while preferring the sharpest frame in each bin.
    selected: list[tuple[int, float, np.ndarray]] = []
    bins = np.array_split(np.asarray(candidates, dtype=object), min(target_frames, len(candidates)))
    for bin_items in bins:
        selected.append(max(bin_items.tolist(), key=lambda item: item[1]))
    selected.sort(key=lambda item: item[0])

    paths: list[Path] = []
    for output_index, (source_index, _, frame) in enumerate(selected):
        path = output_dir / f"frame_{output_index:03d}_{source_index:06d}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 94])
        paths.append(path)
    return paths


def _detect_charuco(image: np.ndarray, board):
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(image)
    return charuco_corners, charuco_ids, marker_corners, marker_ids


def calibrate_camera_from_board(
    image_paths: Sequence[str | Path],
    spec: BoardSpec = BoardSpec(),
    minimum_corners: int = 10,
):
    """Calibrate a fixed-lens video camera from ChArUco observations."""
    _, board = create_charuco_board(spec)
    all_corners: list[np.ndarray] = []
    all_ids: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    accepted_paths: list[str] = []

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        image_size = (width, height)
        corners, ids, _, _ = _detect_charuco(image, board)
        if ids is None or corners is None or len(ids) < minimum_corners:
            continue
        all_corners.append(np.asarray(corners, dtype=np.float32))
        all_ids.append(np.asarray(ids, dtype=np.int32))
        accepted_paths.append(str(path))

    if image_size is None or len(all_corners) < 8:
        raise ValueError(
            "The marker board was not detected in enough frames. Keep most of the board visible while filming."
        )

    error, camera_matrix, distortion, _, _ = cv2.aruco.calibrateCameraCharuco(
        all_corners,
        all_ids,
        board,
        image_size,
        None,
        None,
        flags=cv2.CALIB_RATIONAL_MODEL,
    )
    return camera_matrix, distortion, float(error), accepted_paths


def estimate_board_poses(
    image_paths: Sequence[str | Path],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    spec: BoardSpec = BoardSpec(),
    minimum_corners: int = 8,
) -> list[PoseFrame]:
    _, board = create_charuco_board(spec)
    poses: list[PoseFrame] = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        corners, ids, _, _ = _detect_charuco(image, board)
        if ids is None or corners is None or len(ids) < minimum_corners:
            continue
        object_points, image_points = board.matchImagePoints(corners, ids)
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            continue
        poses.append(
            PoseFrame(
                image_path=str(path),
                rvec=np.asarray(rvec, dtype=np.float64),
                tvec=np.asarray(tvec, dtype=np.float64),
                charuco_corner_count=int(len(ids)),
            )
        )
    if len(poses) < 8:
        raise ValueError(
            "Fewer than eight usable camera poses were recovered. Record a slower full orbit with the board visible."
        )
    return poses


def _largest_centered_component(mask: np.ndarray, center_xy: tuple[float, float]) -> np.ndarray:
    binary = (mask > 127).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return binary * 255
    cx, cy = center_xy
    diagonal = math.hypot(mask.shape[1], mask.shape[0])
    best_label = 0
    best_score = -np.inf
    for label in range(1, count):
        area = float(stats[label, cv2.CC_STAT_AREA])
        distance = math.hypot(centroids[label][0] - cx, centroids[label][1] - cy)
        score = math.log1p(area) - 3.0 * distance / max(diagonal, 1.0)
        if score > best_score:
            best_score = score
            best_label = label
    return (labels == best_label).astype(np.uint8) * 255


def projected_bounds_mask(
    image_shape: tuple[int, int],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    pose: PoseFrame,
    bounds: VolumeBounds,
    padding_px: int = 18,
) -> tuple[np.ndarray, tuple[float, float]]:
    x0, x1 = bounds.x_min_m, bounds.x_max_m
    y0, y1 = bounds.y_min_m, bounds.y_max_m
    z0, z1 = -bounds.height_m, 0.0
    corners_3d = np.asarray(
        [
            [x, y, z]
            for z in (z0, z1)
            for y in (y0, y1)
            for x in (x0, x1)
        ],
        dtype=np.float32,
    )
    projected, _ = cv2.projectPoints(
        corners_3d, pose.rvec, pose.tvec, camera_matrix, distortion
    )
    points = np.round(projected.reshape(-1, 2)).astype(np.int32)
    hull = cv2.convexHull(points)
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    if padding_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (padding_px * 2 + 1, padding_px * 2 + 1)
        )
        mask = cv2.dilate(mask, kernel)
    center_3d = np.asarray(
        [[(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]], dtype=np.float32
    )
    center_2d, _ = cv2.projectPoints(
        center_3d, pose.rvec, pose.tvec, camera_matrix, distortion
    )
    center = tuple(center_2d.reshape(2).tolist())
    return mask, center


def segment_pose_frames_with_rembg(
    poses: Sequence[PoseFrame],
    output_dir: str | Path,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    bounds: VolumeBounds,
    model_name: str = "u2net",
) -> list[PoseFrame]:
    """Generate automatic silhouettes. Import rembg lazily for local testability."""
    from rembg import new_session, remove

    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    session = new_session(model_name)
    segmented: list[PoseFrame] = []

    for index, pose in enumerate(poses):
        image = cv2.imread(pose.image_path)
        if image is None:
            continue
        raw_mask = remove(image, session=session, only_mask=True)
        if raw_mask.ndim == 3:
            raw_mask = raw_mask[:, :, 0]
        roi_mask, center = projected_bounds_mask(
            image.shape[:2], camera_matrix, distortion, pose, bounds
        )
        mask = cv2.bitwise_and(np.asarray(raw_mask, dtype=np.uint8), roi_mask)
        mask = _largest_centered_component(mask, center)
        size = max(3, round(min(image.shape[:2]) * 0.007))
        if size % 2 == 0:
            size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_path = output_dir / f"mask_{index:03d}.png"
        cv2.imwrite(str(mask_path), mask)
        segmented.append(
            PoseFrame(
                image_path=pose.image_path,
                rvec=pose.rvec,
                tvec=pose.tvec,
                charuco_corner_count=pose.charuco_corner_count,
                mask_path=str(mask_path),
            )
        )
    return segmented


def make_voxel_grid(bounds: VolumeBounds, voxel_size_m: float):
    if voxel_size_m <= 0:
        raise ValueError("Voxel size must be positive.")
    xs = np.arange(bounds.x_min_m + voxel_size_m / 2, bounds.x_max_m, voxel_size_m)
    ys = np.arange(bounds.y_min_m + voxel_size_m / 2, bounds.y_max_m, voxel_size_m)
    zs = np.arange(-bounds.height_m + voxel_size_m / 2, 0.0, voxel_size_m)
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())).astype(np.float32)
    return points, (len(zs), len(ys), len(xs)), (xs, ys, zs)


def carve_visual_hull(
    pose_frames: Sequence[PoseFrame],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    bounds: VolumeBounds,
    voxel_size_m: float = 0.004,
    support_ratio: float = 0.72,
    minimum_views: int | None = None,
    chunk_size: int = 250_000,
):
    """Carve a metric occupancy grid by projecting voxels into silhouettes."""
    if not 0.5 <= support_ratio <= 1.0:
        raise ValueError("Support ratio must be between 0.5 and 1.0.")
    frames = [pose for pose in pose_frames if pose.mask_path]
    if len(frames) < 6:
        raise ValueError("At least six silhouette frames are required.")
    minimum_views = minimum_views or max(4, min(8, len(frames) // 3))

    points, grid_shape, axes = make_voxel_grid(bounds, voxel_size_m)
    hit_count = np.zeros(len(points), dtype=np.uint16)
    view_count = np.zeros(len(points), dtype=np.uint16)

    for pose in frames:
        mask = cv2.imread(str(pose.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        height, width = mask.shape
        for start in range(0, len(points), chunk_size):
            stop = min(len(points), start + chunk_size)
            chunk = points[start:stop]
            rotation, _ = cv2.Rodrigues(pose.rvec)
            camera_points = chunk @ rotation.T + pose.tvec.reshape(1, 3)
            positive_depth = camera_points[:, 2] > 1e-6
            projected, _ = cv2.projectPoints(
                chunk, pose.rvec, pose.tvec, camera_matrix, distortion
            )
            uv = np.rint(projected.reshape(-1, 2)).astype(np.int32)
            inside = (
                positive_depth
                & (uv[:, 0] >= 0)
                & (uv[:, 0] < width)
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < height)
            )
            local_indices = np.flatnonzero(inside)
            global_indices = start + local_indices
            view_count[global_indices] += 1
            foreground = mask[uv[local_indices, 1], uv[local_indices, 0]] > 127
            hit_count[global_indices[foreground]] += 1

    support = hit_count.astype(np.float32) / np.maximum(view_count, 1)
    occupied = (view_count >= minimum_views) & (support >= support_ratio)
    occupancy = occupied.reshape(grid_shape)
    return occupancy, axes, hit_count.reshape(grid_shape), view_count.reshape(grid_shape)


def occupancy_to_mesh(
    occupancy: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    voxel_size_m: float,
):
    from skimage.measure import marching_cubes
    import trimesh

    if not np.any(occupancy):
        raise ValueError("No occupied volume remains after carving.")
    padded = np.pad(occupancy.astype(np.uint8), 1)
    vertices_zyx, faces, normals, _ = marching_cubes(
        padded, level=0.5, spacing=(voxel_size_m,) * 3
    )
    xs, ys, zs = axes
    origin_zyx = np.asarray(
        [zs[0] - voxel_size_m, ys[0] - voxel_size_m, xs[0] - voxel_size_m]
    )
    vertices_zyx += origin_zyx
    vertices_xyz = vertices_zyx[:, [2, 1, 0]]
    mesh = trimesh.Trimesh(vertices=vertices_xyz, faces=faces, process=True)
    return mesh


def save_volume_outputs(
    occupancy: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    voxel_size_m: float,
    output_dir: str | Path,
    usable_frames: int,
    minimum_views: int,
    support_ratio: float,
) -> VolumeResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    occupied_voxels = int(np.count_nonzero(occupancy))
    voxel_volume_cm3 = occupied_voxels * voxel_size_m**3 * 1_000_000
    mesh = occupancy_to_mesh(occupancy, axes, voxel_size_m)
    mesh_volume_cm3 = abs(float(mesh.volume)) * 1_000_000 if mesh.is_volume else None
    mesh.export(output_dir / "volume_model.glb")
    mesh.export(output_dir / "volume_model.stl")
    np.savez_compressed(
        output_dir / "occupancy_grid.npz",
        occupancy=occupancy,
        x=axes[0],
        y=axes[1],
        z=axes[2],
        voxel_size_m=voxel_size_m,
    )
    result = VolumeResult(
        voxel_volume_cm3=round(voxel_volume_cm3, 2),
        mesh_volume_cm3=round(mesh_volume_cm3, 2) if mesh_volume_cm3 is not None else None,
        occupied_voxels=occupied_voxels,
        voxel_size_mm=round(voxel_size_m * 1000, 3),
        usable_frames=usable_frames,
        minimum_views=minimum_views,
        support_ratio=support_ratio,
        mesh_watertight=bool(mesh.is_watertight),
        warning=(
            "Experimental visual-hull estimate. Hidden concavities remain filled and may "
            "overestimate true volume. Do not use for critical decisions."
        ),
    )
    (output_dir / "result.json").write_text(result.to_json(), encoding="utf-8")
    return result


def save_contact_sheet(
    image_paths: Sequence[str | Path],
    output_path: str | Path,
    mask_paths: Sequence[str | Path] | None = None,
    columns: int = 4,
    thumb_width: int = 320,
) -> Path:
    images: list[np.ndarray] = []
    for index, path in enumerate(image_paths):
        image = cv2.imread(str(path))
        if image is None:
            continue
        if mask_paths is not None and index < len(mask_paths):
            mask = cv2.imread(str(mask_paths[index]), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                overlay = np.zeros_like(image)
                overlay[:, :, 1] = 220
                alpha = (mask.astype(np.float32) / 255.0 * 0.45)[:, :, None]
                image = (image * (1 - alpha) + overlay * alpha).astype(np.uint8)
        height, width = image.shape[:2]
        new_height = max(1, round(height * thumb_width / width))
        images.append(cv2.resize(image, (thumb_width, new_height)))
    if not images:
        raise ValueError("No images were available for the contact sheet.")
    row_height = max(image.shape[0] for image in images)
    rows = math.ceil(len(images) / columns)
    sheet = np.full((rows * row_height, columns * thumb_width, 3), 245, np.uint8)
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        y = row * row_height
        x = column * thumb_width
        sheet[y : y + image.shape[0], x : x + image.shape[1]] = image
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)
    return output_path


def default_bounds(spec: BoardSpec = BoardSpec(), margin_m: float = 0.025, height_m: float = 0.15):
    return VolumeBounds(
        x_min_m=margin_m,
        x_max_m=spec.width_m - margin_m,
        y_min_m=margin_m,
        y_max_m=spec.height_m - margin_m,
        height_m=height_m,
    )


__all__ = [
    "BoardSpec",
    "PoseFrame",
    "VolumeBounds",
    "VolumeResult",
    "calibrate_camera_from_board",
    "carve_visual_hull",
    "create_charuco_board",
    "default_bounds",
    "estimate_board_poses",
    "extract_video_frames",
    "occupancy_to_mesh",
    "projected_bounds_mask",
    "render_charuco_board",
    "save_contact_sheet",
    "save_volume_outputs",
    "segment_pose_frames_with_rembg",
]
