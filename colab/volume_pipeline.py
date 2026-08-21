"""Core pipeline for the experimental Volume Hakarukun Colab notebook.

The implementation estimates a visual hull from silhouettes observed around a
metric ChArUco board.  It is intentionally conservative: concavities that are
not visible in silhouettes remain filled and can overestimate the true volume.
"""

from __future__ import annotations

import json
import math
import shutil
from csv import writer
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
class MaskQuality:
    frame_index: int
    accepted: bool
    reasons: list[str]
    mask_area_ratio: float
    roi_border_coverage: float
    image_edge_ratio: float
    center_distance_ratio: float


@dataclass
class PoseFrame:
    image_path: str
    rvec: np.ndarray
    tvec: np.ndarray
    charuco_corner_count: int
    mask_path: str | None = None
    frame_index: int = 0
    mask_quality: MaskQuality | None = None


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
class VolumeDiagnostics:
    length_mm: float
    width_mm: float
    height_mm: float
    axis_x_mm: float
    axis_y_mm: float
    max_cross_section_cm2: float
    occupied_layers: int
    touches_height_limit: bool
    touches_horizontal_limit: bool


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
    dimensions: VolumeDiagnostics
    orbit_coverage_degrees: float | None
    accepted_frame_indices: list[int]
    rejected_frame_indices: list[int]
    quality_warnings: list[str]
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
    for frame_index, path in enumerate(image_paths, start=1):
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
                frame_index=frame_index,
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


def assess_mask_quality(
    mask: np.ndarray,
    roi_mask: np.ndarray,
    center_xy: tuple[float, float],
    frame_index: int,
) -> MaskQuality:
    """Detect obvious foreground-removal failures without judging object shape."""
    foreground = mask > 127
    roi = roi_mask > 127
    foreground_area = int(np.count_nonzero(foreground))
    roi_area = max(1, int(np.count_nonzero(roi)))
    mask_area_ratio = foreground_area / roi_area

    # A correct object silhouette normally stays clear of the projected search-volume
    # boundary. Board/background leaks tend to cover a large part of this band.
    band_width = max(3, round(min(mask.shape) * 0.012))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (band_width * 2 + 1, band_width * 2 + 1)
    )
    inner_roi = cv2.erode(roi.astype(np.uint8), kernel) > 0
    roi_border = roi & ~inner_roi
    border_area = max(1, int(np.count_nonzero(roi_border)))
    roi_border_coverage = int(np.count_nonzero(foreground & roi_border)) / border_area

    edge_width = max(2, round(min(mask.shape) * 0.012))
    image_edge = np.zeros(mask.shape, dtype=bool)
    image_edge[:edge_width, :] = True
    image_edge[-edge_width:, :] = True
    image_edge[:, :edge_width] = True
    image_edge[:, -edge_width:] = True
    image_edge_ratio = (
        int(np.count_nonzero(foreground & image_edge)) / max(1, foreground_area)
    )

    if foreground_area:
        moments = cv2.moments(foreground.astype(np.uint8))
        centroid_x = moments["m10"] / moments["m00"]
        centroid_y = moments["m01"] / moments["m00"]
        x, y, width, height = cv2.boundingRect(roi.astype(np.uint8))
        roi_diagonal = max(1.0, math.hypot(width, height))
        center_distance_ratio = math.hypot(
            centroid_x - center_xy[0], centroid_y - center_xy[1]
        ) / roi_diagonal
    else:
        center_distance_ratio = 1.0

    reasons: list[str] = []
    if mask_area_ratio < 0.004:
        reasons.append("mask-too-small")
    if mask_area_ratio > 0.68:
        reasons.append("mask-too-large")
    if roi_border_coverage > 0.42:
        reasons.append("touches-search-border")
    if image_edge_ratio > 0.08:
        reasons.append("touches-image-edge")
    if center_distance_ratio > 0.42:
        reasons.append("far-from-object-center")
    return MaskQuality(
        frame_index=frame_index,
        accepted=not reasons,
        reasons=reasons,
        mask_area_ratio=round(float(mask_area_ratio), 4),
        roi_border_coverage=round(float(roi_border_coverage), 4),
        image_edge_ratio=round(float(image_edge_ratio), 4),
        center_distance_ratio=round(float(center_distance_ratio), 4),
    )


def _apply_temporal_mask_checks(frames: Sequence[PoseFrame]) -> None:
    """Mark isolated mask-area outliers after the per-frame checks."""
    candidates = [
        frame.mask_quality.mask_area_ratio
        for frame in frames
        if frame.mask_quality is not None and frame.mask_quality.accepted
    ]
    if len(candidates) < 6:
        return
    log_areas = np.log(np.maximum(np.asarray(candidates, dtype=np.float64), 1e-6))
    median = float(np.median(log_areas))
    mad = float(np.median(np.abs(log_areas - median)))
    # A minimum spread avoids rejecting normal perspective changes when masks are
    # unusually consistent.
    robust_scale = max(0.18, 1.4826 * mad)
    for frame in frames:
        quality = frame.mask_quality
        if quality is None or not quality.accepted:
            continue
        score = abs(math.log(max(quality.mask_area_ratio, 1e-6)) - median) / robust_scale
        if score > 3.8:
            quality.accepted = False
            quality.reasons.append("area-outlier")


def parse_frame_indices(value: str | Iterable[int]) -> set[int]:
    """Parse one-based frame numbers such as ``1, 4-7, 12``."""
    if not isinstance(value, str):
        return {int(index) for index in value}
    selected: set[int] = set()
    for token in value.replace(" ", "").split(","):
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start <= 0 or end < start:
                raise ValueError(f"Invalid frame range: {token}")
            selected.update(range(start, end + 1))
        else:
            index = int(token)
            if index <= 0:
                raise ValueError("Frame numbers must be one or greater.")
            selected.add(index)
    return selected


def select_pose_frames(
    frames: Sequence[PoseFrame],
    excluded_frames: str | Iterable[int] = "",
    auto_reject_bad_masks: bool = True,
) -> tuple[list[PoseFrame], list[PoseFrame]]:
    """Apply automatic quality decisions and a user's one-based exclusion list."""
    excluded = parse_frame_indices(excluded_frames)
    accepted: list[PoseFrame] = []
    rejected: list[PoseFrame] = []
    for fallback_index, frame in enumerate(frames, start=1):
        frame_index = frame.frame_index or fallback_index
        automatic_rejection = (
            auto_reject_bad_masks
            and frame.mask_quality is not None
            and not frame.mask_quality.accepted
        )
        if automatic_rejection or frame_index in excluded:
            rejected.append(frame)
        else:
            accepted.append(frame)
    return accepted, rejected


def camera_orbit_coverage_degrees(
    frames: Sequence[PoseFrame], bounds: VolumeBounds
) -> float | None:
    """Return horizontal angular coverage after removing the largest unobserved gap."""
    if len(frames) < 2:
        return None
    center = np.asarray(
        [(bounds.x_min_m + bounds.x_max_m) / 2, (bounds.y_min_m + bounds.y_max_m) / 2]
    )
    angles: list[float] = []
    for frame in frames:
        rotation, _ = cv2.Rodrigues(frame.rvec)
        camera_center = (-rotation.T @ frame.tvec.reshape(3, 1)).ravel()
        delta = camera_center[:2] - center
        if np.linalg.norm(delta) > 1e-8:
            angles.append(math.degrees(math.atan2(delta[1], delta[0])) % 360)
    if len(angles) < 2:
        return None
    sorted_angles = np.sort(np.asarray(angles))
    gaps = np.diff(np.r_[sorted_angles, sorted_angles[0] + 360])
    return round(float(360 - np.max(gaps)), 1)


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
        frame_index = pose.frame_index or index + 1
        quality = assess_mask_quality(mask, roi_mask, center, frame_index)
        segmented.append(
            PoseFrame(
                image_path=pose.image_path,
                rvec=pose.rvec,
                tvec=pose.tvec,
                charuco_corner_count=pose.charuco_corner_count,
                mask_path=str(mask_path),
                frame_index=frame_index,
                mask_quality=quality,
            )
        )
    _apply_temporal_mask_checks(segmented)
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
    support_ratio: float = 0.88,
    minimum_views: int | None = None,
    chunk_size: int = 250_000,
):
    """Carve a metric occupancy grid by projecting voxels into silhouettes."""
    if not 0.5 <= support_ratio <= 1.0:
        raise ValueError("Support ratio must be between 0.5 and 1.0.")
    frames = [pose for pose in pose_frames if pose.mask_path]
    if len(frames) < 6:
        raise ValueError("At least six silhouette frames are required.")
    minimum_views = minimum_views or max(6, min(12, math.ceil(len(frames) * 0.4)))

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


def occupancy_diagnostics(
    occupancy: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    voxel_size_m: float,
) -> VolumeDiagnostics:
    """Summarize reconstructed dimensions and search-boundary contact."""
    occupied_indices = np.argwhere(occupancy)
    if not len(occupied_indices):
        raise ValueError("No occupied volume remains after carving.")
    z_indices, y_indices, x_indices = occupied_indices.T
    voxel_mm = voxel_size_m * 1000
    unique_z = np.unique(z_indices)
    unique_y = np.unique(y_indices)
    unique_x = np.unique(x_indices)

    axis_x_mm = (int(unique_x[-1]) - int(unique_x[0]) + 1) * voxel_mm
    axis_y_mm = (int(unique_y[-1]) - int(unique_y[0]) + 1) * voxel_mm
    height_mm = (int(unique_z[-1]) - int(unique_z[0]) + 1) * voxel_mm

    footprint_y, footprint_x = np.nonzero(np.any(occupancy, axis=0))
    footprint = np.column_stack((axes[0][footprint_x], axes[1][footprint_y]))
    if len(footprint) >= 2:
        centered = footprint - np.mean(footprint, axis=0)
        covariance = centered.T @ centered / max(1, len(centered) - 1)
        _, eigenvectors = np.linalg.eigh(covariance)
        projected = centered @ eigenvectors
        spans_mm = (np.ptp(projected, axis=0) + voxel_size_m) * 1000
        length_mm, width_mm = sorted((float(value) for value in spans_mm), reverse=True)
    else:
        length_mm = width_mm = voxel_mm

    cross_sections_cm2 = np.count_nonzero(occupancy, axis=(1, 2)) * voxel_size_m**2 * 10_000
    touches_height_limit = bool(occupancy[0].any())
    touches_horizontal_limit = bool(
        occupancy[:, 0, :].any()
        or occupancy[:, -1, :].any()
        or occupancy[:, :, 0].any()
        or occupancy[:, :, -1].any()
    )
    return VolumeDiagnostics(
        length_mm=round(length_mm, 1),
        width_mm=round(width_mm, 1),
        height_mm=round(float(height_mm), 1),
        axis_x_mm=round(float(axis_x_mm), 1),
        axis_y_mm=round(float(axis_y_mm), 1),
        max_cross_section_cm2=round(float(np.max(cross_sections_cm2)), 2),
        occupied_layers=int(len(unique_z)),
        touches_height_limit=touches_height_limit,
        touches_horizontal_limit=touches_horizontal_limit,
    )


def cross_section_rows(
    occupancy: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    voxel_size_m: float,
) -> list[tuple[float, float]]:
    """Return height-above-board and occupied area for each voxel layer."""
    areas_cm2 = np.count_nonzero(occupancy, axis=(1, 2)) * voxel_size_m**2 * 10_000
    rows = [
        (round(float(-z_center * 1000), 3), round(float(area), 4))
        for z_center, area in zip(axes[2], areas_cm2)
    ]
    return sorted(rows, key=lambda row: row[0])


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
    accepted_frames: Sequence[PoseFrame] = (),
    rejected_frames: Sequence[PoseFrame] = (),
    orbit_coverage_degrees: float | None = None,
    hit_count: np.ndarray | None = None,
    view_count: np.ndarray | None = None,
) -> VolumeResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    occupied_voxels = int(np.count_nonzero(occupancy))
    voxel_volume_cm3 = occupied_voxels * voxel_size_m**3 * 1_000_000
    mesh = occupancy_to_mesh(occupancy, axes, voxel_size_m)
    mesh_volume_cm3 = abs(float(mesh.volume)) * 1_000_000 if mesh.is_volume else None
    mesh.export(output_dir / "volume_model.glb")
    mesh.export(output_dir / "volume_model.stl")
    grid_data = {
        "occupancy": occupancy,
        "x": axes[0],
        "y": axes[1],
        "z": axes[2],
        "voxel_size_m": voxel_size_m,
    }
    if hit_count is not None:
        grid_data["hit_count"] = hit_count
    if view_count is not None:
        grid_data["view_count"] = view_count
    np.savez_compressed(output_dir / "occupancy_grid.npz", **grid_data)

    diagnostics = occupancy_diagnostics(occupancy, axes, voxel_size_m)
    warnings: list[str] = []
    if diagnostics.touches_height_limit:
        warnings.append(
            "復元形状が最大高さの境界に達しています。最大対象物高さを増やして再実行してください。"
        )
    if diagnostics.touches_horizontal_limit:
        warnings.append(
            "復元形状が横方向の探索境界に達しています。ボード余白を小さくして再実行してください。"
        )
    if orbit_coverage_degrees is not None and orbit_coverage_degrees < 280:
        warnings.append(
            f"カメラの周回カバー範囲が{orbit_coverage_degrees:.1f}°です。より完全に一周撮影してください。"
        )
    total_reviewed = len(accepted_frames) + len(rejected_frames)
    if total_reviewed and len(rejected_frames) / total_reviewed > 0.25:
        warnings.append(
            "輪郭の25%超が除外されました。照明や背景とのコントラストを改善して撮り直してください。"
        )
    if usable_frames < 12:
        warnings.append("使用した輪郭が12枚未満のため、再現性が低い可能性があります。")
    if not mesh.is_watertight:
        warnings.append("3Dメッシュが閉じていません。メッシュ体積ではなくボクセル体積を参照してください。")

    accepted_indices = [
        frame.frame_index or index + 1 for index, frame in enumerate(accepted_frames)
    ]
    rejected_indices = [
        frame.frame_index or index + 1 for index, frame in enumerate(rejected_frames)
    ]
    result = VolumeResult(
        voxel_volume_cm3=round(voxel_volume_cm3, 2),
        mesh_volume_cm3=round(mesh_volume_cm3, 2) if mesh_volume_cm3 is not None else None,
        occupied_voxels=occupied_voxels,
        voxel_size_mm=round(voxel_size_m * 1000, 3),
        usable_frames=usable_frames,
        minimum_views=minimum_views,
        support_ratio=support_ratio,
        mesh_watertight=bool(mesh.is_watertight),
        dimensions=diagnostics,
        orbit_coverage_degrees=orbit_coverage_degrees,
        accepted_frame_indices=accepted_indices,
        rejected_frame_indices=rejected_indices,
        quality_warnings=warnings,
        warning=(
            "視体積法による試験的な推定値です。見えない凹みは埋まるため過大評価する場合があります。"
            "重要な判断には使用しないでください。"
        ),
    )
    (output_dir / "result.json").write_text(result.to_json(), encoding="utf-8")
    with (output_dir / "cross_sections.csv").open("w", encoding="utf-8", newline="") as handle:
        csv_writer = writer(handle)
        csv_writer.writerow(["height_above_board_mm", "area_cm2"])
        csv_writer.writerows(cross_section_rows(occupancy, axes, voxel_size_m))
    mask_quality = [
        asdict(frame.mask_quality)
        for frame in [*accepted_frames, *rejected_frames]
        if frame.mask_quality is not None
    ]
    (output_dir / "mask_quality.json").write_text(
        json.dumps(sorted(mask_quality, key=lambda item: item["frame_index"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def save_contact_sheet(
    image_paths: Sequence[str | Path],
    output_path: str | Path,
    mask_paths: Sequence[str | Path] | None = None,
    columns: int = 4,
    thumb_width: int = 320,
    qualities: Sequence[MaskQuality | None] | None = None,
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
        thumbnail = cv2.resize(image, (thumb_width, new_height))
        if qualities is not None and index < len(qualities) and qualities[index] is not None:
            quality = qualities[index]
            assert quality is not None
            color = (40, 170, 40) if quality.accepted else (35, 35, 220)
            label = (
                f"#{quality.frame_index:02d} {'OK' if quality.accepted else 'REJECT'} "
                f"area={quality.mask_area_ratio:.2f}"
            )
            cv2.rectangle(thumbnail, (0, 0), (thumb_width - 1, new_height - 1), color, 5)
            cv2.rectangle(thumbnail, (0, 0), (thumb_width, 31), (20, 20, 20), -1)
            cv2.putText(
                thumbnail,
                label,
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        images.append(thumbnail)
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
    "MaskQuality",
    "BoardSpec",
    "PoseFrame",
    "VolumeBounds",
    "VolumeDiagnostics",
    "VolumeResult",
    "assess_mask_quality",
    "calibrate_camera_from_board",
    "camera_orbit_coverage_degrees",
    "carve_visual_hull",
    "create_charuco_board",
    "cross_section_rows",
    "default_bounds",
    "estimate_board_poses",
    "extract_video_frames",
    "occupancy_to_mesh",
    "occupancy_diagnostics",
    "parse_frame_indices",
    "projected_bounds_mask",
    "render_charuco_board",
    "save_contact_sheet",
    "save_volume_outputs",
    "select_pose_frames",
    "segment_pose_frames_with_rembg",
]
