"""Point cloud geometry: voxel downsampling, normals, camera projection with distortion, occlusion test."""

import cv2
import numpy as np
import open3d as o3d


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    """Indices of the first point in each voxel (keeps original order); voxel <= 0 keeps all points."""
    if voxel <= 0:
        return np.arange(len(points))
    _, keep = np.unique(np.floor(points / voxel).astype(np.int64), axis=0, return_index=True)
    return np.sort(keep)


def estimate_normals(points: np.ndarray, radius: float, max_nn: int) -> np.ndarray:
    """Unit normals oriented towards the cloud origin (the sensor side)."""
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd.orient_normals_towards_camera_location(np.zeros(3))
    return np.asarray(pcd.normals, dtype=np.float32)


def project(points: np.ndarray, camera_T_cloud: np.ndarray, K, D, width, height, min_depth=0.5):
    """Project cloud points into the image. Returns (indices of points inside the image, uv (float pixels), depth)."""
    pc = points.astype(np.float64) @ camera_T_cloud[:3, :3].T + camera_T_cloud[:3, 3]
    front = np.nonzero(pc[:, 2] > min_depth)[0]
    if len(front) == 0:
        return front, np.zeros((0, 2)), np.zeros(0)
    uv, _ = cv2.projectPoints(pc[front], np.zeros(3), np.zeros(3), K, D)
    uv = uv.reshape(-1, 2)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1)
    return front[inside], uv[inside], pc[front[inside], 2]


def visible_mask(uv: np.ndarray, depth: np.ndarray, cell: int = 8, rel_tol: float = 0.05, abs_tol: float = 0.3):
    """Z-buffer test on image cells: keep points close to the nearest depth in their cell.

    WildScenes clouds are accumulated submaps, so points hidden behind closer surfaces still project into the
    image; they must not receive the feature of the surface in front of them.
    """
    if len(uv) == 0:
        return np.zeros(0, dtype=bool)
    cu, cv_ = (uv[:, 0] // cell).astype(np.int64), (uv[:, 1] // cell).astype(np.int64)
    cid = cv_ * (cu.max() + 1) + cu
    zmin = np.full(cid.max() + 1, np.inf)
    np.minimum.at(zmin, cid, depth)
    return depth <= np.maximum(zmin[cid] * (1 + rel_tol), zmin[cid] + abs_tol)
