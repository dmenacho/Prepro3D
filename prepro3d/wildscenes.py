"""Reading WildScenes (CSIRO release): 3D clouds/labels/poses, 2D images/poses, camera calibration, splits.

Layout expected under `root` (the `.../data/WildScenes` folder of the CSIRO download):
    WildScenes3d/<seq>/Clouds/<ts>.bin      float32 x, y, z (cloud frame)
    WildScenes3d/<seq>/Labels/<ts>.label    int32 raw class id per point
    WildScenes3d/<seq>/poses3d.csv          world_T_cloud per timestamp
    WildScenes2d/<seq>/image/<ts>.png       2016 x 1512 RGB (timestamp with '-' instead of '.')
    WildScenes2d/<seq>/poses2d.csv          world_T_camera per timestamp (camera optical frame)
    WildScenes2d/<seq>/camera_calibration.yaml
Splits: the official `opt3d` csv files (train.csv / val.csv / test.csv) from the WildScenes repository.

Pose convention (verified by projecting 3D labels into the images, see scripts/check_projection.py):
poses2d.csv is directly the camera pose; poses3d.csv is the pose of the cloud's local frame.
"""

import csv
import os
from dataclasses import dataclass

import numpy as np
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation

SEQUENCES = ["K-01", "K-03", "V-01", "V-02", "V-03"]

# Raw 3D label id -> 13 benchmark classes (sky=10 and water=14 ignored), as in the official benchmark.
CLASS_NAMES = ["bush", "dirt", "fence", "grass", "gravel", "log", "mud", "object",
               "other-terrain", "rock", "structure", "tree-foliage", "tree-trunk"]
RAW_TO_TRAIN = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9,
                10: -1, 11: 10, 12: 11, 13: 12, 14: -1}


def map_labels(raw: np.ndarray) -> np.ndarray:
    lut = np.full(256, -1, dtype=np.int16)
    for k, v in RAW_TO_TRAIN.items():
        lut[k] = v
    return lut[np.clip(raw, 0, 255)]


def read_poses(path: str) -> dict:
    """{timestamp string: 4x4 world_T_frame} from a WildScenes poses csv (space separated)."""
    rows = list(csv.reader(open(path), delimiter=" ", skipinitialspace=True))[1:]
    poses = {}
    for r in rows:
        x, y, z, qw, qx, qy, qz = map(float, r[2:9])
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        T[:3, 3] = [x, y, z]
        poses[r[1]] = T
    return poses


@dataclass
class Camera:
    K: np.ndarray        # 3x3
    D: np.ndarray        # OpenCV distortion (k1, k2, p1, p2, k3)
    width: int
    height: int


def read_camera(path: str, width: int = 2016, height: int = 1512) -> Camera:
    cal = yaml.safe_load(open(path))["centre-camera"]["intrinsics"]
    fx, fy, cx, cy = cal["K"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return Camera(K=K, D=np.array(cal["D"], dtype=np.float64), width=width, height=height)


@dataclass
class Sample:
    split: str
    seq: str
    cloud_ts: str        # e.g. "1624325374.450389996"
    image_ts: str
    dt: float            # |cloud_ts - image_ts| in seconds

    @property
    def key(self) -> str:   # WebDataset keys must not contain '.'
        return f"{self.seq}_{self.cloud_ts.replace('.', '-')}"


class WildScenes:
    def __init__(self, root: str, splits_dir: str, max_dt: float = 0.05):
        self.root, self.splits_dir, self.max_dt = root, splits_dir, max_dt
        self.poses3d, self.poses2d, self.cameras, self._img_ts = {}, {}, {}, {}
        for seq in SEQUENCES:
            self.poses3d[seq] = read_poses(f"{root}/WildScenes3d/{seq}/poses3d.csv")
            self.poses2d[seq] = read_poses(f"{root}/WildScenes2d/{seq}/poses2d.csv")
            self.cameras[seq] = read_camera(f"{root}/WildScenes2d/{seq}/camera_calibration.yaml")
            keys = sorted(self.poses2d[seq], key=float)
            self._img_ts[seq] = (np.array([float(k) for k in keys]), keys)

    def samples(self, split: str):
        """Cloud/image pairs of a split. Clouds without an image within max_dt are reported, not returned."""
        out, unpaired = [], 0
        with open(os.path.join(self.splits_dir, f"{split}.csv")) as f:
            for row in csv.DictReader(f):
                _, seq, _, fname = row["lidar_path"].split("/")
                ts = fname[: -len(".bin")]
                t, keys = self._img_ts[seq]
                i = int(np.argmin(np.abs(t - float(ts))))
                dt = abs(t[i] - float(ts))
                if dt > self.max_dt or ts not in self.poses3d[seq]:
                    unpaired += 1
                    continue
                out.append(Sample(split, seq, ts, keys[i], dt))
        return out, unpaired

    def load_cloud(self, s: Sample):
        pts = np.fromfile(f"{self.root}/WildScenes3d/{s.seq}/Clouds/{s.cloud_ts}.bin", dtype=np.float32).reshape(-1, 3)
        raw = np.fromfile(f"{self.root}/WildScenes3d/{s.seq}/Labels/{s.cloud_ts}.label", dtype=np.int32)
        assert len(raw) == len(pts), s.key
        return pts, raw

    def load_image(self, s: Sample) -> np.ndarray:
        return np.array(Image.open(f"{self.root}/WildScenes2d/{s.seq}/image/{s.image_ts.replace('.', '-')}.png").convert("RGB"))

    def camera_T_cloud(self, s: Sample) -> np.ndarray:
        return np.linalg.inv(self.poses2d[s.seq][s.image_ts]) @ self.poses3d[s.seq][s.cloud_ts]
