"""Reading SemanticKITTI (scans + labels) together with the KITTI odometry color images, from separate folders.

Each root may point at the folder that contains `sequences/` (or `dataset/sequences/`):
    lidar_root : sequences/<seq>/velodyne/<frame>.bin  (float32 x, y, z, intensity)
    labels_root: sequences/<seq>/labels/<frame>.label  (uint32; semantic id = value & 0xFFFF; 00-10 only)
                 sequences/<seq>/poses.txt             (optional; default labels_root = lidar_root)
    image_root : sequences/<seq>/image_2/<frame>.png   (left color camera; image_3 = right)
    calib_root : sequences/<seq>/calib.txt             (needs P2/P3 and Tr; the color-image download has no Tr)
KITTI odometry images are rectified and synchronized with the scans (same frame index), so no distortion model
and no timestamp matching are needed. Calibration: Tr maps velodyne -> rectified camera 0, P2 = K [I | t] maps
rectified camera 0 -> image_2 pixels. poses.txt holds world_T_cam0 per frame.
"""

import glob
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

from prepro3d.wildscenes import Camera

CLASS_NAMES = ["car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist", "motorcyclist",
               "road", "parking", "sidewalk", "other-ground", "building", "fence", "vegetation", "trunk",
               "terrain", "pole", "traffic-sign"]
# Standard SemanticKITTI learning map (semantic-kitti.yaml): raw id -> 19 classes, -1 = ignore
LEARNING_MAP = {0: -1, 1: -1, 10: 0, 11: 1, 13: 4, 15: 2, 16: 4, 18: 3, 20: 4, 30: 5, 31: 6, 32: 7, 40: 8, 44: 9,
                48: 10, 49: 11, 50: 12, 51: 13, 52: -1, 60: 8, 70: 14, 71: 15, 72: 16, 80: 17, 81: 18, 99: -1,
                252: 0, 253: 6, 254: 5, 255: 7, 256: 4, 257: 4, 258: 3, 259: 4}
SPLITS = {"train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"], "val": ["08"],
          "test": [f"{i:02d}" for i in range(11, 22)]}


def _sequences_dir(root: str) -> str:
    for d in (os.path.join(root, "sequences"), os.path.join(root, "dataset", "sequences"), root):
        if os.path.isdir(d) and any(os.path.isdir(os.path.join(d, s)) for s in ("00", "08", "11")):
            return d
    raise FileNotFoundError(f"no sequences/ folder found under {root}")


def _read_calib(path: str) -> dict:
    out = {}
    for line in open(path):
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = np.array(v.split(), dtype=np.float64)
    return out


@dataclass
class Sample:
    split: str
    seq: str
    frame: str           # "000123"

    @property
    def key(self) -> str:
        return f"{self.seq}_{self.frame}"

    @property
    def order(self) -> int:
        return int(self.frame)


class SemanticKITTI:
    name, class_names, split_names = "semantickitti", CLASS_NAMES, ["train", "val", "test"]

    def __init__(self, lidar_root: str, image_root: str, calib_root: str = None, camera: str = "image_2",
                 labels_root: str = None):
        self.lidar_dir, self.image_dir = _sequences_dir(lidar_root), _sequences_dir(image_root)
        self.labels_dir = _sequences_dir(labels_root) if labels_root else self.lidar_dir
        self.calib_dir = _sequences_dir(calib_root) if calib_root else self.image_dir
        self.camera_name = camera
        self._cams, self._T, self._poses = {}, {}, {}
        lut = np.full(1 << 16, -1, dtype=np.int16)
        for k, v in LEARNING_MAP.items():
            lut[k] = v
        self._lut = lut

    def _load_sequence(self, seq: str):
        if seq in self._cams:
            return
        # The calib.txt of the KITTI color-image download has only P0-P3; Tr (velodyne -> camera) is in the
        # calibration download. Use the first calib.txt that has both.
        cal, tried = None, []
        for root in dict.fromkeys((self.calib_dir, self.image_dir, self.lidar_dir, self.labels_dir)):
            f = os.path.join(root, seq, "calib.txt")
            if os.path.exists(f):
                c = _read_calib(f)
                tried.append(f)
                if "Tr" in c and "P2" in c:
                    cal = c
                    break
        if cal is None:
            raise FileNotFoundError(f"sequence {seq}: no calib.txt with both P2 and Tr (checked {tried or 'none found'}); "
                                    "pass --calib_root pointing to the KITTI odometry calibration download")
        P = cal["P2" if self.camera_name == "image_2" else "P3"].reshape(3, 4)
        Tr = np.vstack([cal["Tr"].reshape(3, 4), [0, 0, 0, 1]])
        K = P[:, :3]
        cam_T_cam0 = np.eye(4)
        cam_T_cam0[:3, 3] = np.linalg.solve(K, P[:, 3])          # P = K [I | t]
        first = sorted(glob.glob(os.path.join(self.image_dir, seq, self.camera_name, "*.png")))[0]
        w, h = Image.open(first).size                             # image size differs slightly per sequence
        self._cams[seq] = Camera(K=K, D=np.zeros(5), width=w, height=h)
        self._T[seq] = (cam_T_cam0 @ Tr, Tr)
        poses = None
        for root in (self.labels_dir, self.lidar_dir, self.calib_dir, self.image_dir):
            f = os.path.join(root, seq, "poses.txt")
            if os.path.exists(f):
                p = np.loadtxt(f).reshape(-1, 3, 4)
                poses = np.concatenate([p, np.tile([[[0, 0, 0, 1]]], (len(p), 1, 1))], axis=1)
                break
        self._poses[seq] = poses

    def samples(self, split: str):
        out, unpaired = [], 0
        for seq in SPLITS[split]:
            scans = sorted(glob.glob(os.path.join(self.lidar_dir, seq, "velodyne", "*.bin")))
            if not scans:
                continue
            self._load_sequence(seq)
            for f in scans:
                frame = os.path.basename(f)[:-4]
                if os.path.exists(os.path.join(self.image_dir, seq, self.camera_name, f"{frame}.png")):
                    out.append(Sample(split, seq, frame))
                else:
                    unpaired += 1
        return out, unpaired

    def load_cloud(self, s: Sample):
        pts = np.fromfile(os.path.join(self.lidar_dir, s.seq, "velodyne", f"{s.frame}.bin"), dtype=np.float32).reshape(-1, 4)[:, :3]
        lf = os.path.join(self.labels_dir, s.seq, "labels", f"{s.frame}.label")
        raw = (np.fromfile(lf, dtype=np.uint32) & 0xFFFF).astype(np.int32) if os.path.exists(lf) else np.zeros(len(pts), np.int32)
        assert len(raw) == len(pts), s.key
        return pts, raw

    def map_labels(self, raw: np.ndarray) -> np.ndarray:
        return self._lut[raw]

    def load_image(self, s: Sample) -> np.ndarray:
        return np.array(Image.open(os.path.join(self.image_dir, s.seq, self.camera_name, f"{s.frame}.png")).convert("RGB"))

    def camera(self, s: Sample) -> Camera:
        self._load_sequence(s.seq)
        return self._cams[s.seq]

    def camera_T_cloud(self, s: Sample) -> np.ndarray:
        self._load_sequence(s.seq)
        return self._T[s.seq][0]

    def world_T_cloud(self, s: Sample) -> np.ndarray:
        self._load_sequence(s.seq)
        poses, Tr = self._poses[s.seq], self._T[s.seq][1]
        return poses[int(s.frame)] @ Tr if poses is not None else np.eye(4)

    def meta(self, s: Sample) -> dict:
        self._load_sequence(s.seq)
        return {"frame": s.frame, "camera": self.camera_name, "pose_available": self._poses[s.seq] is not None}
