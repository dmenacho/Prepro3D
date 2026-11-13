"""WebDataset shards in the layout of Vernata's GrandTour LoftUp data (+ WildScenes labels).

Each sample (key = "<seq>_<cloud timestamp with '-'>"):
    coord.npy            (N, 3)  float32  points after voxel downsampling, cloud frame
    normal.npy           (N, 3)  float32  unit normals
    pose.npy             (4, 4)  float32  world_T_cloud
    downsampled_index.npy(N,)    int32    indices into the original cloud
    segment.npy          (N,)    int16    13 benchmark classes, -1 = ignore (sky, water, unlabeled)
    segment_raw.npy      (N,)    int16    raw WildScenes 3D label ids
    has_color_index.npy  (M,)    int32    indices into coord of points visible in the camera
    color.npy            (M, 3)  uint8    RGB of those points
    point_features.npy   (M, D)  float    2D teacher features of those points
    uv.npy               (M, 2)  float32  their pixel coordinates in the 2016x1512 image
    camera_T_cloud.npy   (4, 4)  float32
    meta.json                    image timestamp, |dt|, backbone, upsampler
Shards are written to "<name>.tar.tmp" and renamed when complete, so a crash never leaves a truncated shard that
a re-run would skip.
"""

import io
import json
import os
import tarfile

import numpy as np


def _npy_bytes(a):
    buf = io.BytesIO()
    np.save(buf, a, allow_pickle=False)
    return buf.getvalue()


def write_shard(path: str, samples: list):
    tmp = path + ".tmp"
    with tarfile.open(tmp, "w") as tar:
        for s in samples:
            key = s.pop("__key__")
            for name, value in s.items():
                data = json.dumps(value).encode() if name.endswith(".json") else _npy_bytes(value)
                info = tarfile.TarInfo(f"{key}.{name}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    os.replace(tmp, path)


def shard_keys(path: str) -> list:
    keys = []
    with tarfile.open(path) as tar:
        for m in tar:
            k = m.name.split(".", 1)[0]
            if not keys or keys[-1] != k:
                keys.append(k)
    return keys


def write_index(output_dir: str, splits, index_name: str):
    lines = []
    for split in splits:
        d = os.path.join(output_dir, split)
        for shard in sorted(f for f in os.listdir(d) if f.endswith(".tar")):
            lines += [f"{split}/{shard}/{k}" for k in shard_keys(os.path.join(d, shard))]
    with open(os.path.join(output_dir, index_name), "w") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines)
