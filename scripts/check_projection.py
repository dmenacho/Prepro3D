"""Check how WildScenes 3D clouds map into the 2D images, using only local data.

For image/cloud pairs close in time, project the labeled cloud into the image under two
hypotheses for what poses2d.csv describes, and measure how consistently 3D labels agree with
2D labels at the projected pixels (best one-to-one class assignment). A shuffled-label baseline
gives the "wrong geometry" level.
"""

import csv
import sys

import cv2
import numpy as np
import yaml
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

W = sys.argv[2] if len(sys.argv) > 2 else "WildScenes"   # path to the WildScenes folder
SEQ = sys.argv[1] if len(sys.argv) > 1 else "K-01"   # usage: python scripts/check_projection.py <seq> <WildScenes folder>


def read_poses(path):
    rows = list(csv.reader(open(path), delimiter=" ", skipinitialspace=True))[1:]
    out = {}
    for r in rows:
        x, y, z, qw, qx, qy, qz = map(float, r[2:9])
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        T[:3, 3] = [x, y, z]
        out[r[1]] = T
    return out


cal = yaml.safe_load(open(f"{W}/WildScenes2d/{SEQ}/camera_calibration.yaml"))["centre-camera"]
fx, fy, cx, cy = cal["intrinsics"]["K"]
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
D = np.array(cal["intrinsics"]["D"])
t, q = cal["extrinsics"]["translation"], cal["extrinsics"]["rotation"]
T_base_cam = np.eye(4)
T_base_cam[:3, :3] = Rotation.from_quat(q).as_matrix()
T_base_cam[:3, 3] = t

p2 = read_poses(f"{W}/WildScenes2d/{SEQ}/poses2d.csv")
p3 = read_poses(f"{W}/WildScenes3d/{SEQ}/poses3d.csv")
t2 = np.array(sorted(float(k) for k in p2))
key2 = {float(k): k for k in p2}

pairs = []
for k3 in sorted(p3, key=float):
    i = np.argmin(np.abs(t2 - float(k3)))
    if abs(t2[i] - float(k3)) < 0.05:
        pairs.append((k3, key2[t2[i]], abs(t2[i] - float(k3))))
print(f"{SEQ}: {len(pairs)} cloud/image pairs within 50 ms (of {len(p3)} clouds, {len(p2)} images)")

rng = np.random.default_rng(0)
hyp = {"H1 poses2d = camera pose": lambda T2: T2, "H2 poses2d = base pose (x extrinsics)": lambda T2: T2 @ T_base_cam}
results = {h: [] for h in hyp}
results["shuffled baseline"] = []
for k3, k2, dt in pairs[:: max(1, len(pairs) // 15)][:15]:
    pts = np.fromfile(f"{W}/WildScenes3d/{SEQ}/Clouds/{k3}.bin", dtype=np.float32).reshape(-1, 3).astype(np.float64)
    lab3 = np.fromfile(f"{W}/WildScenes3d/{SEQ}/Labels/{k3}.label", dtype=np.int32)
    lab2 = np.array(Image.open(f"{W}/WildScenes2d/{SEQ}/indexLabel/{k2.replace('.', '-')}.png"))
    H, Wd = lab2.shape
    pw = (p3[k3] @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
    for name, f in hyp.items():
        T_world_cam = f(p2[k2])
        pc = (np.linalg.inv(T_world_cam) @ np.c_[pw, np.ones(len(pw))].T).T[:, :3]
        front = pc[:, 2] > 0.5
        uv, _ = cv2.projectPoints(pc[front], np.zeros(3), np.zeros(3), K, D)
        uv = uv.reshape(-1, 2).round().astype(int)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < Wd) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        a = lab3[front][inside]
        b = lab2[uv[inside, 1], uv[inside, 0]]
        if len(a) < 1000:
            results[name].append((0.0, len(a)))
            continue
        for tag, aa in ((name, a), ("shuffled baseline", rng.permutation(a))):
            if tag == "shuffled baseline" and name != list(hyp)[1]:
                continue
            C = np.zeros((aa.max() + 1, b.max() + 1))
            np.add.at(C, (aa, b), 1)
            r, c = linear_sum_assignment(-C)
            results[tag].append((C[r, c].sum() / len(aa), len(aa)))

for name, v in results.items():
    acc = np.mean([x for x, _ in v]) if v else float("nan")
    npts = int(np.mean([n for _, n in v])) if v else 0
    print(f"  {name:40s} label agreement {100 * acc:5.1f}%  (avg {npts} projected points per pair)")
