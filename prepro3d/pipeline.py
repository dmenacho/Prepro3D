"""Dataset-independent part: cloud -> voxel grid -> normals -> projection -> occlusion test -> 2D teacher features
-> WebDataset shards + index. Used by scripts/prepare_wildscenes.py and scripts/prepare_semantickitti.py.

A dataset object must provide:
    name, class_names, split_names
    samples(split) -> (list of samples, number of scans without an image)   sample: .key, .seq, .order
    load_cloud(s) -> (points N x 3 float32, raw labels N int)
    map_labels(raw) -> N int16 training labels (-1 = ignore)
    load_image(s) -> H x W x 3 uint8
    camera(s) -> object with K (3x3), D (distortion, OpenCV), width, height
    camera_T_cloud(s), world_T_cloud(s) -> 4x4
    meta(s) -> json-serialisable dict
"""

import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from prepro3d import geometry
from prepro3d.featurizers import Upsampler, build_featurizer
from prepro3d.writer import write_index, write_shard


def add_common_args(p):
    p.add_argument("--output_dir", required=True)
    p.add_argument("--backbone", choices=["sam2", "sam3"], required=True)
    p.add_argument("--checkpoint", default=None, help="local checkpoint (.pt); default: download from HuggingFace")
    p.add_argument("--upsampler", choices=["anyup", "bilinear"], default="anyup")
    p.add_argument("--upsample_scale", type=float, default=0.5, help="AnyUp output size relative to the camera image")
    p.add_argument("--q_chunk_size", type=int, default=None, help="AnyUp query chunking (lower GPU memory)")
    p.add_argument("--voxel", type=float, default=0.1, help="voxel size in metres (Vernata uses 0.1 for GrandTour)")
    p.add_argument("--normal_radius", type=float, default=0.3)
    p.add_argument("--normal_max_nn", type=int, default=30)
    p.add_argument("--no_occlusion_test", action="store_true")
    p.add_argument("--fp16_features", action="store_true", help="store point_features as float16 (half the size)")
    p.add_argument("--max_samples_per_tar", type=int, default=99)
    p.add_argument("--limit", type=int, default=None, help="only N samples per split, evenly spread (quick tests)")
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--random_weights", action="store_true", help="(tests only) random backbone weights")


def _keep_as_is(sample):   # top-level so worker processes can pickle it
    return sample


class _Prepared(Dataset):
    """CPU part, run in DataLoader workers."""

    def __init__(self, ds, samples, args):
        self.ds, self.samples, self.args = ds, samples, args

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        a, s = self.args, self.samples[i]
        pts, raw = self.ds.load_cloud(s)
        keep = geometry.voxel_downsample(pts, a.voxel)
        pts, raw = pts[keep], raw[keep]
        cam = self.ds.camera(s)
        T = self.ds.camera_T_cloud(s)
        idx, uv, depth = geometry.project(pts, T, cam.K, cam.D, cam.width, cam.height)
        if not a.no_occlusion_test:
            vis = geometry.visible_mask(uv, depth)
            idx, uv = idx[vis], uv[vis]
        return {
            "sample": s, "keep": keep.astype(np.int32), "coord": pts.astype(np.float32),
            "normal": geometry.estimate_normals(pts, a.normal_radius, a.normal_max_nn),
            "raw": raw, "idx": idx.astype(np.int32), "uv": uv.astype(np.float32),
            "image": self.ds.load_image(s), "T": T.astype(np.float32),
        }


def run(ds, a, splits):
    info_path = os.path.join(a.output_dir, "dataset_info.json")
    index_name = f"{ds.name}_processed_index.txt"
    previous = json.load(open(info_path)) if os.path.exists(info_path) else {}
    setting = (ds.name, a.backbone, a.upsampler, a.upsample_scale)
    if previous and (previous.get("dataset", ds.name), previous.get("backbone"), previous.get("upsampler"),
                     previous.get("upsample_scale", a.upsample_scale)) != setting:
        sys.exit(f"{a.output_dir} was built with {previous.get('dataset', '?')}/{previous.get('backbone')}/"
                 f"{previous.get('upsampler')}/scale {previous.get('upsample_scale')}; use another --output_dir")
    device = "cuda"
    featurizer = build_featurizer(a.backbone, device, checkpoint=a.checkpoint, random_weights=a.random_weights)
    upsampler = Upsampler(a.upsampler, device, scale=a.upsample_scale, q_chunk_size=a.q_chunk_size)
    os.makedirs(a.output_dir, exist_ok=True)
    report = {}

    for split in splits:
        samples, unpaired = ds.samples(split)
        if not samples:
            print(f"[{split}] no samples found, skipped", flush=True)
            continue
        if a.limit and a.limit < len(samples):   # spread over the whole split (all sequences / places)
            samples = [samples[i] for i in np.linspace(0, len(samples) - 1, a.limit).round().astype(int)]
        os.makedirs(os.path.join(a.output_dir, split), exist_ok=True)
        # deterministic shards per sequence, so a re-run skips exactly the finished ones
        shards, todo = [], []
        for seq in sorted({s.seq for s in samples}):
            seq_samples = sorted((s for s in samples if s.seq == seq), key=lambda s: s.order)
            for j in range(0, len(seq_samples), a.max_samples_per_tar):
                path = os.path.join(a.output_dir, split, f"{seq}_{j // a.max_samples_per_tar:05d}.tar")
                shards.append(path)
                if not os.path.exists(path):
                    todo.append((path, seq_samples[j: j + a.max_samples_per_tar]))
        n_todo = sum(len(c) for _, c in todo)
        print(f"[{split}] {len(samples)} samples with an image ({unpaired} scans without one), "
              f"{len(shards)} shards, {len(shards) - len(todo)} already done, {n_todo} samples to process", flush=True)
        report[split] = {"samples": len(samples), "unpaired_clouds": unpaired, "shards": len(shards)}
        if not todo:
            continue
        flat = [s for _, chunk in todo for s in chunk]
        loader = DataLoader(_Prepared(ds, flat, a), batch_size=None, collate_fn=_keep_as_is,
                            num_workers=a.num_workers, prefetch_factor=2 if a.num_workers else None)
        it, t0, done = iter(loader), time.time(), 0
        for path, chunk in todo:
            out = []
            for _ in chunk:
                d = next(it)
                s, img = d["sample"], d["image"]
                h, w = img.shape[:2]
                feats = upsampler(img, featurizer(img), d["uv"], w, h)
                uvi = np.clip(np.round(d["uv"]).astype(int), 0, [w - 1, h - 1])
                out.append({
                    "__key__": s.key,
                    "coord.npy": d["coord"], "normal.npy": d["normal"],
                    "pose.npy": ds.world_T_cloud(s).astype(np.float32),
                    "downsampled_index.npy": d["keep"],
                    "segment.npy": ds.map_labels(d["raw"]), "segment_raw.npy": d["raw"].astype(np.int16),
                    "has_color_index.npy": d["idx"],
                    "color.npy": img[uvi[:, 1], uvi[:, 0]].astype(np.uint8),
                    "point_features.npy": feats.astype(np.float16 if a.fp16_features else np.float32),
                    "uv.npy": d["uv"], "camera_T_cloud.npy": d["T"],
                    "meta.json": {**ds.meta(s), "backbone": a.backbone, "upsampler": a.upsampler},
                })
            write_shard(path, out)
            done += len(chunk)
            el = time.time() - t0
            print(f"[{split}] {done}/{n_todo} samples | {el / done:.2f} s/sample | ETA {(n_todo - done) * el / done / 3600:.1f} h"
                  f" | {os.path.basename(path)}", flush=True)

    # Index and info cover every split built so far, so splits can be built in separate runs.
    per_split = {**previous.get("per_split", {}), **report}
    n = write_index(a.output_dir, sorted(per_split), index_name)
    info = {"dataset": ds.name, "backbone": a.backbone, "upsampler": a.upsampler, "upsample_scale": a.upsample_scale,
            "feature_dim": featurizer.dim, "per_split": per_split, "index_file": index_name, "index_entries": n,
            "class_names": ds.class_names, "args_last_run": vars(a), "torch": torch.__version__}
    json.dump(info, open(info_path, "w"), indent=2)
    print(f"Done. Index: {n} samples -> {os.path.join(a.output_dir, index_name)}")
