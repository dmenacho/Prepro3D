#!/usr/bin/env python
"""Check a prepared dataset before sharing it: index <-> shards, required keys, shapes, dtypes, value ranges.

Usage: python scripts/verify_dataset.py --output_dir /path/to/wildscenes_sam3_anyup_webdataset [--max_shards 0]
"""

import argparse
import collections
import io
import json
import os
import random
import tarfile

import numpy as np

REQUIRED = ["coord.npy", "normal.npy", "pose.npy", "downsampled_index.npy", "segment.npy", "segment_raw.npy",
            "has_color_index.npy", "color.npy", "point_features.npy", "uv.npy", "camera_T_cloud.npy", "meta.json"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_shards", type=int, default=20, help="shards to open fully (0 = all)")
    a = p.parse_args()

    info = json.load(open(os.path.join(a.output_dir, "dataset_info.json")))
    idx = [l.strip() for l in open(os.path.join(a.output_dir, "wildscenes_processed_index.txt")) if l.strip()]
    by_shard = collections.defaultdict(set)
    for e in idx:
        split, shard, key = e.split("/")
        by_shard[f"{split}/{shard}"].add(key)
    on_disk = {f"{s}/{f}" for s in info["per_split"] for f in os.listdir(os.path.join(a.output_dir, s)) if f.endswith(".tar")}
    leftovers = [f for s in info["per_split"] for f in os.listdir(os.path.join(a.output_dir, s)) if f.endswith(".tmp")]
    problems = []
    if set(by_shard) != on_disk:
        problems.append(f"index/shard mismatch: {len(set(by_shard) ^ on_disk)} shards")
    if leftovers:
        problems.append(f"{len(leftovers)} unfinished .tmp shards (re-run prepare_wildscenes.py)")
    for split, r in info["per_split"].items():
        n = sum(1 for e in idx if e.startswith(split + "/"))
        print(f"{split}: {n} samples in index (expected {r['samples']}), {r['shards']} shards")
        if n != r["samples"]:
            problems.append(f"{split}: index has {n} samples, expected {r['samples']}")

    shards = sorted(by_shard)
    random.seed(0)
    check = shards if a.max_shards == 0 else random.sample(shards, min(a.max_shards, len(shards)))
    dim, stats = info["feature_dim"], collections.Counter()
    for sh in check:
        samples = collections.defaultdict(dict)
        with tarfile.open(os.path.join(a.output_dir, sh)) as tar:
            for m in tar:
                key, name = m.name.split(".", 1)
                raw = tar.extractfile(m).read()
                samples[key][name] = json.loads(raw) if name.endswith(".json") else np.load(io.BytesIO(raw))
        if set(samples) != by_shard[sh]:
            problems.append(f"{sh}: keys differ from index")
        for key, d in samples.items():
            miss = [r for r in REQUIRED if r not in d]
            if miss:
                problems.append(f"{sh}/{key}: missing {miss}")
                continue
            N, M = len(d["coord.npy"]), len(d["has_color_index.npy"])
            ok = (d["normal.npy"].shape == (N, 3) and len(d["segment.npy"]) == N and d["color.npy"].shape == (M, 3)
                  and d["point_features.npy"].shape == (M, dim) and np.isfinite(d["point_features.npy"]).all()
                  and (M == 0 or d["has_color_index.npy"].max() < N) and d["pose.npy"].shape == (4, 4))
            if not ok:
                problems.append(f"{sh}/{key}: bad shapes/values")
            stats["samples"] += 1; stats["points"] += N; stats["visible"] += M
    if stats["samples"]:
        print(f"checked {stats['samples']} samples in {len(check)} shards: {stats['points'] / stats['samples']:.0f} points/sample, "
              f"{100 * stats['visible'] / stats['points']:.0f}% with {dim}-dim features")
    print("OK" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems[:30]))


if __name__ == "__main__":
    main()
