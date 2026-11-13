#!/usr/bin/env python
"""Small quality test of the 2D teacher features: linear probe from point_features to the WildScenes 3D labels.

Trains a linear classifier on the visible points of `--train_split` samples and reports accuracy / mIoU on
`--eval_split` samples (different places, see the opt3d splits). Run it on datasets built with different
backbones/upsamplers to compare them on identical points.

Usage: python scripts/eval_features.py --output_dir /path/to/dataset [--max_samples 300]
"""

import argparse
import collections
import io
import json
import os
import tarfile

import numpy as np
import torch

CLASSES = 13


def load(output_dir, split, max_samples, max_points, rng):
    idx = [l.strip() for l in open(os.path.join(output_dir, "wildscenes_processed_index.txt")) if l.startswith(split + "/")]
    shards = sorted({e.rsplit("/", 1)[0] for e in idx})
    X, Y, n = [], [], 0
    for sh in shards:
        cur = collections.defaultdict(dict)
        with tarfile.open(os.path.join(output_dir, sh)) as tar:
            for m in tar:
                key, name = m.name.split(".", 1)
                if name in ("point_features.npy", "has_color_index.npy", "segment.npy"):
                    cur[key][name] = np.load(io.BytesIO(tar.extractfile(m).read()))
        for d in cur.values():
            f = d["point_features.npy"].astype(np.float32)
            y = d["segment.npy"][d["has_color_index.npy"]].astype(np.int64)
            keep = np.nonzero(y >= 0)[0]
            keep = rng.choice(keep, min(max_points, len(keep)), replace=False)
            X.append(f[keep]); Y.append(y[keep]); n += 1
            if n >= max_samples:
                return np.concatenate(X), np.concatenate(Y), n
    return np.concatenate(X), np.concatenate(Y), n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--train_split", default="train")
    p.add_argument("--eval_split", default="val")
    p.add_argument("--max_samples", type=int, default=300)
    p.add_argument("--max_points", type=int, default=2000, help="points per sample")
    p.add_argument("--epochs", type=int, default=30)
    a = p.parse_args()
    rng = np.random.default_rng(0)
    Xtr, Ytr, ntr = load(a.output_dir, a.train_split, a.max_samples, a.max_points, rng)
    Xev, Yev, nev = load(a.output_dir, a.eval_split, a.max_samples, a.max_points, rng)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xt, yt = torch.tensor((Xtr - mu) / sd, device=dev), torch.tensor(Ytr, device=dev)
    xe = torch.tensor((Xev - mu) / sd, device=dev)
    lin = torch.nn.Linear(Xtr.shape[1], CLASSES).to(dev)
    w = torch.bincount(yt, minlength=CLASSES).float().clamp(min=1)
    loss_fn = torch.nn.CrossEntropyLoss(weight=(w.sum() / w) ** 0.5)   # mild class balancing
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(a.epochs):
        for b in torch.randperm(len(xt), device=dev).split(8192):
            opt.zero_grad(); loss_fn(lin(xt[b]), yt[b]).backward(); opt.step()
    with torch.no_grad():
        pred = lin(xe).argmax(1).cpu().numpy()
    present = [c for c in range(CLASSES) if (Yev == c).any()]
    ious = [((pred == c) & (Yev == c)).sum() / max(((pred == c) | (Yev == c)).sum(), 1) for c in present]
    res = {"train_samples": ntr, "eval_samples": nev, "eval_points": int(len(Yev)),
           "accuracy": float((pred == Yev).mean()), "mIoU_present_classes": float(np.mean(ious)), "classes_present": len(present)}
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
