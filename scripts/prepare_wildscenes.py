#!/usr/bin/env python
"""Build a Vernata-style distillation dataset from WildScenes with SAM2 or SAM3 features upsampled by AnyUp.

Example:
    python scripts/prepare_wildscenes.py \
        --wildscenes_root /path/to/000061541v003/data/WildScenes \
        --splits_dir /path/to/WildScenes/data/splits/opt3d \
        --output_dir /path/to/out/wildscenes_sam3_anyup_webdataset \
        --backbone sam3

Re-running the same command resumes: finished shards are skipped.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prepro3d.pipeline import add_common_args, run  # noqa: E402
from prepro3d.wildscenes import WildScenes  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wildscenes_root", required=True, help="WildScenes folder containing WildScenes2d/ and WildScenes3d/")
    p.add_argument("--splits_dir", required=True, help="folder with the official opt3d train.csv / val.csv / test.csv")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--max_dt", type=float, default=0.05, help="max |t_cloud - t_image| in seconds")
    add_common_args(p)
    a = p.parse_args()
    run(WildScenes(a.wildscenes_root, a.splits_dir, max_dt=a.max_dt), a, a.splits)


if __name__ == "__main__":
    main()
