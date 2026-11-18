#!/usr/bin/env python
"""Build a Vernata-style distillation dataset from SemanticKITTI with SAM2 or SAM3 features upsampled by AnyUp.

Scans/labels, camera images and calibration can live in separate folders. Example:
    python scripts/prepare_semantickitti.py \
        --lidar_root /path/to/semantickitti/dataset \
        --labels_root /path/to/semantickitti_labels/dataset \
        --image_root /path/to/kitti/odometry \
        --output_dir /path/to/out/semantickitti_sam3_anyup_webdataset \
        --backbone sam3 --upsampler anyup --upsample_scale 1.0

Splits: train = 00-07, 09, 10 | val = 08 | test = 11-21 (no labels). Re-running the same command resumes.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prepro3d.pipeline import add_common_args, run  # noqa: E402
from prepro3d.semantickitti import SemanticKITTI  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lidar_root", required=True, help="folder with sequences/<seq>/velodyne (+ labels, poses.txt)")
    p.add_argument("--labels_root", default=None, help="folder with sequences/<seq>/labels (+ poses.txt); default: lidar_root")
    p.add_argument("--image_root", required=True, help="folder with sequences/<seq>/image_2")
    p.add_argument("--calib_root", default=None, help="folder with sequences/<seq>/calib.txt containing P2 and Tr (default: searched in the other folders)")
    p.add_argument("--camera", choices=["image_2", "image_3"], default="image_2", help="left (image_2) or right color camera")
    p.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
    add_common_args(p)
    a = p.parse_args()
    run(SemanticKITTI(a.lidar_root, a.image_root, a.calib_root, a.camera, a.labels_root), a, a.splits)


if __name__ == "__main__":
    main()
