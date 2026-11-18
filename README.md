# Prepro3D

Builds a [Vernata](https://github.com/rai-opensource/vernata)-style **2D→3D feature-distillation dataset** from
**WildScenes** and **SemanticKITTI**, using **SAM2** or **SAM3** image features upsampled with **AnyUp**
(the role LoftUp + DINOv2 play in Vernata's original pipelines).

For every 3D scan: pair it with its camera image (closest in time for WildScenes, same frame for KITTI), voxelize the cloud, estimate normals,
project the points into the image (with lens distortion and an occlusion test), extract the backbone's semantic
feature map, upsample it with AnyUp guided by the full-resolution image, and store one feature vector per visible
point. Output is WebDataset shards + an index file, in the layout of Vernata's GrandTour LoftUp data.

## 1. Install (once)
Needs conda and an NVIDIA GPU (driver supporting CUDA 12.8). One environment holds SAM2, SAM3 and AnyUp:
```bash
git clone git@github.com:dmenacho/Prepro3D.git
cd Prepro3D
bash install.sh            # creates the conda env "prepro3d" (pinned versions)
conda activate prepro3d
```

## 2. Model weights
* **SAM2** (`facebook/sam2.1-hiera-large`) is public: nothing to do, it downloads on first use.
* **SAM3** (`facebook/sam3`) is **gated**: request access on its Hugging Face page (Meta approves manually), then
  log in **in your own terminal** with your own *read* token (never put tokens in code, configs or chats):
  ```bash
  huggingface-cli login        # stored in ~/.cache/huggingface/token, used automatically
  ```
  Alternatively download `sam3.pt` once and pass `--checkpoint /path/to/sam3.pt`.
* **AnyUp** weights download automatically from the official release.

## 3. Data
* WildScenes from the CSIRO Data Access Portal: you need the folder `.../data/WildScenes` containing
  `WildScenes2d/` and `WildScenes3d/`.
* The official **opt3d** split files (`train.csv`, `val.csv`, `test.csv`) from the WildScenes GitHub repository
  (`data/splits/opt3d/`). They are not part of the CSIRO download and not included here.

## 4. Run (WildScenes)
```bash
export CUDA_VISIBLE_DEVICES=0      # pick one GPU
python scripts/prepare_wildscenes.py \
    --wildscenes_root /path/to/000061541v003/data/WildScenes \
    --splits_dir /path/to/opt3d \
    --output_dir /path/to/wildscenes_sam3_anyup_webdataset \
    --backbone sam3 \
    --upsampler anyup \
    --upsample_scale 0.5
python scripts/verify_dataset.py --output_dir /path/to/wildscenes_sam3_anyup_webdataset
```
`--backbone sam2` builds the SAM2 version; for `--upsampler` and `--upsample_scale` see the table below.
Re-running the same command **resumes** (finished shards are skipped; shards are written atomically).
Quick test first: add `--limit 50` (50 samples per split, spread over the whole split).

**Choose the upsampling by GPU memory** (WildScenes images; the scale changes how features are computed, not the
dataset size):

| Setting | GPU memory | Time per sample | Use when |
|---|---|---|---|
| `--upsampler anyup --upsample_scale 0.5` (default) | ~13 GB | ~0.65 s | GPU with 24 GB (recommended for most) |
| `--upsampler anyup --upsample_scale 1.0` | ~42 GB | ~6.3 s (~14 h for the full dataset) | only a GPU with 48 GB (e.g. RTX A6000); one feature per original pixel |
| `--upsampler bilinear` | lowest (SAM model only) | ~0.19 s | alternative: no upsampler, features read directly from the SAM map (fastest) |

At scale 1.0 AnyUp runs without NATTEN, whose kernel crashes at that size, which is why it is much slower.
Name the output folder after the setting you use; a folder built with one backbone/upsampler/scale cannot be extended
with another.

Expected for the full dataset (RTX A6000, default setting): **8,102 samples** (train 5,758 / val 283 / test 2,061),
**~1.5 h** and **~195 GB per backbone** (`--fp16_features`: ~105 GB).
About 23% of the clouds have no image within 50 ms and are skipped (WildScenes has fewer images than clouds).

### Main options
| Option | Default | Meaning |
|---|---|---|
| `--backbone` | (required) | `sam2` (64×64×256 stride-16 map) or `sam3` (72×72×256 map) |
| `--upsampler` | `anyup` | `anyup`, or `bilinear` (no upsampler, for comparison) |
| `--upsample_scale` | `0.5` | AnyUp output size relative to the camera image |
| `--voxel` | `0.1` | voxel size in metres before projection |
| `--max_dt` | `0.05` | WildScenes: max time difference cloud ↔ image (s) |
| `--splits` | WildScenes `train val test`, KITTI `train val` | splits to build |
| `--fp16_features` | off | store features as float16 |
| `--checkpoint` | HF download | local SAM2/SAM3 checkpoint |

## 4b. Run (SemanticKITTI)
Needs the SemanticKITTI scans + labels and the KITTI odometry **color images** (`image_2`), which can live in
**separate folders**. Each folder may be the one containing `sequences/` or its parent (`.../dataset`):

| Option | Folder with | Notes |
|---|---|---|
| `--lidar_root` | `sequences/<seq>/velodyne/*.bin` | required |
| `--labels_root` | `sequences/<seq>/labels/*.label` (+ `poses.txt`) | default: `--lidar_root` |
| `--image_root` | `sequences/<seq>/image_2/*.png` | required (`--camera image_3` for the right camera) |
| `--calib_root` | `sequences/<seq>/calib.txt` with `P2` **and** `Tr` | default: searched in the other folders. The color-image download's `calib.txt` has **no** `Tr`; use the odometry calibration download |

```bash
python scripts/prepare_semantickitti.py \
    --lidar_root /path/to/kitti/dataset \
    --labels_root /path/to/kitti/data_odometry_labels/dataset \
    --calib_root /path/to/kitti/data_odometry_calib/dataset \
    --image_root /path/to/kitti/odometry \
    --output_dir /path/to/semantickitti_sam3_anyup_s10_webdataset \
    --backbone sam3 \
    --upsampler anyup \
    --upsample_scale 1.0
python scripts/verify_dataset.py --output_dir /path/to/semantickitti_sam3_anyup_s10_webdataset
```
Splits: `train` = 00–07, 09, 10 (19,130 scans), `val` = 08 (4,071), `test` = 11–21 (20,351, no labels; add
`--splits train val test` to include it, labels stored as −1). Scans and images are synchronized (same frame).

Measured (RTX A6000, SAM3 + AnyUp, `--upsample_scale 1.0`): **~0.47 s/sample, ~12 GB GPU memory**, so **~3 h** for
train + val. KITTI images are small (1241×376), so scale 1.0 (one feature per pixel) is cheap here, unlike WildScenes.
**~11.4 MB/sample → ~265 GB** for train + val (~500 GB with test). Only **~17%** of each scan's points get features:
the camera sees the front of the car, the LiDAR 360°; the other points keep geometry and labels (`has_color_index`).

## 5. Output
```
<output_dir>/
  train/ val/ test/            <seq>_<nnnnn>.tar shards (≤ 99 samples each)
  <dataset>_processed_index.txt    "split/shard.tar/key" for every sample (wildscenes / semantickitti)
  dataset_info.json                arguments, counts, class names, feature dim
```
Per sample (key `<seq>_<cloud timestamp>` for WildScenes, `<seq>_<frame>` for SemanticKITTI):

| File | Shape | Content |
|---|---|---|
| `coord.npy` | N×3 float32 | points (cloud frame, after voxel grid) |
| `normal.npy` | N×3 float32 | unit normals |
| `pose.npy` | 4×4 float32 | world_T_cloud |
| `segment.npy` | N int16 | WildScenes: 13 benchmark classes; SemanticKITTI: 19 classes; −1 = ignore |
| `segment_raw.npy` | N int16 | raw label ids of the dataset |
| `downsampled_index.npy` | N int32 | indices into the original cloud |
| `has_color_index.npy` | M int32 | points visible in the camera (indices into `coord`) |
| `color.npy` | M×3 uint8 | their RGB |
| `point_features.npy` | M×256 | their SAM2/SAM3 + AnyUp features |
| `uv.npy`, `camera_T_cloud.npy`, `meta.json` | | pixel coords, extrinsics, image timestamp / backbone |

## 6. Check the output
`scripts/verify_dataset.py` checks index ↔ shards, required keys, shapes and finite values. Run it before sharing a
dataset.

## Notes
* KITTI images are wide (1241×376); like the backbones' own preprocessing, the pipeline stretches them to the square
  SAM input (×2.7 vertically). AnyUp then works on the original-resolution image.
* AnyUp uses NATTEN (fast) up to 1M output pixels and its standard attention above (WildScenes at scale 1.0).
* Pose convention (verified by projecting the 3D labels into the images): `poses2d.csv` is the camera pose,
  `poses3d.csv` the cloud pose. Both backbones stretch the image to a square (SAM2 1024², SAM3 1008²), as in their own preprocessing.
* SAM2's stride-4/8 levels are early-layer features without top-down fusion, so the semantic stride-16 map is used;
  SAM3's 288×288 neck level is a transposed-conv expansion of its 72×72 tokens, so the 72×72 map is used. AnyUp
  provides the fine detail from the image in both cases.
* Training with these datasets needs a Vernata dataset config for them, with the 2D-teacher feature size set to
  256 (Vernata's DINOv2-S uses 384).
* Licenses: SAM2 (Apache-2.0), SAM3 (SAM License), AnyUp (CC-BY-4.0), WildScenes (see its CSIRO license).
