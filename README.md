# Prepro3D

Builds a [Vernata](https://github.com/rai-opensource/vernata)-style **2D→3D feature-distillation dataset** from
**WildScenes**, using **SAM2** or **SAM3** image features upsampled with **AnyUp**
(the role LoftUp + DINOv2 play in Vernata's original pipelines).

For every WildScenes 3D cloud: pair it with the camera image closest in time, voxelize the cloud, estimate normals,
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

## 4. Run
```bash
export CUDA_VISIBLE_DEVICES=0      # pick one GPU
python scripts/prepare_wildscenes.py \
    --wildscenes_root /path/to/000061541v003/data/WildScenes \
    --splits_dir /path/to/opt3d \
    --output_dir /path/to/wildscenes_sam3_anyup_webdataset \
    --backbone sam3                # or: sam2
python scripts/verify_dataset.py --output_dir /path/to/wildscenes_sam3_anyup_webdataset
```
Re-running the same command **resumes** (finished shards are skipped; shards are written atomically).
Quick test first: add `--limit 50` (50 samples per split, spread over the whole split).

Expected for the full dataset (RTX A6000): **8,102 samples** (train 5,758 / val 283 / test 2,061),
**~1.5 h** and **~175 GB per backbone** (`--fp16_features` halves the size), ~13 GB GPU memory.
About 23% of the clouds have no image within 50 ms and are skipped (WildScenes has fewer images than clouds).

### Main options
| Option | Default | Meaning |
|---|---|---|
| `--backbone` | (required) | `sam2` (64×64×256 stride-16 map) or `sam3` (72×72×256 map) |
| `--upsampler` | `anyup` | `anyup`, or `bilinear` (no upsampler, for comparison) |
| `--upsample_scale` | `0.5` | AnyUp output size relative to the 2016×1512 image |
| `--voxel` | `0.1` | voxel size in metres before projection |
| `--max_dt` | `0.05` | max time difference cloud ↔ image (s) |
| `--splits` | `train val test` | splits to build |
| `--fp16_features` | off | store features as float16 |
| `--checkpoint` | HF download | local SAM2/SAM3 checkpoint |

## 5. Output
```
<output_dir>/
  train/ val/ test/            <seq>_<nnnnn>.tar shards (≤ 99 samples each)
  wildscenes_processed_index.txt   "split/shard.tar/key" for every sample
  dataset_info.json                arguments, counts, class names, feature dim
```
Per sample (key `<seq>_<cloud timestamp>`):

| File | Shape | Content |
|---|---|---|
| `coord.npy` | N×3 float32 | points (cloud frame, after voxel grid) |
| `normal.npy` | N×3 float32 | unit normals |
| `pose.npy` | 4×4 float32 | world_T_cloud |
| `segment.npy` | N int16 | 13 WildScenes benchmark classes, −1 = ignore |
| `segment_raw.npy` | N int16 | raw WildScenes label ids |
| `downsampled_index.npy` | N int32 | indices into the original cloud |
| `has_color_index.npy` | M int32 | points visible in the camera (indices into `coord`) |
| `color.npy` | M×3 uint8 | their RGB |
| `point_features.npy` | M×256 | their SAM2/SAM3 + AnyUp features |
| `uv.npy`, `camera_T_cloud.npy`, `meta.json` | | pixel coords, extrinsics, image timestamp / backbone |

## 6. Checks and tools
* `scripts/verify_dataset.py` — index ↔ shards, keys, shapes, finite values. Run before sharing a dataset.
* `scripts/eval_features.py` — linear probe from `point_features` to the 3D labels (train → val);
  compare datasets built with different `--backbone` / `--upsampler`.
* `scripts/check_projection.py <seq> <WildScenes folder>` — sanity check of the 2D↔3D geometry: 3D labels projected
  into the images agree with the 2D labels (~70% vs ~46% for shuffled labels).

## Notes
* Pose convention (verified with `check_projection.py`): `poses2d.csv` is the camera pose, `poses3d.csv` the
  cloud pose. Both backbones stretch the image to a square (SAM2 1024², SAM3 1008²), as in their own preprocessing.
* SAM2's stride-4/8 levels are early-layer features without top-down fusion, so the semantic stride-16 map is used;
  SAM3's 288×288 neck level is a transposed-conv expansion of its 72×72 tokens, so the 72×72 map is used. AnyUp
  provides the fine detail from the image in both cases.
* Training with these datasets needs a Vernata dataset config for them, with the 2D-teacher feature size set to
  256 (Vernata's DINOv2-S uses 384).
* Licenses: SAM2 (Apache-2.0), SAM3 (SAM License), AnyUp (CC-BY-4.0), WildScenes (see its CSIRO license).
