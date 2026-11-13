#!/usr/bin/env bash
# Creates the single `prepro3d` conda environment (SAM2 + SAM3 + AnyUp) with the versions this repo was tested with.
# Requires: conda, an NVIDIA GPU with a driver supporting CUDA 12.8 (driver >= 570).
#
# Usage:  bash install.sh            (env name: prepro3d)
#         bash install.sh myenvname
set -e
ENV=${1:-prepro3d}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n "$ENV" python=3.12
PY="$(conda info --base)/envs/$ENV/bin/python"   # use the env's own python (safer than `conda run`)

$PY -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
$PY -m pip install "natten==0.21.1+torch290cu128" -f https://whl.natten.org

# Official code, pinned to the commits this pipeline was tested with
SAM2_BUILD_CUDA=0 $PY -m pip install "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec"
$PY -m pip install "git+https://github.com/facebookresearch/sam3.git@f92a5efc4001"
$PY -m pip install "git+https://github.com/wimmerth/anyup.git@351807a9c428"

# numpy<2 is required by SAM3; contourpy<1.4 keeps matplotlib compatible with it; setuptools<81 still has pkg_resources (used by SAM3)
$PY -m pip install "numpy<2" "contourpy<1.4" "setuptools<81" opencv-python-headless open3d webdataset pyyaml scipy pillow einops pycocotools

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PY -c "
import torch, natten, sam2, sam3, anyup
print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| sam2, sam3, anyup OK')
"
echo "Done. Activate with: conda activate $ENV"
