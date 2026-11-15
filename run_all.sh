#!/bin/bash
# Prepro3D: build the WildScenes datasets for SAM2 then SAM3, one after another, then verify each.
#
#   ./run_all.sh --check    only check paths, dataset layout, GPU, disk and conda (takes seconds)
#   ./run_all.sh            check, quick test (3 val samples per backbone), then the full runs
#
# The dataset can live on the network drive (/mnt/...) or in your home folder: list both under
# DATA_LOCATIONS and the script uses the first one that holds a complete WildScenes copy.
# Re-running resumes: finished shards are skipped (shards are written atomically).

set -u

# ---- edit these -------------------------------------------------------------
REPO=/path/to/Prepro3D
DATA_LOCATIONS=(                                       # searched in this order (up to 5 folders deep)
  /path/to/datasets/wildscenes
)
ROOT=""                                                # leave empty to auto-detect, or set the folder
                                                       # that directly contains WildScenes2d/ and WildScenes3d/
SPLITS=/path/to/WildScenes/data/splits/opt3d           # train.csv / val.csv / test.csv
OUT=/path/to/Prepro3D/logs_out_1.0
SAM3_CKPT=/path/to/weights/sam3.pt                     # local SAM3; SAM2 downloads from HuggingFace

CONDA_ENV=prepro3d
GPU=0                                                  # which GPU (CUDA_VISIBLE_DEVICES)
UPSAMPLER=anyup                                        # anyup | bilinear
SCALE=1.0                                              # 1.0 needs ~42 GB GPU memory and ~6.3 s/sample;
                                                       # 0.5 needs ~13 GB and ~0.65 s/sample
FP16=0                                                 # 1 = --fp16_features (~105 GB instead of ~195 GB per backbone)
BACKBONES="sam2 sam3"

MAX_RETRIES=6                                          # if the network drive drops out mid-run:
RETRY_WAIT=600                                         # wait this many seconds, then resume
# -----------------------------------------------------------------------------

CHECK_ONLY=0; [ "${1:-}" = "--check" ] && CHECK_ONLY=1
LOGDIR=$OUT/logs
export CUDA_VISIBLE_DEVICES=$GPU
stamp() { date '+%F %T'; }
die() { echo "ERROR: $*"; exit 1; }

# ---- 1. preflight checks ----------------------------------------------------
echo "[$(stamp)] checking setup"

# dataset: the folder whose WildScenes3d/<seq>/poses3d.csv exist, next to WildScenes2d/
find_root() {
  local base cand n
  for base in "${DATA_LOCATIONS[@]}"; do
    if [ ! -r "$base" ]; then
      echo "  $base: not found or not readable (not mounted?)" >&2; continue
    fi
    # look for WildScenes3d folders without descending into the (large) image/cloud folders
    while IFS= read -r cand; do
      cand=$(dirname "$cand")
      n=$(ls "$cand"/WildScenes3d/*/poses3d.csv 2>/dev/null | wc -l)
      if [ -d "$cand/WildScenes2d" ] && [ "$n" -gt 0 ]; then
        echo "  $base: dataset found at $cand ($n sequences with poses3d.csv)" >&2
        echo "$cand"; return 0
      fi
      echo "  $base: $cand has WildScenes3d/ but no WildScenes2d/ or no */poses3d.csv" >&2
    done < <(find "$base" -maxdepth 5 \( -name WildScenes2d -prune \) -o \( -type d -name WildScenes3d -print -prune \) 2>/dev/null)
    echo "  $base: no WildScenes3d/ folder found within 5 levels" >&2
  done
  return 1
}

if [ -z "$ROOT" ]; then
  ROOT=$(find_root) || die "no complete WildScenes copy in: ${DATA_LOCATIONS[*]}
       Run: find <location> -maxdepth 6 -name poses3d.csv | head   to see where the files are."
fi
[ -d "$ROOT/WildScenes3d" ] && [ -d "$ROOT/WildScenes2d" ] \
  || die "$ROOT must contain WildScenes2d/ and WildScenes3d/"
echo "  dataset: $ROOT"

# every 3D sequence needs its poses3d.csv, otherwise the loader crashes at start-up
missing=()
for d in "$ROOT"/WildScenes3d/*/; do
  seq=$(basename "$d")
  [ -f "$d/poses3d.csv" ] || missing+=("$seq")
done
n_seq=$(ls -d "$ROOT"/WildScenes3d/*/ 2>/dev/null | wc -l)
[ ${#missing[@]} -eq 0 ] || die "sequences without WildScenes3d/<seq>/poses3d.csv: ${missing[*]}
       (incomplete download/extraction?)  ls $ROOT/WildScenes3d/${missing[0]}"
echo "  sequences: $n_seq, all with poses3d.csv"

for s in train val test; do
  [ -f "$SPLITS/$s.csv" ] || die "split file not found: $SPLITS/$s.csv"
done
echo "  splits: $SPLITS"

cd "$REPO" 2>/dev/null || die "cannot cd to $REPO"
[ -f scripts/prepare_wildscenes.py ] || die "$REPO has no scripts/prepare_wildscenes.py"
[ -f "$SAM3_CKPT" ] || die "checkpoint not found: $SAM3_CKPT"
echo "  repo + SAM3 checkpoint: ok"

# GPU memory: scale 1.0 needs ~42 GB
gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | head -1 | tr -dc 0-9)
[ -n "$gpu_mem" ] || die "cannot read GPU $GPU with nvidia-smi"
echo "  GPU $GPU: $(( gpu_mem / 1024 )) GB"
if [ "$UPSAMPLER" = anyup ] && awk "BEGIN{exit !($SCALE > 0.5)}" && [ "$gpu_mem" -lt 45000 ]; then
  die "SCALE=$SCALE needs ~42 GB GPU memory; GPU $GPU has $(( gpu_mem / 1024 )) GB. Use SCALE=0.5."
fi

# output folder named after the setting (a folder can't be extended with another backbone/upsampler/scale)
tag() { if [ "$UPSAMPLER" = anyup ]; then echo "${1}_anyup_s${SCALE}"; else echo "${1}_bilinear"; fi; }
fp16_args=(); [ "$FP16" = 1 ] && fp16_args=(--fp16_features)

# disk space: README estimate per full backbone, minus what earlier runs already wrote
mkdir -p "$LOGDIR" || die "cannot create $LOGDIR"
per_bb=$([ "$FP16" = 1 ] && echo 105 || echo 195)
need=0
for bb in $BACKBONES; do
  have=$(du -s -BG "$OUT/wildscenes_$(tag "$bb")" 2>/dev/null | cut -f1 | tr -d G); have=${have:-0}
  need=$(( need + (per_bb > have ? per_bb - have : 0) ))
done
free=$(df -BG --output=avail "$OUT" | tail -1 | tr -dc 0-9)
echo "  disk: ~${need} GB still needed, ${free} GB free on $OUT (a per-user quota is not visible here)"
[ "$free" -ge $(( need + 10 )) ] || die "not enough disk space (set FP16=1 to roughly halve it, or move OUT)"

# conda env (a fresh tmux shell may not have it active)
if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$CONDA_ENV" \
    || die "cannot activate conda env $CONDA_ENV"
fi
echo "  conda env: $CONDA_ENV"

if awk "BEGIN{exit !($SCALE > 0.5)}" && [ "$UPSAMPLER" = anyup ]; then
  echo "  note: SCALE=$SCALE takes ~14 h per backbone (README); 0.5 takes ~1.5 h"
fi
echo "[$(stamp)] all checks passed"
[ $CHECK_ONLY = 1 ] && exit 0

# ---- helpers ------------------------------------------------------------------
prepare() {   # prepare <backbone> <output_dir> <log> [extra args...]
  local bb=$1 out=$2 log=$3; shift 3
  local ckpt_args=(); [ "$bb" = sam3 ] && ckpt_args=(--checkpoint "$SAM3_CKPT")
  python -u scripts/prepare_wildscenes.py \
      --wildscenes_root "$ROOT" --splits_dir "$SPLITS" --output_dir "$out" \
      --backbone "$bb" --upsampler "$UPSAMPLER" --upsample_scale "$SCALE" \
      "${ckpt_args[@]}" "${fp16_args[@]}" "$@" >> "$log" 2>&1
}

data_readable() { ls "$ROOT/WildScenes3d" > /dev/null 2>&1 && [ -r "$ROOT/WildScenes3d/$(ls "$ROOT/WildScenes3d" | head -1)/poses3d.csv" ]; }

# run prepare; if it fails because the dataset became unreadable (network drive dropped out),
# wait and resume (finished shards are kept), up to MAX_RETRIES times
prepare_with_retry() {
  local attempt=0
  while true; do
    prepare "$@" && return 0
    data_readable && return 1                     # data fine -> a real error, don't retry
    attempt=$(( attempt + 1 ))
    [ $attempt -le "$MAX_RETRIES" ] || { echo "[$(stamp)] dataset still unreadable after $MAX_RETRIES retries"; return 1; }
    echo "[$(stamp)] dataset at $ROOT unreadable (drive dropped?) -> retry $attempt/$MAX_RETRIES in $(( RETRY_WAIT / 60 )) min"
    sleep "$RETRY_WAIT"
  done
}

# ---- 2. quick test ------------------------------------------------------------
OK=""
for bb in $BACKBONES; do
  echo "[$(stamp)] TEST  $bb"
  if prepare "$bb" "$OUT/_test_$(tag "$bb")" "$LOGDIR/test_$bb.log" --splits val --limit 3; then
    echo "[$(stamp)] TEST OK  $bb"; OK="$OK $bb"
  else
    echo "[$(stamp)] TEST FAILED $bb -> skipping it; last lines of $LOGDIR/test_$bb.log:"
    tail -n 5 "$LOGDIR/test_$bb.log" | sed 's/^/    /'
  fi
done

# ---- 3. full runs + verification ----------------------------------------------
for bb in $OK; do
  out="$OUT/wildscenes_$(tag "$bb")"
  echo "[$(stamp)] START $bb -> $out  (log: $LOGDIR/$bb.log)"
  if ! prepare_with_retry "$bb" "$out" "$LOGDIR/$bb.log"; then
    echo "[$(stamp)] FAILED $bb; last lines of $LOGDIR/$bb.log:"
    tail -n 5 "$LOGDIR/$bb.log" | sed 's/^/    /'
    continue
  fi
  echo "[$(stamp)] DONE  $bb, verifying"
  if python -u scripts/verify_dataset.py --output_dir "$out" > "$LOGDIR/verify_$bb.log" 2>&1; then
    echo "[$(stamp)] VERIFY OK  $bb"
  else
    echo "[$(stamp)] VERIFY FAILED $bb, see $LOGDIR/verify_$bb.log"
  fi
done

echo "[$(stamp)] finished:$OK"