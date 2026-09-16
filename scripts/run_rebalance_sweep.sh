#!/usr/bin/env bash
# The synthetic set outnumbers the real one 6.6:1, which measurably traded seen-slice
# precision for unseen-slice recall. Sweep how heavily to oversample the real pairs.
#
# Selection is on VAL. Test is read once at the end, for reporting only.
#
# mini-batch is the GradCache chunk that must fit in VRAM -- NOT the contrastive batch.
# With 6 hard negatives each example is 8 texts, so a chunk of 32 left the A100 at ~28%
# utilisation. 128 keeps the same optimisation (identical --batch) and just fills the GPU.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=/home/nvidia/23BRS1361_lpdev/.hf_cache TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
GPU=${GPU:-1}
HN=${HN:-data/processed/hardnegs_r2.npz}

for R in 3 6; do
  echo "[sweep] === real-repeat=$R ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/03_train_biencoder.py \
    --out "models/biencoder_r3_rr$R" --name "biencoder_r3_rr$R" \
    --synth data/synth/filtered_p1.parquet --synth-cap-per-mid 16 \
    --hard-negs "$HN" --n-hard 6 --real-repeat $R \
    --epochs 3 --batch 128 --mini-batch 128 --lr 3e-5 2>&1 \
    | grep -E "^(train pairs|saved|--- |  overall|  unseen|  seen)"
done
echo "[sweep] done $(date -Is)"
