#!/usr/bin/env bash
# Teacher training only, fully detached, output UNFILTERED.
# The previous attempt piped torchrun through grep, so when it died the reason was
# discarded and the chain marched on to score with a model that did not exist.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=/home/nvidia/23BRS1361_lpdev/.hf_cache TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=1,2,5
rm -f results/.done_teacher results/.fail_teacher
.venv/bin/torchrun --nproc_per_node=3 --master_port=29572 \
  scripts/05_train_reranker.py \
  --cand-train results/cand_r3_train.npy \
  --synth data/synth/filtered_p1.parquet --synth-cap-per-mid 4 \
  --group 8 --groups-per-step 2 --accum 4 --epochs 1.0 --max-groups 8000 \
  --out models/reranker7b
rc=$?
if [ $rc -eq 0 ] && [ -f models/reranker7b/adapter_model.safetensors ]; then
  touch results/.done_teacher
  echo "[teacher] OK $(date -Is)"
else
  touch results/.fail_teacher
  echo "[teacher] FAILED rc=$rc $(date -Is)"
fi
