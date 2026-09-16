#!/usr/bin/env bash
# Waits for the synthetic-generation shards to exit, then runs the retriever chain:
#   filter -> train r2 (with synthetic) -> mine hard negatives -> train r3.
# Each stage evaluates on val+test and appends to results/metrics.json, so the
# ablation (real-only -> +synthetic -> +hard-negatives) falls out of the run log.
set -uo pipefail
# NOTE on waiting for other stages: do NOT use `pgrep -f "<script>.sh"`. Claude Code (and
# any wrapper that evals a command string) puts the full text of a monitoring command into
# ITS OWN process command line, so a diagnostic that merely MENTIONS the script name matches
# the pgrep and the waiter blocks forever. Wait on a marker file the producing stage writes.

cd "$(dirname "$0")/.."
export HF_HOME=/home/nvidia/23BRS1361_lpdev/.hf_cache
export TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
GPU=${GPU:-1}

echo "[driver] waiting for generation shards…"
while pgrep -f "02_generate_synthetic.py --shard" > /dev/null; do sleep 30; done
echo "[driver] generation done at $(date -Is)"
ls -la data/synth/synth_p1_shard*.parquet

echo "[driver] === filter ==="
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/02b_filter_synthetic.py \
  --inputs data/synth/synth_p1_shard0.parquet data/synth/synth_p1_shard1.parquet data/synth/synth_p1_shard2.parquet \
  --model models/biencoder_r1 --out data/synth/filtered_p1.parquet \
  --report results/synth_filter_report.json 2>&1 | tail -25

echo "[driver] === bi-encoder r2 (+ synthetic) ==="
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/03_train_biencoder.py \
  --out models/biencoder_r2 --name biencoder_r2_synth \
  --synth data/synth/filtered_p1.parquet --synth-cap-per-mid 16 \
  --epochs 4 --batch 384 --mini-batch 128 --lr 5e-5 2>&1 | grep -E "^(train pairs|saved|--- |  overall|  unseen|  seen)"

echo "[driver] === mine hard negatives ==="
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/04_mine_hard_negatives.py \
  --model models/biencoder_r2 --out data/processed/hardnegs_r2.npz 2>&1 | tail -8

echo "[driver] === bi-encoder r3 (+ hard negatives) ==="
CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/03_train_biencoder.py \
  --out models/biencoder_r3 --name biencoder_r3_hardneg \
  --synth data/synth/filtered_p1.parquet --synth-cap-per-mid 16 \
  --hard-negs data/processed/hardnegs_r2.npz --n-hard 6 \
  --epochs 3 --batch 128 --mini-batch 32 --lr 3e-5 2>&1 | grep -E "^(train pairs|saved|--- |  overall|  unseen|  seen)"

echo "[driver] === retriever chain complete $(date -Is) ==="
touch results/.done_retriever_chain
