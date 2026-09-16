#!/usr/bin/env bash
# Precompute the pedagogical payload for all 2,587 misconceptions across 2 GPUs.
set -uo pipefail
# NOTE on waiting for other stages: do NOT use `pgrep -f "<script>.sh"`. Claude Code (and
# any wrapper that evals a command string) puts the full text of a monitoring command into
# ITS OWN process command line, so a diagnostic that merely MENTIONS the script name matches
# the pgrep and the waiter blocks forever. Wait on a marker file the producing stage writes.

cd "$(dirname "$0")/.."
export HF_HOME=/home/nvidia/23BRS1361_lpdev/.hf_cache TOKENIZERS_PARALLELISM=false
echo "[expl] waiting for generation shards…"
while pgrep -f "02_generate_synthetic.py --shard" > /dev/null; do sleep 30; done
for i in 0 1; do
  gpu=$(echo "2 5" | cut -d' ' -f$((i+1)))
  CUDA_VISIBLE_DEVICES=$gpu nohup .venv/bin/python -u scripts/09_precompute_explanations.py \
    --shard $i --num-shards 2 --batch 48 --max-tokens 900 > logs/expl_shard$i.log 2>&1 &
  echo "[expl] shard $i -> GPU $gpu (pid $!)"
done
wait
echo "[expl] done $(date -Is)"
touch results/.done_explanations
