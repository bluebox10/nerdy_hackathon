#!/usr/bin/env bash
# Two held-out evaluations that need a GPU, run once distillation has the GPUs free:
#   - the 7B teacher on test  -> gives us the distillation gap, measured not assumed
#   - a general 7B zero-shot  -> what you get from model size with NO task training,
#                                the honest local stand-in for "throw an API at it"
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=/home/nvidia/23BRS1361_lpdev/.hf_cache TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python

echo "[eval] waiting for distillation to finish…"
while [ ! -f results/.done_distill ] && [ ! -f results/.fail_distill ]; do sleep 30; done
echo "[eval] distillation stage done $(date -Is)"

CUDA_VISIBLE_DEVICES=2 $PY -u scripts/06c_eval_teacher.py \
  --cands results/cand_r3_test.npy --split test > logs/eval_teacher.log 2>&1 &
T=$!
CUDA_VISIBLE_DEVICES=5 $PY -u scripts/01b_baseline_llm_zeroshot.py \
  --cands results/cand_r3_test.npy --split test > logs/eval_zeroshot.log 2>&1 &
Z=$!
wait $T; wait $Z
echo "[eval] --- teacher ---";  grep -E "^(--- |  overall|  unseen|  seen)" logs/eval_teacher.log  | tail -5
echo "[eval] --- zero-shot ---"; grep -E "^(--- |  overall|  unseen|  seen)" logs/eval_zeroshot.log | tail -5
touch results/.done_evals
echo "[eval] done $(date -Is)"
