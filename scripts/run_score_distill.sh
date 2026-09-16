#!/usr/bin/env bash
# Teacher scoring (sharded) -> merge -> distil into the 22M student we ship.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=/home/nvidia/23BRS1361_lpdev/.hf_cache TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
BI=$(cat results/selected_retriever.txt)
SYNTH=data/synth/filtered_p1.parquet
rm -f results/.done_distill results/.fail_distill data/processed/tscores_[0-9].npz

echo "[sd] === teacher scores train+synth, sharded over 3 GPUs ==="
for i in 0 1 2; do
  gpu=$(echo "1 2 5" | cut -d' ' -f$((i+1)))
  CUDA_VISIBLE_DEVICES=$gpu $PY -u scripts/06_teacher_scores.py \
    --teacher models/reranker7b --biencoder "$BI" --splits train \
    --synth "$SYNTH" --synth-cap-per-mid 4 --shard $i --num-shards 3 \
    --out data/processed/tscores_$i.npz > logs/tscore_$i.log 2>&1 &
done
wait
n=$(ls data/processed/tscores_[0-9].npz 2>/dev/null | wc -l)
if [ "$n" -ne 3 ]; then
  echo "[sd] ABORT: only $n/3 score shards produced; see logs/tscore_*.log"
  tail -5 logs/tscore_0.log; touch results/.fail_distill; exit 1
fi
$PY scripts/06b_merge_scores.py data/processed/tscores.npz 'data/processed/tscores_[0-9].npz'

echo "[sd] === distil 7B -> MiniLM-L6 (22M) ==="
CUDA_VISIBLE_DEVICES=1 $PY -u scripts/07_distill_student.py \
  --scores data/processed/tscores.npz --out models/student_ce --epochs 4 --batch 8 --lr 6e-5
if [ -f models/student_ce/model.safetensors ]; then
  touch results/.done_distill; echo "[sd] OK $(date -Is)"
else
  touch results/.fail_distill; echo "[sd] FAILED $(date -Is)"
fi
