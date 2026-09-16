#!/usr/bin/env bash
# Teacher (7B LoRA) -> sharded scoring -> distil to the 22M student we actually ship.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=/home/nvidia/23BRS1361_lpdev/.hf_cache TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python
BI=${BI:-models/biencoder_r3}
SYNTH=${SYNTH:-data/synth/filtered_p1.parquet}

echo "[rr] === dump stage-1 candidates from $BI ==="
CUDA_VISIBLE_DEVICES=1 $PY -u scripts/04b_dump_candidates.py --model "$BI" --tag r3 --topk 50 2>&1 | tail -5

echo "[rr] === train 7B LoRA teacher on 3 GPUs ==="
CUDA_VISIBLE_DEVICES=1,2,5 torchrun --nproc_per_node=3 --master_port=29571 \
  scripts/05_train_reranker.py --cand-train results/cand_r3_train.npy \
  --synth "$SYNTH" --synth-cap-per-mid 4 --group 8 --groups-per-step 2 --accum 4 \
  --epochs 1.0 --max-groups 8000 --out models/reranker7b
# Output above is deliberately NOT filtered. An earlier version piped torchrun through
# grep, so when the run died the reason was discarded and the pipeline went on to score
# with a model that did not exist -- failing 3.5 hours later, for the wrong reason.
if [ ! -f models/reranker7b/adapter_model.safetensors ]; then
  echo "[rr] ABORT: teacher checkpoint missing -- refusing to score with a model that does not exist"
  exit 1
fi

echo "[rr] === score with the teacher, sharded over 3 GPUs ==="
for i in 0 1 2; do
  gpu=$(echo "1 2 5" | cut -d' ' -f$((i+1)))
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY -u scripts/06_teacher_scores.py \
    --teacher models/reranker7b --biencoder "$BI" --splits train \
    --synth "$SYNTH" --synth-cap-per-mid 4 --shard $i --num-shards 3 \
    --out data/processed/tscores_$i.npz > logs/tscore_$i.log 2>&1 &
done
wait
n_shards=$(ls data/processed/tscores_[0-9].npz 2>/dev/null | wc -l)
if [ "$n_shards" -eq 0 ]; then
  echo "[rr] ABORT: no teacher score shards were produced; see logs/tscore_*.log"
  exit 1
fi
echo "[rr] merging $n_shards score shards"
$PY scripts/06b_merge_scores.py data/processed/tscores.npz 'data/processed/tscores_[0-9].npz'

if [ ! -f data/processed/tscores.npz ]; then
  echo "[rr] ABORT: merged teacher scores missing -- nothing to distil from"
  exit 1
fi

echo "[rr] === distil 7B -> MiniLM-L6 (22M) ==="
CUDA_VISIBLE_DEVICES=1 $PY -u scripts/07_distill_student.py \
  --scores data/processed/tscores.npz --out models/student_ce \
  --epochs 4 --batch 8 --lr 6e-5 2>&1 | grep -E "groups|top1_agree|saved"

echo "[rr] === done $(date -Is) ==="
