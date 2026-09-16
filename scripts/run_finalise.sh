#!/usr/bin/env bash
# Rebuild every shipped artifact from the trained models, in dependency order.
# Safe to re-run: each step overwrites its own outputs.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=/home/nvidia/23BRS1361_lpdev/.hf_cache
export CUDA_VISIBLE_DEVICES=""          # the shipped path is CPU-only; prove it here too
export OMP_NUM_THREADS=1
PY=.venv/bin/python

BI=$(cat results/selected_retriever.txt 2>/dev/null || echo models/biencoder_r1)
CE_ARG=""
[ -d models/student_ce ] && CE_ARG="--cross-encoder models/student_ce"
echo "[final] retriever=$BI  cross-encoder=${CE_ARG:-<none, bi-only>}"

echo "[final] === 1/6 export INT8 ONNX ==="
# Unfiltered, and gated: a failed export previously left the PREVIOUS run's encoder in
# place and the benchmark happily measured that instead, reporting the wrong model's score.
$PY scripts/08_export_onnx.py --biencoder "$BI" $CE_ARG
rc=$?
if [ $rc -ne 0 ] || [ ! -f serve/artifacts/bi_encoder/model_quantized.onnx ]; then
  echo "[final] ABORT: ONNX export failed (rc=$rc) -- refusing to benchmark a stale model"
  exit 1
fi
if [ -n "$CE_ARG" ] && [ ! -f serve/artifacts/cross_encoder/model_quantized.onnx ]; then
  echo "[final] ABORT: cross-encoder export missing while a cross-encoder was requested"
  exit 1
fi
echo "[final] export ok: $(du -sh serve/artifacts/bi_encoder serve/artifacts/cross_encoder 2>/dev/null | tr '\n' ' ')"

echo "[final] === 2/6 build bundle (explanations + demo pools) ==="
# Retry file FIRST: the builder keeps the first record it sees per misconception, and the
# retry holds regenerated versions of the 19 whose original JSON failed to parse and fell
# back to boilerplate (one of them is #172, the fractions case the demo opens on).
$PY scripts/10_build_bundle.py --explanations \
  data/processed/explanations_retry.jsonl 'data/processed/explanations_shard*.jsonl' 2>&1 | tail -4

echo "[final] === 3/6 benchmark the SHIPPED artifact ==="
$PY scripts/11_build_benchmark.py 2>&1 | grep -vE "^  rerank " | tail -22

echo "[final] === 4/6 analyse failures ==="
$PY scripts/11b_analyse_failures.py 2>&1 | head -60

echo "[final] === 5/6 fill submission copy ==="
$PY scripts/12_fill_submission.py 2>&1 | tail -20

echo "[final] === 6/6 render the project page ==="
$PY scripts/13_build_artifact.py 2>&1 | tail -3

echo "[final] restarting API so it serves the new bundle"
pkill -f "uvicorn app:app" 2>/dev/null || true
echo "[final] done $(date -Is) -- keepalive will bring the API back within 30s"
