#!/usr/bin/env bash
# Waits for the retriever chain AND the explanation shards, then runs the reranker
# (teacher -> score -> distil) and rebuilds every shipped artifact from scratch.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=/home/nvidia/23BRS1361_lpdev/.hf_cache TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python

echo "[rr] waiting for retriever chain + explanation shards…"
while pgrep -f "run_after_generation.sh" >/dev/null || pgrep -f "run_explanations.sh" >/dev/null; do sleep 30; done
echo "[rr] upstream complete $(date -Is)"

if [ ! -d models/biencoder_r3 ]; then
  echo "[rr] biencoder_r3 missing -- falling back to r2, then r1"
  BI=models/biencoder_r2; [ -d "$BI" ] || BI=models/biencoder_r1
else
  BI=models/biencoder_r3
fi
export BI
echo "[rr] retriever for the reranker stage: $BI"

bash scripts/run_reranker.sh
echo "[rr] reranker chain finished $(date -Is)"
