#!/usr/bin/env bash
# Tail of the pipeline:
#   1. wait for the retriever chain + explanation shards
#   2. sweep how heavily to oversample real data against synthetic
#      (17.5k synthetic vs 2.6k real measurably traded seen-slice precision for
#       unseen-slice recall, so the ratio is worth a sweep rather than a guess)
#   3. select the retriever on VAL -- test is read once, for reporting only
#   4. run the reranker chain (7B teacher -> sharded scoring -> distil) on the winner
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=/home/nvidia/23BRS1361_lpdev/.hf_cache TOKENIZERS_PARALLELISM=false
PY=.venv/bin/python

echo "[tail] waiting for retriever chain + explanations…"
# Marker files, not pgrep -- see the NOTE in run_after_generation.sh for why pgrep on a
# script name livelocks the moment anything else mentions that name.
while [ ! -f results/.done_retriever_chain ] || [ ! -f results/.done_explanations ]; do
  sleep 20
done
echo "[tail] upstream done $(date -Is)"

bash scripts/run_rebalance_sweep.sh

echo "[tail] === selecting retriever on VAL ==="
BEST=$($PY - <<'PY'
import json, pathlib, sys
rows = {r["name"]: r for r in json.loads(pathlib.Path("results/metrics.json").read_text())}
cands = [("models/biencoder_r3_rr6", "biencoder_r3_rr6:val"),
         ("models/biencoder_r3_rr3", "biencoder_r3_rr3:val"),
         ("models/biencoder_r3",     "biencoder_r3_hardneg:val"),
         ("models/biencoder_r2",     "biencoder_r2_synth:val"),
         ("models/biencoder_r1",     "biencoder_r1_realonly:val")]
scored = []
for path, key in cands:
    r = rows.get(key)
    if not r or not pathlib.Path(path).exists():
        continue
    o = r["metrics"]["overall"]
    # rank on val MAP@25; recall@25 breaks ties because it hard-caps what the reranker can reach
    scored.append((o["map@25"], o["recall@25"], path, key))
scored.sort(reverse=True)
for m, rc, p, k in scored:
    print(f"#   {k:32s} val MAP@25={m:.4f}  R@25={rc:.4f}", file=sys.stderr)
print(scored[0][2] if scored else "models/biencoder_r1")
PY
)
echo "[tail] selected retriever: $BEST"
export BI="$BEST"
echo "$BEST" > results/selected_retriever.txt

bash scripts/run_reranker.sh
echo "[tail] reranker chain finished $(date -Is)"
