"""Mine hard negatives: misconceptions the current retriever ranks highly but that
are wrong. Training round 2 against these is where most of the retrieval gain lives --
random negatives are trivially separable, near-miss negatives are not."""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text_templates import query_text, doc_text, BGE_QUERY_INSTRUCTION

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--top", type=int, default=60)
ap.add_argument("--skip-top", type=int, default=1,
                help="drop the very top hits: too likely to be near-synonyms of the gold label")
args = ap.parse_args()

from sentence_transformers import SentenceTransformer
misc = pd.read_parquet(ROOT / "data/processed/misconceptions.parquet")
train = pd.read_parquet(ROOT / "data/processed/train.parquet")
m = SentenceTransformer(args.model, device="cuda"); m.max_seq_length = 320

ids = misc["misconception_id"].to_numpy()
D = m.encode([doc_text(n) for n in misc["misconception_name"]], batch_size=256,
             normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

# Negatives are mined per *misconception*, from the queries that carry it, plus from
# the misconception's own text -- so classes with zero real queries still get negatives.
per_mid = {}
for _, r in train.iterrows():
    per_mid.setdefault(int(r["misconception_id"]), []).append(BGE_QUERY_INSTRUCTION + query_text(r))
for mid, name in zip(misc["misconception_id"], misc["misconception_name"]):
    per_mid.setdefault(int(mid), []).append(BGE_QUERY_INSTRUCTION + f"A student's error caused by: {name}")

mids, negs = [], []
keys = sorted(per_mid)
flat = [t for k in keys for t in per_mid[k]]
E = m.encode(flat, batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
off = 0
for k in keys:
    n = len(per_mid[k])
    sims = (E[off:off + n] @ D.T).mean(0)
    off += n
    order = np.argsort(-sims)
    cand = [int(ids[i]) for i in order if int(ids[i]) != k][args.skip_top:args.skip_top + args.top]
    mids.append(k); negs.append(cand)

np.savez_compressed(args.out, mids=np.array(mids),
                    negs=np.array(negs, dtype=object), allow_pickle=True)
print(f"mined hard negatives for {len(mids)} misconceptions -> {args.out}")
print("example:", misc.set_index('misconception_id').loc[mids[0],'misconception_name'])
for c in negs[0][:4]:
    print("   NEG:", misc.set_index('misconception_id').loc[c,'misconception_name'])
