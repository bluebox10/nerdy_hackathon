"""Dump stage-1 top-K candidates per query for any split.

The reranker trains on the retriever's actual mistakes, so its negatives must be
the ones stage 1 really ranks highly -- not random misconceptions.
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text_templates import query_text, doc_text, BGE_QUERY_INSTRUCTION

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
ap.add_argument("--topk", type=int, default=50)
ap.add_argument("--tag", required=True)
args = ap.parse_args()

from sentence_transformers import SentenceTransformer
misc = pd.read_parquet(ROOT / "data/processed/misconceptions.parquet")
m = SentenceTransformer(args.model, device="cuda"); m.max_seq_length = 320
ids = misc["misconception_id"].to_numpy()
D = m.encode([doc_text(n) for n in misc["misconception_name"]], batch_size=256,
             normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

for sp in args.splits:
    df = pd.read_parquet(ROOT / f"data/processed/{sp}.parquet")
    Q = m.encode([BGE_QUERY_INSTRUCTION + query_text(r) for _, r in df.iterrows()], batch_size=256,
                 normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    order = np.argsort(-(Q @ D.T), axis=1)[:, : args.topk]
    out = ROOT / f"results/cand_{args.tag}_{sp}.npy"
    np.save(out, ids[order])
    gold = df["misconception_id"].to_numpy()
    hit = float(np.mean([gold[i] in ids[order[i]] for i in range(len(df))]))
    print(f"{sp}: {ids[order].shape} -> {out.name}   gold in top-{args.topk}: {hit:.4f}")

# nearest misconceptions by name, used to build negatives for synthetic queries
nn = np.argsort(-(D @ D.T), axis=1)[:, 1:51]
np.save(ROOT / "data/processed/misc_nn.npy", ids[nn])
print(f"misc_nn.npy: {ids[nn].shape}")
