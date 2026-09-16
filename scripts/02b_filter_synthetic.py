"""Round-trip quality filter for synthetic pairs.

A generator asked to write a question for misconception M sometimes drifts to a
neighbouring topic, or writes a distractor that no student holding M would pick.
Training on that teaches the retriever noise. The filter: embed each synthetic
query with the *current* retriever and require the intended misconception to land
in the top-K. Cheap, and it removes exactly the drifted cases.

We report the rejection rate rather than hiding it -- it is the honest measure of
how good the generator actually was.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text_templates import query_text, doc_text, BGE_QUERY_INSTRUCTION
from latex_repair import repair_latex

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", nargs="+", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--keep-topk", type=int, default=50)
ap.add_argument("--min-chars", type=int, default=25)
ap.add_argument("--report", default=None)
args = ap.parse_args()

from sentence_transformers import SentenceTransformer

frames = [pd.read_parquet(p) for p in args.inputs if Path(p).exists()]
df = pd.concat(frames, ignore_index=True)
n0 = len(df)

# rows generated before the escape-order fix carry LaTeX commands that JSON turned
# into control characters; reconstruct them before anything else looks at the text
for col in ("question", "correct", "incorrect", "subject", "construct", "why_tempting"):
    df[col] = df[col].map(repair_latex)

# ---- cheap structural filters first
df = df.drop_duplicates(subset=["misconception_id", "question", "incorrect"])
n_dedup = n0 - len(df)
bad = (df["question"].str.len() < args.min_chars) | (df["correct"] == df["incorrect"]) \
      | (df["incorrect"].str.len() == 0) | (df["correct"].str.len() == 0)
df = df[~bad].reset_index(drop=True)
n_struct = int(bad.sum())

# ---- round-trip semantic filter
misc = pd.read_parquet(ROOT / "data/processed/misconceptions.parquet")
m = SentenceTransformer(args.model, device="cuda"); m.max_seq_length = 320
ids = misc["misconception_id"].to_numpy()
pos = {int(v): i for i, v in enumerate(ids)}
D = m.encode([doc_text(n) for n in misc["misconception_name"]], batch_size=256,
             normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
Q = m.encode([BGE_QUERY_INSTRUCTION + query_text(r) for _, r in df.iterrows()], batch_size=256,
             normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

sims = Q @ D.T
gold_col = np.array([pos[int(g)] for g in df["misconception_id"]])
gold_sim = sims[np.arange(len(df)), gold_col]
rank = (sims > gold_sim[:, None]).sum(1)          # 0 = gold is top-1
keep = rank < args.keep_topk
out = df[keep].copy()
out["roundtrip_rank"] = rank[keep]

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
out.to_parquet(args.out, index=False)

rep = dict(
    input_rows=int(n0), removed_duplicate=int(n_dedup), removed_structural=n_struct,
    round_trip_checked=int(len(df)), round_trip_rejected=int((~keep).sum()),
    round_trip_reject_rate=round(float((~keep).mean()), 4),
    kept=int(keep.sum()), keep_rate_overall=round(float(keep.sum() / max(n0, 1)), 4),
    keep_topk=args.keep_topk,
    misconceptions_covered=int(out["misconception_id"].nunique()),
    median_rows_per_misconception=float(out.groupby("misconception_id").size().median()),
    filter_model=args.model, out=args.out)
print(json.dumps(rep, indent=2))
if args.report:
    Path(args.report).write_text(json.dumps(rep, indent=2))
