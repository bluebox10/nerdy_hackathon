"""Baseline 1/2: off-the-shelf embedding retrieval, no fine-tuning. The floor."""
import sys, time, json, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from whywrong_eval import evaluate, fmt, save
from text_templates import query_text, doc_text, BGE_QUERY_INSTRUCTION

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
ap.add_argument("--name", default=None)
ap.add_argument("--split", default="test")
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--no-instruction", action="store_true")
args = ap.parse_args()
name = args.name or f"offtheshelf:{args.model.split('/')[-1]}"

from sentence_transformers import SentenceTransformer

misc = pd.read_parquet(ROOT / "data/processed/misconceptions.parquet")
df = pd.read_parquet(ROOT / f"data/processed/{args.split}.parquet")

model = SentenceTransformer(args.model, device="cuda" if torch.cuda.is_available() else "cpu")
model.max_seq_length = 384

instr = "" if args.no_instruction else BGE_QUERY_INSTRUCTION
q_texts = [instr + query_text(r) for _, r in df.iterrows()]
d_texts = [doc_text(m) for m in misc["misconception_name"]]

t0 = time.time()
D = model.encode(d_texts, batch_size=args.batch, normalize_embeddings=True,
                 show_progress_bar=False, convert_to_numpy=True)
t_doc = time.time() - t0
t0 = time.time()
Q = model.encode(q_texts, batch_size=args.batch, normalize_embeddings=True,
                 show_progress_bar=False, convert_to_numpy=True)
t_q = time.time() - t0

sims = Q @ D.T                                    # (n_q, 2587)
top = np.argsort(-sims, axis=1)[:, :25]
ids = misc["misconception_id"].to_numpy()
rankings = ids[top]

res = evaluate(rankings, df["misconception_id"].tolist(), df["unseen_in_train"].tolist())
print(fmt(name, res))
save(ROOT / "results/metrics.json", name, res, extra=dict(
    split=args.split, model=args.model, dim=int(D.shape[1]),
    n_params_M=round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
    encode_docs_s=round(t_doc, 2), encode_queries_s=round(t_q, 2),
    ms_per_query_gpu_batched=round(1000 * t_q / len(q_texts), 2)))
np.save(ROOT / f"results/cand_{name.replace(':','_')}_{args.split}.npy", ids[np.argsort(-sims, axis=1)[:, :50]])
