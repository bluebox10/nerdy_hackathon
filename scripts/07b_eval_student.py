"""Evaluate a distilled cross-encoder on a held-out split, on GPU, in torch.

Deciding whether a student is worth shipping should not cost a 30-minute ONNX export and
CPU benchmark. This scores the same stage-1 candidates the shipped pipeline would see and
reports the same metrics, so the ship/don't-ship call can be made in about a minute.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text_templates import query_text, doc_text
from whywrong_eval import evaluate, fmt, save

ap = argparse.ArgumentParser()
ap.add_argument("--student", required=True)
ap.add_argument("--cands", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 25])
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--max-len", type=int, default=256)
ap.add_argument("--name", required=True)
args = ap.parse_args()

from transformers import AutoTokenizer, AutoModelForSequenceClassification

misc = pd.read_parquet(ROOT / "data/processed/misconceptions.parquet")
MID2NAME = dict(zip(misc["misconception_id"], misc["misconception_name"]))
df = pd.read_parquet(ROOT / f"data/processed/{args.split}.parquet").reset_index(drop=True)
full = np.load(args.cands)
K = max(args.ks)
cands = full[:, :K]

tok = AutoTokenizer.from_pretrained(args.student)
model = AutoModelForSequenceClassification.from_pretrained(args.student).cuda().eval()

A, B = [], []
for i in range(len(df)):
    qt = query_text(df.iloc[i])
    for c in cands[i]:
        A.append(qt); B.append(doc_text(MID2NAME[int(c)]))

scores = np.zeros(len(A), dtype=np.float32)
t0 = time.time()
with torch.inference_mode():
    for b in range(0, len(A), args.batch):
        enc = tok(A[b: b + args.batch], B[b: b + args.batch], padding=True, truncation=True,
                  max_length=args.max_len, return_tensors="pt").to("cuda")
        scores[b: b + enc["input_ids"].shape[0]] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
S = scores.reshape(len(df), K)
print(f"scored {len(A)} pairs in {time.time()-t0:.0f}s")

gold, unseen = df["misconception_id"].tolist(), df["unseen_in_train"].tolist()
base = evaluate(full[:, :50], gold, unseen)
print(f"  stage-1 only          MAP@25={base['overall']['map@25']:.4f} top1={base['overall']['top1_acc']:.4f}")
for k in args.ks:
    ranked = []
    for i in range(len(df)):
        head = cands[i][:k][np.argsort(-S[i, :k])]
        tail = [c for c in full[i] if c not in set(head.tolist())]
        ranked.append(np.concatenate([head, np.array(tail, dtype=head.dtype)]))
    res = evaluate(np.array(ranked), gold, unseen)
    o = res["overall"]
    d_map = o["map@25"] - base["overall"]["map@25"]
    d_top1 = o["top1_acc"] - base["overall"]["top1_acc"]
    print(f"  rerank top-{k:<2d}          MAP@25={o['map@25']:.4f} ({d_map:+.4f})  "
          f"top1={o['top1_acc']:.4f} ({d_top1:+.4f})")
    save(ROOT / "results/metrics.json", f"{args.name}_k{k}:{args.split}", res,
         extra=dict(split=args.split, student=args.student, rerank_k=k,
                    delta_map25_vs_stage1=round(d_map, 4),
                    delta_top1_vs_stage1=round(d_top1, 4)))
