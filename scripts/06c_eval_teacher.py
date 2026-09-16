"""Evaluate the 7B LoRA teacher on a held-out split.

The teacher is never deployed -- it exists to be distilled. But we need its accuracy to
state the distillation gap honestly: "the 22M student gives up N points against its
teacher and runs on a CPU" is only meaningful if N was measured, not assumed.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text_templates import query_text, doc_text
from whywrong_eval import evaluate, fmt, save

ap = argparse.ArgumentParser()
ap.add_argument("--teacher", default=str(ROOT / "models/reranker7b"))
ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--cands", required=True, help="npy of stage-1 top-50 for this split")
ap.add_argument("--split", default="test")
ap.add_argument("--topk", type=int, default=25)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--max-len", type=int, default=288)
ap.add_argument("--name", default="teacher7b")
args = ap.parse_args()

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

misc = pd.read_parquet(ROOT / "data/processed/misconceptions.parquet")
MID2NAME = dict(zip(misc["misconception_id"], misc["misconception_name"]))
df = pd.read_parquet(ROOT / f"data/processed/{args.split}.parquet").reset_index(drop=True)
cands = np.load(args.cands)[:, : args.topk]

tok = AutoTokenizer.from_pretrained(args.teacher)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForSequenceClassification.from_pretrained(
    args.base, num_labels=1, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
model.config.pad_token_id = tok.pad_token_id
model = PeftModel.from_pretrained(model, args.teacher).eval()

INSTR = "Judge whether the misconception explains the student's wrong answer.\n\n"
pairs = []
for i in range(len(df)):
    qt = query_text(df.iloc[i])
    pairs += [INSTR + qt + "\n\n" + doc_text(MID2NAME[int(c)]) for c in cands[i]]

scores = np.zeros(len(pairs), dtype=np.float32)
t0 = time.time()
with torch.inference_mode():
    for b in range(0, len(pairs), args.batch):
        chunk = pairs[b: b + args.batch]
        enc = tok(chunk, padding=True, truncation=True, max_length=args.max_len,
                  return_tensors="pt").to("cuda")
        scores[b: b + len(chunk)] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
        if (b // args.batch) % 50 == 0:
            print(f"  {b + len(chunk)}/{len(pairs)}  {(b + len(chunk)) / max(time.time() - t0, 1e-9):.0f} pair/s",
                  flush=True)
dt = time.time() - t0
S = scores.reshape(len(df), args.topk)

# reranked head, then whatever stage 1 had below it
full = np.load(args.cands)
ranked = []
for i in range(len(df)):
    head = cands[i][np.argsort(-S[i])]
    tail = [c for c in full[i] if c not in set(head.tolist())]
    ranked.append(np.concatenate([head, np.array(tail, dtype=head.dtype)]))
ranked = np.array(ranked)

res = evaluate(ranked, df["misconception_id"].tolist(), df["unseen_in_train"].tolist())
print(fmt(f"{args.name} [{args.split}]", res))
save(ROOT / "results/metrics.json", f"{args.name}:{args.split}", res, extra=dict(
    split=args.split, base=args.base, topk=args.topk, n_params_M=7620,
    gpu_seconds=round(dt, 1), gpu_ms_per_query=round(1000 * dt / len(df), 1),
    runtime="1x A100 (bf16) -- never deployed, exists to be distilled"))
