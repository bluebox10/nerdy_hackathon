"""Run the 7B LoRA teacher over stage-1 candidates and cache its scores.

These soft scores (rather than hard gold labels) are what the 22M student learns
from in script 07. The teacher's *relative ordering of near-miss candidates* carries
far more signal than a one-hot label, which is why distillation recovers most of a
7B model's ranking ability at 1/300th the parameters.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text_templates import query_text, doc_text, BGE_QUERY_INSTRUCTION

ap = argparse.ArgumentParser()
ap.add_argument("--teacher", required=True)
ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--biencoder", required=True)
ap.add_argument("--splits", nargs="*", default=["train"])
ap.add_argument("--synth", nargs="*", default=[])
ap.add_argument("--synth-cap-per-mid", type=int, default=8)
ap.add_argument("--topk", type=int, default=25)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--max-len", type=int, default=288)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--num-shards", type=int, default=1)
ap.add_argument("--out", required=True)
args = ap.parse_args()

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

misc = pd.read_parquet(ROOT / "data/processed/misconceptions.parquet")
MID2NAME = dict(zip(misc["misconception_id"], misc["misconception_name"]))
ids = misc["misconception_id"].to_numpy()

rows = []
for s in args.splits:
    d = pd.read_parquet(ROOT / f"data/processed/{s}.parquet")
    d["__src"] = s
    rows.append(d[["subject", "construct", "question", "correct", "incorrect", "misconception_id", "__src"]])
if args.synth:
    fr = [pd.read_parquet(p) for p in args.synth if Path(p).exists()]
    if fr:
        sdf = pd.concat(fr, ignore_index=True).drop_duplicates(subset=["misconception_id", "question", "incorrect"])
        # groupby.apply drops the grouping column on pandas 3.x; shuffle+head does not.
        sdf = (sdf.sample(frac=1.0, random_state=0)
                  .groupby("misconception_id", sort=False)
                  .head(args.synth_cap_per_mid))
        sdf["__src"] = "synth"
        rows.append(sdf[["subject", "construct", "question", "correct", "incorrect", "misconception_id", "__src"]])
df = pd.concat(rows, ignore_index=True)
if args.num_shards > 1:
    df = df.iloc[args.shard::args.num_shards].reset_index(drop=True)
print(f"[shard {args.shard}/{args.num_shards}] scoring {len(df)} queries x top-{args.topk}", flush=True)

# ---- stage 1: candidates
bi = SentenceTransformer(args.biencoder, device="cuda"); bi.max_seq_length = 320
D = bi.encode([doc_text(n) for n in misc["misconception_name"]], batch_size=256,
              normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
Q = bi.encode([BGE_QUERY_INSTRUCTION + query_text(r) for _, r in df.iterrows()], batch_size=256,
              normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
order = np.argsort(-(Q @ D.T), axis=1)[:, : args.topk]
cands = ids[order]                                        # (n, topk)
# guarantee the gold label is in the candidate set, else there is nothing to learn
gold = df["misconception_id"].to_numpy()
for i in range(len(df)):
    if gold[i] not in cands[i]:
        cands[i, -1] = gold[i]
del bi, D, Q; torch.cuda.empty_cache()

# ---- stage 2: teacher scores
tok = AutoTokenizer.from_pretrained(args.teacher)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForSequenceClassification.from_pretrained(
    args.base, num_labels=1, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
model.config.pad_token_id = tok.pad_token_id
model = PeftModel.from_pretrained(model, args.teacher).eval()

INSTR = "Judge whether the misconception explains the student's wrong answer.\n\n"
pairs = []
for i, (_, r) in enumerate(df.iterrows()):
    qt = query_text(r)
    for c in cands[i]:
        pairs.append(INSTR + qt + "\n\n" + doc_text(MID2NAME[int(c)]))

scores = np.zeros(len(pairs), dtype=np.float32)
t0 = time.time()
with torch.inference_mode():
    for b in range(0, len(pairs), args.batch):
        chunk = pairs[b : b + args.batch]
        enc = tok(chunk, padding=True, truncation=True, max_length=args.max_len, return_tensors="pt").to("cuda")
        scores[b : b + len(chunk)] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
        if (b // args.batch) % 40 == 0:
            done = b + len(chunk)
            print(f"  {done}/{len(pairs)}  {done/max(time.time()-t0,1e-9):.0f} pair/s", flush=True)
S = scores.reshape(len(df), args.topk)

np.savez_compressed(args.out, cands=cands, scores=S, gold=gold,
                    src=df["__src"].to_numpy().astype(str))
df.to_parquet(str(args.out).replace(".npz", "_queries.parquet"), index=False)
hit = float(np.mean([gold[i] in cands[i] for i in range(len(df))]))
teacher_top1 = float(np.mean([cands[i][S[i].argmax()] == gold[i] for i in range(len(df))]))
print(json.dumps(dict(n=len(df), topk=args.topk, candidate_hit_rate=round(hit, 4),
                      teacher_top1_on_train=round(teacher_top1, 4),
                      wall_min=round((time.time()-t0)/60, 1), out=args.out), indent=2))
