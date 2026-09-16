"""Baseline 3: a general instruct LLM reranking stage-1 candidates, with NO task training.

This is the honest local stand-in for "throw a frontier API at it". Same 7B backbone as
our teacher, same candidate set, same scoring position -- the only difference from
script 05 is that this model has never seen the task. It answers the question the panel
will actually ask: how much of our gain came from training rather than from model size?

Scoring is monoT5-style: read the logit of " Yes" vs " No" at the next-token position.
No generation, so 882 queries x 25 candidates is one batched forward pass sweep.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text_templates import query_text, doc_text
from whywrong_eval import evaluate, fmt, save

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--cands", required=True, help="npy of stage-1 top-50 candidates for the split")
ap.add_argument("--split", default="test")
ap.add_argument("--topk", type=int, default=25)
ap.add_argument("--batch", type=int, default=48)
ap.add_argument("--max-len", type=int, default=320)
ap.add_argument("--few-shot", action="store_true")
ap.add_argument("--name", default=None)
args = ap.parse_args()
name = args.name or ("llm7b_fewshot_zeroshot_rerank" if args.few_shot else "llm7b_zeroshot_rerank")

from transformers import AutoTokenizer, AutoModelForCausalLM

misc = pd.read_parquet(ROOT / "data/processed/misconceptions.parquet")
MID2NAME = dict(zip(misc["misconception_id"], misc["misconception_name"]))
df = pd.read_parquet(ROOT / f"data/processed/{args.split}.parquet").reset_index(drop=True)
cands = np.load(args.cands)[:, : args.topk]

tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    args.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda").eval()

YES = tok(" Yes", add_special_tokens=False)["input_ids"][-1]
NO = tok(" No", add_special_tokens=False)["input_ids"][-1]

SHOTS = ("\nExample: Question 'What is 7/12 - 3/12?', correct '4/12', student chose '4/24'. "
         "Misconception: 'When subtracting fractions, subtracts the numerators and denominators'. "
         "Answer: Yes\n"
         "Example: same question. Misconception: 'Believes there are 100 degrees in a full turn'. "
         "Answer: No\n") if args.few_shot else ""


def prompt(row, mid):
    user = (f"A student answered a maths question incorrectly.\n\n{query_text(row)}\n\n"
            f"Candidate misconception: {MID2NAME[int(mid)]}\n{SHOTS}\n"
            "Does this misconception explain the student's specific wrong answer? Answer Yes or No.")
    return tok.apply_chat_template([{"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)

pairs = [prompt(df.iloc[i], c) for i in range(len(df)) for c in cands[i]]
scores = np.zeros(len(pairs), dtype=np.float32)
t0 = time.time()
with torch.inference_mode():
    for b in range(0, len(pairs), args.batch):
        chunk = pairs[b: b + args.batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_len).to("cuda")
        logits = model(**enc).logits[:, -1, :].float()
        scores[b: b + len(chunk)] = (logits[:, YES] - logits[:, NO]).cpu().numpy()
        if (b // args.batch) % 50 == 0:
            print(f"  {b+len(chunk)}/{len(pairs)}  {(b+len(chunk))/max(time.time()-t0,1e-9):.0f} pair/s", flush=True)
dt = time.time() - t0
S = scores.reshape(len(df), args.topk)

ranked = np.array([cands[i][np.argsort(-S[i])] for i in range(len(df))])
res = evaluate(ranked, df["misconception_id"].tolist(), df["unseen_in_train"].tolist())
print(fmt(name, res))
save(ROOT / "results/metrics.json", f"{name}:{args.split}", res, extra=dict(
    split=args.split, model=args.model, topk=args.topk, few_shot=args.few_shot,
    n_forward_passes=len(pairs), gpu_seconds=round(dt, 1),
    pairs_per_s=round(len(pairs) / dt, 1),
    gpu_ms_per_query=round(1000 * dt / len(df), 1),
    note="no task-specific training; reranks the same stage-1 candidates as the trained models"))
