"""Distil the 7B teacher into a 22M MiniLM cross-encoder -- the model we actually ship.

Loss = KL(student || teacher) over each 25-candidate group at temperature T, plus a
hard-label CE term. The student never sees the 7B at inference; it just inherits its
ranking behaviour. Output is a plain HF cross-encoder that exports cleanly to ONNX.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text_templates import query_text, doc_text

ap = argparse.ArgumentParser()
ap.add_argument("--scores", required=True)
ap.add_argument("--student", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
ap.add_argument("--out", default=str(ROOT / "models/student_ce"))
ap.add_argument("--epochs", type=float, default=4.0)
ap.add_argument("--batch", type=int, default=8, help="groups per step")
ap.add_argument("--lr", type=float, default=6e-5)
ap.add_argument("--temp", type=float, default=2.0)
ap.add_argument("--real-repeat", type=int, default=1,
                help="oversample real (non-synthetic) groups; the raw set is only 23.5% real")
ap.add_argument("--alpha", type=float, default=0.7, help="weight on the KL (distillation) term")
ap.add_argument("--max-len", type=int, default=256)
args = ap.parse_args()

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup

Z = np.load(args.scores, allow_pickle=True)
cands, tscores, gold = Z["cands"], Z["scores"], Z["gold"]
qdf = pd.read_parquet(str(args.scores).replace(".npz", "_queries.parquet"))
misc = pd.read_parquet(ROOT / "data/processed/misconceptions.parquet")
MID2NAME = dict(zip(misc["misconception_id"], misc["misconception_name"]))
K = cands.shape[1]

# The scored set is 2,647 real groups against 8,620 synthetic ones -- only 23.5% real.
# Distilling on that teaches the student to rank SYNTHETIC queries well, which is not the
# distribution it is judged on. Same failure the retriever hit; same fix. Index list, so
# oversampling costs no extra memory.
src = Z["src"].astype(str) if "src" in Z else np.array(["train"] * len(cands))
idx = list(range(len(cands)))
if args.real_repeat > 1:
    real = [i for i in idx if src[i] == "train"]
    idx = idx + real * (args.real_repeat - 1)
n_real = int((src == "train").sum())
print(f"distillation groups: {len(idx)} x {K}  "
      f"(from {len(cands)} scored: {n_real} real x{args.real_repeat}, {len(cands)-n_real} synthetic)")

tok = AutoTokenizer.from_pretrained(args.student)

class Groups(Dataset):
    def __len__(self): return len(idx)
    def __getitem__(self, j):
        i = idx[j]
        qt = query_text(qdf.iloc[i])
        docs = [doc_text(MID2NAME[int(c)]) for c in cands[i]]
        g = int(np.where(cands[i] == gold[i])[0][0]) if gold[i] in cands[i] else -1
        return qt, docs, tscores[i], g

def collate(batch):
    a, b = [], []
    for qt, docs, _, _ in batch:
        a += [qt] * K; b += docs
    enc = tok(a, b, padding=True, truncation=True, max_length=args.max_len, return_tensors="pt")
    return enc, torch.tensor(np.stack([x[2] for x in batch])), torch.tensor([x[3] for x in batch])

model = AutoModelForSequenceClassification.from_pretrained(args.student, num_labels=1,
                                                           ignore_mismatched_sizes=True).cuda()
dl = DataLoader(Groups(), batch_size=args.batch, shuffle=True, collate_fn=collate,
                num_workers=4, drop_last=True)
steps = int(len(dl) * args.epochs)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
sched = get_cosine_schedule_with_warmup(opt, int(0.06 * steps) + 1, steps)
scaler_dtype = torch.bfloat16

t0, done, ep = time.time(), 0, 0
model.train()
while done < steps:
    for enc, ts, g in dl:
        enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
        ts, g = ts.cuda(), g.cuda()
        with torch.autocast("cuda", dtype=scaler_dtype):
            logits = model(**enc).logits.view(-1, K).float()
        t_soft = F.log_softmax(ts / args.temp, dim=-1)
        s_soft = F.log_softmax(logits / args.temp, dim=-1)
        kl = F.kl_div(s_soft, t_soft, log_target=True, reduction="batchmean") * (args.temp ** 2)
        valid = g >= 0
        ce = F.cross_entropy(logits[valid], g[valid]) if valid.any() else logits.sum() * 0
        loss = args.alpha * kl + (1 - args.alpha) * ce
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        done += 1
        if done % 100 == 0:
            agree = (logits.argmax(-1) == ts.argmax(-1)).float().mean().item()
            print(f"  {done}/{steps} loss={loss.item():.4f} kl={kl.item():.4f} ce={ce.item():.4f} "
                  f"top1_agree_with_teacher={agree:.3f}  {(time.time()-t0)/60:.1f}m", flush=True)
        if done >= steps: break
    ep += 1

Path(args.out).mkdir(parents=True, exist_ok=True)
model.save_pretrained(args.out); tok.save_pretrained(args.out)
json.dump(dict(student=args.student, groups=len(idx), scored=len(cands),
               real_repeat=args.real_repeat, K=K, epochs=args.epochs,
               temp=args.temp, alpha=args.alpha, steps=steps,
               train_minutes=round((time.time()-t0)/60, 1),
               n_params_M=round(sum(p.numel() for p in model.parameters())/1e6, 1)),
          open(Path(args.out)/"distill_meta.json","w"), indent=2)
print(f"saved -> {args.out}  ({(time.time()-t0)/60:.1f} min)")
