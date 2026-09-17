"""Stage 1 of the retriever: contrastive fine-tune of bge-small-en-v1.5 (33M).

Uses CachedMultipleNegativesRankingLoss (GradCache) so the effective contrastive
batch (the number of in-batch negatives that drives
retrieval quality) can be pushed far past what fits in activation memory.
Optionally mixes in synthetic long-tail pairs and mined hard negatives.
"""
import argparse, json, math, os, random, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text_templates import query_text, doc_text, BGE_QUERY_INSTRUCTION
from whywrong_eval import evaluate, fmt, save

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="BAAI/bge-small-en-v1.5")
ap.add_argument("--init-from", default=None, help="resume from a previous round's checkpoint")
ap.add_argument("--out", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--synth", nargs="*", default=[], help="synthetic parquet files to mix in")
ap.add_argument("--synth-cap-per-mid", type=int, default=24)
ap.add_argument("--real-repeat", type=int, default=1,
                help="oversample the real pairs so synthetic data does not swamp them")
ap.add_argument("--hard-negs", default=None, help="npz with mined hard negatives")
ap.add_argument("--n-hard", type=int, default=8)
ap.add_argument("--epochs", type=float, default=3.0)
ap.add_argument("--batch", type=int, default=512, help="contrastive batch (in-batch negatives)")
ap.add_argument("--mini-batch", type=int, default=64, help="GradCache chunk that must fit in VRAM")
ap.add_argument("--lr", type=float, default=4e-5)
ap.add_argument("--warmup", type=float, default=0.1)
ap.add_argument("--max-seq", type=int, default=320)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments)
from sentence_transformers.losses import (CachedMultipleNegativesRankingLoss,
                                          MultipleNegativesRankingLoss)
from datasets import Dataset

misc = pd.read_parquet(ROOT / "data/processed/misconceptions.parquet")
MID2NAME = dict(zip(misc["misconception_id"], misc["misconception_name"]))
train_df = pd.read_parquet(ROOT / "data/processed/train.parquet")
val_df = pd.read_parquet(ROOT / "data/processed/val.parquet")
test_df = pd.read_parquet(ROOT / "data/processed/test.parquet")

def q(row):
    return BGE_QUERY_INSTRUCTION + query_text(row)

# ---------------- build training pairs ----------------
anchors, positives, mids = [], [], []
# Real pairs are repeated `--real-repeat` times. With ~17.5k synthetic against 2.6k real,
# a 1x weighting lets generated data dominate the contrastive objective: measured, that
# lifted the unseen slice but cost the seen slice ~9 MAP points. Oversampling the real
# pairs buys the long-tail gain without paying for it on the classes we have data for.
for _ in range(max(1, args.real_repeat)):
    for _, r in train_df.iterrows():
        anchors.append(q(r)); positives.append(doc_text(MID2NAME[r["misconception_id"]]))
        mids.append(int(r["misconception_id"]))
n_real = len(anchors)

n_synth = 0
if args.synth:
    frames = [pd.read_parquet(p) for p in args.synth if Path(p).exists()]
    if frames:
        s = pd.concat(frames, ignore_index=True)
        s = s.drop_duplicates(subset=["misconception_id", "question", "incorrect"])
        if args.synth_cap_per_mid:
            # NB: groupby.apply drops the grouping column on pandas 3.x: shuffle+head
            # gives the same random per-misconception cap and keeps misconception_id.
            s = (s.sample(frac=1.0, random_state=args.seed)
                   .groupby("misconception_id", sort=False)
                   .head(args.synth_cap_per_mid))
        for _, r in s.iterrows():
            anchors.append(q(r)); positives.append(doc_text(r["misconception_name"]))
            mids.append(int(r["misconception_id"]))
        n_synth = len(s)

# every misconception also gets a name->name identity anchor so the 1,498 classes
# with zero examples are at least placed in the embedding space by their own text
for mid, name in MID2NAME.items():
    anchors.append(BGE_QUERY_INSTRUCTION + f"A student's error caused by: {name}")
    positives.append(doc_text(name)); mids.append(int(mid))

cols = {"anchor": anchors, "positive": positives}
loss_kind = "mnrl"
if args.hard_negs and Path(args.hard_negs).exists():
    hn = np.load(args.hard_negs, allow_pickle=True)
    neg_map = {int(k): list(v) for k, v in zip(hn["mids"], hn["negs"])}
    for j in range(args.n_hard):
        col = []
        for mid in mids:
            cands = [c for c in neg_map.get(int(mid), []) if int(c) != int(mid)]
            col.append(doc_text(MID2NAME[int(random.choice(cands))]) if cands
                       else doc_text(MID2NAME[int(random.choice(list(MID2NAME)))]))
        cols[f"negative_{j}"] = col
    loss_kind = f"mnrl+{args.n_hard}hard"

ds = Dataset.from_dict(cols).shuffle(seed=args.seed)
print(f"train pairs: {len(ds)}  (real={n_real} synth={n_synth} identity={len(MID2NAME)})  loss={loss_kind}")

model = SentenceTransformer(args.init_from or args.base, device="cuda")
model.max_seq_length = args.max_seq
loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch, scale=20.0)

targs = SentenceTransformerTrainingArguments(
    output_dir=str(ROOT / "models/_hf_trainer" / args.name),
    num_train_epochs=args.epochs, per_device_train_batch_size=args.batch,
    learning_rate=args.lr, warmup_ratio=args.warmup, bf16=True,
    logging_steps=10, save_strategy="no", report_to=[], seed=args.seed,
    dataloader_num_workers=4, remove_unused_columns=False)

t0 = time.time()
SentenceTransformerTrainer(model=model, args=targs, train_dataset=ds, loss=loss).train()
train_s = time.time() - t0
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
model.save(args.out)
print(f"saved -> {args.out}  ({train_s/60:.1f} min)")

# ---------------- evaluate ----------------
D = model.encode([doc_text(n) for n in misc["misconception_name"]], batch_size=256,
                 normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
ids = misc["misconception_id"].to_numpy()
for split, df in [("val", val_df), ("test", test_df)]:
    Q = model.encode([q(r) for _, r in df.iterrows()], batch_size=256,
                     normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    order = np.argsort(-(Q @ D.T), axis=1)
    res = evaluate(ids[order[:, :25]], df["misconception_id"].tolist(), df["unseen_in_train"].tolist())
    print(fmt(f"{args.name} [{split}]", res))
    save(ROOT / "results/metrics.json", f"{args.name}:{split}", res, extra=dict(
        split=split, base=args.base, loss=loss_kind, n_train_pairs=len(ds),
        n_real=n_real, n_synth=n_synth, batch=args.batch, epochs=args.epochs,
        train_minutes=round(train_s / 60, 2),
        n_params_M=round(sum(p.numel() for p in model.parameters()) / 1e6, 1)))
    np.save(ROOT / f"results/cand_{args.name}_{split}.npy", ids[order[:, :50]])
