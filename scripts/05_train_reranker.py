"""Stage 2: the heavy cross-encoder teacher: Qwen2.5-7B + LoRA, scalar score head.

Trained listwise: each example is one query with its gold misconception plus K hard
negatives mined by the stage-1 retriever, optimised with softmax cross-entropy over
the group. Listwise beats pointwise BCE here because the task is *ranking* 25 near-
miss candidates, not deciding absolute relevance. This model is never deployed --
it exists to be distilled (script 06).

Launch: torchrun --nproc_per_node=3 scripts/05_train_reranker.py ...
"""
import argparse, json, math, os, random, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from text_templates import query_text, doc_text

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--out", default=str(ROOT / "models/reranker7b"))
ap.add_argument("--cand-train", required=True, help="npy of stage-1 top-50 candidates for TRAIN queries")
ap.add_argument("--synth", nargs="*", default=[])
ap.add_argument("--synth-cap-per-mid", type=int, default=8)
ap.add_argument("--group", type=int, default=8, help="1 positive + group-1 negatives")
ap.add_argument("--groups-per-step", type=int, default=2)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--epochs", type=float, default=2.0)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--lora-r", type=int, default=32)
ap.add_argument("--max-len", type=int, default=288)
ap.add_argument("--max-groups", type=int, default=0)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

LOCAL = int(os.environ.get("LOCAL_RANK", 0))
WORLD = int(os.environ.get("WORLD_SIZE", 1))
IS_DDP = WORLD > 1
if IS_DDP:
    dist.init_process_group("nccl"); torch.cuda.set_device(LOCAL)
DEV = torch.device(f"cuda:{LOCAL}")
RANK0 = (not IS_DDP) or dist.get_rank() == 0
def log(*a):
    if RANK0: print(*a, flush=True)

random.seed(args.seed + LOCAL); torch.manual_seed(args.seed)

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

misc = pd.read_parquet(ROOT / "data/processed/misconceptions.parquet")
MID2NAME = dict(zip(misc["misconception_id"], misc["misconception_name"]))
ALL_MIDS = list(MID2NAME)
train_df = pd.read_parquet(ROOT / "data/processed/train.parquet").reset_index(drop=True)
cand_train = np.load(args.cand_train)          # (n_train, 50)

# ---- assemble listwise groups: real queries use retriever-mined negatives,
# ---- synthetic queries use nearest-neighbour negatives of their misconception.
examples = []
for i, r in train_df.iterrows():
    gold = int(r["misconception_id"])
    negs = [int(c) for c in cand_train[i] if int(c) != gold]
    if len(negs) >= args.group - 1:
        examples.append(dict(row=dict(r), gold=gold, negs=negs))

if args.synth:
    frames = [pd.read_parquet(p) for p in args.synth if Path(p).exists()]
    if frames:
        s = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["misconception_id", "question", "incorrect"])
        # groupby.apply drops the grouping column on pandas 3.x; shuffle+head does not.
        s = (s.sample(frac=1.0, random_state=args.seed)
               .groupby("misconception_id", sort=False)
               .head(args.synth_cap_per_mid))
        gold_pos = {int(m): i for i, m in enumerate(misc["misconception_id"])}
        # negatives for synthetic rows: text-nearest misconceptions, precomputed cheaply below
        nn_path = ROOT / "data/processed/misc_nn.npy"
        nn = np.load(nn_path) if nn_path.exists() else None
        for _, r in s.iterrows():
            gold = int(r["misconception_id"])
            if nn is not None:
                negs = [int(c) for c in nn[gold_pos[gold]] if int(c) != gold]
            else:
                negs = random.sample(ALL_MIDS, 50)
            if len(negs) >= args.group - 1:
                examples.append(dict(row=dict(r), gold=gold, negs=negs))

random.Random(args.seed).shuffle(examples)
if args.max_groups:
    examples = examples[: args.max_groups]
log(f"listwise groups: {len(examples)}  group_size={args.group}")

tok = AutoTokenizer.from_pretrained(args.base)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

INSTR = ("Judge whether the misconception explains the student's wrong answer.\n\n")

class Groups(Dataset):
    def __init__(self, ex): self.ex = ex
    def __len__(self): return len(self.ex)
    def __getitem__(self, i):
        e = self.ex[i]
        negs = random.sample(e["negs"][:40], args.group - 1) if len(e["negs"]) >= args.group - 1 \
               else random.choices(e["negs"], k=args.group - 1)
        mids = [e["gold"]] + negs
        qt = query_text(e["row"])
        return [INSTR + qt + "\n\n" + doc_text(MID2NAME[int(m)]) for m in mids]

def collate(batch):
    flat = [t for g in batch for t in g]
    enc = tok(flat, padding=True, truncation=True, max_length=args.max_len, return_tensors="pt")
    return enc, len(batch)

model = AutoModelForSequenceClassification.from_pretrained(
    args.base, num_labels=1, dtype=torch.bfloat16, attn_implementation="sdpa")
model.config.pad_token_id = tok.pad_token_id
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(
    r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05, bias="none",
    task_type="SEQ_CLS", modules_to_save=["score"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.to(DEV)
if RANK0: model.print_trainable_parameters()
if IS_DDP:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL], find_unused_parameters=False)

from torch.utils.data.distributed import DistributedSampler
ds = Groups(examples)
sampler = DistributedSampler(ds, shuffle=True, seed=args.seed) if IS_DDP else None
dl = DataLoader(ds, batch_size=args.groups_per_step, shuffle=(sampler is None),
                sampler=sampler, collate_fn=collate, num_workers=2, drop_last=True)

steps = int(len(dl) * args.epochs) // args.accum
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps) + 1, max(steps, 1))
log(f"optimizer steps: {steps}  (world={WORLD})")

t0, step, done, ep = time.time(), 0, 0, 0
model.train()
target_updates = steps
while done < target_updates:
    if sampler: sampler.set_epoch(ep)
    for enc, nb in dl:
        enc = {k: v.to(DEV, non_blocking=True) for k, v in enc.items()}
        logits = model(**enc).logits.view(nb, args.group)     # gold is column 0
        loss = F.cross_entropy(logits, torch.zeros(nb, dtype=torch.long, device=DEV)) / args.accum
        loss.backward()
        step += 1
        if step % args.accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); done += 1
            if done % 20 == 0:
                acc = (logits.argmax(-1) == 0).float().mean().item()
                log(f"  step {done}/{target_updates}  loss={loss.item()*args.accum:.4f} "
                    f"group_acc={acc:.3f}  lr={sched.get_last_lr()[0]:.2e}  {(time.time()-t0)/60:.1f}m")
            if done >= target_updates: break
    ep += 1

if RANK0:
    (model.module if IS_DDP else model).save_pretrained(args.out)
    tok.save_pretrained(args.out)
    json.dump(dict(base=args.base, group=args.group, lora_r=args.lora_r, epochs=args.epochs,
                   groups=len(examples), gpu_hours=round((time.time()-t0)/3600*WORLD, 3),
                   train_minutes=round((time.time()-t0)/60, 1)),
              open(Path(args.out)/"train_meta.json","w"), indent=2)
    log(f"saved -> {args.out}   {(time.time()-t0)/60:.1f} min x {WORLD} GPUs")
if IS_DDP: dist.destroy_process_group()
