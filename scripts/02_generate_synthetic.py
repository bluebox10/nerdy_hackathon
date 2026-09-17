"""Synthetic augmentation for the long tail.

58% of the 2,587-misconception taxonomy never appears in our training split, so
supervised contrastive learning has literally nothing to learn for those classes.
Fix: ask a locally-hosted instruct model to *write* diagnostic questions whose
distractor embodies each misconception, especially the ones with zero real
examples. Runs offline on our own A100s; never touches the serving path.
"""
import argparse, json, os, random, re, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--n-per", type=int, default=12, help="questions requested per misconception")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--num-shards", type=int, default=1)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--only-unseen", action="store_true", help="restrict to misconceptions absent from train")
ap.add_argument("--temperature", type=float, default=0.9)
ap.add_argument("--pass-tag", default="p1")
ap.add_argument("--max-tokens", type=int, default=2600)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--out", default=None)
args = ap.parse_args()

misc = pd.read_parquet(ROOT / "data/processed/misconceptions.parquet")
train = pd.read_parquet(ROOT / "data/processed/train.parquet")
seen = set(train["misconception_id"])

# Real in-train examples, keyed by misconception, used as few-shot style anchors.
by_mid = {}
for _, r in train.iterrows():
    by_mid.setdefault(int(r["misconception_id"]), []).append(r)

targets = misc.copy()
if args.only_unseen:
    targets = targets[~targets["misconception_id"].isin(seen)]
targets = targets.sort_values("misconception_id").reset_index(drop=True)
targets = targets.iloc[args.shard::args.num_shards]
if args.limit:
    targets = targets.head(args.limit)
print(f"[shard {args.shard}/{args.num_shards}] targets={len(targets)}", flush=True)

OUT_PATH = Path(args.out or ROOT / f"data/synth/synth_{args.pass_tag}_shard{args.shard}.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

SUBJECTS = sorted(train["subject"].dropna().unique().tolist())

SYSTEM = (
    "You are a UK secondary-school mathematics assessment writer who specialises in "
    "diagnostic multiple-choice questions. You write questions where the wrong option is "
    "chosen deliberately: it is exactly the answer a student produces if they hold one "
    "specific, nameable misconception. You output one JSON object per line and nothing else."
)

# Fixed few-shot demonstrations, taken from real training rows, chosen to span topics.
# These teach the *relationship* (misconception -> the wrong answer it forces). An earlier
# version instead listed plausible subject areas; the model latched onto the subject list
# and ignored the misconception entirely, so the hint was removed.
def _demo_pool():
    pool = []
    for _, r in train.sample(frac=1.0, random_state=7).iterrows():
        if 40 < len(r["question"]) < 260 and r["correct"] and r["incorrect"]:
            pool.append(r)
        if len(pool) >= 400:
            break
    return pool

DEMOS = _demo_pool()

def _fmt_demo(r):
    return json.dumps(dict(subject=r["subject"], construct=r["construct"],
                           question=r["question"], correct=r["correct"],
                           incorrect=r["incorrect"],
                           why_tempting="the misconception forces this exact value"),
                      ensure_ascii=False)

TEMPLATE = """MISCONCEPTION TO TARGET:
"{misconception}"

Write {n} DIVERSE diagnostic maths questions for exactly this misconception.

Hard requirements:
- The question must be about the mathematical topic THIS misconception is about.
  Do not drift to another topic. If the misconception is about gradients, write about
  gradients; if it is about fractions, write about fractions.
- "incorrect" must be the answer a student produces *because* they hold this
  misconception, mechanically derivable from it. Not a random wrong answer,
  not an arithmetic slip.
- "correct" must be the genuinely correct answer, and different from "incorrect".
- Each of the {n} questions must probe the misconception by a DIFFERENT route.
  Changing only the numbers in one template is a failure; vary the sub-topic, the
  representation (symbolic / word problem / table / "who is correct?" / "which
  statement is true"), and the difficulty (UK Key Stage 2 to GCSE).
- "why_tempting" must name the exact mechanical step the misconception causes,
  e.g. "subtracts the denominators as well as the numerators, giving 24".
- Use LaTeX inline maths like \\( \\frac{{3}}{{4}} \\) or \\[ 3 + 5 \\times 2 \\].

Here are examples of the required style and JSON shape (DIFFERENT misconceptions --
copy the format, not the content):
{demos}
{anchor}
Now output exactly {n} lines. One JSON object per line, keys:
"subject", "construct", "question", "correct", "incorrect", "why_tempting"
No markdown, no numbering, no array brackets, no prose. JSON lines only."""


def build_prompt(mid, name):
    demos = "\n".join(_fmt_demo(r) for r in random.sample(DEMOS, k=min(3, len(DEMOS))))
    real = by_mid.get(int(mid), [])
    if real:
        r = random.choice(real)
        anchor = ("\nA REAL question that targets the TARGET misconception: match its rigour, "
                  "do NOT copy it:\n" + _fmt_demo(r) + "\n")
    else:
        anchor = ("\nThere are no existing examples for the target misconception. Reason first about "
                  "what mathematical operation a student holding it would perform, then build "
                  "questions where that operation yields the 'incorrect' value.\n")
    return TEMPLATE.format(n=args.n_per, misconception=name, demos=demos, anchor=anchor)


FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.M)

# Only ", \, / and \uXXXX are preserved. \n \t \r \b \f are deliberately treated as
# LaTeX (\neq, \times, \right, \begin, \frac) and doubled: strict json.loads is always
# tried first, so this repair only ever sees output that was already invalid JSON,
# where a backslash before a letter is LaTeX essentially without exception.
_ESC = re.compile(r'\\(["\\\\/]|u[0-9a-fA-F]{4})|\\')


def repair_escapes(blob: str) -> str:
    """Models write LaTeX as \\( x^2 \\) inside JSON strings, but a lone backslash is an
    illegal JSON escape, so json.loads rejects the whole object. Consume valid escape
    sequences untouched and double every remaining lone backslash."""
    return _ESC.sub(lambda m: m.group(0) if m.group(1) else "\\\\", blob)


def parse_items(text):
    """Accept JSON-lines (what we ask for), a JSON array (returned anyway sometimes),
    or a truncated mixture. Every complete object is salvaged independently, so a
    completion cut off mid-array still yields all the objects that did finish."""
    text = FENCE.sub("", text).strip()
    out, seen = [], set()
    for m in re.finditer(r"\{(?:[^{}]|\{[^{}]*\})*\}", text, re.S):
        blob = m.group(0)
        if blob in seen:
            continue
        seen.add(blob)
        for cand in (repair_escapes(blob), blob):
            try:
                o = json.loads(cand)
            except Exception:
                continue
            if isinstance(o, dict):
                out.append(o)
            break
    return out


REQUIRED = ("subject", "construct", "question", "correct", "incorrect")

def clean(item, mid, name):
    if not isinstance(item, dict) or any(k not in item for k in REQUIRED):
        return None
    v = {k: str(item[k]).strip() for k in REQUIRED}
    if not v["question"] or not v["incorrect"] or v["correct"] == v["incorrect"]:
        return None
    if len(v["question"]) < 15 or len(v["question"]) > 1200:
        return None
    v["why_tempting"] = str(item.get("why_tempting", "")).strip()[:400]
    v["misconception_id"] = int(mid)
    v["misconception_name"] = name
    v["source"] = f"synthetic:{args.pass_tag}"
    return v


def main():
    """HF batched generation. vLLM hangs at NCCL init on this shared box; raw
    transformers is fast enough; length-sorted batches of 64 keep one A100
    saturated, and we shard the misconception list across GPUs by process."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda").eval()

    prompts, meta = [], []
    for _, row in targets.iterrows():
        mid, name = int(row["misconception_id"]), row["misconception_name"]
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_prompt(mid, name)}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        meta.append((mid, name))

    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    rows, n_parse_fail, gen_tok = [], 0, 0
    t0 = time.time()
    B = args.batch
    for bi in range(0, len(order), B):
        idx = order[bi:bi + B]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=1536).to("cuda")
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=args.max_tokens, do_sample=True,
                                 temperature=args.temperature, top_p=0.95,
                                 pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        gen_tok += int((gen != tok.pad_token_id).sum())
        for j, i in enumerate(idx):
            mid, name = meta[i]
            raw = tok.decode(gen[j], skip_special_tokens=True)
            if os.environ.get("WW_DUMP_RAW"):
                Path(os.environ["WW_DUMP_RAW"]).open("a").write(
                    "\n===== mid=%s :: %s =====\n%s\n" % (mid, name, raw))
            items = parse_items(raw)
            if not items:
                n_parse_fail += 1
            for it in items:
                c = clean(it, mid, name)
                if c:
                    rows.append(c)
        el = time.time() - t0
        seen_n = min(bi + B, len(order))
        print(f"[shard {args.shard}] {seen_n}/{len(order)} prompts  {len(rows)} rows  "
              f"{gen_tok/el:.0f} tok/s  {el/60:.1f}m  "
              f"eta {(len(order)-seen_n)/max(seen_n,1)*el/60:.1f}m", flush=True)
        pd.DataFrame(rows).to_parquet(OUT_PATH, index=False)

    dt = time.time() - t0
    print(json.dumps(dict(shard=args.shard, prompts=len(prompts), rows=len(rows),
                          parse_failures=n_parse_fail, wall_s=round(dt, 1),
                          gen_tokens=int(gen_tok), tok_per_s=round(gen_tok / dt, 1),
                          out=str(OUT_PATH)), indent=2), flush=True)

main()
