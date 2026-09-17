"""Generate the pedagogical payload for all 2,587 misconceptions once offline.

This is the architectural bet: school mathematics has a *finite* misconception
taxonomy, so the expensive part (writing a good Socratic hint ladder) can be done
once on our own GPUs and stored. Production then serves a primary-key lookup with
no LLM in the hot path, which is where the 600x cost gap comes from.

Per misconception we store: a student-facing restatement, why students fall for it,
a 3-rung hint ladder (rung 1 is a QUESTION, never an answer), 2 probe questions
that discriminate this misconception from its neighbours, and a tutor opener.
"""
import argparse, json, os, re, sqlite3, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--num-shards", type=int, default=1)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--max-tokens", type=int, default=900)
ap.add_argument("--only-ids", default=None,
                help="json list of misconception ids to (re)generate")
ap.add_argument("--temperature", type=float, default=0.6)
ap.add_argument("--out", default=None)
args = ap.parse_args()

OUT = Path(args.out or ROOT / f"data/processed/explanations_shard{args.shard}.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

misc = pd.read_parquet(ROOT / "data/processed/misconceptions.parquet")
train = pd.read_parquet(ROOT / "data/processed/train.parquet")
ex_by_mid = {}
for _, r in train.iterrows():
    ex_by_mid.setdefault(int(r["misconception_id"]), []).append(r)

targets = misc.sort_values("misconception_id")
if args.only_ids:
    keep = set(json.loads(Path(args.only_ids).read_text()))
    targets = targets[targets["misconception_id"].isin(keep)]
targets = targets.iloc[args.shard::args.num_shards]
if args.limit:
    targets = targets.head(args.limit)
print(f"[shard {args.shard}] explaining {len(targets)} misconceptions", flush=True)

SYSTEM = ("You are an expert mathematics tutor-trainer. You turn a terse diagnostic "
          "misconception label into material a tutor can use in the next 30 seconds of a "
          "live session. You answer with one JSON object and nothing else.")

TEMPLATE = """MISCONCEPTION (diagnostic label): "{name}"
{example}
Produce a JSON object with exactly these keys:

"plain": one sentence a tutor could say aloud describing what the student is doing wrong,
         in plain English, no jargon. Address the behaviour, never call the student wrong.
"why": one or two sentences on WHY this is a natural mistake (the over-generalised rule)
       or surface pattern that makes it feel right to the student.
"hints": array of exactly 3 strings, a Socratic ladder.
         hints[0] MUST be a question that makes the student notice the problem themselves.
                  It must NOT contain the answer or the correction.
         hints[1] a more concrete nudge, still not the answer: point at the specific step.
         hints[2] the direct explanation of the correct rule.
"probes": array of exactly 2 short diagnostic questions that separate a student who has
          cleared this misconception from one who has not. Include the expected correct answer
          in parentheses at the end of each.
"opener": one question a tutor should ask to open the conversation about this,
          phrased warmly and curiously.
"topic": a 1-3 word topic tag, e.g. "Fractions", "Order of operations", "Angles".

JSON object only. No markdown fences, no prose."""


def build(mid, name):
    ex = ""
    rows = ex_by_mid.get(int(mid), [])
    if rows:
        r = rows[0]
        ex = (f'\nA real question where a student with this misconception answered wrongly:\n'
              f'  Topic: {r["subject"]} / {r["construct"]}\n  Question: {r["question"][:320]}\n'
              f'  Correct: {r["correct"]}\n  Student chose: {r["incorrect"]}\n')
    return TEMPLATE.format(name=name, example=ex)


FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.M)
_ESC = re.compile(r'\\(["\\/]|u[0-9a-fA-F]{4})|\\')


def parse_obj(text):
    text = FENCE.sub("", text).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    for cand in (m.group(0), _ESC.sub(lambda x: x.group(0) if x.group(1) else "\\\\", m.group(0))):
        try:
            o = json.loads(cand)
            if isinstance(o, dict):
                return o
        except Exception:
            continue
    return None


def normalise(o, mid, name):
    """Never ship a half-formed record: fall back to the raw label rather than a blank.

    A record that had to fall back is tagged generated_by="degraded:<model>" so the
    count of genuinely-generated explanations stays honest, avoiding boilerplate credit
    row to the model is how a quality number quietly becomes a lie.
    """
    ok = isinstance(o, dict) and bool(str(o.get("plain", "")).strip())
    if not isinstance(o, dict):
        o = {}
    hints = [str(h).strip() for h in o.get("hints", []) if str(h).strip()][:3]
    while len(hints) < 3:
        hints.append("Walk through the step you just did, out loud, one operation at a time.")
    probes = [str(p).strip() for p in o.get("probes", []) if str(p).strip()][:2]
    while len(probes) < 2:
        probes.append("Try a simpler version of the same question and talk me through it.")
    return dict(
        misconception_id=int(mid), misconception_name=name,
        plain=str(o.get("plain", "") or name).strip()[:500],
        why=str(o.get("why", "") or "").strip()[:600],
        hints=hints, probes=probes,
        opener=str(o.get("opener", "") or f"Talk me through how you got that: what was your first step?").strip()[:300],
        topic=str(o.get("topic", "") or "Mathematics").strip()[:60],
        generated_by=args.model if ok else f"degraded:{args.model}",
    )


def main():
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
                {"role": "user", "content": build(mid, name)}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        meta.append((mid, name))

    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    t0, n_fail = time.time(), 0
    with OUT.open("w") as fh:
        for bi in range(0, len(order), args.batch):
            idx = order[bi:bi + args.batch]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                      truncation=True, max_length=1400).to("cuda")
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=args.max_tokens, do_sample=True,
                                     temperature=args.temperature, top_p=0.9, pad_token_id=tok.pad_token_id)
            for j, i in enumerate(idx):
                mid, name = meta[i]
                o = parse_obj(tok.decode(out[j, enc["input_ids"].shape[1]:], skip_special_tokens=True))
                if o is None:
                    n_fail += 1
                fh.write(json.dumps(normalise(o, mid, name), ensure_ascii=False) + "\n")
            fh.flush()
            seen = min(bi + args.batch, len(order))
            el = time.time() - t0
            print(f"[shard {args.shard}] {seen}/{len(order)}  fails={n_fail}  "
                  f"{el/60:.1f}m  eta {(len(order)-seen)/max(seen,1)*el/60:.1f}m", flush=True)
    print(json.dumps(dict(shard=args.shard, n=len(order), parse_failures=n_fail,
                          wall_min=round((time.time()-t0)/60, 1), out=str(OUT)), indent=2))

main()
