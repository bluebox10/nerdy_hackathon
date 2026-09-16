"""Assemble the serving bundle: SQLite of precomputed explanations + demo examples."""
import argparse, json, random, sqlite3, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "serve/artifacts"
ART.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--explanations", nargs="+", default=[])
ap.add_argument("--n-examples", type=int, default=60)
args = ap.parse_args()

import re as _re

# Many Eedi questions reference a diagram we do not have the asset for, or embed a LaTeX
# table. Either is unanswerable for someone reading the demo, so they are excluded from
# the demo pool -- they stay in the eval split, where they are scored like everything else.
_NEEDS_ASSET = _re.compile(r"!\[|\\begin\{tabular\}|\\begin\{array\}")


def demoable(q: str) -> bool:
    return not _NEEDS_ASSET.search(str(q))


misc = pd.read_parquet(ROOT / "data/processed/misconceptions.parquet")
MID2NAME = dict(zip(misc["misconception_id"], misc["misconception_name"]))

# ---------------- explanations -> sqlite ----------------
db = ART / "whywrong.sqlite"
if db.exists():
    db.unlink()
con = sqlite3.connect(str(db))
con.execute("""CREATE TABLE explanations(
  misconception_id INTEGER PRIMARY KEY, misconception_name TEXT NOT NULL,
  plain TEXT, why TEXT, hints TEXT, probes TEXT, opener TEXT, topic TEXT, generated_by TEXT)""")

rows, seen = [], set()
for p in args.explanations:
    for path in sorted(Path().glob(p)) or ([Path(p)] if Path(p).exists() else []):
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            mid = int(o["misconception_id"])
            if mid in seen:
                continue
            seen.add(mid)
            rows.append((mid, o["misconception_name"], o.get("plain", ""), o.get("why", ""),
                         json.dumps(o.get("hints", [])), json.dumps(o.get("probes", [])),
                         o.get("opener", ""), o.get("topic", ""), o.get("generated_by", "")))

# any misconception without a generated record still gets a usable row
for mid, name in MID2NAME.items():
    if int(mid) not in seen:
        rows.append((int(mid), name, name, "",
                     json.dumps(["Talk me through your first step on this one.",
                                 "Look again at the operation you applied — does it do what you expect?",
                                 f"The rule being missed here: {name}."]),
                     json.dumps(["Try a simpler version of the same question.",
                                 "Explain the rule back to me in your own words."]),
                     "Talk me through how you got that — what was your first step?", "Mathematics", "fallback"))

con.executemany("INSERT INTO explanations VALUES (?,?,?,?,?,?,?,?,?)", rows)
con.commit()
# Count honestly: a row whose JSON failed to parse fell back to boilerplate and is
# tagged "degraded:<model>" by script 09. Crediting it to the model would inflate the
# quality number that goes on the benchmark page.
n_degraded = sum(1 for r in rows if str(r[8]).startswith("degraded:"))
n_missing = sum(1 for r in rows if r[8] == "fallback")
n_good = len(rows) - n_degraded - n_missing
print(f"explanations: {len(rows)} rows -> {n_good} fully generated, "
      f"{n_degraded} degraded (parse failed, boilerplate), {n_missing} never generated")
json.dump(dict(total=len(rows), generated=n_good, degraded=n_degraded, missing=n_missing),
          open(ROOT / "results/explanation_quality.json", "w"), indent=2)

# ---------------- demo examples ----------------
# Reconstruct full multiple-choice items: one question row per distractor, so grouping
# by question_id recovers the original 4-option question.
test = pd.read_parquet(ROOT / "data/processed/test.parquet")
val = pd.read_parquet(ROOT / "data/processed/val.parquet")
pool = pd.concat([test, val], ignore_index=True)

examples = []
for qid, g in pool.groupby("question_id"):
    if len(g) < 2:
        continue
    r0 = g.iloc[0]
    distractors = list(dict.fromkeys(g["incorrect"].tolist()))
    opts = list(dict.fromkeys([r0["correct"]] + distractors))[:4]
    if len(opts) < 3 or any(len(str(o)) > 60 for o in opts) or len(r0["question"]) > 420:
        continue
    if not demoable(r0["question"]) or not all(demoable(o) for o in opts):
        continue
    random.Random(int(qid)).shuffle(opts)
    examples.append(dict(
        question_id=int(qid), subject=str(r0["subject"]), construct=str(r0["construct"]),
        question=str(r0["question"]), correct=str(r0["correct"]), options=[str(o) for o in opts],
        distractor=str(r0["incorrect"]),
        gold_misconception_id=int(r0["misconception_id"]),
        gold_misconception_name=MID2NAME[int(r0["misconception_id"])]))

random.Random(7).shuffle(examples)
# prefer a spread of subjects so the demo doesn't show four fraction questions in a row
by_subject, picked = {}, []
for e in examples:
    if len(by_subject.get(e["subject"], [])) < 4:
        by_subject.setdefault(e["subject"], []).append(e)
        picked.append(e)
    if len(picked) >= args.n_examples:
        break
(ART / "examples.json").write_text(json.dumps(picked, indent=1))
# The carousel is a curated 60. The PROBE pool is every demoable question we have, so
# "next problem, chosen to probe this misconception" can actually find a same-family
# follow-up instead of shrugging -- a 60-item pool spread over 29 subjects almost never can.
(ART / "probe_pool.json").write_text(json.dumps(examples, indent=1))
print(f"probe pool: {len(examples)} questions covering "
      f"{len({e['gold_misconception_id'] for e in examples})} misconceptions -> probe_pool.json")
print(f"examples: {len(picked)} multiple-choice items across "
      f"{len({e['subject'] for e in picked})} subjects -> examples.json")


# ---------------- tutor session script ----------------
# A believable struggling-student session. We pick a misconception whose real questions
# span SEVERAL different topics, because that is exactly the case where the tutor panel
# earns its place: the student looks like they are failing four separate subjects, and
# the panel shows it is one root cause. One off-cause error is included -- real sessions
# are not monocausal, and a panel that only ever shows 100% concentration is a lie.
demo_pool = pool[pool.question.map(demoable)]
counts = demo_pool.groupby("misconception_id").agg(n=("qid", "size"), topics=("subject", "nunique"))
eligible = counts[(counts.n >= 5) & (counts.topics >= 3)].sort_values(
    ["topics", "n"], ascending=False)
script = []
if len(eligible):
    root = int(eligible.index[0])
    g = pool[(pool.misconception_id == root) & pool.question.map(demoable)]
    g = g.drop_duplicates(subset=["question"])
    g = g.groupby("subject", group_keys=False).head(2).head(5)
    for _, r in g.iterrows():
        script.append(dict(subject=str(r["subject"]), construct=str(r["construct"]),
                           question=str(r["question"]), correct=str(r["correct"]),
                           student_answer=str(r["incorrect"]),
                           gold_misconception_id=int(r["misconception_id"]),
                           gold_misconception_name=MID2NAME[int(r["misconception_id"])],
                           is_root_cause=True))
    other = demo_pool[(demo_pool.misconception_id != root)
                      & (demo_pool.question.str.len() < 220)].iloc[0]
    script.append(dict(subject=str(other["subject"]), construct=str(other["construct"]),
                       question=str(other["question"]), correct=str(other["correct"]),
                       student_answer=str(other["incorrect"]),
                       gold_misconception_id=int(other["misconception_id"]),
                       gold_misconception_name=MID2NAME[int(other["misconception_id"])],
                       is_root_cause=False))
    (ART / "session_script.json").write_text(json.dumps(dict(
        student="Priya", root_misconception_id=root,
        root_misconception_name=MID2NAME[root],
        topics=sorted({s["subject"] for s in script if s["is_root_cause"]}),
        steps=script), indent=1))
    print(f"session script: {len(script)} questions, root #{root} "
          f"'{MID2NAME[root]}' spanning {len({s['subject'] for s in script if s['is_root_cause']})} topics")
