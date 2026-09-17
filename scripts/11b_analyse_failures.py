"""Compute diagnostic signals for each failure case, so the written note is evidence-based.

A failure gallery is only credible if the explanation of each case is right. Guessing
"ambiguous distractor" for everything is worse than no gallery. This computes signals --
how close the prediction is to the gold label in embedding space, whether other labels
in the taxonomy are near-duplicates of gold, whether the question needs an asset we do
not have, and proposes a category. The prose note is still written by hand against
these signals; the script never invents one.
"""
import json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "serve/artifacts"
sys.path.insert(0, str(ROOT / "serve"))

raw = json.loads((ART / "failure_gallery_raw.json").read_text())
E = np.load(ART / "misc_emb.npy")
rows = json.loads((ART / "misconceptions.json").read_text())
names = [r["misconception_name"] for r in rows]
ids = [int(r["misconception_id"]) for r in rows]
by_name = {n: i for i, n in enumerate(names)}
NEEDS_ASSET = re.compile(r"!\[|\\begin\{tabular\}|\\begin\{array\}")

out = []
for f in raw:
    gi, pi = by_name.get(f["gold_name"]), by_name.get(f["pred_name"])
    sim = float(E[gi] @ E[pi]) if (gi is not None and pi is not None) else None
    # how crowded is the neighbourhood around the GOLD label?
    twins = []
    if gi is not None:
        s = E @ E[gi]
        for j in np.argsort(-s)[1:6]:
            if s[j] >= 0.72:
                twins.append(dict(name=names[j], sim=round(float(s[j]), 3)))
    needs_asset = bool(NEEDS_ASSET.search(f["question"]))

    if needs_asset:
        cat = "needs the diagram"
    elif sim is not None and sim >= 0.80:
        cat = "near-duplicate labels"
    elif twins:
        cat = "crowded neighbourhood"
    elif sim is not None and sim >= 0.55:
        cat = "same family, wrong member"
    elif f.get("gold_rank") is None:
        cat = "retrieval miss"
    else:
        cat = "ranking miss"
    out.append({**f, "category": cat,
                "signal_pred_gold_cosine": None if sim is None else round(sim, 3),
                "signal_gold_near_twins": twins,
                "signal_needs_asset": needs_asset})

(ART / "failure_gallery_analysed.json").write_text(json.dumps(out, indent=2))
print(f"{len(out)} cases analysed -> failure_gallery_analysed.json\n")
for i, f in enumerate(out, 1):
    print(f"[{i}] {f['category']}   pred~gold cosine={f['signal_pred_gold_cosine']}   "
          f"gold_rank={f['gold_rank']}   needs_asset={f['signal_needs_asset']}")
    print(f"     Q: {f['question'][:110]}")
    print(f"     chose {f['student_answer'][:34]!r} | gold: {f['gold_name'][:62]}")
    print(f"     pred: {f['pred_name'][:62]}")
    for t in f["signal_gold_near_twins"][:2]:
        print(f"     twin of gold ({t['sim']}): {t['name'][:60]}")
    print()
