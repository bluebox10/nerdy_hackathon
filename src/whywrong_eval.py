"""Evaluation harness for WhyWrong misconception retrieval.

Primary metric is MAP@25 (the official Eedi/Kaggle metric). We also report
Recall@k and top-1 accuracy, and every metric is additionally broken out on the
`unseen` slice: test queries whose gold misconception never appears in training.
That slice is the one that actually matters -- 58% of the taxonomy is never seen.
"""
from __future__ import annotations
import json
import numpy as np


def average_precision_at_k(ranked_ids, gold_id, k=25):
    """Eedi's MAP@25: exactly one relevant document, so AP = 1/rank if hit else 0."""
    for i, mid in enumerate(ranked_ids[:k]):
        if mid == gold_id:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked_ids, gold_id, k):
    return float(gold_id in ranked_ids[:k])


def evaluate(rankings, gold, unseen_mask=None, ks=(1, 5, 10, 25)):
    """rankings: (N, >=25) array/list of misconception ids, best first.
       gold: (N,) gold misconception ids.  unseen_mask: (N,) bool."""
    rankings = [list(r) for r in rankings]
    gold = list(gold)
    assert len(rankings) == len(gold), "rankings/gold length mismatch"

    def _agg(idx):
        if not len(idx):
            return None
        out = {"n": len(idx)}
        out["map@25"] = float(np.mean([average_precision_at_k(rankings[i], gold[i], 25) for i in idx]))
        for k in ks:
            out[f"recall@{k}"] = float(np.mean([recall_at_k(rankings[i], gold[i], k) for i in idx]))
        out["top1_acc"] = out["recall@1"]
        # mean reciprocal rank over the full returned list (0 if absent)
        rr = []
        for i in idx:
            rr.append(1.0 / (rankings[i].index(gold[i]) + 1) if gold[i] in rankings[i] else 0.0)
        out["mrr"] = float(np.mean(rr))
        return out

    all_idx = list(range(len(gold)))
    res = {"overall": _agg(all_idx)}
    if unseen_mask is not None:
        unseen_mask = np.asarray(unseen_mask, dtype=bool)
        res["unseen"] = _agg([i for i in all_idx if unseen_mask[i]])
        res["seen"] = _agg([i for i in all_idx if not unseen_mask[i]])
    return res


def fmt(name, res):
    lines = [f"--- {name} ---"]
    for slice_name, m in res.items():
        if m is None:
            continue
        lines.append(
            f"  {slice_name:8s} n={m['n']:5d}  MAP@25={m['map@25']:.4f}  "
            f"R@25={m['recall@25']:.4f}  R@10={m['recall@10']:.4f}  "
            f"R@5={m['recall@5']:.4f}  top1={m['top1_acc']:.4f}  MRR={m['mrr']:.4f}")
    return "\n".join(lines)


def save(path, name, res, extra=None):
    import datetime, pathlib
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(p.read_text()) if p.exists() else []
    rows = [r for r in rows if r.get("name") != name]
    rec = {"name": name, "metrics": res}
    if extra:
        rec.update(extra)
    rows.append(rec)
    p.write_text(json.dumps(rows, indent=2))
    return rec
