"""Populate placeholder slots in submission/SUBMISSION.md from measured metrics.

Values are populated directly from results/metrics.json and benchmark.json.
Any unmeasured metric is written as 'not measured'.
"""
import json, re, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
metrics = {r["name"]: r for r in json.loads((ROOT / "results/metrics.json").read_text())}
bench_p = ROOT / "serve/artifacts/benchmark.json"
bench = json.loads(bench_p.read_text()) if bench_p.exists() else {}
url = (ROOT / "serve/artifacts/PUBLIC_URL").read_text().strip() if (ROOT / "serve/artifacts/PUBLIC_URL").exists() else "<live-url>"

NM = "not measured"
def m(name, slice_="overall", key="map@25", fmt="{:.3f}"):
    r = metrics.get(name)
    if not r or not r["metrics"].get(slice_):
        return NM
    return fmt.format(r["metrics"][slice_][key])

def first(names, **kw):
    for n in names:
        v = m(n, **kw)
        if v != NM:
            return v
    return NM

final = ["shipped_onnx_full:test", "biencoder_r3_hardneg:test", "biencoder_r2_synth:test"]
n_synth = NM
fp = ROOT / "results/synth_filter_report.json"
if fp.exists():
    n_synth = f"{json.loads(fp.read_text())['kept']:,}"

cost_rows = {r["name"]: r for r in bench.get("cost", {}).get("rows", [])}
ours = next((r for r in cost_rows.values() if r.get("shipped")), None)
frontier = next((r for r in cost_rows.values() if not r.get("shipped")), None)
ratio = NM
for h in bench.get("headline", []):
    if "cheaper" in h.get("key", ""):
        ratio = h["value"].rstrip("x")

teacher = m("teacher7b:test")
shipped = first(final)
gap = NM
if teacher != NM and shipped != NM:
    gap = f"{float(teacher) - float(shipped):.3f}"

vals = {
    "N_SYNTH": n_synth,
    "R25_UNSEEN_BEFORE": m("biencoder_r1_realonly:test", "unseen", "recall@25"),
    "R25_UNSEEN_AFTER": first(["biencoder_r3_hardneg:test", "biencoder_r2_synth:test"],
                              slice_="unseen", key="recall@25"),
    "MAP_FINAL": shipped,
    "MAP_FLOOR": m("offtheshelf:bge-small-en-v1.5"),
    "P50": str(bench.get("latency", {}).get("p50", NM)),
    "COST_OURS": ours["per_1k"] if ours else NM,
    "COST_FRONTIER": frontier["per_1k"] if frontier else NM,
    "COST_RATIO": ratio,
    "DISTILL_GAP": gap,
    "URL": url,
}

src = (ROOT / "submission/SUBMISSION.md").read_text(encoding="utf-8")
out = re.sub(r"\[\[(\w+)\]\]", lambda x: vals.get(x.group(1), NM), src)
(ROOT / "submission/SUBMISSION_FILLED.md").write_text(out, encoding="utf-8")
print(json.dumps(vals, indent=2))
missing = [k for k, v in vals.items() if v == NM]
print(f"\n-> submission/SUBMISSION_FILLED.md updated successfully")
if missing:
    print(f"Unmeasured: {', '.join(missing)}")
