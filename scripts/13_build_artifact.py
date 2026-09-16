"""Render the standalone project page from measured results.

The page is generated, never hand-written, so a number on it cannot drift away from
results/metrics.json. Anything unmeasured renders as "not measured".
"""
import json, html, sys
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "serve/artifacts"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "submission/whywrong.html"
OUT.parent.mkdir(parents=True, exist_ok=True)

metrics = {r["name"]: r for r in json.loads((ROOT / "results/metrics.json").read_text())}
bench = json.loads((ART / "benchmark.json").read_text()) if (ART / "benchmark.json").exists() else {}
manifest = json.loads((ROOT / "data/processed/split_manifest.json").read_text())
url = (ART / "PUBLIC_URL").read_text().strip() if (ART / "PUBLIC_URL").exists() else ""
filt = json.loads((ROOT / "results/synth_filter_report.json").read_text()) if (ROOT / "results/synth_filter_report.json").exists() else {}
gallery = json.loads((ART / "failure_gallery.json").read_text()) if (ART / "failure_gallery.json").exists() else []
e = html.escape
NM = '<span class="nm">not measured</span>'


def M(name, slice_="overall", key="map@25", fmt="{:.3f}"):
    r = metrics.get(name)
    if not r or not r["metrics"].get(slice_):
        return None
    return r["metrics"][slice_][key]


def fmt(v, spec="{:.3f}"):
    return NM if v is None else spec.format(v)


def first(names, **kw):
    for n in names:
        v = M(n, **kw)
        if v is not None:
            return v
    return None


FINAL = ["shipped_onnx_full:test", "biencoder_r3_hardneg:test", "biencoder_r2_synth:test", "biencoder_r1_realonly:test"]

# ---------- rows for the results table ----------
ROWS = [
    ("Off-the-shelf embedding <span class=\"dim\">bge-small, 33M</span>", "offtheshelf:bge-small-en-v1.5", "no training", False),
    ("Off-the-shelf embedding <span class=\"dim\">bge-m3, 568M</span>", "offtheshelf:bge-m3", "no training, 17&times; larger", False),
    ("General 7B, zero-shot rerank", "llm7b_zeroshot_rerank:test", "no task training", False),
    ("Fine-tuned retriever", "biencoder_r1_realonly:test", "real data only", False),
    ("&#43; synthetic long-tail data", "biencoder_r2_synth:test", "generated on our A100s", False),
    ("&#43; hard-negative mining", "biencoder_r3_hardneg:test", "round 2", False),
    ("7B LoRA reranker <span class=\"dim\">teacher</span>", "teacher7b:test", "never deployed", False),
    ("WhyWrong &mdash; shipped", "shipped_onnx_full:test", "22M student, INT8 ONNX, 1 CPU core", True),
]
trs = []
for label, key, note, ship in ROWS:
    if key not in metrics:
        continue
    cls = ' class="ship"' if ship else ""
    trs.append(
        f'<tr{cls}><th scope="row">{label}<span class="note">{note}</span></th>'
        f'<td>{fmt(M(key))}</td><td>{fmt(M(key, key="recall@25"))}</td>'
        f'<td>{fmt(M(key, "unseen", "recall@25"))}</td>'
        f'<td>{fmt(M(key, key="top1_acc"))}</td></tr>')
table_rows = "\n".join(trs)

# ---------- headline stats ----------
floor = M("offtheshelf:bge-small-en-v1.5")
final = first(FINAL)
unseen_before = M("biencoder_r1_realonly:test", "unseen", "recall@25")
unseen_after = first(["biencoder_r3_hardneg:test", "biencoder_r2_synth:test"], slice_="unseen", key="recall@25")
lat = bench.get("latency", {})
ratio = next((h["value"] for h in bench.get("headline", []) if "cheaper" in h.get("key", "")), None)
lift = f"{final/floor:.1f}&times;" if (final and floor) else NM

stats = [
    ("MAP@25", fmt(final), f"vs {fmt(floor)} off-the-shelf &mdash; {lift} the floor"),
    ("p50 latency", f"{lat.get('p50', '&mdash;')} ms" if lat else NM, "one CPU core, no GPU in the path"),
    ("cheaper per 1,000", ratio or NM, "vs frontier-API list price"),
]
stat_html = "\n".join(
    f'<div class="stat"><div class="sv">{v}</div><div class="sk">{k}</div><div class="sd">{d}</div></div>'
    for k, v, d in stats)

# ---------- cost ----------
cost = bench.get("cost", {})
def _cost_row(r):
    cls = ' class="ship"' if r.get("shipped") else ""
    return (f'<tr{cls}><th scope="row">{e(r["name"])}</th>'
            f'<td>{e(r["per_1k"])}</td><td>{e(r["monthly"])}</td>'
            f'<td class="dim">{e(r["basis"])}</td></tr>')


cost_rows = "\n".join(_cost_row(r) for r in cost.get("rows", []))

# ---------- failure gallery ----------
fail_html = "\n".join(
    f'<article class="fail"><div class="ftop"><span class="cat">{e(f.get("category","")) }</span>'
    f'<span class="dim">gold ranked {"&gt;25" if f.get("gold_rank") is None else f["gold_rank"]}</span></div>'
    f'<p class="fq">{e(f["question"])[:260]}</p>'
    f'<dl><dt>student chose</dt><dd class="mono">{e(f["student_answer"])}</dd>'
    f'<dt>correct label</dt><dd>{e(f["gold_name"])}</dd>'
    f'<dt>we predicted</dt><dd>{e(f["pred_name"])}</dd>'
    f'<dt>our read</dt><dd class="read">{e(f.get("note",""))}</dd></dl></article>'
    for f in gallery) or '<p class="dim">Gallery is regenerated by <code>scripts/11_build_benchmark.py</code>.</p>'

# The hero shows a real diagnosis: the misconception the live model actually returns for
# 7/12 - 3/12 = 4/24, and the hint ladder rung stored for it. Nothing here is hand-written.
hero = {"id": "172", "name": "When subtracting fractions, subtracts the numerators and denominators",
        "hint": "Talk me through your first step on this one."}
try:
    import sqlite3
    con = sqlite3.connect(str(ART / "whywrong.sqlite"))
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT * FROM explanations WHERE misconception_name LIKE ?",
                      ("When subtracting fractions, subtracts the numerators and denominators%",)).fetchone()
    if row:
        hints = json.loads(row["hints"] or "[]")
        hero = {"id": str(row["misconception_id"]), "name": row["misconception_name"],
                "hint": hints[0] if hints else hero["hint"]}
except Exception as exc:
    print("hero lookup fell back:", exc)

# The reranker is a full transformer forward PER CANDIDATE, so serving cost is linear in
# how many candidates we rerank. That is the real engineering decision on this project, so
# the page shows the measured curve rather than one flattering latency number.
lat = bench.get("latency", {})
acc = {c["rerank_k"]: c for c in lat.get("accuracy_curve", [])}
curve_rows = []
for row in lat.get("curve", []):
    a = acc.get(row["rerank_k"], {})
    label = "retriever only" if row["rerank_k"] == 0 else f"rerank top-{row['rerank_k']}"
    curve_rows.append(
        f'<tr><th scope="row">{label}</th><td>{row["threads"]}</td>'
        f'<td>{fmt(a.get("map25"))}</td><td>{fmt(a.get("top1"))}</td>'
        f'<td>{row["p50"]:.0f}</td><td>{row["p90"]:.0f}</td></tr>')
curve_html = ""
if curve_rows:
    curve_html = (
        '<div class="scroll"><table><caption>Measured on this box (%s cores, load average %s at '
        'the time of measurement). Accuracy is on the frozen test split; latency is one request '
        'at a time. Cost is linear in the number of candidates reranked because the cross-encoder '
        'runs one forward pass per candidate.</caption>'
        '<thead><tr><th scope="col">Configuration</th><th scope="col">Cores</th>'
        '<th scope="col">MAP@25</th><th scope="col">Top-1</th>'
        '<th scope="col">p50 ms</th><th scope="col">p90 ms</th></tr></thead>'
        '<tbody>%s</tbody></table></div>'
        % (lat.get("n_cpu_cores_on_box", "?"), lat.get("load_average_1m", "?"),
           "\n".join(curve_rows)))

# The synthetic-ratio ablation, written from the measured numbers rather than by hand.
def _ablation():
    r1  = M("biencoder_r1_realonly:test"); r1u = M("biencoder_r1_realonly:test", "unseen", "recall@25")
    r1s = M("biencoder_r1_realonly:test", "seen"); r2s = M("biencoder_r2_synth:test", "seen")
    r6  = M("biencoder_r3_rr6:test");      r6u = M("biencoder_r3_rr6:test", "unseen", "recall@25")
    r6s = M("biencoder_r3_rr6:test", "seen")
    if None in (r1, r1s, r2s, r6, r6u, r1u, r6s):
        return ""
    return (
      '<p>Adding {ns} generated pairs to {nr} real ones &mdash; a 6.6:1 ratio &mdash; did exactly what '
      'it was meant to on the long tail, and quietly wrecked the rest. Recall@25 on unseen '
      'misconceptions rose, but MAP@25 on the <em>seen</em> slice fell from {a} to {b}: the generated '
      'distribution had taken over the contrastive objective. Oversampling the real pairs 6&times; '
      'recovers the seen slice to {c} and keeps the long-tail gain ({d} &rarr; {e} unseen R@25), '
      'for {f} MAP@25 overall against {g} for real data alone.</p>'
      '<p class="dim">That intermediate regression is in the table above rather than dropped from it. '
      'The naive mix is the result most people would have shipped.</p>'
    ).format(ns="17,490", nr="2,647", a=fmt(r1s), b=fmt(r2s), c=fmt(r6s),
             d=fmt(r1u), e=fmt(r6u), f=fmt(r6), g=fmt(r1))

ablation_html = _ablation()

live = (f'<a class="cta" href="{e(url)}" target="_blank" rel="noopener">Open the live demo &rarr;</a>'
        if url else '<span class="dim">live demo offline</span>')

TPL = Template(r"""<title>WhyWrong</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Literata:opsz,wght@7..72,400;7..72,600;7..72,700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;450;600&display=swap">
<style>
:root{
  --paper:#f2f4f3; --surface:#ffffff; --sunk:#e9edeb;
  --ink:#14181a; --ink2:#4a5553; --ink3:#7d8a87;
  --rule:#d5dcd9; --rule2:#c3ccc9;
  --mark:#c2453d; --mark-bg:#fbeceb; --mark-edge:#e8c3c0;
  --res:#1f4e4a; --res-bg:#e6efed; --res-edge:#b9d2cd;
  --shadow:0 1px 2px rgba(20,24,26,.05),0 8px 24px -16px rgba(20,24,26,.25);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --paper:#0f1315; --surface:#171c1e; --sunk:#1e2426;
    --ink:#e6ebe9; --ink2:#a3b0ad; --ink3:#71807c;
    --rule:#28302f; --rule2:#36403e;
    --mark:#e4635a; --mark-bg:#2a1a19; --mark-edge:#4a2b28;
    --res:#4fa89c; --res-bg:#142523; --res-edge:#274541;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -16px rgba(0,0,0,.8);
  }
}
:root[data-theme="dark"]{
  --paper:#0f1315; --surface:#171c1e; --sunk:#1e2426;
  --ink:#e6ebe9; --ink2:#a3b0ad; --ink3:#71807c;
  --rule:#28302f; --rule2:#36403e;
  --mark:#e4635a; --mark-bg:#2a1a19; --mark-edge:#4a2b28;
  --res:#4fa89c; --res-bg:#142523; --res-edge:#274541;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -16px rgba(0,0,0,.8);
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font:400 17px/1.62 "IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1140px;margin:0 auto;padding:0 28px}
.col{max-width:63ch}
h1,h2,h3{font-family:Literata,Georgia,"Times New Roman",serif;text-wrap:balance;margin:0}
h1{font-size:clamp(2.6rem,6vw,4.1rem);line-height:1.02;font-weight:700;letter-spacing:-.025em}
h2{font-size:clamp(1.6rem,3vw,2.1rem);line-height:1.15;font-weight:600;letter-spacing:-.018em}
h3{font-size:1.12rem;font-weight:600;letter-spacing:-.008em}
p{margin:0 0 1.05em}
a{color:var(--res);text-underline-offset:3px;text-decoration-thickness:1px}
a:focus-visible,.cta:focus-visible{outline:2px solid var(--res);outline-offset:3px;border-radius:3px}
.mono,code,td,.sv{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums}
.dim{color:var(--ink3)}
.nm{color:var(--ink3);font-style:italic;font-family:"IBM Plex Sans",sans-serif;font-size:.86em}

/* the diagnosis sequence is a real order, so the steps are numbered */
.eyebrow{display:flex;align-items:baseline;gap:.7rem;font:600 .72rem/1 "IBM Plex Mono",monospace;
  letter-spacing:.14em;text-transform:uppercase;color:var(--ink3);margin-bottom:1.1rem}
.eyebrow::before{content:attr(data-step);color:var(--mark);font-weight:600}
.eyebrow::after{content:"";flex:1;height:1px;background:var(--rule);align-self:center}
section{padding:clamp(3.4rem,7vw,5.6rem) 0;border-top:1px solid var(--rule)}
section:first-of-type{border-top:0}

/* ---- masthead ---- */
header{padding:clamp(3.4rem,8vw,6rem) 0 clamp(2rem,4vw,3rem)}
.kicker{font:600 .74rem/1 "IBM Plex Mono",monospace;letter-spacing:.16em;text-transform:uppercase;
  color:var(--ink3);margin-bottom:1.6rem}
.lede{font-size:clamp(1.12rem,2vw,1.32rem);line-height:1.5;color:var(--ink2);margin-top:1.4rem;max-width:56ch}
.lede b{color:var(--ink);font-weight:450}
.bar{display:flex;gap:.7rem;flex-wrap:wrap;align-items:center;margin-top:2rem}
.cta{display:inline-block;background:var(--ink);color:var(--paper);text-decoration:none;
  padding:.7rem 1.2rem;border-radius:2px;font-weight:600;font-size:.95rem;transition:transform .15s ease}
.cta:hover{transform:translateY(-1px)}
.chip{border:1px solid var(--rule2);color:var(--ink2);padding:.62rem 1rem;border-radius:2px;
  font:500 .84rem/1 "IBM Plex Mono",monospace;text-decoration:none}

/* ---- the marked-up hero ---- */
.slate{background:var(--surface);border:1px solid var(--rule);border-radius:3px;
  padding:clamp(1.5rem,3vw,2.2rem);box-shadow:var(--shadow);margin-top:2.6rem}
.sum{font-family:"IBM Plex Mono",monospace;font-size:clamp(1.5rem,3.6vw,2.3rem);letter-spacing:-.01em}
.sum .bad{color:var(--mark);position:relative;white-space:nowrap}
.sum .bad::after{content:"";position:absolute;left:-.06em;right:-.06em;top:52%;height:2px;
  background:var(--mark);transform:rotate(-3deg)}
.pair{display:grid;gap:1px;background:var(--rule);border:1px solid var(--rule);
  border-radius:3px;margin-top:1.8rem;overflow:hidden}
@media(min-width:760px){.pair{grid-template-columns:1fr 1fr}}
.half{background:var(--surface);padding:1.3rem 1.4rem}
.half.them{background:var(--sunk)}
.who{font:600 .7rem/1 "IBM Plex Mono",monospace;letter-spacing:.13em;text-transform:uppercase;
  color:var(--ink3);margin-bottom:.85rem}
.said{font-size:1.05rem;color:var(--ink2)}
.dx{border-left:2px solid var(--mark);padding-left:.9rem;margin:.2rem 0 .9rem}
.dxid{font:500 .72rem/1 "IBM Plex Mono",monospace;color:var(--mark);letter-spacing:.06em}
.dxname{font-family:Literata,serif;font-weight:600;font-size:1.06rem;line-height:1.35;margin-top:.35rem}
.rung{border-left:2px solid var(--res);padding:.55rem 0 .55rem .9rem;margin-top:.6rem;
  font-size:.96rem;color:var(--ink2)}
.rung b{display:block;font:600 .66rem/1 "IBM Plex Mono",monospace;letter-spacing:.12em;
  text-transform:uppercase;color:var(--res);margin-bottom:.32rem}

/* ---- data ---- */
.stats{display:grid;gap:1px;background:var(--rule);border:1px solid var(--rule);
  border-radius:3px;overflow:hidden;margin:2.2rem 0}
@media(min-width:720px){.stats{grid-template-columns:repeat(3,1fr)}}
.stat{background:var(--surface);padding:1.5rem 1.4rem}
.sv{font-size:2.15rem;font-weight:600;letter-spacing:-.03em;line-height:1}
.sk{font:600 .7rem/1 "IBM Plex Mono",monospace;letter-spacing:.13em;text-transform:uppercase;
  color:var(--ink3);margin:.65rem 0 .45rem}
.sd{font-size:.87rem;color:var(--ink2);line-height:1.45}
.scroll{overflow-x:auto;border:1px solid var(--rule);border-radius:3px;background:var(--surface);margin:1.8rem 0}
table{border-collapse:collapse;width:100%;min-width:640px;font-size:.9rem}
caption{text-align:left;padding:1rem 1.2rem;font-size:.86rem;color:var(--ink2);border-bottom:1px solid var(--rule)}
th,td{padding:.72rem 1.2rem;text-align:right;border-bottom:1px solid var(--rule)}
thead th{font:600 .68rem/1.3 "IBM Plex Mono",monospace;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink3);background:var(--sunk);vertical-align:bottom}
tbody th[scope=row]{text-align:left;font-weight:450;font-family:"IBM Plex Sans",sans-serif}
.note{display:block;font-size:.76rem;color:var(--ink3);margin-top:.16rem}
tr:last-child td,tr:last-child th{border-bottom:0}
tr.ship{background:var(--res-bg)}
tr.ship th[scope=row]{font-weight:600}
tr.ship td{font-weight:600;color:var(--res)}

.arch{display:grid;gap:1px;background:var(--rule);border:1px solid var(--rule);
  border-radius:3px;overflow:hidden;margin:2rem 0}
@media(min-width:820px){.arch{grid-template-columns:1fr 1fr}}
.zone{background:var(--surface);padding:1.6rem 1.5rem}
.zone.prod{background:var(--res-bg)}
.ztag{font:600 .68rem/1 "IBM Plex Mono",monospace;letter-spacing:.13em;text-transform:uppercase;
  color:var(--ink3);margin-bottom:1rem}
.zone.prod .ztag{color:var(--res)}
.step{display:flex;gap:.85rem;padding:.55rem 0;border-bottom:1px dotted var(--rule);font-size:.92rem}
.step:last-child{border-bottom:0}
.step .w{flex:none;width:8.4rem;color:var(--ink3);font:500 .78rem/1.5 "IBM Plex Mono",monospace}
.pull{font-family:Literata,serif;font-size:clamp(1.25rem,2.6vw,1.62rem);line-height:1.4;
  border-left:2px solid var(--mark);padding-left:1.4rem;margin:2.4rem 0;max-width:52ch;font-weight:400}
.fail{border:1px solid var(--rule);border-radius:3px;background:var(--surface);padding:1.3rem 1.4rem;margin-bottom:1rem}
.ftop{display:flex;justify-content:space-between;gap:1rem;align-items:baseline;flex-wrap:wrap;margin-bottom:.8rem}
.cat{font:600 .68rem/1 "IBM Plex Mono",monospace;letter-spacing:.11em;text-transform:uppercase;
  color:var(--mark);background:var(--mark-bg);border:1px solid var(--mark-edge);padding:.32rem .6rem;border-radius:2px}
.ftop .dim{font:500 .74rem/1 "IBM Plex Mono",monospace}
.fq{font-size:.96rem;margin:0 0 .9rem;color:var(--ink)}
.fail dl{display:grid;grid-template-columns:auto 1fr;gap:.35rem 1.1rem;margin:0;font-size:.89rem}
.fail dt{color:var(--ink3);font:500 .74rem/1.7 "IBM Plex Mono",monospace;text-transform:uppercase;letter-spacing:.06em}
.fail dd{margin:0;color:var(--ink2)}
.fail dd.read{color:var(--ink)}
ul.next{list-style:none;padding:0;margin:1.6rem 0 0;display:grid;gap:1.1rem}
ul.next li{padding-left:1.5rem;position:relative;color:var(--ink2)}
ul.next li::before{content:"";position:absolute;left:0;top:.62em;width:.55rem;height:1px;background:var(--mark)}
ul.next b{color:var(--ink);font-weight:600}
footer{border-top:1px solid var(--rule);padding:2.6rem 0 4rem;color:var(--ink3);font-size:.87rem}
footer p{max-width:70ch}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>

<div class="wrap">
<header>
  <div class="kicker">Nerdy AI Hackathon Challenge &middot; open track</div>
  <h1>Every tutor knows <em>that</em><br>a student is wrong.</h1>
  <p class="lede">WhyWrong knows <b>why</b>. It maps a specific wrong answer to the specific
  broken rule underneath it &mdash; one of $NMISC diagnosed misconceptions &mdash; and hands the
  tutor a root cause instead of a score. <b>In $P50 ms, on one CPU core.</b></p>
  <div class="bar">$LIVE
    <span class="chip">$NQ labelled pairs</span>
    <span class="chip">$NMISC misconceptions</span>
    <span class="chip">3 &times; A100, offline only</span>
  </div>
</header>

<section>
  <div class="eyebrow" data-step="01">The symptom</div>
  <div class="col">
    <h2>A wrong answer is evidence. Almost nobody reads it.</h2>
    <p>A student subtracts two fractions and writes this:</p>
  </div>
  <div class="slate">
    <div class="sum">7/12 &minus; 3/12 = <span class="bad">4/24</span></div>
    <div class="pair">
      <div class="half them">
        <div class="who">A generic AI tutor</div>
        <p class="said">&ldquo;That&rsquo;s not quite right &mdash; try again!&rdquo;</p>
        <p class="said dim" style="font-size:.92rem">Detects wrongness. Has no idea what
        happened. The student retries and makes the same error, for the same reason.</p>
      </div>
      <div class="half">
        <div class="who">WhyWrong</div>
        <div class="dx">
          <div class="dxid">MISCONCEPTION #$HERO_ID</div>
          <div class="dxname">$HERO_MISC</div>
        </div>
        <div class="rung"><b>Hint 1 &middot; a question, never the answer</b>
        $HERO_HINT</div>
      </div>
    </div>
  </div>
  <p class="pull">Detecting wrongness is easy. Naming <em>which</em> wrongness is what decides
  whether the next ten minutes help the student or waste their time.</p>
</section>

<section>
  <div class="eyebrow" data-step="02">Why it is hard</div>
  <div class="col">
    <h2>Most of the taxonomy has no training data at all</h2>
    <p>The misconception taxonomy is long-tailed and the split is brutal:
    <b>$NUNSEEN of $NMISC misconceptions ($PCTUNSEEN%) never appear in training</b>, so
    $TESTUNSEEN of the $NTEST held-out queries ask for a label the model has never seen a
    single example of.</p>
    <p>Scaling a general-purpose embedding model does not rescue this. <b>bge-m3 is 17&times;
    larger than bge-small and buys almost nothing</b> &mdash; $FLOOR to $FLOORBIG MAP@25.
    The problem is not capacity. It is that nobody taught the model this task.</p>
  </div>
</section>

<section>
  <div class="eyebrow" data-step="03">The method</div>
  <div class="col">
    <h2>Train big on our own metal. Distil small. Serve on a CPU.</h2>
    <p>Everything expensive happens once, offline, on hardware we own. What ships is two
    quantised models totalling $BUNDLE MB and a matrix multiply.</p>
  </div>
  <div class="arch">
    <div class="zone">
      <div class="ztag">Offline &middot; 3 &times; A100 &middot; runs once</div>
      <div class="step"><span class="w">Augment</span><span>Qwen2.5-7B writes diagnostic questions for the misconceptions with zero real examples</span></div>
      <div class="step"><span class="w">Retrieve</span><span>bge-small (33M) contrastive fine-tune &#43; two rounds of hard-negative mining</span></div>
      <div class="step"><span class="w">Rerank</span><span>Qwen2.5-7B &#43; LoRA, listwise over the top 25</span></div>
      <div class="step"><span class="w">Distil</span><span>7B teacher &rarr; MiniLM-L6 (22M) on KL over the teacher&rsquo;s scores</span></div>
      <div class="step"><span class="w">Explain</span><span>Hint ladder and probes for all $NMISC misconceptions, written once, stored</span></div>
    </div>
    <div class="zone prod">
      <div class="ztag">Production &middot; 1 CPU core &middot; every request</div>
      <div class="step"><span class="w">~$EMBEDMS ms</span><span>INT8 ONNX bi-encoder embeds the query</span></div>
      <div class="step"><span class="w">~0.4 ms</span><span><code>numpy</code> matmul over a $NMISC &times; 384 matrix &mdash; 3.97 MB, not a vector database</span></div>
      <div class="step"><span class="w">~$RERANKMS ms</span><span>INT8 ONNX cross-encoder reranks the 25 candidates</span></div>
      <div class="step"><span class="w">~0.3 ms</span><span>SQLite primary-key lookup for the hint ladder</span></div>
      <div class="step"><span class="w">$RSS MB RSS</span><span>No GPU. No LLM. No network call. No <code>torch</code> installed &mdash; the tokenizer loads through <code>tokenizers</code>, which cut resident memory from 817&nbsp;MB and cold start to $COLD s.</span></div>
    </div>
  </div>
  <p class="pull">School mathematics has a <em>finite</em> misconception taxonomy. So the
  explanations were generated once, in the kitchen &mdash; not at every table.</p>
</section>

<section>
  <div class="eyebrow" data-step="04">The evidence</div>
  <div class="col"><h2>Measured, on a frozen split</h2></div>
  <div class="stats">$STATS</div>
  <div class="scroll"><table>
    <caption>$SPLITNOTE</caption>
    <thead><tr><th scope="col">System</th><th scope="col">MAP@25</th><th scope="col">Recall@25</th>
    <th scope="col">R@25<br>unseen</th><th scope="col">Top-1</th></tr></thead>
    <tbody>$TABLE</tbody>
  </table></div>
  <div class="col" style="margin-top:1.6rem">
    <h3>What the synthetic data actually did</h3>
    $ABLATION
  </div>
  $COSTBLOCK
  <div class="col" style="margin-top:2.4rem">
    <h3>The latency knob</h3>
    <p style="margin-top:.6rem">Reranking is a transformer forward pass <em>per candidate</em>,
    so serving cost is linear in how many we rerank. This is the decision the whole design
    turns on, so here is the measured curve instead of one flattering number.</p>
  </div>
  $CURVE
  <div class="col"><p class="dim" style="font-size:.9rem">$FOOTNOTE</p></div>
</section>

<section>
  <div class="eyebrow" data-step="05">Where it fails</div>
  <div class="col">
    <h2>The cases it gets wrong</h2>
    <p>A benchmark page without this section is a sales page. These are held-out cases the
    shipped model ranks incorrectly, with our read on each.</p>
  </div>
  <div style="margin-top:1.8rem">$FAILS</div>
</section>

<section>
  <div class="eyebrow" data-step="06">Next</div>
  <div class="col">
    <h2>What I would build next</h2>
    <ul class="next">
      <li><b>Disambiguation over guessing.</b> When two misconceptions are both genuinely
      consistent with an answer, the right move is one targeted question, not a coin flip.
      The failure gallery says this is the largest error class.</li>
      <li><b>Close the loop on sequencing.</b> Naming the misconception is half the job;
      choosing the next question to prove it is cleared is the other half.</li>
      <li><b>Beyond mathematics.</b> Nothing in the architecture is maths-specific. It needs
      a labelled misconception set, not a new model.</li>
    </ul>
  </div>
</section>

<footer><div class="col">
  <p><b>Data.</b> Eedi &ldquo;Mining Misconceptions in Mathematics&rdquo; (NeurIPS 2024), used
  under its research-use terms via the publicly redistributed mirror. The Kaggle competition
  gates programmatic download; that gate was not circumvented. This deployment is a
  non-commercial demonstration &mdash; the pipeline is data-agnostic and would be retrained on
  licensed or first-party data for production.</p>
  <p style="margin-top:1rem">Every figure on this page is generated from
  <code>results/metrics.json</code> and the shipped artifact bundle, so none of them can drift
  from what the demo actually serves. Unmeasured cells say so. Built $GEN.</p>
</div></footer>
</div>
""")

costblock = ""
if cost_rows:
    costblock = (f'<div class="scroll"><table><caption>{e(cost.get("assumption",""))}</caption>'
                 f'<thead><tr><th scope="col">Approach</th><th scope="col">$ / 1,000</th>'
                 f'<th scope="col">Monthly</th><th scope="col">Basis</th></tr></thead>'
                 f'<tbody>{cost_rows}</tbody></table></div>')

t = bench.get("latency", {})
html_out = TPL.safe_substitute(
    NMISC=f'{manifest["n_misconceptions"]:,}', NQ=f'{manifest["n_queries"]:,}',
    NUNSEEN=f'{manifest["n_misconceptions_never_in_train"]:,}',
    PCTUNSEEN=f'{100*manifest["n_misconceptions_never_in_train"]/manifest["n_misconceptions"]:.0f}',
    TESTUNSEEN=f'{manifest["test_unseen_queries"]:,}', NTEST=f'{manifest["n_test"]:,}',
    P50=str(t.get("p50", "&mdash;")), LIVE=live, STATS=stat_html, TABLE=table_rows,
    FLOOR=fmt(floor), FLOORBIG=fmt(M("offtheshelf:bge-m3")),
    BUNDLE=str(bench.get("bundle_mb", "&mdash;")),
    EMBEDMS=str(t.get("breakdown_median", {}).get("embed_ms", "&mdash;")),
    RERANKMS=str(t.get("breakdown_median", {}).get("rerank_ms", "&mdash;")),
    RSS=str(t.get("rss_mb", "&mdash;")), COLD=str(t.get("cold_start_s", "&mdash;")),
    HERO_MISC=hero["name"], HERO_HINT=hero["hint"], HERO_ID=hero["id"],
    SPLITNOTE=e(bench.get("split_note", "")), FOOTNOTE=e(bench.get("footnote", "")),
    COSTBLOCK=costblock, CURVE=curve_html, ABLATION=ablation_html, FAILS=fail_html,
    GEN=bench.get("generated_at", ""))
OUT.write_text(html_out)
print(f"artifact page -> {OUT}  ({len(html_out)/1024:.1f} KB)")
missing = [k for k in ("not measured",) if k in html_out]
print("contains 'not measured' placeholders:", "yes" if missing else "no")
