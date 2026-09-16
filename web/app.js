const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const api = (p, o) => fetch(p, o).then(r => r.ok ? r.json() : r.json().then(e => Promise.reject(e)));
const pct = x => (100 * x).toFixed(0) + '%';

/* LaTeX in this dataset is light: \( .. \), \[ .. \], \frac{a}{b}. Render it
   readably without pulling in a 300 KB typesetting library the CSP would block. */
function tex(s) {
  return esc(s ?? '')
    // Eedi questions sometimes carry an image reference we do not have the asset for.
    // Say so rather than printing raw markdown at the reader.
    .replace(/!\[[^\]]*\]\([^)]*\)/g, ' [diagram] ')
    .replace(/!\[[^\]]*\]/g, ' [diagram] ')
    .replace(/\\begin\{[^}]*\}|\\end\{[^}]*\}/g, ' ')
    .replace(/\\hline/g, ' ')
    .replace(/\\(?:mathbf|mathrm|text|textbf|mathit)\{([^{}]*)\}/g, (m, g) => g.replace(/\s+/g, ''))
    .replace(/\\[[\]]|\\[()]/g, '')
    // one level of brace nesting, so \frac{5x^2}{2y(x-3)} survives
    .replace(/\\d?frac\{((?:[^{}]|\{[^{}]*\})*)\}\{((?:[^{}]|\{[^{}]*\})*)\}/g,
             '<span class="mono">($1)/($2)</span>')
    .replace(/\\sqrt\{([^{}]*)\}/g, '√($1)')
    .replace(/\^\{?\\?(?:circ|degree)\}?/g, '°')
    .replace(/\\times/g, '×').replace(/\\div/g, '÷').replace(/\\pm/g, '±')
    .replace(/\\neq/g, '≠').replace(/\\leq/g, '≤').replace(/\\geq/g, '≥')
    .replace(/\\degree|\\circ/g, '°').replace(/\\pi/g, 'π').replace(/\\cdot/g, '·')
    .replace(/\\([%$#&_])/g, '$1')
    .replace(/\\\\/g, ' ')
    .replace(/\\[a-zA-Z]+/g, '')
    .replace(/~/g, ' ')
    .replace(/[{}]/g, '')
    .replace(/&amp;/g, ' ')
    .replace(/\^(\d)/g, (m, d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+d])
    .replace(/\s+/g, ' ').trim();
}

/* ---------------- nav ---------------- */
function switchView(name, push = true) {
  const btn = $(`nav button[data-v="${name}"]`);
  if (!btn) return;
  $$('nav button').forEach(x => x.classList.toggle('on', x === btn));
  $$('.view').forEach(v => v.classList.toggle('on', v.id === 'v-' + name));
  if (name === 'benchmark') loadBench();
  if (name === 'how') loadHow();
  if (push) {
    const target = name === 'student' ? '/' : '/' + name;
    if (window.location.pathname !== target) history.pushState({ v: name }, '', target);
  }
}
$$('nav button').forEach(b => b.onclick = () => switchView(b.dataset.v));
window.onpopstate = e => {
  const v = e.state?.v || window.location.pathname.replace(/^\//, '') || 'student';
  switchView(v, false);
};

/* ---------------- student ---------------- */
let EX = [], cur = 0, answered = false;

function renderQ() {
  const e = EX[cur];
  if (!e) { $('#qbox').innerHTML = '<p class="muted">No demo problems bundled.</p>'; return; }
  answered = false;
  $('#qbox').innerHTML = `
    <div class="qcard">
      <div class="muted">${esc(e.subject)} · ${esc(e.construct)}</div>
      <div class="qtext">${tex(e.question)}</div>
      <div class="opts">${e.options.map((o, i) =>
        `<div class="opt" data-i="${i}"><b>${'ABCD'[i]}</b><span>${tex(o)}</span></div>`).join('')}</div>
    </div>
    <div style="display:flex;gap:9px;align-items:center">
      <button class="ghost" id="nextq">Next problem →</button>
      <span class="muted">${cur + 1} of ${EX.length}</span>
    </div>`;
  $('#nextq').onclick = () => { cur = (cur + 1) % EX.length; renderQ(); };
  $$('#qbox .opt').forEach(el => el.onclick = () => pick(el, e, +el.dataset.i));
}

async function pick(el, e, i) {
  if (answered) return;
  answered = true;
  const chosen = e.options[i], right = chosen === e.correct;
  $$('#qbox .opt').forEach(o => {
    const t = e.options[+o.dataset.i];
    if (t === e.correct) o.classList.add('right');
    else if (o === el) o.classList.add('wrong');
  });
  if (right) {
    $('#resultbox').innerHTML = `<h3>Diagnosis</h3>
      <span class="pill good">Correct</span>
      <p style="margin-top:11px">Nothing to diagnose. WhyWrong only fires on a wrong answer —
      that's the whole point: it reads the <em>error</em>, not the score.</p>
      <p class="muted" style="margin-top:9px">Try a wrong option to see the diagnosis.</p>`;
    return;
  }
  $('#resultbox').innerHTML = '<h3>Diagnosis</h3><p class="muted"><span class="spinner"></span> diagnosing…</p>';
  try {
    const r = await api('/api/diagnose', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: e.question, correct: e.correct, student_answer: chosen,
                             subject: e.subject, construct: e.construct, top_n: 3 })
    });
    renderDiag(r, $('#resultbox'), e);
  } catch (err) { $('#resultbox').innerHTML = `<h3>Diagnosis</h3><p class="pill bad">error: ${esc(err.detail || err)}</p>`; }
}

function renderDiag(r, box, ctx) {
  const c = r.candidates[0], t = r.timing;
  const alt = r.candidates.slice(1);
  box.innerHTML = `
    <h3>Diagnosis</h3>
    <span class="pill acc">confidence ${pct(c.confidence)}</span>
    <span class="pill">${esc(c.topic || 'Mathematics')}</span>
    <span class="pill">${t.total_ms} ms · CPU</span>
    <div class="diag">
      <div class="muted">Misconception #${c.misconception_id}</div>
      <div class="diagname">${esc(c.misconception_name)}</div>
      <div class="bar"><i style="width:${100 * c.confidence}%"></i></div>
      ${c.plain ? `<p style="margin-top:12px">${esc(c.plain)}</p>` : ''}
      ${c.why ? `<p class="muted" style="margin-top:8px">${esc(c.why)}</p>` : ''}
    </div>
    ${c.hints?.length ? `<h3 style="margin-top:18px">Socratic hint ladder</h3>
      <div class="ladder" id="ladder"></div>
      <button class="ghost" id="hintbtn" style="margin-top:10px">Reveal first hint</button>` : ''}
    ${alt.length ? `<h3 style="margin-top:18px">Other candidates</h3>
      <div class="timeline">${alt.map(a => `<div class="tl"><span class="dot" style="background:var(--ink3)"></span>
        <div><div>${esc(a.misconception_name)}</div>
        <div class="muted">confidence ${pct(a.confidence)}</div></div></div>`).join('')}</div>` : ''}
    <div id="probe"></div>
    <h3 style="margin-top:18px">Where the time went</h3>
    <div class="lat">
      <div style="flex:${t.embed_ms};background:#5b8cff">embed ${t.embed_ms}</div>
      <div style="flex:${Math.max(t.search_ms, .05)};background:#7c5cff">search ${t.search_ms}</div>
      <div style="flex:${Math.max(t.rerank_ms, .05)};background:#3ddc97">rerank ${t.rerank_ms}</div>
      <div style="flex:${Math.max(t.lookup_ms, .05)};background:#ffb545">db ${t.lookup_ms}</div>
    </div>
    <p class="muted" style="margin-top:7px">ms, single CPU core. The hint ladder was written
    offline and fetched with a primary-key lookup — no LLM in this path.</p>`;

  // offer a follow-up chosen to probe THIS misconception, not just the next item
  if (ctx) probeNext(c.misconception_id, ctx.question_id);

  if (c.hints?.length) {
    let n = 0;
    const labels = ['Hint 1 — a question, not an answer', 'Hint 2 — a concrete nudge', 'Hint 3 — the rule'];
    $('#hintbtn').onclick = () => {
      if (n >= c.hints.length) return;
      $('#ladder').insertAdjacentHTML('beforeend',
        `<div class="rung"><span class="n">${labels[n]}</span>${esc(c.hints[n])}</div>`);
      n++;
      $('#hintbtn').textContent = n >= c.hints.length ? 'Ladder complete' : 'Next hint';
      $('#hintbtn').disabled = n >= c.hints.length;
    };
  }
}

async function probeNext(mid, excludeQid) {
  try {
    const r = await api(`/api/next_problem?after_misconception_id=${mid}` +
                        `&exclude_question_id=${excludeQid ?? -1}`);
    const box = $('#probe');
    if (!box) return;
    box.innerHTML = `<h3 style="margin-top:18px">Next problem, chosen to probe this</h3>
      <div class="rung" style="border-left-color:var(--warn)">
        <span class="n">${esc(r.reason)}</span>
        ${tex(r.problem.question).slice(0, 150)}
        <div class="muted" style="margin-top:6px">targets: ${esc(r.targets || '')}</div>
        <button class="ghost" id="probego" style="margin-top:9px">Serve this problem →</button>
      </div>`;
    $('#probego').onclick = () => {
      let idx = EX.findIndex(e => e.question_id === r.problem.question_id);
      if (idx < 0) { EX.splice(cur + 1, 0, r.problem); idx = cur + 1; }   // not in the carousel yet
      cur = idx; renderQ(); window.scrollTo({ top: 0, behavior: 'smooth' });
    };
  } catch {}
}

/* ---------------- tutor ---------------- */
const SID = 'demo-' + Math.random().toString(36).slice(2, 8);

$('#resetbtn').onclick = async () => {
  await api(`/api/session/${SID}/reset`, { method: 'POST' });
  $('#tutorbox').innerHTML = '<div class="card"><p class="muted">Session reset.</p></div>';
};

$('#simbtn').onclick = async () => {
  const btn = $('#simbtn'); btn.disabled = true;
  await api(`/api/session/${SID}/reset`, { method: 'POST' });
  let script = [];
  try { script = (await api('/api/session_script')).steps || []; } catch {}
  if (!script.length) script = EX.filter(e => e.distractor).slice(0, 6)
    .map(e => ({ ...e, student_answer: e.distractor }));
  for (let i = 0; i < script.length; i++) {
    $('#simstat').innerHTML = `<span class="spinner"></span> question ${i + 1} of ${script.length} — ${esc(script[i].subject)}`;
    const e = script[i];
    await api('/api/diagnose', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: e.question, correct: e.correct,
                             student_answer: e.student_answer, subject: e.subject,
                             construct: e.construct, session_id: SID, student_name: 'Priya' })
    });
    await renderTutor();
  }
  $('#simstat').textContent = `Session complete — ${script.length} wrong answers across ` +
    `${new Set(script.map(s => s.subject)).size} different topics.`;
  btn.disabled = false;
};

async function renderTutor() {
  const s = await api(`/api/session/${SID}`);
  if (!s.n_errors) { $('#tutorbox').innerHTML = '<div class="card"><p class="muted">No errors yet.</p></div>'; return; }
  const r = s.root_causes[0];
  $('#tutorbox').innerHTML = `
    <div class="card" style="margin-bottom:16px">
      <div class="tutorbar">
        <div class="avatar">${esc(s.student[0])}</div>
        <div><div style="font-weight:650">${esc(s.student)}</div>
             <div class="muted">${esc(r.topic || 'Mathematics')} · ${Math.round(s.duration_s)}s into session ·
             ${s.n_errors} error${s.n_errors > 1 ? 's' : ''}</div></div>
        <span class="pill ${s.concentration >= .5 ? 'warn' : ''}" style="margin-left:auto">
          ${pct(s.concentration)} of errors share one cause</span>
      </div>
    </div>
    <div class="alert">
      <span class="pill warn">⚠ Root cause</span>
      <span class="pill">confidence ${pct(r.mean_confidence)}</span>
      <span class="pill">first seen ${Math.round(r.first_seen_s_ago)}s ago</span>
      <div class="rootname">#${r.misconception_id} — ${esc(r.misconception_name)}</div>
      <p>${esc(r.plain)}</p>
      ${r.why ? `<p class="muted" style="margin-top:8px">${esc(r.why)}</p>` : ''}
      <div class="opener"><strong style="font-style:normal">Suggested opener:</strong> “${esc(r.opener)}”</div>
      ${r.probes?.length ? `<h3 style="margin-top:16px">Probes to check it's cleared</h3>
        <div class="timeline">${r.probes.map(p =>
          `<div class="tl"><span class="dot" style="background:var(--good)"></span><div>${esc(p)}</div></div>`).join('')}</div>` : ''}
      ${r.also_at_risk?.length ? `<h3 style="margin-top:16px">Also at risk</h3>
        <div class="chips">${r.also_at_risk.map(a => `<span class="pill">${esc(a.misconception_name)}</span>`).join('')}</div>` : ''}
    </div>
    <div class="card" style="margin-top:16px">
      <h3>Error timeline</h3>
      <div class="timeline">${s.events.map(e => `<div class="tl"><span class="dot"></span>
        <div><div>${tex(e.question).slice(0, 110)}…</div>
        <div class="muted">answered <span class="mono">${tex(e.student_answer)}</span> →
        ${esc(e.misconception_name)} (${pct(e.confidence)})</div></div></div>`).join('')}</div>
    </div>
    ${s.root_causes.length > 1 ? `<div class="card"><h3>Secondary causes</h3>
      <div class="timeline">${s.root_causes.slice(1).map(x => `<div class="tl">
        <span class="dot" style="background:var(--ink3)"></span><div><div>${esc(x.misconception_name)}</div>
        <div class="muted">${x.n_errors} error(s) · ${pct(x.share)} of session</div></div></div>`).join('')}</div></div>` : ''}`;
}

/* ---------------- try it ---------------- */
$('#trybtn').onclick = async () => {
  $('#trybox').innerHTML = '<h3>Result</h3><p class="muted"><span class="spinner"></span> diagnosing…</p>';
  try {
    const r = await api('/api/diagnose', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: $('#t-q').value, correct: $('#t-c').value,
        student_answer: $('#t-a').value, subject: $('#t-s').value,
        construct: $('#t-t').value, top_n: 3 })
    });
    renderDiag(r, $('#trybox'));
  } catch (e) { $('#trybox').innerHTML = `<h3>Result</h3><p class="pill bad">${esc(e.detail || e)}</p>`; }
};

/* ---------------- benchmark ---------------- */
let benchLoaded = false;
async function loadBench() {
  if (benchLoaded) return; benchLoaded = true;
  let b;
  try { b = await api('/api/benchmark'); }
  catch { $('#benchbox').innerHTML = '<div class="card"><p class="muted">Benchmark not built yet.</p></div>'; benchLoaded = false; return; }

  const cell = (v, d = 4) => v === null || v === undefined ? '<span class="muted">not measured</span>'
    : (typeof v === 'number' ? v.toFixed(d) : esc(v));
  const rows = b.systems.map(s => `<tr class="${s.shipped ? 'ship' : ''}">
      <td>${esc(s.name)}${s.shipped ? ' <span class="pill acc">shipped</span>' : ''}</td>
      <td class="num">${cell(s.params_M, 0)}</td>
      <td class="num">${cell(s.map25)}</td>
      <td class="num">${cell(s.recall25)}</td>
      <td class="num">${cell(s.recall25_unseen)}</td>
      <td class="num">${cell(s.top1)}</td>
      <td class="num">${cell(s.p50_ms, 1)}</td>
      <td class="num">${cell(s.cost_per_1k_usd, 4)}</td>
      <td>${esc(s.runs_on)}</td></tr>`).join('');

  $('#benchbox').innerHTML = `
    <div class="grid g3" style="margin-bottom:18px">
      ${(b.headline || []).map(h => `<div class="stat"><div class="v" style="color:${h.color || 'var(--ink)'}">${esc(h.value)}</div>
        <div class="k">${esc(h.key)}</div><div class="d">${esc(h.detail)}</div></div>`).join('')}
    </div>
    <div class="card">
      <h2>Held-out results</h2>
      <p class="sub">${esc(b.split_note || '')}</p>
      <div class="tw"><table><thead><tr>
        <th>System</th><th>Params (M)</th><th>MAP@25</th><th>Recall@25</th>
        <th>R@25 unseen</th><th>Top-1</th><th>p50 ms</th><th>$ / 1k</th><th>Runs on</th>
      </tr></thead><tbody>${rows}</tbody></table></div>
      <p class="muted" style="margin-top:11px">${esc(b.footnote || '')}</p>
    </div>
    ${b.live_latency && b.live_latency.n ? `<div class="card"><h2>Live latency</h2>
      <p class="sub">Measured on this server, including your clicks. n=${b.live_latency.n} requests.</p>
      <div class="grid g3">
        <div class="stat"><div class="v">${b.live_latency.p50}</div><div class="k">p50 ms</div></div>
        <div class="stat"><div class="v">${b.live_latency.p95}</div><div class="k">p95 ms</div></div>
        <div class="stat"><div class="v">${b.live_latency.p99}</div><div class="k">p99 ms</div></div>
      </div></div>` : ''}
    ${b.cost ? `<div class="card"><h2>Cost at scale</h2>
      <p class="sub">${esc(b.cost.assumption)}</p>
      <div class="tw"><table><thead><tr><th>Approach</th><th>$ / 1k diagnoses</th><th>Monthly</th><th>Basis</th></tr></thead>
      <tbody>${b.cost.rows.map(r => `<tr class="${r.shipped ? 'ship' : ''}"><td>${esc(r.name)}</td>
        <td class="num">${esc(r.per_1k)}</td><td class="num ${r.shipped ? 'win' : ''}">${esc(r.monthly)}</td>
        <td class="muted">${esc(r.basis)}</td></tr>`).join('')}</tbody></table></div></div>` : ''}
    <div class="card" id="failbox"><h2>Where it fails</h2>
      <p class="sub">Ten held-out cases the shipped model gets wrong, with our read on why.
      A benchmark page without this is a sales page.</p>
      <div id="faillist"><span class="spinner"></span></div></div>`;

  try {
    const f = await api('/api/failures');
    $('#faillist').innerHTML = f.length ? f.map(x => `<div class="fail">
      <span class="pill bad">${esc(x.category)}</span>
      <span class="pill">gold rank ${x.gold_rank === null ? '>25' : x.gold_rank}</span>
      <div style="margin-top:9px">${tex(x.question)}</div>
      <dl class="kv">
        <dt>Student chose</dt><dd class="mono">${tex(x.student_answer)}</dd>
        <dt>Gold label</dt><dd>${esc(x.gold_name)}</dd>
        <dt>We predicted</dt><dd>${esc(x.pred_name)}</dd>
        <dt>Our read</dt><dd>${esc(x.note)}</dd>
      </dl></div>`).join('') : '<p class="muted">Gallery not built yet.</p>';
  } catch { $('#faillist').innerHTML = '<p class="muted">Gallery not built yet.</p>'; }
}

/* ---------------- how ---------------- */
let howLoaded = false;
async function loadHow() {
  if (howLoaded) return; howLoaded = true;
  let b = {}; try { b = await api('/api/benchmark'); } catch {}
  $('#howbox').innerHTML = `
    <div class="card"><h2>The pipeline</h2>
      <div class="tw"><table><thead><tr><th>Stage</th><th>Where</th><th>Model</th><th>What it does</th></tr></thead><tbody>
        <tr><td>Synthetic augmentation</td><td>3 × A100</td><td>Qwen2.5-7B-Instruct</td>
            <td>Writes diagnostic questions for the 1,498 misconceptions with zero real examples</td></tr>
        <tr><td>Retriever</td><td>1 × A100</td><td>bge-small-en-v1.5 (33M)</td>
            <td>Contrastive fine-tune + hard-negative mining · 2,587 → 25</td></tr>
        <tr><td>Teacher reranker</td><td>3 × A100</td><td>Qwen2.5-7B + LoRA</td>
            <td>Listwise reranking. Never deployed — exists to be distilled</td></tr>
        <tr><td>Student reranker</td><td>3 × A100</td><td>MiniLM-L6 (22M)</td>
            <td>KL-distilled from the 7B · 25 → 1</td></tr>
        <tr><td>Explanations</td><td>3 × A100</td><td>Qwen2.5-7B-Instruct</td>
            <td>Hint ladder + probes for all 2,587, generated once, stored in SQLite</td></tr>
        <tr class="ship"><td>Serving</td><td>1 CPU core</td><td>2 × INT8 ONNX</td>
            <td>Everything above, compressed into ~${esc(b.bundle_mb ?? '?')} MB and a matmul</td></tr>
      </tbody></table></div>
    </div>
    <div class="card"><h2>The bet</h2>
      <p>School mathematics has a <strong>finite</strong> misconception taxonomy — about 2,587 of them.
      A finite set means the expensive part, writing a good Socratic hint ladder, can be done
      <strong>once, offline, on our own GPUs</strong> and stored. Production is then a primary-key lookup.</p>
      <p style="margin-top:11px">You don't need a frontier model in the hot path to do this well.
      You need it once, in the kitchen — not at every table.</p>
      <p style="margin-top:11px" class="muted">The vector index is 2,587 × 384 floats = 3.8 MB.
      That's a <code>numpy</code> matmul, not a vector database.</p>
    </div>
    <div class="card"><h2>Data &amp; licensing</h2>
      <p>Trained and evaluated on Eedi's publicly released <em>Mining Misconceptions in Mathematics</em>
      research dataset under its research-use terms. This deployment is a non-commercial demonstration.
      For production the same pipeline would be retrained on licensed or first-party data —
      the architecture is data-agnostic.</p>
    </div>`;
}

/* ---------------- boot ---------------- */
(async () => {
  try {
    const h = await api('/api/health');
    $('#foot-health').textContent =
      `${h.n_misconceptions} misconceptions · pipeline ${h.pipeline} · cold start ${h.load_s}s`;
  } catch {}
  try { EX = await api('/api/examples'); } catch { EX = []; }
  renderQ();
  const initial = window.location.pathname.replace(/^\//, '') || window.location.hash.replace(/^#/, '');
  if (initial && ['student', 'tutor', 'try', 'benchmark', 'how'].includes(initial)) {
    switchView(initial, false);
  }
})();
