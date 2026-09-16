"""WhyWrong API + demo site. One process, one CPU core, no GPU."""
from __future__ import annotations
import json, os, sqlite3, time, uuid, warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine import WhyWrongEngine

ROOT = Path(__file__).resolve().parent
WEB = ROOT.parent / "web"

app = FastAPI(title="WhyWrong", version="1.0",
              description="Misconception-level diagnosis for adaptive practice.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

engine: Optional[WhyWrongEngine] = None
SESSIONS: dict[str, dict] = defaultdict(lambda: {"events": [], "student": "Student", "started": time.time()})


@app.on_event("startup")
def _load():
    global engine
    t0 = time.time()
    engine = WhyWrongEngine(rerank_k=int(os.environ.get("WW_RERANK_K", "25")),
                            threads=int(os.environ.get("WW_THREADS", "1")))
    app.state.load_s = round(time.time() - t0, 2)
    app.state.started = time.time()
    print(f"engine loaded in {app.state.load_s}s", flush=True)


class DiagnoseIn(BaseModel):
    model_config = {"protected_namespaces": ()}
    question: str = Field(..., min_length=1)
    student_answer: str = Field(..., min_length=1)
    correct: str = ""
    subject: str = ""
    construct: str = ""
    top_n: int = 3
    session_id: Optional[str] = None
    student_name: Optional[str] = None


@app.get("/api/health")
def health():
    return dict(ok=engine is not None, load_s=getattr(app.state, "load_s", None),
                uptime_s=round(time.time() - getattr(app.state, "started", time.time()), 1),
                n_misconceptions=int(len(engine.ids)) if engine else 0,
                rerank_k=engine.rerank_k if engine else None,
                threads=engine.threads if engine else None,
                pipeline="bi+cross" if (engine and engine.ce is not None) else "bi-only")


@app.post("/api/diagnose")
def diagnose(inp: DiagnoseIn):
    if engine is None:
        raise HTTPException(503, "engine not loaded")
    res = engine.diagnose(inp.subject, inp.construct, inp.question,
                          inp.correct, inp.student_answer, top_n=max(1, min(inp.top_n, 10)))
    if inp.session_id:
        s = SESSIONS[inp.session_id]
        if inp.student_name:
            s["student"] = inp.student_name
        top = res["candidates"][0]
        s["events"].append(dict(
            t=time.time(), question=inp.question, student_answer=inp.student_answer,
            correct=inp.correct, subject=inp.subject, construct=inp.construct,
            misconception_id=top["misconception_id"], misconception_name=top["misconception_name"],
            confidence=top["confidence"], topic=top.get("topic", "")))
        res["session_id"] = inp.session_id
    return res


@app.get("/api/session/{sid}")
def session(sid: str):
    """The tutor panel: collapse a stream of individual errors into a root cause."""
    if engine is None:
        raise HTTPException(503, "engine not loaded")
    s = SESSIONS.get(sid)
    if not s or not s["events"]:
        return dict(session_id=sid, student=s["student"] if s else "Student",
                    n_errors=0, root_causes=[], events=[])
    ev = s["events"]
    clusters = engine.cluster_errors([e["misconception_id"] for e in ev])
    roots = []
    for cl in clusters[:3]:
        member_set = set(cl["members"])
        hits = [e for e in ev if e["misconception_id"] in member_set]
        # represent the cluster by its highest-confidence member, not just the modal id
        lead = max(hits, key=lambda h: h["confidence"])["misconception_id"]
        expl = engine.explain(int(lead))
        roots.append(dict(
            misconception_id=int(lead), misconception_name=expl["misconception_name"],
            plain=expl["plain"], why=expl["why"], opener=expl["opener"],
            topic=expl.get("topic", ""), hints=expl["hints"], probes=expl["probes"],
            n_errors=cl["n"], share=round(cl["n"] / len(ev), 3),
            cluster_size=len(cl["members"]),
            cluster_members=[dict(misconception_id=int(m),
                                  misconception_name=engine.names[engine.pos[int(m)]])
                             for m in cl["members"]],
            topics_spanned=sorted({h["subject"] for h in hits if h["subject"]}),
            mean_confidence=round(sum(h["confidence"] for h in hits) / len(hits), 3),
            first_seen_s_ago=round(time.time() - min(h["t"] for h in hits), 1),
            also_at_risk=engine.neighbours(int(lead), 3),
            examples=[dict(question=h["question"], student_answer=h["student_answer"],
                           correct=h["correct"]) for h in hits[:3]]))
    top_n = clusters[0]["n"] if clusters else 0
    return dict(session_id=sid, student=s["student"], n_errors=len(ev),
                duration_s=round(time.time() - s["started"], 1),
                concentration=round(top_n / len(ev), 3),
                root_causes=roots,
                events=[dict(question=e["question"], student_answer=e["student_answer"],
                             misconception_name=e["misconception_name"],
                             subject=e.get("subject", ""),
                             confidence=e["confidence"],
                             s_ago=round(time.time() - e["t"], 1)) for e in ev[-12:]])


@app.post("/api/session/{sid}/reset")
def reset(sid: str):
    SESSIONS.pop(sid, None)
    return dict(ok=True, session_id=sid)


@app.get("/api/benchmark")
def benchmark():
    p = ROOT / "artifacts/benchmark.json"
    if not p.exists():
        raise HTTPException(404, "benchmark.json not built yet")
    d = json.loads(p.read_text())
    d["live_latency"] = engine.latency_stats() if engine else None
    return d


@app.get("/api/examples")
def examples():
    p = ROOT / "artifacts/examples.json"
    return json.loads(p.read_text()) if p.exists() else []


@app.get("/api/next_problem")
def next_problem(after_misconception_id: int, exclude_question_id: int = -1):
    """Pick the next problem to serve after a diagnosis.

    Not 'the next item in the list'. We choose the demo problem whose own gold
    misconception sits closest to the one just diagnosed, so the follow-up actually
    probes whether that specific broken rule is still there -- which is the whole
    point of diagnosing it rather than just scoring the answer.
    """
    if engine is None:
        raise HTTPException(503, "engine not loaded")
    p = ROOT / "artifacts/probe_pool.json"
    if not p.exists():
        p = ROOT / "artifacts/examples.json"
    if not p.exists():
        raise HTTPException(404, "no examples bundled")
    pool = [e for e in json.loads(p.read_text()) if e["question_id"] != exclude_question_id]
    if not pool:
        raise HTTPException(404, "pool exhausted")
    i = engine.pos.get(int(after_misconception_id))
    if i is None:
        return dict(problem=pool[0], reason="unknown misconception; served next available")
    scored = []
    for e in pool:
        j = engine.pos.get(int(e["gold_misconception_id"]))
        scored.append((float(engine.E[i] @ engine.E[j]) if j is not None else -1.0, e))
    scored.sort(key=lambda x: -x[0])
    sim, best = scored[0]
    return dict(problem=best, similarity=round(sim, 4),
                targets=engine.names[engine.pos[int(best["gold_misconception_id"])]],
                reason=("probes the same misconception family" if sim >= 0.62
                        else "no close probe in the demo pool; serving the nearest available"))


@app.get("/api/session_script")
def session_script():
    p = ROOT / "artifacts/session_script.json"
    return json.loads(p.read_text()) if p.exists() else {"steps": []}


@app.get("/api/failures")
def failures():
    p = ROOT / "artifacts/failure_gallery.json"
    if not p.exists():
        p = ROOT / "artifacts/failure_gallery_analysed.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


@app.get("/api/misconceptions")
def misconceptions(q: str = "", limit: int = 25):
    if engine is None:
        raise HTTPException(503, "engine not loaded")
    ql = q.lower().strip()
    out = [dict(misconception_id=int(m), misconception_name=n)
           for m, n in zip(engine.ids, engine.names) if not ql or ql in n.lower()]
    return dict(total=len(out), results=out[:limit])


if WEB.exists():
    app.mount("/static", StaticFiles(directory=str(WEB)), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(WEB / "index.html"))

    @app.get("/{page}")
    def page(page: str):
        if page in {"student", "tutor", "try", "benchmark", "how"}:
            return FileResponse(str(WEB / "index.html"))
        if page in {"submission", "project"}:
            sub_f = ROOT.parent / "submission" / "whywrong.html"
            if sub_f.exists():
                return FileResponse(str(sub_f))
        if page == "slides":
            slides_f = ROOT.parent / "submission" / "slides.html"
            if slides_f.exists():
                return FileResponse(str(slides_f))
        f = WEB / f"{page}.html"
        if f.exists():
            return FileResponse(str(f))
        raise HTTPException(404, "not found")

