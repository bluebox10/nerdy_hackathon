"""WhyWrong inference engine on CPU with no GPU or external network calls.

Pipeline per diagnosis:
  1. bi-encoder (33M, INT8 ONNX) embeds the query                    ~15 ms
  2. numpy matmul against a 2,587 x 384 float32 matrix (3.8 MB)      ~0.1 ms
  3. cross-encoder (22M, INT8 ONNX) reranks the top-25               ~40 ms
  4. SQLite primary-key lookup for the precomputed hint ladder       ~0.1 ms

Step 4 is the whole architectural argument: the expensive pedagogical text was
generated once, offline, for a finite taxonomy, so serving is a lookup.
"""
from __future__ import annotations
import json, sqlite3, threading, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

ART = Path(__file__).resolve().parent / "artifacts"


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


@dataclass
class Timing:
    embed_ms: float = 0.0
    search_ms: float = 0.0
    rerank_ms: float = 0.0
    lookup_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return round(self.embed_ms + self.search_ms + self.rerank_ms + self.lookup_ms, 2)

    def as_dict(self):
        return dict(embed_ms=round(self.embed_ms, 2), search_ms=round(self.search_ms, 3),
                    rerank_ms=round(self.rerank_ms, 2), lookup_ms=round(self.lookup_ms, 3),
                    total_ms=self.total_ms)


class WhyWrongEngine:
    def __init__(self, artifacts: Path = ART, max_len: int = 256, rerank_k: int = 25,
                 threads: int = 1):
        """rerank_k and threads are the two knobs on the latency/accuracy curve.

        The cross-encoder dominates serving cost: it is a full transformer forward per
        candidate, so cost is linear in rerank_k. Measured on one core it is ~28 ms per
        candidate (approximately 700 ms at k=25), which is why both are configurable and why
        scripts/11 sweeps them rather than asserting a single latency number.
        """
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.art = Path(artifacts)
        self.max_len, self.rerank_k, self.threads = max_len, rerank_k, threads
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # ORT's default arena over-reserves badly for our tiny, bursty workload
        opts.enable_cpu_mem_arena = False

        self.bi_tok = self._tok(self.art / "bi_encoder")
        self.bi = ort.InferenceSession(str(self.art / "bi_encoder/model_quantized.onnx"),
                                       opts, providers=["CPUExecutionProvider"])
        self.bi_inputs = {i.name for i in self.bi.get_inputs()}

        ce_dir = self.art / "cross_encoder"
        self.ce = None
        if (ce_dir / "model_quantized.onnx").exists():
            self.ce_tok = self._tok(ce_dir)
            self.ce = ort.InferenceSession(str(ce_dir / "model_quantized.onnx"),
                                           opts, providers=["CPUExecutionProvider"])
            self.ce_inputs = {i.name for i in self.ce.get_inputs()}

        self.E = np.load(self.art / "misc_emb.npy").astype(np.float32)      # (2587, 384)
        rows = json.loads((self.art / "misconceptions.json").read_text())
        self.ids = np.array([r["misconception_id"] for r in rows], dtype=np.int64)
        self.names = [r["misconception_name"] for r in rows]
        self.pos = {int(m): i for i, m in enumerate(self.ids)}

        self.db = sqlite3.connect(str(self.art / "whywrong.sqlite"),
                                  check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self.latencies: list[float] = []

    def _tok(self, d: Path):
        """Load the tokenizer through `tokenizers`, not `transformers`.

        `transformers.AutoTokenizer` imports torch when it is installed, which added
        ~500 MB of RSS to a process that never runs a torch op. The serving box installs
        requirements-serve.txt (no torch, no CUDA, no sentence-transformers) and the
        tokenizer.json in the export is all that is needed.
        """
        from tokenizers import Tokenizer
        cfg = json.loads((d / "tokenizer_config.json").read_text()) if (d / "tokenizer_config.json").exists() else {}
        t = Tokenizer.from_file(str(d / "tokenizer.json"))
        pad = cfg.get("pad_token") or "[PAD]"
        pid = t.token_to_id(pad)
        t.enable_padding(pad_id=pid if pid is not None else 0, pad_token=pad)
        t.enable_truncation(max_length=self.max_len)
        return t

    @staticmethod
    def _feed(encs, wanted):
        arr = lambda attr: np.array([getattr(e, attr) for e in encs], dtype=np.int64)
        out = {"input_ids": arr("ids"), "attention_mask": arr("attention_mask")}
        if "token_type_ids" in wanted:
            out["token_type_ids"] = arr("type_ids")
        return {k: v for k, v in out.items() if k in wanted}

    # ---------- stage 1 ----------
    def embed(self, texts: list[str]) -> np.ndarray:
        encs = self.bi_tok.encode_batch(texts)
        hidden = self.bi.run(None, self._feed(encs, self.bi_inputs))[0]
        vec = hidden[:, 0, :]                                   # bge pools with CLS
        return vec / np.clip(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9, None)

    # ---------- stage 2 ----------
    def rerank(self, query: str, cand_idx: np.ndarray) -> np.ndarray:
        if self.ce is None:
            return None
        a = [query] * len(cand_idx)
        b = [f"Misconception: {self.names[i]}" for i in cand_idx]
        encs = self.ce_tok.encode_batch(list(zip(a, b)))
        return self.ce.run(None, self._feed(encs, self.ce_inputs))[0].reshape(-1).astype(np.float32)

    # ---------- explanation ----------
    def explain(self, mid: int) -> dict:
        row = self.db.execute(
            "SELECT * FROM explanations WHERE misconception_id = ?", (int(mid),)).fetchone()
        if row is None:
            return dict(misconception_id=int(mid),
                        misconception_name=self.names[self.pos[int(mid)]],
                        plain="", why="", hints=[], probes=[], opener="", topic="")
        d = dict(row)
        d["hints"] = json.loads(d.get("hints") or "[]")
        d["probes"] = json.loads(d.get("probes") or "[]")
        return d

    def neighbours(self, mid: int, k: int = 4) -> list[dict]:
        """Retrieve misconceptions closest in embedding space."""
        i = self.pos.get(int(mid))
        if i is None:
            return []
        sims = self.E @ self.E[i]
        order = np.argsort(-sims)[1: k + 1]
        return [dict(misconception_id=int(self.ids[j]), misconception_name=self.names[j],
                     similarity=round(float(sims[j]), 4)) for j in order]

    # ---------- the public call ----------
    def diagnose(self, subject: str, construct: str, question: str,
                 correct: str, student_answer: str, top_n: int = 3) -> dict:
        q = (f"Subject: {subject} | Topic: {construct}\n"
             f"Question: {question}\nCorrect answer: {correct}\n"
             f"Student chose: {student_answer}")
        t = Timing()

        t0 = time.perf_counter()
        vec = self.embed(["Represent this sentence for searching relevant passages: " + q])[0]
        t.embed_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        sims = self.E @ vec
        cand = np.argpartition(-sims, self.rerank_k)[: self.rerank_k]
        cand = cand[np.argsort(-sims[cand])]
        t.search_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        ce_scores = self.rerank(q, cand)
        if ce_scores is not None:
            order = np.argsort(-ce_scores)
            ranked, scores, stage = cand[order], ce_scores[order], "bi+cross"
        else:
            ranked, scores, stage = cand, sims[cand], "bi-only"
        t.rerank_ms = (time.perf_counter() - t0) * 1000

        conf = _softmax(scores[: max(top_n, 5)].astype(np.float64))

        t0 = time.perf_counter()
        out = []
        for r, (j, sc) in enumerate(zip(ranked[:top_n], scores[:top_n])):
            e = self.explain(int(self.ids[j]))
            e.update(rank=r + 1, score=round(float(sc), 4),
                     confidence=round(float(conf[r]), 4),
                     retriever_similarity=round(float(sims[j]), 4))
            out.append(e)
        t.lookup_ms = (time.perf_counter() - t0) * 1000

        with self._lock:
            self.latencies.append(t.total_ms)
            if len(self.latencies) > 5000:
                del self.latencies[:1000]

        return dict(query=q, pipeline=stage, candidates=out,
                    also_at_risk=self.neighbours(int(self.ids[ranked[0]])),
                    timing=t.as_dict())

    def cluster_errors(self, mids: list[int], threshold: float = 0.62):
        """Group a session's diagnosed misconceptions into root causes by embedding
        neighbourhood rather than exact id.

        Requiring exact-id agreement across a session is the wrong test: a student with
        one broken rule produces errors that land on several *adjacent* labels ("only
        multiplies the first term in the bracket", "believes the sign changes when
        factorising"). Those are the same root cause to a tutor. Greedy clustering:
        seed on the most frequent label, absorb every label whose embedding cosine to
        the seed exceeds `threshold`, repeat on what is left.

        Threshold calibrated on the trained index: random misconception pairs sit at
        cosine p99 = 0.43, while the genuine family around "only multiplies the first
        term in the expansion of a bracket" spans 0.63-0.93. 0.62 separates them.
        """
        from collections import Counter
        counts = Counter(int(m) for m in mids)
        remaining = dict(counts)
        clusters = []
        while remaining:
            seed = max(remaining, key=lambda k: (remaining[k], k))
            si = self.pos.get(seed)
            members = [seed]
            if si is not None:
                for other in list(remaining):
                    if other == seed:
                        continue
                    oi = self.pos.get(other)
                    if oi is not None and float(self.E[si] @ self.E[oi]) >= threshold:
                        members.append(other)
            n = sum(remaining[m] for m in members)
            for m in members:
                remaining.pop(m, None)
            clusters.append(dict(seed=seed, members=members, n=n))
        clusters.sort(key=lambda c: -c["n"])
        return clusters

    def latency_stats(self) -> dict:
        with self._lock:
            xs = np.array(self.latencies) if self.latencies else np.array([0.0])
        return dict(n=int(len(self.latencies)),
                    p50=round(float(np.percentile(xs, 50)), 2),
                    p95=round(float(np.percentile(xs, 95)), 2),
                    p99=round(float(np.percentile(xs, 99)), 2),
                    mean=round(float(xs.mean()), 2))
