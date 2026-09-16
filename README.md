# WhyWrong

**Every AI tutor can tell a student they're wrong. WhyWrong tells them — and their tutor — exactly *why*.**

A wrong answer is a symptom. The useful signal is the *misconception* underneath it: the
specific broken rule the student is applying, which will keep producing wrong answers until
someone names it. WhyWrong maps a student's wrong answer to one of **2,587 diagnosed
misconceptions**, returns a Socratic hint ladder targeting that misconception, and surfaces a
live root-cause panel to the tutor mid-session.

It runs on **one CPU core**.

> Built for the Nerdy AI Hackathon Challenge · Open track ("bring your own idea")

---

## The problem, in one line

A student answers `7/12 − 3/12 = 4/24`.

| | Response |
|---|---|
| **Generic AI tutor** | "That's not quite right, try again!" → student retries, makes the same error, disengages |
| **A good human tutor** | instantly recognises the student is subtracting denominators as well as numerators, and asks *one* question that breaks it |
| **WhyWrong** | `#172 — When subtracting fractions, subtracts the numerators and denominators` · confidence 0.21 · hint ladder · tutor opener · **55 ms, one CPU core** |

Detecting wrongness is easy. Diagnosing *which* wrongness is the thing that decides whether
the next ten minutes of practice help or waste the kid's time.

---

## What's here

Three surfaces, one engine:

1. **Student view** — practice that responds to the cause. On a wrong answer you get a named
   diagnosis and a 3-rung Socratic ladder whose first rung is a *question*, not an answer.
2. **Tutor view** — the session-intelligence surface. Collapses a stream of individual errors
   into one root cause, with a suggested opener and probes to check it's cleared.
3. **`/benchmark`** — measured numbers against baselines, a cost model, live latency, and a
   gallery of cases the model gets **wrong** with our read on why.

---

## Architecture

```
OFFLINE  (our A100s — runs once, never in production)
  [A] Synthetic augmentation      Qwen2.5-7B-Instruct  → diagnostic questions for the
                                                          1,498 misconceptions with zero
                                                          real training examples
  [B] Retriever fine-tune         bge-small-en-v1.5 (33M), contrastive + hard-negative mining
  [C] Teacher reranker            Qwen2.5-7B + LoRA, listwise over top-25
  [D] Distillation                7B teacher → MiniLM-L6 (22M) via KL on teacher scores
  [E] Explanation precompute      hint ladder + probes for all 2,587 misconceptions
                          ↓ ship artifacts ↓
PRODUCTION  (one small CPU box — no GPU, ever)
  bi-encoder INT8 ONNX ──→ 2,587×384 matrix, np.matmul ──→ top-25
        ↓
  distilled reranker INT8 ONNX ──→ top-1
        ↓
  SQLite primary-key lookup ──→ precomputed explanation + hint ladder
        ↓
  FastAPI JSON
```

### The bet

School mathematics has a **finite** misconception taxonomy — about 2,587 of them. A finite set
means the expensive part, writing a good pedagogical explanation, can be done **once, offline,
on our own GPUs** and stored. Production is then a primary-key lookup.

*You don't need a frontier model in the hot path to do this well. You need it once, in the
kitchen — not at every table.*

The vector index is 2,587 × 384 floats = **3.97 MB**. That's a `numpy` matmul, not a vector database.

---

## Data

**Eedi — *Mining Misconceptions in Mathematics*** (NeurIPS 2024 competition dataset).
Real multiple-choice maths questions where each distractor was deliberately written to embody
a specific misconception.

- 4,370 labelled `(question, wrong answer) → misconception` pairs
- 2,587-class misconception taxonomy, long-tailed

**The frozen split** (`scripts/00_prepare_data.py`, grouped by question so distractors of one
question never straddle folds):

| | count |
|---|---|
| train | 2,647 |
| val | 841 |
| test | 882 |
| misconceptions appearing in train | 1,089 |
| **misconceptions never seen in train** | **1,498 (58%)** |
| **test queries whose gold label is unseen** | **562 of 882 (64%)** |

That last row is the whole difficulty. Nearly two-thirds of the test set asks the model to
retrieve a misconception it has never seen a single example of. Pure supervised learning
plateaus there — which is what the synthetic augmentation stage exists to fix.

### Licensing — stated, not hand-waved

The Kaggle competition data carries research-use terms, and the competition's rules gate
programmatic download (we hit a `403` and did not work around it). We trained and evaluated on
the publicly redistributed HuggingFace mirror of the same release under those research-use
terms, and this deployment is a **non-commercial demonstration**. For production the same
pipeline would be retrained on licensed or first-party data — the architecture is data-agnostic
and nothing in it depends on this particular taxonomy.

---

## Results

See `/benchmark` on the live demo for the full measured table, or `results/metrics.json`
for raw numbers.

<!-- RESULTS:BEGIN (generated by scripts/12_fill_submission.py -- do not edit by hand) -->
| System | MAP@25 | Recall@25 | R@25 unseen | Top-1 |
|---|---|---|---|---|
| Off-the-shelf `bge-small` (33M, no training) | 0.153 | 0.466 | 0.504 | 0.078 |
| Off-the-shelf `bge-m3` (568M, no training) | 0.160 | 0.478 | 0.536 | 0.086 |
| General 7B, zero-shot rerank (no task training) | 0.287 | 0.662 | 0.712 | 0.162 |
| Fine-tuned retriever, real data only | 0.246 | 0.643 | 0.676 | 0.142 |
| + synthetic, naive 1:6.6 mix | 0.226 | 0.659 | 0.724 | 0.108 |
| + hard-negative mining | 0.229 | 0.638 | 0.710 | 0.117 |
| + real data oversampled 6x **(selected on val)** | 0.266 | 0.661 | 0.710 | 0.161 |
| 7B LoRA reranker (teacher, never deployed) | 0.361 | 0.662 | 0.712 | 0.237 |
| **WhyWrong shipped** (22M student, INT8 ONNX, CPU) | 0.261 | 0.650 | 0.696 | 0.136 |
<!-- RESULTS:END -->

Two things in that table are worth pausing on.

**Scaling the embedding model does nothing.** `bge-m3` is 17x larger than `bge-small` and
buys almost nothing off the shelf. The problem is not capacity; it is that nobody taught the
model this task. The same 33M model, fine-tuned, is worth far more than 17x the parameters.

**Naive synthetic mixing made things worse before it made them better.** Dropping 17,490
generated pairs on top of 2,647 real ones (a 6.6:1 ratio) lifted the unseen slice exactly as
intended -- and cost 9 MAP points on the *seen* slice, because the generated distribution
dominated the contrastive objective. Oversampling the real pairs 6x recovers the seen slice
and keeps the long-tail gain. That ablation is in the table rather than quietly dropped,
because the intermediate result is the interesting part.

## Reproducing

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
export HF_HOME=/path/to/cache

python scripts/00_prepare_data.py                    # frozen split + manifest
python scripts/01_baseline_embeddings.py --model BAAI/bge-small-en-v1.5   # the floor
python scripts/02_generate_synthetic.py  --shard 0 --num-shards 3         # per GPU
python scripts/02b_filter_synthetic.py   --inputs data/synth/*.parquet --model models/biencoder_r1
python scripts/03_train_biencoder.py     --out models/biencoder_r2 --synth data/synth/filtered.parquet
python scripts/04_mine_hard_negatives.py --model models/biencoder_r2 --out data/processed/hardnegs.npz
torchrun --nproc_per_node=3 scripts/05_train_reranker.py --cand-train ...  # 7B LoRA teacher
python scripts/06_teacher_scores.py      ...          # cache teacher soft scores
python scripts/07_distill_student.py     ...          # 7B → 22M
python scripts/09_precompute_explanations.py --shard 0 --num-shards 3
python scripts/08_export_onnx.py         --biencoder models/biencoder_r3 --cross-encoder models/student_ce
python scripts/10_build_bundle.py        --explanations 'data/processed/explanations_shard*.jsonl'
python scripts/11_build_benchmark.py                  # evaluates the SHIPPED artifact

cd serve && uvicorn app:app --port 8077                # 1 CPU core is enough
```

### Running the demo locally (production bundle)

To run the shipped model locally without retraining:

```bash
# Windows
.\run_serve.bat

# Linux / macOS
pip install -r requirements-serve.txt
cd serve && uvicorn app:app --port 8077

# Docker (e.g. for Hugging Face Spaces / cloud deployment)
docker build -t whywrong .
docker run -p 8077:7860 whywrong
```
Open [http://localhost:8077](http://localhost:8077) in your browser.


---

## Repo layout

```
scripts/    numbered pipeline, each step standalone and re-runnable
src/        eval harness + the single source of truth for text templates
serve/      FastAPI app, CPU inference engine, and the shipped artifact bundle
web/        the demo site (no build step, no npm)
results/    every measured metric, appended by name
```

`src/text_templates.py` exists because a drift between how training and serving build the
query string is the kind of bug that silently halves your accuracy and never raises. Every
stage imports from that one file.
