# WhyWrong: Mathematics Misconception Diagnosis System

WhyWrong maps incorrect student answers in mathematics to specific diagnosed misconceptions from a 2,587-class taxonomy. Rather than simply evaluating whether an answer is correct or incorrect, the system identifies the underlying conceptual error, generates structured Socratic hints for self-guided learning, and clusters error sequences for tutor review.

The production runtime is distilled to quantized ONNX models running on a single CPU core with under 150 MB of memory and no external API dependencies.

---

## Core Capabilities

1. **Student Diagnostic Interface**: Matches student errors to diagnosed misconceptions and serves a three-step Socratic hint sequence (guiding question, conceptual clue, and full rule).
2. **Tutor Session Analysis**: Aggregates mistakes across multiple topics and questions to identify shared root-cause misconceptions during a tutoring session.
3. **Interactive Benchmark & Failure Gallery**: Includes latency and accuracy metrics against baseline models, cost models, and analyzed failure cases from the held-out test split.

---

## System Architecture

```
OFFLINE TRAINING PIPELINE
  [A] Synthetic Augmentation   Qwen2.5-7B-Instruct generates diagnostic questions
                               for 1,498 zero-shot misconceptions
  [B] Retriever Fine-Tuning    bge-small-en-v1.5 (33M), contrastive training + hard negatives
  [C] Teacher Reranker         Qwen2.5-7B + LoRA listwise reranker over top-25 candidates
  [D] Distillation             7B teacher distilled into MiniLM-L6 (22M) via KL divergence
  [E] Pedagogical Precompute   Hint progressions and verification probes stored in SQLite
                                       ↓
PRODUCTION SERVING (Single CPU core, no GPU required)
  Bi-encoder INT8 ONNX         → Matmul against 2,587 × 384 embedding matrix → Top 25
         ↓
  Cross-encoder INT8 ONNX      → Reranks top 25 candidates to top 1
         ↓
  SQLite Lookup                → Fetches pre-indexed explanation and hint ladder
         ↓
  FastAPI Service              → Returns JSON response (<400ms p50 latency)
```

### Design Rationale

School mathematics follows a defined taxonomy of approximately 2,587 core misconceptions. Because this taxonomy is bounded, pedagogical explanations and hint sequences can be generated once offline and stored in a indexed database. At runtime, diagnosis requires only bi-encoder retrieval and cross-encoder reranking, eliminating LLM latency and cloud API costs during live student interaction.

The full candidate vector index consists of 2,587 × 384 float32 values (3.97 MB), which is evaluated using NumPy matrix operations in under 1 millisecond.

---

## Data and Experimental Split

The project uses the Eedi *Mining Misconceptions in Mathematics* dataset (NeurIPS 2024 competition release), consisting of real multiple-choice math questions where each distractor reflects a specific misconception:

- 4,370 labeled `(question, incorrect answer) -> misconception` pairs
- 2,587-class misconception taxonomy with a heavy long tail

The dataset split groups entries by question so that distractors for a single question never cross folds (`scripts/00_prepare_data.py`):

| Split | Count | Notes |
|---|---|---|
| Train | 2,647 | Labeled question pairs |
| Validation | 841 | Tuning split |
| Test | 882 | Held-out evaluation split |
| Classes in training set | 1,089 | Observed during supervised training |
| Classes unseen in training set | 1,498 (58%) | Zero-shot target classes |
| Test queries with unseen gold label | 562 of 882 (64%) | Evaluates generalization to unobserved misconceptions |

### Licensing

The dataset is used under its public research-use terms for non-commercial evaluation and demonstration. The architecture is data-agnostic and can be retrained on custom curricula or proprietary assessment data.

---

## Evaluation Results

Evaluation metrics measured on the held-out test split (see `results/metrics.json` for detailed logs):

| Model / Configuration | MAP@25 | Recall@25 | R@25 (Unseen) | Top-1 Accuracy |
|---|---|---|---|---|
| Off-the-shelf bge-small (33M, baseline) | 0.153 | 0.466 | 0.504 | 0.078 |
| Off-the-shelf bge-m3 (568M, baseline) | 0.160 | 0.478 | 0.536 | 0.086 |
| General 7B zero-shot reranking | 0.287 | 0.662 | 0.712 | 0.162 |
| Fine-tuned retriever (real data only) | 0.246 | 0.643 | 0.676 | 0.142 |
| + Synthetic data (unweighted 1:6.6 ratio) | 0.226 | 0.659 | 0.724 | 0.108 |
| + Hard-negative mining | 0.229 | 0.638 | 0.710 | 0.117 |
| + Real data oversampled 6x (selected model) | 0.266 | 0.661 | 0.710 | 0.161 |
| 7B LoRA reranker (teacher model) | 0.361 | 0.662 | 0.712 | 0.237 |
| **WhyWrong Shipped (22M INT8 ONNX, CPU)** | **0.261** | **0.650** | **0.696** | **0.136** |

### Key Findings

- **Model size vs task fine-tuning**: Increasing model parameters 17x (`bge-m3` vs `bge-small`) yielded minimal gain out-of-the-box (0.160 vs 0.153 MAP@25). Fine-tuning the 33M parameter model directly on the diagnostic objective yielded substantially higher performance (0.266 MAP@25).
- **Synthetic data balancing**: Adding 17,490 generated pairs directly alongside 2,647 real pairs improved the unseen misconception slice but degraded the seen slice by 9 MAP points due to distribution skew. Applying 6x oversampling to the real pairs preserved unseen generalization while restoring performance on observed misconceptions.

---

## Quickstart & Local Serving

### Running Locally (Pre-built Artifacts)

```bash
# Windows
.\run_serve.bat

# Linux / macOS
pip install -r requirements-serve.txt
cd serve && uvicorn app:app --port 8077

# Docker
docker build -t whywrong .
docker run -p 8077:7860 whywrong
```

Navigate to `http://localhost:8077` in your browser.

---

## Pipeline Execution

To reproduce training, distillation, and export from scratch:

```bash
# Environment setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Step-by-step pipeline execution
python scripts/00_prepare_data.py
python scripts/01_baseline_embeddings.py --model BAAI/bge-small-en-v1.5
python scripts/02_generate_synthetic.py  --shard 0 --num-shards 3
python scripts/02b_filter_synthetic.py   --inputs data/synth/*.parquet --model models/biencoder_r1
python scripts/03_train_biencoder.py     --out models/biencoder_r2 --synth data/synth/filtered.parquet
python scripts/04_mine_hard_negatives.py --model models/biencoder_r2 --out data/processed/hardnegs.npz
torchrun --nproc_per_node=3 scripts/05_train_reranker.py --cand-train ...
python scripts/06_teacher_scores.py
python scripts/07_distill_student.py
python scripts/08_export_onnx.py         --biencoder models/biencoder_r3 --cross-encoder models/student_ce
python scripts/09_precompute_explanations.py --shard 0 --num-shards 3
python scripts/10_build_bundle.py        --explanations "data/processed/explanations_shard*.jsonl"
python scripts/11_build_benchmark.py

# Start local server
cd serve && uvicorn app:app --port 8077
```

---

## Project Structure

```
scripts/     Pipeline scripts for training, evaluation, distillation, and bundle export
src/         Shared evaluation metrics and standardized text templates
serve/       FastAPI application, ONNX inference engine, and indexed SQLite database
web/         Frontend interface (HTML, CSS, vanilla JavaScript)
results/     Evaluation metrics and ablation logs
submission/  Documentation, presentation deck, and demonstration resources
```
