# Nerdy AI Hackathon Challenge — submission pack

**Prompt:** My own idea
**Project:** WhyWrong — misconception-level diagnosis for adaptive practice

> Numbers in `[[ ]]` are auto-filled from `results/metrics.json` / `serve/artifacts/benchmark.json`
> by `scripts/12_fill_submission.py`. Do not type them by hand.

---

## What did you build?

**WhyWrong — misconception-level diagnosis for adaptive practice.**

Most AI tutoring detects *that* a student is wrong. WhyWrong identifies *why*: it maps a
student's specific wrong answer to the underlying misconception (from a 2,587-class
taxonomy), returns a Socratic hint ladder targeting that misconception, and surfaces a
live root-cause panel to the tutor mid-session.

**How it's built.** Two-stage retrieval. A fine-tuned bi-encoder narrows 2,587
misconceptions to 25 candidates; a cross-encoder reranks them. The reranker was trained
as a Qwen2.5-7B LoRA on 3×A100 and then **distilled to a 22M-parameter MiniLM**,
INT8-quantised to ONNX. To handle the long tail — 58% of the taxonomy never appears in
our training split, and 64% of test queries have a gold label the model has never seen —
I generated `[[N_SYNTH]]` synthetic training pairs with a locally-hosted 7B, which lifted
Recall@25 on unseen misconceptions from `[[R25_UNSEEN_BEFORE]]` to `[[R25_UNSEEN_AFTER]]`.
Explanations and hint ladders for all 2,587 misconceptions were generated once, offline,
and stored — so production is a database lookup with no LLM in the hot path.

**Result:** MAP@25 `[[MAP_FINAL]]` on the held-out split (off-the-shelf embedding floor:
`[[MAP_FLOOR]]`), `[[P50]]` ms p50 latency on a **single CPU core**, at roughly
`[[COST_OURS]]` per 1,000 diagnoses versus `[[COST_FRONTIER]]` for a frontier-API
approach — `[[COST_RATIO]]`× cheaper. Full comparison, including a gallery of the cases it
gets wrong, is live at **`[[URL]]`/benchmark**.

**What I'd do next.**
1. **Disambiguation.** When two misconceptions are both genuinely consistent with an
   answer, ask one targeted question rather than guess. The failure gallery shows this is
   the single largest error class.
2. **Close the loop on sequencing.** Knowing the misconception is half the problem;
   choosing the next question is the other half.
3. **Beyond maths.** The architecture is subject-agnostic — it needs a labelled
   misconception set, not a maths-specific model.

*Data note: trained and evaluated on Eedi's publicly released "Mining Misconceptions in
Mathematics" research dataset under its research-use terms. The Kaggle competition gates
programmatic download; we used the publicly redistributed mirror and did not circumvent
the gate. This deployment is a non-commercial demonstration; the pipeline is data-agnostic
and would be retrained on licensed or first-party data for production.*

---

## Demo video script (target 2:30, hard cap 3:00)

**0:00–0:20 — The problem, shown not told.**
*Screen: a student types `7/12 − 3/12 = 4/24`. A generic tutor panel says "Not quite, try again!". Hold on it, silent, 4 seconds.*

> "This is what almost every AI tutor does with a wrong answer. It detects wrongness. It
> has no idea what just happened in that kid's head."

**0:20–0:50 — The product.**
*Same input into WhyWrong. Diagnosis appears; reveal the hint ladder one rung at a time.*

> "WhyWrong diagnoses the specific misconception — here, subtracting the denominators as
> well as the numerators — and responds to the cause, not the symptom. The first hint is a
> question, never the answer."

**0:50–1:25 — The tutor view. Slow down here.**
*Run the session simulation. Six wrong answers across four different topics.*

> "But the real user is the tutor. This student just got six questions wrong across four
> different topics — brackets, linear equations, factorising, algebraic fractions. It
> looks like four separate problems. It's one. The panel says so in a glance, mid-session,
> with an opening question to use right now. That's the difference between a tutor spending
> ten minutes finding the problem and ten minutes fixing it."

**1:25–2:00 — How it's built.**
*Show the architecture panel.*

> "I trained this on my own hardware — three A100s. 58% of the misconception taxonomy has
> zero training examples, so I generated `[[N_SYNTH]]` synthetic ones with a local 7B; that
> alone moved recall on unseen misconceptions from `[[R25_UNSEEN_BEFORE]]` to
> `[[R25_UNSEEN_AFTER]]`. Then a fine-tuned retriever, a 7B LoRA reranker, and I distilled
> that 7B teacher down to 22 million parameters, INT8, on a single CPU core."

**2:00–2:25 — The cost slide.**
*Show the benchmark table.*

> "The distilled model gives up `[[DISTILL_GAP]]` points against its teacher. In exchange
> it's `[[COST_RATIO]]`× cheaper. There are only about 2,500 misconceptions in school
> maths — a finite set — so I generated every explanation once, offline, and production is
> a database lookup. You don't need a frontier model in the hot path. You need it once, in
> the kitchen — not at every table."

**2:25–2:40 — One honest failure, then close.**
*Show a failure-gallery case.*

> "Here's one it gets wrong. Two misconceptions are both genuinely consistent with this
> answer, and guessing is the wrong move — the right fix is to ask the student one
> disambiguating question. That's what I'd build next."

---

## Checklist

- [ ] Live demo URL reachable from outside the network
- [x] `/benchmark` numbers regenerated from the shipped artifact (`scripts/11_build_benchmark.py`)
- [x] Failure gallery categorised with a real note per case
- [ ] Repo public, README renders, no secrets committed
- [ ] Video under 3:00
- [ ] Submission form: name, email, "My own idea", video link, repo link, live URL
