# Nerdy AI Hackathon: WhyWrong Submission Pack

**Prompt:** My own idea  
**Project:** WhyWrong: Misconception-Level Diagnosis for Mathematics  

> Note: Metrics in `[[ ]]` are populated from `results/metrics.json` and `serve/artifacts/benchmark.json` via `scripts/12_fill_submission.py`.

---

## Project Overview

**WhyWrong: Misconception-level diagnosis for adaptive mathematics practice.**

WhyWrong identifies the underlying misconception behind a student's incorrect mathematics answer from a 2,587-class taxonomy. Rather than merely flagging an error, it classifies the specific conceptual mistake, returns a Socratic hint sequence targeting that mistake, and clusters live errors for the tutor.

**Pipeline Architecture:** Two-stage retrieval. A fine-tuned bi-encoder retrieves the top 25 candidate misconceptions from the 2,587-class taxonomy, and a distilled cross-encoder reranks them to the top prediction. The cross-encoder was trained as a Qwen2.5-7B LoRA model and distilled into a 22M-parameter MiniLM, quantized to INT8 ONNX for CPU execution. To handle the long tail (58% of classes do not appear in the training split, and 64% of test queries have unseen gold labels), `[[N_SYNTH]]` synthetic training pairs were generated using a local 7B model. This improved Recall@25 on unseen misconceptions from `[[R25_UNSEEN_BEFORE]]` to `[[R25_UNSEEN_AFTER]]`. Pedagogical explanations and hint ladders for all 2,587 misconceptions were precomputed and stored in SQLite, allowing fast runtime retrieval without cloud LLM calls.

**Results:** MAP@25 `[[MAP_FINAL]]` on the held-out test split (baseline floor: `[[MAP_FLOOR]]`), `[[P50]]` ms latency on a single CPU core, and approximately `[[COST_OURS]]` per 1,000 queries compared to `[[COST_FRONTIER]]` for frontier API alternatives (`[[COST_RATIO]]`x lower cost). The failure gallery and benchmark comparison are live at **`[[URL]]`/benchmark**.

**Future Work:**
1. **Targeted Disambiguation:** When multiple misconceptions align with a student response, ask a single diagnostic question rather than guessing.
2. **Adaptive Problem Sequencing:** Automatically select subsequent questions to verify whether a diagnosed misconception has been resolved.
3. **Multi-Domain Taxonomy:** Expand the retrieval architecture to other domains such as science, grammar, and introductory computer science.

*Data note: Trained and evaluated on Eedi's publicly released "Mining Misconceptions in Mathematics" research dataset under its research-use terms. This deployment is a non-commercial demonstration; the architecture can be retrained on custom curricula or proprietary datasets.*

---

## Demo Video Script (Target: 2:30, Cap: 3:00)

**0:00 - 0:20: The Problem**  
*Screen: Student enters 7/12 - 3/12 = 4/24. Standard generic interface displays "Try again".*

> "Most math learning software only checks whether an answer is correct or incorrect. When a student solves 7/12 minus 3/12 and answers 4/24, a basic system just asks them to try again. But the student repeats the mistake because they applied a specific broken rule: subtracting the denominators. WhyWrong is built to diagnose that specific misconception."

**0:20 - 0:50: Student Diagnostic Experience**  
*Action: Demonstrate the Student view with WhyWrong. Show the diagnosis and reveal the 3-step hint ladder.*

> "Here is WhyWrong on the same problem. In milliseconds on CPU, the model identifies misconception #172: subtracting numerators and denominators. Instead of giving away the answer, it provides a 3-step Socratic ladder. Step 1 asks a guiding question, Step 2 provides a conceptual clue, and Step 3 details the rule. Below, it suggests a targeted follow-up problem to test whether the concept has cleared."

**0:50 - 1:25: Tutor Session View**  
*Action: Run the session simulation and highlight the root-cause card.*

> "For 1-on-1 tutoring, a tutor has limited time. Here we simulate Priya's session. Across six errors in different topics like brackets and factorising, the model clusters the mistakes to show that 100% share one root cause: ignoring the negative sign when expanding brackets. The dashboard provides an opening prompt and verification questions ready for the tutor to use immediately."

**1:25 - 2:00: Architecture and Training**  
*Action: Display the Architecture view.*

> "The pipeline uses a fine-tuned BGE retriever to get top 25 candidates, followed by a cross-encoder reranker. We trained a 7B LoRA teacher on A100 GPUs and distilled it into a 22-million parameter MiniLM. For the long tail of misconceptions without training pairs, synthetic question generation boosted unseen recall from `[[R25_UNSEEN_BEFORE]]` to `[[R25_UNSEEN_AFTER]]`."

**2:00 - 2:25: Benchmark and Cost**  
*Action: Show the benchmark table.*

> "Because the math taxonomy is bounded to 2,587 concepts, all pedagogical explanations are stored offline in SQLite. Runtime diagnosis is just a fast matrix operation and local database lookup. That makes it `[[COST_RATIO]]` times cheaper than calling frontier cloud APIs, while running fully offline on basic CPU hardware."

**2:25 - 2:40: Failure Analysis and Closing**  
*Action: Open the Failure Gallery view.*

> "The benchmark page also includes an open failure gallery analyzing where the model misses the gold label, such as ambiguous distractors or missing diagrams. Addressing these with interactive disambiguation questions is our next development step."

---

## Submission Checklist

- [ ] Live demo URL reachable from outside the network
- [x] Benchmark figures measured from the shipped artifact
- [x] Failure gallery categorized with notes per case
- [ ] Repository public, README verified, no credentials committed
- [ ] Video under 3:00 duration
- [ ] Form submitted with repo, video, and demo URL
