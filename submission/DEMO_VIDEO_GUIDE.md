# WhyWrong: Demo Video Production Guide

This guide provides the presentation workflow and voiceover outline for recording the WhyWrong demo video.

**Target Length:** 2:30 (Cap: 3:00)  
**Recommended Screen Recorders:**
- **Loom**: [loom.com](https://www.loom.com)
- **Windows Game Bar**: Press `Windows Key + Alt + R` to start/stop recording.
- **OBS Studio**: Clean 1080p recording.

**Setup:**
1. Start the WhyWrong server (`run_serve.bat`).
2. Open two browser tabs:
   - Tab 1: **App**: `http://localhost:8077`
   - Tab 2: **Slides**: `http://localhost:8077/slides`

---

## Video Script & Actions

### 0:00 - 0:25: The Problem
- **On Screen:** Tab 2 (Slides: Slide 1, "WhyWrong: Mathematics Misconception Diagnosis").
- **Voiceover:**
  > "Most math learning software only flags whether an answer is correct or incorrect. But an incorrect answer is just a symptom; the real signal is the misconception behind it.
  > 
  > For example, when a student solves 7/12 minus 3/12 and writes 4/24, standard software says 'Not quite, try again'. The student repeats the mistake because they applied a specific procedural error: subtracting denominators as well as numerators.
  > 
  > An experienced tutor recognizes that pattern immediately. WhyWrong automates this diagnostic step."

---

### 0:25 - 0:55: Student Experience
- **On Screen:** Switch to Tab 1 (`http://localhost:8077`).
- **Action:**
  1. On the Student view, select the problem `m(2m - 7)`.
  2. Click the incorrect answer `3m - 7`.
  3. The diagnosis box appears.
  4. Click **"Reveal first hint"** twice to demonstrate the ladder.
- **Voiceover:**
  > "Here is the student experience. When a student chooses '3m - 7', WhyWrong identifies the underlying misconception locally on CPU: 'Adds instead of multiplying when expanding bracket'.
  > 
  > Rather than spoiling the solution, the interface provides a 3-step Socratic ladder. Step 1 asks a targeted question about the operation between the terms. Step 2 provides a conceptual clue, and Step 3 details the rule.
  > 
  > Below, the system surfaces a follow-up problem chosen to verify if that specific error has been cleared."

---

### 0:55 - 1:35: Tutor Dashboard
- **On Screen:** Click **"Tutor view"** in the navigation bar.
- **Action:**
  1. Click **"Simulate session"**.
  2. Watch the questions stream and resolve into the root cause summary.
  3. Highlight the suggested opener and check probes.
- **Voiceover:**
  > "In live 1-on-1 tutoring, session time is limited. Here we simulate Priya's session. She made errors across six problems spanning multiple topics like linear equations, single brackets, and factorising.
  > 
  > Rather than treating these as separate errors, WhyWrong clusters the embeddings: all of these mistakes stem from one root cause: 'Ignores negative sign when expanding bracket'.
  > 
  > The panel immediately gives the tutor a conversational opener and two targeted probe questions to confirm understanding during the lesson."

---

### 1:35 - 2:05: Architecture & Distillation
- **On Screen:** Switch to Tab 2 (Slides: Slide 3) or click **"How it's built"** in the app.
- **Voiceover:**
  > "To keep latency and costs minimal, the runtime runs entirely on a single CPU core.
  > 
  > The taxonomy covers 2,587 standard math misconceptions. Because this taxonomy is bounded, pedagogical hint sequences are generated once offline and indexed into SQLite.
  > 
  > We fine-tuned a 33M parameter BGE retriever, trained a 7B LoRA teacher reranker, and distilled that teacher down to a 22M parameter MiniLM cross-encoder quantized to INT8 ONNX. At runtime, inference is simply a 3.9 MB matrix multiplication and a database lookup with zero GPU requirements."

---

### 2:05 - 2:30: Benchmark & Cost Analysis
- **On Screen:** Click the **"Benchmark"** tab in the app.
- **Voiceover:**
  > "On the held-out NeurIPS 2024 Eedi test split, WhyWrong achieves 0.261 MAP@25, substantially outperforming generic baseline embeddings.
  > 
  > In terms of cost and speed: calling external frontier LLM APIs averages $1.15 per 1,000 requests and adds noticeable network latency. WhyWrong operates at roughly $0.002 per 1,000 queries (over 260 times cheaper) and runs locally in sub-500ms."

---

### 2:30 - 2:50: Failure Analysis & Future Steps
- **On Screen:** Scroll down to the failure gallery on the Benchmark tab.
- **Action:** Highlight a failure example (e.g. diagram reliance or overlapping concepts).
- **Voiceover:**
  > "The benchmark page also includes an open failure gallery analyzing cases where the model misses the gold label, such as questions requiring visual diagrams.
  > 
  > As a next step, we are adding interactive disambiguation: when a student choice aligns with two related misconceptions, the system prompts with one targeted diagnostic question instead of guessing.
  > 
  > Thank you!"

---

## Recording Checklist
- [ ] Video duration is under 3 minutes (recommended: 2:30 - 2:45).
- [ ] Audio is clear and audible.
- [ ] Uploaded to YouTube (Unlisted), Loom, or Google Drive with public view access.
- [ ] Video URL added to the submission form.
