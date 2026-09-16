# WhyWrong — 2:30 Demo Video Production Guide

This guide gives you the **exact visual workflow and word-for-word voiceover script** for recording your Nerdy AI Hackathon demo video.

**Target Length:** 2:30 (Hard submission cap: 3:00)  
**Recommended Screen Recorders:**
- **Loom** (Free, instant shareable link): [loom.com](https://www.loom.com)
- **Windows Built-in Recorder**: Press `Windows Key + Alt + R` to start/stop recording.
- **OBS Studio**: Clean 1080p recording.

**Before You Start:**
1. Ensure WhyWrong is running locally (`run_serve.bat`).
2. Open two browser tabs:
   - Tab 1: **App**: `http://localhost:8077`
   - Tab 2: **Slides**: `http://localhost:8077/slides`

---

## Word-for-Word Video Script & Actions

### 0:00 – 0:25 · The Problem, Shown Not Told
- **On Screen:** Tab 2 (Slides — Slide 1: *"Every AI tutor says you're wrong. WhyWrong tells you exactly why"*).
- **Voiceover:**
  > "Every AI tutor can tell a student that they're wrong. But detecting wrongness is trivial. The actual signal is the *misconception* underneath it — the specific broken rule the student is applying.
  > 
  > When a student solves 7/12 minus 3/12 and writes 4/24, a generic AI tutor says 'Not quite, try again!'. The student repeats the same mistake and disengages.
  > 
  > A human expert tutor instantly recognizes they subtracted the denominators as well as the numerators. That's why I built **WhyWrong**."

---

### 0:25 – 0:55 · The Student Experience
- **On Screen:** Switch to Tab 1 (`http://localhost:8077`).
- **Action:**
  1. On the Student view, point to the problem `m(2m - 7)`.
  2. Click the wrong answer `3m - 7`.
  3. Diagnosis box appears instantly.
  4. Click **"Reveal first hint"** twice to show the Socratic ladder.
- **Voiceover:**
  > "Here's WhyWrong in action. When the student picks '3m - 7', WhyWrong diagnoses the root misconception in 380 milliseconds on a single CPU core: *'Adds instead of multiplying when expanding bracket'*.
  > 
  > Instead of spoiling the answer, it returns a 3-rung Socratic ladder. Rung 1 is a question, not an answer, nudging them to think about what happens between the letter and the bracket.
  > 
  > And below, the system automatically sequences the next problem specifically chosen to probe whether that exact broken rule has cleared."

---

### 0:55 – 1:35 · The Tutor Experience (The Superpower)
- **On Screen:** Click **"Tutor view"** in the top navigation bar.
- **Action:**
  1. Click the blue **"▶ Simulate a live session"** button.
  2. Watch the 6 questions stream and resolve into the orange **Root cause** panel for Priya.
  3. Point to the suggested opener and probes.
- **Voiceover:**
  > "The real superpower isn't just for the student — it's for the tutor. In a 45-minute live session, a tutor doesn't have time to review ten disjointed mistakes.
  > 
  > Here we simulate Priya's session. She got six questions wrong across four totally different math topics — linear equations, single brackets, factorising, and algebraic fractions.
  > 
  > To a human eye at a glance, it looks like four separate gaps. But WhyWrong clusters the embeddings: 100% of these errors share one root cause: *'Ignores negative sign when expanding bracket'*.
  > 
  > It immediately arms the tutor with a conversational opener: *'Can you walk me through how you expanded these brackets?'* and two check probes to confirm mastery."

---

### 1:35 – 2:05 · How It's Built: The Finite Taxonomy Bet
- **On Screen:** Switch to Tab 2 (Slides — Slide 3: *"The Architecture"*), or click **"How it's built"** in the app.
- **Voiceover:**
  > "How is this possible on a single CPU core?
  > 
  > School mathematics has a finite misconception taxonomy — roughly 2,587 of them. A finite set means the heavy GPU work can be done **once, offline, in the kitchen** — not at every table.
  > 
  > I trained a 7B LoRA teacher on 3×A100 GPUs, generated 17,490 synthetic pairs to cover zero-shot tail concepts, and then **distilled the teacher into a 22-million parameter cross-encoder INT8 model**.
  > 
  > All explanations and hint ladders were precomputed into SQLite. At inference time, serving is just a 3.9 MB vector matmul and a primary-key database lookup. Zero GPUs needed."

---

### 2:05 – 2:30 · Benchmark & 264× Cost Advantage
- **On Screen:** Click **"Benchmark"** tab in the app. Point to the results table and cost comparison.
- **Voiceover:**
  > "On the frozen NeurIPS 2024 Eedi test split, WhyWrong achieves a 0.261 MAP@25, outperforming off-the-shelf embeddings by over 70%.
  > 
  > More importantly, look at cost and latency: frontier LLM APIs cost around $1.15 per 1,000 diagnoses and introduce seconds of streaming lag. WhyWrong costs **$0.002** per 1,000 queries — **264 times cheaper** — running locally in 380 milliseconds."

---

### 2:30 – 2:50 · Honest Failure Analysis & What's Next
- **On Screen:** Scroll down to the **"Where it fails"** gallery on the Benchmark tab.
- **Action:** Point to one of the failure cards (e.g. diagram reliance or near twins).
- **Voiceover:**
  > "A credible engineering benchmark shows where it fails. Our gallery inspects 10 held-out failure cases. For example, questions that require visual diagrams like number line ticks mislead a text-only retriever.
  > 
  > For production, what I'd build next is **active disambiguation**: when a student's answer is equally consistent with two near-twin misconceptions, don't guess — prompt them with one targeted disambiguation question.
  > 
  > WhyWrong turns wrong answers from wasted time into actionable learning signal. Thank you!"

---

## Post-Recording Checklist
- [ ] Video length is under 3 minutes (ideal: 2:30 – 2:45).
- [ ] Audio is clear and easy to understand.
- [ ] Upload to YouTube (Unlisted), Loom, or Google Drive (Ensure link sharing is set to *"Anyone with the link can view"*).
- [ ] Paste video link into your hackathon submission form!
