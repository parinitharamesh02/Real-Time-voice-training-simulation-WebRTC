# Bank Voice Training – Realtime WebRTC + LangGraph Trainer

A **real-time, voice-based training environment** where a learner practices handling **bank customer support** calls against an **LLM-driven “customer”**.

This version extends the original turn-based audio prototype with:

- **OpenAI Realtime (WebRTC) voice calls**
- **Streaming transcripts & customer partials**
- **Per-turn LangGraph evaluation & coaching**
- **Latency / micro-cost tracking**
- **End-of-session assessment**

The focus is on **AI quality**, **stateful multi-agent orchestration**, **observability**, and **training usefulness**, not on flashy UI.

---

## What’s New in This Version

Compared to the original turn-based audio demo, this repo now includes:

### 1. Realtime WebRTC Voice Trainer

- Browser establishes a **WebRTC call** to `gpt-4o-realtime`:
  - Mic → OpenAI Realtime → TTS back to browser audio.
- Customer persona is an **angry / stressed / confused bank customer** (depending on scenario).
- The call supports:
  - **Barge-in** (you can interrupt)
  - **Streaming text partials** from the customer
  - **Full transcripts** for both sides

### 2. LangGraph “Trainer” for Live Coaching

Each **agent turn** is sent to a LangGraph “trainer” pipeline:

- Uses the **existing multi-agent graph**:
  - **Evaluator node** – scores the turn
  - **Coach node** – explains scores & suggests next move
  - **Safety node** – enforces safety / tone constraints
- Trainer runs **side-by-side** with the realtime audio; it does **not** block the call.

For each turn, the trainer produces:

- **Scores** on:
  - `greeting_and_rapport`
  - `empathy`
  - `probing_questions`
  - `resolution_progress`
  - `compliance`
- A **live hint**: “What to do on your very next turn”
- A **coaching narrative**: why you got that score, concrete improvement tips
- **Latency** and **cost estimate** per trainer turn

### 3. End-of-Session Assessment (Realtime Session)

The trainer keeps a **turn log** and a **latency / cost log** in memory.  
At any time, you can click **“✨ View assessment”** and the backend will:

- Summarize the full session
- Highlight **what you did well** (with quotes)
- Highlight **what needs improvement** (with quotes)
- Comment on how your scores evolved over the call

---

## Features Overview

### Realtime Voice Interaction (WebRTC)

- Browser mic → OpenAI Realtime (WebRTC) → streamed TTS back to browser.
- **AI customer starts the call** (e.g. “I’ve lost my card and there are charges I don’t recognise.”).
- You respond as the **bank support agent**.
- You can **interrupt** the customer at any time (barge-in).
- **Live transcript** shows turns from both sides.

### Multi-Agent LangGraph Trainer

For each agent turn, the trainer runs:

- **Evaluator Agent** – scores the turn against the rubric.
- **Coaching Agent** – explains the scores and gives a next-step hint.
- **Safety Agent** – ensures the advice stays within safe / compliant behaviour.

The trainer is **stateful**: it sees the history of the session, not just one isolated line.

### Skill Evaluation (Per Turn)

Each agent turn is scored on:

- Greeting & rapport  
- Empathy  
- Probing questions  
- Resolution progress  
- Compliance / safety

Trainer returns:

- Structured **scores** dict  
- A **coaching explanation**  
- A concise **live hint** (“Ask them to confirm which transactions they recognise…”)  
- A **mode** flag (e.g. `normal` / `support` / `strict` / `safe`)

### End-of-Session Assessment

From the accumulated state, the backend builds a **session assessment**:

- Summary of what happened
- Positive example quote
- Needs-improvement example quote
- Comments across the rubric dimensions

### Latency & Cost Awareness

For both the turn-based and realtime trainer:

- **Latency tracked per trainer turn**:
  - `llm` latency (evaluator + coach + safety)
  - Simple per-turn latency log for a chart
- **Rough cost estimation**:
  - Approximated tokens from text length
  - Tiny “micro-dollars” for LLM / ASR / TTS
  - Useful for showing how training cost scales

### Personas & Scenarios

Three personas are implemented:

1. **Lost Card — Angry customer**
2. **Account Locked — Stressed customer**
3. **Failed Transfer — Confused customer**

Each persona has:

- Scenario context
- Emotional baseline
- Difficulty
- Target skill set
- Persona tone & constraints

The **WebRTC trainer UI** lets you pick any persona from a dropdown; the backend returns:

- Realtime **instructions** for the model
- Scenario **context**
- A **starting utterance** for the AI customer

---

## Architecture

At a high level:

### Frontend

- `client_webrtc.html`
  - WebRTC setup to OpenAI Realtime
  - Mic streaming + remote TTS audio playback
  - Full transcript & per-turn conversation log
  - Right-hand panel:
    - Live **scores**
    - **Hint** & **coaching**
    - **Latency** chart
    - **Cost** per trainer turn
    - **Assessment** viewer

### Backend (FastAPI + LangGraph)

- `api.py`
  - `/health` – healthcheck
  - `/session/start` – original turn-based session start
  - `/session/turn-text` – original text-only mode
  - `/session/turn-audio` – original turn-based audio mode
  - `/session/assessment/{session_id}` – builds session assessment
  - `/realtime/score-turn` – **trainer API** called by the WebRTC UI
- `realtime_webrtc.py`
  - `/realtime/config` – returns realtime model, voice, persona instructions, and scenario context
  - `/realtime/call-setup` – optional server-side relay to OpenAI Realtime (HTML demo currently calls OpenAI directly with the user’s key)
- `agents.py`
  - LangGraph workflow: customer, evaluator, coach, safety
- `personas.py`
  - Persona definitions (angry / stressed / confused customers)
- `scenarios.py`
  - Scenario metadata + initial state builders
- `state.py`
  - `SimulationState` structure
- `voice_pipeline.py`
  - ASR / TTS utilities (for turn-based mode)
- `config.py`
  - Environment keys & model configuration

---

## Models Used

- **Realtime Voice (WebRTC):** `gpt-4o-realtime`  
- **ASR (turn-based mode):** Whisper (`whisper-1`)  
- **LLM (customer, evaluation, coaching, safety):** `gpt-4o-mini`  
- **TTS (turn-based mode):** OpenAI Speech  

Models are chosen to balance **latency**, **cost**, and **behavior quality**.

How to Run Locally

Clone the Repositary

git clone https://github.com/parinitharamesh02/bank-voice-training-simulation.git

cd bank-voice-training-simulation

Create & Activate Virtual Environment

python -m venv venv Set-ExecutionPolicy -Scope CurrentUser RemoteSigned venv\Scripts\activate

On macOS/Linux source venv/bin/activate

Install Dependencies

pip install -r requirements.txt

Configure Environment

In the existing .env file in project root, add the API key: OPENAI_API_KEY=your_key_here

Start Backend

uvicorn api:app --reload

The backend will start at:

http://127.0.0.1:8000 or http://127.0.0.1:8000/docs

Open the UI

Open client.html directly in your browser.

(No web server required for the frontend.)

You can now start a session, speak via microphone, and interact with the AI customer.

Known Limitations

The Realtime demo uses browser-side OpenAI key (for simplicity).
The /realtime/call-setup route is available if we want to hide the key server-side.

Long-term, persistent analytics are in-memory only (no database).

RAG is static policy context, not a full document management system.

No avatar / visual agent; UI is intentionally minimal for clarity.

Enrich the scoring rubric (e.g. de-escalation, time to resolution).

Add exportable session summaries for LMS integration.

---





