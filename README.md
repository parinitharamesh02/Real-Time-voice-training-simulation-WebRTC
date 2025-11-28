Bank Voice Training Simulation
Convergent AI - Technical Task (Option A)
Candidate - Parinitha 

A real-time, voice-based simulation environment where a learner practices handling bank customer support scenarios with an LLM-driven “customer.” The emphasis is on AI quality, stateful multi-agent orchestration, latency/cost observability, safety, and evaluation logic.
UI is intentionally minimal, as specified in the assignment.
•	A detailed architectural write-up is included in the repository as:
Technical_Documentation.pdf (Architecture, design rationale, multi-agent system, reliability, state management, trade-offs).

•	A demonstration video showcasing a live session is provided as well.

Please find the video via this link: https://www.loom.com/share/2ddf6387d111445587806343fc52c495

(Note: In the recording, the browser microphone sometimes filtered out the bot TTS audio as “background noise,” causing it not to be captured in the screencast. For completeness, I have included the audio samples that were recorded automatically of the AI customer reply inside the /audio_out folder.)
________________________________________

Features

1.	Real-Time Voice Interaction
o	Mic input → ASR → LLM simulation → TTS output
o	Turn-based real-time voice with immediate playback
o	Latency logged for ASR / LLM / TTS per turn
o	Cost estimation per turn
o	Three fully implemented scenarios/personas

2.	Multi-Agent LangGraph Simulation
    Each turn activates:
o	Customer Generation Agent
o	Evaluation Agent (6 skills)
o	Coaching Agent (explains scores + next-step hint)
o	Safety Agent (prevents unsafe model behavior)

3.	Skill Evaluation (Per Turn)
    Each learner response is scored on:
o	Greeting & rapport
o	Empathy
o	Probing questions
o	Clarity
o	Resolution progress
o	Compliance
    Includes:
o	Coaching narrative
o	Live hint for next turn
o	Mode switching (normal / support / strict / safe)

4.	End-of-Session Assessment
    Provides:
o	A full session summary
o	Quoted positive example
o	Quoted needs-improvement example
o	Skill trends across turns

5.	Latency & Cost Awareness
o	ASR, LLM, TTS latency per turn
o	Cost estimation (Whisper + GPT + TTS)
o	Displayed directly in the UI

6.	 Optional RAG (Partial Implementation)
The customer agent has access to a small, embedded “bank policy” document indexed with FAISS.  
This is used internally to keep behaviour consistent with basic support guidelines  
(e.g., verifying identity, not asking for PINs, blocking lost cards).
There is no live document upload in the current version; policy context is static by design.

7.	 Safety & Reliability
o	Strict safety rules
o	Deterministic evaluator prompts
o	Turn caps
o	Memory truncation
o	Graceful handling of missing data / unexpected turns
________________________________________

2. Personas & Scenarios
    Three required personas:
1.	Lost Card — Angry customer
2.	Account Locked — Stressed customer
3.	Failed Transfer — Confused customer
    Each includes:
•	Scenario context
•	Emotional baseline
•	Difficulty
•	Target skill set
•	Persona tone & constraints
________________________________________

Browser UI
→ Audio recording
→ Upload to /session/turn-audio
→ ASR transcription
→ LLM simulation (LangGraph)
→ Evaluation + coaching + safety
→ TTS synthesis
→ Audio playback in browser
All state is stored per session in memory.

High-Level Components:
client.html             # Minimal UI (mic input + audio playback + metrics)
api.py                  # FastAPI backend (ASR, LLM pipeline, TTS, session mgmt)
voice_pipeline.py       # ASR/TTS utilities, temp handling, file outputs
agents.py               # LangGraph workflow: customer, evaluator, coach, safety
personas.py             # Persona definitions
scenarios.py            # Scenario metadata + initial state builders
state.py                # SimulationState structure
config.py               # Environment keys & model configuration
main_cli.py             # CLI version of the simulation (text-only)
requirements.txt
.env.example
README.md               # This document
Technical_Documentation.pdf   # Deep architecture explanation
________________________________________

4. Models Used
•	ASR: Whisper (OpenAI)
•	LLM (customer, evaluation, coaching, safety): gpt-4o-mini
•	TTS: OpenAI Speech (MP3 output)
All models chosen for latency, cost efficiency, and deterministic behavior.
________________________________________

5. How to Run Locally

1. Install Dependencies

pip install -r requirements.txt

2. Configure Environment

3. In the existing .env file in project root, add the API key:
OPENAI_API_KEY=your_key_here

4. Start Backend
uvicorn api:app --reload
You might have to open this link - http://127.0.0.1:8000/docs 

5. Open the UI
Open client.html directly in your browser.
(No web server required for the frontend.)
You can now start a session, speak via microphone, and interact with the AI customer.
________________________________________

6. Deliverables Included
•	Full runnable repository
•	Live voice demo video
•	Technical_Documentation.pdf
Includes:
o	Architecture diagrams
o	Multi-agent workflow
o	Reliability & safety considerations
o	Human-centric design philosophy
o	RAG integration
o	Latency/cost analysis
o	Limitations & future improvements
________________________________________
7. Known Limitations
•	Audio is turn-based, not full WebRTC duplex streaming
•	RAG is partial (sufficient for demonstration)
•	No avatar / facial animation (optional extension)
•	No multi-session long-term memory
________________________________________
8. What I Would Do With More Time
•	Add WebRTC full-duplex with barge-in interruption
•	Introduce a talking-head avatar with emotion-aware lip sync
•	Expand RAG to semantic policy lookup with authoritative citations
•	Create more advanced scoring rubrics
•	Add datastore-backed analytics & session replay
•	Support long-term adaptive learning across sessions
________________________________________
Final Note
This project fulfills all required elements of Option A, plus multiple extensions (live hints, assessment, RAG integration, cost/latency tracking). The emphasis remains on AI rigor, state correctness, and simulation reliability, as requested in the task specification.

