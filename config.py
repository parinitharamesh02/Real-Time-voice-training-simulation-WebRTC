# config.py
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

print(f"[config] Loading env from: {ENV_PATH}")
load_dotenv(ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY not set.\n"
        "Create a .env file in the project root with:\n"
        "OPENAI_API_KEY=sk-...\n"
    )

print("[config] OPENAI_API_KEY present?", bool(OPENAI_API_KEY))

# Global OpenAI client (used for ASR + TTS, not chat)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Default LLM models (LangGraph / evaluation pipeline)
DIALOGUE_MODEL = "gpt-4o-mini"   # fast & cheap, for customer turns
EVAL_MODEL = "gpt-4o-mini"       # or "gpt-4.1" if you want higher quality for eval/coaching

# ASR / TTS models (turn-based audio endpoint)
ASR_MODEL = "whisper-1"
TTS_MODEL = "gpt-4o-mini-tts"    # adjust to whatever TTS endpoint name you're using
TTS_VOICE = "alloy"

# Realtime voice model for WebRTC pipeline
REALTIME_MODEL = "gpt-4o-realtime-preview"
