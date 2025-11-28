# voice_pipeline.py
import os
import time
import tempfile
from pathlib import Path
from typing import Tuple

from config import openai_client, ASR_MODEL, TTS_MODEL, TTS_VOICE

BASE_DIR = Path(__file__).resolve().parent
AUDIO_OUT_DIR = BASE_DIR / "audio_out"
AUDIO_OUT_DIR.mkdir(exist_ok=True)


def transcribe_audio_file(path: str) -> Tuple[str, float]:
    """
    Run ASR (Whisper) on an audio file.
    Returns (transcript, latency_seconds).
    """
    start = time.time()
    with open(path, "rb") as f:
        result = openai_client.audio.transcriptions.create(
            model=ASR_MODEL,
            file=f,
        )
    elapsed = time.time() - start
    text = result.text.strip()
    return text, elapsed


def synthesize_speech_to_file(text: str, session_id: str, turn: int) -> Tuple[str, float]:
    """
    Run TTS on `text` and write to an mp3 file in audio_out/.
    Returns (file_path, latency_seconds).
    """
    filename = f"{session_id}_turn{turn}.mp3"
    out_path = str(AUDIO_OUT_DIR / filename)

    start = time.time()
    # NOTE: Depending on your OpenAI SDK version / model, you may want to
    # switch TTS_MODEL to "tts-1" or "tts-1-hd". This code assumes a modern
    # SDK where .audio.speech.create returns a binary-like object.
    response = openai_client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
    )
    # Some SDK versions support .stream_to_file; others expose .read().
    try:
        # Newer streaming interface
        with open(out_path, "wb") as f:
            f.write(response.read())
    except AttributeError:
        # Fallback: older style with stream_to_file
        try:
            response.stream_to_file(out_path)
        except Exception:
            # Last-resort: assume response is raw bytes
            with open(out_path, "wb") as f:
                f.write(response)  # type: ignore[arg-type]

    elapsed = time.time() - start
    return out_path, elapsed


def save_upload_to_temp(upload_file) -> str:
    """
    Utility for FastAPI UploadFile -> temp file path.
    """
    suffix = Path(upload_file.filename or "").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = upload_file.file.read()
        tmp.write(content)
        return tmp.name
