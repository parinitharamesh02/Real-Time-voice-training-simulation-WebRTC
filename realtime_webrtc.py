# realtime_webrtc.py
from typing import Any
import json
import time

import requests
from fastapi import APIRouter, HTTPException, Form, Response
from pydantic import BaseModel

from personas import PersonaId, persona_description
from scenarios import SCENARIOS
from config import REALTIME_MODEL, TTS_VOICE, OPENAI_API_KEY

router = APIRouter(prefix="/realtime", tags=["realtime"])


class RealtimeConfigRequest(BaseModel):
    persona_id: PersonaId


class RealtimeConfigResponse(BaseModel):
    model: str
    voice: str
    persona_id: PersonaId
    instructions: str
    scenario_context: str
    starting_utterance: str


def build_realtime_instructions(persona_id: PersonaId) -> str:
    """
    Build a concise system prompt for the OpenAI Realtime voice model,
    reusing your existing persona + scenario definitions.
    The HTML later wraps this with extra instructions for ANGRY tone.
    """
    if persona_id not in SCENARIOS:
        raise HTTPException(status_code=400, detail="Unknown persona_id")

    scenario = SCENARIOS[persona_id]
    persona_text = persona_description(persona_id)

    return f"""
You are playing the role of a BANK CUSTOMER in a training simulation.

Persona:
{persona_text}

Scenario context:
{scenario['context']}

Guidelines:
- Stay strictly in character as the customer.
- Respond in short, conversational turns (1–3 sentences).
- Reflect realistic emotions: frustration, confusion, relief, gratitude.
- If the agent is helpful, you gradually calm down.
- If they are unclear or dismissive, you may get more frustrated.
- Do NOT reveal that you are an AI or mention 'simulation' or 'LLM'.
""".strip()


@router.post("/config", response_model=RealtimeConfigResponse)
def get_realtime_config(req: RealtimeConfigRequest) -> Any:
    """
    Used by the browser to:
      - choose a persona,
      - get persona instructions,
      - know which realtime model + voice to use.
    """
    persona_id = req.persona_id
    if persona_id not in SCENARIOS:
        raise HTTPException(status_code=400, detail="Unknown persona_id")

    scenario = SCENARIOS[persona_id]
    instructions = build_realtime_instructions(persona_id)

    return RealtimeConfigResponse(
        model=REALTIME_MODEL,
        voice=TTS_VOICE,
        persona_id=persona_id,
        instructions=instructions,
        scenario_context=scenario["context"],
        starting_utterance=scenario["starting_utterance"],
    )


@router.post("/call-setup")
def realtime_call_setup(
    sdp: str = Form(...),
    session: str = Form(...),
) -> Response:
    """
    OPTIONAL: server-side relay for SDP exchange with OpenAI Realtime.

    The current HTML demo talks directly to OpenAI with the user API key,
    so this endpoint is *not* used by that page. You can use this instead
    if you later want to hide your key.
    """
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured on server")

    # Validate JSON
    try:
        json.loads(session)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid session JSON")

    url = f"https://api.openai.com/v1/realtime/calls?model={REALTIME_MODEL}"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
        "Accept": "application/sdp",
    }

    files = {
        "sdp": (None, sdp),
        "session": (None, session),
    }

    resp = None
    last_exc: Exception | None = None

    for _ in range(3):
        try:
            resp = requests.post(url, headers=headers, files=files, timeout=20)
        except Exception as e:
            last_exc = e
            time.sleep(1.0)
            continue

        if not (500 <= resp.status_code < 600):
            break

        time.sleep(1.0)

    if resp is None:
        raise HTTPException(
            status_code=502,
            detail=f"Error calling OpenAI Realtime API: {last_exc}",
        )

    content_type = resp.headers.get("content-type", "application/sdp")

    if 500 <= resp.status_code < 600:
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=content_type,
        )

    return Response(content=resp.content, status_code=resp.status_code, media_type=content_type)
