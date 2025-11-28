# realtime_webrtc.py
"""
WebRTC + Realtime API support.

The browser:
  1) Asks this server for a short-lived Realtime ephemeral key via /realtime/token.
  2) Uses that key to open a WebRTC call directly with OpenAI.

This file only handles the token creation. It does NOT proxy the SDP.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import httpx

from config import OPENAI_API_KEY
from personas import PersonaId, persona_description

router = APIRouter()


class RealtimeTokenRequest(BaseModel):
    persona_id: PersonaId


class RealtimeTokenResponse(BaseModel):
    client_secret: str


@router.post("/realtime/token", response_model=RealtimeTokenResponse)
async def create_realtime_token(req: RealtimeTokenRequest) -> RealtimeTokenResponse:
    """
    Create a short-lived client secret (ephemeral key) for a WebRTC Realtime session.

    The client secret is safe for the browser. It does NOT expose your real OPENAI_API_KEY.
    """
    persona_text = persona_description(req.persona_id)
    instructions = (
        "You are a BANK CUSTOMER in a training simulation.\n"
        f"{persona_text}\n\n"
        "Speak in short, natural sentences (1–3 at a time). "
        "React realistically based on how helpful the agent is. "
        "Never mention that you are an AI or a simulation."
    )

    # 👇 This matches the official minimal client_secrets body:
    # { "session": { "type": "realtime", "model": "gpt-realtime", ... } }
    session_config = {
        "type": "realtime",
        "model": "gpt-realtime",
        "instructions": instructions,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/realtime/client_secrets",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                    "OpenAI-Beta": "realtime=v1",
                },
                json={"session": session_config},
            )
    except httpx.RequestError as e:
        # Network/problem talking to OpenAI
        raise HTTPException(
            status_code=502,
            detail=f"Error contacting OpenAI: {repr(e)}",
        ) from e

    # Debug logging so you can see *exactly* what OpenAI returns in your terminal
    print("Realtime client_secrets status:", resp.status_code)
    print("Realtime client_secrets body:", resp.text)

    if not resp.is_success:
        # Forward OpenAI's error text to the Swagger response
        raise HTTPException(status_code=500, detail=f"OpenAI error: {resp.text}")

    # ✅ Correct shape: { "value": "ek_...", "expires_at": ... }
    data = resp.json()
    if "value" not in data:
        # Defensive: if schema is not as expected, surface it clearly
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected client_secrets response: {data}",
        )

    client_secret = data["value"]
    return RealtimeTokenResponse(client_secret=client_secret)
