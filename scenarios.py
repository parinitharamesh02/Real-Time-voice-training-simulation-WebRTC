# scenarios.py
from typing import Dict, Any

from langchain_core.messages import AIMessage
from state import SimulationState
from personas import PersonaId, persona_description, default_target_skills


# ---- Seed scenarios for 3 bank personas ----

SCENARIOS: Dict[PersonaId, Dict[str, Any]] = {
    "lost_card_angry": {
        "difficulty": "medium",
        "context": (
            "Customer reports their debit card is missing and has noticed "
            "several suspicious transactions in the last 24 hours."
        ),
        "starting_utterance": (
            "Hi, I just noticed a bunch of charges on my card that I did NOT make, "
            "and now I can't even find my card. What is going on? This is ridiculous."
        ),
        "target_skills": [
            "greeting_and_rapport",
            "empathy",
            "probing_questions",
            "resolution_progress",
            "compliance",
        ],
        "scenario_description": (
            "Angry customer with a lost card and suspected fraud. The agent should "
            "stay calm, show empathy, verify identity, and clearly explain next steps "
            "to block the card and investigate the transactions."
        ),
    },
    "account_locked_stressed": {
        "difficulty": "medium",
        "context": (
            "Customer has been locked out of online banking after multiple failed "
            "login attempts. They need to pay an urgent bill today."
        ),
        "starting_utterance": (
            "I keep getting this 'account locked' message when I log in, and I have a "
            "bill due TODAY. I don’t have time for this. Can you please just fix it?"
        ),
        "target_skills": [
            "greeting_and_rapport",
            "clarity",
            "empathy",
            "resolution_progress",
        ],
        "scenario_description": (
            "Stressed customer locked out of their account before an urgent payment. "
            "The agent should de-escalate, explain the lock clearly, and guide the "
            "customer through safe recovery steps while reassuring them about the payment."
        ),
    },
    "failed_transfer_confused": {
        "difficulty": "easy",
        "context": (
            "Customer attempted a bank transfer which failed with a vague error. "
            "They are not very technical and are worried their money is 'stuck'."
        ),
        "starting_utterance": (
            "I tried to send money to my friend and it just said 'transfer failed'. "
            "Did my money disappear? I don’t really understand what happened."
        ),
        "target_skills": [
            "greeting_and_rapport",
            "clarity",
            "empathy",
            "probing_questions",
            "resolution_progress",
        ],
        "scenario_description": (
            "Confused customer whose transfer failed. The agent should avoid jargon, "
            "check basic details, reassure the customer their funds are safe, and give "
            "simple, step-by-step next actions."
        ),
    },
}


def init_state_for_persona(persona: PersonaId) -> SimulationState:
    """
    Build an initial SimulationState for a given persona using our seed scenarios.
    The AI 'customer' will typically start the conversation.
    """
    scenario = SCENARIOS[persona]

    state: SimulationState = {
        "messages": [],
        "persona": persona,
        "turn": 0,
        "evaluation": {},
        "coaching": "",
        "live_hint": "",
        "mode": "normal",
        "difficulty": scenario["difficulty"],
        "context": scenario["context"],
        "start_mode": "customer_starts",
        "starting_utterance": scenario["starting_utterance"],
        "target_skills": scenario.get("target_skills", default_target_skills()),
        "scenario_description": scenario["scenario_description"],
        "next": "customer",
        "latency": {},
        "latency_log": [],
        "turn_log": [],
    }

    # If the customer starts, we inject their first utterance as an AIMessage
    if state["start_mode"] == "customer_starts" and state["starting_utterance"]:
        first_ai = AIMessage(content=state["starting_utterance"])
        state["messages"].append(first_ai)  # type: ignore[arg-type]

    return state
