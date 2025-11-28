# state.py
from typing import List, Dict, Optional, TypedDict, Any


class SimulationState(TypedDict, total=False):
    """
    Shared state for one training simulation session.

    This flows through the LangGraph multi-agent pipeline.
    """
    # Conversation
    messages: List[Any]          # list[BaseMessage] at runtime (HumanMessage/AIMessage)
    persona: str                 # "lost_card_angry" | "account_locked_stressed" | "failed_transfer_confused"
    turn: int                    # number of USER turns so far

    # Evaluation / coaching
    evaluation: Dict[str, float] # scores from evaluator agent
    coaching: str                # text coaching from coach agent
    live_hint: str               # short live hint for next turn
    mode: str                    # "support" | "normal" | "advanced"

    # Scenario metadata
    difficulty: str              # "easy" | "medium" | "hard"
    context: str                 # short description of scenario
    start_mode: str              # "customer_starts" | "agent_starts"
    starting_utterance: Optional[str]
    target_skills: List[str]     # e.g. ["empathy", "verification", "probing", "resolution", "compliance"]
    scenario_description: str

    # Control flow
    next: str                    # "customer" | "evaluator" | "coach" | "safety" | "end"

    # Latency metrics for last turn
    latency: Dict[str, float]    # {"asr": ..., "llm": ..., "tts": ...}

    # Session-level logs (optional)
    latency_log: List[Dict[str, float]]
    turn_log: List[Dict[str, Any]]
