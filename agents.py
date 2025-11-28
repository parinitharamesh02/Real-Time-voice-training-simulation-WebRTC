# agents.py
from typing import Dict, Any, List
import json
import re

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from config import OPENAI_API_KEY, DIALOGUE_MODEL, EVAL_MODEL
from state import SimulationState
from personas import persona_description
from scenarios import SCENARIOS

# Optional tiny RAG context to make the customer slightly more grounded.
# (You can expand this later if you want.)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

_docs = [
    Document(
        page_content=(
            "Bank support guidelines:\n"
            "- Always verify the customer's identity before discussing account details.\n"
            "- Show empathy and acknowledge emotions when the customer is stressed or upset.\n"
            "- Explain next steps clearly, in plain language, and avoid jargon.\n"
            "- Never ask for full card numbers, PINs, or passwords over the phone.\n"
            "- For lost cards: block the card, review recent transactions, and reassure the customer.\n"
            "- For locked accounts: explain lockout reason and guide through recovery safely.\n"
            "- For failed transfers: confirm whether funds left the account and give clear guidance."
        ),
        metadata={"source": "bank_policy"},
    ),
]

_embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
_vectorstore = FAISS.from_documents(_docs, embedding=_embeddings)
_retriever = _vectorstore.as_retriever(search_kwargs={"k": 2})


MAX_TURNS = 5  # number of AGENT (learner) turns before auto-ending the sim


# ---------- Orchestrator ----------

def orchestrator_node(state: SimulationState) -> SimulationState:
    """
    Simple control node: if we've reached max user turns, end; else go to customer.
    """
    turn = state.get("turn", 0)
    if turn >= MAX_TURNS:
        return {**state, "next": "end"}
    else:
        return {**state, "next": "customer"}


# ---------- Customer (LLM persona) ----------

def customer_node(state: SimulationState) -> SimulationState:
    """
    Customer agent: responds as a bank customer persona using full conversation state.
    The human (trainee) is the support agent, speaking via HumanMessage.
    """
    messages: List[BaseMessage] = state["messages"]  # type: ignore[assignment]
    persona = state["persona"]
    context_text = state.get("context", "")
    mode = state.get("mode", "normal")

    persona_text = persona_description(persona)

    # Retrieve small bits of "policy" context based on last user message if available
    last_user = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user = msg.content
            break

    query = last_user or "Typical bank support call."
    docs = _retriever.invoke(query)
    kb_context = "\n\n".join([d.page_content for d in docs])

    if mode == "support":
        mode_instructions = (
            "You are in SUPPORT mode: if the agent seems confused or weak, slow down, "
            "be a bit more patient, and give them chances to recover. You can be stressed "
            "but not hostile."
        )
    elif mode == "advanced":
        mode_instructions = (
            "You are in ADVANCED mode: the agent is doing well. You can challenge them with "
            "more detailed questions, push on edge cases, and expect them to handle multiple "
            "concerns while staying calm and clear."
        )
    else:
        mode_instructions = (
            "You are in NORMAL mode: act like a realistic customer, with natural emotions "
            "based on how helpful the agent is."
        )

    system_prompt = f"""
You are playing the role of a BANK CUSTOMER in a training simulation.

Persona:
{persona_text}

Scenario context:
{context_text}

Current adaptive mode: {mode}
{mode_instructions}

Guidelines:
- Stay strictly in character as the customer.
- Respond in short, conversational turns (1–3 sentences).
- Reflect realistic emotions: frustration, confusion, relief, gratitude.
- If the agent is helpful, you gradually calm down.
- If they are unclear or dismissive, you may get more frustrated.
- Do NOT reveal that you are an AI or mention 'simulation' or 'LLM'.

Policy hints (for your internal consistency, not to quote explicitly):
{kb_context}
"""

    llm = ChatOpenAI(model=DIALOGUE_MODEL, temperature=0.8, api_key=OPENAI_API_KEY)
    convo = [SystemMessage(content=system_prompt)] + messages
    customer_reply = llm.invoke(convo)

    new_messages: List[BaseMessage] = messages + [customer_reply]  # type: ignore[assignment]

    return {
        **state,
        "messages": new_messages,
        "turn": state.get("turn", 0),  # turn is incremented on the human side
        "next": "evaluator",
    }


# ---------- Evaluator ----------

def evaluator_node(state: SimulationState) -> SimulationState:
    """
    Evaluate the agent's last answer as a customer support response.
    """
    messages: List[BaseMessage] = state["messages"]  # type: ignore[assignment]
    difficulty = state.get("difficulty", "medium")
    target_skills = state.get("target_skills", [])

    # Last HumanMessage (agent's answer)
    last_agent_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_agent_idx = i
            break

    if last_agent_idx is None:
        # no agent answer yet, skip evaluation
        return {**state, "next": "coach"}

    last_agent_msg = messages[last_agent_idx].content

    # Last AIMessage before that human (customer's utterance)
    last_customer_msg = None
    for j in range(last_agent_idx - 1, -1, -1):
        if isinstance(messages[j], AIMessage):
            last_customer_msg = messages[j].content
            break

    if last_customer_msg is None:
        last_customer_msg = "Customer described a general banking issue."

    eval_prompt = f"""
You are an evaluation agent for a BANK CUSTOMER SUPPORT training simulation.

Scenario difficulty: {difficulty}
Target skills for this call: {target_skills}

Score the AGENT's last response using this rubric (0–10 each):

- greeting_and_rapport: did they greet politely and build a bit of rapport?
- empathy: did they acknowledge the customer's emotions (stress, frustration, confusion)?
- clarity: did they speak clearly and avoid unnecessary jargon?
- probing_questions: did they ask relevant follow-up questions to understand the issue?
- resolution_progress: did they move the issue meaningfully towards a solution?
- compliance: did they respect basic banking safety (e.g., not asking for PIN/password, hinting at verification steps)?

If the agent ignores the customer's emotion or question, scores should be LOW (0–3).
Only give scores above 7 when the answer is clearly strong and well-structured.

Customer said:
\"\"\"{last_customer_msg}\"\"\"


Agent responded:
\"\"\"{last_agent_msg}\"\"\"


Return STRICT JSON:
{{
  "greeting_and_rapport": float,
  "empathy": float,
  "clarity": float,
  "probing_questions": float,
  "resolution_progress": float,
  "compliance": float
}}
No explanation, no extra keys, just JSON.
"""

    llm = ChatOpenAI(model=EVAL_MODEL, temperature=0.0, api_key=OPENAI_API_KEY)
    response = llm.invoke([HumanMessage(content=eval_prompt)])
    text = response.content

    scores: Dict[str, float] = {}
    try:
        scores = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                scores = json.loads(match.group(0))
            except Exception:
                pass

    default_scores = {
        "greeting_and_rapport": 5.0,
        "empathy": 5.0,
        "clarity": 5.0,
        "probing_questions": 5.0,
        "resolution_progress": 5.0,
        "compliance": 5.0,
    }
    default_scores.update({k: float(v) for k, v in scores.items() if k in default_scores})

    return {
        **state,
        "evaluation": default_scores,
        "next": "coach",
    }


# ---------- Coach + live hint ----------

def coach_node(state: SimulationState) -> SimulationState:
    """
    Generate coaching feedback AND a short live hint for the next turn.
    Also updates the adaptive 'mode' (support/normal/advanced).
    """
    scores = state.get("evaluation", {})
    messages: List[BaseMessage] = state["messages"]  # type: ignore[assignment]
    target_skills = state.get("target_skills", [])

    # Last agent response
    last_agent_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_agent_msg = msg.content
            break

    avg_score = 5.0
    if scores:
        avg_score = sum(scores.values()) / len(scores.values())

    if avg_score < 4.0:
        new_mode = "support"
    elif avg_score > 7.5:
        new_mode = "advanced"
    else:
        new_mode = "normal"

    coaching_prompt = f"""
You are a coaching agent in a BANK CUSTOMER SUPPORT training simulation.

Scores:
{json.dumps(scores, indent=2)}

Average score: {avg_score:.2f}
New adaptive mode for the CUSTOMER persona: {new_mode}
Target skills: {target_skills}

Agent's last response:
\"\"\"{last_agent_msg}\"\"\"


Your job:
1. Give ONE short positive reinforcement sentence.
2. Give 2–3 specific, actionable coaching tips, focusing on the weakest scores and target skills.
3. Explain briefly what the agent can do next turn to move towards or stay in a higher mode.
4. Keep it concise (under 6 bullet points total).

Return plain text.
"""

    llm = ChatOpenAI(model=EVAL_MODEL, temperature=0.7, api_key=OPENAI_API_KEY)
    coaching_response = llm.invoke([HumanMessage(content=coaching_prompt)])

    # Live hint: very short, actionable cue for the NEXT turn
    hint_prompt = f"""
You are a concise coach for a bank support agent.

Based on these scores:
{json.dumps(scores)}

And the agent's last answer:
\"\"\"{last_agent_msg}\"\"\"


Give ONE very short, actionable hint (max 15 words) for what the agent should do in the NEXT turn.
Example style: "First acknowledge their frustration, then explain how you'll protect their money."

Return just the hint text, no bullets, no quotes.
"""

    hint_response = llm.invoke([HumanMessage(content=hint_prompt)])

    return {
        **state,
        "coaching": coaching_response.content,
        "live_hint": hint_response.content.strip(),
        "mode": new_mode,
        "next": "safety",
    }


# ---------- Safety ----------

def safety_node(state: SimulationState) -> SimulationState:
    """
    Simple safety/compliance checker on the agent's last message.
    """
    messages: List[BaseMessage] = state["messages"]  # type: ignore[assignment]

    last_agent_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_agent_msg = msg.content
            break

    safety_prompt = f"""
You are a safety/compliance checker for a bank support training simulation.

Review the AGENT's last message for:
- abusive or harassing language
- hateful or discriminatory content
- asking for sensitive info (full card number, PIN, password)

If everything is acceptable, respond with exactly:
SAFE

If there is a problem, respond with:
UNSAFE: <one sentence explanation>
"""

    llm = ChatOpenAI(model=EVAL_MODEL, temperature=0.0, api_key=OPENAI_API_KEY)
    safety_resp = llm.invoke([HumanMessage(content=safety_prompt)])
    safety_flag = str(safety_resp.content).strip()

    coaching = state.get("coaching", "")
    coaching += f"\n\n[Safety check]: {safety_flag}"

    return {
        **state,
        "coaching": coaching,
        "next": "end",
    }


# ---------- Build LangGraph app ----------

def build_simulation_graph():
    workflow = StateGraph(SimulationState)

    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("customer", customer_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("coach", coach_node)
    workflow.add_node("safety", safety_node)

    def route_from_orchestrator(state: SimulationState) -> str:
        return state["next"]

    workflow.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {"customer": "customer", "end": END},
    )

    workflow.add_edge("customer", "evaluator")
    workflow.add_edge("evaluator", "coach")
    workflow.add_edge("coach", "safety")
    # safety is the last step in a single turn:
    workflow.add_edge("safety", END)


    workflow.set_entry_point("orchestrator")

    app = workflow.compile()
    return app


# Single global app instance
simulation_app = build_simulation_graph()

# ---------- Session-level assessment (for end-of-run summary) ----------

def build_session_assessment(state: SimulationState) -> str:
    """
    Build a concise end-of-session assessment using the turn_log + scores.

    This is used as the "simulation assessment" extension:
    - summary of overall performance
    - at least one positive example with [Turn N] + quote
    - at least one needs-improvement example with [Turn N] + quote
    """

    turn_log = state.get("turn_log", [])
    if not turn_log:
        # Fallback: just summarize from messages if no structured log
        from langchain_core.messages import BaseMessage

        messages: list[BaseMessage] = state.get("messages", [])  # type: ignore[assignment]
        transcript_lines = []
        for msg in messages:
            role = "AGENT" if isinstance(msg, HumanMessage) else "CUSTOMER"
            transcript_lines.append(f"{role}: {msg.content}")

        transcript_text = "\n".join(transcript_lines)
        prompt = f"""
You are an assessment agent for a bank support training simulation.

Here is the raw transcript:

{transcript_text}

Write a concise assessment (max 250 words) with:
- a short summary of how the AGENT performed,
- one positive example citing the exact AGENT quote,
- one improvement example citing the exact AGENT quote.

Format:
Summary:
- ...

Positive example:
- [Turn ?] "..."

Needs improvement:
- [Turn ?] "..."
"""
        llm = ChatOpenAI(model=EVAL_MODEL, temperature=0.3, api_key=OPENAI_API_KEY)
        resp = llm.invoke([HumanMessage(content=prompt)])
        return str(resp.content)

    # Structured path using turn_log
    # turn_log entries should look like:
    # { "turn": int, "agent": str, "customer": str, "scores": {...} }
    prompt = f"""
You are an assessment agent for a BANK CUSTOMER SUPPORT training simulation.

You will receive a list of turns. Each turn has:
- turn: integer (agent turn number, starting at 1)
- agent: what the support agent said
- customer: what the customer said next
- scores: rubric scores for this agent turn

Turn log (JSON):
{json.dumps(turn_log, indent=2)}

Your job:
1. Give a brief summary (3–5 bullet points) of how the agent performed overall
   across greeting/rapport, empathy, clarity, probing, resolution, and compliance.
2. Provide at least ONE clearly positive example:
   - cite it like: [Turn N] "<agent quote>"
   - explain in 1–2 sentences why this was good.
3. Provide at least ONE needs-improvement example:
   - cite it like: [Turn N] "<agent quote>"
   - explain in 1–2 sentences what should change.
4. Keep the total under 250 words.
5. Be specific and actionable, as if coaching a real trainee.

Return plain text only in this structure:

Summary:
- ...

Positive example:
- [Turn N] "..."

Needs improvement:
- [Turn N] "..."
"""
    llm = ChatOpenAI(model=EVAL_MODEL, temperature=0.3, api_key=OPENAI_API_KEY)
    resp = llm.invoke([HumanMessage(content=prompt)])
    return str(resp.content)

