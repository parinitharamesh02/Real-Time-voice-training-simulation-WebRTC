# main_cli.py
from langchain_core.messages import HumanMessage, AIMessage

from scenarios import init_state_for_persona
from personas import PersonaId
from agents import simulation_app, MAX_TURNS, build_session_assessment


def choose_persona() -> PersonaId:
    print("Choose a training scenario:")
    print("1) Lost card (angry customer)")
    print("2) Account locked (stressed customer)")
    print("3) Failed transfer (confused customer)")

    while True:
        choice = input("Enter 1, 2, or 3: ").strip()
        if choice == "1":
            return "lost_card_angry"
        if choice == "2":
            return "account_locked_stressed"
        if choice == "3":
            return "failed_transfer_confused"
        print("Invalid choice, try again.")


def run_cli_simulation():
    persona = choose_persona()
    state = init_state_for_persona(persona)

    print("\n=== Bank Support Training Simulation ===")
    print("Persona:            ", state["persona"])
    print("Difficulty:         ", state["difficulty"])
    print("Context:            ", state["context"])
    print("Target skills:      ", state["target_skills"])
    print("Scenario:           ", state["scenario_description"])
    print("========================================\n")

    # Show starting customer utterance if any
    if state.get("starting_utterance"):
        print("[Customer]:", state["starting_utterance"])
        print()

    print(" You are the bank support AGENT. Type your responses below.\n")

    while True:
        # stop after MAX_TURNS agent replies
        if state.get("turn", 0) >= MAX_TURNS:
            print("Simulation ended (max turns reached).")
            break

        user_text = input("Your reply (or 'quit'): ").strip()
        if user_text.lower() in {"quit", "exit"}:
            print("Simulation ended by user.")
            break

        # Add agent's response
        state["messages"].append(HumanMessage(content=user_text))  # type: ignore[arg-type]
        state["turn"] = state.get("turn", 0) + 1

        # Run one full multi-agent cycle (customer → evaluator → coach → safety)
        state = simulation_app.invoke(state)

        # Get last customer reply (AIMessage)
        last_customer = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                last_customer = msg.content
                break

        print("\n----------------------")
        if last_customer:
            print("[Customer]:", last_customer)

        print("\n[Scores]:")
        for k, v in state.get("evaluation", {}).items():
            print(f"  {k}: {v:.1f}")

        print("\n[Coaching]:")
        print(state.get("coaching", "No coaching yet."))

        print("\n[Live hint for next turn]:", state.get("live_hint", ""))

        print("\n[Adaptive mode]:", state.get("mode", "normal"))
        print("----------------------\n")

     # ---- Log this turn for later session assessment ----
        turn_log = state.get("turn_log") or []
        turn_log.append(
            {
                "turn": state.get("turn", 0),
                "agent": user_text,
                "customer": last_customer or "",
                "scores": state.get("evaluation", {}),
                "mode": state.get("mode", "normal"),
            }
        )
        state["turn_log"] = turn_log  # type: ignore[assignment]



        # Add agent's response
        state["messages"].append(HumanMessage(content=user_text))  # type: ignore[arg-type]
        state["turn"] = state.get("turn", 0) + 1

        # Run one full multi-agent cycle
        state = simulation_app.invoke(state)

        # Get last customer reply (AIMessage)
        last_customer = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                last_customer = msg.content
                break

        print("\n----------------------")
        if last_customer:
            print("[Customer]:", last_customer)

        print("\n[Scores]:")
        for k, v in state.get("evaluation", {}).items():
            print(f"  {k}: {v:.1f}")

        print("\n[Coaching]:")
        print(state.get("coaching", "No coaching yet."))

        print("\n[Live hint for next turn]:", state.get("live_hint", ""))

        print("\n[Adaptive mode]:", state.get("mode", "normal"))
        print("----------------------\n")

        print("Simulation ended (max turns reached or user quit).")

    # ---- End-of-session assessment ----
    print("\n=== Session Assessment ===")
    try:
        assessment = build_session_assessment(state)
        print(assessment)
    except Exception as e:
        print(f"(Assessment generation failed: {e})")

    print("\n CLI simulation complete.")



if __name__ == "__main__":
    run_cli_simulation()
