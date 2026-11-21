"""Sparring partner: raises the pot every time it can, whatever it holds. Never folds."""


def decide(game_state: dict) -> dict:
    if game_state.get("type") == "warmup":
        return {"action": "check"}
    pot = game_state["pot"]
    return {"action": "raise", "amount": game_state["current_bet"] + pot + game_state["amount_owed"]}
