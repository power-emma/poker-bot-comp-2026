"""
Baseline bot: estimate equity, then make the pot-odds fold/call decision.

  - Equity is a Monte Carlo estimate of how often our hand wins at showdown
    against one random opponent hand (ties count half), using eval7.
  - Free to continue -> check. Facing a bet -> call when equity beats the
    pot odds (the share of the final pot we have to put in), otherwise fold.

Never raises yet. Equity against a *random* hand overestimates us against an
opponent who bets; adjusting for their range is the next step.
"""

import random

import eval7

SIMULATIONS = 2000          # fixed count (not a time budget) so seeded matches replay exactly

FULL_DECK = [eval7.Card(r + s) for r in "23456789TJQKA" for s in "shdc"]


def equity(hole, board, sims=SIMULATIONS):
    """Share of showdowns won (ties count half) against one random hand."""
    hero = [eval7.Card(c) for c in hole]
    known = [eval7.Card(c) for c in board]
    used = set(hole) | set(board)
    deck = [c for c in FULL_DECK if str(c) not in used]
    to_come = 5 - len(known)

    score = 0.0
    for _ in range(sims):
        drawn = random.sample(deck, 2 + to_come)
        full_board = known + drawn[2:]
        ours = eval7.evaluate(hero + full_board)
        theirs = eval7.evaluate(drawn[:2] + full_board)
        if ours > theirs:
            score += 1.0
        elif ours == theirs:
            score += 0.5
    return score / sims


def hand_label(hole, board):
    """The made hand so far, e.g. 'Two Pair' (preflop: just the hole cards)."""
    if not board:
        return "".join(hole)
    return eval7.handtype(eval7.evaluate([eval7.Card(c) for c in hole + board]))


def pot_odds(state):
    """Equity needed to call: call / (pot after we call).

    If the opponent bet more than we have, we can only call our stack and the
    excess goes back to them, so it isn't part of the pot we're playing for.
    """
    owed = state["amount_owed"]
    call = min(owed, state["your_stack"])
    pot = state["pot"] - (owed - call)
    return call / (pot + call)


def decide(game_state: dict) -> dict:
    if game_state.get("type") == "warmup":
        equity(["As", "Kd"], [], sims=200)       # warm up eval7 before the clock matters
        return {"action": "check"}

    try:
        hole = game_state["your_cards"]
        board = game_state["community_cards"]

        if game_state["can_check"]:
            return {"action": "check", "debug": {"hand": hand_label(hole, board)}}

        eq = equity(hole, board)
        needed = pot_odds(game_state)
        action = "call" if eq >= needed else "fold"
        return {"action": action,
                "debug": {"hand": hand_label(hole, board),
                          "equity": round(eq, 3), "needed": round(needed, 3)}}
    except Exception as e:                       # never crash: check if free, else fold
        return {"action": "check" if game_state.get("can_check") else "fold",
                "debug": {"error": repr(e)}}
