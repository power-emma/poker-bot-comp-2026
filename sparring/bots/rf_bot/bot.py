"""Random-forest sparring partner.

A learned bot, built independently of my_bots/tag's hand-written logic. At each decision it lists
its legal actions (fold, check/call, raise half pot, raise pot, all-in) and asks a random forest
regressor "how many big blinds will I net from here to the end of the hand if I take this
action?". Fold is worth exactly 0 by construction; it picks the action with the best prediction.

The forest is trained by sparring/rf/policy_train.py: matches against tag and the sparring
presets, first with a simple equity heuristic plus random exploration, then with the forest's
own policy (a few rounds of policy iteration). The forest lives in data/rf_policy.json.

data/config.json (optional, used only while collecting training data):
    {"eps": 0.3}        chance of a uniformly random legal action
    {"model": false}    ignore the forest and play the heuristic
Without a forest the bot plays the heuristic, so it always works.
"""

import json
import random
from pathlib import Path

import eval7

HERE = Path(__file__).resolve().parent
FULL_DECK = [eval7.Card(r + s) for r in "23456789TJQKA" for s in "shdc"]
RANKS = "23456789TJQKA"
SIMS = 250
STREETS = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
# action ids: 0 fold, 1 check/call, 2 raise half pot, 3 raise pot, 4 all-in
FEATURES = ("street", "equity", "pot_bb", "owed_frac", "eff_bb", "spr", "button", "opp_raises_hand",
            "opp_raises_street", "paired", "suit_max", "opp_aggr", "opp_fold", "opp_vpip_proxy",
            "action", "commit_pot", "commit_eff")


def _load(name, default):
    try:
        return json.loads((HERE / "data" / name).read_text())
    except Exception:
        return default


CONFIG = _load("config.json", {})
MODEL = _load("rf_policy.json", None) if CONFIG.get("model", True) else None
EPS = float(CONFIG.get("eps", 0.0))


def equity(hole, board, sims=SIMS):
    hero = [eval7.Card(c) for c in hole]
    known = [eval7.Card(c) for c in board]
    used = set(hole) | set(board)
    deck = [c for c in FULL_DECK if str(c) not in used]
    k = 5 - len(known)
    score = 0.0
    for _ in range(sims):
        d = random.sample(deck, 2 + k)
        b = known + d[2:]
        a, o = eval7.evaluate(hero + b), eval7.evaluate(d[:2] + b)
        score += 1.0 if a > o else 0.5 if a == o else 0.0
    return score / sims


def opp_stats(state, me):
    """Match-long tendencies of the opponent from match_action_log (last 200 entries)."""
    acts = raises = folds = vol = 0
    for e in state.get("match_action_log", []):
        if e["bot_id"] == me or e["action"] in ("small_blind", "big_blind", "ante", "show"):
            continue
        acts += 1
        if e["action"] in ("raise", "all_in"):
            raises += 1
        elif e["action"] == "fold":
            folds += 1
        if e["action"] in ("call", "raise", "all_in"):
            vol += 1
    return (raises + 1) / (acts + 3), (folds + 1) / (acts + 4), (vol + 1) / (acts + 2)


def base_features(state, eq):
    me = state["your_bot_id"]
    bb = state["blinds"]["bb"]
    pot, owed = state["pot"], state["amount_owed"]
    opp = next(p for p in state["players"] if p["bot_id"] != me)
    eff = min(state["your_stack"] + state["your_bet_this_street"], opp["stack"] + opp["bet_this_street"])
    log = state.get("action_log", [])
    opp_r_hand = sum(1 for e in log if e["bot_id"] != me and e["action"] in ("raise", "all_in"))
    opp_r_street = sum(1 for e in log if e["bot_id"] != me and e["action"] in ("raise", "all_in")
                       and e["street"] == state["street"])
    board = state["community_cards"]
    ranks = [c[0] for c in board]
    suits = [c[1] for c in board]
    aggr, fold, vol = opp_stats(state, me)
    return [STREETS[state["street"]], eq, pot / bb, owed / max(1, pot), eff / bb, eff / max(1, pot),
            int(state["button_seat"] == state["seat_to_act"]), opp_r_hand, opp_r_street,
            int(len(set(ranks)) < len(ranks)), max([suits.count(s) for s in "shdc"] + [0]),
            aggr, fold, vol]


def candidates(state):
    """[(action_id, reply, chips we add now)] for every legal action."""
    owed, pot, stack = state["amount_owed"], state["pot"], state["your_stack"]
    cur, mine = state["current_bet"], state["your_bet_this_street"]
    out = []
    if owed > 0:
        out.append((0, {"action": "fold"}, 0))
        out.append((1, {"action": "call"}, min(owed, stack)))
    else:
        out.append((1, {"action": "check"}, 0))
    if stack > owed:
        seen = set()
        for aid, frac in ((2, 0.5), (3, 1.0)):
            target = max(state["min_raise_to"], int(cur + frac * (pot + owed)))
            if target - mine >= stack or target in seen:
                continue
            seen.add(target)
            out.append((aid, {"action": "raise", "amount": target}, target - mine))
        out.append((4, {"action": "all_in"}, stack))
    return out


def tree_value(t, x):
    l, r, f, th, v = t["l"], t["r"], t["f"], t["t"], t["v"]
    n = 0
    while l[n] != -1:
        n = l[n] if x[f[n]] <= th[n] else r[n]
    return v[n]


def predict(x):
    trees = MODEL["trees"]
    return sum(tree_value(t, x) for t in trees) / len(trees)


def heuristic(state, eq, cands):
    owed, pot = state["amount_owed"], state["pot"]
    by = {a: c for a, *c in cands}
    if owed > 0:
        need = owed / (pot + owed)
        if eq > 0.70 and 3 in by:
            return 3
        if eq > 0.82 and 4 in by:
            return 4
        return 1 if eq > need + 0.05 else 0
    if eq > 0.65 and 3 in by:
        return 3
    if eq > 0.52 and 2 in by and random.random() < 0.6:
        return 2
    return 1


def decide(state):
    if state.get("type") == "warmup":
        equity(["As", "Kd"], [], 50)
        return {"action": "check"}
    try:
        eq = equity(state["your_cards"], state["community_cards"])
        cands = candidates(state)
        base = base_features(state, eq)
        pot = max(1, state["pot"])
        eff = max(1.0, base[FEATURES.index("eff_bb")] * state["blinds"]["bb"])
        vals = {}
        if MODEL is not None:
            for aid, _, add in cands:
                vals[aid] = 0.0 if aid == 0 else predict(base + [aid, add / pot, add / eff])
        if EPS and random.random() < EPS:
            choice = random.choice(cands)[0]
            how = "explore"
        elif vals:
            choice = max(vals, key=lambda a: (vals[a], -a))
            how = "model"
        else:
            choice = heuristic(state, eq, cands)
            how = "heuristic"
        aid, reply, add = next(c for c in cands if c[0] == choice)
        reply = dict(reply)
        reply["debug"] = {"x": [round(v, 4) for v in base + [aid, add / pot, add / eff]], "how": how,
                          "v": {str(k): round(v, 2) for k, v in vals.items()}}
        return reply
    except Exception as e:
        return {"action": "check" if state.get("can_check") else "fold", "debug": {"error": repr(e)}}
