"""
Tight-aggressive bot with value betting by expected take (see strategy.md).

  - Preflop: every starting hand has a "top %" from a table of equity against a
    random hand. Deep stacks play TAG ranges; at 15 big blinds or less it's
    push or fold.
  - Postflop: Monte Carlo equity against the opponent's estimated range (the
    hands that would have bet, raised and called the way they did this hand,
    given how often this opponent does each). Every option (check, call, fold,
    and each bet size) is scored by EV: how likely this opponent is to fold,
    call or raise at that size, and our equity against the hands that call.
  - Seven-two suited is played as a slightly better hand than its rank, and
    its flush draws like high flush draws (strategy.md 7.1). Seven-two offsuit
    folds unless the opponent folds too much to preflop raises.
"""

import random

import eval7

SIMULATIONS = 2000          # fixed count (not a time budget) so seeded matches replay exactly
RANKS = "AKQJT98765432"
FULL_DECK = [eval7.Card(r + s) for r in RANKS for s in "shdc"]

# Equity of each starting hand against one random hand (60,000 deals each),
# strongest first.
PREFLOP_EQUITY = {
    "AA": 0.852, "KK": 0.827, "QQ": 0.799, "JJ": 0.775, "TT": 0.75, "99": 0.718, "88": 0.689, "AKs": 0.672,
    "77": 0.665, "AQs": 0.662, "AJs": 0.654, "AKo": 0.652, "ATs": 0.649, "AQo": 0.646, "AJo": 0.637, "66": 0.634,
    "KQs": 0.632, "KJs": 0.628, "A9s": 0.627, "ATo": 0.625, "A8s": 0.621, "KTs": 0.618, "KQo": 0.616, "A7s": 0.612,
    "A9o": 0.605, "KJo": 0.604, "QJs": 0.603, "55": 0.602, "A6s": 0.601, "K9s": 0.6, "A8o": 0.598, "A5s": 0.598,
    "KTo": 0.595, "A4s": 0.593, "QTs": 0.593, "A7o": 0.589, "A3s": 0.583, "K8s": 0.582, "QJo": 0.58, "A6o": 0.577,
    "A5o": 0.577, "K9o": 0.577, "Q9s": 0.576, "JTs": 0.576, "K7s": 0.575, "QTo": 0.574, "A2s": 0.573, "44": 0.568,
    "A4o": 0.567, "K6s": 0.565, "A3o": 0.561, "Q8s": 0.559, "K8o": 0.558, "K5s": 0.557, "J9s": 0.556, "JTo": 0.553,
    "K7o": 0.552, "K4s": 0.551, "Q9o": 0.551, "A2o": 0.548, "K3s": 0.545, "Q7s": 0.545, "K6o": 0.544, "T9s": 0.541,
    "Q8o": 0.539, "J8s": 0.539, "Q6s": 0.536, "33": 0.536, "K5o": 0.535, "J9o": 0.535, "K2s": 0.531, "T8s": 0.529,
    "Q5s": 0.527, "J7s": 0.526, "K4o": 0.521, "Q4s": 0.517, "Q7o": 0.516, "T9o": 0.515, "K3o": 0.513, "J8o": 0.512,
    "Q6o": 0.51, "J6s": 0.509, "K2o": 0.508, "Q3s": 0.508, "T7s": 0.508, "98s": 0.505, "Q5o": 0.504, "22": 0.503,
    "Q2s": 0.501, "J5s": 0.5, "J7o": 0.498, "T8o": 0.495, "Q4o": 0.489, "J4s": 0.489, "T6s": 0.489, "97s": 0.489,
    "Q3o": 0.481, "J6o": 0.481, "J3s": 0.481, "98o": 0.481, "87s": 0.48, "T7o": 0.478, "Q2o": 0.474, "T5s": 0.472,
    "96s": 0.472, "J2s": 0.47, "J5o": 0.468, "J4o": 0.466, "T4s": 0.466, "97o": 0.465, "T6o": 0.464, "86s": 0.463,
    "T3s": 0.459, "95s": 0.459, "76s": 0.456, "J3o": 0.454, "87o": 0.45, "T2s": 0.449, "85s": 0.448, "J2o": 0.445,
    "T5o": 0.445, "96o": 0.445, "94s": 0.44, "T4o": 0.433, "93s": 0.433, "75s": 0.433, "65s": 0.432, "86o": 0.43,
    "95o": 0.429, "T3o": 0.425, "76o": 0.425, "92s": 0.424, "84s": 0.424, "74s": 0.418, "85o": 0.415, "64s": 0.415,
    "T2o": 0.414, "54s": 0.413, "83s": 0.408, "94o": 0.406, "82s": 0.404, "75o": 0.403, "93o": 0.401, "73s": 0.399,
    "65o": 0.398, "63s": 0.397, "84o": 0.395, "53s": 0.393, "92o": 0.39, "74o": 0.385, "43s": 0.384, "64o": 0.383,
    "72s": 0.381, "62s": 0.38, "54o": 0.379, "52s": 0.379, "83o": 0.372, "82o": 0.37, "42s": 0.367, "73o": 0.366,
    "53o": 0.363, "63o": 0.361, "32s": 0.358, "43o": 0.35, "72o": 0.346, "62o": 0.345, "52o": 0.342, "42o": 0.331,
    "32o": 0.321,
}

# Share of all 1,326 starting combos at least as strong as each class: AA is
# the top 0.5%, 32o is 100%.
TOP_PCT = {}
_combos = 0
for _cls in PREFLOP_EQUITY:
    _combos += 6 if len(_cls) == 2 else 4 if _cls.endswith("s") else 12
    TOP_PCT[_cls] = _combos / 1326

SHORT_STACK_BB = 15
PREFLOP_RANGE_SIMS = 1000
POSTFLOP_SIMS = 3000             # more than preflop: narrow calling ranges use a slice of them
RERAISE_EQUITY = 0.60            # re-raise preflop with this much equity against their raising range
OUT_OF_POSITION_MARGIN = 0.08    # extra equity a preflop call needs, for the streets still to play

# Opponent stats, kept for the whole match (the process lasts one match).
STATS = {
    "our_preflop_raises": 0, "folds_to_them": 0,   # their folds to our first preflop raise
    "hands": 0, "preflop_raised": 0,               # hands seen, hands they raised preflop
    "bet_chances": 0, "bets": 0,                   # postflop, nothing to call: did they bet?
    "faced_bets": 0, "calls": 0, "raises": 0,      # postflop, facing a bet: call / raise?
    "faced_by_size": [0, 0, 0, 0],                 # ... the same, split by the bet's size
    "continued_by_size": [0, 0, 0, 0],             #     (see SIZE_BUCKETS): didn't fold
}
_seen_hands = set()

# Bet sizes as a fraction of the pot before the bet: up to 0.6, up to 1.1,
# up to 2.5, overbets (usually all-ins). Each bucket's typical size is used
# for its prior.
SIZE_BUCKETS = (0.6, 1.1, 2.5)
BUCKET_TYPICAL = (0.5, 0.85, 1.6, 5.0)

# Value betting (strategy.md section 3).
BET_FRACTIONS = (0.33, 0.5, 0.75, 1.0, 1.5)     # candidate sizes, as a fraction of the pot
STACK_OFF_EQUITY = 0.55          # if they raise our bet, call with this much equity against their raises
RAISE_MULTIPLE = 3               # ...assuming they raise to about this many times our bet
SIZE_PRIOR_WEIGHT = 6            # a size bucket's prior counts as this many observed bets
STREETS_LEFT = {"flop": 3, "turn": 2, "river": 1}
BLUFF_MAX_FRACTION = 1.0         # bets that lose when called: at most the pot
BET_MAX_FRACTION = 1.5           # other bets: at most this (or the geometric size) ...
STATION_MIN_BETS = 5             # ... unless they've called at least this many large bets
STATION_CALL_RATE = 0.7          #     at this rate or more

# Opponent hands that haven't made a pair but have a draw rank like a middling
# pair when we estimate their range.
DRAW_SCORE = eval7.evaluate([eval7.Card(c) for c in ("8s", "8d", "Kc", "5h", "3d")])
MIN_RANGE_COMBOS = 20


# ── Hand helpers ──────────────────────────────────────────────────────────────

def hand_class(cards):
    """['7h', '2h'] -> '72s', ['Kd', 'As'] -> 'AKo', ['9c', '9d'] -> '99'."""
    a, b = sorted(cards, key=lambda c: RANKS.index(c[0]))
    if a[0] == b[0]:
        return a[0] + b[0]
    return a[0] + b[0] + ("s" if a[1] == b[1] else "o")


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


def has_draw(cards, board_len):
    """A flush draw or open-ended straight draw (cards = hole cards + board)."""
    if board_len == 5:
        return False
    suits = [c[1] for c in cards]
    if any(suits.count(s) == 4 for s in "shdc"):
        return True
    ranks = {RANKS.index(c[0]) for c in cards}
    # Four in a row with room at both ends (an ace can't be an open end).
    return any(all(r + k in ranks for k in range(4)) and r >= 1 and r + 3 <= 11
               for r in range(13))


def strength(opp, board):
    """How strong an opponent hand looks now: its made hand, with draws counted as a middling pair."""
    value = eval7.evaluate(opp + board)
    if value >> 24 <= 1 and has_draw([str(c) for c in opp + board], len(board)):
        value = max(value, DRAW_SCORE)
    return value


class RangeSample:
    """Simulated showdowns against the opponent's estimated range.

    The range is every hand in the top `width` of starting hands, then the
    strongest `keep` share of those on this board, ordered strongest first.
    Each simulation records which hand in that order the opponent held, so
    `equity(share)` gives our equity against just the strongest `share` of the
    range (the hands that would call a big bet) from the same simulations.
    """

    def __init__(self, hole, board, width, keep, sims=SIMULATIONS):
        self.results = range_showdowns(hole, board, width, keep, sims)
        self.size = max(i for i, _ in self.results) + 1 if self.results else 1

    def equity(self, share=1.0):
        """Equity (ties count half) against the strongest `share` of the range."""
        cutoff = max(MIN_RANGE_COMBOS, round(self.size * share))
        outcomes = [r for i, r in self.results if i < cutoff]
        return sum(outcomes) / len(outcomes) if outcomes else 0.5


def range_equity(hole, board, width, keep, sims=SIMULATIONS):
    """Equity against the opponent's estimated range (ties count half)."""
    return RangeSample(hole, board, width, keep, sims).equity()


def range_showdowns(hole, board, width, keep, sims):
    """[(index of the opponent's hand in the range, our result 1 / 0.5 / 0)] per simulation."""
    hero = [eval7.Card(c) for c in hole]
    known = [eval7.Card(c) for c in board]
    used = set(hole) | set(board)
    deck = [c for c in FULL_DECK if str(c) not in used]
    to_come = 5 - len(known)

    candidates = []
    for i in range(len(deck)):
        for j in range(i + 1, len(deck)):
            opp = [deck[i], deck[j]]
            cls = hand_class([str(deck[i]), str(deck[j])])
            if TOP_PCT[cls] <= width:
                candidates.append((strength(opp, known) if known else PREFLOP_EQUITY[cls], opp))
    candidates.sort(key=lambda x: -x[0])
    n = min(len(candidates), max(MIN_RANGE_COMBOS, round(len(candidates) * keep)))
    opp_range = [opp for _, opp in candidates[:n]]

    results = []
    for _ in range(sims):
        index = random.randrange(n)
        opp = opp_range[index]
        drawn = random.sample(deck, to_come + 2)
        full_board = known + [c for c in drawn if c not in opp][:to_come]
        ours = eval7.evaluate(hero + full_board)
        theirs = eval7.evaluate(opp + full_board)
        results.append((index, 1.0 if ours > theirs else 0.5 if ours == theirs else 0.0))
    return results


def hand_label(hole, board):
    """The made hand so far, e.g. 'Two Pair' (preflop: the hand class)."""
    if not board:
        return hand_class(hole)
    return eval7.handtype(eval7.evaluate([eval7.Card(c) for c in hole + board]))


def is_seven_two(hole):
    return hand_class(hole) in ("72s", "72o")


def flush_draw_72s(hole, board):
    """Suited 7-2 with four of its suit counting the board, cards still to come."""
    if hand_class(hole) != "72s" or len(board) == 5:
        return False
    suit = hole[0][1]
    return sum(c[1] == suit for c in board) == 2


# ── Game-state helpers ────────────────────────────────────────────────────────

def pot_odds(state):
    """Equity needed to call: call / (pot after we call).

    If the opponent bet more than we have, we can only call our stack and the
    excess goes back to them, so it isn't part of the pot we're playing for.
    """
    owed = state["amount_owed"]
    call = min(owed, state["your_stack"])
    pot = state["pot"] - (owed - call)
    return call / (pot + call)


def opponent(state):
    return next(p for p in state["players"] if p["bot_id"] != state["your_bot_id"])


def effective_bb(state):
    """The smaller stack, counting chips already bet this street, in big blinds."""
    opp = opponent(state)
    ours = state["your_stack"] + state["your_bet_this_street"]
    theirs = opp["stack"] + opp["bet_this_street"]
    return min(ours, theirs) / state["blinds"]["bb"]


def preflop_raises(state, by_us):
    me = state["your_bot_id"]
    return sum(1 for e in state["action_log"]
               if e["street"] == "preflop" and e["action"] in ("raise", "all_in")
               and (e["bot_id"] == me) == by_us)


def update_stats(state):
    """Count, over finished hands, how the opponent plays: folds to our first
    preflop raise, how often they raise preflop, and postflop how often they
    bet when they can and how they answer a bet."""
    me = state["your_bot_id"]
    hands = {}
    for e in state.get("match_action_log") or []:
        h = e.get("hand_num")
        if h is not None and h < state["hand_num"] and h not in _seen_hands:
            hands.setdefault(h, []).append(e)
    for h, entries in hands.items():
        _seen_hands.add(h)
        if not any(e["action"] == "big_blind" for e in entries):
            continue                             # start of the hand fell out of the 200-entry log
        moves = [e for e in entries if e["action"] in ("fold", "check", "call", "raise", "all_in")]
        preflop_moves = [e for e in moves if e["street"] == "preflop"]
        for i, e in enumerate(preflop_moves):
            if e["bot_id"] == me and e["action"] in ("raise", "all_in"):
                reply = next((x for x in preflop_moves[i + 1:] if x["bot_id"] != me), None)
                if reply:
                    STATS["our_preflop_raises"] += 1
                    STATS["folds_to_them"] += reply["action"] == "fold"
                break
        STATS["hands"] += 1
        STATS["preflop_raised"] += any(e["bot_id"] != me and e["action"] in ("raise", "all_in")
                                       for e in preflop_moves)

        # Postflop, replaying the pot so we know how big each bet they faced was.
        pot, street, bet_open, put_in = 0, None, False, {}
        for e in entries:
            action, amount, who = e["action"], e.get("amount") or 0, e["bot_id"]
            if e["street"] != street:
                street, bet_open, put_in = e["street"], False, {}
            aggressive = action in ("raise", "all_in")
            if who != me and street != "preflop" and action in ("fold", "check", "call", "raise", "all_in"):
                if bet_open:
                    to_call = max(put_in.values()) - put_in.get(who, 0)
                    bucket = size_bucket(to_call / max(1, pot - to_call))
                    STATS["faced_bets"] += 1
                    STATS["raises"] += aggressive
                    STATS["calls"] += action == "call"
                    STATS["faced_by_size"][bucket] += 1
                    STATS["continued_by_size"][bucket] += action != "fold"
                else:
                    STATS["bet_chances"] += 1
                    STATS["bets"] += aggressive
            if action == "ante":
                pot += amount
            elif action in ("small_blind", "big_blind", "call"):
                pot += amount
                put_in[who] = put_in.get(who, 0) + amount
            elif aggressive:
                pot += amount - put_in.get(who, 0)
                put_in[who] = amount
            if street != "preflop":
                bet_open = bet_open or aggressive


def opponent_folds_to_raise():
    """Their fold rate to our preflop raises, shrunk toward 40% (prior worth 5 hands)."""
    return (STATS["folds_to_them"] + 2) / (STATS["our_preflop_raises"] + 5)


def shrunk(count, chances, prior, weight=10):
    """count / chances, pulled toward `prior` as if we'd seen `weight` extra chances."""
    return (count + prior * weight) / (chances + weight)


def opponent_range(state):
    """(width, keep) for range_equity, from what the opponent did this hand.

    Each action keeps the share of hands this opponent takes it with: if they
    bet 30% of the time when checked to, a bet keeps their strongest 30%. A
    maniac who bets everything barely narrows; a nit narrows a lot.
    """
    me = state["your_bot_id"]
    moves = [e for e in state["action_log"]
             if e["action"] in ("fold", "check", "call", "raise", "all_in")]

    preflop_moves = [e for e in moves if e["street"] == "preflop"]
    their_raises = sum(e["bot_id"] != me and e["action"] in ("raise", "all_in")
                       for e in preflop_moves)
    width = 1.0
    if their_raises:
        width = shrunk(STATS["preflop_raised"], STATS["hands"], 0.45) * 0.35 ** (their_raises - 1)
    raises = [e for e in preflop_moves if e["action"] in ("raise", "all_in")]
    if raises and raises[-1]["bot_id"] == me:
        width *= 0.75                            # they called our raise
    width = max(width, 0.03)

    bet_freq = shrunk(STATS["bets"], STATS["bet_chances"], 0.45)
    raise_freq = shrunk(STATS["raises"], STATS["faced_bets"], 0.25)
    continue_freq = shrunk(STATS["raises"] + STATS["calls"], STATS["faced_bets"], 0.60)

    keep, street, bet_open = 1.0, None, False
    for e in moves:
        if e["street"] == "preflop":
            continue
        if e["street"] != street:
            street, bet_open = e["street"], False
        aggressive = e["action"] in ("raise", "all_in")
        if e["bot_id"] != me:
            if aggressive:
                keep *= raise_freq if bet_open else bet_freq
            elif e["action"] == "call":
                keep *= continue_freq
        bet_open = bet_open or aggressive
    return width, keep


def size_bucket(fraction):
    """0, 1 or 2: a bet of `fraction` of the pot is small, medium or large."""
    return sum(fraction > edge for edge in SIZE_BUCKETS)


def prior_continue(fraction):
    """Before we know them: the share of hands an opponent continues with
    against a bet of `fraction` of the pot. Close to the rational defence rate
    1 / (1 + fraction), tilted slightly toward calling, since hackathon bots
    fold too little more often than too much. The tilt is kept small because
    we assume callers hold the strongest hands: a big floor would make huge
    overbets look profitable against opponents we know nothing about."""
    return 0.05 + 0.95 / (1 + fraction)


def continue_probability(fraction):
    """How likely this opponent is to call or raise a bet of `fraction` of the pot.

    Starts from the prior, scaled by how much they continue overall, then
    learns from how they answered bets of this size.
    """
    overall = shrunk(STATS["raises"] + STATS["calls"], STATS["faced_bets"], 0.60) / 0.60
    bucket = size_bucket(fraction)
    typical = BUCKET_TYPICAL[bucket]
    bucket_prior = min(0.98, max(0.02, prior_continue(typical) * overall))
    learned = shrunk(STATS["continued_by_size"][bucket], STATS["faced_by_size"][bucket],
                     bucket_prior, SIZE_PRIOR_WEIGHT)
    # Smooth within the bucket: a slightly bigger bet is called slightly less.
    return min(0.98, max(0.02, learned * prior_continue(fraction) / prior_continue(typical)))


def raise_share():
    """Of the times they don't fold to a bet, how often they raise it."""
    return shrunk(STATS["raises"], STATS["raises"] + STATS["calls"], 0.30)


def bet_ev(state, sample, extra):
    """(EV, expected take, call chance, equity when called) of calling and raising by `extra` more.

    EV counts the chips we get back minus the chips we put in from now on, so
    folding is 0. Expected take is the part won when they call: the chips we
    expect to extract with a value bet.
    """
    pot = state["pot"]
    owed = min(state["amount_owed"], state["your_stack"])
    behind = pot + owed                              # the pot once we've matched their bet
    all_in = extra >= max_raise(state)

    cont = continue_probability(extra / behind)
    raised = 0.0 if all_in else cont * raise_share()
    called = cont - raised

    eq_called = sample.equity(cont)                  # against the hands that would continue
    take = called * (eq_called * (behind + 2 * extra) - owed - extra)
    ev = (1 - cont) * pot + take
    if raised:
        eq_raised = sample.equity(raised)
        if eq_raised >= STACK_OFF_EQUITY:            # they raise to about 3x our bet, we call
            total = min(max_raise(state), RAISE_MULTIPLE * extra)
            ev += raised * (eq_raised * (behind + 2 * total) - owed - total)
        else:                                        # they raise, we fold
            ev -= raised * (owed + extra)
    return ev, take, cont, eq_called


def confirmed_station():
    """Seen calling large bets often enough that overbetting them for value is safe."""
    faced, continued = STATS["faced_by_size"][2], STATS["continued_by_size"][2]
    return faced >= STATION_MIN_BETS and continued / faced >= STATION_CALL_RATE


def size_allowed(state, extra, eq_called):
    """Risk limits on top of the EV model, where its call estimates are least
    reliable and a mistake costs the stack: bluffs (bets that lose when
    called) at most the pot, and overbets only against a confirmed station."""
    behind = state["pot"] + min(state["amount_owed"], state["your_stack"])
    fraction = extra / behind
    if eq_called < 0.5 and fraction > BLUFF_MAX_FRACTION:
        return False
    cap = max(BET_MAX_FRACTION, geometric_fraction(state))
    return fraction <= cap + 1e-9 or confirmed_station()


def max_raise(state):
    """The most we can usefully raise by, beyond calling: the smaller remaining stack."""
    owed = min(state["amount_owed"], state["your_stack"])
    return max(0, min(state["your_stack"] - owed, opponent(state)["stack"]))


def geometric_fraction(state):
    """The pot fraction that, bet on every street left, gets the stacks in by the river."""
    behind = state["pot"] + min(state["amount_owed"], state["your_stack"])
    streets = STREETS_LEFT[state["street"]]
    return ((1 + 2 * max_raise(state) / behind) ** (1 / streets) - 1) / 2


def bet_sizes(state):
    """Candidate raises (chips beyond calling): pot fractions, the geometric
    size that gets the stacks in by the river, and all-in."""
    room = max_raise(state)
    if room <= 0:
        return []
    behind = state["pot"] + min(state["amount_owed"], state["your_stack"])
    smallest = state["min_raise_to"] - state["current_bet"]
    sizes = {room}
    for fraction in BET_FRACTIONS + (geometric_fraction(state),):
        sizes.add(min(room, max(smallest, int(fraction * behind))))
    return sorted(s for s in sizes if s > 0)


def raise_to(state, amount):
    """A raise to `amount` total this street, as all-in if we can't cover more."""
    amount = max(int(amount), state["min_raise_to"])
    if amount >= state["your_stack"] + state["your_bet_this_street"]:
        return {"action": "all_in"}
    return {"action": "raise", "amount": amount}


def check_or_fold(state):
    return {"action": "check"} if state["can_check"] else {"action": "fold"}


# ── Preflop ───────────────────────────────────────────────────────────────────

def push_pct(eff_bb):
    """Approximate heads-up push range from the button (stand-in for a Nash chart)."""
    return min(0.85, 0.25 + 3.0 / eff_bb)


def call_shove_pct(eff_bb):
    """Approximate range for calling an all-in from the big blind."""
    return min(0.70, 0.10 + 2.7 / eff_bb)


def preflop(state, cls, top):
    bb = state["blinds"]["bb"]
    eff = effective_bb(state)
    on_button = state["seat_to_act"] == state["button_seat"]
    they_raised = preflop_raises(state, by_us=False) > 0
    we_raised = preflop_raises(state, by_us=True) > 0

    # Short stacks: push or fold.
    if eff <= SHORT_STACK_BB:
        if not they_raised:
            shove = top <= (push_pct(eff) if on_button else 0.25)
            return {"action": "all_in"} if shove else check_or_fold(state)
        return {"action": "all_in"} if top <= call_shove_pct(eff) else check_or_fold(state)

    # Button, first in: raise to 2.5bb or fold.
    if on_button and not they_raised:
        steal_72o = cls == "72o" and STATS["our_preflop_raises"] >= 8 \
            and opponent_folds_to_raise() > 0.60
        if top <= 0.50 or cls == "72s" or steal_72o:
            return raise_to(state, 2.5 * bb)
        return {"action": "fold"}

    # Big blind, the button just called: raise our good hands, check the rest.
    if state["can_check"]:
        return raise_to(state, 4 * bb) if top <= 0.20 else {"action": "check"}

    # Facing a raise: compare our hand with the hands this opponent raises with
    # (wide for a maniac, narrow for a nit). Re-raise when well ahead of that
    # range, call when the pot odds allow, otherwise fall back on the chart.
    width, _ = opponent_range(state)
    eq = range_equity(state["your_cards"], [], width, 1.0, sims=PREFLOP_RANGE_SIMS)
    if eq >= RERAISE_EQUITY:
        return raise_to(state, 3 * state["current_bet"])
    if eq >= pot_odds(state) + OUT_OF_POSITION_MARGIN:
        return {"action": "call"}
    if we_raised:
        return {"action": "fold"}

    # Defend a range that shrinks as the raise grows.
    raise_bb = state["current_bet"] / bb
    if raise_bb <= 3:
        defend = 0.40
    elif raise_bb <= 5:
        defend = 0.25
    elif raise_bb <= 12:
        defend = 0.12
    else:
        defend = 0.06
    if top <= defend or (cls == "72s" and raise_bb <= 3):
        return {"action": "call"}
    return {"action": "fold"}


# ── Postflop ──────────────────────────────────────────────────────────────────

def postflop(state, sample):
    """Pick the option with the highest EV (strategy.md section 3).

    Without betting: check (our equity share of the pot), or call / fold.
    Betting: every candidate size, scored by how likely this opponent is to
    fold, call or raise at that size and our equity against the hands that
    would call. Returns (move, debug info).
    """
    hole, board = state["your_cards"], state["community_cards"]
    pot = state["pot"]
    eq = sample.equity()

    if state["can_check"]:
        options = [({"action": "check"}, eq * pot, 0.0, None)]
    else:
        call = min(state["amount_owed"], state["your_stack"])
        live_pot = pot - (state["amount_owed"] - call)
        options = [({"action": "fold"}, 0.0, 0.0, None),
                   ({"action": "call"}, eq * (live_pot + call) - call, 0.0, None)]
    bets = []
    for extra in bet_sizes(state):
        ev, take, cont, eq_called = bet_ev(state, sample, extra)
        if size_allowed(state, extra, eq_called):
            bets.append((raise_to(state, state["current_bet"] + extra), ev, take, cont))

    best = max(options + bets, key=lambda o: o[1])

    # 7-2 suited with a flush draw: play it like a high flush draw (strategy.md 7.1).
    if flush_draw_72s(hole, board) and best[0]["action"] in ("check", "call", "fold") and bets:
        if state["street"] == "flop" and state["your_bet_this_street"] > 0:
            best = next(o for o in options if o[0]["action"] == "call")   # get it in on the flop
        else:
            best = max(bets, key=lambda o: o[1])                           # semi-bluff

    move, ev, take, cont = best
    info = {"ev": round(ev), "check_or_call_ev": round(max(o[1] for o in options))}
    if cont is not None:
        info.update(expected_take=round(take), call_chance=round(cont, 2))
    return move, info


# ── Entry point ───────────────────────────────────────────────────────────────

def decide(game_state: dict) -> dict:
    if game_state.get("type") == "warmup":
        equity(["As", "Kd"], [], sims=200)       # warm up eval7 before the clock matters
        return {"action": "check"}

    try:
        update_stats(game_state)
        hole = game_state["your_cards"]
        board = game_state["community_cards"]
        cls = hand_class(hole)
        debug = {"hand": hand_label(hole, board), "top_pct": round(TOP_PCT[cls], 3)}

        if game_state["street"] == "preflop":
            move = preflop(game_state, cls, TOP_PCT[cls])
        else:
            width, keep = opponent_range(game_state)
            sample = RangeSample(hole, board, width, keep, sims=POSTFLOP_SIMS)
            debug.update(equity=round(sample.equity(), 3),
                         range_width=round(width, 3), range_keep=round(keep, 3))
            if not game_state["can_check"]:
                debug["needed"] = round(pot_odds(game_state), 3)
            move, info = postflop(game_state, sample)
            debug.update(info)

        if is_seven_two(hole):
            debug["seven_two"] = cls
        move["debug"] = debug
        return move
    except Exception as e:                       # never crash: check if free, else fold
        return {"action": "check" if game_state.get("can_check") else "fold",
                "debug": {"error": repr(e)}}
