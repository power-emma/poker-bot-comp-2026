"""Feature vector for "how will this opponent answer a postflop bet?".

Shared by the data collector and (pasted verbatim) by my_bots/tag/bot.py, so training and
play see identical features. Only uses what a bot can see: the bet, the pot, the board and
running counts of how this opponent has behaved so far in the match (tag's STATS dict).
No hole cards (the opponent's are hidden) and no per-hand identity, to limit overfitting.
"""

RF_RANKS = "23456789TJQKA"
RF_FEATURES = ("street", "frac", "spr", "pot_bb", "paired", "suit_max", "high", "spread",
               "hands", "pfr", "bet_rate", "n_faced", "cont_rate", "raise_rate",
               "bucket_cont", "bucket_n", "fold_to_pfr", "n_our_raises", "later")


def rf_features(street, frac, pot_before, bb, behind, board, stats, bucket, later=0):
    """street 'flop'|'turn'|'river'; frac = bet / pot before the bet; behind = chips the
    bettor's opponent has left after calling; bucket = tag's size bucket for frac;
    later = 1 once they have already continued against a bet this hand."""
    ranks = sorted(RF_RANKS.index(c[0]) for c in board)
    suits = [c[1] for c in board]
    s = stats
    faced = s["faced_bets"]
    return [
        {"flop": 0, "turn": 1, "river": 2}[street],
        frac,
        behind / max(1, pot_before),
        pot_before / max(1, bb),
        int(len(set(ranks)) < len(ranks)),
        max(suits.count(x) for x in "shdc"),
        ranks[-1],
        ranks[-1] - ranks[0],
        s["hands"],
        (s["preflop_raised"] + 1) / (s["hands"] + 2),
        (s["bets"] + 1) / (s["bet_chances"] + 2),
        faced,
        (s["raises"] + s["calls"] + 1) / (faced + 2),
        (s["raises"] + 1) / (faced + 2),
        (s["continued_by_size"][later][bucket] + 1) / (s["faced_by_size"][later][bucket] + 2),
        s["faced_by_size"][later][bucket],
        (s["folds_to_them"] + 1) / (s["our_preflop_raises"] + 2),
        s["our_preflop_raises"],
        later,
    ]
