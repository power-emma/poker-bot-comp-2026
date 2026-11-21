#!/usr/bin/env python3
"""Profile a bot's own tendencies the way a strong opponent would: frequencies, bet-size tells, think time.

    python3 sparring/profile.py my_bots/tag --pairs 10

Plays the bot against every sparring bot (minus the degenerate ones) and measures, from its decisions:
  preflop  button open / limp / fold, big blind fold / call / 3-bet vs opens, fold to 3-bet and to shoves
  postflop c-bet and probe frequency, fold / call / raise vs bets (by street and bet size), bet-size mix
  tells    does the size of its bets say anything about its hand strength? (equity vs a random hand by size)
  speed    think-time mean / p99 / max
"""
import argparse
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SKIP = {"folder", "fuzz_badsize", "fuzz_crasher", "fuzz_random", "fuzz_wrongcheck", "shover"}


class Collect:
    def __init__(self):
        self.events = []

    def emit(self, t, **d):
        if t == "decision":
            self.events.append(d)
        return {}


def play(task):
    focal, opp, seed, ff = task
    from pokercore import structures
    from pokercore.seats import BotSeat
    from pokercore.seeds import derive_seed
    from pokercore.table import MatchConfig, play_match
    from play import Bot
    f, o = Bot("focal", focal), Bot("opp", opp)
    order = [f, o] if ff else [o, f]
    cfg = MatchConfig(mode="freezeout", hands=600, starting_stack=10_000,
                      structure=structures.parse("standard", 10_000), seed=seed)
    seats = [BotSeat(b, seed=derive_seed(seed, "bot", b.name), use_docker=False) for b in order]
    log = Collect()
    try:
        play_match(seats, cfg, "prof", log)
    finally:
        for s in seats:
            try:
                s.close()
            except Exception:
                pass
    out = []
    for e in log.events:
        if e["bot"] != "focal":
            continue
        st = e["state"]
        out.append(dict(street=st["street"], hole=st["your_cards"], board=st["community_cards"],
                        pot=st["pot"], owed=st["amount_owed"], can_check=st["can_check"], cur=st["current_bet"],
                        stack=st["your_stack"], bb=st["blinds"]["bb"], btn=st["seat_to_act"] == st["button_seat"],
                        log=[(x["bot_id"], x["action"], x.get("amount"), x["street"]) for x in st["action_log"]],
                        act=e["applied"]["action"], amt=e["applied"]["amount"], think=e.get("think_ms") or 0,
                        opp=Path(opp).name))
    return out


def equity(hole, board, sims=150):
    import eval7
    deck = [eval7.Card(r + s) for r in "23456789TJQKA" for s in "shdc"]
    h = [eval7.Card(c) for c in hole]
    b = [eval7.Card(c) for c in board]
    used = set(hole) | set(board)
    deck = [c for c in deck if str(c) not in used]
    w = 0.0
    for _ in range(sims):
        d = random.sample(deck, 2 + 5 - len(b))
        fb = b + d[2:]
        a, o = eval7.evaluate(h + fb), eval7.evaluate(d[:2] + fb)
        w += 1 if a > o else 0.5 if a == o else 0
    return w / sims


def pct(n, d):
    return f"{100 * n / d:5.1f}%" if d else "   n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("focal")
    ap.add_argument("--pairs", type=int, default=10)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--opp", help="comma list of sparring bot names (default: all non-degenerate)")
    args = ap.parse_args()
    if args.opp:
        opps = [str(ROOT / "sparring" / "bots" / o) for o in args.opp.split(",")]
    else:
        opps = sorted(str(p.parent) for p in (ROOT / "sparring" / "bots").glob("*/bot.py") if p.parent.name not in SKIP)
    tasks = [(args.focal, o, 9000 + i, ff) for o in opps for i in range(args.pairs) for ff in (True, False)]
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(play, tasks, chunksize=2):
            rows.extend(r)
    print(f"{args.focal}: {len(tasks)} matches vs {len(opps)} bots, {len(rows)} decisions\n")

    # ---- speed
    th = sorted(r["think"] for r in rows)
    print(f"THINK TIME (ms): mean {sum(th) / len(th):.0f}  p99 {th[int(.99 * len(th))]:.0f}  max {th[-1]:.0f}"
          f"   (limit 2000; on a 0.5-core sandbox multiply by ~2)")
    slow = Counter(r["opp"] for r in rows if r["think"] > 800)
    print(f"  decisions over 800ms: {sum(slow.values())}" + (f"  {dict(slow.most_common(4))}" if slow else ""))

    # ---- preflop
    pre = [r for r in rows if r["street"] == "preflop"]
    def opp_raises(r):
        return sum(1 for b, a, amt, s in r["log"] if s == "preflop" and b != "focal" and a in ("raise", "all_in"))
    def my_raises(r):
        return sum(1 for b, a, amt, s in r["log"] if s == "preflop" and b == "focal" and a in ("raise", "all_in"))
    deep = [r for r in pre if min(r["stack"] + 1, 10 ** 9) / r["bb"] > 25]
    print("\nPREFLOP (stacks > 25bb)")
    first_in = [r for r in deep if r["btn"] and opp_raises(r) == 0 and my_raises(r) == 0]
    c = Counter("raise" if r["act"] in ("raise", "all_in") else "limp" if r["act"] == "call" else "fold" for r in first_in)
    print(f"  button first in (n={len(first_in)}): raise {pct(c['raise'], len(first_in))}  limp {pct(c['limp'], len(first_in))}  fold {pct(c['fold'], len(first_in))}")
    bb_vs = [r for r in deep if not r["btn"] and opp_raises(r) == 1 and my_raises(r) == 0]
    c = Counter("3bet" if r["act"] in ("raise", "all_in") else r["act"] for r in bb_vs)
    print(f"  big blind vs open (n={len(bb_vs)}): fold {pct(c['fold'], len(bb_vs))}  call {pct(c['call'], len(bb_vs))}  3-bet {pct(c['3bet'], len(bb_vs))}")
    by_size = defaultdict(Counter)
    for r in bb_vs:
        sz = r["cur"] / r["bb"]
        by_size["<=2.5bb" if sz <= 2.6 else "<=4bb" if sz <= 4 else "<=10bb" if sz <= 10 else ">10bb"][r["act"]] += 1
    for k in ("<=2.5bb", "<=4bb", "<=10bb", ">10bb"):
        n = sum(by_size[k].values())
        if n:
            print(f"      open {k:<8} n={n:>5}  fold {pct(by_size[k]['fold'], n)}")
    vs3 = [r for r in deep if my_raises(r) >= 1 and opp_raises(r) >= 1]
    c = Counter("4bet" if r["act"] in ("raise", "all_in") else r["act"] for r in vs3)
    print(f"  after we opened, facing a re-raise (n={len(vs3)}): fold {pct(c['fold'], len(vs3))}  call {pct(c['call'], len(vs3))}  4-bet/jam {pct(c['4bet'], len(vs3))}")
    jams = [r for r in vs3 if r["owed"] >= r["stack"]]
    c = Counter(r["act"] for r in jams)
    print(f"      ... of which facing a shove (n={len(jams)}): fold {pct(c['fold'], len(jams))}  call {pct(c['call'] + c['all_in'], len(jams))}")

    # ---- postflop
    post = [r for r in rows if r["street"] != "preflop"]
    print("\nPOSTFLOP")
    for street in ("flop", "turn", "river"):
        rs = [r for r in post if r["street"] == street]
        chk = [r for r in rs if r["can_check"]]
        bets = [r for r in chk if r["act"] in ("raise", "all_in")]
        face = [r for r in rs if not r["can_check"]]
        c = Counter("raise" if r["act"] in ("raise", "all_in") else r["act"] for r in face)
        print(f"  {street:<6} checked-to (n={len(chk):>5}): bet {pct(len(bets), len(chk))}   "
              f"facing bet (n={len(face):>5}): fold {pct(c['fold'], len(face))}  call {pct(c['call'], len(face))}  raise {pct(c['raise'], len(face))}")
    print("  folds when facing a bet, by bet size (fraction of pot before the bet):")
    for lo, hi, nm in ((0, .4, "<=40%"), (.4, .8, "40-80%"), (.8, 1.3, "80-130%"), (1.3, 3, "130-300%"), (3, 1e9, ">300%")):
        rs = [r for r in post if not r["can_check"] and r["pot"] - r["owed"] > 0 and lo < r["owed"] / (r["pot"] - r["owed"]) <= hi]
        if rs:
            print(f"      {nm:<9} n={len(rs):>5}  fold {pct(sum(r['act'] == 'fold' for r in rs), len(rs))}")

    # ---- size tells
    print("\nBET-SIZE TELLS: tag's bets (first bet on a street), equity vs a random hand, by size")
    rng = random.Random(1)
    sample = [r for r in post if r["can_check"] and r["act"] in ("raise", "all_in")]
    rng.shuffle(sample)
    for street in ("flop", "turn", "river"):
        print(f"  {street}:")
        buckets = defaultdict(list)
        for r in [x for x in sample if x["street"] == street][:1500]:
            frac = r["amt"] / max(1, r["pot"])
            b = "<=33%" if frac <= .38 else "33-60%" if frac <= .65 else "60-90%" if frac <= .95 else "90-150%" if frac <= 1.6 else ">150%"
            buckets[b].append(equity(r["hole"], r["board"]))
        for b in ("<=33%", "33-60%", "60-90%", "90-150%", ">150%"):
            v = buckets.get(b)
            if v and len(v) >= 15:
                print(f"      {b:<8} n={len(v):>4}  mean equity {sum(v) / len(v):.2f}   bluffs (<40% eq) {pct(sum(x < .4 for x in v), len(v))}")


if __name__ == "__main__":
    main()
