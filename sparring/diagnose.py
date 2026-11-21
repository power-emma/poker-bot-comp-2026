#!/usr/bin/env python3
"""Where does the focal bot win and lose chips against an opponent?

    python3 sparring/diagnose.py my_bots/tag nash --seeds 100

Plays duplicate pairs, then splits every hand's result (in big blinds of that hand) by effective
stack depth at the start of the hand, by position (button = small blind heads-up), and by how the
hand went: folded preflop, preflop all-in, or saw a flop. Duplicate pairs cancel card luck, so
persistent negative cells are leaks, not variance (though each cell is still noisy: see n and se).
"""
import argparse
import math
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEPTHS = [(0, 6, "<6bb"), (6, 10, "6-10"), (10, 15, "10-15"), (15, 18, "15-18"), (18, 25, "18-25"),
          (25, 40, "25-40"), (40, 80, "40-80"), (80, 1e9, "80+")]


class Collect:
    def __init__(self):
        self.events = []

    def emit(self, t, **d):
        if t in ("hand_start", "decision", "match_start"):
            self.events.append({"type": t, **d})
        return {}


def run_one(task):
    focal, opp, seed, focal_first = task
    from pokercore import structures
    from pokercore.seats import BotSeat
    from pokercore.seeds import derive_seed
    from pokercore.table import MatchConfig, play_match
    from play import Bot
    f, o = Bot("focal", focal), Bot("opp", opp)
    order = [f, o] if focal_first else [o, f]
    cfg = MatchConfig(mode="freezeout", hands=1000, starting_stack=10_000,
                      structure=structures.parse("standard", 10_000), seed=seed)
    seats = [BotSeat(b, seed=derive_seed(seed, "bot", b.name), use_docker=False) for b in order]
    log = Collect()
    try:
        res = play_match(seats, cfg, "diag", log)
    finally:
        for s in seats:
            try:
                s.close()
            except Exception:
                pass
    final = {r["bot"]: r["final_stack"] for r in res["results"]}
    starts = [e for e in log.events if e["type"] == "hand_start"]
    decs = defaultdict(list)
    for e in log.events:
        if e["type"] == "decision":
            decs[e["hand"]].append(e)
    rows = []
    for i, hs in enumerate(starts):
        h = hs["hand"]
        st = hs["stacks"]
        nxt = starts[i + 1]["stacks"]["focal"] if i + 1 < len(starts) else final["focal"]
        mine = [e for e in decs[h] if e["bot"] == "focal"]
        if not mine or "focal" not in st:
            continue
        first = decs[h][0]["state"]
        bb = first["blinds"]["bb"]
        depth = min(st["focal"], st["opp"]) / bb
        on_button = mine[0]["state"]["seat_to_act"] == mine[0]["state"]["button_seat"]
        pre_allin = any(e["street"] == "preflop" and e["applied"]["action"] == "all_in" for e in decs[h])
        flop = any(e["street"] != "preflop" for e in decs[h])
        i_shoved = any(e["bot"] == "focal" and e["street"] == "preflop" and e["applied"]["action"] == "all_in" for e in decs[h])
        i_folded_pre = any(e["bot"] == "focal" and e["street"] == "preflop" and e["applied"]["action"] == "fold" for e in decs[h])
        kind = ("fold-pre" if i_folded_pre else "pre-allin" if pre_allin else "saw-flop" if flop else "other-pre")
        if pre_allin and kind == "pre-allin":
            kind = "pre-allin (we shove)" if i_shoved else "pre-allin (call shove)"
        if i_folded_pre and pre_allin:
            kind = "fold-pre (to shove)"
        rows.append((depth, on_button, kind, (nxt - st["focal"]) / bb))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("focal")
    ap.add_argument("opp")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    opp = args.opp if Path(args.opp).exists() else str(ROOT / "sparring" / "bots" / args.opp)
    tasks = [(args.focal, opp, args.seed_base + i, ff) for i in range(args.seeds) for ff in (True, False)]
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(run_one, tasks, chunksize=2):
            rows.extend(r)
    print(f"{args.focal} vs {args.opp}: {len(tasks)} matches, {len(rows)} hands, "
          f"total {sum(r[3] for r in rows):+.0f} bb ({sum(r[3] for r in rows) / len(tasks):+.1f} bb/match)\n")

    def table(title, keyf, keys):
        print(title)
        print(f"  {'':<26}{'hands':>7}{'bb total':>10}{'bb/hand':>9}{'se':>7}")
        for k in keys:
            xs = [r[3] for r in rows if keyf(r) == k]
            if not xs:
                continue
            m = sum(xs) / len(xs)
            sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1))
            print(f"  {str(k):<26}{len(xs):>7}{sum(xs):>+10.0f}{m:>+9.2f}{sd / math.sqrt(len(xs)):>7.2f}")
        print()

    def dkey(r):
        return next(n for lo, hi, n in DEPTHS if lo <= r[0] < hi)
    table("by effective depth:", dkey, [n for _, _, n in DEPTHS])
    table("by position:", lambda r: "button (SB)" if r[1] else "big blind", ["button (SB)", "big blind"])
    kinds = sorted({r[2] for r in rows})
    table("by how the hand went:", lambda r: r[2], kinds)
    print("depth x position x line, only cells with |total| >= 40 bb:")
    cells = defaultdict(list)
    for r in rows:
        cells[(dkey(r), "BTN" if r[1] else "BB", r[2])].append(r[3])
    for k, xs in sorted(cells.items(), key=lambda kv: sum(kv[1])):
        if abs(sum(xs)) >= 40:
            print(f"  {k[0]:<7}{k[1]:<4}{k[2]:<26}n={len(xs):>5}  total {sum(xs):>+7.0f} bb  ({sum(xs) / len(xs):+.2f}/hand)")


if __name__ == "__main__":
    main()
