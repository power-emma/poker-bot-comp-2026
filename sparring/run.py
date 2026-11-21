#!/usr/bin/env python3
"""Duplicate round-robin of one focal bot against many sparring bots.

    python3 sparring/run.py my_bots/tag                       # every bot in sparring/bots, 20 seeds each
    python3 sparring/run.py my_bots/tag --seeds 50 --workers 4
    python3 sparring/run.py my_bots/tag --opp nit,trapper,sparring/bots/nash
    python3 sparring/run.py my_bots/tag --json results.json   # also dump every match

For each seed S and opponent O two freezeout matches are played with the SAME seed
(so the same cards) and the seats swapped, exactly like the tournament's duplicate
pairs. A pairing is a WIN (both matches), LOSS (neither) or DRAW (one each). Losses
and draws are listed with the seed so you can replay them:

    python3 play.py my_bots/tag sparring/bots/nit --seed 17 --show
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class NullLog:
    def emit(self, event_type, **data):
        return {"type": event_type, **data}


def play(task):
    focal, opp, seed, focal_first, hands = task
    from pokercore import structures
    from pokercore.seats import BotSeat
    from pokercore.seeds import derive_seed
    from pokercore.table import MatchConfig, play_match
    from play import Bot

    names = [Path(focal).resolve().name, Path(opp).resolve().name]
    if names[0] == names[1]:
        names[1] += "_2"
    fb, ob = Bot(names[0], focal), Bot(names[1], opp)
    order = [fb, ob] if focal_first else [ob, fb]
    config = MatchConfig(mode="freezeout", hands=hands, starting_stack=10_000,
                         structure=structures.parse("standard", 10_000), seed=seed)
    seats = [BotSeat(b, seed=derive_seed(seed, "bot", b.name), use_docker=False) for b in order]
    t0 = time.time()
    try:
        res = play_match(seats, config, "spar", NullLog())
        row = next(r for r in res["results"] if r["bot"] == fb.name)
        orow = next(r for r in res["results"] if r["bot"] == ob.name)
        out = dict(won=row["final_stack"] > orow["final_stack"], hands=res["hands"], stopped=res["stopped"],
                   final=row["final_stack"], errors=row.get("errors", 0) + row.get("timeouts", 0),
                   opp_errors=orow.get("errors", 0) + orow.get("timeouts", 0))
    except Exception as e:                                     # a harness/engine failure, not a bot loss
        out = dict(won=None, hands=0, stopped="error: " + repr(e), final=0, errors=0, opp_errors=0)
    finally:
        for s in seats:
            try:
                s.close()
            except Exception:
                pass
    out.update(focal=focal, opp=opp, seed=seed, focal_first=focal_first, secs=round(time.time() - t0, 1))
    return out


def resolve(name):
    p = Path(name)
    if p.exists():
        return str(p)
    q = ROOT / "sparring" / "bots" / name
    if q.exists():
        return str(q)
    sys.exit(f"no such bot: {name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("focal", help="the bot under test, e.g. my_bots/tag")
    ap.add_argument("--opp", help="comma-separated bots (names from sparring/bots, or paths); default: all")
    ap.add_argument("--seeds", type=int, default=20, help="duplicate pairs per opponent (default 20)")
    ap.add_argument("--seed-base", type=int, default=1000, help="first seed (default 1000)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--hands", type=int, default=1000, help="safety cap per match (default 1000)")
    ap.add_argument("--json", help="write every match result to this file")
    args = ap.parse_args()

    focal = resolve(args.focal)
    if args.opp:
        opps = [resolve(o) for o in args.opp.split(",")]
    else:
        opps = sorted(str(p.parent) for p in (ROOT / "sparring" / "bots").glob("*/bot.py"))
    tasks = [(focal, o, args.seed_base + i, ff, args.hands)
             for o in opps for i in range(args.seeds) for ff in (True, False)]
    print(f"{Path(focal).name} vs {len(opps)} bots x {args.seeds} duplicate pairs = {len(tasks)} matches "
          f"({args.workers} workers)", flush=True)

    rows, t0 = [], time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        for k, r in enumerate(ex.map(play, tasks), 1):
            rows.append(r)
            if k % 20 == 0 or k == len(tasks):
                print(f"  {k}/{len(tasks)} matches, {time.time() - t0:.0f}s", flush=True)

    per = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        per[r["opp"]][r["seed"]][r["focal_first"]] = r
    print(f"\n{'opponent':<16}{'win':>5}{'draw':>6}{'loss':>6}  {'match%':>7}{'avg hands':>10}{'slowest':>9}  notes")
    bad = []
    for o in opps:
        w = d = l = mw = mn = hs = 0
        slowest, notes = 0.0, set()
        for seed, pair in sorted(per[o].items()):
            ms = [pair[True], pair[False]]
            if any(m["won"] is None for m in ms):
                notes.add("ENGINE ERROR " + next(m["stopped"] for m in ms if m["won"] is None)[:60])
                continue
            wins = sum(m["won"] for m in ms)
            w, d, l = w + (wins == 2), d + (wins == 1), l + (wins == 0)
            mw, mn = mw + wins, mn + 2
            hs += sum(m["hands"] for m in ms)
            slowest = max(slowest, max(m["secs"] for m in ms))
            if any(m["errors"] for m in ms):
                notes.add("focal errors/timeouts")
            if any(m["stopped"] == "hand_limit" for m in ms):
                notes.add("hit hand cap")
            if wins < 2:
                bad.append((o, seed, wins, [m["hands"] for m in ms]))
        name = Path(o).name
        print(f"{name:<16}{w:>5}{d:>6}{l:>6}  {100 * mw / max(mn, 1):>6.0f}%{hs / max(mn, 1):>10.0f}{slowest:>8.0f}s  "
              f"{'; '.join(sorted(notes))}")

    if bad:
        print("\nnot-won pairings (replay with: python3 play.py <focal> <opp> --seed S --show):")
        for o, seed, wins, hands in bad[:40]:
            print(f"  {'LOSS' if wins == 0 else 'draw'}  {Path(o).name:<16} seed {seed}  hands {hands}")
        if len(bad) > 40:
            print(f"  ... {len(bad) - 40} more (use --json)")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
