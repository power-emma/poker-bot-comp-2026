#!/usr/bin/env python3
"""Collect (features -> opponent's answer to a postflop bet) rows from tag playing many opponents.

    python3 sparring/rf/collect.py --out /tmp/rf/rows.pkl

Opponents: N random puppets (random PARAMS) + the sparring presets. Presets in HOLDOUT go to the
test split (never used for training), so the forest is scored on opponent *types* it hasn't seen.
Label: 0 fold, 1 call, 2 raise. Stats features are tag's own STATS, replayed with tag's
update_stats exactly as it would have seen them at that point of the match.
"""
import argparse
import importlib.util
import pickle
import random
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sparring"))
sys.path.insert(0, str(ROOT / "sparring" / "rf"))
from rf_features import rf_features  # noqa: E402

HOLDOUT = {"trapper", "size_liar", "switcher", "nash", "hero_caller", "polar_overbet", "floater"}
SKIP = {"folder", "shover"}   # no postflop play


def random_params(rng):
    r = rng.random
    return dict(
        vpip=0.15 + 0.85 * r(), pfr=0.05 + 0.6 * r(), threebet=0.02 + 0.5 * r(), call_raise=0.05 + 0.6 * r(),
        open_bb=rng.choice((2.0, 2.5, 3.0, 4.0)), limp=r() < 0.3, pf_bluff=0.3 * r() * (r() < 0.5),
        cbet=r(), bluff=0.7 * r(), value_min=0.45 + 0.3 * r(),
        bet_frac=rng.choice(((0.33,), (0.5, 0.75), (0.75, 1.0), (0.5, 1.0, 1.5), (1.5,))),
        size_tell=rng.choice(("none", "none", "none", "inverted", "polar")), trap=rng.choice((0, 0, 0.5, 0.9)),
        call_min=0.2 + 0.5 * r(), size_sens=0.6 * r(), raise_min=0.7 + 0.2 * r(), raise_mult=rng.choice((2.5, 3.0, 4.0)),
        bluff_raise=0.3 * r() * (r() < 0.5), float=0.6 * r() * (r() < 0.5),
        min_raise_only=r() < 0.07)


def load_tag(path):
    spec = importlib.util.spec_from_file_location("tagmod_%d" % random.randrange(10 ** 9), path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Collect:
    def __init__(self):
        self.events = []

    def emit(self, t, **d):
        if t in ("hand_start", "decision"):
            self.events.append({"type": t, **d})
        return {}


def run_one(task):
    tag_path, opp_path, opp_name, split, seed, tag_first = task
    from pokercore import structures
    from pokercore.seats import BotSeat
    from pokercore.seeds import derive_seed
    from pokercore.table import MatchConfig, play_match
    from play import Bot
    tagbot, opp = Bot("tag", tag_path), Bot("opp", opp_path)
    order = [tagbot, opp] if tag_first else [opp, tagbot]
    cfg = MatchConfig(mode="freezeout", hands=400, starting_stack=10_000,
                      structure=structures.parse("standard", 10_000), seed=seed)
    seats = [BotSeat(b, seed=derive_seed(seed, "bot", b.name), use_docker=False) for b in order]
    log = Collect()
    try:
        play_match(seats, cfg, "rf", log)
    finally:
        for s in seats:
            try:
                s.close()
            except Exception:
                pass
    mod = load_tag(str(Path(tag_path) / "bot.py"))
    by_hand = {}
    for e in log.events:
        if e["type"] == "decision":
            by_hand.setdefault(e["hand"], []).append(e)
    rows, entries = [], []
    for h in sorted(by_hand):
        evs = by_hand[h]
        # tag's view of the match so far: completed hands only
        mod.update_stats({"your_bot_id": "tag", "hand_num": h, "match_action_log": entries[-200:]})
        stats = {k: (list(v) if isinstance(v, list) else v) for k, v in mod.STATS.items()}
        for e in evs:
            st = e["state"]
            if e["bot"] != "opp" or st["street"] == "preflop" or st["amount_owed"] <= 0 or e.get("error"):
                continue
            owed, pot = st["amount_owed"], st["pot"]
            before = pot - owed
            if before <= 0:
                continue
            frac = owed / before
            a = e["applied"]["action"]
            label = 0 if a == "fold" else 2 if a in ("raise",) or (a == "all_in" and e["applied"]["amount"] > st["current_bet"]) else 1
            behind = max(0, st["your_stack"] - owed)
            x = rf_features(st["street"], frac, before, st["blinds"]["bb"], behind, st["community_cards"],
                            stats, mod.size_bucket(frac))
            rows.append((x, label, opp_name, split, h, mod.continue_probability(frac), mod.raise_share()))
        last = evs[-1]
        done = [dict(x, hand_num=h) for x in last["state"]["action_log"]]
        done.append({"hand_num": h, "seat": last["seat"], "bot_id": last["bot"], "street": last["street"],
                     "action": last["applied"]["action"], "amount": last["applied"]["amount"]})
        entries.extend(done)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=str(ROOT / "my_bots" / "tag"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--random-bots", type=int, default=80)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--preset-seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    import gen
    scratch = Path(tempfile.mkdtemp(prefix="rfbots_"))
    tasks, rng = [], random.Random(7)
    for i in range(args.random_bots):
        d = scratch / f"rand{i}"
        d.mkdir()
        (d / "bot.py").write_text(gen.render(f"rand{i}", random_params(rng)))
        for k in range(args.seeds):
            for tf in (True, False):
                tasks.append((args.tag, str(d), f"rand{i}", "train", 20_000 + i * 100 + k, tf))
    for name in gen.PRESETS:
        if name in SKIP:
            continue
        d = ROOT / "sparring" / "bots" / name
        for k in range(args.preset_seeds):
            for tf in (True, False):
                tasks.append((args.tag, str(d), name, "test" if name in HOLDOUT else "train", 30_000 + k, tf))
    print(f"{len(tasks)} matches", flush=True)
    t0, rows = time.time(), []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(run_one, tasks, chunksize=2), 1):
            rows.extend(r)
            if i % 100 == 0:
                print(f"  {i}/{len(tasks)}  rows {len(rows)}  {time.time() - t0:.0f}s", flush=True)
    pickle.dump(rows, open(args.out, "wb"))
    print(f"{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
