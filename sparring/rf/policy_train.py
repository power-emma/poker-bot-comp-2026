#!/usr/bin/env python3
"""Train sparring/bots/rf_bot: a random forest that values actions, by policy iteration.

    python3 sparring/rf/policy_train.py --iters 3 --workers 10

Round 0 plays rf_bot's equity heuristic with random exploration; each later round plays the
forest from the previous round (still exploring a little). Every round's matches are against
my_bots/tag and the sparring presets. Each rf_bot decision becomes a row:
    features of the state + the action taken  ->  big blinds netted from that point to hand end
(folds are skipped: they are worth exactly 0). The forest is fit on the last two rounds and
exported to sparring/bots/rf_bot/data/rf_policy.json.
"""
import argparse
import json
import pickle
import random
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sparring"))
BOT = ROOT / "sparring" / "bots" / "rf_bot"
SKIP = {"folder"}


class Collect:
    def __init__(self):
        self.events = []

    def emit(self, t, **d):
        if t in ("hand_start", "decision", "hand_end"):
            self.events.append({"type": t, **d})
        return {}


def run_one(task):
    rf_path, opp_path, opp_name, seed, rf_first = task
    from pokercore import structures
    from pokercore.seats import BotSeat
    from pokercore.seeds import derive_seed
    from pokercore.table import MatchConfig, play_match
    from play import Bot
    rf, opp = Bot("rf", rf_path), Bot("opp", opp_path)
    order = [rf, opp] if rf_first else [opp, rf]
    cfg = MatchConfig(mode="freezeout", hands=400, starting_stack=10_000,
                      structure=structures.parse("standard", 10_000), seed=seed)
    seats = [BotSeat(b, seed=derive_seed(seed, "bot", b.name), use_docker=False) for b in order]
    log = Collect()
    try:
        res = play_match(seats, cfg, "rfp", log)
    finally:
        for s in seats:
            try:
                s.close()
            except Exception:
                pass
    won = next(r for r in res["results"] if r["bot"] == "rf")["final_stack"] > 10_000
    bb, pending, rows = {}, [], []
    for e in log.events:
        if e["type"] == "hand_start":
            bb[e["hand"]] = e["blinds"]["bb"]
            pending = []
        elif e["type"] == "decision" and e["bot"] == "rf":
            d = e.get("debug") or {}
            if "x" in d and e["applied"]["action"] != "fold":
                pending.append((d["x"], e["state"]["your_stack"]))
        elif e["type"] == "hand_end":
            end = e["stacks"]["rf"]
            for x, before in pending:
                rows.append((x, (end - before) / bb[e["hand"]], opp_name))
            pending = []
    return rows, opp_name, won


def make_bot(scratch, it, eps, model_json):
    d = scratch / f"rf_it{it}"
    (d / "data").mkdir(parents=True)
    shutil.copy(BOT / "bot.py", d / "bot.py")
    cfg = {"eps": eps, "model": model_json is not None}
    (d / "data" / "config.json").write_text(json.dumps(cfg))
    if model_json is not None:
        shutil.copy(model_json, d / "data" / "rf_policy.json")
    return d


def fit(rows, trees, leaf, depth):
    X = np.array([r[0] for r in rows], float)
    y = np.clip(np.array([r[1] for r in rows], float), -150, 150)
    m = RandomForestRegressor(n_estimators=trees, min_samples_leaf=leaf, max_depth=depth, max_features=0.6,
                              n_jobs=-1, random_state=0).fit(X, y)
    return m


def export(model, path, features):
    trees = []
    for est in model.estimators_:
        t = est.tree_
        trees.append({"l": t.children_left.tolist(), "r": t.children_right.tolist(), "f": t.feature.tolist(),
                      "t": [round(float(x), 5) for x in t.threshold],
                      "v": [round(float(x), 3) for x in t.value[:, 0, 0]]})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"features": list(features), "trees": trees}, separators=(",", ":")))
    print(f"  exported {len(trees)} trees, {path.stat().st_size / 1e6:.1f} MB -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--tag-seeds", type=int, default=300, help="seeds vs tag per round (x2 seat orders)")
    ap.add_argument("--preset-seeds", type=int, default=12, help="seeds per preset per round (x2)")
    ap.add_argument("--eps0", type=float, default=0.35)
    ap.add_argument("--eps", type=float, default=0.12)
    ap.add_argument("--trees", type=int, default=60)
    ap.add_argument("--leaf", type=int, default=40)
    ap.add_argument("--depth", type=int, default=14)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--rows-out")
    args = ap.parse_args()

    import gen
    spec = __import__("importlib.util").util.spec_from_file_location("rfbot", BOT / "bot.py")
    rfbot = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(rfbot)

    scratch = Path(tempfile.mkdtemp(prefix="rfpolicy_"))
    opps = [("tag", str(ROOT / "my_bots" / "tag"), args.tag_seeds)]
    opps += [(n, str(ROOT / "sparring" / "bots" / n), args.preset_seeds) for n in gen.PRESETS if n not in SKIP]
    history, model_json = [], None
    for it in range(args.iters):
        eps = args.eps0 if it == 0 else args.eps
        d = make_bot(scratch, it, eps, model_json)
        tasks = [(str(d), p, n, 50_000 + it * 10_000 + k, tf)
                 for n, p, ns in opps for k in range(ns) for tf in (True, False)]
        random.Random(it).shuffle(tasks)
        t0, rows, wins = time.time(), [], {}
        with ProcessPoolExecutor(args.workers) as ex:
            for i, (r, n, won) in enumerate(ex.map(run_one, tasks, chunksize=2), 1):
                rows.extend(r)
                w = wins.setdefault(n, [0, 0])
                w[0] += won
                w[1] += 1
                if i % 200 == 0:
                    print(f"  round {it}: {i}/{len(tasks)} matches, {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
        tw = wins["tag"]
        print(f"round {it} ({'heuristic' if model_json is None else 'forest'}, eps {eps}): "
              f"{len(rows)} rows; match wins vs tag {tw[0]}/{tw[1]}, vs presets "
              f"{sum(v[0] for k, v in wins.items() if k != 'tag')}/{sum(v[1] for k, v in wins.items() if k != 'tag')}",
              flush=True)
        history.append(rows)
        train = [r for rs in history[-2:] for r in rs]
        model = fit(train, args.trees, args.leaf, args.depth)
        model_json = scratch / f"rf_policy_{it}.json"
        export(model, model_json, rfbot.FEATURES)
        imp = sorted(zip(model.feature_importances_, rfbot.FEATURES), reverse=True)[:8]
        print("  top features:", ", ".join(f"{n} {v:.2f}" for v, n in imp), flush=True)
    if args.rows_out:
        pickle.dump(history, open(args.rows_out, "wb"))
    shutil.copy(model_json, BOT / "data" / "rf_policy.json")
    print(f"final forest -> {BOT / 'data' / 'rf_policy.json'}")


if __name__ == "__main__":
    main()
