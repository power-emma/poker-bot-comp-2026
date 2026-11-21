#!/usr/bin/env python3
"""Score variants of a bot (module-level constants overridden) against a sparring suite.

    python3 sparring/tune.py my_bots/tag --variant base= --variant "sb12:SHORT_STACK_BB=12" \
        --variant "rr:RERAISE_EQUITY=0.55,OUT_OF_POSITION_MARGIN=0.05" --seeds 30

A variant is NAME:CONST=VALUE,CONST=VALUE (VALUE is Python source; an empty spec = unchanged).
Variants are copied to a scratch folder with the constants rewritten, then every variant plays
the same suite on the same seeds (duplicate pairs), so differences are paired, not noise from
different deals. Score = (pairs won + 0.5 * pairs drawn) / pairs.
"""
import argparse
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sparring"))
from run import play  # noqa: E402

SUITE = ["nash", "jam_fold", "overcaller", "shover", "trapper", "size_liar", "min_raiser", "three_bet",
         "button_spam", "bluff_bot", "hero_caller", "floater", "switcher", "seven_two", "limp_check",
         "polar_overbet"]


def make_variant(src_dir, spec, out_root):
    name, _, overrides = spec.partition(":")
    d = Path(out_root) / name
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(src_dir, d, ignore=shutil.ignore_patterns("__pycache__"))
    f = d / "bot.py"
    text = f.read_text()
    for item in filter(None, overrides.split(";") if ";" in overrides else overrides.split(",")):
        const, _, value = item.partition("=")
        pat = re.compile(rf"^{re.escape(const.strip())}( *)=[^\n#]*", re.M)
        if not pat.search(text):
            sys.exit(f"{name}: no module-level constant {const} in {src_dir}")
        text = pat.sub(lambda m: f"{const.strip()}{m.group(1)}= {value.strip()}  ", text, count=1)
    f.write_text(text)
    return name, str(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bot")
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--opp", help="comma list (default: built-in suite)")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--seed-base", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--hands", type=int, default=1000)
    args = ap.parse_args()

    scratch = tempfile.mkdtemp(prefix="tune_")
    variants = [make_variant(args.bot, v, scratch) for v in args.variant]
    opps = [str(ROOT / "sparring" / "bots" / o) if (ROOT / "sparring" / "bots" / o).exists() else o
            for o in (args.opp.split(",") if args.opp else SUITE)]
    tasks = [(path, o, args.seed_base + i, ff, args.hands)
             for _, path in variants for o in opps for i in range(args.seeds) for ff in (True, False)]
    print(f"{len(variants)} variants x {len(opps)} opps x {args.seeds} pairs = {len(tasks)} matches", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(play, tasks, chunksize=4))
    print(f"done in {time.time() - t0:.0f}s\n")

    res = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))   # variant -> opp -> seed -> ff -> won
    for r in rows:
        res[r["focal"]][r["opp"]][r["seed"]][r["focal_first"]] = r["won"]
    names = {path: name for name, path in variants}
    short = [Path(o).name for o in opps]
    print(f"{'variant':<14}{'score':>7}{'W':>5}{'D':>5}{'L':>5}{'match%':>8}   " + " ".join(f"{s[:7]:>7}" for s in short))
    for _, path in variants:
        W = D = L = 0
        cells = []
        for o in opps:
            w = d = l = 0
            for pair in res[path][o].values():
                n = sum(bool(x) for x in pair.values())
                w, d, l = w + (n == 2), d + (n == 1), l + (n == 0)
            W, D, L = W + w, D + d, L + l
            cells.append((w + 0.5 * d) / max(1, w + d + l))
        N = W + D + L
        print(f"{names[path]:<14}{(W + 0.5 * D) / N:>7.3f}{W:>5}{D:>5}{L:>5}{100 * (2 * W + D) / (2 * N):>7.0f}%   "
              + " ".join(f"{c:>7.2f}" for c in cells))
    shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
