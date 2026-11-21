"""
Heads-up Nash push/fold chart (strategy.md section 4).

The small blind (the button heads-up) either goes all-in or folds; the big
blind either calls or folds. For each effective stack depth this finds the
equilibrium ranges by fictitious play over the 169 starting hand classes,
using an all-in equity matrix that counts card removal (AsAh can only meet
one other pair of aces).

Prints a Python dict for bot.py: hand class -> (push up to, call up to,
push up to with antes, call up to with antes), in big blinds of effective
stack counted at the start of the hand. A hand pushes (calls) at every depth
up to its value; MAX_BB means "at every depth we solved".

With --data DIR it also writes DIR/preflop_equity.json: the class list, the
169x169 all-in equity matrix and the compatible-combo counts, for the bot to
compute exploitative push and call ranges during a match.

Run in the sandbox image (needs eval7 + numpy), all cores:
  docker run --rm --network=none -v "$PWD":/kit -w /kit \
    --entrypoint python3 fullhouse-sandbox:latest tools/pushfold_nash.py --data my_bots/tag/data
"""

import itertools
import json
import multiprocessing
import os
import sys

import eval7
import numpy as np

RANKS = "AKQJT98765432"
SUITS = "shdc"
TRIALS = 40000          # Monte Carlo deals per hand-class pair (C code, ~4 ms each)
ANTE = 0.1              # this tournament's antes are 10% of the big blind, from hand 101
MAX_BB = 30
GRID = [x / 2 for x in range(2, 2 * MAX_BB + 1)]    # 1, 1.5, ..., 30 bb
ITERATIONS = 3000


def classes():
    out = []
    for i, a in enumerate(RANKS):
        for j, b in enumerate(RANKS):
            if i == j:
                out.append(a + b)
            elif i < j:
                out.append(a + b + "s")
                out.append(a + b + "o")
    return out


def combos(cls):
    a, b = cls[0], cls[1]
    if len(cls) == 2:
        return [(a + s, a + t) for s, t in itertools.combinations(SUITS, 2)]
    if cls.endswith("s"):
        return [(a + s, b + s) for s in SUITS]
    return [(a + s, b + t) for s in SUITS for t in SUITS if s != t]


CLASSES = classes()


def row(i):
    """Equity of class i (one representative combo; suits are symmetric) against
    each class, and how many of that class's combos don't share a card with it."""
    hero = combos(CLASSES[i])[0]
    cards = [eval7.Card(c) for c in hero]
    eq, weight = [], []
    for cls in CLASSES:
        ok = [c for c in combos(cls) if not set(c) & set(hero)]
        weight.append(len(ok))
        if not ok:
            eq.append(0.5)
            continue
        villain = eval7.HandRange(",".join(a + b for a, b in ok))
        eq.append(eval7.py_hand_vs_range_monte_carlo(cards, villain, [], TRIALS))
    return eq, weight


def solve(E, W, stack, ante):
    """Fictitious play; returns (push frequency, call frequency) per class."""
    n = len(CLASSES)
    push, call = np.full(n, 0.5), np.full(n, 0.5)
    fold_sb, fold_bb, steal = -(0.5 + ante), -(1 + ante), 1 + ante
    called_net = E * 2 * stack - stack                  # [i, j]: i's net when i and j get it in
    for k in range(1, ITERATIONS + 1):
        # Small blind: EV of pushing each hand against the big blind's calling range.
        w = W * 1.0
        ev_push = (w * (call * called_net + (1 - call) * steal)).sum(1) / w.sum(1)
        best_push = (ev_push > fold_sb).astype(float)
        # Big blind: EV of calling with each hand against the pushing range.
        wp = W * push
        ev_call = (wp * called_net).sum(1) / np.maximum(wp.sum(1), 1e-9)
        best_call = (ev_call > fold_bb).astype(float)
        push += (best_push - push) / (k + 1)
        call += (best_call - call) / (k + 1)
    return push, call


def up_to(freqs):
    """Per class: the deepest grid stack such that the hand is in the range at
    that depth and every shallower one."""
    out = []
    for c in range(len(CLASSES)):
        deepest = 0
        for g, f in zip(GRID, freqs):
            if f[c] < 0.5:
                break
            deepest = g
        out.append(deepest)
    return out


def main():
    with multiprocessing.Pool() as pool:
        rows = pool.map(row, range(len(CLASSES)))
    E = np.array([r[0] for r in rows])
    W = np.array([r[1] for r in rows], dtype=float)
    E = (E + (1 - E.T)) / 2                              # average the two estimates of each pair
    print(f"# max |E + E.T - 1| before averaging: {np.abs(np.array([r[0] for r in rows]) + np.array([r[0] for r in rows]).T - 1).max():.4f}",
          file=sys.stderr)
    if "--data" in sys.argv:
        out = sys.argv[sys.argv.index("--data") + 1]
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "preflop_equity.json"), "w") as f:
            json.dump({"classes": CLASSES, "equity": np.round(E, 4).tolist(),
                       "weight": W.astype(int).tolist()}, f, separators=(",", ":"))

    charts = []
    for ante in (0.0, ANTE):
        pushes, calls = [], []
        for stack in GRID:
            p, c = solve(E, W, stack, ante)
            pushes.append(p)
            calls.append(c)
            n_push = sum(p[i] * len(combos(cls)) for i, cls in enumerate(CLASSES)) / 1326
            n_call = sum(c[i] * len(combos(cls)) for i, cls in enumerate(CLASSES)) / 1326
            print(f"# ante {ante} stack {stack:>4}: push {n_push:.0%}, call {n_call:.0%}", file=sys.stderr)
        charts.append((up_to(pushes), up_to(calls)))

    print("NASH_PUSH_FOLD = {")
    print("    # class: (push up to, call up to, push up to with antes, call up to with antes), in bb")
    line = "   "
    for i, cls in enumerate(CLASSES):
        item = f' "{cls}": ({charts[0][0][i]:g}, {charts[0][1][i]:g}, {charts[1][0][i]:g}, {charts[1][1][i]:g}),'
        if len(line) + len(item) > 100:
            print(line)
            line = "   "
        line += item
    print(line)
    print("}")


if __name__ == "__main__":
    main()
