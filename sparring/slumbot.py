#!/usr/bin/env python3
"""Play a bot against Slumbot (slumbot.com), a strong heads-up NLHE equilibrium bot.

    python3 sparring/slumbot.py my_bots/tag --hands 1000 --workers 4
    python3 sparring/slumbot.py my_bots/tag --hands 2000 --log out.jsonl

Slumbot's format is NOT the tournament's: 50/100 blinds, 200bb (20,000) stacks
reset every hand, no rising blinds, no antes. So it tests deep-stack preflop and
postflop play, not push/fold. Results are in bb/100 (big blinds won per 100 hands).

Each worker is its own process: it imports the bot once and plays its hands as a
single "match" against Slumbot, so the bot's opponent model sees a continuous
match_action_log (with Slumbot's cards from showdowns) exactly like the engine's.
The bot itself never touches the network; only this harness does.

Runs on the host or in the sandbox image (needs host networking there, e.g.
docker run --network=host ...).
"""

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Lock, Value
from pathlib import Path

HOST = "https://slumbot.com/slumbot/api/"
SB, BB, STACK = 50, 100, 20_000
STREETS = ["preflop", "flop", "turn", "river"]
ME, SLUM = "hero", "slumbot"


def post(endpoint, data, tries=6):
    body = json.dumps(data).encode()
    for k in range(tries):
        try:
            req = urllib.request.Request(HOST + endpoint, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=20) as r:
                out = json.loads(r.read())
            if "error_msg" in out:
                raise RuntimeError(out["error_msg"])
            return out
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
            if k == tries - 1:
                raise
            time.sleep(1.5 * (k + 1))


def load_bot(path):
    p = Path(path).resolve()
    f = p / "bot.py" if p.is_dir() else p
    if (f.parent / "data").is_dir():
        os.environ["BOT_DATA_DIR"] = str(f.parent / "data")
    sys.path.insert(0, str(f.parent))
    spec = importlib.util.spec_from_file_location("spar_bot", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Hand:
    """Replays a Slumbot action string. pos 0 = big blind, pos 1 = small blind (button).

    `cap` emulates a shorter stack: the bot sees stacks, pot and bets as if each
    player had `cap` chips, a Slumbot bet past that is an all-in it can only
    call or fold, and capped_result() scores the hand at that depth. Slumbot
    itself still plays its 200bb strategy, so this approximates a short-stack
    opponent rather than being one.
    """

    def __init__(self, action, client_pos, hand_num, cap=STACK):
        self.me, self.hand_num, self.cap = client_pos, hand_num, cap
        self.ids = {client_pos: ME, 1 - client_pos: SLUM}
        self.seat = {client_pos: 0, 1 - client_pos: 1}
        self.committed = {0: BB, 1: SB}
        self.street_bet = {0: BB, 1: SB}
        self.street, self.to_act, self.last_size, self.n_on_street = 0, 1, BB - SB, 0
        self.over, self.folder = False, None
        self.log = [self._entry(1, "small_blind", SB), self._entry(0, "big_blind", BB)]
        self._replay(action)

    def _entry(self, pos, action, amount):
        return {"seat": self.seat[pos], "bot_id": self.ids[pos], "action": action,
                "amount": amount, "street": STREETS[min(self.street, 3)], "hand_num": self.hand_num}

    def capped(self, p):
        """Chips `p` has in the pot, counting at most the cap."""
        return min(self.committed[p], self.cap)

    def street_capped(self, p):
        """Chips `p` has in on this street, counting at most the cap overall."""
        return self.capped(p) - min(self.committed[p] - self.street_bet[p], self.cap)

    def all_in_matched(self):
        """Someone is all-in at the cap and the other has matched it: nothing left to decide."""
        return max(self.capped(0), self.capped(1)) >= self.cap and self.capped(0) == self.capped(1)

    def _log(self, p, action, amount):
        if self.capped(p) >= self.cap and action != "fold" or self.all_in_matched():
            return                                          # an all-in player doesn't act
        self.log.append(self._entry(p, action, amount))

    def _close_street(self):
        if self.street == 3:
            self.over = True
            return
        self.street += 1
        self.street_bet = {0: 0, 1: 0}
        self.to_act, self.last_size, self.n_on_street = 0, 0, 0

    def _replay(self, s):
        i = 0
        while i < len(s):
            c = s[i]
            i += 1
            if c == "/":
                continue                                   # street ends are inferred from the action
            p, o = self.to_act, 1 - self.to_act
            if c == "f":
                self._log(p, "fold", 0)
                self.over, self.folder = True, p
                return
            if c == "k":
                self._log(p, "check", 0)
                self.n_on_street += 1
                if self.n_on_street >= 2:
                    self._close_street()
                else:
                    self.to_act = o
            elif c == "c":
                add = self.street_bet[o] - self.street_bet[p]
                before = self.capped(p)
                self._log(p, "call", min(self.committed[p] + add, self.cap) - before)
                self.street_bet[p] += add
                self.committed[p] += add
                self.n_on_street += 1
                self.last_size = 0
                if self.committed[p] >= STACK:             # all-in called: run it out
                    self.over = True
                    return
                if self.street == 0 and self.n_on_street == 1:
                    self.to_act = o                       # small blind limped, big blind to act
                else:
                    self._close_street()
            elif c == "b":
                j = i
                while i < len(s) and s[i].isdigit():
                    i += 1
                to = int(s[j:i])
                add = to - self.street_bet[p]
                prev = min(self.committed[p] - self.street_bet[p], self.cap)
                kind = "all_in" if self.committed[p] + add >= self.cap else "raise"
                self._log(p, kind, min(to, self.cap - prev))
                self.last_size = to - self.street_bet[o]
                self.street_bet[p] = to
                self.committed[p] += add
                self.n_on_street += 1
                self.to_act = o
            else:
                raise ValueError(f"bad action char {c!r} in {s!r}")

    def needs_decision(self):
        """False when the bot has nothing to decide at the capped depth (it, or both, are all-in)."""
        return self.capped(self.me) < self.cap and not self.all_in_matched()

    def state(self, hole, board, match_log):
        me, op = self.me, 1 - self.me
        stack = self.cap - self.capped(me)
        bets = {p: self.street_capped(p) for p in (0, 1)}
        owed = max(0, bets[op] - bets[me])
        cur = max(bets.values())
        min_to = min(cur + max(self.last_size, BB), bets[me] + stack)
        sb_seat = self.seat[1]
        players = []
        for pos in (me, op):
            st = self.cap - self.capped(pos)
            players.append({"seat": self.seat[pos], "bot_id": self.ids[pos], "stack": st,
                            "state": "all_in" if st == 0 else "active", "is_folded": False,
                            "is_all_in": st == 0, "bet_this_street": bets[pos], "hole_cards": None})
        players.sort(key=lambda p: p["seat"])
        alog = [{k: v for k, v in e.items() if k != "hand_num"} for e in self.log]
        return {
            "your_cards": list(hole), "community_cards": list(board), "street": STREETS[self.street],
            "pot": self.capped(0) + self.capped(1), "your_stack": stack,
            "your_bet_this_street": bets[me], "amount_owed": owed, "can_check": owed == 0,
            "current_bet": cur, "min_raise_to": min_to, "your_bot_id": ME, "seat_to_act": self.seat[me],
            "button_seat": sb_seat, "sb_seat": sb_seat, "bb_seat": self.seat[0],
            "hand_num": self.hand_num, "hand_id": f"slum-{self.hand_num}",
            "blinds": {"sb": SB, "bb": BB, "ante": 0}, "players": players,
            "action_log": alog, "match_action_log": (match_log + self.log)[-200:], "time_limit_ms": 2000,
        }

    def passive(self):
        return "c" if self.street_bet[1 - self.me] > self.street_bet[self.me] else "k"

    def to_incr(self, reply):
        """The bot's reply -> Slumbot's incremental action, following the engine's rules."""
        me, op = self.me, 1 - self.me
        owed = self.street_bet[op] > self.street_bet[me]
        max_to = self.street_bet[me] + (self.cap - self.committed[me])     # our all-in at the cap
        a = (reply or {}).get("action") if isinstance(reply, dict) else None
        passive = self.passive()
        if a in ("check", "call"):
            return passive
        if a in ("raise", "all_in"):
            if self.cap - self.capped(op) == 0 or max_to <= self.street_bet[op]:
                return passive                             # nothing to raise into
            cur = max(self.street_bet.values())
            legal_min = cur + max(self.last_size, BB)      # Slumbot's minimum (it doesn't know the cap)
            if a == "all_in":
                amt = max_to
            else:
                try:
                    amt = int(reply.get("amount"))
                except (TypeError, ValueError):
                    return "f" if owed else "k"
                amt = min(max(amt, legal_min), max_to)
            if amt < legal_min:
                return passive                             # a capped all-in below a legal raise: call
            return f"b{amt}"
        return "f" if owed else "k"                       # fold, or anything unknown

    def capped_result(self, winnings):
        """Our net chips at the capped depth, given Slumbot's (200bb) result."""
        if self.cap >= STACK:
            return winnings
        if self.folder is not None:
            return -self.capped(self.me) if self.folder == self.me else self.capped(1 - self.me)
        matched = min(self.capped(0), self.capped(1))
        return matched if winnings > 0 else -matched if winnings < 0 else 0


# Running totals shared by the workers, for --every progress lines.
_PROGRESS = None


def _init_progress(count, total, lock, every, t0):
    global _PROGRESS
    _PROGRESS = (count, total, lock, every, t0)


def report_progress(won):
    if not _PROGRESS:
        return
    count, total, lock, every, t0 = _PROGRESS
    with lock:
        count.value += 1
        total.value += won
        if every and count.value % every == 0:
            n, bb = count.value, total.value / BB
            print(f"{n} hands: {bb / n * 100:+.1f} bb/100 (total {bb:+.0f} bb, {time.time() - t0:.0f}s)", flush=True)


def worker(task):
    bot_path, n_hands, wid, seed, verbose, cap = task
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % 2**32)
    except ImportError:
        pass
    bot = load_bot(bot_path)
    try:
        bot.decide({"type": "warmup"})
    except Exception:
        pass
    token, match_log, out = None, [], []
    for h in range(n_hands):
        try:
            r = post("new_hand", {"token": token} if token else {})
            token = r.get("token") or token
            decisions = []
            while "winnings" not in r:
                hand = Hand(r.get("action", ""), r["client_pos"], h, cap)
                if not hand.needs_decision():              # all-in at the capped depth: check/call it down
                    r = post("act", {"token": token, "incr": hand.passive()})
                    token = r.get("token") or token
                    continue
                st = hand.state(r["hole_cards"], r.get("board", []), match_log)
                t0 = time.time()
                try:
                    reply = bot.decide(st)
                except Exception as e:
                    reply = {"action": "fold", "error": repr(e)}
                dt = time.time() - t0
                incr = hand.to_incr(reply)
                decisions.append({"street": st["street"], "reply": reply if isinstance(reply, dict) else str(reply),
                                  "incr": incr, "ms": round(dt * 1000), "pot": st["pot"], "owed": st["amount_owed"]})
                r = post("act", {"token": token, "incr": incr})
                token = r.get("token") or token
            action = r.get("action", "")
            hand = Hand(action, r["client_pos"], h, cap)
            match_log.extend(hand.log)
            showdown = not action.endswith("f")
            if showdown and r.get("bot_hole_cards"):
                for pos, cards in ((hand.me, r["hole_cards"]), (1 - hand.me, r["bot_hole_cards"])):
                    match_log.append({"seat": hand.seat[pos], "bot_id": hand.ids[pos], "action": "show",
                                      "amount": 0, "cards": list(cards), "street": "showdown", "hand_num": h})
            match_log = match_log[-400:]
            rec = {"w": wid, "h": h, "pos": "SB" if r["client_pos"] == 1 else "BB", "won": hand.capped_result(r["winnings"]), "raw_won": r["winnings"],
                   "hole": r["hole_cards"], "opp": r.get("bot_hole_cards"), "board": r.get("board", []),
                   "action": action, "showdown": showdown, "end_street": STREETS[min(hand.street, 3)],
                   "decisions": decisions}
            out.append(rec)
            report_progress(rec["won"])
            if verbose:
                print(f"[w{wid} {h:4d}] {rec['pos']} {' '.join(rec['hole'])} | {' '.join(rec['board']):<14} "
                      f"| {action:<30} | {rec['won']:+6d}  vs {' '.join(rec['opp'] or [])}", flush=True)
        except Exception as e:
            print(f"[w{wid}] hand {h} aborted: {e!r}", file=sys.stderr, flush=True)
            token = None
            time.sleep(2)
    return out


def summarize(hands):
    n = len(hands)
    if not n:
        print("no hands played")
        return
    won = [x["won"] / BB for x in hands]
    mean = sum(won) / n
    sd = math.sqrt(sum((w - mean) ** 2 for w in won) / max(1, n - 1))
    print(f"\n{n} hands: {mean * 100:+.1f} bb/100  (±{1.96 * sd / math.sqrt(n) * 100:.1f} at 95%, "
          f"total {sum(won):+.0f} bb)")

    def table(title, keyf):
        g = defaultdict(list)
        for x in hands:
            g[keyf(x)].append(x["won"] / BB)
        print(f"\n{title:<22}{'hands':>7}{'bb/100':>10}{'total bb':>10}")
        for k in sorted(g, key=str):
            v = g[k]
            print(f"  {str(k):<20}{len(v):>7}{sum(v) / len(v) * 100:>+10.1f}{sum(v):>+10.0f}")

    table("position", lambda x: x["pos"])
    table("hand ended", lambda x: ("showdown " if x["showdown"] else "fold on ") + x["end_street"])

    def who_folded(x):
        if x["showdown"]:
            return "showdown"
        hand = Hand(x["action"], 1 if x["pos"] == "SB" else 0, 0)
        return ("hero" if hand.log[-1]["bot_id"] == ME else "slumbot") + " folds " + hand.log[-1]["street"]
    table("result", who_folded)

    pre = [x for x in hands]
    vpip = sum(1 for x in pre if any(d["street"] == "preflop" and d["incr"][0] in "cb" and
                                     not (d["incr"] == "c" and x["pos"] == "BB" and d["owed"] == 0)
                                     for d in x["decisions"]))
    pfr = sum(1 for x in pre if any(d["street"] == "preflop" and d["incr"][0] == "b" for d in x["decisions"]))
    ms = [d["ms"] for x in hands for d in x["decisions"]]
    print(f"\nhero VPIP {vpip / n:.0%}  PFR {pfr / n:.0%}  | decisions {len(ms)}, "
          f"avg {sum(ms) / max(1, len(ms)):.0f} ms, max {max(ms, default=0)} ms")
    errs = sum(1 for x in hands for d in x["decisions"] if isinstance(d["reply"], dict) and d["reply"].get("error"))
    if errs:
        print(f"bot errors: {errs}")
    leaks(hands)
    big = sorted(hands, key=lambda x: x["won"])[:5]
    print("\nbiggest losses:")
    for x in big:
        print(f"  {x['won'] / BB:+7.0f}bb {x['pos']} {' '.join(x['hole'])} vs {' '.join(x['opp'] or ['??'])} "
              f"board {' '.join(x['board'])}  {x['action']}")


def leaks(hands):
    """Hero's folds to bets (vs. minimum defence frequency) and betting when checked to, per street."""
    facing = defaultdict(lambda: [0, 0])            # (street, size bucket) -> [folds, chances]
    open_ = defaultdict(lambda: [0, 0])             # street -> [bets, chances]
    for x in hands:
        for d in x["decisions"]:
            if d["street"] == "preflop":
                continue
            if d["owed"]:
                frac = d["owed"] / max(1, d["pot"] - d["owed"])
                b = "<40%" if frac < 0.4 else "40-75%" if frac < 0.75 else "75-125%" if frac < 1.25 else ">125%"
                f = facing[(d["street"], b)]
                f[0] += d["incr"] == "f"
                f[1] += 1
            else:
                o = open_[d["street"]]
                o[0] += d["incr"][0] == "b"
                o[1] += 1
    mdf_fold = {"<40%": 0.22, "40-75%": 0.37, "75-125%": 0.5, ">125%": 0.6}   # max fold before bluffs auto-profit
    print(f"\nhero facing a bet      {'n':>6}{'fold':>8}{'max fold (MDF)':>16}")
    for st in STREETS[1:]:
        for b in mdf_fold:
            f = facing.get((st, b))
            if f and f[1]:
                flag = "  <-- overfolds" if f[0] / f[1] > mdf_fold[b] + 0.08 and f[1] >= 20 else ""
                print(f"  {st:<6} {b:<13}{f[1]:>6}{f[0] / f[1]:>8.0%}{mdf_fold[b]:>16.0%}{flag}")
    print(f"\nhero checked to        {'n':>6}{'bets':>8}")
    for st in STREETS[1:]:
        o = open_.get(st)
        if o and o[1]:
            print(f"  {st:<21}{o[1]:>6}{o[0] / o[1]:>8.0%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bot")
    ap.add_argument("--hands", type=int, default=500, help="total hands (split across workers)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--stack-bb", type=int, default=STACK // BB,
                    help="emulate this stack depth (default 200, Slumbot's own); scored at that depth")
    ap.add_argument("--log", help="write every hand as JSON lines")
    ap.add_argument("--every", type=int, default=100, help="print running bb/100 every N hands (0 = off)")
    ap.add_argument("--load", help="skip playing; summarize an earlier --log file")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every hand")
    a = ap.parse_args()

    if a.load:
        summarize([json.loads(l) for l in open(a.load)])
        return
    per = [a.hands // a.workers + (1 if i < a.hands % a.workers else 0) for i in range(a.workers)]
    cap = min(STACK, a.stack_bb * BB)
    tasks = [(a.bot, n, i, a.seed * 1000 + i, a.verbose, cap) for i, n in enumerate(per) if n]
    t0 = time.time()
    hands = []
    progress = (Value("i", 0), Value("d", 0.0), Lock(), a.every, t0)
    with ProcessPoolExecutor(len(tasks), initializer=_init_progress, initargs=progress) as ex:
        for res in ex.map(worker, tasks):
            hands.extend(res)
    print(f"played in {time.time() - t0:.0f}s")
    if a.log:
        with open(a.log, "w") as f:
            for x in hands:
                f.write(json.dumps(x) + "\n")
    summarize(hands)


if __name__ == "__main__":
    main()
