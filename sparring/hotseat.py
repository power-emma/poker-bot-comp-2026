#!/usr/bin/env python3
"""Play a bot yourself, one decision per command (built for an LLM or a human at a shell).

    python3 sparring/hotseat.py serve my_bots/tag --dir /tmp/hs --hands 40 &   # starts the match
    python3 sparring/hotseat.py view --dir /tmp/hs                             # current decision
    python3 sparring/hotseat.py act  --dir /tmp/hs raise 300                   # fold|check|call|raise N|all_in
                                                                                # (prints the next decision)
The match is cash mode with flat blinds (every hand starts at the full stack), the real engine and the
real bot sandbox. Every event, including the bot's own debug output, goes to DIR/events.jsonl.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

NAME = "claude"


def write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def render(st, results):
    me = next(p for p in st["players"] if p["bot_id"] == NAME)
    opp = next(p for p in st["players"] if p["bot_id"] != NAME)
    bb = st["blinds"]["bb"]
    lines = []
    if results:
        lines += results + [""]
    pos = "BUTTON/SB (acts first preflop, last postflop)" if st["seat_to_act"] == st["button_seat"] else "BIG BLIND (acts last preflop, first postflop)"
    lines.append(f"HAND {st['hand_num'] + 1} | {st['street'].upper()} | you are {pos}")
    lines.append(f"your cards: {' '.join(st['your_cards'])}    board: {' '.join(st['community_cards']) or '-'}")
    lines.append(f"pot {st['pot']} | you have {st['your_stack']} (bet {st['your_bet_this_street']}) | tag has {opp['stack']} (bet {opp['bet_this_street']})")
    if st["can_check"]:
        lines.append(f"you may: check | raise (total this street, min {st['min_raise_to']}) | all_in")
    else:
        lines.append(f"to call: {st['amount_owed']} (pot odds {st['amount_owed'] / (st['pot'] + st['amount_owed']):.0%}) | fold | call | raise (min {st['min_raise_to']}) | all_in")
    acts = [f"{x['bot_id']}:{x['action']}{(' ' + str(x['amount'])) if x.get('amount') else ''}@{x['street'][:2]}"
            for x in st["action_log"] if x["action"] not in ("ante",)]
    lines.append("log: " + "  ".join(acts))
    return "\n".join(lines)


def serve(args):
    from pokercore import structures
    from pokercore.seats import BotSeat, Seat
    from pokercore.seeds import derive_seed
    from pokercore.table import MatchConfig, play_match
    from play import Bot

    d = Path(args.dir)
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*"):
        f.unlink()
    events = open(d / "events.jsonl", "w")
    state = {"seq": 0, "results": []}

    class Log:
        def emit(self, t, **data):
            ev = {"type": t, **data}
            events.write(json.dumps(ev, default=str) + "\n")
            events.flush()
            if t == "showdown":
                state["results"].append("showdown: " + ", ".join(f"{n} {' '.join(c)}" for n, c in (data.get("reveal") or {}).items()))
            elif t == "award":
                state["results"].append(f"{data['bot']} wins {data['amount']}")
            elif t == "hand_start":
                state["results"].append(None)  # marker, filtered below
            return ev

    class HumanSeat(Seat):
        kind = "human"

        def request_action(self, st):
            res = [r for r in state["results"] if r]
            state["results"] = []
            state["seq"] += 1
            write_json(d / "view.json", {"seq": state["seq"], "text": render(st, ["-- last hand: " + "; ".join(res)] if res else []), "done": False})
            while True:
                mv = d / "move.json"
                if mv.exists():
                    m = json.loads(mv.read_text())
                    mv.unlink()
                    if m.get("seq") == state["seq"]:
                        return {"action": m["action"], **({"amount": m["amount"]} if "amount" in m else {}), "think_ms": 0}
                time.sleep(0.05)

    seed = args.seed or int(time.time()) % 100000
    config = MatchConfig(mode="cash", hands=args.hands, starting_stack=10_000,
                         structure=structures.parse("50/100", 10_000), seed=seed)
    tag = Bot("tag", args.bot)
    seats = [BotSeat(tag, seed=derive_seed(seed, "bot", "tag"), use_docker=False), HumanSeat(NAME)]
    if args.human_first:
        seats.reverse()
    result = play_match(seats, config, "hotseat", Log())
    res = [r for r in state["results"] if r]
    final = "\n".join(f"{r['bot']}: {r['final_stack']:,} ({r['delta']:+,}, {r.get('bb_per_100')} bb/100)" for r in result["results"])
    write_json(d / "view.json", {"seq": state["seq"] + 1, "done": True,
                                 "text": ("-- last hand: " + "; ".join(res) + "\n" if res else "") + f"MATCH OVER after {result['hands']} hands\n{final}"})


def wait_view(d, after, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = d / "view.json"
        if v.exists():
            try:
                j = json.loads(v.read_text())
            except ValueError:
                j = None
            if j and j["seq"] > after:
                return j
        time.sleep(0.05)
    sys.exit("timed out waiting for the bot")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("bot")
    s.add_argument("--dir", required=True)
    s.add_argument("--hands", type=int, default=40)
    s.add_argument("--seed", type=int)
    s.add_argument("--human-first", action="store_true")
    v = sub.add_parser("view")
    v.add_argument("--dir", required=True)
    a = sub.add_parser("act")
    a.add_argument("--dir", required=True)
    a.add_argument("move", nargs="+")
    args = ap.parse_args()
    if args.cmd == "serve":
        return serve(args)
    d = Path(args.dir)
    cur = json.loads((d / "view.json").read_text()) if (d / "view.json").exists() else {"seq": 0}
    if args.cmd == "view":
        j = wait_view(d, 0)
        print(j["text"])
        return
    mv = {"seq": cur["seq"], "action": args.move[0]}
    if len(args.move) > 1:
        mv["amount"] = int(args.move[1])
    write_json(d / "move.json", mv)
    j = wait_view(d, cur["seq"])
    print(j["text"])


if __name__ == "__main__":
    main()
