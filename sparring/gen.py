#!/usr/bin/env python3
"""Write the preset sparring bots (sparring/bots/<name>/bot.py) from puppet_core.py.

    python3 sparring/gen.py            # regenerate every preset
    python3 sparring/gen.py --out DIR  # ...somewhere else

Each preset is the puppet core with a PARAMS dict injected, so it is a single
self-contained bot.py that passes `make validate` and runs in the sandbox.
"""
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent

NIT = dict(vpip=0.08, pfr=0.08, threebet=0.04, call_raise=0.08, cbet=0.5, bluff=0.0, call_min=0.6)
MANIAC = dict(vpip=1.0, pfr=1.0, threebet=0.6, call_raise=0.9, cbet=1.0, bluff=0.8,
              value_min=0.0, bluff_raise=0.4, call_min=0.1, size_sens=0.0, bet_frac=(0.75, 1.0, 1.5))
STATION = dict(station=True)
BLUFFER = dict(pfr=0.45, vpip=0.8, pf_bluff=0.3, cbet=0.9, bluff=0.6, bluff_raise=0.2, call_min=0.45)

PRESETS = {
    # -- extremes / degenerate ------------------------------------------------
    "folder": dict(always_fold=True),
    "shover": dict(always_shove=True),
    "nit": NIT,
    "min_raiser": dict(min_raise_only=True, pfr=0.5, vpip=0.7, threebet=0.1, bluff=0.3, cbet=0.8),
    "limp_check": dict(limp=True, vpip=0.7, pfr=0.05, threebet=0.04, call_raise=0.35, cbet=0.0, bluff=0.0,
                       call_min=0.45, raise_min=0.9),
    # -- preflop pressure -----------------------------------------------------
    "three_bet": dict(pfr=0.5, vpip=0.6, threebet=0.45, call_raise=0.5, threebet_mult=3.5),
    "button_spam": dict(pfr=1.0, vpip=1.0, open_bb=2.2, threebet=0.04, call_raise=0.0, cbet=0.7, bluff=0.4),
    "seven_two": dict(pfr=0.3, vpip=0.4, raise_classes=("72o", "72s"), threebet_mult=3.0, cbet=0.8,
                      bluff=0.5, bluff_raise=0.2),
    # -- attacks on tag's opponent model --------------------------------------
    "size_liar": dict(size_tell="inverted", pfr=0.4, cbet=0.8, bluff=0.5, value_min=0.5),
    "polar_overbet": dict(size_tell="polar", pfr=0.45, cbet=0.5, bluff=0.4, value_min=0.65, call_min=0.5),
    "trapper": dict(pfr=0.35, vpip=0.5, trap=0.9, cbet=0.3, bluff=0.05, raise_min=0.8, raise_mult=3.5),
    "floater": dict(pfr=0.35, vpip=0.55, float=0.6, cbet=0.3, bluff=0.35, call_min=0.45),
    "hero_caller": dict(pfr=0.3, vpip=0.5, call_min=0.28, size_sens=0.0, cbet=0.4, bluff=0.05),
    "bluff_bot": BLUFFER,
    # -- short-stack phase ----------------------------------------------------
    "overcaller": dict(shove_bb=25, shove_pct=0.25, call_shove_pct=0.60, pfr=0.3, vpip=0.45),
    "jam_fold": dict(shove_bb=18, shove_pct=0.5, call_shove_pct=0.22, pfr=0.45, vpip=0.55),
    "nash": dict(nash=True, pfr=0.65, vpip=0.65, threebet=0.15, call_raise=0.35, cbet=0.55, bluff=0.2,
                 raise_min=0.8),
    # -- attacks written against tag's design (see report): ----------------------
    # re-shove over every raise, at any depth
    "reshove": dict(pfr=0.5, vpip=0.5, threebet=1.0, threebet_mult=100, call_raise=0.0, cbet=0.6, bluff=0.2),
    # limp, then jam over any raise
    "limp_shove": dict(limp=True, vpip=0.95, pfr=0.0, threebet=0.35, threebet_mult=100, call_raise=0.0,
                       cbet=0.0, bluff=0.0, call_min=0.5),
    # bets 25% pot on every street, folds to any resistance
    "tiny_bettor": dict(pfr=0.45, vpip=0.6, cbet=1.0, bluff=1.0, value_min=0.0, bet_frac=(0.25,),
                        call_min=0.6, size_sens=0.0, raise_min=0.95, call_raise=0.4),
    # raises every bet it faces (all-in over it) whatever it holds
    "jam_facing": dict(pfr=0.45, vpip=0.6, cbet=0.5, bluff=0.2, raise_min=0.0, trap=1.0, raise_mult=100),
    # raises ~half the bets it faces with air, to 3x
    "raise_bluffer": dict(pfr=0.4, vpip=0.6, cbet=0.6, bluff=0.2, bluff_raise=0.5, raise_mult=3.0, call_min=0.35),
    # bets every street at a fixed size whatever it holds (tag bets turn/river rarely and folds to them)
    "barrel_66": dict(pfr=0.5, vpip=0.6, call_raise=0.45, cbet=1.0, bluff=1.0, value_min=0.0, bet_frac=(0.66,),
                      call_min=0.35, raise_min=0.9),
    "barrel_100": dict(pfr=0.5, vpip=0.6, call_raise=0.45, cbet=1.0, bluff=1.0, value_min=0.0, bet_frac=(1.0,),
                       call_min=0.35, raise_min=0.9),
    # reads tag's bet-size tells: raises small bets, folds to big ones unless strong
    "size_reader": dict(pfr=0.45, vpip=0.7, call_raise=0.5, cbet=0.6, bluff=0.2, call_min=0.3, size_sens=0.0,
                        small_raise=0.6, big_fold_above=0.85, raise_min=0.85, raise_mult=3.0),
    # everything at once: nit for 80 hands (poison the stats), then big barrels + bet-size reading
    "combo_80": ([NIT, dict(pfr=0.5, vpip=0.7, call_raise=0.5, cbet=1.0, bluff=1.0, value_min=0.0, bet_frac=(1.0,),
                            call_min=0.3, size_sens=0.0, small_raise=0.6, big_fold_above=0.85, raise_min=0.85)], 80),
    # tight for 40 / 80 hands (builds a nit profile), then bluff-heavy for as long
    "poison_40": ([NIT, BLUFFER], 40),
    "poison_80": ([NIT, MANIAC], 80),
    # -- non-stationary -------------------------------------------------------
    "switcher": [NIT, MANIAC, STATION, BLUFFER],
}

# Protocol-edge bots (fuzz_core.py): they mostly check that tag survives odd inputs.
FUZZ = {"fuzz_random": "random", "fuzz_badsize": "badsize", "fuzz_wrongcheck": "wrongcheck",
        "fuzz_crasher": "crasher"}


def render(name, params, period=25):
    src = (HERE / "puppet_core.py").read_text()
    assert "PARAMS = {}\n" in src
    src = src.replace("PARAMS = {}\n", f"PARAMS = {params!r}\n", 1).replace("PERIOD = 25\n", f"PERIOD = {period}\n", 1)
    return f'# GENERATED by sparring/gen.py: preset "{name}". Edit puppet_core.py or gen.py, not this file.\n' + src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "bots"))
    out = Path(ap.parse_args().out)
    fuzz = (HERE / "fuzz_core.py").read_text()
    for name, mode in FUZZ.items():
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "bot.py").write_text(f'# GENERATED by sparring/gen.py: fuzz mode "{mode}".\n'
                                  + fuzz.replace('MODE = "random"\n', f'MODE = "{mode}"\n', 1))
    for name, params in PRESETS.items():
        period = 25
        if isinstance(params, tuple):
            params, period = params
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "bot.py").write_text(render(name, params, period))
    print(f"wrote {len(PRESETS) + len(FUZZ)} bots to {out}")


if __name__ == "__main__":
    main()
