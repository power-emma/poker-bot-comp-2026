# Sparring bots

Opponents for stress-testing `my_bots/tag`, plus a duplicate round-robin harness.

```bash
python3 sparring/gen.py                          # regenerate sparring/bots/* from puppet_core.py / fuzz_core.py
python3 sparring/run.py my_bots/tag              # every sparring bot, 20 duplicate pairs each
python3 sparring/run.py my_bots/tag --opp nit,trapper --seeds 50 --workers 8 --json out.json
python3 play.py my_bots/tag sparring/bots/nit --seed 1008 --show   # replay a loss
```

A pairing is two freezeout matches with the same seed and seats swapped: WIN = both, DRAW = one each, LOSS = neither.

- `puppet_core.py`: parametric bot (VPIP, PFR, 3-bet, c-bet, bet sizing tells, trapping, floating, push/fold, ...).
  `PARAMS` can be a list of profiles cycled every `PERIOD` hands (the `switcher`).
- `gen.py`: the preset table (`PRESETS`) and the fuzz modes. Add a bot by adding a line.
- `fuzz_core.py`: protocol-edge bots (random actions, bad raise sizes, wrong check/call, crashes and garbage replies).
- `run.py`: the harness.

`sparring/bots/` is generated but committed, so `play.py` can use the bots directly.
Needs eval7 (Python 3.10 per the kit; it also ran on 3.9 in a venv here, but `make validate` needs 3.10).
