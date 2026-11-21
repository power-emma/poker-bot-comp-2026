# Handoff: poker bot, state and next steps

Context for picking up this work in a new session. Read this, then
`strategy.md` (the design), then `my_bots/tag/bot.py` (the bot).

## The competition

- IEEE uOttawa × LARUS poker bot hackathon, Saturday November 21, coding
  until 10 PM. Teams of 1-2. Full rules are in `README.md` (`Rules.txt` is
  empty).
- **Format:** heads-up No-Limit Hold'em freezeout. 10,000 chips each, blinds
  start at 25/50 and rise every 20 hands, antes from hand 101.
- **Qualification:** a round robin of *duplicate* pairs (same cards, seats
  swapped, and you must win both to win the pairing). The top 4 go to a
  knockout final.
- **Interface:** the bot is `my_bots/<team>/bot.py` with `decide(game_state)`.
  - 2 seconds per decision. The sandbox has 0.5 CPU and 768 MB.
  - Allowed libraries: the Python 3.10 standard library, eval7, numpy, scipy,
    treys, scikit-learn.
  - No network, no file writes, and no `eval`/`exec`.
- **Prizes** include side prizes. The one we target is **most chips won with
  7-2**.
- **Submitting:** only `my_bots/<team_name>/` is collected. The folder **must
  match the team name** on the submission form, because `my_bots/` now holds
  several bots. The repo must be public and pushed to `main` by the deadline.
  The team name hasn't been decided yet. Rename `my_bots/tag` to it.
- **Don't edit `pokercore/`.**

## Files

| Path | What |
|---|---|
| `strategy.md` | The full strategy and design, with results and roadmap (section 10). Keep it updated. |
| `my_bots/tag/bot.py` | **The main bot.** Tight-aggressive preflop, range-aware equity, value betting by EV, 7-2 rules. |
| `my_bots/tag_v1/bot.py` | The same bot with fixed bet sizes instead of value betting. Used for comparison. |
| `my_bots/baseline/bot.py` | First bot: equity against a random hand plus pot odds, call or fold only. Sparring partner. |
| `my_bots/maniac/bot.py` | Sparring partner: raises the pot every time. |
| `bots/caller`, `bots/template` | Kit bots: a calling station and a simple template. |

Everything is pushed to `main` on `power-emma/poker-bot-comp-2026`. Commits
are named `v1`, `v2`, ... for now.

## How `my_bots/tag` works (short version, details in strategy.md)

- **Preflop:**
  - `PREFLOP_EQUITY` holds the equity of all 169 starting hands against a
    random hand (60,000 deals each). `TOP_PCT` turns that into a "top %"
    score.
  - Deep stacks use tight-aggressive charts: open the top 50% on the button,
    raise the top 20% from the big blind when the button just calls.
  - Facing a raise, the bot computes equity against this opponent's raising
    range and re-raises at 60% or more. It calls when the pot odds plus an
    8% margin allow, and otherwise uses a defend chart.
  - At 15bb or less it's push or fold, using rough formulas (`push_pct`,
    `call_shove_pct`), which are placeholders for a real Nash chart.
- **Opponent model:**
  - `update_stats` parses `match_action_log` for finished hands. Each hand is
    processed once (`_seen_hands`); the log only keeps 200 entries.
  - It tracks their preflop raise rate, how often they fold to our preflop
    raise, how often they bet postflop when checked to, and how they respond
    to bets (call or raise) per size bucket. To get bet sizes, it replays the
    pot from the log.
  - Every rate is shrunk toward a prior (`shrunk()`).
- **Range-aware equity:**
  - `opponent_range` turns this hand's actions into `(width, keep)`. Each
    action keeps the share of hands this opponent takes it with, so a
    maniac's bets barely narrow the range.
  - `RangeSample` ranks the opponent's possible hands by strength on this
    board (made hand; draws count as a middling pair) and runs one
    simulation. `sample.equity(share)` then gives our equity against the
    strongest `share` of the range, i.e. the hands that would call.
- **Postflop (`postflop`):** it's EV maximisation.
  - Options: check or call or fold, and bets of 0.33, 0.5, 0.75, 1 and 1.5
    pot, the geometric size, and all-in.
  - `bet_ev` uses `continue_probability(fraction)`, `raise_share()`, and our
    equity against the hands that would call.
  - **Risk caps** (`size_allowed`): bluffs at most 1x the pot. Other bets at
    most 1.5 pot or the geometric size. Overbets only against a confirmed
    station (at least 70% calls of large bets, over at least 5).
  - Debug output includes `ev`, `expected_take` and `call_chance`.
- **7-2 (strategy.md 7.1):**
  - 7-2 suited is played a little better than its rank: raised from the
    button, defended against raises of 3bb or less.
  - Its flush draws are played like nut flush draws: semi-bluff, and call to
    get all-in on the flop.
  - 7-2 offsuit folds unless the opponent folds to more than 60% of our
    preflop raises (with at least 8 observed).
- **Safety:** `decide` is wrapped in try/except, falling back to check, or
  fold if checking isn't allowed. Simulation counts are fixed (not
  time-based) so seeded matches replay exactly.

## Running and testing

**Environment quirks on this machine:**

- System Python is 3.14 and eval7 isn't installed on the host. There's no
  `uv`, so there's no `.venv`.
- `make sandbox` fails: Docker's bridge network errors with "operation not
  supported", likely needing a reboot after a kernel update.
- The image was built with:
  `docker build --network=host -t fullhouse-sandbox:latest pokercore/sandbox`
- **Run everything inside that image:**

```bash
# one match, tournament format, at half a core like the real sandbox
docker run --rm --network=none -v "$PWD":/kit:ro -w /kit --cpus=0.5 \
  --entrypoint python3 fullhouse-sandbox:latest play.py my_bots/tag my_bots/baseline --seed 7

# add --show to watch hands, or --log /out/m.jsonl with -v <dir>:/out to get the
# full log (decision events include state, applied action and the bot's debug)

# validator (must pass before submitting)
docker run --rm --network=none -v "$PWD":/kit:ro -w /kit \
  --entrypoint python3 fullhouse-sandbox:latest pokercore/validator.py my_bots/tag
```

**Batch testing (ad hoc; there's no harness yet).** For each seed, run both
seat orders, in parallel with `xargs -P 8`, and parse the `1. <winner>` line
and the `avg think` lines. Use a scratch directory for results. The bots are
read live from disk, so **don't edit a bot while a batch is running**.

## Latest results (12 seeds × both seat orders)

| Opponent | `tag` | `tag_v1` |
|---|---|---|
| caller | 24/24 | 24/24 |
| template | 24/24 | 24/24 |
| baseline | 20/24 | 23/24 |
| maniac | 20/24 | 16/24 |

- Head to head, `tag` beat `tag_v1` in 28 of 40.
- The slowest match averaged 94ms per decision. No errors.
- The maniac losses that were checked were all coolers (for example AK all-in
  against J4).

## Next steps, in priority order

1. **Rename `my_bots/tag`** to the team name once it's decided (not yet;
   keep versioned commits `v2`, `v3`, ... until then).
2. **Duplicate-match harness** in `tools/`, the teammate's task.
   - Hundreds of seeds, both seat orders, in parallel, in the sandbox image.
   - Report pairing wins, chips per hand, and 7-2 hands with chips won.
   - Needed because 24 matches can't separate a better bot from a lucky one.
3. ~~Find out why `tag` loses more to `baseline`~~ Done in v3: it was mostly
   noise, plus over-bluffed later barrels (strategy.md, "Later bets get called
   more"). Ad hoc scripts used for it: `batch.sh` (seeds in both seat orders,
   with logs), `analyze.py` (bb won by hand type, worst hands dumped) and
   `barrels.py` (opponent continue rates, first against later bets). They are
   worth folding into the `tools/` harness.
4. **Check predicted take against real results** (strategy.md 3.7). Log
   `expected_take` against actual chips won over many matches and tune
   `prior_continue`, `SIZE_PRIOR_WEIGHT`, `RAISE_MULTIPLE` and
   `STACK_OFF_EQUITY`.
5. **A real heads-up Nash push/fold chart** as a dict, replacing `push_pct` and
   `call_shove_pct`. That matters because late in the match everything is
   push or fold.
6. **More sparring bots:** a nit, an always-all-in bot, a "check when owing" bot
   (the engine plays that as a call), a min-raiser, a random bot, and a
   fixed-size bettor.
7. **Use showdown cards** (`show` entries in `match_action_log`) to calibrate
   the range assumptions. Showdowns show what they actually bet or called
   with.
8. **Side prizes** (strategy.md 7): track 7-2 results in testing, and consider
   thin river value bets against stations for "lowest winning showdown".

## Gotchas

- `raise.amount` is the **total** for the street. Below the minimum it's
  raised to the minimum; above our stack it becomes all-in.
- `check` when chips are owed is played as a call. Always look at `can_check`.
- Log `amount` means different things by action. For `raise` and `all_in`
  it's the street total. For `call` and the blind posts it's chips added.
  Antes are dead money and don't count toward the street bet.
- Use the global `random` module, which is seeded per match. Never seed from
  the clock.
- Corrections already made in strategy.md:
  - 3-2 offsuit is the worst hand against a random hand, not 7-2 offsuit.
  - A 7-2 suited flush draw on the flop has about 47% equity against a random
    hand, which is above the bluff prize's 40% cutoff, so those semi-bluffs
    don't count as bluffs.
