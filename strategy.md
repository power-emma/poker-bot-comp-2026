# Bot Strategy

Tight-aggressive (TAG) poker with **value betting as the secret sauce**: we don't
just try to win the hand, we estimate how many chips the opponent will pay us
and choose the bet size that takes the most.

## 1. The format, and what it rewards

- **Heads-up freezeout.** 10,000 chips each. Blinds rise every 20 hands
  (25/50 at the start, 150/300 by hand 81, antes from hand 101).
- **Two phases per match:**
  - *Deep (hands 1–60, roughly 200bb down to 65bb).* Real postflop poker. This
    is where value betting pays most, because stacks are big enough to
    take several streets of bets.
  - *Short (about 15bb or less).* Push or fold preflop, from a chart.
- **Duplicate pairs.** The same cards are played twice with the seats swapped,
  and only winning both matches wins the pairing. Card luck mostly cancels
  out, so skill and exploiting the opponent decide results.
- **The field is other hackathon bots.** Expect calling stations, maniacs,
  bots that fold too much, and buggy bots. They make mistakes we can bet into.

## 2. Style: tight-aggressive

- **Tight preflop.** Play a ranked list of starting hands and fold the rest.
  Widen the list on the button and as stacks get shorter, and against
  opponents who fold too much.
- **Aggressive.** When we play a hand we raise or bet rather than call. Being
  the one who bets gives us two ways to win: the opponent folds, or we win at
  showdown.
- **Hand ranking.** Rank the 169 starting hands (pairs, suited, offsuit) by
  equity against a random hand. Compute the table once at warmup (30s limit)
  or ship it in `data/`.

## 3. The secret sauce: value betting by expected take

For every bet or raise decision, we compute how many chips we expect to win
for each candidate size and pick the largest.

### 3.1 The calculation

For a bet of size `b` into a pot of `P`:

```
EV(b) = P(fold | b) * P
      + P(call | b) * [ E(b) * (P + 2b) - b ]
```

where:

- `P(fold | b)` and `P(call | b)` are how likely the opponent is to fold or
  call this size (raises handled in 3.4),
- `E(b)` is our equity **against the hands they would call with**, not
  against a random hand. Bigger bets get called by stronger hands, so
  `E(b)` usually falls as `b` grows.

The **value part**, the chips we expect to extract, is the second line:
`P(call | b) * [E(b) * (P + 2b) - b]`. We log it with every decision as
`expected_take` so we can check it against what actually happened.

Candidate sizes: check, 33%, 50%, 75%, 100% and 150% of the pot, and all-in.
Also always compare against checking (EV of checking = our equity in the pot,
plus whatever we expect to win when they bet into us).

### 3.2 How likely they are to call, by size

Model `P(call | b)` as a function of the bet as a fraction of the pot,
`f = b / P`:

- **Prior (before we know them):** the pot-odds-rational fold rate, roughly
  `P(call) = 1 - f / (1 + f)` (minimum defence frequency), blended toward
  "calls a lot" because hackathon bots fold too little more often than too
  much.
- **Learned during the match:** count how often they call, fold or raise
  against our bets, bucketed by size (small up to 0.5 pot, medium 0.5–1,
  large above 1 pot, all-in) and by street. Blend the counts with the prior
  using Bayesian shrinkage (prior worth about 10 observations), so a few
  hands don't swing the model.
- **Calling stations** barely react to size, so `P(call)` stays near 1 and the
  biggest bet wins. **Nits** fold to anything large, so small bets take more.
  The formula finds this automatically.

### 3.3 Equity against the hands they'd call with

- Treat the opponent's range as their starting hands ranked by strength,
  narrowed each street by what they did (they called preflop, so they're in
  the top X%).
- If they call size `b` with probability `c`, assume they call with the
  **strongest `c` share** of their current range. Simulate our equity against
  just those hands with eval7, using weighted sampling of their hole cards.
- Showdowns calibrate this. Each `show` entry in `match_action_log` tells us
  what they actually called with at a given size. If they call large bets
  with weak hands, their calling range is wider than "the strongest `c`
  share", and we bet bigger.

### 3.4 When they raise back

Add a third outcome: `P(raise | b)` times our EV against their raising range
(usually we fold, and lose `b`, unless our hand is very strong). Against
maniacs who raise often, this pulls our sizes down or toward checking and
calling.

### 3.5 Planning bets over several streets

With a strong hand early, we want the whole stack in by the river. The
**geometric bet size** grows the pot evenly so that `n` equal pot-fraction
bets put the effective stack `S` in:

```
f = ((1 + 2S/P) ** (1/n) - 1) / 2
```

Add the geometric size to the candidate list on the flop and turn. Against
stations it is usually the best choice. Against folders, one smaller bet that
gets called often may take more.

### 3.6 Thin value

Bet for value whenever `E(b) > 50%` against their calling range, not only
with strong hands. Against stations, second pair or king-high on the river can
be a value bet. This also helps with the "lowest winning showdown" prize.

### 3.7 Checking whether it works

Log `expected_take` and the actual chips won on every hand where we value bet.
Over test matches, compare the predicted take with the real take. If we
consistently overestimate, the calling model is too generous or the equity
against their calling range is too high.

## 4. Push/fold for short stacks

At 15bb or less, use the Nash push/fold chart (when to go all-in as the small
blind, and when to call an all-in as the big blind). Precompute it and ship it
in `data/`. Value betting still applies: against opponents who call all-ins
too often, push only hands that beat their calling range. Against opponents
who fold too much, push wider.

## 5. Opponent modelling

Keep statistics in module-level variables. Each match is one process, and
`match_action_log` only holds the last 200 entries.

Track, with Bayesian smoothing toward a prior:

- VPIP (how often they put chips in preflop) and PFR (how often they raise)
- How often they fold to a raise, per street
- How often they fold to a bet after the flop, per size bucket (feeds 3.2)
- Aggression: bets and raises compared with calls
- Showdown cards compared with how they bet those hands (feeds 3.3)

Rough types and the main adjustment for each:

| Type | Signs | Adjustment |
|---|---|---|
| Calling station | High VPIP, rarely folds, rarely raises | Never bluff. Value bet big and thin. |
| Nit | Low VPIP, folds to aggression | Raise and bet small and often. Respect their raises. |
| Maniac | Raises constantly | Call down wider. Let them bet. Value bet by check-raising. |
| Fixed sizing | Same size every time | Read their hand strength from the size. |

## 6. Robustness

- Wrap `decide()` in try/except. If anything fails, check when free, otherwise
  fold.
- Never `check` when `amount_owed > 0`: the engine plays it as a call.
- `raise.amount` is our **total** for the street. Below `min_raise_to` it is
  raised to the minimum. Above our stack it becomes all-in.
- Use a fixed simulation count rather than a time budget, so seeded matches
  replay exactly. Use the global `random` module, which is seeded per match.
- 2 seconds per decision on half a CPU core. 2,000 simulations currently take
  about 9ms, so there is plenty of headroom.
- Do slow loading during the warmup call.

## 7. Side prizes

### 7.1 Main side-prize target: most won off 7-2

The score is the chips won on hands we win holding a seven and a two, divided
by how many times we were dealt 7-2. Losing hands don't count against the
score, but they still cost chips in the match, so the two kinds of 7-2 are
played differently.

**Seven-two suited (slightly favoured).** About 38% equity against a random
hand, 3 points better than offsuit. Weak, but not a disaster, and the flush
potential is real.

- **Preflop:** treat it as a playable hand, a little better than its ranking.
  Raise it from the button. Defend it from the big blind against a normal
  raise. Fold it to big raises or all-ins unless the pot odds say call.
- **Flush draw (four of our suit with the board):** play it like a high
  (nut-style) flush draw, not a low one. Bet or raise it as a semi-bluff,
  call bets the pot odds allow, and be willing to get all-in on the flop.
  When the flush hits, bet it for value using section 3, sizing up against
  calling stations.
- **Why this is safe enough:** heads-up, the opponent only rarely holds two
  higher cards of our suit, so a 7-high flush wins most of the time it's
  made. Be careful only on four-flush boards, or when the opponent raises big
  on the river after the flush card arrives.
- **Doesn't help the bluffing prize on the flop:** a 7-2 suited flush draw
  has about 47% equity against a random hand there, above the prize's 40%
  cutoff, so those semi-bluffs don't count as bluffs. On the turn, with one
  card to come, the equity is lower and they may count.

**Seven-two offsuit (still bad).** About 35% equity against a random hand,
in the bottom 4% of starting hands (by this measure 3-2 offsuit, at 32%, is
the very worst).

(Both figures are from 200,000 simulated deals each with the baseline bot's
`equity()` function.)

- Fold it by default.
- Exception: raise it from the button against opponents who fold too much to
  preflop raises (fold rate above about 60%). Winning the blinds without a
  fight still counts toward the score.
- After the flop, play it like any other hand (two pair or better is a value
  hand, anything less is usually a give-up).

**Track it:** log every 7-2 hand (suited or not, what we did, chips won) so we
can measure the score in testing. The prize needs at least 30 7-2 hands to be
ranked. It's dealt about once every 80 hands, so that comes from playing many
matches, not from playing more 7-2.

### 7.2 Other side prizes

- **Biggest bluffer:** a bluff is a bet after the flop with under 40% equity
  against a random hand that gets a fold. Barrel against opponents who fold
  too much.
- **Lowest winning showdown:** thin river value bets against stations (3.6).
- **Best exploitative strategy:** the opponent model plus value sizing,
  explained with logs showing how the bot adapted.
- **Best game-theoretic strategy:** the push/fold Nash chart and minimum
  defence frequency priors.

## 8. Why not a neural net

There is no training data from the real opponents, a match is too short to
learn online, and eval7 plus lookup tables already give exact equities. Simple
counts with smoothing beat a net at this sample size and are easier to explain
to the judges.

## 9. Testing

- Sparring bots in `my_bots/`: maniac, nit, always-all-in, calling station,
  "check means call", min-raiser, random.
- A duplicate-match runner: same `--seed`, seats swapped, many seeds, scored
  as pairings.
- Compare predicted take with actual take (3.7).
- `make validate` and a sandbox run before every push.

## 10. Status and roadmap

**Done:** `my_bots/baseline/bot.py`. Equity against a random hand by
simulation, then the pot-odds call or fold decision. Never raises. Beat the
calling station in its one test match (seed 7, 160 hands) and passes the
validator's test decisions.

**Done:** `my_bots/tag/bot.py`, built on the baseline.
- Preflop: a ranked table of all 169 starting hands, with tight-aggressive
  opening and defending ranges.
- Short stacks: an approximate push/fold rule.
- Postflop: fixed bet sizes, flop c-bets, calls by pot odds.
- 7-2 rules from section 7.1.
- One opponent stat: how often they fold to our preflop raise.

Won 35 of 36 test matches against the calling station, the template and the
baseline (6 seeds, both seat orders).

**Done: range-aware equity (3.3).** Equity is now measured against the
opponent's estimated range, not a random hand. Every bet, raise or call they
make this hand keeps the share of hands this opponent takes that action with,
from their match stats (shrunk toward a prior). A maniac's bets barely narrow
the range and a nit's narrow it a lot. Preflop re-raise decisions use the same
range. This fixed the A7-against-AA stack-off (it now folds to the flop
raise). Test results: 36 of 36 against the calling station, template and
baseline, and 8 of 12 against `my_bots/maniac`. The maniac losses were
coolers that lost to bad luck (AK against J4 all-in preflop, for example), but
12 matches is too few to be sure.

**Done: value betting by expected take (section 3), in `my_bots/tag`.**
Every postflop option is scored by EV and the best one is played:

- **Options:** check, call or fold, and bets of 33%, 50%, 75%, 100% and 150%
  of the pot, the geometric size and all-in.
- **Call chance:** learned per size bucket (up to 0.6 pot, up to 1.1, up to
  2.5, overbets), starting from the prior `0.05 + 0.95 / (1 + f)` scaled by
  how often they continue overall.
- **Equity when called:** against the strongest share of their range that
  continues. One simulation pass gives every share at once.
- **Raises:** assumed to go to about 3x our bet. We call with 55% equity
  against their raising range, and fold otherwise.
- **Logging:** `ev`, `expected_take` and `call_chance` go into the debug
  output.

Checks on hand-made states:

- **Against a station:** goes all-in with the nuts and with top pair, and
  value bets second pair on the river.
- **Against a nit:** bets small with top pair and bluffs the river with
  nothing.
- **Against an unknown opponent:** checks second pair on the river.

The prior's loose-caller floor was cut from 15% to 5%. At 15% the bot went
all-in with top pair against opponents it knew nothing about, because the
model treats every caller as holding the strongest hands.

**Risk limits.** The first version lost to the baseline with huge all-in
bluffs, such as 8-high betting 8,493 into an 800 pot, because the prior
assumed opponents fold almost everything to a big overbet. Where the model
knows least, a mistake costs the stack, so two limits sit on top of it:

- **Bluffs** (bets that lose on average when called): at most the pot.
- **Other bets:** at most 1.5 pot, or the geometric size if that's bigger.
  Overbets are allowed only once the opponent has called at least 70% of
  large bets, over at least 5.

`my_bots/tag_v1` is the same bot with the old fixed-size postflop play, kept
for comparison. Results, 12 seeds in both seat orders, in the sandbox:

| Opponent | `tag` (value betting) | `tag_v1` |
|---|---|---|
| Calling station | 24/24 | 24/24 |
| Template | 24/24 | 24/24 |
| Baseline | 20/24 | 23/24 |
| Maniac | 20/24 | 16/24 |

Head to head, `tag` beat `tag_v1` in 28 of 40 matches. No errors, and the
slowest match averaged 94ms per decision.

**Next, split across the two computers.**

Computer 1 (`my_bots/tag/bot.py`):

1. Check predicted take against real results (3.7), and tune the priors.
2. More opponent statistics (section 5), including showdown cards.
3. Side-prize adjustments.

Computer 2 (separate files only, so there are no merge conflicts in `bot.py`):

1. Duplicate-match harness (`tools/`). Runs many seeds in both seat orders, in
   parallel, in the sandbox image. Reports pairing wins, chips per hand and
   7-2 results. Needed because a dozen matches can't tell a better bot from a
   lucky one.
2. Sparring bots in `my_bots/`: nit, always-all-in, "check means call",
   min-raiser, random, fixed-size bettor.
3. A real heads-up Nash push/fold chart as a Python dict (hand class to
   push-up-to and call-up-to stack depth in big blinds), to replace
   `push_pct` and `call_shove_pct`.
