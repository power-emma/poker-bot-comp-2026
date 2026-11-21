# huff_and_even_more_puff

Team huff_and_even_more_puff's heads-up No-Limit Hold'em bot (v10), all in `bot.py`. It ships with
`data/preflop_equity.json`, a 169x169 all-in equity matrix used for push/fold and preflop range equity.

Each time it has to act, it estimates what the opponent holds, works out the expected chips (EV) of
every move it could make, and takes the highest. It keeps statistics on the opponent across the match,
so its read of their hands, bluffs and calls changes as the match goes on.

## 1. What happens on each decision (`decide`)

1. **Update opponent stats** (`update_stats`) from the action log. This counts how often they bet,
   raise, call or fold, broken down by bet size. It also tracks how often they fold to our raises before
   the flop and how often they re-raise. And it records how often their bets turned out to be bluffs in
   hands that reached showdown.
2. **Before the flop:** a decision tree driven by charts and equity (section 2).
3. **After the flop:** build a model of the opponent's likely hands, simulate showdowns, and score every
   option by EV (sections 3 and 4).
4. **Seven-two rules** for the 7-2 side prize, then return the move.
5. **Safety:** everything runs inside a try/except, so if anything crashes it checks, or folds if it
   can't check.

Each starting hand has a ranking, its "top %", from a table of how often it beats a random hand. AA is
near 0%, 72o near 100%.

## 2. Before the flop (`preflop`, `short_stack`)

The bot plays differently depending on stack depth: the smaller stack, measured in big blinds.

**15bb or less: all-in or fold.**

- **When it opens:** it goes all-in with the hands that beat how often this opponent calls all-ins. It
  starts from the Nash push/fold chart and adjusts as it sees real calls.
- **Facing a raise:** it compares three options: fold, call (with its equity discounted because it may
  not see a showdown), and re-shove using the opponent's observed calling rate.

**15–40bb (where most of a freezeout is played):**

- **Opening:** from the button it opens the top 75% of hands to 2bb.
- **Re-shoving:** over the opponent's open it re-shoves when that beats calling, if stacks are 25bb or
  less.
- **Calling:** calls out of position count only 85% of their equity.
- **After re-raising:** if it re-raised and then gets re-raised, it goes all-in or folds; it never just
  calls.

**Over 40bb:**

- **Button:** opens the top 50% to 2.5bb.
- **Big blind after the button just calls:** raises the top 20% to 4bb.
- **Facing a raise:**
  - It works out how wide the opponent raises from what it has seen.
  - It re-raises to 3 times their raise if its equity against that range is 60% or more.
  - It calls if the pot odds cover it.
  - Otherwise it uses a defend chart: the top 40% against a raise of 3bb or less, falling to 6% against
    big raises.

## 3. Estimating the opponent's hand after the flop (`opponent_range`, `RangeSample`)

1. **Starting point:** the hands they'd have played the way they did before the flop.
2. **Narrowing:** each bet, raise or call keeps only their strongest share. The share is how often this
   opponent takes that action. If they bet 30% of the time when checked to, a bet keeps their strongest
   30%. So a maniac's range barely narrows, while a nit's narrows a lot.
3. **Bluffs:** after a bet, a share of bluffs is added from their weaker hands. It starts at the standard
   bluff ratio for that bet size and is adjusted by how often this opponent's bets turned out to be
   bluffs at showdown.
4. **Simulation:** about 3,000 showdowns against hands drawn from that range. Each result is tagged with
   where the opponent's hand sits in the range, from strongest to weakest. That lets it ask, for
   example, "what's my equity against just the top 40% that would call this bet?"

## 4. Choosing a move after the flop (`postflop`, `bet_ev`)

Every option gets an EV:

- **Check:** its equity times the pot.
- **Call:** its equity times the final pot, minus the call. Out of position, flop and turn calls count
  only 85% and 90% of its equity. This is new in v10.
- **Each bet size:** a third of the pot, half, three quarters, the full pot, 1.5 times the pot, the size
  that gets all the chips in by the river, and all-in. For each size:
  - The chance they fold, based on how often this opponent has folded to that size.
  - If they call, its equity against the strongest hands that would call.
  - If they raise, it either calls against their raising range or folds and loses its bet.

It takes the option with the highest EV, after these limits:

- **Bluffs:** bets that lose when called are capped at the size of the pot.
- **Value bets:** capped at 1.5 times the pot, unless the opponent has been seen calling big bets a lot.
- **Overbets (v10):** a bet bigger than the pot needs two pair or better, not counting pairs on the
  board.
- **River all-ins (v10):** facing a river bet, it only goes all-in, or puts in half its stack or more, if
  3 or fewer of the opponent's possible hands beat it.
- **Early in the match:** it commits at most 30% of its stack, rising to all of it by hand 30, while it
  learns the opponent. Near-unbeatable hands are exempt.

## 5. Seven-two (for the "most won with 7-2" prize)

- **72 suited** is played better than its ranking: it opens from the button and defends against small
  raises. When it picks up a flush draw, it plays it like a strong flush draw and raises.
- **72 offsuit** folds, unless this opponent folds to raises before the flop more than 60% of the time.
  Then it steals with it.
- **All-ins:** 7-2 never goes all-in, so a win counts as chips won with 7-2 rather than as a coin flip.

## 6. Limits of the design

- **Calling too little against strong opponents:** the range model narrows hard when someone bets, so
  against a balanced bot like Slumbot it folds too often on the turn and river.
- **Stats need time:** early on it relies on its default assumptions until it has seen enough hands.
- **Fixed simulation counts:** they're a set number, not a time budget, so a seeded match replays
  exactly. Decisions take about 25–60 ms, well inside the 2-second limit.

## Results (v10)

| Test | Result |
|---|---|
| Sparring A/B, 10 bots × 10 duplicate pairs | 0.755 (51 W / 49 D / 0 L), vs 0.720 for the same bot without the v10 overbet and out-of-position changes |
| Slumbot, 2,000 hands (200bb, flat 50/100) | −31 bb/100 (±81 at 95%) |
| Live against Sonnet, 50 hands (seed 4242) | +3,190 |
| Live against Opus, 50 hands (seed 4242) | +17,806 (mostly two stacks Opus put in badly) |
