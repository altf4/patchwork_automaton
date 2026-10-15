# Deterministic weak Modern Burn baseline

`patchwork_automaton.baselines.modern_burn.BurnBaseline` is an explicit scripted
opponent for the numerical learner. It uses the actor's XMage observation and
returns `core.Response`. It has no neural dependency, RNG, hidden-hand access or
search. Identical decision tokens return the same cached response, including
decisions that update local bookkeeping such as selecting all attackers.

## Pinned 60-card deck

The baseline uses [DesertSamambaia's Boros Burn list](https://www.mtggoldfish.com/deck/7983760),
listed as a 5-0 result in the October 4, 2026 Modern League. The
[downloaded list](https://www.mtggoldfish.com/deck/download/7983760) has exactly
60 mainboard and 15 sideboard cards. The source metadata and exact quantities
ship in `data/modern-burn-2026-10-04.json`; no network fetch occurs at runtime.

Mainboard: 4 each of Boltwave, Boros Charm, Eidolon of the Great Revel, Goblin
Guide, Lava Spike, Lightning Bolt, Monastery Swiftspear, Searing Blaze, Skewer the
Critics and Skullcrack. Lands: 3 Arid Mesa, 3 Barbarian Ring, 2 Bloodstained Mire,
1 Elegant Parlor, 3 Inspiring Vantage, 3 Mountain, 1 Sacred Foundry and 4 Sunbaked
Canyon. The loader asserts the mainboard count is 60.

The original sideboard is retained. `Wear/Tear` is normalized to XMage's
`Wear // Tear` card name for resolution. This milestone plays single games with
XMage's `Constructed - Modern` validator; it does not perform sideboarding or
run tournament matches. Both evaluation seats use the identical published list.

## Intentionally exploitable behavior

The default policy:

- Keeps every opening hand, including zero-land and flooded hands.
- Plays a land first when available, except on every fourth **own turn**.
  It skips that turn's land drop in both main phases. This is seat-independent.
- Casts creatures before burn using a fixed card-name priority. Available spells
  are played only during its own main phases with an empty stack. It never saves
  an instant to cast on the opponent's turn.
- Activates fetch lands in its own main phase, prefers Sacred Foundry, and
  accepts shock-land life payments without considering its remaining life.
- Attacks with everything, without evaluating blocks or trades, and never blocks.
- Aims burn at the opponent whenever offered. It does not calculate lethal,
  remove a valuable creature instead, or time Skullcrack against lifegain.
- Chooses Boros Charm's four-damage mode. Uses simple color-aware mana payment
  and declines casts it cannot finish rather than retrying them indefinitely.
- Does not use Sunbaked Canyon's draw ability or Barbarian Ring's damage ability
  as strategic finishers.

The skipped land interval is configurable; the weak baseline default is four.
Forced target choices, mana payments and server triggers are still answered on
either player's turn. The restriction concerns discretionary spell/ability play,
not refusing mandatory interactions.

For Searing Blaze, the baseline prefers an opposing player/planeswalker and a
creature that player or planeswalker's controller controls. Targeting ourselves
and our own creature is also legal. The script declines that play as a targeting
preference; this is not a restriction imposed on the neural model's action space.
It therefore also misses rare winning lines where self-targeting Searing Blaze
provides a prowess trigger for lethal combat damage.
XMage can advertise an initial card click even when the script's desired target
sequence cannot be completed, so the script checks for an opposing creature
before starting that sequence. Actual target legality remains the server's job.

The script may interpret fixed backend labels and mana-cost symbols. Those
strings remain outside the neural model. No protocol or neural strategy fallback
is installed globally; using this opponent is explicit.

## Run and evaluate on CPU with uv

With local XMage installed:

```sh
uv sync --python 3.12 --extra xmage --extra neural --extra rl
uv run --no-sync python scripts/xmage-local.py start

# Two deterministic controllers; game shuffles are still random.
uv run --no-sync python -m examples.burn_baseline \
  --opponent baseline --episodes 2 \
  --output .local/xmage/burn-mirror.json

# Existing untrained model versus the baseline; alternate the baseline seat.
uv run --no-sync python -m examples.burn_baseline \
  --opponent neural --episodes 8 --seed 41 \
  --trace .local/xmage/burn-vs-neural.jsonl \
  --output .local/xmage/burn-vs-neural.json
```

In this workspace uv is also available as `.local/uv/uv`. The runner selects CPU
execution, performs zero optimizer updates, and never concedes to create an
evaluation result. Decision/time limits are truncations and cause a failed run.
Tables and sessions are closed after evaluation. The server remains running.

The benchmark below used the original small semantic registry, so most Burn
cards had unknown semantic identity. Their visible numerical properties and
offered candidates still reached the model. The current default registry now
annotates all 26 mainboard/sideboard identities; see [semantics](semantics.md).
The earlier comparison does not measure the updated feature set or the ceiling
of this architecture on Modern Burn.

Baseline unit tests cover land skipping for both seats, off-turn passing,
idempotence, blind attacks, no blocking, target selection, mana colors, modes and
the Searing Blaze legality guard. Future training should retain this opponent
unchanged as a regression benchmark and evaluate against additional opponents.
Its design is intended to be exploitable by competent play; no human-strength
claim follows from beating the random policy.

## 100-game sanity check — October 14, 2026

The fixed scripted baseline won **96/100** games against the untrained neural
policy (initialization/sampling seed 41), with four neural wins, zero draws,
zero failures and zero truncations. Both seats used the same 60-card mainboard.
The baseline alternated seats: 49/50 wins as `player_0`, 47/50 as `player_1`.
These are seat counts, not a measured play/draw split. No optimizer steps ran.
XMage shuffles were not seeded. This is one initialized model, not a comparison
across multiple model initializations.

The run used CPU inference and sequential games, totaling 16,382 decisions and
1,727.193 seconds (28.8 minutes) of gameplay, excluding setup and cleanup.
The full per-game results and configuration are saved in
[`benchmarks/burn-baseline-100-2026-10-14.json`](benchmarks/burn-baseline-100-2026-10-14.json).
The local action trace is `.local/xmage/burn-baseline-100.jsonl`.

A separate diagnostic game, excluded from those 100, took 11.784 seconds:
7.452 seconds waiting for decisions, 4.153 seconds choosing neural actions
(4.062 inside model selection), and 0.126 seconds submitting responses.
This single-game timing includes model initialization and ran alongside the
evaluation. Decision wait includes server processing and delivery, rather than
isolating network latency. It suggests concurrent games and then batched
inference as useful performance work; it does not establish their speedup.
