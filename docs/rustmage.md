# In-memory RustMage backend

RustMage runs rules enforcement inside the Python process through its upstream
`ctypes` binding and a release-built Rust shared library. No socket, JVM, or
server participates in a RustMage game. MTGO and XMage remain available.

This integration pins [RustMage revision 78d71785](https://github.com/altf4/rustmage/tree/78d717856af36274018634b54a3fc8cbacf4c829),
native ABI 1 and observation schema **131**. Both the binding and the loaded
binary's observation schema are checked. RustMage rejects unsupported cards;
catalogue presence alone is not evidence of implemented rules.

## Setup

On the tested Linux host, use Rust's edition-2024 toolchain, Git, uv, and a user
systemd session for RustMage's build resource guard:

```sh
uv sync --python 3.12 --extra xmage
uv run --no-sync python scripts/rustmage-local.py
```

The setup script clones the pinned revision into ignored `.local/rustmage` and
builds its release shared library using the upstream **mandatory** resource
wrapper (one Cargo job, shared 6 GiB cgroup). It refuses to reset an existing dirty
checkout or one at a different revision. Build logs/resource reports are retained
by the upstream wrapper under `~/.cache/rustmage-build-reports`.
The build/library hash is written to `target/patchwork-build.json`.

No additional Python dependency is needed for RustMage. Set `RUSTMAGE_ROOT` or
pass `checkout=...` for a different checkout location. An explicit `library=...`
can select a retained shared library. Keep the schema file, Python binding and
library from the same build. The shared library is not bundled into our wheel.

## Shared game interface

```python
import asyncio
from patchwork_automaton.baselines.foundations import FoundationsAggro, foundations_deck
from patchwork_automaton.rl import RustMageSelfPlay, EpisodeEnd

async def play():
    factory = RustMageSelfPlay(foundations_deck('red'), foundations_deck('white'))
    policy = FoundationsAggro()
    try:
        runner = await factory.reset(seed=801)
        while True:
            turn = await runner.next()
            if isinstance(turn, EpisodeEnd):
                return turn
            await runner.submit(policy.choose(turn.observation))
    finally:
        await factory.close()

asyncio.run(play())
```

Each match has two seat-private `GameHandle`s. They provide the existing async
`snapshot`, `wait_for_decision`, `submit`, `concede` and `wait_for_result` methods.
Observations contain visible objects, player counts/life/mana, turn/phase,
pending decisions and verified terminal outcomes. `native` retains the exact
structured RustMage view, including stack, targeting, combat, counters, effective
keywords, current card types, and pending payments. Opponent hands and library
contents are never assembled from the other seat's view.

`RustMageSelfPlay.reset(seed=n)` replays a shuffle; omitting the seed draws fresh
entropy and records the chosen value in `factory.seed` for reproduction.

Actions are finite typed Rust actions (`AnnounceCast`, `ChooseTarget`, `Attack`,
etc.) carried by `Option.constraints['action']`. Select an offered option with
`Response(option.id)`. Numeric domains additionally accept:

| Option | Response value |
| --- | --- |
| `domain:x_choice` | Integer in one of the exact offered intervals |
| `domain:damage_assignment` | Integer in the inclusive damage range |
| `domain:mana_payment` | Six integers in W/U/B/R/G/C order satisfying required, available and generic constraints |
| `domain:mana_lot_spend` | Integer in the offered range and optional exact allowed set |

Native action representatives are not treated as exhaustive for large numeric
domains. The adapter checks arguments and Rust enforces legality. There is no
freeform text action generation. Failed validation leaves the decision usable;
successful actions invalidate its token. Tokens include game, session, actor and
revision; old actions cannot be reused after restoration or on another fork.

The match's native calls are synchronous. Use one owner event-loop thread per
match and independent matches on separate workers. This first comparison uses
one sequential worker, with neither batching nor parallel games.

## Branching for search

The factory exposes its `match`, or create `backends.rustmage.RustMageMatch`
directly. `checkpoint()`, `restore(checkpoint)`, `drop_checkpoint(checkpoint)` and
`fork()` operate on owned native state. Use `with match.fork() as branch:` to
release a branch reliably. Close the original match explicitly too.

A restore advances the revision. Obtain a fresh decision before submitting an
action. A checkpoint belongs to its original match, cannot be restored on a
fork, and becomes invalid when released. Fork mutations do not change the parent.
If exploring while a `DecisionRunner` has pending tasks/current decisions, use
an independent fork rather than restoring underneath that runner.

Checkpoints contain the full engine state, including hidden cards. They are
coordinator capabilities, **not observation features**. This is the branching
foundation for future search; information-set sampling/determinization and MCTS
itself are not implemented by this adapter.

## Foundations workload

Both decks have exactly 60 mainboard cards, no sideboard, 24 basics and four each
of nine spells. Every printing is explicitly `FDN`, including Foundations
starter-product reprints. Printing membership, complete implementation status,
Oracle IDs and source-definition hashes are recorded in
[`foundations-aggro.json`](../src/patchwork_automaton/data/foundations-aggro.json).

| Mono-red: 24 Mountain, 4 each below | Mono-white: 24 Plains, 4 each below |
| --- | --- |
| Fanatical Firebrand | Savannah Lions |
| Ghitu Lavarunner | Healer's Hawk |
| Swab Goblin | Bishop's Soldier |
| Firebrand Archer | Leonin Skyhunter |
| Viashino Pyromancer | Helpful Hunter |
| Goblin Smuggler | Dazzling Angel |
| Raging Redcap | Inspiring Paladin |
| Firespitter Whelp | Prideful Parent |
| Burst Lightning | Inspiring Overseer |

`FoundationsAggro` uses the same strategy on both engines: keep all hands, play
lands, cast affordable spells cheapest-first then alphabetically in its own main
phase with an empty stack, target the opposing player, attack all and never
block. It declines kicker/optional payments and does not activate abilities.
Mandatory targets, trigger ordering, payments and cleanup discards are answered.
This provides casting, haste, flying, lifelink, draw/entry/cast triggers, token
creation and first/double-strike combat workloads. It is intentionally simple,
not a competitive deck benchmark or an exhaustive card-rules validation suite.

## Performance comparison

Start the existing local XMage installation, then run:

```sh
uv run --no-sync python scripts/xmage-local.py start
uv run --no-sync python -m examples.compare_engines \
  --games 40 --warmup 2 --one-cpu \
  --output .local/foundations-comparison.json
```

The reported **games/minute** counts complete natural games, including each
game's setup, policy/adapter work, gameplay, result delivery and cleanup. Engine
startup, compilation and neural inference/training are excluded. Setup and play
times are also reported separately. Errors and limits are retained as failures,
not replaced by successful retries or converted to wins by concession.

The engines alternate within the run; seats and starting players rotate
independently. `--one-cpu` pins Python and every managed XMage JVM thread to one
shared logical CPU and restores/verifies their previous affinities afterward.
The report includes per-game results, source/library hashes, warmups and a
bootstrap interval over game timings. Native decision granularity differs, so
decisions/second is diagnostic rather than a matched-work comparison.

**This is a matched workload, not a deterministic cross-engine replay.** RustMage
seeds its shuffle; stock XMage has no verified shuffle-seed API. Both engines
play the exact same decks and strategy, but draw orders and game lengths vary.
This measures the actual Python integration (including RustMage's JSON FFI and
observation conversion), not the maximum speed of a native Rust rollout loop.

### Measured on 2026-10-18

40 measured games per engine, two excluded warmups each, on logical CPU 7 of an
Intel Core Ultra 5 335. **All 80 measured games completed naturally**, with zero
errors or cutoffs. Original thread affinities were restored and verified.

| Metric | RustMage | XMage |
| --- | ---: | ---: |
| Complete games/minute | **94.14** | **4.58** |
| Bootstrap 95% interval, games/minute | 89.82–99.33 | 4.42–4.75 |
| Mean seconds/game, including setup/cleanup | 0.637 | 13.094 |
| Mean setup seconds/game | 0.0019 | 3.2161 |
| Mean gameplay seconds/game | 0.6354 | 9.8646 |
| Mean global turn at completion | 11.35 | 11.65 |
| White / red wins | 34 / 6 | 28 / 12 |

RustMage was **20.55× faster end to end** for this workload. Different shuffles
and outcomes mean this does not establish cross-engine rules equivalence.
The bootstrap intervals describe variation in this sample, not other hardware,
parallel worker configurations or arbitrary decks.

The [full measured report](benchmarks/foundations-engines-2026-10-18.json) retains
all attempts, excluded warmups, configuration, hashes and audit results. Two
small post-measurement hardenings made unseeded factory resets choose fresh
entropy and made the harness verify the pinned clean checkout itself. The run
used explicit seeds and an independently verified clean checkout; its exact
versions of those two source files are preserved in the report.

## Tests and current boundary

```sh
uv run --no-sync python -m unittest discover -s tests -p test_rustmage.py -v
uv run --no-sync python -m unittest discover -s tests -p test_foundations_policy.py -v
```

Tests cover private views, deterministic full-game replay, natural outcomes,
stale/foreign action rejection, failed-action transactionality, nonactive-seat
concession, close notification, seeded reset, cutoffs, exact numeric domains,
and fork/restore/replay during a partially paid spell. Native integration tests
skip explicitly when the library has not been built. The live comparison also
exercises the Foundations policy through XMage's Standard deck validation.

Final validation on 2026-10-18: **280 repository tests passed**, including 12 new
adapter/policy/benchmark tests. The wheel builds and contains the backend,
episode factory, controller and deck data. The setup command and the hardened
benchmark CLI were also exercised successfully.

The existing Burn Keras encoder/checkpoint remains XMage-specific. A separate
[mono-red numerical encoder and CPU PPO experiment](rustmage-training.md) now
supports the Foundations red deck on RustMage, with batched inference, mirror
self-play, checkpoint/resume and turns-to-lethal evaluation. It starts with new
weights and rejects incompatible Burn checkpoints. The neural schema currently
covers the red deck and an inert 60-Mountain opponent; the scripted adapter and
throughput comparison continue to support both red and white. Sideboarding and
match-level play are not exposed. Real training rewards remain terminal-only.
