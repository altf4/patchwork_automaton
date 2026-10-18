# Foundations mono-red CPU PPO

This experiment uses the existing 60-card **Foundations Red Aggro** list: 24
Mountains and four each of Fanatical Firebrand, Ghitu Lavarunner, Swab Goblin,
Firebrand Archer, Viashino Pyromancer, Goblin Smuggler, Raging Redcap, Firespitter
Whelp and Burst Lightning. Training games are mono-red mirrors under normal
RustMage rules. Both seats use the same frozen behavior checkpoint within a
collection round and sample their own actions independently.

There is **no imitation warm-up and no intermediate reward**. A verified natural
win gives +1, a loss −1, and a draw 0. Every recorded decision in a seat's game
receives that terminal return, undiscounted. Turns, life totals, damage, lands,
and spell casts are observation features, never reward bonuses. Errors and
compute-limit cutoffs are excluded from PPO and reported explicitly. They are
not converted to losses, draws, concessions, or successful replacement games.

## Run and resume

Install the neural dependencies and build the pinned RustMage backend as
described in [the backend setup](rustmage.md):

```sh
uv sync --python 3.12 --extra neural
uv run --no-sync python scripts/rustmage-local.py
uv run --no-sync python -m examples.train_rustmage \
  --output .local/red-ppo --rounds 32 --games-per-round 32 \
  --eval-games 40 --eval-horizon 25 --workers 4
```

That collects 1,024 self-play games, with 32 games per update round, using four
independent native match workers and one shared batched TensorFlow inference
service. Updates occur only after all workers finish their collection round.
CPU execution is the default; two TensorFlow intra-operation threads and one
inter-operation thread are used unless explicitly configured in the environment.
`--rounds` is the total target round number, including previously completed rounds.

```sh
uv run --no-sync python -m examples.train_rustmage \
  --output .local/red-ppo --rounds 64 --games-per-round 32 \
  --eval-games 40 --eval-horizon 25 --workers 4 --resume
```

Resume preserves weights, Adam state, optimizer iteration count and the PPO
shuffle generator. Game and action-sampling seeds derive from the round and game
index. The remaining experiment configuration must match. Checkpoints are
written atomically at round boundaries; the incomplete current round is rerun
after interruption. Parallel TensorFlow floating-point results are not promised
to be bitwise identical across machines or different batching schedules.

`report.json` retains every evaluation attempt, every training game, per-seat
returns, action counts, update diagnostics, source hashes, the library hash and
inference batch statistics. Trusted local rollout files live in `rollouts/`;
they use Python pickle and must not be loaded from untrusted sources. Model and
optimizer checkpoints live in `checkpoints/round-NNNN/`.

## Numerical model and action space

The experiment reuses the Keras 3 entity-attention/recurrent actor-critic with a
smaller CPU configuration: width 32, memory 32, one attention block and **125,026
parameters**. It has an action-scoring head and a scalar value head. This is not
a language model. Freeform prompts, card rules text and display labels never
enter its tensors.

The new RustMage feature schema includes:

- Seat-private visible objects with exact card IDs, zone, relative controller,
  current power/toughness, damage, tap/summoning state, type/color/keyword bits,
  counters and mana value; player life, counts and mana pools.
- Reviewed partial semantic labels for these ten cards, including red mana,
  damage, tap/sacrifice costs, cast/entry triggers, graveyard conditions,
  unblockability and kicker. Effective keywords come from the rules engine.
- Turn, phase, active player, typed pending decision, announcing spell identity,
  pending payment, hand/land counts and actor-private numerical observation
  history. The policy observes its own view after each engine action, including
  opponent actions and forced choices.
- Typed legal actions with separately ordered source, target and secondary
  reference features, plus ability index, payments, amounts and booleans.
  Thus casting Firebrand, activating Firebrand, targeting yourself and targeting
  the opponent have distinct representations. Live object/stack IDs bind
  references but are never learned numeric identities.

The engine supplies legality. The model samples every non-forced choice; forced
single-option prompts bypass inference while their state changes accumulate in
history. There is no scripted gameplay fallback. **Voluntary Concede is excluded
explicitly in both training and evaluation.** Self-inflicted natural losses still
count. This avoids spending the first experiment learning to avoid immediate
resignation; it does not enforce playing lands, attacking, saving mana or
choosing good targets.

The adapter retains general RustMage numeric domains. This scoped neural encoder
exactly enumerates the small domains needed by the mono-red deck, deduplicates
native representative actions and raises an explicit unsupported-interaction
error if enumeration would exceed 4,096 combinations. It does not truncate menus
or claim support for arbitrary X spells or the entire catalogue. Unknown cards,
decisions and action fields fail visibly. Mono-white neural training is outside
this experiment. Burn/XMage checkpoints cannot be loaded as RustMage policies.

The training coordinator projects each private native view to the fields actually
consumed by this encoder before constructing immutable Python objects. This avoids
copying unrelated mechanics on every action. Integration tests compare every
model tensor, decision and private-history delta with the full adapter through a
complete game. The repeat detector still hashes the full private native state,
and every selected action still passes RustMage's actor/revision/legality checks.

PPO uses frozen full behavior distributions, clipped policy ratios, a value
baseline, entropy regularization, gradient clipping, and KL-guarded epoch
backtracking. Recurrent states are detached rollout states during optimization:
this is not sequence backpropagation through whole games. The critic receives
the same terminal outcome target; no intermediate value labels are supplied.

## Evaluation: turns to lethal

Before and after training, the frozen sampled policy plays the same held-out
shuffle seeds, seats and starting-player assignments against **60 Mountains and
a completely passive opponent**. The opponent keeps, passes, and answers forced
cleanup choices; it never plays lands, casts, activates, attacks, blocks or
concedes. Evaluation data never enters PPO. Policy temperature is 1 for both
checkpoints, so a greedy trained policy is not compared against a sampled random
one.

The primary figures are successful kills / all attempts and the mean number of
the agent's **own turns** in successful games. This is not the engine's global
turn counter, which also counts opponent turns. The report includes median,
90th percentile, kill-turn histogram and fraction of all attempts won by turns
4, 5, 6 and 10. Losses, draws, action-limit cutoffs, turn-limit cutoffs, repeated
visible-state cutoffs and errors remain visible.

To compare two policies without hiding failed attempts, the report also gives
an evaluation-only capped score: a win on turn T scores T; a failure by horizon
H scores H+1. Lower is better. The paired comparison reports the mean reduction
and a bootstrap interval over matched evaluation seeds. This score is **never
an RL reward**, and its arbitrary failure value should always be read alongside
the actual win and failure counts. A positive point estimate whose interval
crosses zero is not convincing evidence of improvement.

Evaluate a checkpoint independently, or omit `--checkpoint` for the scripted
Foundations baseline:

```sh
uv run --no-sync python -m examples.evaluate_rustmage \
  --checkpoint .local/red-ppo/checkpoints --games 100 --horizon 25 \
  --output .local/red-goldfish.json
uv run --no-sync python -m examples.evaluate_rustmage \
  --games 100 --horizon 25 --output .local/red-baseline-goldfish.json
```

`--greedy` is available for a separately labeled evaluation. Checkpoint hashes
are verified unchanged by the standalone evaluator. The goldfish metric measures
basic execution; it does not establish competence at blocking, interaction or
competitive play. A win-only training objective also does not directly prefer
fast kills once a policy reliably wins.

The earlier roughly 20× RustMage/XMage result measured scripted-policy games.
Neural training also pays for observation encoding, batched network inference,
trajectory storage and gradient updates. Its throughput must be measured
separately; multiplying the old neural games/minute by 20 is not justified.

## Measured run: October 18, 2026

The [complete experiment report](benchmarks/foundations-red-ppo-2026-10-18.json)
records **1,024 natural terminal self-play games**, 32 PPO rounds and 306,223
recorded decisions. There were no training errors, draws or compute cutoffs.
Both seats trained from their terminal outcomes; the report's seat-relative
win/loss counts are not a measure of improvement in mirror self-play.

The same 40 held-out shuffle seeds, seat assignments, starting players and
action-sampling seeds were used before and after training. Each policy faced the
passive 60-Mountain opponent, with a 25-own-turn horizon:

| Policy | Wins / attempts | Failed attempts | Mean own turns to win | Median | 90th percentile |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scripted Foundations baseline | 40 / 40 | 0 | 6.075 | 6 | 7 |
| Fresh neural policy | 40 / 40 | 0 | 15.025 | 14 | 18 |
| Neural policy after 1,024 self-play games | 40 / 40 | 0 | 8.600 | 8 | 12 |

The trained policy used **42.8% fewer turns on average**. It improved on 39 of
40 paired attempts and worsened on one. The paired mean reduction was 6.425
turns, with a 95% bootstrap interval of **5.450–7.425 turns**. Because every
attempt succeeded, the capped-score comparison equals the ordinary turn
comparison here. Wins by turn 10 increased from 0/40 to 33/40.

This demonstrates improved basic execution for this deck after win-only PPO.
It is one training seed and one deck, and does not establish a head-to-head win
rate against the scripted baseline or robustness to interactive opponents.
The scripted baseline still kills faster. Evaluation was performed only after
the fixed training run, without choosing among intermediate checkpoints.

The final local checkpoint is
`.local/red-ppo-2026-10-18/checkpoints/round-0032/policy.keras`, SHA-256
`cb03805656200916826e58f0dcf2c3c71cfe4e8ac54cda0484e664aebc24d865`.
The checkpoint and compressed trajectories remain local; the JSON audit report
is retained with the repository documentation. The full test suite passed
291 tests, and the 11 RustMage neural tests passed after the final checkpoint
vocabulary validation change.

For the **960 games with phase timing**, collection took 1,924.88 seconds and
PPO updates took 1,171.36 seconds: **29.92 collected games/minute**, or **18.60
games/minute including updates**. These figures exclude evaluation, checkpoint
serialization and startup. The first 64 games preceded the compact-view
optimization and timing instrumentation; the resumed run used an equivalent
feature projection verified by integration tests. Its final inference batches
averaged 1.79 decisions, with one TensorFlow graph trace.

The update phase accounted for **37.8%** of measured collection-plus-update time.
It includes backpropagation, forward/KL checks and CPU trajectory preparation.
Later policy-change backtracking increased this share from the roughly 33%
observed partway through the run. Offloading only this entire phase to an
infinitely fast device would therefore cap the measured overall speedup at
**1.61×**; actual GPU speedup has not been measured. A hypothetical 5× faster
whole update phase would yield 1.43× overall, before transfer overhead.

The host has eight CPU cores. Short read-only process samples found about 1.16
cores used during collection and 1.30–1.34 during updates. These are brief
samples, not whole-run CPU averages. The collector currently uses four Python
threads and one shared inference service. Native Rust calls release the GIL;
Python state conversion and encoding still share it. Separate collection
processes with centralized batched inference are a candidate optimization,
whose IPC costs and throughput need measurement.

A subsequent [frozen-policy worker benchmark](benchmarks/foundations-red-workers-2026-10-18.json)
compared four and eight **threads**, using the final checkpoint, 32 matched game
seeds per setting and trajectory writing enabled. Two 16-game groups used
opposite configuration orders; four warm-up games preceded each trial and were
excluded from timing. All 64 timed games terminated naturally, and all 32 paired
games had identical action counts, decision counts, turn counts, final life and
outcomes. The checkpoint hash was unchanged.

| Collection threads | Collected games/minute | Mean CPU cores used | Mean inference batch |
| --- | ---: | ---: | ---: |
| 4 | 30.16 | 1.14 | approximately 1.8 |
| 8 | 29.36 | 1.15 | approximately 3.1 |

Doubling threads produced no useful throughput improvement in this short
diagnostic, despite larger inference batches. These are collection-only rates,
not rates including PPO updates. The result supports testing separate worker
processes rather than simply raising the thread count; it does not establish a
process-based speedup. The benchmark report embeds its exact source and seeds.

## Collection pipeline profile

The [stage profile and native replay report](benchmarks/foundations-red-pipeline-profile-2026-10-18.json)
uses the same final checkpoint, four collection threads, two TensorFlow
intra-operation threads, and 32 fixed mono-red mirror seeds starting at
23,000,000. There are no gradient updates in this measurement. Model loading,
graph warm-up and four warm-up games per pass are excluded. Trajectory writing
and the repeated-state safeguard are enabled.

Uninstrumented passes before and after took 65.24 and 65.88 seconds. The
instrumented pass took 68.51 seconds, or **4.5% longer** than their mean, and
used 77.92 CPU-seconds across threads. These percentages describe **collection
CPU consumption**, not fractions of elapsed time or promised speedups:

| Stage | CPU-seconds | Share of process CPU |
| --- | ---: | ---: |
| Rust API calls, including native observations and JSON output | 4.85 | 6.2% |
| Response copying and UTF-8 decoding | 0.59 | 0.8% |
| JSON parsing into Python dictionaries | 11.64 | 14.9% |
| Building immutable Python observation objects | 15.66 | 20.1% |
| Numerical feature and legal-action encoding | 8.37 | 10.7% |
| History tracking and history arrays | 2.38 | 3.1% |
| Packing individual and inference batches | 1.45 | 1.9% |
| NumPy-to-TensorFlow conversion | 2.24 | 2.9% |
| TensorFlow forward call plus native runtime workers | 11.63 | 14.9% |
| TensorFlow outputs to NumPy | 0.15 | 0.2% |
| Repeated-state safeguard: JSON encoding | 3.78 | 4.8% |
| Compressed trajectory writing | 1.99 | 2.5% |
| Other coordination, selection, allocation and residual CPU | 13.19 | 16.9% |

Python input preparation (response copying/parsing, observation objects,
features, history, batching and tensor conversion) accounts for approximately
**54% of collection CPU**. The largest individual conversion stage is building
Python observation objects. A separate two-game call-count diagnostic recorded
1,056,989 calls to `freeze`, 107,018 immutable `__post_init__` calls, and 3,192,890
`isinstance` calls across the pipeline. These counts support investigating
recursive freezing and generic object initialization; they are not an
additional timing attribution.

The timers use per-thread stacks and exclusive thread CPU time, so nested calls
are not counted twice. Worker inference waits are recorded separately; their
elapsed times overlap and must not be added into an elapsed-time percentage.
Linux thread counters capture 6.66 CPU-seconds in persistent native runtime
workers, principally TensorFlow. The main event loop accounts for another 2.63
CPU-seconds. After these measurements, about 1.1% of process CPU remains
unattributed. The optional cProfile diagnostic reports **call counts only**:
per-thread custom clocks are unsuitable for this runtime's global profiling
callbacks, and its timing totals are deliberately excluded.

The collector received **1,345,986,851 bytes of JSON** in 67,342 responses. Of
these, 22,426 action responses contained 448,293,827 bytes (**33.3%**) of
observations that the collector discards before querying both private views
again. This is a concrete redundant conversion path. The compact learning-view
projection currently runs after JSON parsing, so it does not avoid decoding
unused fields in the full native observation.

To separate native stages, the replay helper links the existing release rlib
and reconstructs the exact captured protocol request sequence. All response
byte lengths and Adler-32 checksums matched, across three replays of all 32
games. Mean native stage times for that workload were:

| Native replay stage | Seconds |
| --- | ---: |
| Apply rules/actions | 0.168 |
| Construct player-visible observations | 0.283 |
| Serialize response JSON | 0.865 |
| Parse request JSON | 0.015 |
| Response deallocation and game initialization | 0.013 |

These are **isolated single-thread replay measurements**. They exclude Python,
the live FFI boundary, trace-file IO and validation; they are not additive to the
live collector timings. Allocation, cache, linking and concurrency differences
also prevent attributing the difference simply to FFI overhead. Within the
native replay, JSON serialization is the largest stage, but the live native
calls together consume only about 6% of collection CPU.

Every encoded feature array and selected action index matched exactly across
the two reference passes, the instrumented pass and trace capture: 64 seat
trajectories and 9,910 sampled decisions in each pass. Aggregate game outcomes,
action counts, turns and final life also matched. The checkpoint and engine
shared-library hashes stayed unchanged. Three focused tests verify nested
exclusive timing, thread isolation and stack restoration after exceptions.

At the time of this profile, the next optimization experiments were:

1. Reuse the observation returned when an action is submitted, preserving its
   viewer and revision checks, to eliminate the redundant response path.
2. Let the numerical training encoder consume a compact private representation
   without constructing the general immutable Python object graph. A later
   native numerical-buffer API could also remove JSON parsing and Python feature
   assembly. Preserve feature equivalence and hidden-information boundaries.
3. Avoid re-serializing the full observation for repeated-state detection, using
   an equivalent native visible-state fingerprint or a cheaper canonical form.

Separate game-worker processes remain a useful parallelism experiment for this
Python-heavy path. They will add IPC and serialization costs, so the measured
CPU shares should not be translated directly into a multiprocessing speedup.

Reproduce the profile and native replay with the saved checkpoint:

```sh
uv run --no-sync python -m examples.profile_rustmage \
  --checkpoint .local/red-ppo-2026-10-18/checkpoints/round-0032/policy.keras \
  --output .local/red-profile --games 32 --workers 4 \
  --trace-games 32 --cprofile-games 2
uv run --no-sync python scripts/profile-rustmage-native.py \
  --traces .local/red-profile/native-traces \
  --output .local/red-profile/native-report.json --repeats 3
```

The native helper uses RustMage's mandatory memory-limited build guard and links
existing release artifacts. `--resume` on the Python profiler reuses completed
passes after an optional diagnostic fails; `--redo-call-profile` repeats only
the call-count diagnostic. Call profiling is sequential and excluded from
throughput measurements. Raw `.pstats`, request traces and rollouts stay local;
the repository report retains the measured results and source hashes.

## Pipeline quick fixes

The collector now reuses the observation returned by a successful native action,
checking its schema, viewer and advancing revision before caching it. The other
seat still receives its own independently redacted observation. Public adapter
snapshots remain deeply immutable. The internal collector instead borrows the
worker-owned private JSON tree through shallow records, avoiding the recursive
freezing and copying needed for the public API. Native responses are separate
trees, so advancing the game does not mutate previous views.

The neural extra includes `orjson` for response decoding and sorted full-state
repetition fingerprints. The guard still compares every private state field
except revision. Numerical actions are expanded once, and inference requests
carry unpadded arrays until the shared batcher packs them together.

PPO retains decoded trajectories and packed minibatches in an update-local cache
with a default 256 MiB data budget. Three quarters is reserved for trajectory
records and one quarter for packed batches; oversized data falls back to the
streaming path. The budget covers retained cache data, not total process RSS or
TensorFlow workspaces. Set `StreamingPPOTrainer(cache_bytes=0)` to disable it.
Permutation draws, row order, frozen behavior probabilities, terminal rewards,
optimizer rollback and KL acceptance limits are preserved. For the current
deterministic `PolicyNetwork`, loss statistics and exact KL share one forward
pass. Models without the explicit deterministic-forward contract retain the
separate training/inference-mode evaluations.

The matched collection/update benchmark uses a frozen checkpoint, fixed seeds,
four workers, and a two-epoch PPO update. Compilation and warm-up are excluded.
Use the earlier report as the update reference to compare learner performance
on exactly the same frozen trajectory files:

```sh
uv run --no-sync python -m examples.benchmark_pipeline \
  --checkpoint .local/red-ppo-2026-10-18/checkpoints/round-0032/policy.keras \
  --output .local/pipeline-benchmark --games 32 --workers 4 \
  --update-reference .local/pipeline-quickfix-before/report.json
```

Omit `--update-reference` for a standalone benchmark. Updated weights are written
only to the benchmark output directory; the original checkpoint is unchanged.

The [October 18 matched benchmark](benchmarks/foundations-red-pipeline-quickfixes-2026-10-18.json)
measured the following on 32 fixed mono-red mirror games:

| Measurement | Before | After | Throughput improvement |
| --- | ---: | ---: | ---: |
| Collection | 64.57 s | 40.11 s | 1.61× |
| Two-epoch PPO update | 33.73 s | 22.12 s | 1.52× |
| Collection plus update | 98.30 s | 62.23 s | 1.58× |
| Games/minute including update | 19.53 | 30.85 | 1.58× |

This is one matched run per implementation, not a long-run throughput estimate.
All 32 terminal game records and all feature/action hashes matched across 64
seat trajectories and 9,910 decisions. Using the identical frozen rollouts,
both learners accepted 620 optimizer steps with no rejected epochs and produced
bit-for-bit identical final weights and update metrics. The optimized update
cached all 64 trajectories plus 46 packed batches, retaining about 143 MiB
within the 256 MiB budget. The full suite passed 299 tests, including cached
versus streaming PPO equivalence, response-viewer checks, private history and
tensor equality, and repetition-guard semantics.
