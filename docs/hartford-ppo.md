# Ten-deck Hartford self-play PPO

## Default for future experiments

Use the diverse opponent recipe for new Hartford learning experiments: 580
current-policy games, 100 each against three pinned historical identity policies,
and 120 StandardAggro games per 1,000-game round. The validated historical
opponents are identity checkpoints 25, 50 and 75. Scripts occupy supported
mono-green, Izzet and BW opponent-deck cells only; the 800/200 training/holdout
split and balanced deck, seat and start assignments remain intact.

New `--fork-from` continuations default to `--opponent-recipe diverse`.
`--opponent-recipe current` remains available for explicit control arms. Resume
without a recipe restores the manifest's saved choice. The current continuation
admission is deliberately limited to the tested checkpoint-100, 50-round setup;
other parents or durations require validated admission and provenance support.
The older fresh-run CLI configuration and frozen runs retain their original
protocol and do not silently adopt this change.

The [opponent-diversity experiment](benchmarks/hartford-identity-opponents-ab-2026-11-03/README.md)
records the implementation and audited results. Diverse scored 51.725% against
control over 30,000 games (nominal 95% interval 50.941%–52.509%). This is a modest
advantage for one trained agent per recipe; it does not establish training-seed
variation or generalization to unseen opponents. The sections below describe
the original compact experiment and its reproduction protocol.

The experiment uses RustMage commit
`e732a4bb42774fc20672dc2aad3952f9d1b57ae7` (observation schema 198), which
supports all 164 distinct cards in the ten exact mainboards. The driver refuses
to create a run, initialize a model, or collect games until its readiness check
passes. Launch requires an explicit command; it does not watch for updates or
start itself later.

The merged engine includes [RustMage PR #561](https://github.com/altf4/rustmage/pull/561),
which prevents ordinary back-face casts for Braided Net, The Enigma Jewel, and
Spring-Loaded Sawblades while preserving crafting. Two independent reviews,
3,014 native rules tests, 10 Python binding tests, the native simulation check,
and GitHub CI passed before the merge.

## Protocol

`examples.train_hartford_ppo` now defaults to a fresh shared `CompactActorCritic`
(`--architecture compact`, configuration version `compact-standard-v1`) with
random, independent actor and critic weights and zero optimizer steps. There is no imitation
source, supervised critic bootstrap, scripted opponent, or greedy action mode.
The ten exact mainboards come from
`src/patchwork_automaton/data/hartford-standard-decklists.json`. Sideboarding is
outside this experiment. The older Standard drivers retain their original
three-deck catalog and resume configuration.

The compact architecture uses the current private observation without recurrent
memory, event history, or previous-action features. Entity facts and card traits
are stored once, action roles reference shared rows, and exact engine-provided
candidate references retain relational context. Sparse feature coordinates and
identical rows are interned across the frozen rollout, then gathered on the
learner device. The actor and critic have separate encoders and Adam states.
This is an explicit architecture change, not a compatible execution backend for
the old recurrent model. The three-deck imitation checkpoint is not promoted or
used to initialize Hartford.

Existing recurrent checkpoints, including the stopped checkpoint 10, remain
intact and cannot be resumed into this architecture. Start a new output directory.
`--architecture legacy` reproduces the earlier recurrent configuration for new
legacy runs; resuming an old source-pinned run still requires its original frozen
source and original command. Compact resumes retain the usual strict model,
optimizer, RNG, configuration, source and engine checks.

Each of 100 rounds contains 1,000 games: all 100 ordered learner/opponent deck
matchups, including mirrors, with ten games per matchup. Eight games in each
cell train PPO and two are whole-game holdouts. This produces 80,000 training
games and 20,000 holdouts. Training seats and starting players balance within
each cell; the two holdout assignments rotate across rounds. Holdouts never
enter either optimizer or advantage normalization.

The learner always uses the latest committed checkpoint, frozen throughout
collection. Only its trajectory is recorded. Opponent quotas apply separately
to both partitions:

| Opponent | Total | Training | Holdout |
| --- | ---: | ---: | ---: |
| Latest checkpoint | 700 | 560 | 140 |
| One checkpoint earlier | 100 | 80 | 20 |
| Two checkpoints earlier | 100 | 80 | 20 |
| Three checkpoints earlier | 100 | 80 | 20 |

With no historical checkpoint, round one uses checkpoint zero for all games.
With one historical checkpoint, it receives all 30%; with two, each receives
15%. Once all three exist, every matchup gets seven current-opponent games and
one against each predecessor. The schedule fixes and shuffles these quotas;
collection dispatches the shuffled schedule through one shared queue per runner.
Each opponent inference process holds an immutable bank of up to four compact
checkpoint models and routes requests by each game's pinned version. There are
still two learner and two opponent CPU inference processes in total; weights
stay frozen throughout collection. Both seats sample at temperature 1.
Milestone evaluations also interleave their three predecessors.

After every PPO update, heldout trajectories provide read-only value/KL/policy
diagnostics. At the selected `--eval-every-games` cadence, fresh sampled games evaluate the saved checkpoint against
its previous three distinct checkpoints. Checkpoint one has only checkpoint
zero available; checkpoint two has two predecessors. Every opponent gets all
100 ordered matchups, 40 games each, including ten games in each seat/start
combination. Evaluation uses a separate, nonoverlapping seed range and writes
no training trajectories.

With evaluation every 1,000 games, there are 4,000 evaluation games after round one, 8,000 after round two, and
12,000 after each subsequent round: **1,188,000 evaluation games**, additional
to the 100,000 collection games. Preliminary engine smoke tests are separate.
The current compact launch command retains the amended 10,000-game evaluation
cadence: ten 12,000-game milestone evaluations, or 120,000 evaluation games for a
fresh 100-round experiment. Training and holdout validation remain every 1,000
games. The final round is always evaluated, including shorter acceptance runs.

Scores are `(wins + 0.5 * draws) / games`, with equal matchup and opponent
weighting. Reports include each opponent's score, raw win rates, confidence
intervals, full 10-by-10 matrices, play/draw splits, natural draws and resource
adjudications. Individual 40-game cells remain noisy. Rolling comparisons
measure recent improvement; all checkpoints remain available for later fixed
reference comparisons.

## Readiness and raytracer preparation

Run the metadata-only check from the project directory:

```sh
PYTHONPATH=src .venv/bin/python -m examples.train_hartford_ppo --preflight-only
```

This command reports readiness without importing TensorFlow, creating an output
directory, loading native gameplay, or playing a game. On raytracer the existing
neural environment is
`/home/altf4/benchmarks/mtgo-cpu-gpu-20261024/venv/bin/python`; set `PYTHONPATH=src`
to use this staged project's code rather than that environment's old checkout.

Engine integration must update the bot's RustMage revision/schema, matching
Python bindings, exported card traits, and direct bridge dependency/build.
`scripts/build-direct-standard.py` writes a provenance sidecar after a successful
build. Admission verifies all ten deck sizes, every mainboard card's complete
implementation and authored-definition provenance, source/schema/catalogue
agreement, and the selected native library's build hashes. An old or absent
direct build sidecar is rejected. The optional Python playback path checks its
own selected library provenance. There is no bypass flag.

Passing metadata admission does not establish rules correctness or encoding
coverage. Once the engine supports these decks, complete the engine's rules
tests and all-matchup sampled smoke checks, including new decision types and
large legal-action domains, before the production run.

After integration and verification, the production command is:

```sh
PYTHONPATH=src /home/altf4/benchmarks/mtgo-cpu-gpu-20261024/venv/bin/python \
  -u -m examples.train_hartford_ppo \
  --architecture compact --eval-every-games 10000 \
  --output /home/altf4/benchmarks/hartford-compact-ppo-20261101/run
```

Defaults are
100 rounds, 1,000 games per round, 40 evaluation games per cell, two runners,
48 game workers, two learner inference replicas and two opponent replicas in
total. Both policies run on CPU; PPO runs on the RTX 4090. Compact rollout inputs
have a configurable 512 MiB encoded-payload budget (`--compact-input-mib`). They
are uploaded once per update; minibatches use device-side sparse gathers without
the legacy numerical spill or actor/policy caches. Budget overflow fails before
optimization. The bound excludes Python/allocator overhead and activations; the
physical-memory guard remains authoritative. Host-memory reclamation runs after
updates, holdout validation and head-to-head evaluation. Compact runs require
32 GiB free disk; legacy runs retain their 200 GiB cache requirement. Use a frozen
source directory for the run.

`examples.smoke_hartford_selfplay` runs a separate 400-game sampled acceptance
suite, covering every matchup and seat/start combination with fresh checkpoint
zero. It checks recorded behavior against model outputs and preserves failures.
The schema-198 encoder includes public named counters, the authoritative nested
stack source, all typed actions, and legal actor-visible search/scry/selection
cards. Private or nonlegal pool entries do not become model entities.
The inherited stack representation still compresses packets into source-object,
target, and adjacency features; it is not a lossless encoding of every selected
mode or triggered ability on the stack.

On raytracer, `scripts/guard-hartford-run.py` supervises a JSON argv command in
its own process group. It stops that group after available host RAM remains
below 1,536 MiB for five seconds, records resource samples and exit status, and
never retries automatically. `--keep-windows-awake` holds a temporary Windows
execution-state request while the child runs and releases it on exit; it does
not change persistent power settings. Both the smoke and production runs use
separate, newly created guard output directories.

The previous PPO recipe is retained: actor LR 1e-4, critic LR 3e-4, batch size
512 decisions, two epochs, entropy weight .01, PPO clip .2, mean KL limit .02,
maximum-state KL limit .5, critic maximum 157 steps. Games use terminal +1/-1/0
rewards and a 50-turn limit. Compact Hartford now uses an approved **20-second
hard wall-clock deadline** (`--game-timeout 20`); legacy defaults stay at 60.
The deadline starts when the isolated game child launches, excluding worker
queue time, and interrupts native calls as well as slow inference. Timeouts
preserve readable learner prefixes and diagnostics, count once as explicitly
reported draws with target zero, and are never replayed. Decision-limit cutoffs and
engine failures stop the run and preserve diagnostics. They are not silently
reclassified or replaced.

Compact runs use `sparse-device-resident-v1`: close collection processes before
updates, stream and intern compact records under the input budget, and keep the
frozen sparse inputs on the learner device throughout actor/critic optimization
and canonical KL scans. Concurrent independent optimizers, global shuffle/RNG
reservation order, terminal targets, frozen advantages/probabilities, clipping,
and complete actor/Adam rollback on KL rejection are retained. Host prefetch and
parallel host measurements are unnecessary for the resident-input path.

The collection policy is versioned as `mixed-frozen-opponents-v1`. The first
9,000 compact-run games had a 14.5-second p99 including timeouts and a 12.2-second
p99 among natural completions. A 20-second cutoff would affect 52/9,000 (0.58%),
including 26 games already cut off at 60 seconds. These historical durations
do not guarantee the runtime of later policies.

The paused production run must adopt the new frozen source explicitly using
`--resume --adopt-collection-policy-from ORIGINAL_PROJECT --previous-guard
ORIGINAL_GUARD_STATUS`. Admission permits only the four collection, inference
and evaluation source changes and the mixed-scheduling/20-second configuration
amendment. It requires a stopped process and unchanged engine, preserves all
checkpoint and completed-game bytes, and archives the old manifest/report under
`collection-policy-recovery-v1`. Completed games in the partial round retain
their original deadline; only missing games use 20 seconds. Historical checkpoint
configurations remain pinned through the amendment boundary. Admission is limited
to the boundary before the first evaluation manifest. Later resumes use the new
source and ordinary `--resume`.

Native gameplay still constructs a dense **single-decision** observation before
the worker compacts it; it never collates wide dense inference/PPO minibatches
for compact models. Game/inference IPC, stored trajectories, learner batches and
holdout batches carry the sparse representation. Native sparse emission is a
possible further optimization, not part of this change.

Legacy raytracer runs use the `bounded-host-v1` execution policy: finish and close all
self-play game/inference processes before training, measure canonical batches
on one worker, and disable batch prefetch. Processes are recreated for the
saved checkpoint's evaluation and reused for the next collection. The effective
policy is recorded in `report.json`; the original checkpoint configuration and
PPO minibatches, update concurrency, seeds, learning rates and KL limits remain
pinned.

The first run's memory guard stopped during its initial measurement scan.
Recovery reuses its complete 1,000-game rollout and checkpoint zero. The explicit
`--resume --adopt-memory-fix-from ORIGINAL_PROJECT --previous-guard GUARD_STATUS`
admission permits only the driver replacement, checks every saved trajectory,
the complete schedule, engine/card pins and unchanged untrained checkpoint, and
archives the previous manifest/report under `memory-recovery`. The original
frozen source and guard evidence remain intact. Later resumes from the corrected
source use ordinary `--resume` without the one-time adoption options.

Standard candidate expansion retains the first 256 distinct legal actions in
the existing deterministic order, then stops generating more. Python and Rust
use the same prefix. Decisions below the cap retain all choices, and policies
still sample at temperature 1 from the retained candidates. This bounds dense
inputs for the 512-decision training minibatches; model feature widths and saved
weights remain compatible. The effective cap is recorded in the run report and
new evaluation manifests.

The cap amendment resumed checkpoint two after an oversized mono-green decision
stopped evaluation. `candidate-recovery-256` preserves the prior manifests,
checkpoint pins, completed-result hashes, and the original failed attempt. All
2,000 collected games and 7,999 successful evaluation games were retained; only
the failed game was retried. Round two's `candidate-policy-amendment.json` records
the retained results and the change of action policy for its remaining games.

## Durability and inspection

The output contains:

- `manifest.json`: immutable experiment configuration, source and engine pins.
- `checkpoints/round-0000` through `round-0100`: policies, both Adam states, RNG,
  and committed round state.
- `report.json` and `progress.json`: cumulative results and active phase.
- `rounds/round-NNNN/games.json`: durable collection outcomes and trajectory hashes.
- `rollouts/round-NNNN`: resumable per-game results and numerical trajectories.
- `evaluation/round-NNNN`: immutable duel manifest, individual outcomes,
  opponent summaries, matrices and aggregate summary.

Resume using the identical command with `--resume`. Configuration, sources,
deck data, engine and checkpoint pins must match. An interrupted collection
reuses completed games. A committed PPO update is restored exactly, and an
interrupted evaluation finishes before the next training round. Completed
evaluation games are never replayed to replace missing or modified outcomes;
genuine failed attempts remain blocking evidence. Validation, opponent loading,
evaluation and memory reclamation assert unchanged learner/optimizer/RNG state.

All checkpoints and compact summaries remain. Only committed rollout directories
older than the current and preceding round are removed. The first three full
production rounds exercise the one/two/three-opponent evaluation startup and
provide measured runtime/resource data; they count toward the 100,000 games.

## Verification

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_hartford*.py' -v
```

The schedule and orchestration tests use synthetic results, and the neural test
uses synthetic numerical trajectories on CPU. Encoder tests compare native and
Python arrays on bounded real Hartford trajectories and test private choice
visibility. Full sampled gameplay and GPU acceptance run separately on the
frozen raytracer source before production.
