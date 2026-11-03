# Hartford identity PPO: 100,000 additional collection games

Status: **COMPLETE**, finished 2026-11-03 11:53 UTC (04:53 MST).
All 100 rounds and 100,000 new collection outcomes are committed: 80,000 PPO
training games and 20,000 whole-game holdouts. Checkpoint 100 adds this exposure
to the parent's 200,000 games. The final production guard exited normally with
code zero. The complete read-only audit passed; see
[completion.json](completion.json) and its linked timestamped audit. No
policy-strength evaluations were run as part of training. The user subsequently
authorized a separate [11,200-game comparison](../post-training-comparisons/README.md)
against the original 200K agent and scripted baselines; those games are not
training outcomes or milestone evaluations of this run.

The [compressed report](report.json.gz) preserves all 100 rounds' training,
holdout and timing metrics. The [preservation inventory](../preservation/README.md)
links the exact frozen source archive and its hashes. Models, both optimizer
histories, saved RNG states, training outcomes and retained raw diagnostics
remain in the Raytracer output below; this Git archive does not contain those
production binaries or raw training data.

This is one identity-only training arm, transferring the completed 200k compact
policy into the experimental identity architecture. It adds 100,000 collection
games to the parent's 200,000-game exposure. It is a separate experiment with
fresh actor/critic Adam state and trainer RNG. Original production results and
optimizers remain intact. Local checkpoints are numbered 0 through 100,
with checkpoint zero representing the unchanged transferred 200k policy.

| Setting | Recorded value |
| --- | --- |
| Collection | 100 rounds × 1,000 games |
| PPO training | 800 whole games per round; 80,000 total |
| Holdout | 200 whole games per round; 20,000 total |
| Matchups | All 100 ordered Hartford deck matchups every round |
| New input | Visible card identities, 16-dimensional embeddings, exact ordered references |
| Optional rule descriptors | Disabled |
| Lookahead | Not implemented or enabled |
| Milestone head-to-head evaluations | Disabled, including final checkpoint |
| Collection seeds | 412610510000–412610609999; disjoint from original 200k |
| Self-play | Sampled temperature 1; steady quotas 700 current / 100 each against three predecessors |
| Startup | Transferred checkpoint zero first; normal predecessor startup quotas |
| Processes | Two collectors, 48 game workers, two learner and two opponent CPU inference processes |
| Per-game limits | 20 seconds, 50 turns, 6,000 decisions |
| Learner | GPU, same PPO recipe as completed compact experiment |
| Learning rates | Actor 0.0001; critic 0.0003 |
| PPO | Two epochs, batch 512, clip 0.2, entropy 0.01 |
| KL guards | Mean 0.02; maximum state 0.5; rejection/rollback retained |
| Rollout payload | Bounded GPU-resident store, 512 MiB |
| Physical-memory guard | Minimum 1,536 MiB available for five seconds |
| Disk admission | At least 32 GiB free |
| PPO phase timeout | 1,800 seconds |

The parent policy SHA256 is
eb56f40c7e3519a9cbb69dbbc9d85129bec91a6bf600d13ce3d599c5ee64fef6.
Its policy, model manifest, and training state were copied and verified without
changing the originals. Frozen source includes explicit CLI transfer support.
The initial runtime was 4bedacf19687e98345f26c7b7492cd2bc149fe38. After the
recorded round-43 recovery, current runtime is 4f64940e46e0181bf963c0de0f6f5a9a51a926b5
in project-trigger-v2, with fresh guard/control directories. Read launch.json
for authoritative current paths. The feature ABI remains separately pinned to
e732a4b, as in the completed run.

Final remote paths:

- Source: /home/altf4/benchmarks/hartford-identity-ppo-20261102/project-trigger-v2
- Parent: /home/altf4/benchmarks/hartford-identity-ppo-20261102/inputs/source-200k
- Output: /home/altf4/benchmarks/hartford-identity-ppo-20261102/run-v1
- Launcher: /home/altf4/benchmarks/hartford-identity-ppo-20261102/run-trigger-recovery-v1-launch/launch.sh
- Guard: /home/altf4/benchmarks/hartford-identity-ppo-20261102/run-trigger-recovery-v1-guard

Source and copied parent inputs are read-only. The launcher verifies all 387
frozen source files, copied and original parent hashes, and runtime admission
before invoking the supervisor. The experiment started as a fresh run, with no
parent-run resume or historical adoption flags. The round-43 recovery used
ordinary resume after its recorded amendment. The initial project-v1 source
and original guard remain preserved.

Preparation evidence:

- launch.json: authoritative paths, environment, commands and hashes.
- configuration.json: complete immutable training recipe.
- preflight.json: passed card/schema/catalogue/native-build admission.
- schedule-audit.json: 100,000 unique seeds, exact partitions/opponent quotas.
- launch-tests.log: 14 passing local tests for transfer/configuration, unchanged
  PPO settings, disabled evaluations and schedule invariants. Tests initialized
  temporary models without collecting games or applying optimizer updates.
- prepare-stdout.log: confirms absent output/guard and unchanged parent.

At preparation time, a full GPU training round with the new inputs had not yet
been tested. Earlier local acceptance and its limitations are recorded in the
parent README. The user disabled all policy-strength evaluations for this
training run, including a final panel, so training completion reported only
training and holdout metrics. Those metrics alone do not establish improved
play. The separately authorized post-training comparison is linked above.

The user-authorized launch used this command on Raytracer:

~~~sh
tmux new-session -d -s hartford-identity-100k \
  '/bin/bash /home/altf4/benchmarks/hartford-identity-ppo-20261102/run-launch-v1/launch.sh'
~~~

This historical initial-launch command was executed once. The run is complete;
do not repeat it against the preserved output. A reproduction needs a new
output and guard/control directory with explicit configuration.

Initial launch validation: checkpoint 1 committed all 1,000 games (800 training /
200 holdout), both PPO epochs passed without rejection, GPU update completed
in 42.6 seconds, and frozen-source/engine/checkpoint/optimizer checks passed.
See launch-verification.json for the preserved first-round evidence.

Recovery: a mandatory exile cycle with an unplaceable Sawblades trigger reached
the decision cutoff in game 42776. The reviewed repair (RustMage PR #572) passed
10 focused and 3,024 native tests and full guarded release validation. The same
failed identity became a natural rules draw after 196 decisions. All 999 other
round-43 outcomes, checkpoints 0–42 and the complete original attempt/archive
were preserved. Ordinary resume reused all 1,000 outcomes without collection
inference, committed checkpoint 43, and continued training. No policy-strength
evaluations were run during recovery. See
recovery-inert-trigger-v1/resume-verification.json.

Completion audit verified all 100,000 scheduled identities, seeds, deck/seat/start
assignments, opponent versions and checkpoint hashes; all 101 saved models,
both optimizer histories and saved RNG states; current and historical frozen
source/runtime provenance and the copied/original parent. It also rehashed all
2,172 files in the original recovery archive and verified equality of the 999
reused successful round-43 outcomes. No successful game was replayed. Routine
rollout retention had pruned older trajectory payloads; their pinned committed
game summaries and checkpoints remain intact.

Elapsed time was about 7 hours 9 minutes including the repair interval; total
guarded runtime was about 5 hours 47 minutes. Mean collection time was 148.0
seconds per round (round 43 reused already-collected games), mean PPO update
43.0 seconds, and mean holdout measurement 4.3 seconds. The final holdout value
MSE changed from 0.6411 to 0.6191 across the last update; final training exact
KL was 0.00267. Five epoch proposals were rejected by unchanged guards across
the run; all recorded model/optimizer metrics are finite. Minimum observed
physical MemAvailable was 2.47 GiB, with no memory-guard breach. There were
3,374 wall-clock timeout draws and three turn-limit draws; the original failed
mandatory-loop attempt and repaired rules draw remain explicitly documented.
These training and holdout metrics do not measure policy strength.

The hourly health automation was removed after the passing completion audit.
The completed output remains preserved; no continuation is part of this
experiment record.
