# CPU self-play with terminal-reward reinforcement learning

`examples.train_selfplay` trains the existing structured Keras network on both
seats of Modern Burn self-play, then evaluates its saved checkpoint against the
fixed scripted Burn baseline. All network inputs and actions remain numerical.
Both players receive only their own observations and retain independent recurrent
state, action RNG and visible-event history. The model weights are shared.

## Objective and update schedule

The only game reward is the verified terminal result: win +1, loss -1, draw 0.
Every sampled selection receives its player's eventual result (discount 1).
Truncation, timeout and missing outcomes do not supply a win/loss reward.

The learner is a Monte Carlo actor-critic. For each sampled action, its advantage
is terminal return minus the value predicted during collection. The policy loss
is `-log_probability * stop_gradient(advantage)`. The critic fits terminal return
with squared error, weighted 0.5. An entropy bonus weighted 0.01 encourages
exploration; it is a regularizer, not an intermediate game reward. Adam uses
learning rate 0.0003 and global gradient norm clipping at 1.0 by default.

One completed self-play game produces one optimizer update. Weights stay fixed
through collection and all training minibatches; minibatch gradients are
accumulated before applying the update. Both players' trajectories participate,
with opposite returns for a decisive game. Candidate masks are retained, as are
all conditional integer/distribution selections. The critic is fitted once per
external decision, rather than once per internal digit.

Rollouts save the exact numerical inputs and incoming recurrent states used to
sample actions. Recurrent states are detached between decisions: gradients pass
through the current decision and its visible-event GRU, but not backward across
the entire game. This is a practical first CPU implementation; learning long
memory strategies would benefit from sequence replay and longer truncated BPTT.
Training minibatches weight sampled selections equally.

## Run

Start local XMage, then use uv:

```sh
uv run --no-sync python scripts/xmage-local.py start
uv run --no-sync python -m examples.train_selfplay \
  --games 12 --evaluation-games 20 --seed 41 \
  --directory .local/training/cpu-first
```

This command forces CPU execution. In this workspace uv is also available at
`.local/uv/uv`. The directory contains `initial.keras`, `trained.keras`, their
semantic manifests, `training.json`, `evaluation.json` and `evaluation.jsonl`.
The initial checkpoint is saved before the first gradient update. Model
checkpoints preserve inference weights and vocabulary; optimizer state is not
saved and this runner does not resume interrupted optimizer sessions.

The baseline evaluator can independently load a checkpoint:

```sh
uv run --no-sync python -m examples.burn_baseline \
  --opponent neural --load .local/training/cpu-first/trained.keras \
  --episodes 20 --seed 41 --output .local/training/cpu-first/evaluation.json
```

Evaluation alternates seats and never updates weights. Its `optimizer_steps: 0`
means zero updates **during evaluation**; `training.json` records training steps.
Stock XMage game shuffles are not seeded. The model/action seed does not make
matches reproducible. Existing untrained evaluation yielded 4 wins out of 100;
a small trained evaluation has wide uncertainty and cannot by itself establish
improvement. The fixed baseline is never used to generate training actions or
rewards. The current default registry annotates all 26 Burn identities; the
first run below predates that expansion and used the original small card pool.

## Small-scenario learning diagnostics

`examples.train_scenarios` trains the full-sized Keras policy on synthetic
one-decision tasks using the real XMage choice adapter and numeric encoder:

- Complete Boros Charm's red/white payment with the missing color.
- Aim lethal Lightning Bolt at the opponent, rather than self or cancel.
- Choose Boros Charm damage when no creature is attacking, or double strike
  when an unblocked Swiftspear needs it for lethal. The scripted continuation
  accounts for the Charm cast's prowess trigger and normal combat damage.

These are terminal task-success rewards (+1/-1), **not intermediate rewards
for real games**. Real-game training continues to use only win/loss/draw.
The diagnostic oracle is separate from all model inputs. Training samples an
action, receives its reward, and performs an on-policy actor-critic update;
there are no demonstration labels or supervised policy targets.
An explicit `--supervised-control` flag instead uses oracle action labels to
test representational capacity; its reports are labeled supervised and must not
be presented as RL results. Neither diagnostic modifies the real-game trainer.

Option order, UUIDs, seat order and nuisance state vary. Evaluation uses new
states, disjoint life/power values and, for mana, Elegant Parlor instead of
the training lands Sacred Foundry and Inspiring Vantage. Both required colors
and both Charm branches are scored separately. An additional fresh evaluation
uses the training distribution to distinguish learning from transfer failure.
Reported success probability is the policy's probability mass on successful
actions; greedy accuracy is reported separately. The pass gate requires at
least 90% success probability and 95% greedy accuracy in every held-out branch.

```sh
UV_CACHE_DIR=.local/uv-cache KERAS_HOME=.local/keras TF_ENABLE_ONEDNN_OPTS=0 \
  .local/uv/uv run --no-sync python -m examples.train_scenarios \
  --seeds 41 42 43 --updates 1500 --candidate-interactions --player-readout \
  --require-pass --directory .local/training/scenario-check
```

The two architecture flags are opt-in experiments: explicit state/action
products in the scoring head and direct access to the self/opponent numeric
player rows. Defaults for real self-play are unchanged, and the flags serialize
with each checkpoint. Omit them for the original scorer. Each run saves its
configuration, initial/final metrics, update history and reload-verified model.
Per-branch development metrics in `progress-SEED.json` reveal regressions that
an aggregate reward can hide; their states are separate from final evaluation.
Use a new directory for each experiment. Synthetic success does not establish
full-game strength, multi-step planning, or successful recurrent credit assignment.

### PPO diagnostic

Select `--algorithm ppo` to reuse each frozen rollout for up to four optimizer
epochs. `PPOTrainer` consumes numeric inputs, sampled actions, behavior log
probabilities, behavior values, and externally supplied returns. It leaves the
policy/value model interface unchanged, including for a future search caller.
It does not implement MCTS or a forkable simulator.

PPO uses a clipped probability-ratio surrogate (clip 0.2), squared-error value
loss (weight 0.5), entropy regularization (0.01), and gradient clipping (1.0).
Advantages are return minus frozen behavior value, normalized across genuine
choices in the rollout when their variance is nonzero. Forced actions contribute
no policy or entropy loss. The value head is trained once per external decision.
After each epoch, a sampled KL estimate on the entire rollout stops additional
epochs if it exceeds 0.02. This is a drift check, not a hard bound or rollback.
Old probabilities, values, and advantages are never refreshed during reuse.

```sh
UV_CACHE_DIR=.local/uv-cache KERAS_HOME=.local/keras TF_ENABLE_ONEDNN_OPTS=0 \
  .local/uv/uv run --no-sync python -m examples.train_scenarios \
  --algorithm ppo --ppo-epochs 4 --ppo-clip .2 --target-kl .02 \
  --learning-rate .001 --seeds 41 42 43 --updates 1500 \
  --candidate-interactions --player-readout --require-pass \
  --directory .local/training/scenarios-v4-ppo
```

The comparison holds architecture, learning rate, environment-example budget,
evaluation states, seeds and success thresholds fixed against the earlier
actor-critic experiment. PPO uses more optimizer steps and additionally
normalizes advantages; this compares the complete PPO update recipe, not an
isolated clipping ablation. Optimizer permutations use a separate RNG so they
do not change scenario generation. Reports include optimizer-step counts,
KL stops, and sampled progress metrics. Checkpoints contain inference weights;
the scenario runner does not resume optimizer state.

PPO is currently connected to the diagnostic runner only. Real XMage self-play
still uses the existing actor-critic trainer and terminal win/loss/draw rewards.
Rollout recurrent states remain detached; sequence replay must be addressed
before claiming that repeated PPO updates train long-term memory reliably.

#### Matched PPO results — October 15, 2026

All three runs completed and checkpoints reloaded successfully. None passed
the unchanged gate, versus one of three for the earlier actor-critic recipe.
The command exited 1 because `--require-pass` detected those failures, not
because training crashed. Initial evaluation metrics matched exactly for all
three paired seeds.

| Seed | Need white | Need red | Bolt target | Charm damage | Charm double strike |
| --- | ---: | ---: | ---: | ---: | ---: |
| 41 | 81.4% | 98.3% | ~100% | 99.8% | 69.5% |
| 42 | 50.0% | 50.0% | 50.0% | 72.7% | 27.2% |
| 43 | 98.3% | 99.1% | 50.0% | 0.3% | 99.7% |

These are held-out correct-action probabilities. Seed 41 had perfect greedy
accuracy on every branch, but insufficient sampling confidence. Development
metrics show late regressions in all three runs; for example, seed 43 had
learned both Charm branches before losing the damage branch. The final
checkpoint was evaluated without selecting a favorable earlier checkpoint.

Each algorithm received 216,000 fresh examples across its three seeds. PPO
performed 15,856 optimizer steps versus 4,500 for actor-critic, taking about
22.5 minutes on CPU. With the matched learning rate of 0.001 and rollout size
48, this experiment provides no evidence that PPO improves reliability.
It does not establish that PPO is generally worse. Larger rollout batches and
a smaller learning rate are reasonable next experiments; no reward shaping is
needed to test them. PPO remains opt-in in the scenario runner.

Full configuration, metrics, progress history, and the paired control are in
[the PPO benchmark](benchmarks/ppo-scenarios-2026-10-15.json).

### Stabilizing synthetic learning

`--stable-heads` keeps the same entity/semantic/recurrent model and numeric
features, but normalizes candidate and query vectors before their products,
uses GELU instead of the saturated tanh scorer, and normalizes the hidden
scoring vector. Normalization has fixed scale. The value output is linear so
the critic can recover from extreme predictions. These changes serialize with
the model and do not change existing checkpoints or real-game defaults.

`--standardize-inputs` fits frozen per-feature means and scales from 512
additional training-distribution observations. It uses valid entity/candidate
rows only, without rewards or held-out states, and caps amplification at 10.
Constant features retain unit scale. This makes small numeric differences,
such as one floating mana, visible alongside the larger categorical and learned
features. Statistics are saved inside the Keras model and never adapted during
evaluation. These calibration observations are reported separately from RL
training examples.

`--advantage-scale-floor .1` limits amplification when centering/scaling
advantages. Otherwise, small differences in an almost-correct critic can be
rescaled to unit-size actor updates even when every sampled outcome succeeds.
The original PPO benchmark used a near-zero floor; the current default is 0.1.

`--trust-guard` measures exact categorical KL across every offered action after
each proposed epoch, against the frozen behavior distribution. It checks both
mean KL and the worst state. A rejected epoch restores all model variables,
Adam moments, iteration counter, and learning rate before retrying with half
the step size. At most six backtracks are allowed; if no proposal is acceptable,
the last accepted state is retained and further epochs stop. A new rollout
starts with the configured learning rate again. This extends the clipped PPO
objective with backtracking; it is not an implementation of TRPO. Limits apply
to collected states, not every possible game state.

The diagnostic reports accepted-step counts, rejected proposals, zero-step
rollouts, and maximum accepted KL over the whole run. The policy and value
interface remains suitable for a later search caller. Rewards still come only
from terminal task success; no demonstration targets or intermediate rewards
are used by this recipe.

```sh
UV_CACHE_DIR=.local/uv-cache KERAS_HOME=.local/keras TF_ENABLE_ONEDNN_OPTS=0 \
  .local/uv/uv run --no-sync python -m examples.train_scenarios \
  --algorithm ppo --stable-heads --standardize-inputs --isolate-value \
  --candidate-interactions --player-readout \
  --trust-guard --target-kl .01 --max-state-kl .05 \
  --learning-rate .0003 --final-learning-rate .000003 \
  --advantage-scale-floor 1 --batch 96 --updates 600 \
  --distribution diverse --identity-dropout .5 \
  --seeds 81 82 83 84 85 --test-cases 500 --retention-checks 3 --require-pass \
  --directory .local/training/scenarios-stable-check
```

Development experiments use `--development-only`, which avoids the held-out
distribution. The retention gate requires every branch to pass at each of the
last three development checkpoints as well as the final held-out evaluation.
No favorable earlier checkpoint is selected. This is a test of one-decision
learning and retention, not of full-game strategy or recurrent credit assignment.

`--outcome-noise .2` independently flips 20% of terminal task outcomes. A
correct action then wins with probability 0.8, and an incorrect action wins
with probability 0.2. Only the sampled outcome reaches the learner. Evaluation
reports correct-action probability separately from expected win probability,
which is `0.2 + 0.6 * correct_action_probability`. Thus perfect decisions still
lose 20% of these synthetic episodes. Value calibration is measured against
the expected return, rather than requiring the critic to predict random luck.
This is a controlled noise test, not a model of Magic's complete uncertainty.

`--distribution diverse` broadens the training states without changing the
held-out generator. Own life varies below and above the held-out range, and
Charm's attacking/nonattacking examples have overlapping powers. The held-out
life and power values remain excluded from the corresponding training fields.
This removes shortcuts such as always targeting the player with less life or
using a fixed creature-power threshold instead of combat state.

`--identity-dropout .5` masks some current card IDs and the ability IDs of
candidates bound to those sources, while retaining every numeric cost/effect,
color, stat, relation and legal-action mask. It applies before sampling actions;
the same masked inputs and behavior probabilities are retained for every PPO
epoch. Evaluation uses complete identities. The mask RNG is separate from the
scenario, outcome and optimization RNGs. This encourages use of shared
semantics when an identity's embedding has received no training.

`--isolate-value` is a controlled gradient ablation: the critic receives a
detached copy of the actor's state representation. Its value head still learns
from terminal outcomes, but its loss cannot update the shared encoder. Model
capacity and forward computation are unchanged. The critic consequently relies
on features learned by the actor; this experiment does not establish that this
tradeoff will be preferable for full games. The stronger
`--advantage-scale-floor 1` also avoids amplifying small, nonconstant advantage
residuals to unit scale.

`--final-learning-rate .000003` linearly decreases the configured learning
rate from 0.0003 over the run. Backtracking may reduce a particular proposal
further, but the next rollout restarts at its scheduled rate. This tests whether
smaller late updates preserve decisions after the policy has learned them,
despite continuing terminal-outcome noise. Omitting the flag retains the
constant rate used by the controls. Logs record the scheduled and actually
accepted proposal rates separately.

The suite defaults to the validated recipe above, with clean seeds 81–85 and
noisy seeds 91–93. It uses isolated CPU processes, each launched through uv. It
writes its plan before starting, keeps every failed run, and evaluates final
checkpoints with 500 held-out states per task and the same retention gate:

```sh
UV_CACHE_DIR=.local/uv-cache KERAS_HOME=.local/keras TF_ENABLE_ONEDNN_OPTS=0 \
  .local/uv/uv run --no-sync python -m examples.train_scenario_suite \
  --directory .local/training/scenarios-reliable-check --workers 3
```

Reproduce saved-checkpoint evaluations without training:

```sh
UV_CACHE_DIR=.local/uv-cache KERAS_HOME=.local/keras TF_ENABLE_ONEDNN_OPTS=0 \
  .local/uv/uv run --no-sync python -m examples.verify_scenario_suite \
  .local/training/scenarios-reliable-check \
  --output .local/training/scenarios-reliable-check/reload-verification.json
```

The verifier checks frozen source fingerprints and model flags, then regenerates
both the held-out and fresh training-distribution evaluations, including value
calibration. It refuses an incomplete suite or changed sources.

#### Reliable recipe results — October 15, 2026

All eight fresh confirmation seeds passed the unchanged final and retention
gates. Each trained the full-size 1,432,368-parameter model on 57,600 sampled
one-decision terminal outcomes, plus 512 unlabeled calibration observations.
The final checkpoint was used in every case.

| Terminal outcomes | Seeds passing all gates | Held-out greedy accuracy | Lowest branch-mean correct-action probability |
| --- | ---: | ---: | ---: |
| Deterministic | 5/5 | 7,500/7,500 | 99.91% |
| 20% independent outcome flips | 3/3 | 4,500/4,500 | 99.87% |

The noisy policies averaged 79.97% expected wins against an 80% ceiling, with a
mean critic prediction of 0.594 (the optimal expected return is approximately
0.6). Correct decisions can still lose in this test. The learner received only
the sampled terminal result; correct-action labels were used solely to evaluate
decision quality. These are synthetic outcomes, not full MTG match win rates.

The investigation found several interacting problems. The old scorer had
82–93% saturated tanh units, but a lower-rate control with the old head also
learned, so saturation was not established as the sole cause. Counterfactuals
showed spurious dependence on land identity and unrelated life totals. Broader
training states and identity masking improved that transfer. A paired noisy
development run passed retention after isolating value gradients; its initial
metrics were identical to the shared-gradient control. Isolation alone still
had a transfer gap. Decreasing the learning rate reduced late confidence drift
and passed the established transfer regression cases. Routing untrained IDs to
unknown tokens did not resolve the gap and was not adopted.

An earlier stability attempt passed only 5/8 complete gates: one clean seed
missed a retention checkpoint, and two noisy seeds failed transfer. Those
failures and the valid development controls remain in the benchmark. The final
recipe was frozen before the new eight-seed confirmation. Both confirmation
cohorts use the same held-out distribution inspected during development; the
held-out generator itself is unchanged, with fresh seeds and observations.

All accepted guarded updates stayed within mean KL 0.01 and maximum-state KL
0.05 on their rollout states. All eight checkpoints reproduced all reported
policy and value metrics exactly over 24,000 regenerated evaluation states
after reload. All 219 unit tests passed. Real XMage training remains unchanged:
terminal win/loss/draw rewards and the existing actor-critic runner. This result
establishes learning and retention on these one-decision task families; it does
not test planning, full-game strength or recurrent credit assignment.

The full results, negative controls, configurations, source hashes and reload
checks are in [the reliability benchmark](benchmarks/reliable-scenarios-2026-10-15.json).
Actual checkpoints are under `.local/training/scenarios-reliable-confirmation/`.

### CPU scenario results — October 15, 2026

The full-size model can learn these synthetic choices, but learning is not yet
reliable across seeds. With both optional heads enabled, 1,500 updates of 48
fresh one-step examples (72,000 examples per seed) produced these held-out
success probabilities. These are exact policy probabilities, not sampled wins.

| Seed | Need white | Need red | Bolt target | Charm damage | Charm double strike | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 41 | 98.4% | 98.2% | ~100% | 62.7% | 37.6% | No |
| 42 | 89.0% | 91.8% | ~100% | ~100% | ~100% | No |
| 43 | 93.0% | 92.8% | ~100% | 99.2% | 98.0% | Yes |

Seeds 42 and 43 selected the correct action by argmax on every held-out case.
Seed 42 nevertheless failed the preselected 90% probability threshold for
white mana. Seed 41 had learned both Charm branches after 600 updates but lost
that distinction by 1,500, showing that aggregate progress can hide forgetting.

The original scorer, tested at the same 1,500-update budget with seed 41, failed
both conditional tasks. Isolated 600-update RL tests with the optional heads
passed Charm (at least 99.7% per branch). Isolated mana achieved perfect greedy
accuracy but only 73.3–80.0% success probability on the unseen land, compared
with 87.7–96.7% on fresh states from the training distribution. A separate
600-update supervised capacity control also failed the full gate; its results
are explicitly distinguished from RL.

The evidence supports learning state-dependent decisions, not robust learning
or improved full-game play. The architecture changes remain opt-in and real
game rewards remain terminal win/loss/draw only. All saved checkpoints passed
a reload/logit comparison. Configurations and per-branch results are retained
in [the scenario benchmark](benchmarks/scenario-learning-2026-10-15.json).

## Parallel 1,000-game experiment

`examples.train_parallel` uses concurrent XMage games, policy worker threads and
one shared compiled TensorFlow inference service. Requests wait up to 5 ms to
form batches of at most 16. Each game and player has separate recurrent state
and action RNG; history buffers are keyed by game and actor. Model inputs remain
numerical, including the expanded Burn annotations.

```sh
UV_CACHE_DIR=.local/uv-cache KERAS_HOME=.local/keras TF_ENABLE_ONEDNN_OPTS=0 \
  .local/uv/uv run --no-sync python -m examples.train_parallel \
  --games 1000 --workers 8 --round-games 16 \
  --directory .local/training/burn-parallel-1000
```

The new runner freezes weights while collecting 16 completed games, streams
compressed trajectories through training minibatches, accumulates gradients and
then applies one optimizer update. The last round contains eight games, for
63 updates total. Rewards and detached recurrent training follow the objective
above. Inference and gradient calculation use compiled graphs with dynamic
batch, object, candidate and history lengths.

Baseline evaluation runs 100 games before training, 20 after crossing each of
250/500/750 completed training games (normally 256/512/752), and 100 at the end.
Seats alternate. These evaluation games never contribute gradients. Reports
include wins/losses/draws and Wilson 95% intervals for win rate. The old 4/100
result is historical; this experiment evaluates its own fresh initial model.

Failed or truncated games are excluded and replaced. Three failures within one
collection phase stop the run for diagnosis. `failures.jsonl` records the last
visible state and choices. Successful training rollouts are deleted only after
their round checkpoint is committed; paths retained in reports record their
provenance. Failed rollouts remain available. Evaluation traces under `traces/`
retain choices, conditional probabilities, responses, actor-visible states and
terminal results, without revealing the opponent's hidden cards.

`progress.json` reports the current phase. `report.json` summarizes committed
training rounds and evaluations. Each `checkpoints/round-NNNN/` stores Keras
weights, the semantic manifest, Adam variables, RNG state and run configuration;
`checkpoints/latest.json` points to the latest committed round. Resume with the
same command plus `--resume`. An unfinished round is replayed from its beginning;
completed optimizer updates are preserved. XMage's shuffles remain unseeded.
Do not run two coordinators against the same output directory.

For a throughput probe, use `--benchmark --benchmark-workers 4 8` with a separate
directory. The local server currently permits ten game threads; a sixteen-game
probe needs a larger server limit. Worker concurrency and mean inference batch
size are measured separately: Python preprocessing can limit scaling despite
concurrent games. The benchmark excludes optimizer work, so its games/hour is
not an end-to-end training estimate.

The October 14 cached-state probe measured 19.58 decisions/s with four workers
and 20.91 with eight. Eight workers averaged 2.70 requests per inference batch
and used 686 MiB peak Python-process RSS (excluding the Java server and proxy).
All twelve benchmark games completed without failures. Eight workers were
selected for the first 1,000-game run; the small throughput difference should
not be treated as a precise scaling estimate.

## First CPU run — October 14, 2026

Completed 12 self-play games, 5,168 external decisions and 5,170 sampled
selections, producing 12 optimizer updates. All training games completed
naturally. Both initial and final checkpoints reload successfully; 1,128,904 of
1,257,346 parameters changed and all saved weights are finite.

The frozen trained model won **2/20 (10%)** against the baseline, compared with
the earlier untrained result of **4/100 (4%)**. Evaluation alternated seats evenly
and had zero draws, failures or truncations. These are different unseeded games;
20 evaluation games are too few to establish that training improved strength.
This run validates the collection, terminal reward, gradient update, checkpoint
and evaluation pipeline. It does not establish competitive play.

Full results and checkpoint hashes are in
[`benchmarks/cpu-selfplay-2026-10-14.json`](benchmarks/cpu-selfplay-2026-10-14.json).
The trained model is `.local/training/cpu-first/trained.keras`, accompanied by
its required `.manifest.json` file. The initial checkpoint is retained beside it.
These historical checkpoints use feature version 2 and cannot be loaded with
the current version-6 Burn vocabulary without an explicit migration. They are
preserved as experiment artifacts; use a new output directory for a fresh run
with the updated annotations.

## Synthetic Burn mulligans and bottom-card choices

The opening-hand oracle is a deliberately simple user-specified heuristic:

- With four cards or fewer, keep anything.
- Otherwise, keep every two- or three-land hand.
- Keep a one-land hand if it contains at most two spells with mana value exactly 2.
- Mulligan all other hands. Cards with mana value 3 are not counted as two-mana spells.

This is the synthetic teacher, not an optimal Magic strategy or a fallback in
`NeuralPolicy`. `baselines/burn_mulligan.py` defines the rule independently of
network inputs. Both actions receive terminal task-success rewards, never a
reward proportional to lands, mana, or life in real games.

`--tasks mulligan mulligan_bottom` selects seven keep/mulligan branches and two
bottom-card branches. Two- and three-land hands include expensive spell-heavy
hands so a model cannot substitute a blanket low-curve rule. Training uses four-,
five-, and seven-card keep prompts; held-out prompts use three and six cards.
The legacy distribution additionally withholds card identities. Diverse and
native distributions draw without replacement from the full pinned main deck
and hold out hands instead. Player/candidate ordering and UUIDs vary. The oracle
and branch name are never encoded.

The bottom objectives are separate simple exercises: remove excess lands while
preserving at least two, or remove the lone two-mana spell while preserving two
lands and one-mana plays. Equivalent copies are all accepted. These are not
additional user-specified mulligan rules. `advance_bottom` removes each chosen
card and updates hand/library counts, pending count, and decision generation;
`examples.evaluate_openings` exercises complete sequences with actor-local
history and carried state, scoring success only after the last choice.

Actual XMage 1.4.62 prompts were captured in a setup-only session with two
redraws followed by keep. This server bottoms cards after each redraw, before
asking about another mulligan. The stripped fixture is
`tests/fixtures/xmage_mulligan_prompts.json`. Feature version 6 supplies a finite
prompt-purpose enum, next mulligan hand size, and bottom-cards remaining. Raw
prose stays outside the network. Earlier checkpoints remain preserved and fail
manifest validation against the expanded feature schema.

The earlier seed-212 PPO experiment below did not pass its generalization gate;
it is retained as a negative diagnostic, not a validated training recipe.

```sh
UV_CACHE_DIR=.local/uv-cache KERAS_HOME=.local/keras TF_ENABLE_ONEDNN_OPTS=0 \
  .local/uv/uv run --no-sync python -m examples.train_scenarios \
  --directory .local/training/mulligan-four-card-floor-seed212 --seeds 212 \
  --tasks mulligan mulligan_bottom --algorithm ppo \
  --candidate-interactions --player-readout --stable-heads --standardize-inputs \
  --isolate-value --trust-guard --learning-rate .0003 \
  --final-learning-rate .000003 --advantage-scale-floor 1 \
  --target-kl .01 --max-state-kl .05 --batch 96 --updates 300 \
  --test-cases 300 --identity-dropout .5 --distribution diverse --require-pass
```

Use a new directory when repeating the experiment. Task success does not
establish full-game strength. The original three tactical tasks remain available
via `--tasks mana bolt_target charm_mode`; the default task list now includes
both new opening tasks.

## Five-skill Burn curriculum

`examples.train_curriculum` trains one model on five families of short,
multi-decision puzzles:

| Family | Terminal objective |
| --- | --- |
| `openings` | Follow the user keep rule, redraw seven, bottom the London cards, and reach a correct keep. |
| `lands_and_casting` | Play a land if needed, pay for an untapped shock land if needed, cast and resolve Bolt at the opponent; pass in the unpayable branch. |
| `mana_colors` | Finish paying an already-cast Charm using the actual remaining red or white debt. |
| `lethal` | Resolve one or two burn spells to win this turn. Self-targeting, creature targets, and premature passing can fail. |
| `creatures_and_attacking` | Play a land, deploy Guide or Swiftspear, advance into combat, declare the eligible creature, and finish the attack. |

Opening keeps use the exact requested teacher: keep anything at four or fewer;
otherwise keep two or three lands, or one land with at most two MV-2 spells.
The additional bottoming teacher preserves two lands where possible, removing
excess lands first and then the highest printed-mana-value spell. This is a
deliberately limited puzzle teacher, not an optimal bottoming claim.

Observations use the production XMage choice decoder and numeric encoder,
captured sideboard/helper layouts with stable IDs throughout each episode,
shuffled candidate order, actual observation history, and carried recurrent
state. Main-deck identities occur in training; evaluation reserves new seeds,
opening hand sizes, life totals, and combinations rather than withholding whole
card identities. Teachers and task/branch names never enter model inputs.

The warm start is **supervised imitation**, using all equivalent valid teacher
actions as positives. It is not presented as reward-only learning. The optional
PPO phase samples the network with no teacher fallback and supplies only one
terminal +1/-1 outcome for each entire puzzle. Full-game training still uses
verified server win/loss/draw rewards only. Recurrent states are detached between
decisions during optimization; this runner does not implement BPTT.

`--balance-prompts` gives each task equal supervised weight and then balances
prompt groups within each task, so long sequences do not drown out short choices. This is demonstration-loss
weighting, not an intermediate RL reward. `--calibrate-context` estimates factual
context statistics from training observations only. It compensates the model's
two context-consuming dense layers so initial logits, value, and memory are
preserved; subsequent learning uses normalized numeric coordinates. Tests cover
both raw and already-standardized warm starts. It adds no strategy labels or new
input fields.

```sh
UV_CACHE_DIR=.local/uv-cache KERAS_HOME=.local/keras TF_ENABLE_ONEDNN_OPTS=0 \
  .local/uv/uv run --no-sync python -m examples.train_curriculum \
  --directory .local/training/five-skills-repeat --seed 415 \
  --load .local/training/five-skills-input-414.keras \
  --calibrate-context --balance-prompts --learning-rate .0001 \
  --warmup-rounds 40 --ppo-rounds 2 --round-episodes 80 \
  --eval-cases 100 --evaluate-every 20 --require-pass
```

Use a fresh directory. Reports retain initial, warmup, final held-out, and final
IID results for both greedy and normal sampled policies. The pass gate requires
at least 90% complete-episode success in every family for both modes and both
final distributions. Intermediate development evaluations are also retained.
`policy.keras` plus its manifest is loadable by the normal neural policy;
`warmup.keras` preserves the pre-PPO model. The report records source hashes,
checkpoint hash, optimizer counts, and batching statistics.

This command reproduces the local warm-start experiment. Omit `--load` to start
a fresh model; inherited training must not be counted as zero when comparing
sample efficiency. `--require-pass` exits unsuccessfully if any family misses
the threshold, while retaining its checkpoint and complete evaluation reports.
`examples.diagnose_curriculum` evaluates a saved model on a separate seed set and
records the first teacher deviation for each failed episode, without updates.

These are controlled continuations with scripted opponent passes. Combat has
no opposing blockers, Guide reveal and Eidolon damage triggers are omitted, and the puzzles do not
cover general spell interactions or entire games. Passing them establishes
these elementary skills, not full-game XMage strength.

### Corrected experiment: 2026-10-15

The final warm-start run is `.local/training/five-skills-corrected-415`.
It adds 3,200 supervised puzzle episodes and 160 terminal-only PPO episodes to
an existing synthetic checkpoint. This is not a claim that a fresh model learned
the skills from only those additional episodes.

On a separate paired seed set, normal sampled actions completed these puzzles:

| Family (100 episodes each) | Fresh untrained model | Trained model |
| --- | ---: | ---: |
| Openings | 30% | 76% |
| Lands and casting | 10% | 100% |
| Mana colors | 12% | 99% |
| Lethal | 16% | 100% |
| Creatures and attacking | 0% | 100% |

Independent greedy completion was 86% for openings and 100% for all other
families. No trained independent episode reached the 32-decision bound. Of the
24 sampled opening failures, the first deviation was a keep/mulligan choice in
23 cases and a bottom choice in one. The all-family 90% graduation gate **did
not pass**; opening decisions still need work. The main final held-out and IID
evaluations also retain this failure rather than reporting only easy branches.

Offline replay got 5/6 captured live mulligan views and 174/200 valid hands
inserted into a captured native template correct with greedy choices. Those
are individual decisions with fresh per-case history, not live game results.

Earlier prototype measurements were discarded from final validation after
finding incorrect fetch-land types in redraws and incorrect creature facts.
Regression tests now check generated card types and printed mana values through
complete teacher trajectories. The final repository suite passes 254 tests.

The loadable model is
`.local/training/five-skills-corrected-415/policy.keras` with its adjacent
manifest. Full results, source/checkpoint hashes, controls and limitations are
recorded in [the benchmark](benchmarks/five-skill-curriculum-2026-10-15.json).

### Manual payment and cancellation recovery

The subsequent XMage comparison exposed a missing adapter action: when mana was
already floating but locked, the model could only cancel a payment. The decoder
now offers each positive own-pool color through `sendPlayerManaType`. The weak
baseline understands the same protocol action. This mechanical fix is necessary
before training can solve the observed cast/target/cancel/recast loops.

Four additional synthetic sequence families are available through
`examples.train_curriculum --tasks`:

| Task | Controlled continuation |
| --- | --- |
| `spend_pool` | Pay Bolt, partially paid Charm, or Skullcrack from floating mana, matching colored and generic debt. |
| `tap_then_spend` | Tap a source, select its color when needed, then explicitly spend the produced mana. |
| `cancel_retry` | Cancel an unpayable cast, recover the card, and pass instead of retrying it; payable counterexamples must resolve. |
| `blaze_targets` | Complete player and creature targets and pay RR. Kill a damaged blocker, or target one's own Swiftspear to trigger prowess and win. |

These use the production decoder/encoder, stable native auxiliary objects,
random UUIDs and shuffled menus. Teacher labels remain external. Cancellation
returns the card to hand in recovery puzzles; it is not universally labeled a
bad action. Wrong-color pool clicks can leave the prompt unchanged, allowing
evaluation to detect actual no-progress loops. The 32-decision puzzle limit is a
defined synthetic failure, never an invented server loss. Reward is supplied
only at the end; supervised demonstration loss is identified separately.

Blaze scenarios fix landfall to false, start the opponent at three life, and
script both players passing into combat. Blaze deals one to each target; prowess
makes Swiftspear 2/3. With no blocker, targeting that Swiftspear leaves it alive
and its attack wins. With a 2/2 blocker already marked with one damage, removing
the blocker wins instead. These are controlled targeting examples, not a general
combat engine or a rule against targeting one's own creatures.

Half the Blaze cases instead start during targeting in ordinary nonlethal
positions, varying life totals, hand contents and opposing creatures. Their
terminal task goal is to finish the cast at the opponent and a legal creature;
they do not require the opponent to reach zero life. This is explicitly a
synthetic mechanical objective, not a real-game intermediate reward. It avoids
teaching the mistaken association that only lethal targeting prompts should be
completed. One damage does not automatically kill an undamaged 2-toughness
creature in these continuations.

`--balance-actions` additionally balances the teacher's winning primitive kinds
within each task and prompt. For example, priority examples that cast a payable
spell (UUID) receive the same total demonstration weight as examples that pass
with an unpayable spell (boolean). This affects supervised loss only; teacher
classes are never observations or rewards.

For a bounded CPU warm start that also checks retention of previous skills:

```sh
UV_CACHE_DIR=.local/uv-cache KERAS_HOME=.local/keras TF_ENABLE_ONEDNN_OPTS=0 \
  .local/uv/uv run --no-sync python -m examples.train_curriculum \
  --directory .local/training/payment-repeat --seed 603 \
  --load .local/training/five-skills-corrected-415/policy.keras \
  --tasks openings lands_and_casting mana_colors lethal creatures_and_attacking \
    spend_pool tap_then_spend cancel_retry blaze_targets \
  --warmup-rounds 50 --ppo-rounds 0 --round-episodes 90 \
  --learning-rate .0001 --balance-prompts --eval-cases 40 --evaluate-every 25
```

This adds 4,500 supervised episodes; PPO is deliberately off for this diagnostic
run. Existing real-game reward semantics are unchanged. For an independent
evaluation, `examples.diagnose_curriculum` accepts the same `--tasks` list.

The follow-up uses the expanded Blaze distribution and balanced teacher classes:

```sh
UV_CACHE_DIR=.local/uv-cache KERAS_HOME=.local/keras TF_ENABLE_ONEDNN_OPTS=0 \
  .local/uv/uv run --no-sync python -m examples.train_curriculum \
  --directory .local/training/payment-balanced-repeat --seed 605 \
  --load .local/training/payment-curriculum-603/policy.keras \
  --tasks openings lands_and_casting mana_colors lethal creatures_and_attacking \
    spend_pool tap_then_spend cancel_retry blaze_targets \
  --warmup-rounds 40 --ppo-rounds 0 --round-episodes 90 \
  --learning-rate .0003 --balance-actions --eval-cases 20 --evaluate-every 20
```

The original 603 run used the narrower, lethal-only Blaze generator; its source
snapshot and measurements are preserved locally. Running the first command with
the current generator uses the expanded examples. The follow-up adds another
3,600 supervised episodes to that checkpoint; inherited training is not zero.

The pool command was also exercised in two baseline-versus-baseline XMage games
with `examples.burn_baseline --manual-mana --episodes 2`: both ended naturally,
using 19 explicit pool-spend responses. A second two-game smoke test with normal
automatic payment also ended naturally. These validate protocol mechanics,
not neural playing strength.

The saved follow-up is `.local/training/payment-balanced-605/policy.keras` and
its adjacent manifest. Across the two retained stages it adds 8,100 supervised
puzzles and 538 optimizer steps to checkpoint 415, with no additional PPO.
An independent paired set (seed 270001, 100 puzzles per family) produced:

| Family | Previous model, sampled | Saved follow-up, sampled |
| --- | ---: | ---: |
| Spend floating mana | 85/100 | 100/100 |
| Tap, then spend | 99/100 | 100/100 |
| Cancel/retry recovery | 56/100 | 94/100 |
| Searing Blaze targets | 58/100 | 100/100 |

The previous model was evaluated with the repaired decoder too, separating the
effect of training from the missing-action fix. Independent sampled recovery
timeouts dropped from 34 to zero; none of the 400 final independent sampled
puzzles timed out. Independent greedy recovery was 86/100, with 100/100 for each
other new family. Remaining independent errors passed on a payable spell with
floating red mana. A separate 20-case IID recovery set still had one sampled
timeout, so recovery is not fully solved. Openings also remain below the 90%
all-family threshold; the runner correctly records its overall gate as failed.

Fresh-history probes of two captured stuck prompts gave 99.0% probability to
spending red mana for Lava Spike, and 96.6% total probability to selecting a
creature for Blaze. These reconstruct prompt metadata and reset recurrent
history per snapshot; they are not a replay of the original policy state or
proof of optimal targeting. No neural full-game win-rate claim is made here.

All 263 repository tests pass. Configurations, source/checkpoint hashes,
retention measurements, negative prototypes and transfer limitations are in
[the payment benchmark](benchmarks/payment-curriculum-2026-10-15.json).
The earlier checkpoint-415 comparison was interrupted with three completed
losses; incomplete games were not relabeled as losses. A separate 20-game run
of checkpoint 605 finished 4–16, with no errors or cutoffs; see
[its benchmark](benchmarks/payment-baseline-20-2026-10-15.json).

### Goldfish competency metric

`examples.evaluate_goldfish` measures the evaluated player's **own turns until
a verified XMage win** against an inert opponent. The opponent has a legal
60-Mountain deck, always keeps, never plays a card, passes priority, and makes
deterministic required discards. It starts at the normal 20 life. The model
keeps control of its mulligans and all gameplay decisions. This is a basic
test of converting an uncontested hand into a win; it does not measure response
to interaction, defense or matchup strength.

The harness independently alternates seat and play/draw, and checks the actual
starting player. Pregame prompts are excluded from turn counts. A turn counts
once regardless of the number of priority prompts, and consecutive extra turns
would count separately. An opponent-turn kill is measured by the number of
own turns already started. The report includes play/draw breakdowns.

The primary mean and median are conditional on winning. Report them together
with the win rate, failure rate and cutoff counts, never alone: failed games have no defined
turn-to-win and are not assigned fabricated win turns. Win-by-turn rates retain
**every attempted game** in their denominator. Defaults permit 20 complete own
turns, 3,000 total decisions and 240 seconds per game. The next own turn stops
the game before any action; losses, draws, cutoffs and errors remain separate.
Self-inflicted defeats under normal Magic rules count as failed attempts too:
an inert opponent cannot attack or cast spells, but the agent can still kill
itself with its own cards. The diagnostic does not grant invulnerability.
The failure rate counts every attempt that did not finish in a verified win;
the breakdown preserves infrastructure errors separately from game defeats.

A Burn-specific stall guard also stops the attempt after 25 visits to an
identical visible decision state. The key excludes callback revisions and map
presentation order, while retaining turn, phase, players, visible cards, stack
order, offered choices and typed prompt metadata. State changes such as damage,
mana, cards moving or turns advancing produce different keys. This only marks
an evaluation cutoff; it does not concede a server loss or change training
rewards. `--repeat-limit 0` disables it for future decks that can use repeated
states productively. The 100-game calibration uses the same guard for both
controllers and starts fresh rather than mixing the earlier pilot measurements.

```sh
UV_CACHE_DIR=.local/uv-cache KERAS_HOME=.local/keras TF_ENABLE_ONEDNN_OPTS=0 \
  .local/uv/uv run --no-sync python -m examples.evaluate_goldfish \
  --load .local/training/payment-balanced-605/policy.keras \
  --directory .local/xmage/goldfish-neural-repeat \
  --games 20 --workers 4 --seed 702
```

For the deterministic Burn reference, use `--controller baseline`, omit
`--load`, and choose a fresh output directory. The model is frozen and its
checkpoint hash is checked; no optimizer is constructed. This diagnostic adds
no reward or intermediate training objective. Seeds control policy sampling;
stock XMage shuffles are not seeded, so the two controllers' draws are not
paired. Full traces and incremental `results.json` files are retained locally.

For the 100-game comparison use `--games 100 --workers 8 --seed 703` with the
same settings for both controllers. Play/draw and seats each balance 50/50.

The 2026-10-15 samples completed 100 attempts per controller:

| Controller | Wins | Mean own turns to win | Median | Self-defeats | Repeated-state cutoffs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Weak deterministic Burn baseline | 90 | 4.79 | 5 | 10 | 0 |
| Frozen checkpoint 605 | 31 | 6.39 | 6 | 8 | 61 |

Both samples had zero draws or infrastructure errors. Means and medians include
only wins; all other attempts count as failures. Across all attempts, the
baseline won by its sixth own turn in 84% of games, versus 17% for the model.
The model therefore still struggles to finish uncontested games despite its
isolated synthetic-task scores. The 61 model cutoffs all hit the repeated-state
guard, not the wall-clock or turn limits. No policy fallback or training update
was applied.

All 200 traces were audited: the opponent never played a card, its battlefield
remained empty, and its only responses were passing, keeping and required
cleanup discards. Actual play/draw matched setup in every attempt, and the model
checksum remained unchanged. All 268 repository tests pass. Full configurations,
per-game rows, source/checkpoint hashes, audits and limitations are in
[the 100-game goldfish benchmark](benchmarks/goldfish-100-2026-10-15.json).
