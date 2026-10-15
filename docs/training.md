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
the current version-3 Burn vocabulary without an explicit migration. They are
preserved as experiment artifacts; use a new output directory for a fresh run
with the updated annotations.
