# RustMage expert imitation diagnostic

This experiment trains the existing Keras policy to predict `FoundationsAggro`'s
legal actions. Both seats use the unchanged mono-red deck and scripted baseline.
The dataset contains no model predictions, rewards, advantages, or value targets.
Collection does not import TensorFlow or start an inference service.

```bash
uv run --no-sync python -m examples.collect_rustmage_demonstrations \
  --output .local/imitation/dataset --games 10000 --games-per-batch 250 --workers 24

uv run --no-sync python -m examples.train_rustmage_imitation \
  --dataset .local/imitation/dataset --output .local/imitation/training
```

Each 250-game collection batch is partitioned into 200 training games and 50
validation games before creating neural minibatches. Both players' decisions
stay with their game. Failed or incomplete demonstrations stop the experiment;
they are not silently replaced. The completed dataset is fixed and reusable.

Training starts from fresh random weights. There are two supervised epochs on
each new batch of 200 training games, for 40 updates over the 10,000-game dataset.
Cross entropy on the exact expert action is the sole objective. The default
learning rate is 0.0003. A minibatch contains 32 seat sequences, with truncated
backpropagation through eight decisions. Sequences start from zero recurrent
memory; state is carried between chunks and reset between seats. Training replays
expert observations and previous actions; game evaluation uses the learner's own
actions. No expert or
learner hidden states are saved in the dataset. The separate value network is
retained for checkpoint compatibility and verified unchanged after each update.

Validation replays sequences using current weights without gradients. It does
not update optimizers, control stopping, or enter the training sampler. Metrics
include cross entropy, exact candidate accuracy, action-kind accuracy, macro
accuracy across expert action types, non-Pass accuracy, and per-action counts.
Exact accuracy distinguishes interchangeable physical copies of a card, so it
must be interpreted alongside action-kind accuracy and actual game performance.

At initialization and after each 250-game batch, evaluate against the same
do-nothing opponent using both sampled actions (the primary historical metric)
and greedy actions (a separate classifier diagnostic). Failures count as
`horizon + 1` only in evaluation. The scripted baseline provides a reference on
the same seeds. All legal neural actions remain available; no baseline fallback
is used when the model plays.

The final policy also evaluates all 2,000 held-out expert games. Reports, optimizer
states, policy checkpoints, and `imitation-learning.png`, `.svg`, and `.csv` are
saved locally. The report records the dataset hash, source hashes, unchanged-value
checks, and read-only validation audit. To resume an interrupted training run,
repeat its exact command with `--resume`. Models and generated data belong in
ignored `.local/`, not Git.

## Transfer the cloned policy to PPO

`examples.train_rustmage --initialize-from` starts a new PPO experiment from an
existing `policy.keras` or its checkpoint directory. It preserves the source
checkpoint and records its hash and actual training/held-out game counts. PPO
starts with fresh actor and critic Adam states and a fresh trainer RNG; it does
not restore the imitation optimizer. The imitation run leaves the critic at its
initial weights, so the transfer can first fit that critic with
`--critic-warmup-games 1000 --critic-warmup-max-steps 157`.

Warm-up collects 1,000 new self-play games with the frozen cloned actor, using a
separate `seed + 30,000,000` range and `warmup-rollouts/attempt-*` directories.
Both seats of each game stay together in an 80% training / 20% holdout split.
Only the training games enter the critic optimizer. The report compares value
error before and after warm-up on both partitions, replaying recurrent memory
from zero with current weights. Actor weights and every actor Adam variable
must remain identical. The warm-up step cap applies only to this phase; ordinary
PPO rounds retain their own `--value-max-steps` setting.

The runner first saves an immutable `prewarm-checkpoints/round-0000` snapshot.
After successful warm-up, it durably saves the warmed critic, both optimizers,
trainer RNG and warm-up result in the normal `checkpoints/round-0000`, before
any actor update. A resume during unfinished warm-up restores the prewarm
snapshot and repeats collection in a new attempt directory with the same seeds.
A committed normal round zero resumes without repeating warm-up, including
when its checkpoint was saved just before report publication. To resume, repeat
the transfer command with `--initialize-from ...` replaced by `--resume`.

The October 24 transfer pilot was launched with the following command from the
prepared raytracer project directory. The sourced environment supplies the WSL
CUDA libraries used by the GPU learner. CPU inference uses 48 game workers and
four frozen replicas; gradient minibatches contain 512 decisions. Actor and
critic learning rates are 0.0001 and 0.0003 respectively.

```bash
. /home/altf4/benchmarks/mtgo-cpu-gpu-20261024/env.sh
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES=0 TF_ENABLE_ONEDNN_OPTS=0 \
TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=4 \
/home/altf4/benchmarks/mtgo-cpu-gpu-20261024/venv/bin/python -u -m examples.train_rustmage \
  --output .local/raytracer-bc-ppo-4000-2026-10-24 \
  --initialize-from .local/red-imitation-10000-2026-10-19/training/checkpoints/round-0040 \
  --rounds 8 --games-per-round 500 --seed 21261024 \
  --workers 48 --inference-processes 4 \
  --learner-device gpu --inference-device cpu --validation-device cpu \
  --train-batch 512 --learning-rate 0.0001 --value-learning-rate 0.0003 \
  --value-max-steps 157 --critic-warmup-games 1000 --critic-warmup-max-steps 157 \
  --concurrent-updates --prefetch-batches 2 --measurement-workers 4 --rollout-cache-mib 512 \
  --eval-games 200 --baseline-eval-games 1000 --eval-every-games 1000 --eval-horizon 25
```

The eight PPO rounds collect **4,000 additional games**, split into 3,200
training and 800 held-out games before terminal-outcome filtering. The 1,000
critic warm-up games (800 training / 200 held out) remain separate from all PPO
progress counters. Fixed sampled evaluations use 200 passive-opponent games and
1,000 active-baseline games initially and after each 1,000 PPO games. This recipe
records a launched experiment; it does not establish a learning or performance
improvement.

### Audit the PPO guard before changing its limits

The saved first PPO update can be replayed without collecting games. From the
prepared project directory, use an independent output directory:

```bash
. /home/altf4/benchmarks/mtgo-cpu-gpu-20261024/env.sh
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES=0 \
/home/altf4/benchmarks/mtgo-cpu-gpu-20261024/venv/bin/python -u -m examples.diagnose_rustmage_guard \
  --source .local/raytracer-bc-ppo-4000-2026-10-24 \
  --output .local/raytracer-guard-control-attempt2-2026-10-24 \
  --device gpu --validation-device cpu --save-checkpoint
```

The control restores warmed checkpoint zero, including both Adam states and the
trainer RNG, and reuses exactly the recorded first training partition. Recorded
behavior probabilities and values remain frozen; advantages and minibatch
shuffles are reproduced from those same inputs and restored RNG. The source
checkpoint and its `latest.json` remain unchanged. Per-state audit reads share
the trajectory-reader lock with concurrent critic preparation.

Each proposal immediately writes a flushed `guard-proposals.jsonl` event with
its learning rate, mean/tail/maximum KL, rejection reasons and worst decision
identities. The report compares resulting actor weights, optimizer state and
RNG with the original first update, and records before/after validation using
current recurrent-memory replay. `--save-checkpoint` saves the diagnostic result
in its own output directory.

First confirm that the control reproduces the original update. **Only if the
logged maximum-state veto dominates rejection**, repeat with a new output
directory and `--max-state-kl 0.5` to isolate that limit. This frozen-data test
diagnoses guard behavior and optimization cost; it establishes no playing-strength
improvement. The main training runner also exposes `--target-kl`,
`--max-state-kl` and `--guard-audit`; their default limits remain **0.02 mean KL
and 0.1 maximum-state KL**, with auditing off.
