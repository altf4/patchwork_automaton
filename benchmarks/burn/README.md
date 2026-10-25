# Burn CPU experiment — 2026-10-20

The faithful Burn port is faster for individual decisions and has a much smaller
process footprint, but **does not improve the measured training-step latency**.
Keep TensorFlow for training on the evidence from this experiment. Moving
observation encoding and batching into Rust remains a separate opportunity;
this experiment does not measure that work.

## Results

Intel Core Ultra 5 335, 8 logical CPUs, Linux x86-64. TensorFlow 2.21.0 versus
Burn 0.22.1, release build, Flex backend with SIMD and Rayon. No GPU is used.
TensorFlow uses the project's current configuration: 1 intra-op thread,
4 inter-op threads, oneDNN disabled. Burn uses the indicated Rayon pool size.

Median milliseconds per call in the confirmation run; lower is better:

| Real examples | Operation | TensorFlow | Burn, 1 thread | Burn, 4 threads |
|---|---|---:|---:|---:|
| Typical, batch 1 | Actor + critic inference | 1.348 | 0.418 | 0.480 |
| Typical, batch 4 | Actor + critic inference | 2.030 | 1.799 | 1.744 |
| Typical, batch 12 | Actor + critic inference | 5.025 | 10.513 | 11.971 |
| Typical, batch 32 | Actor + critic inference | 12.071 | 35.498 | 32.972 |
| Large board, batch 1 | Actor + critic inference | 1.616 | 1.000 | 1.023 |
| Large board, batch 4 | Actor + critic inference | 2.658 | 3.384 | 2.758 |
| Large board, batch 12 | Actor + critic inference | 5.429 | 10.090 | 10.662 |
| Large board, batch 32 | Actor + critic inference | 14.380 | 32.572 | 36.386 |
| Typical, batch 32 | Actor PPO update | 16.658 | 24.000 | 25.513 |
| Typical, batch 32 | Critic update | 22.598 | 33.461 | 36.318 |
| Large board, batch 32 | Actor PPO update | 18.943 | 25.457 | 31.377 |
| Large board, batch 32 | Critic update | 27.609 | 41.344 | 46.624 |

The one-thread Burn run takes 34–50% longer per training step across these four
cases. Typical single-decision inference is 3.22× faster. Four Rayon threads
do not improve training in this experiment. These are results for this port,
backend, machine and shapes, not a ceiling on Burn's performance.

Peak process RSS is 826 MiB for TensorFlow versus 63–64 MiB for Burn. This
includes framework imports, loaded fixtures and warmed graphs; it is not the
memory requirement of a complete training session. Burn with one thread also
uses less process CPU time per training step, despite its longer wall time.
That might help under contention, but concurrent session throughput has not
been measured here.

The machine-readable [results](results-2026-10-20.json) include all individual
trials, process CPU time, runtime settings, source and checkpoint hashes, full
numerical comparisons, and a separate TensorFlow oneDNN control.

The oneDNN control reduces the four training-step medians by 8–14%:
15.332/20.374 ms for typical actor/critic updates and 17.229/23.828 ms for
large-board updates. That is a smaller change worth evaluating in a full
training run before considering a framework migration. Production defaults
have not been changed.

An additional attempt to build Burn's CubeCL/LLVM CPU backend with fusion and
autotune failed during dependency resolution: `cubek 0.3.1` requires
`cubek-fft ^0.3.1`, while the available registry returned `0.3.0` and older
prereleases. The exact build log is included in the results. **LLVM CPU was
not benchmarked.** No dependency was locally patched to manufacture a result.

## Model and correctness

This is the actual `SeparateActorCritic` network, with 367,714 trainable
parameters in 86 arrays: actor width/memory 32, critic width/memory 64, one
actor attention block, four attention heads, semantic and history GRUs,
relation messages, candidate interactions and separate actor/critic heads.
The port preserves Keras reset-after GRUs, gate ordering, population layer
variance, epsilon 0.001, exact GELU, masking and recurrent state.

The fixture uses the round-9 checkpoint and real saved training decisions from
16 evenly spaced training games in round 10 of
`.local/red-batch500-5000-2026-10-19/report.json`. Both seats are sampled. Seed
721 fixes selection. Stored behavior memories and log probabilities and the
round's advantage-normalization constants are retained. Typical batches are
random selections from that pool; large-board selections have at least 60
entities. All exported samples are genuine choices with memory advancement.
There are eight input batches at sizes 1, 4, 12 and 32. Their padded entity
counts are 35–64, candidate counts 6–11 and history lengths 1–13.

Validation passes all **1,056 arrays**, using `atol=3e-5, rtol=5e-4` set before
examining the results:

| Comparison | Largest absolute difference |
|---|---:|
| Policy logits | 2.38e-7 |
| Critic values | 9.54e-7 |
| Recurrent memory | 4.17e-7 |
| Every actor and critic parameter gradient | 9.54e-7 |
| Weights after each of three Adam updates | 1.26e-5 |
| Adam first and second moments after three updates | 3.63e-8 |

Gradient and optimizer validation uses both batch-32 shapes. Adam matches
Keras' update equation: learning rate 0.0003, betas 0.9/0.999, epsilon 1e-7,
global gradient-norm clipping at 1. Actor loss uses PPO ratio clipping at
0.8/1.2 and entropy weight 0.01; critic loss is masked squared error. Both
frameworks start with fresh optimizer state for each trial. This is not an
optimizer-state checkpoint migration or evidence of identical long-run
learning curves.

## Timing boundaries

- Seven confirmation trials per case. Training trials each perform 20 updates;
  inference iteration counts target roughly 0.3 seconds, capped at 256.
  Preliminary five-trial runs showed the same broad tradeoff.
- Model loading, feature extraction, collation, input-tensor construction,
  graph compilation, five warmup calls, and optimizer reset are outside timers.
- Inference materializes logits, values and combined recurrent memory in both
  frameworks. TensorFlow executes a compiled `tf.function`.
- Training includes forward, loss, backward, finite checking, global norm,
  clipping and Adam. TensorFlow uses the existing production `PPOTrainer`
  compiled step methods. Burn uses tensor autodiff and a Keras-compatible Adam
  implementation. Both materialize gradient norm; Burn also materializes loss.
  Burn synchronizes work before stopping the timer.
- Benchmarks run sequentially, without overlapping compilation, outside the
  build resource guard. CPU affinity is not pinned and this is a shared desktop,
  so individual trials vary; raw ranges are retained, not treated as confidence
  intervals. Thread configurations are explicit, not equal CPU-core quotas.
- This excludes the game engine, Rust↔Python object conversion, IPC, rollout
  serialization, Python observation encoding, PPO acceptance/backtracking and
  evaluation scheduling. Actor and critic steps are timed separately. It does
  not establish a speedup or slowdown for an entire concurrent training round.

## Reproduce

Run from the `patchwork_automaton` repository root. The existing Python virtual
environment and trusted local rollout/checkpoint files are required. Exporting
and validating create new directories and refuse to overwrite an existing
archive. The results JSON is portable; the 12 MiB numeric fixture and larger
validation arrays stay under `.local/`.

```bash
.venv/bin/python benchmarks/burn/export_fixture.py \
  --report .local/red-batch500-5000-2026-10-19/report.json \
  --checkpoint .local/red-batch500-5000-2026-10-19/checkpoints/round-0009/policy.keras \
  --output .local/burn-repeat/fixture

CARGO_TARGET_DIR="$PWD/.local/burn-target" \
  python3 .local/rustmage/scripts/resource_guard.py -- \
  cargo build --release --locked --manifest-path benchmarks/burn/Cargo.toml

.venv/bin/python benchmarks/burn/tensorflow_reference.py validate \
  --fixture .local/burn-repeat/fixture --output .local/burn-repeat/tf-validation

RAYON_NUM_THREADS=1 .local/burn-target/release/patchwork-burn-benchmark \
  validate .local/burn-repeat/fixture .local/burn-repeat/burn-validation

.venv/bin/python benchmarks/burn/compare.py \
  --reference .local/burn-repeat/tf-validation \
  --burn .local/burn-repeat/burn-validation \
  --output .local/burn-repeat/validation.json

TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=4 \
  TF_ENABLE_ONEDNN_OPTS=0 CUDA_VISIBLE_DEVICES=-1 \
  .venv/bin/python benchmarks/burn/tensorflow_reference.py bench \
  --fixture .local/burn-repeat/fixture --output .local/burn-repeat/tensorflow.json \
  --trials 7 --train-steps 20

RAYON_NUM_THREADS=1 .local/burn-target/release/patchwork-burn-benchmark \
  bench .local/burn-repeat/fixture .local/burn-repeat/burn-1.json 7 20

RAYON_NUM_THREADS=4 .local/burn-target/release/patchwork-burn-benchmark \
  bench .local/burn-repeat/fixture .local/burn-repeat/burn-4.json 7 20

.venv/bin/python benchmarks/burn/summarize.py \
  --fixture .local/burn-repeat/fixture \
  --validation .local/burn-repeat/validation.json \
  --baseline tensorflow \
  --run tensorflow=.local/burn-repeat/tensorflow.json \
  --run burn-1=.local/burn-repeat/burn-1.json \
  --run burn-4=.local/burn-repeat/burn-4.json \
  --output .local/burn-repeat/results.json
```

For the oneDNN control, repeat the TensorFlow benchmark with
`TF_ENABLE_ONEDNN_OPTS=1` and a different output filename. To change model
architecture, port the changed layers and rerun numerical validation before
timing; the Rust model is deliberately scoped to this checkpoint architecture.
The experiment does not change the production training backend.
