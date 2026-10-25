# Native TensorFlow input pipeline

This library moves the Foundations mono-red observation encoder into Rust while
retaining the existing TensorFlow model, optimizer, vocabulary, tensor shapes,
history ordering and checkpoint format.

The native path is experimental: it did not improve full-session throughput on
the measured CPU, so Python remains the default. Build once from the
`patchwork_automaton` repository root and explicitly select Rust:

```bash
.venv/bin/python scripts/build-observation-pipeline.py
.venv/bin/python -m examples.train_rustmage --output .local/native-training \
  --observation-pipeline rust
```

Use `--observation-pipeline python` for the reference implementation. A checkpoint
can resume with either implementation; the chosen runtime and native library
hash are recorded per invocation. Outside the training command, set
`PATCHWORK_OBSERVATION_PIPELINE=rust` to opt in. `PATCHWORK_OBSERVATION_LIBRARY`
can select an explicitly built shared library. Missing libraries fail with a
build instruction instead of silently selecting a different implementation.

## Boundary and ownership

The engine's existing C ABI still produces a redacted JSON observation for one
seat. Rust parses that response directly from the engine allocation, before it
is freed. The training collector receives compact metadata and a native
observation handle. It does not build a Python tree of card objects unless a
baseline or public adapter explicitly requests those objects.

Rust performs exact legal-domain enumeration and deduplication, entity and
candidate encoding, relation construction, context features, actor-private
history deltas and repetition fingerprints. Python retains orchestration,
action sampling, recurrent-state bookkeeping and TensorFlow. Production batching
uses NumPy: constructing ctypes descriptors cost more than the Rust padding
saved in session measurements. The native padding implementation remains
available for direct experiments and equivalence tests.
The engine revision and rules implementation are unchanged. JSON serialization
inside the engine and NumPy-to-TensorFlow conversion remain; this is not a claim
that the whole engine-to-TensorFlow path is zero-copy.

Numerical outputs are owned Rust vectors exposed as contiguous float32, int32
and boolean NumPy views. Each view holds an allocation owner through its ctypes
base. Slices remain valid after the observation and game close. No returned
array borrows the engine response, a scratch buffer or a later decision's state.
Pickle/IPC continues to serialize ordinary NumPy arrays, not raw pointers.

Batching calls borrow contiguous input arrays only for the duration of the
synchronous FFI call. ctypes releases the GIL for native operations. Match
locks and actor-local history locks prevent concurrent mutation of native
handles. Policies must continue treating borrowed observations as read-only.

The rollout cache counts the full native allocation, including vector capacity,
even when only a small slice survives. Shared allocations are charged once per
cache entry. This preserves the existing memory budget instead of counting only
Python view objects.
The same correction also accounts for ordinary memoryview-backed NumPy arrays,
including decoded protocol-5 rollouts. Both benchmark implementations use this
corrected accounting and the same 256 MiB rollout-cache budget.

## Schema and validation

The Python vocabulary manifest configures the library once per process. Ordered
card identities, semantics, feature enums and action fields come from that
manifest, and configuration mismatches fail explicitly. The native encoder is
scoped to the existing 105/25/87/408/40 feature widths and the reviewed mono-red
vocabulary. Unknown cards, action fields and unsupported domains are errors;
legal choices are neither truncated nor strategically pruned.

`tests/test_native_pipeline.py` compares all 14 arrays bit-for-bit with the
independent Python encoder through four complete games, including both private
views at every revision, action ordering, exact payment/damage domains, history,
repetition fingerprints and mixed-shape batching. It also checks buffer lifetime,
explicit failure cases and native-buffer cache accounting. Existing policy,
spawned-collector, checkpoint, PPO scheduling, cache and spill tests exercise the
new path under TensorFlow. An additional 1,000-decision stochastic walk checks
encoding and private history through blocks, damage assignments and other
choices the baseline policy does not necessarily make.

```bash
.venv/bin/python -m unittest discover -s tests -p test_native_pipeline.py -v
PATCHWORK_OBSERVATION_PIPELINE=rust PYTHONPATH=tests \
  .venv/bin/python -m unittest test_rustmage_neural test_rustmage_processes \
  test_update_pipeline test_streaming_ppo -v
```

## Full-session benchmark

```bash
.venv/bin/python -m examples.benchmark_native_pipeline \
  --output .local/native-pipeline-comparison
```

The default order is Python → Rust → Rust → Python. Every run starts from the
same seeded weights and performs two 32-game rounds, an 80/20 whole-game split,
PPO updates, held-out validation, baseline and before/after evaluation,
checkpoints and shutdown. Eight engine workers and two frozen inference replicas
run with TensorFlow intra/inter-op threads 1/4 and oneDNN disabled. There is no
GPU use or simultaneous benchmark/build activity.

The benchmark records both complete subprocess wall time (including startup and
shutdown) and individual training phases, CPU use, initial-weight identity,
game-record identity, source/library hashes and raw per-run results. Two runs
per implementation are useful local evidence, not a statistical guarantee or an
extrapolation to every training configuration. The interrupted preliminary run
used during development is excluded from reported results.

For a comparison with the same frozen collection policy throughout every
self-play game, use `--rounds 1 --games-per-round 64`. In multi-round sessions,
small floating-point differences from concurrent TensorFlow inference can change
updated policies and subsequent game lengths, even between two Python runs.

## Results: 2026-10-20

**No overall improvement on this machine.** Median full-session time was
120.50 seconds with Python and 123.45 seconds with Rust: Rust took 2.45% longer.
The four runs were sequential, in Python/Rust/Rust/Python order, with one
64-game round each. Hardware: Intel Core Ultra 5 335, eight CPU cores. Software:
TensorFlow CPU 2.21.0, Keras 3.15.1, NumPy 2.5.4 and orjson 3.13.0.

| Metric | Python median | Rust median |
| --- | ---: | ---: |
| Complete session | 120.50 s | 123.45 s |
| Self-play collection | 34.74 s | 35.42 s |
| PPO update | 52.97 s | 54.10 s |
| Held-out validation | 16.93 s | 17.55 s |
| Process-tree CPU time | 444.79 CPU-s | 455.08 CPU-s |

Individual complete-session times were 120.72/120.29 seconds for Python and
123.44/123.47 seconds for Rust. All four runs had identical initial weights and
self-play game records: 77,845 decisions, 29,467 neural choices, 23,543 training
samples, 1,472 accepted actor steps and 500 critic steps. All 64 self-play games
finished naturally without collection errors or cutoffs. Evaluation remains the
existing 20-own-turn test and includes cutoffs for the initially untrained
policy. Post-update weights and evaluation games are not bitwise deterministic.

The complete session includes startup, compilation/warmup, baseline and
before/after evaluation, self-play, updates, held-out validation, checkpoints,
weight synchronization and shutdown. Both modes use the corrected memory
accounting described above. These are two local repeats per mode, not a
confidence interval or evidence about larger configurations or GPUs.

The earlier two-round prototype, with Rust batch padding and a full-state clone
for fingerprints, took 136.49 versus 128.27 seconds (6.4% longer). The final
version borrows the fingerprint's state and retains NumPy batching. That removed
most of the regression, but did not produce a training-session gain. Profiling
identified JSON parsing/allocation as the remaining boundary cost. The next
candidate is a typed engine ABI that fills numeric buffers directly, avoiding
the engine's JSON serialization and the helper's JSON parsing; this benchmark
does not measure that design.

Reproduce the final comparison:

```bash
.venv/bin/python -m examples.benchmark_native_pipeline \
  --rounds 1 --games-per-round 64 \
  --output .local/native-pipeline-comparison-matched
```

The [benchmark record](../../docs/benchmarks/native-observation-pipeline-2026-10-20.json)
preserves individual trials, source/library hashes, environment, raw-report
locations and the earlier prototype result. After timing, the CLI default was
returned to Python and the native history reader acquired the observation lock
as well as its history lock. The record lists those source changes explicitly;
the measured native binary is unchanged.
