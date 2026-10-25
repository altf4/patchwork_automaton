# CPU rollout pipeline experiments — 2026-10-20

A 1 GiB rollout cache reduced median full-session time by **1.80%**.
Indexed numeric storage at the original 256 MiB budget reduced it by
**3.94%**, with **21.9×** the persistent rollout-file size.
The two experiments use separate, contemporaneous controls; their percentages
should not be added to estimate a combined result.

Defaults remain 256 MiB and compressed pickle. The cache gain is small relative
to observed variation; indexed storage trades persistent disk space for speed.
Both are available as explicit settings without changing TensorFlow or the model.

## Full-session measurements

Every trial starts from the same fresh seeded weights and runs one 64-game
self-play round, an 80/20 whole-game holdout, PPO, training/held-out validation,
8-game baseline and before/after evaluation, checkpoints, synchronization and
shutdown. Subprocess wall time includes imports, model compilation and warmup.
All trials are CPU-only (`CUDA_VISIBLE_DEVICES=-1`, TensorFlow CPU 2.21.0).
Hardware: Intel Core Ultra 5 335, eight cores; eight game workers and two
inference replicas. TensorFlow intra/inter-op threads 1/4, oneDNN off,
minibatch 32, two PPO epochs, concurrent actor/critic, prefetch 2, four measurement
workers and a 500-step critic cap. The observation encoder stays Python.

Two sequential runs per setting were performed in forward/reverse order.
All ten trials matched initial weights and self-play records: 64 naturally
finished games, 77,845 decisions, 29,467 neural choices, 23,543 training samples,
1,472 accepted actor steps and 500 critic steps, without rejected epochs.
Concurrent TensorFlow batching can introduce small floating-point differences;
post-update weights and evaluation outcomes are not required to be bit-identical.
These are local measurements, not statistical confidence intervals or a forecast
for larger rollouts, cold disks, other CPUs or GPUs.

## Cache budget sweep

The existing admission-only cache and 75/25 record/minibatch split are unchanged.
Order: 256 → 512 → 1024 → 1024 → 512 → 256 MiB. Medians:

| Budget | Complete session | PPO update | Session time reduction | Retained cache | Peak summed RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256 MiB | 122.35 s | 53.75 s | 0.00% | 252.7 MiB | 6.93 GiB |
| 512 MiB | 121.95 s | 53.19 s | 0.33% | 507.6 MiB | 7.00 GiB |
| 1024 MiB | 120.15 s | 51.78 s | 1.80% | 1021.5 MiB | 7.37 GiB |

Temporary spill declined from 1.24 GB
to 0.72 GB at 1 GiB. The retained
cache budget is distinct from total memory. RSS is sampled every second and
summed across the process tree; shared pages count repeatedly, and filesystem
page cache is excluded. The same accounting correction from the observation
pipeline experiment applies to all settings.

```bash
.venv/bin/python -m examples.benchmark_rollout_pipeline \
  --output .local/cache-sweep-reproduction
```

## Indexed numeric rollout storage

Order: pickle → indexed → indexed → pickle, always at 256 MiB. Medians:

| Format | Complete session | Collection | PPO update | Validation | Session time reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| pickle | 124.90 s | 36.35 s | 54.70 s | 17.28 s | 0.00% |
| indexed | 119.98 s | 36.11 s | 52.55 s | 15.58 s | 3.94% |

Persistent rollout files were 80.4 MB
for gzip/pickle and 1.76 GB for
indexed storage (21.9×). Gzip/pickle additionally wrote
1.24 GB to its temporary seekable
spill file; indexed storage wrote no spill. Primary data plus spill was roughly
1.32 GB versus
1.76 GB during an
update. The persistent footprint matters across many rounds even when transient
spill sizes are comparable. Both formats leave model/checkpoint formats intact.

The `.rlt` format stores contiguous numeric payloads and a versioned footer with
field dtypes, dimensions, offsets and scalar behavior fields. Readers fetch index
rows on demand, return owned NumPy arrays and keep at most 64 files open. Existing
cache admission/accounting applies; records do not depend on an open file or
mmap. PPO accesses uncached numeric samples directly instead of decoding gzip,
writing a temporary pickle copy and unpickling random samples from it. The shared
reader supports both existing `.pkl.gz` and new `.rlt` paths. Writers commit their
footer on close; incomplete/truncated files fail explicitly.

```bash
.venv/bin/python -m examples.benchmark_rollout_pipeline \
  --cases 256:pickle 256:indexed 256:indexed 256:pickle \
  --output .local/indexed-storage-reproduction

CUDA_VISIBLE_DEVICES=-1 .venv/bin/python -m examples.train_rustmage \
  --output .local/larger-cache --rollout-cache-mib 1024
CUDA_VISIBLE_DEVICES=-1 .venv/bin/python -m examples.train_rustmage \
  --output .local/indexed-training --rollout-cache-mib 256 --rollout-storage indexed
```

## Correctness and provenance

Twenty-one distinct targeted checks passed: five new storage tests and sixteen
existing regression checks. They cover exact tensor/scalar/dtype preservation,
variable shapes, missing behavior probabilities, malformed/partial files,
record ownership, file-reader limits/cleanup, deterministic normalization and
validation, model/optimizer/RNG equivalence with forced PPO backtracking, bounded
cache/spill, checkpoints, concurrent failures, global shuffle and whole-game
holdout isolation. No rewards, action pruning, recurrence or objective changed.

The [machine-readable benchmark record](benchmarks/rollout-cache-and-storage-2026-10-20.json)
contains source and engine hashes, software/runtime settings, CPU use, memory,
cache/spill statistics and raw report paths/hashes. Source inputs stayed fixed
while their consuming sessions ran. The storage integration started after the
cache sweep, and the storage stage used a fresh reference under the same code.

Individual full-session times:

| Stage | Setting | Trial | Complete session |
| --- | --- | ---: | ---: |
| Cache | 256:pickle | 1 | 120.93 s |
| Cache | 512:pickle | 2 | 119.91 s |
| Cache | 1024:pickle | 3 | 118.51 s |
| Cache | 1024:pickle | 4 | 121.79 s |
| Cache | 512:pickle | 5 | 123.99 s |
| Cache | 256:pickle | 6 | 123.77 s |
| Storage | 256:pickle | 1 | 124.72 s |
| Storage | 256:indexed | 2 | 119.87 s |
| Storage | 256:indexed | 3 | 120.10 s |
| Storage | 256:pickle | 4 | 125.09 s |
