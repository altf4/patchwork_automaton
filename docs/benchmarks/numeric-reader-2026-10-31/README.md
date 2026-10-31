# Numeric rollout reader benchmark

On October 31, 2026, the 250K Standard model was used to compare batch
preparation strategies for PPO training on raytracer. The workload used the
same 800 training games (64,888 decisions), batch size 512, and two PPO epochs;
200 validation games were excluded from the training set. All runs started
from matching actor, critic, Adam, and random-number-generator states. The
checkpoint SHA-256 was
`12f263e2024014574c96130fc707fee95cdb9837130e498f27d27661fa5ce623`.

Two runs of each configuration were interleaved on the same warm-cache,
5.73 GB rollout file:

| Reader | Elapsed seconds | Mean | Change from baseline |
| --- | --- | --- | --- |
| Four Python workers (previous default) | 115.45, 118.50 | 116.97 | — |
| One Python worker (new default) | 97.22, 96.78 | 97.00 | 17.1% less time |
| Experimental native reader | 87.42, 88.97 | 88.20 | 24.6% less time |

The native reader was a benchmark prototype and was not adopted. The
one-worker default keeps the existing reader and learning algorithm; set
`PATCHWORK_NUMERIC_READ_WORKERS=4` to restore the prior worker count. These
results do not establish the best worker count for files larger than host
memory.

All benchmark runs had the same guard outcomes: 254 retained actor steps,
157 critic steps, and three rejected actor proposals, with the same final RNG
state. Final weights were not bitwise identical across runs, including the
baseline controls. Separate reader tests check exact data parity, including
ragged records, duplicate indices, and negative zero.

The [comparison results](results.json) contain the measurements. A separate
full GPU training run verified the new default ([report](default-verification.json),
[guard decisions](default-guard-status.json)): it used one reader and took
97.86 seconds, with the same decision count and guard outcomes. A smaller GPU
smoke test verified the four-worker environment override
([report](override-verification.json), [guard decisions](override-guard-status.json)).
The original run artifacts are on raytracer under
`/home/altf4/benchmarks/training-batch-prep-20261031/` and
`/home/altf4/benchmarks/numeric-reader-default-20261031/`.
