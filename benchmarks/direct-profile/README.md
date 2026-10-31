# Direct Standard playback: profile and inference experiments

The frozen 250K Standard checkpoint (SHA-256 `12f263e2024014574c96130fc707fee95cdb9837130e498f27d27661fa5ce623`) was played against itself on raytracer with 48 CPU game workers, two runners and two inference replicas per player. The direct Rust playback path was used throughout. Weights were frozen and no learning took place. Timed phases exclude startup and a 48-game warmup but include game execution, inference, trajectory writing and result collection. Source files, library and checkpoint were hashed and checked before and after each run; memory guards did not retry.

## 1,000-game profile

The profiled run completed 1,000 natural games in 64.61 seconds, or 928.6 games/minute, for 340,129 decisions. The table accounts for **764.82 CPU-seconds** measured across game children, worker supervisors, coordinator and inference replicas. Percentages are of this summed CPU, not elapsed time. Intervals are exclusive within each game child; CPU from separate processes is summed.

| Measured category | CPU seconds | Share |
| --- | ---: | ---: |
| Inference replicas: model, batching, IPC and runtime | 413.65 | 54.1% |
| Other game work and instrumentation | 82.92 | 10.8% |
| Game-side inference pipe | 75.77 | 9.9% |
| Rust observation construction | 50.15 | 6.6% |
| Rollout writing and close | 35.25 | 4.6% |
| Rust history, choice expansion and encoding | 30.07 | 3.9% |
| Rust action application | 29.67 | 3.9% |
| Python/NumPy array-view setup | 28.25 | 3.7% |
| Other native and action-label overhead | 14.41 | 1.9% |
| Worker, coordinator and unaccounted child CPU | 4.69 | 0.6% |

Rust observation and action calls include engine work; they do not separately isolate all game-rule computation. The game-side inference pipe CPU includes serialization and communication, not waiting time. Inference-replica CPU includes model execution, batching, communication and TensorFlow runtime, not just the network. Separately timed service intervals total 179.94 seconds for model forward, 54.11 for batch packing and 1.86 for result conversion. Those are summed **wall intervals that overlap across replicas**, not fractions of job wall time or additional CPU.

The inference replicas served 164,573 requests in 30,601 calls, averaging 5.38 requests per batch. The 240-game pilot matched the uninstrumented control on all recorded game summaries; instrumented elapsed time was 1.031 times the control, so the profile has measurable but modest observed timing overhead. It does not establish the overhead for every workload. The guard exited successfully with at least 6.42 GiB sampled available memory. [Profile manifest](results/profile/manifest.json) and [per-stage summary](results/profile/profile/summary.json).

## Shared inference pilots

The candidate shared replicas across both seats only because both used the same frozen model and version; recurrent states and random sampling remained player-specific. Four shared replicas preserve the baseline's total replica count; two shared replicas reduce it. These are 240-game pilots, not tightly bounded estimates. Each configuration used the same scheduled seeds, decks, seats and starting players.

| Configuration | Baseline games/min | Candidate games/min | Baseline mean batch | Candidate mean batch | Baseline replica CPU | Candidate replica CPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Four shared replicas | 1,004.9 | 979.0 | 5.10 | 5.31 | 87.0 s | 84.0 s |
| Two shared replicas | 1,013.6 | 841.9 | 5.12 | 10.02 | 86.6 s | 52.7 s |

Both pilots matched all 240 game summaries and all 18,520 recorded feature/action records. Inference outputs exhibited the small floating-point variation expected from different batch compositions. Shared replicas were not adopted: in these pilots the lower replica CPU did not translate into higher throughput, especially with two replicas. These results apply to same-checkpoint self-play and do not establish behavior with independently updated player models. [Four-replica manifest](results/preflight-shared4/manifest.json), [two-replica manifest](results/preflight-shared2/manifest.json).

## CPU inference thread test

The promising change keeps the existing two-plus-two topology and reduces **intra-op threads per CPU inference replica from two to one**. Inter-op threads remained at two. Each freshly spawned replica explicitly set and reported its effective TensorFlow thread counts; all before/after snapshots matched the intended settings. This changes inference workers only in the test; it does not change a learner's settings. The 240-game pilot was approximately 5.1% faster and matched all games and 18,566 recorded feature/action records. The pilot uses the first games of the main schedule and is not an independent replication. [Pilot manifest](results/preflight-threads1/manifest.json).

The main test used two different matched 1,000-game schedules in baseline/candidate then candidate/baseline order, with fresh warmed processes per phase. All 4,000 attempted games completed naturally. Aggregate throughput is 2,000 games divided by the sum of each configuration's two timed intervals.

| Matched schedule | Two intra-op threads | One intra-op thread |
| --- | ---: | ---: |
| 1 | 942.9 games/min (63.64 s) | 998.6 games/min (60.09 s) |
| 2 | 923.2 games/min (64.99 s) | 993.3 games/min (60.41 s) |
| **Combined** | **933.0 games/min (128.62 s)** | **995.9 games/min (120.49 s)** |

One thread increased aggregate throughput **6.75%**, reduced elapsed collection time **6.32%**, and reduced measured inference-replica CPU from **802.04 to 674.95 seconds** (15.85%). These figures apply to this machine, 250K model and two-neural-seat playback workload; two paired schedules do not tightly bound future variability.

All **2,000 matched game summaries** agreed on seed, result, winner, decisions, turn, life and action counts. All **159,745 recorded player feature/action records** matched exactly, with no skipped adjudicated games. Recorded inference outputs were not bit-identical: 225 recurrent-memory records and 173 value/log-probability records exceeded the strict `1e-6` absolute / `1e-5` relative tolerance. The maximum absolute log-probability difference was `1.43e-5` and maximum value difference `6.89e-6`; small differences also occur between separately batched Python baseline runs. These results establish recorded game/input agreement in the tested sample, not universal bitwise equivalence.

The guard exited successfully; the minimum sampled available host memory was 5.98 GiB. The checkpoint, source and direct library hashes were unchanged after the run. The one-thread setting was subsequently adopted as the CPU inference-replica default; it is configured inside the replica, not globally for a potentially concurrent GPU learner. Set `PATCHWORK_CPU_INFERENCE_INTRA_THREADS` to a positive integer before creating a collector to override it. Full game records and trajectories are on raytracer under `/home/altf4/benchmarks/direct-profile-20261031/`. The [full matched manifest](results/full-threads1/manifest.json), [guard status](results/full-threads1/guard-status.json), and [computed summary](results/summary.json) are saved here. Run `python3 benchmarks/direct-profile/summarize.py` to validate and regenerate the summary.

`compare_threads.py` records the historical experiment and is intended for its pinned pre-default source; the later `compare_default.py` tests the production default against the explicit two-thread override.

A subsequent guarded verification exercised the actual production default with the parent environment still set to two intra-op threads and both inter-op threads unchanged. All four CPU replicas reported one intra-op thread in the default phase; an explicit replica override reported two in the baseline phase. All 240 game summaries and 19,104 recorded feature/action records matched. The default ran at 1,026.6 versus 973.1 games/minute for the explicit two-thread setting in this short check. Small separately batched floating-point differences persisted. The guard exited successfully with at least 6.20 GiB sampled available memory. [Production-default manifest](results/default-check/manifest.json).
