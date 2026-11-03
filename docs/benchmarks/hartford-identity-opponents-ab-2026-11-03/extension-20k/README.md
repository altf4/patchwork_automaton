# Additional 20K head-to-head games

The user requested a fixed additional 20,000 games after reviewing the original
10,000-game result. The combined sample contains 30,000 outcomes from the same
diverse and control checkpoint-50 models.

Each new ordered matchup has 200 games, with 50 in each seat/start combination.
Combined with the original batch, every matchup has 300 games and every seat/start
combination has 75. New game IDs run from 10000 through 29999 and new seeds from
432611040000 through 432611059999. All original outcomes retain their recorded
identities, hashes and adjudication diagnostics.

The evaluation retains sampled temperature 1, 48 workers, two collectors, two
learner and two opponent inference processes, a 20-second game deadline, 50 turns
and 6,000 decisions. The physical-memory guard requires 1,536 MiB for five
seconds. Preparation requires at least 32 GiB free disk. Frozen gameplay source
and engine hashes are checked alongside both completed training histories.

`manage.py prepare` reaudits the original batch and creates a separate pinned
runner and guard. `manage.py start` launches the fixed additional sample.
`manage.py status` archives observations and checks durable results and hashes.
`finalize.py` audits both batches, checks disjoint seeds/IDs and exact balance,
then writes the combined results and graph under `combined-30k`.

The additional sample size was chosen after the initial result was observed.
Combined error bars use nominal fixed-sample Hoeffding bounds. The comparison
contains one training run per recipe.

The combined 30K evaluation is complete and audited. The diverse agent scored
51.725%, with nominal 95% bounds of 50.9409%–52.5091%.
See [combined results](combined-30k/README.md) for the graph, both batch
summaries, and the explicitly recorded publication recovery.
