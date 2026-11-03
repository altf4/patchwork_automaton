# Hartford experiment defaults

For new Hartford PPO experiments, use the diverse opponent recipe unless the
user requests another recipe or an experiment explicitly needs a control arm:
580 current-policy games, 100 each against three pinned historical identity
policies, and 120 StandardAggro scripted-opponent games per 1,000-game round.
The validated reference opponents are identity checkpoints 25, 50 and 75.
Scripts use only their supported mono-green, Izzet and BW opponent decks.
Retain the 800-training/200-whole-game-holdout split and balanced matchups.

Record the exact opponent checkpoints, hashes and schedule in every new run's
manifest. Preserve existing runs' saved recipes on resume. Never change an
existing experiment's protocol or frozen source implicitly to adopt this default.

The current `--fork-from` implementation supports the validated 50-round fork
from identity checkpoint 100 and defaults to `--opponent-recipe diverse`.
For other parents or run lengths, extend and validate continuation admission
and schedule provenance before launching; do not bypass those checks or fall
back silently to the old current/predecessor recipe.

See `docs/hartford-ppo.md` and the audited opponent-diversity benchmark linked
there for results and limitations.
