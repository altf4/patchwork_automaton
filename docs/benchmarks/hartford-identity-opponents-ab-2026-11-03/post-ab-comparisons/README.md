# Fixed-opponent evaluation of the two A/B agents

Status is **COMPLETE**. All 40,000 games passed their schedule, outcome, handoff,
source, engine, checkpoint and resource-guard audit. The hourly monitor is removed.
See [results and graph](results/README.md).

The user authorized four evaluations of exactly 10,000 games each, totaling
40,000 games. Control and diverse are the final local checkpoint-50 models from
the completed A/B experiment. The user confirmed that the reference should be
identity checkpoint 100, which started both training arms. That reference has
100K identity-training games and 300K total inherited exposure. Both A/B agents
have 350K total exposure.

| Agent | Opponent | Games | Matchups | Deadline |
| --- | --- | ---: | --- | --- |
| Control | StandardAggro baseline pool | 10,000 | Original three-deck 3×3 panel | 60 seconds |
| Diverse | StandardAggro baseline pool | 10,000 | Identical panel and matched seeds | 60 seconds |
| Control | Shared identity checkpoint 100 | 10,000 | All 100 ordered Hartford matchups | 20 seconds |
| Diverse | Shared identity checkpoint 100 | 10,000 | Identical panel and matched seeds | 20 seconds |

The original scripted lists are mono-green landfall, Izzet prowess and BW
lifegain. Exactly 10,000 games cannot divide evenly across nine cells and four
seat/start combinations. Seven baseline cells contain 1,112 games and two contain
1,108, with every cell balanced across all four seat/start combinations.
Each neural matchup contains 100 games, 25 per seat/start combination.

All panels use sampled temperature 1, 48 workers, two collectors, two learner
and two opponent CPU inference processes, 50 turns and 6,000 decisions. The
supervisor requires 1,536 MiB available physical RAM for five seconds. Disk
admission requires 32 GiB free. The baseline deadline reproduces the earlier
benchmark. Comparison seeds are fresh and matched between agents within each
opponent panel. Baseline seeds start at 452611030000 and neural seeds at
462611030000. Game IDs are unique across all four panels.

The frozen gameplay source is copied into a new snapshot with one publication
amendment. Completed outcomes are atomically handed off before inference reset.
If reset cleanup fails, collectors close before validated handoffs are published.
The transport error and its explicit resolution stay in the panel artifacts.
A specific regression test verifies that reset failure cannot hide a completed
outcome or allow replay of a handed-off game. Further tests check exact panel
counts, matched seeds, role balance and the established evaluation validators.

`manage.py` prepares, launches and observes the suite. `evaluate.py` runs each
panel in a separate process and verifies all frozen artifacts before and after.
`audit.py` verifies schedules, models, outcomes, diagnostics, handoffs and guard
history. `finalize.py` fetches the audited results and generates PNG/SVG graphs.
The authoritative launch and resource paths are recorded in `launch.json`.

The diverse arm has practiced against the scripted policies. Scores on that
panel therefore measure performance against those fixed policies. There is one
trained agent per recipe, with error bars describing evaluation-game uncertainty.
