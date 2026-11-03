# Identity PPO: current versus diverse opponents

User-authorized comparison: 100,000 additional collection games in two independent
50K arms, followed by a 10,000-game head-to-head and a graph. No intermediate
policy-strength evaluations or extra baseline panels are scheduled.

Both arms start from completed identity checkpoint 100 (300K total exposure),
preserving model weights, both Adam states and trainer RNG. Each local checkpoint
zero is pinned to the same parent; local checkpoints 1–50 correspond to lineage
checkpoints 101–150 and end at 350K total exposure. Original experiments stay intact.

| Arm | Opponents in each 1,000-game round |
| --- | --- |
| Control | 700 current; 100 each against its three immediate predecessors |
| Diverse | 580 current; 100 each against frozen identity checkpoints 25, 50 and 75; 120 StandardAggro scripts |

The control initially uses the real parent checkpoints 97–99. It does not reset
to fresh-run opponent quotas. Scripts use only the supported mono-green landfall,
Izzet prowess and BW lifegain opponent decks. Every one of the 100 ordered deck
cells retains ten games, eight training and two whole-game holdouts. Each
supported scripted-opponent cell has four scripted games. Opponent categories
are also balanced across seat/start roles and training/holdout partitions.

Both arms share game seeds, deck/seat/start assignments, queue order and holdout
membership. Their outputs and scheduled identities are namespaced by arm; the
same seed intentionally defines one distinct game in each arm. The paired
collection seed range is 422611030000–422611079999, disjoint from the prior
Hartford training, acceptance and comparison ranges. Bounded plumbing acceptance
uses seeds starting at 442611030000 and is never imported into production.

Model, PPO, rewards, temperature 1, 48 game workers, two collectors and two plus
two CPU inference processes remain fixed. Each arm retains the 512 MiB payload
budget, 20-second game deadline, 50-turn and 6,000-decision limits, 1,800-second
PPO timeout, 1,536 MiB physical-memory guard for five seconds and 32 GiB minimum
free disk. The two arms run sequentially with the same resource allocation on
Raytracer, avoiding contention on its shared GPU.

After both complete and pass training audits, the diverse agent plays the control
for 10,000 games: 100 per ordered matchup, 25 in each seat/start combination.
Evaluation seeds are 432611030000–432611039999. Draws count as half a win; all
timeout diagnostics and failed attempts remain visible. The final graph reports
the diverse agent's score with uncertainty. One run per recipe measures this
pair of agents; evaluation-game intervals do not measure training-seed variation.

`prepare.py` only stages and validates independent forks. `acceptance.py` checks
real checkpoint restoration, native mixed collection and unchanged training state
on 48 disjoint plumbing games without optimizer updates or strength scores.
`run_suite.py` runs the two prepared arms, read-only audits and the endpoint under
the resource supervisor. `monitor.py` archives local observations; `finalize.py`
audits and renders completed results. Read `launch.json` for authoritative frozen
source, commands, environment, original artifact hashes and guard paths.

Recovery must preserve committed checkpoints, original failures and every
successful outcome. Diagnose and validate a targeted repair before a controlled
resume; never replay successful games, reset optimizers, weaken limits or silently
replace failures. A failed arm must not automatically start the next phase.
