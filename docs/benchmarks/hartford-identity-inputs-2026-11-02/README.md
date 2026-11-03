# Hartford identity-input experiment

Status: **complete and audited**, 2026-11-03. The identity-only arm completed
100,000 additional collection games from the original 200K policy, reaching
300K total exposure. Training retained its 80,000/20,000 whole-game
training/holdout split and ran zero policy-strength evaluations. A separately
user-authorized post-training suite then completed 11,200 evaluation games:
the identity agent scored **60.01% against the 200K parent** and **48.44% against
the scripted panel**, versus the parent's 43.92%. Scores count draws as half
a win. Extra training and identity inputs both changed, so this comparison
does not isolate the architecture's contribution.

Preserved experiment records:

- [Training protocol, completion and recovery](training-100k/README.md),
  [completion audit](training-100k/completion-audit/20261103T115903Z-audit.json)
  and [complete per-round report](training-100k/report.json.gz).
- [Post-training results and figures](post-training-comparisons/README.md),
  including all 11,200 compressed evaluation outcomes and their audit.
- [Preservation inventory](preservation/README.md),
  [frozen source archive](preservation/frozen-source.tar.gz) and
  [source manifest](preservation/source-manifest.json). The archive preserves
  the 387 files pinned by the final launch, including the runtime pin and
  Cargo overrides used by the frozen source.

Production models, optimizer/RNG checkpoints, training outcomes and retained
raw diagnostics remain on Raytracer at the paths pinned by
[launch.json](training-100k/launch.json); they are not included in this Git
archive. The original parent experiment remains unchanged. Launch and recovery
scripts record historical absolute paths: reproductions require new output
directories and explicit path configuration, not rerunning against these
completed artifacts.

The opt-in `sparse-identity-standard-v2` input adds learned visible card identities
and exact ordered object references to the existing sparse actor/critic. The
default for this experimental model is identity enabled, rule descriptors
disabled. The existing production model and training default remain unchanged.

This addresses an information loss in the old encoder: distinct cards such as
Spell Pierce/Spell Snare and Blooming Marsh/Overgrown Tomb can have identical
shared trait rows. Increasing network size alone cannot distinguish identical
inputs. Identity embeddings add a fixed-width vector per visible object; their
parameter count grows linearly with the card vocabulary. They do not enumerate
combinations of card abilities or possible game states.

The old feature tensors and all old actor/critic weights are preserved. A new
residual branch starts with an exactly zero output. Transfer from checkpoint 200
therefore preserves its logits and values before learning. This is a separate
experiment with fresh optimizer state, not a resume of the completed production
run. Old rollouts cannot be adopted because they do not contain the new identity
information. Unknown and concealed identities use zero, including named stack
sources marked concealed. Model manifests pin the vocabulary and descriptor
hashes; v1/v2 records cannot be mixed in an inference batch.

The optional rule ablation contains only a narrow typed land-entry/counterspell
export with explicit partial-coverage markers. It is not a general rules
representation and is not the foundation of the identity-only experiment.

## Discrimination probe

`probe-v1/probe.json` records a synthetic, supervised local-objective probe:
choose an untapped land while minimizing life payment, between a fastland and
a shockland. The shockland label assumes the later pay-two-life choice. This
is not an assertion about the strategically optimal move in a complete game.

There were 168 training and 168 held-out examples, two land pairs, both action
orders, and varying land counts. Holdout examples used disjoint life totals and
fresh object handles/order. All experimental arms have the same parameter
count. Only the new actor residual was trained for 300 steps.

| Model | Held-out accuracy |
| --- | ---: |
| Unchanged 200k checkpoint | 50% |
| Additional capacity without identity or rule inputs | 50% |
| Identity only | 100% |
| Identity plus narrow rules ablation | 100% |

This establishes that the new input can resolve this particular ambiguity.
It does not establish improved policy strength, broad rules understanding, or
generalization to unseen card identities.

## Pipeline acceptance

`acceptance-v1/acceptance.json` and `acceptance-artifact-audit.json` preserve the
local CPU acceptance evidence. Both the unchanged 200k model and its zero-output
identity transfer completed the same 12 games against StandardAggro, with
identical outcomes, action counts, turn counts, final life and 5,286 decisions
per arm. Collection took 17.45 and 17.16 seconds respectively. This small serial
panel is not a Raytracer throughput benchmark or a strength comparison.

The identity arm supplied eight training games (515 decision samples) and four
whole-game holdouts. Both PPO epochs completed with four accepted optimizer
steps, zero rejected epochs, and exact mean KL 0.00048863. Holdout measurement
did not mutate the model or optimizers. A fresh checkpoint restored both Adam
optimizers and trainer RNG exactly. The bounded input budget remains enforced.

The acceptance guard used the unchanged 1,536 MiB / five-second physical-memory
threshold and exited zero without a breach. The initial local harness delayed
at asyncio executor shutdown after writing all results, then exited normally
with a shutdown warning. The harness now keeps bounded selector wakeups through
executor shutdown. Independent inference-process testing uses the existing
`PATCHWORK_ARENA_POLL_INFERENCE=1` sandbox option. Original logs are retained;
successful acceptance games were not replayed to repair shutdown.

Native acceptance used the available local bridge pinned to engine `e732a4b`.
The optional typed descriptors were exported from `4bedacf`, after checking all
5,487 identity mappings against the unchanged feature catalogue. This acceptance
preceded the completed Raytracer GPU training run documented above. The training
run initially used runtime `4bedacf`, then the recorded round-43 repair moved it
to `4f64940` without changing the feature ABI or the parent's artifacts.

Regression coverage includes privacy, malformed sidecars, budget/version
boundaries, copied stack sources, identity gradients, object permutations,
mixed-size/repeated batch entries, exact v1 transfer, Keras serialization,
PPO/holdout/checkpoint/RNG restoration, independent compiled inference, and
native/private Python choice and feature parity over three complete games.

Reproduce the local probe and acceptance in a new output directory, supplying
the preserved parent checkpoint at the configured source path:

```sh
PYTHONPATH=src:tests:. CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m examples.experiment_hartford_identity \
  --source .local/hartford-identity-inputs/source-200k/policy.keras \
  --output /tmp/hartford-identity-new --mode both
```

## Bounded lookahead proposal

One-step afterstate inputs are a separate promising experiment. Reuse the same
state encoder to describe the public consequences of each candidate action,
and score current state, action, afterstate, and the change between them. This
can expose immediate costs and consequences without creating a category for
every possible ability combination. Continue training the policy/value model;
do not simply trust the existing critic to rank unfamiliar afterstates.

The engine already supports independent game clones. The current fast bridge
does not expose a bounded preview operation. A safe implementation needs to
stop before another player's choice, hidden information is revealed, or a
random outcome is sampled. Applying one engine micro-action may only announce
a spell, not resolve its costs/targets/effects. Auto-passing opponent responses
would describe a conditional hypothetical, not a guaranteed consequence.

Cloning the true hidden state and redacting afterward is insufficient: a
preview can leak the next real draw or an outcome caused by hidden information.
Before deployment, previews must be invariant across hidden worlds with the
same player-visible information. Also test that previews leave live game state
and RNG untouched. Begin with deterministic public consequences and an explicit
incomplete-preview marker; batch neural evaluation and benchmark extra engine
cost. Compare identity-only with identity-plus-preview under both equal-game
and equal-wall-clock budgets. No lookahead has been implemented or enabled by
this identity-input experiment.
