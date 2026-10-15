# Untrained numerical Keras policy

The optional `patchwork_automaton.neural` package implements a Keras 3 model
using TensorFlow, a numerical observation encoder, and a structured action
adapter. It can control both seats in local XMage. All gameplay responses are
sampled from the network, with random initial weights and no optimizer updates.
There is no scripted strategy fallback, mandatory keep, forced target preference,
or concession to manufacture a completed episode.

## Run with uv

Use Python 3.12 for the TensorFlow wheels. The default system Python on the
development host is 3.14, which is not the tested model environment.

```sh
uv sync --python 3.12 --extra neural --extra xmage --extra rl
uv run --no-sync python scripts/xmage-local.py start
uv run --no-sync python -m examples.neural_xmage \
  --episodes 2 --deck combat --seed 29 \
  --save .local/neural-policy.keras \
  --trace .local/xmage/neural.jsonl \
  --output .local/xmage/neural.json
```

The server must already be installed; see [the setup instructions](xmage.md).
On the current workspace, uv itself is available at `.local/uv/uv` and the
isolated `.venv` has been synced. `uv.lock` records the tested dependency
resolution. Runtime files and checkpoints stay under ignored `.local/`.

Reload the same weights:

```sh
uv run --no-sync python -m examples.neural_xmage \
  --load .local/neural-policy.keras --episodes 1 --deck burn --seed 31
```

`--seed` seeds model initialization and action sampling, not XMage's shuffle.
`--limit` truncates an episode and causes the runner to report failure if a
natural result was not reached. It does not concede. The example removes its
tables and closes sessions; the server remains running until explicitly stopped.

Both example decks use Freeform Unlimited to exercise interactions. `burn` is
24 Mountains and 36 Lightning Bolts. `combat` is 24 Mountains and 12 each of
Lightning Bolt, Goblin Piker and Raging Goblin. These are transport/model test
decks, not format-legal competitive deck lists.

## Input and output contract

An observation becomes only integer, boolean and float arrays. Names are used
outside the network to resolve reviewed card annotations. Display labels,
prompt prose, card rules text, user names, live UUIDs and backend credentials
are never model features. A UUID is only an adapter reference to an entity row.

| Input | Per-observation shape | Meaning |
| --- | --- | --- |
| entities | N × 149 | Relative seats, zones, coverage, base keywords, current numeric/boolean features, players and mana |
| card_ids | N | Stable semantic identity, embedded by the model; unknown = 0 |
| semantics | N × S × 199 | Ordered parameterized ability, cost and effect records with numerical rule tags |
| relations | N × N × 4 | Controller, parent, target and stack-order links exposed by the gateway |
| context | 42 | Phase, prompt kind, actor-relative active/priority seats, turn and bounds |
| candidates | A × 38 | Option kind, semantic role, binding coverage, bounds and argument features |
| candidate_refs | A × N | Reference from each choice to its entity, when known |
| history | H × 40 | Visible appearances/disappearances/moves, player counts/life and turn changes |
| history_cards | H | Known semantic card identities associated with history events |
| selection | 38 | Previous selected option, or the parent option during argument assembly |
| memory_in | 256 | Actor-local recurrent state |

Associated masks identify real entities, semantic records, history events and
candidates. `collate()` pads each batch to its largest axes without dropping
objects or actions. Missing numerical values have explicit known flags.

Outputs are candidate logits, a scalar value in [-1, 1], and updated memory.
The sampling layer uses softmax with configurable temperature. Initial final
layer weights are small and random, making the untrained distribution close to
uniform while still computing every choice through the network.

The model has 64-dimensional identity embeddings, an ordered semantic-record
GRU, three 128-dimensional entity-attention blocks, history and decision GRUs
with 256-dimensional state, a shared candidate scorer, and a value head.
Relationship features aggregate linked entity representations before attention.
The value estimate is untrained and should not be interpreted as a calibrated
win probability.

## Actions and semantics

Finite XMage choices map the sampled index back to `core.Response`. Targets and
payment sources are offered object references. Generic string choices select a
finite backend binding; no string generation occurs. Unknown ability semantics
remain marked unknown rather than being inferred from display text.

The default registry includes all 26 cards in the pinned Burn deck, with 15
complete and 11 partial base-feature annotations. Playable-object metadata
separates land plays, spell casts and mana abilities. Unambiguous single-ability
source/category matches on fully covered cards bind to reviewed annotations.
Multiple modes, alternative costs and arbitrary ability-picker UUIDs remain
unresolved rather than being assumed to have the
same meaning across games or backends.

Feature version 3 uses 29 card-embedding entries, including unknown ID 0.
See [semantic coverage and checkpoint compatibility](semantics.md). The earlier
version-2 checkpoints and benchmark results below predate these annotations.

Numeric choices use bounded decimal-digit selections, so a 32-bit upper bound
does not allocate billions of logits. Offsetting by the lower bound permits
negative ranges. Every digit prefix is masked to retain a feasible completion.
For distributions, recipient bounds are tightened against the remaining total
before choosing each amount. Internal digit selections do not advance game
memory. Their log probabilities are retained with the parent option selection.

The model samples the adapter's advertised candidates. Some XMage mana fallback
clicks require server validation; the library does not certify all candidates
as legal moves in every game state. It does not retry failed submissions or
silently replace a model choice. A callback that cannot be represented stops
the run with an explicit error.

## History, checkpoints and validation

The runner attaches `policy.observe` as the XMage client's synchronous
`on_observation` listener. This records structured changes in each seat's own
server view between decisions, including transient visible cards. The history
buffer is bounded and fails on overflow rather than silently dropping events.
It records observed changes, not an inferred complete event log or opponent-hand
reconstruction. Each actor has independent recurrent state; reset clears it.

Checkpoints use Keras `.keras` files plus `.manifest.json` files that fingerprint
the feature schema and reviewed semantic registry. Reload rejects mismatched
vocabularies. Model dimensions are serialized and weights are built before
restoration. Runtime memories and sampler RNG state are reset, not checkpointed.

```sh
uv run --no-sync python -m unittest discover -s tests
uv build --wheel
```

Tests cover text independence, hidden identity isolation, transient-history
capture, dynamic padding, integer/distribution bounds, masked logits, finite
training gradients, memory handling and checkpoint output equivalence.

The first policy encoder supports XMage only. MTGO's bundled action arguments
still need a dedicated binding/assembly implementation. Complex cards can have
partial or unknown semantic annotations, and live keyword extraction, combat
relationships, cost restrictions and effect semantics need broader coverage.
No learning algorithm or optimizer loop is included in this milestone.

## Live verification on 2026-10-14

The final 1,257,346-parameter model completed these games with complementary
server win/loss results. In all three, the losing player's final life was -1.

| Model | Deck | Decisions | Elapsed gameplay seconds |
| --- | --- | ---: | ---: |
| Random initialization, seed 29 | Combat | 459 | 36.366 |
| Same weights, next episode | Combat | 622 | 50.634 |
| Reloaded checkpoint, sampling seed 31 | Burn | 397 | 29.758 |

That is 1,478 model-selected decisions, with zero optimizer steps and no forced
concessions. The combat traces include creature selections during both attacker
and blocker declarations. Earlier exploratory Burn and combat runs also ended
naturally; the table reports only the final implementation and its reload test.
Finite numeric/distribution selectors are covered by offline tests, not by these
four-card live scenarios.

All 164 offline tests passed with optional dependencies installed; the wheel
built successfully. Checkpoint output equivalence and gradient propagation are
tested. Threaded RL tests require working local event-loop wakeup sockets; they
passed outside the restrictive execution sandbox used during development.

The [machine-readable results](benchmarks/neural-xmage-2026-10-14.json) record
versions, feature fingerprint, checkpoint hash, callback counts and outcomes.
Detailed local traces are `.local/xmage/neural-final.jsonl` and
`.local/xmage/neural-reloaded.jsonl`. The checkpoint is
`.local/neural-policy.keras` with its adjacent manifest. These timings are
execution checks, not measurements of competitive strength or optimized training
throughput.
