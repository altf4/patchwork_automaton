# Modern Burn synthetic checkpoint 605

Saved research progress from 2026-10-15. This is a numerical Keras 3/TensorFlow
policy with 1,476,766 parameters, feature schema v6, and a scalar value head.
Keep `policy.keras` and `policy.manifest.json` together. The manifest validates
the feature and semantic vocabulary before loading.

The retained payment stages added 8,100 supervised synthetic puzzles to the
earlier checkpoint. They cover mana payments, cancellation recovery and Searing
Blaze targets alongside earlier opening, land, casting and combat tasks. The
latest stage is described in `training-report.json`; ancestry and measurements
are in [the payment curriculum benchmark](../../docs/benchmarks/payment-curriculum-2026-10-15.json).
Those additions used supervised demonstrations, with no additional PPO or
intermediate real-game rewards.

This model is still weak:

- Against the scripted Burn baseline: 4 wins in 20 games.
- Against an inert opponent: 31 wins, 8 self-defeats and 61 repeated-state
  cutoffs in 100 attempts. Successful games averaged 6.39 own turns.

See [the goldfish benchmark](../../docs/benchmarks/goldfish-100-2026-10-15.json)
for the comparison with the baseline, cutoffs and limitations. Synthetic puzzle
scores do not establish full-game competence.

Load using the installed neural extras:

```python
from patchwork_automaton.neural.policy import NeuralPolicy

policy = NeuralPolicy.load("models/modern-burn-synthetic-605/policy.keras")
```

With the local XMage server and gateway ready, run the competency metric:

```sh
uv run python -m examples.evaluate_goldfish \
  --load models/modern-burn-synthetic-605/policy.keras \
  --directory .local/xmage/goldfish-restored --games 100 --workers 8 --seed 703
```

The archive preserves policy/value weights and feature metadata. It does not
include optimizer state or live recurrent history. Further training can warm
start from these weights with a new optimizer. Runtime traces and intermediate
experiments referenced by reports use paths from the original workspace.
