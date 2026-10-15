# Numerical card and ability semantics

`patchwork_automaton.semantics` implements the initial parameterized annotation
schema for a non-language policy. It has no TensorFlow dependency and does not
execute effects, infer legality, or change backend observations/submissions.

Each card has a stable semantic identity, keyword flags and separate abilities.
Each ability has costs, a trigger/condition, a mode and an ordered list of effects.
Effects specify a kind, quantity expression, recipient/filter, duration,
destination and optional keyword/counter/template identity. Integers are enum
IDs or numeric values; no prompt prose or card text enters encoded features.

```python
from patchwork_automaton.semantics import lookup_visible, encode_card

# obj is an ObjectView from the acting seat's common Observation.
features = encode_card(lookup_visible(obj))
```

The initial reviewed registry contains Mountain's tap-for-red mana ability,
Lightning Bolt's red casting cost and three damage to any target, and the
casting costs and permanent-spell resolution of Goblin Piker and Raging Goblin
(including the latter's base haste keyword). IDs are local
semantic IDs, not collection IDs, printing IDs, object handles or action UUIDs.
`INITIAL_CARDS` retains those original four records. The default
`burn_semantics.CARD_REGISTRY` additionally annotates all 26 distinct cards in
the pinned Modern Burn mainboard and sideboard (28 identities total including
Goblin Piker and Raging Goblin). `lookup_visible` and the neural `Encoder` use
this expanded registry by default. One base
annotation is shared across printings; backend identity resolution remains the
adapter's responsibility. Names are exact registry lookup keys outside the model.

## Tensor layout and batching

`encode_card` returns scalar schema/card/coverage IDs, a keyword multi-hot vector,
and variable-length ability, cost and effect rows. `ABILITY_COLUMNS`,
`COST_COLUMNS` and `EFFECT_COLUMNS` declare their ordered column layouts.
Cost/effect rows reference an ability row, preserving grouping; effect row order
preserves sequencing. Pool/attend over ordered effect rows to create ability
vectors for the entity encoder and action scorer. Preserve mode grouping.
Schema version 2 adds aligned `ability_tags` and `effect_tags` multi-hot rows.
`RuleTag` is a closed integer vocabulary for reviewed mechanical qualifiers;
the model receives those flags, never their Python names or explanatory prose.

Batch collation must pad these arrays and supply row masks. Empty arrays have
shapes `(0, len(COST_COLUMNS))`, etc., before padding, rather than an ambiguous
one-dimensional empty array. Keep categorical IDs in integer tensors and embed
them; handle quantities separately as numerical features. Enum/column changes
require schema versioning, persisted vocabulary manifests and checkpoint checks.
Do not truncate oversized records or treat padding as an unknown real effect.

## Coverage, visibility and current state

Coverage is UNKNOWN, PARTIAL or COMPLETE, independently for card, keywords and
individual abilities. COMPLETE means the annotation covers the card's abilities
under this schema; it does not certify all rules interactions, type/cost changes,
or live action availability. Zero keyword flags with UNKNOWN coverage mean
unknown, not verified absence. Unknown identity is card ID 0.

`lookup_visible` rejects absent names and explicit hidden/face-down flags, even
when a name accidentally remains in the object payload. Only pass the actor's
view; do not merge seats or enrich unseen cards using private backend data.
Remembered identities require a separate knowledge tracker with visibility rules.

These annotations describe base abilities. Keep them separate from current
engine properties: gained/lost flying, changed costs, copied abilities and
temporary effects require live normalization. Do not fill missing current
properties with printed defaults and pretend they are verified current state.

## Burn review and coverage

The reviewed source IDs, URLs, Oracle-text hashes, coverage and limitations are
in [`burn-semantics-review.json`](../src/patchwork_automaton/data/burn-semantics-review.json).
Fifteen of the 26 deck identities have complete base-feature coverage; eleven
remain partial. Partial records still supply reviewed costs, effects, keywords
and rule qualifiers. Keyword coverage refers to the supported `Keyword`
vocabulary; other mechanics such as spectacle and fuse have dedicated tags.

The annotation distinguishes damage from life-payment costs (Barbarian Ring
versus Sunbaked Canyon), independent red/white mana alternatives, Boros Charm's
three modes, spectacle's alternative cost, and prowess/Eidolon trigger scopes.
Skullcrack's restrictions precede its damage. Lands retain entry restrictions
and fetch-search subtype filters. Wear/Tear aliases share identity 28; face
names retain only that half's spell record, while the combined card has both
halves and a fused record.

`QuantityKind.LANDFALL_DAMAGE` specifically means one damage, or three if a land
entered under your control this turn, evaluated at resolution. Searing Blaze's
first target is a player or planeswalker; `SAME_CONTROLLER_AS_FIRST_TARGET`
qualifies the second creature target as controlled by that player or that
planeswalker's controller. It does not restrict either target to an opponent.
These tags describe rules; they do not impose a policy preference or legality
mask. `PREVENTED_DAMAGE` represents Deflecting Palm's dependent amount.

Partial coverage explicitly retains unresolved runtime dependencies: searched
or revealed card identity, optional entry payments, linked targets, delayed
death triggers, last-known controllers, replacement events and split-card
zone/face behavior. Conditional effect tags prevent presenting Molten Rain's
extra damage or Searing Blood's delayed damage as unconditional effects.

The neural feature version is now 4. Card embeddings have 29 entries including
unknown ID 0, and semantic rows have 199 columns. Original card IDs 1–4 remain
stable, but vocabulary and tensor shapes changed. Version-2 and version-3 checkpoints are
preserved on disk and rejected by the default loader; start a fresh model or
implement an explicit weight migration before using them with these features.

## Next integration work

The Burn adapter now binds reviewed ability-picker rules to individual annotated
abilities/modes, scoped by visible source identity and native links or click
context. Candidate features include the specific ability ID and typed semantic
summary. Broader rule variants and multi-trigger ambiguities still need mappings.
Never equate an opaque UUID or option position with a verified identity. Unresolved
bindings retain unknown semantics; the backend options still determine what can
be submitted. Dynamic cost and target constraints remain authoritative inputs.

The [initial Keras policy](neural.md) includes a collator, XMage feature encoder,
entity attention, GRU memory and candidate scorer. Expand verified action
bindings and semantic coverage as the card pool grows. A quantity expression such as MATCHING_COUNT
needs its filter/context supplied; symbolic X is not a measured zero.

The vocabulary is intentionally incomplete. Compound conditions, hybrid/Phyrexian
costs, replacement effects, detailed searches and token templates need additional
structured fields/definitions. Mark those annotations partial rather than
silently flattening away meaningful conditions or dependencies. No automatic
rules-text parser or external card-data download is included.
