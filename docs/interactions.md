# Interaction API (0.3.0)

See the [135-pattern coverage checklist](interaction-coverage.md) for the inventory.

The server supplies actions and runs the rules. The library decodes its choices
and sends the caller's selections. All choices go through `game.submit()` with
the current `Observation.decision_id`; session, revision, player ownership,
game-over, and duplicate checks still apply.

## Targets and modes

`Observation.actions` contains `CardAction` descriptors. Each has target groups,
mode options, casting-option text, mana/cast traits, and the original attributes.
Targets can be cards, players, stack objects, payment selections, or effects;
use the advertised object IDs, not catalog IDs.

```python
from patchwork_automaton import CardSelection

action = observation.actions[0]  # The caller chooses the action.
groups = action.target_groups_for()  # For modal actions, supply modes=(...).
candidates = action.legal_targets(observation, 0)

await game.submit(observation.decision_id, CardSelection(
    action.card_id,
    action.action_id,
    target_groups=((chosen_target_id,),),
))
```

Supply one tuple per active target group, including empty tuples for optional
or deferred groups. The library checks minimum/maximum counts, advertised IDs,
duplicates, and descriptor relationships. `action.legal_targets(..., chosen=...)`
filters candidates after partial selections. For example, `chosen=((id,),)`
means group zero already has that target. Relationships include controller,
toughness, names/types, property sums, differing property values, and matching
values across groups. Required metadata that cannot be resolved raises
`UnsupportedInteraction`; this is not a replacement for the server's rules.

```python
await game.submit(observation.decision_id, CardSelection(
    action.card_id,
    action.action_id,
    modes=(0, 2),                    # Advertised mode indices, not raw bytes.
    target_groups=((first_target,), (second_target,)),
    retain_priority=True,
))
```

Inspect `action.modes.minimum`, `.maximum`, and `.options`. Options carry an
index, label, target-property ID, allowed repetition count, and weight. Modal
selections are encoded as the native ASCII mapping, sorted by mode index.
Repeated modes repeat their target groups. Weights and the advertised limits
are checked. Some casting menus advertise separate actions for fixed mode
combinations; select the matching action instead of changing its combination.

For effects divided among targets, provide matching distribution tuples:

```python
CardSelection(
    action.card_id, action.action_id,
    target_groups=((target_a, target_b),),
    distributions=((2, 1),),
)
```

The encoder writes target IDs and group separators first, then the distribution
values and corresponding separators. The byte target count excludes the
distribution section. Distribution totals and minimum bucket amounts are
checked against the descriptors.

`CardChoice` remains the low-level alternative with raw mode bytes, flat target
values, extra mana, flags, and an optional explicit `target_count` for divided
buffers. It checks the action ID but bypasses structured selection validation.

## Counters, stack, and card faces

`observation.objects[id]` exposes zone, controller, owner, texture, immutable
counter counts, raw properties, and facet IDs. Counter changes are applied by
the server; there is no local operation that independently adds a counter.
To pay/remove/distribute counters, respond to the advertised numeric, card,
target-group, or counter-type selector.

`observation.stack` returns visible stack objects sorted by their server order.
Countering a spell is a targeted advertised card action against one of those
objects. There is no special client-side counterspell rules implementation.

`observation.actions_for(parent_id)` includes actions on split, adventure, room,
or other facets referenced by the server. Submit each action's actual
`card_id`, which may be a child object ID. The library does not manufacture
actions for a face that the server has not offered.

## Numeric, player, color, card-name, and type selectors

`observation.selectors` contains immutable `Selector` descriptors with stable
IDs for that observation. The decoder handles every UI descriptor in the
examined client: prompt button, mana button, number spinner, player buttons,
list selector, choose color, card selector, and wish selector.

```python
from patchwork_automaton import SelectorChoice

selector = observation.selectors[0]
print(selector.kind, selector.minimum, selector.maximum, selector.options)
await game.submit(observation.decision_id, SelectorChoice(selector.id, value))
```

The library checks numeric bounds, enabled card choices, player indices,
available colors, and list exclusion/inclusion filters. Legacy number/pile
prompts are normalized too. Unknown descriptor types retain their bytes and
cannot be submitted through this API.

There are distinct identifier namespaces:

| Selector | Submitted value |
| --- | --- |
| Number | Chosen integer |
| Player | Game-server player index |
| Color | `PromptCode` color flag |
| Card selector / wish | Card texture number |
| Card-name list | Card name token |
| Creature, land, planeswalker, counter, card type | Corresponding property/type ID |

The package includes the recovered type/counter ID tables. Card names and
definitions are optional local data rather than bundled client assets. If a
menu needs missing catalog data, `selector.catalog_required` is true. A caller
that already knows the correct numeric list value can submit it; the library
can then validate server filters but cannot certify catalog membership.

Build a local catalog once from an installed client's data, using the optional
`dnfile` discovery dependency:

```sh
python3 scripts/extract-choice-catalog.py \
  /path/to/Mtgo.Shared.Parser.Attributes.dll \
  /path/to/CardDataSource .local/choice-catalog.json
```

```python
from patchwork_automaton import ChoiceCatalog, MTGOClient

catalog = ChoiceCatalog.from_json('.local/choice-catalog.json')
client = MTGOClient(choice_catalog=catalog, strings=string_table)
```

`ChoiceCatalog` can also take caller-supplied definitions and source tables.
For format-specific naming menus, supply `allowed_name_tokens=` when loading
the catalog. No Modern legality database is inferred. Server-provided type
flags override static definitions when present; special dynamic names/types
still need representative edge-case validation.

## Mana and costs

`observation.mana` contains pool entries with a player index, ID, amount, raw
color, normalized color flag, and properties such as snow/source restrictions.

```python
from patchwork_automaton import ManaChoice, PromptCode

await game.submit(observation.decision_id, ManaChoice(PromptCode.RED, mana_id))
```

Use an offered color and an available matching source ID. The native mana
message uses timestamp zero; freshness is still checked against the observation
before sending. Restricted mana remains server-directed and caller-selected.
The library does not decide which mana is spendable on a particular spell.

Alternative costs are separate advertised actions or later prompts. X uses a
number selector; sacrifice, discard, exile, and counter costs use the supplied
card actions, target groups, distributions, or selectors. Follow each new
observation rather than sending a guessed series of payment messages.

## Combat, ordering, and damage

Attacker and blocker choices use `CardSelection` against advertised action
IDs. `attack_victim_id` identifies a preconfigured attack destination when
provided. For target-bearing block/attack actions, choose from the target groups.

The server may flag actions `can_perform_locally`. Such actions can be sent for
different objects under the same decision, as the client does when declaring
multiple attackers or tapping multiple sources. Duplicate submissions of the
same action are rejected. Snapshots remain authoritative server observations;
they are not optimistically mutated.

`observation.ordering` supplies source IDs and the objects to order:

```python
from patchwork_automaton import OrderingChoice

await game.submit(observation.decision_id,
                  OrderingChoice(source_id, (first_id, second_id)))
```

`observation.damage` supplies source IDs, exact totals, and recipient bounds:

```python
from patchwork_automaton import DamageChoice

await game.submit(observation.decision_id,
                  DamageChoice(source_id, ((recipient_a, 2), (recipient_b, 1))))
```

Ordering is checked as an exact permutation. Damage must include each offered
recipient exactly once, satisfy its bounds, and match the total. Separate
sources may be submitted under the same decision before its confirmation
button; submitting the same source twice is rejected. Ordering uses native
one-based ranks with flags zero; damage uses amounts with flags one.

Trigger ordering, replacement choices, searches, and arranging library cards
use whichever of these descriptors/actions the server offers. Do not assume
all kinds of ordering use a combat-order message: some are sequences of card
actions or target selections followed by a Done button.

## Locally staged batches

`submit_batch()` validates all choices against one decision before sending the
first message. Use it when the server offers actions flagged local, or several
independent damage/ordering sources followed by a confirmation:

```python
receipts = await game.submit_batch(observation.decision_id, (
    CardSelection(first_object_id, first_action_id),
    CardSelection(second_object_id, second_action_id),
    PromptChoice(done_button_code),
))
```

Every choice except the optional final one must be a partial choice. Ordinary
casts or choices that require a new observation cannot be chained in a batch.
Repeated partial keys are rejected. Build or edit the caller's selection list
before submitting it; the library does not optimistically mutate snapshots.
Three-pile arrangements use the offered per-card action IDs; there is no guessed
pile-number wire command. `prompt['arranging_three_piles']` exposes the native
legacy flag/text indicator, and special-player/type fields are preserved.

Validation is atomic; delivery is not. A failed transport write ends the session
and the library never retries the sequence. A sent receipt proves no acceptance.

Advertised non-game object menu actions require explicit
`CardSelection(..., allow_non_game=True)`. These are partial choices, allowing an
object-specific preference action followed by a prompt response where required.

## Game controls and extra state

```python
from patchwork_automaton import PlayerStops, Step, YieldControl, YieldPreference

await game.undo(expected_revision=observation.revision)
await game.yield_control(YieldControl.DISABLE_ALL)
await game.yield_control(YieldControl.AUTO_STACK_TRIGGERS)
await game.remove_yield(advertised_yield_id)
await game.update_stops(
    checksum=deployed_string_checksum,
    stops=(PlayerStops(0, Step.PRECOMBAT_MAIN | Step.END),),
    yields=(PlayerStops(1, Step.END),),
    preferences=YieldPreference.NORMAL_CARD_DRAW,
)
```

These are independent example operations, not a recommended sequence. Native
function-key yield controls are **toggles**, except `DISABLE_ALL`; repeating a
request can reverse the setting. There is no automatic retry or idempotent-set
claim. Stop preference `player_id` values are game-server indices, not account
IDs. Step masks start at upkeep; first-strike and ordinary damage share the
combat-damage stop. The deployed string-table checksum can be saved with:

```sh
python3 scripts/extract-strings.py /path/to/WotC.MtGO.Client.Model.Play.dll \
  .local/strings.json --checksum-output .local/string-checksum.txt
```

`game.undo()` and `game.concede()` invalidate the current decision immediately;
wait for a fresh state or result. The server decides whether undo is legal.
`game.concede()` concedes one game; `client.concede_match(match)` concedes the
match. Other controls can use `expected_revision=` to reject stale observations.
`game.detach()` closes only that game route and makes its handle unusable.

Additional immutable observation fields:

- `commanders`: owner index, live object ID, texture, damage to each player.
- `dungeons`: texture and per-player completion counts.
- `auto_yields`: object yield ID, player mask, description, effect.
- PlayerStatus summaries: active-player index and displayed yield preferences.
- Mana entries: `source_id`, `snow`, and `restriction` convenience properties.

Dynamic ranged creature types replace static types. Unknown enum values remain
unresolved, so relationship checks fail explicitly. Changeling/wildcard and
lost-abilities flags are normalized. Face-down objects are nameless and do not
inherit printed creature types. Room name sets follow unlocked doors on the
battlefield and the chosen half on the stack; local catalog extraction now
retains the required nested half definitions. Regenerate older local catalogs
for Room support. This still does not simulate arbitrary copy/layer effects.

## Sideboarding, subsequent games, and results

```python
match = await client.challenge(opponent_id, deck, format='CMODERN', best_of=3)
# Accept/start using the normal workflow, then play the first game.

# When sideboarding is permitted, supply the entire revised deck. DeckItem's
# sideboard field distinguishes main-deck and sideboard entries.
result = await client.submit_sideboard(match, revised_complete_deck)
if result.accepted:
    next_game = await client.wait_for_game(match, after_game_id=finished_game.id)
else:
    print(result.errors, result.expired)
```

The native match type is `BES` for first two game wins; the default `best_of=1`
remains `SGM`. Match sync metadata retains `match_type`. The server validates
format/deck eligibility and controls the sideboarding window. The caller must
establish that the match continues rather than blindly sideboarding after the
last game. Tournament and league event decoding/entry remain outside this API.

Sideboarding disconnects any ended game routes for that match before sending
the full revised deck. Active games, unknown matches, empty decks, and nonpositive
quantities are rejected locally. Rejections preserve every native validation
error (card ID, annotation, code, region, card-specific flag). Expiration is
reported separately. A next-game wait handles either notification order and
refreshes cached match metadata instead of reusing an old game ID.

`game.ended` exposes the received game-over status. `game.result` retains the
latest typed `GameResult` (moved-first index, status, per-player result code and
playing time). Winner indices follow the native code 2. A `GameMessage` for
`GameResultsMessage` also carries `.result`; raw payload access is retained.
A pending `wait_for_update()` wakes and raises `GameEnded` with the result if
the game finishes without a newer observation.

## Validation and limits

New encoders/decoders are based on native serialization and choice construction
from client 3.4.159.4704. Tests use synthetic framed responses and rejection
cases. All 82 saved observations from the previous direct match were replayed
through the expanded model. The new advanced choices have not been sent in a
fresh live match; both account connections stayed closed during implementation.

This covers the shared interaction families, not an exhaustive certification
of every Modern card. Remaining work is live acceptance coverage, unusual
dynamic name/type/facet combinations, missing definitions, and future server
schema changes. Raw attributes and `send_raw()` remain available for research.
No bot policy, rules simulation, automatic payment, or action retry is added.
