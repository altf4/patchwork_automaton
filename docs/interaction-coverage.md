# Gameplay interaction checklist (0.3.0)

This checklist inventories interaction patterns, not implementations of Magic's
rules. The server supplies actions, legal candidate IDs, choices, and new states;
the caller selects answers. No card-name dispatch or per-card handler registry
is required for the patterns below.

**Generic** means an implemented descriptor/message path can carry the choice
when the server advertises it in that form. It does not mean a named mechanic's
complete sequence has been tested with a real card. **Dedicated** means a
specific additional API or decoder exists. **State** means the authoritative
information is exposed; its consequences remain server-managed. All newly
added paths have offline tests; live evidence remains the earlier Burn games.

The API reference is [interactions.md](interactions.md). Regression fixtures live
in `tests/test_interactions.py`, `tests/test_mechanics.py`, and
`tests/test_match_lifecycle.py`.

## Casting and activating

| Interaction | Implementation | Coverage |
| --- | --- | --- |
| Play a land | Advertised `CardSelection` | Generic |
| Cast a spell from hand | Advertised `CardSelection` | Generic |
| Cast from graveyard / flashback / escape | Zone-specific advertised action, then payment choices | Generic |
| Cast from exile / foretell / suspend permission / plot | Zone-specific advertised action, then payment choices | Generic |
| Cast from command zone / commander tax | Advertised action and server payment prompts | Generic |
| Cast an opponent's card with permission | Advertised object/action IDs | Generic |
| Activate a permanent's ability | Advertised `CardSelection` | Generic |
| Activate from hand / cycling / channel | Advertised `CardSelection`, then costs | Generic |
| Activate from graveyard / unearth | Advertised `CardSelection`, then costs | Generic |
| Loyalty abilities | Advertised action; server changes loyalty | Generic |
| Equip / reconfigure / attach | Action with target groups | Generic |
| Morph / disguise / turn face up | Advertised action, payment prompts | Generic |
| Special actions / companion / suspend | Advertised actions rather than keyword inference | Generic |
| Alternative cost / evoke / overload / prototype | Separate actions or subsequent selectors | Generic |
| Additional cost / kicker / buyback / entwine | Advertised menu actions, modes, and payment prompts | Generic |
| Flash / timing permissions | Server-advertised availability | State |
| Cast without choosing targets yet | Deferred target groups, including divided effects | Dedicated |
| Activated versus cast versus mana traits | `CardAction` flags and immutable attributes | State |
| Parent menu / submenus / casting variants | Preserve separate advertised action IDs and attributes | Generic |
| Object-specific non-game menu actions | `CardSelection(..., allow_non_game=True)` | Dedicated |

## Modes and target selection

| Interaction | Implementation | Coverage |
| --- | --- | --- |
| Single target | One target group | Generic |
| Multiple target groups | Separate tuples with native separators | Generic |
| Optional targets / up to N | Advertised group bounds, including empty tuple | Generic |
| Ordered targets / first and second roles | Preserve tuple and group order | Generic |
| Player target | Advertised game-player object ID | Generic |
| Spell / ability on stack target | Stack object ID in target group | Generic |
| Target own / opponent / any controller | Server candidate list plus descriptor relationships | Generic |
| Target during resolution | New observation and fresh decision | Generic |
| Random targets | Deferred descriptor; server chooses | Dedicated |
| Cannot repeat targets across groups | Cross-group distinctness validation | Generic |
| Same / different controllers | Descriptor validation and partial candidate filtering | Generic |
| Equal toughness | Descriptor validation | Generic |
| Distinct names | Native name-set comparison, including nameless objects and Rooms | Dedicated |
| Shared creature type | Type metadata, changeling and wildcard support | Dedicated |
| Shared card / permanent type | Current type metadata and descriptor validation | Generic |
| Sum of target properties bounded | Advertised property ID and maximum sum | Generic |
| Different values of a target property | Advertised property ID | Generic |
| Same property value across earlier groups | Target-list bitmap and property ID | Generic |
| Choose one mode | `CardSelection(modes=(...))` | Generic |
| Choose multiple modes | Weighted minimum/maximum and capacity checks | Generic |
| Repeat a mode / spree-style combinations | Repeated mode indices and corresponding target groups | Generic |
| Fixed combinations offered as separate actions | Preserve the advertised fixed mode mapping | Generic |
| Divide damage / counters / effect among targets | `distributions=...`, native divided buffer | Generic |
| Distribution reminder / threshold / ordered comments | `TargetGroup` metadata; no local effect calculation | State |
| Retain priority after casting / activating | `retain_priority=True` | Generic |

## Selectors and costs

| Interaction | Implementation | Coverage |
| --- | --- | --- |
| Number / X / repeat count | Bounded numeric selector | Generic |
| Choose a player without targeting | Player selector with server index | Generic |
| Choose a color | Color selector and native color flag | Generic |
| Choose a card definition / texture | Enabled card-selector choices | Generic |
| Name a card | Card-name token selector; optional local catalog | Generic |
| Name a creature / artifact / land | Source-filtered name-token catalog | Generic |
| Name a nonland / nonbasic / noncreature nonland | Source-filtered name-token catalog | Generic |
| Choose a creature type | Native enum ID selector | Generic |
| Choose a land / nonbasic land type | Native enum ID selector | Generic |
| Choose a planeswalker type | Native enum ID selector | Generic |
| Choose a card type | Native property ID selector | Generic |
| Choose a counter type | Recovered counter-property ID selector | Generic |
| Exclude listed values / choose only listed values | List selector exception/inversion filter | Generic |
| Wish / choose outside-game card definition | Wish selector with explicit local definitions | Generic |
| Choose one of two piles | Native legacy pile selector | Generic |
| Spend a chosen color from mana pool | `ManaChoice` | Generic |
| Spend mana from a particular source entry | `ManaChoice(..., mana_id=...)` | Generic |
| Snow mana | Mana-entry snow flag and source ID; server validates spending | State |
| Restricted mana / riders | Source, restriction text, and raw properties | Dedicated |
| Hybrid / Phyrexian / alternate payment decisions | Advertised menu, color, number, or yes/no choices | Generic |
| Sacrifice / discard / exile for a cost | Advertised card actions or target groups | Generic |
| Tap creatures / convoke / crew / saddle | Advertised actions/groups; local batches when flagged | Generic |
| Exile cards / delve / escape payment | Advertised actions/groups and numeric prompts | Generic |
| Remove counters / loyalty / energy payment | Server counter state plus offered selectors/actions | Generic |
| Pay life / decline payment | Numeric or prompt choice offered by server | Generic |
| Cancel an unfinished payment | Advertised Cancel button | Generic |

## Resolution, zones, and persistent state

| Interaction | Implementation | Coverage |
| --- | --- | --- |
| Accept / decline optional trigger | Enabled prompt button | Generic |
| Choose among replacement or prevention effects | Advertised effect actions / selectors | Generic |
| Order triggers | Advertised ordering descriptor or sequential card actions | Generic |
| Counter a spell / counter an ability | Card action targeting advertised stack object | Generic |
| Copy a spell / choose new targets | Advertised stack action and subsequent target groups | Generic |
| Choose a target for a resolving delayed trigger | Fresh advertised action/selector | Generic |
| Choose a library-search result / fail to find | Offered cards and optional bounds / Done | Generic |
| Scry / surveil / select top or bottom | Offered card actions and final confirmation | Generic |
| Arrange revealed cards / library order | Offered ordering or sequential card actions | Generic |
| Reveal a card / choose from revealed cards | Offered cards/actions; raw messages retained | Generic |
| Split cards / adventure / modal faces | `actions_for(parent)` follows server facet IDs | Generic |
| Transform / copy / clone | Server object properties and advertised actions | State |
| Face-down / morph / manifest / cloak / disguise | Explicit nameless state; printed subtypes not assumed | Dedicated |
| Room doors / locked versus chosen-half names | Nested definition metadata and current door flags | Dedicated |
| Dynamic creature subtypes | Ranged enum IDs replace printed type metadata | Dedicated |
| Unknown dynamic subtype | Metadata stays unresolved rather than inheriting a false printed type | Dedicated |
| Lost abilities / lost changeling | Authoritative remove-text flag clears changeling interpretation | Dedicated |
| Added / removed counters | Immutable server counter containers | State |
| Poison / energy / experience / other player counters | Player Thing properties and counters | State |
| Monarch / initiative / Ring-bearer / day-night state | Server properties/effect objects preserved | State |
| Dungeon identity and repeated completions | Typed `Observation.dungeons` | Dedicated |
| Commander identities / owners / partner commanders | Typed `Observation.commanders` | Dedicated |
| Commander damage by player | Commander damage matrix | Dedicated |
| Attachments / mutate / meld relationships | Immutable server IDs and properties; no local reconstruction of rules | State |
| Extra turns / extra combats / skipped phases | Follow server phase/status changes | State |
| Emblems / continuous effects / tokens | Server Thing objects and attributes | State |

## Combat and locally staged decisions

| Interaction | Implementation | Coverage |
| --- | --- | --- |
| Declare attackers | Advertised card actions | Generic |
| Choose attacked player / planeswalker / battle | Action destination or advertised target group | Generic |
| Declare several attackers before Done | `submit_batch()` on actions flagged local | Dedicated |
| Declare blockers / multi-block | Advertised actions and target groups | Generic |
| Order blocking participants | `OrderingChoice` when descriptor is supplied | Generic |
| Assign combat damage | `DamageChoice`, per-recipient bounds and total | Generic |
| Trample / alternate recipients / damage to defending object | Offered damage recipients; server determines legal bounds | Generic |
| First strike and regular damage steps | Separate server observations; no local scheduling | Generic |
| Several damage sources before confirmation | Separate source keys or one validated batch | Dedicated |
| Arrange cards into three piles | Advertised per-card pile actions in a local batch | Dedicated |
| Stage several searches/selections before Done | Local actions followed by terminal prompt, only when advertised | Dedicated |
| Change a draft before sending it | Build/edit caller's batch before `submit_batch()` | Generic |
| Reject an invalid later choice before sending earlier choices | Whole-batch validation before first write | Dedicated |
| Partial write during a staged sequence | Session fails; no automatic retry or atomic-delivery claim | Dedicated |

## Game and match controls

| Interaction | Implementation | Coverage |
| --- | --- | --- |
| Undo the last undoable action | `game.undo()`; server decides whether permitted | Dedicated |
| Disable all automatic yields | `yield_control(DISABLE_ALL)` | Dedicated |
| Toggle yield through this turn | `yield_control(THROUGH_THIS_TURN)` | Dedicated |
| Toggle automatic trigger stacking | `yield_control(AUTO_STACK_TRIGGERS)` | Dedicated |
| Toggle bluffing preference | `yield_control(DISABLE_BLUFFING)` | Dedicated |
| Toggle yield through opponent-controlled actions | `yield_control(THROUGH_OPPONENT_ACTIONS)` | Dedicated |
| Read object-specific automatic yields | Typed `Observation.auto_yields` | Dedicated |
| Remove an advertised object-specific yield | `game.remove_yield(id)` | Dedicated |
| Set per-player phase stops | `update_stops(stops=...)`, native player indices and phase masks | Dedicated |
| Set per-player yield-until-step preferences | `update_stops(yields=...)` | Dedicated |
| Set automatic-yield preference flags | `update_stops(preferences=...)` | Dedicated |
| Read active player and displayed yield settings | Decoded PlayerStatus fields | Dedicated |
| Concede the current game | `game.concede()` | Dedicated |
| Concede the match | `client.concede_match(match)` | Dedicated |
| Detach game while keeping login alive | `game.detach()` | Dedicated |
| Create a single-game direct match | `challenge(..., best_of=1)` | Dedicated |
| Create a first-two-wins direct match | `challenge(..., best_of=3)` using native `BES` code | Dedicated |
| Submit revised deck between games | `submit_sideboard()` with complete main/sideboard list | Dedicated |
| Handle sideboard rejection details | Typed `SideboardResult.errors` | Dedicated |
| Handle sideboard deadline expiry | `SideboardResult.expired` | Dedicated |
| Wait for game two / three | `wait_for_game(..., after_game_id=...)` | Dedicated |
| Observe game result / playing time / winners | `game.result`, typed result attached to `GameMessage` | Dedicated |
| Reject stale choices / completed game actions | Session/game/revision checks; game-over invalidation | Dedicated |
| Resume after explicit re-login | Existing account login and `connect_game()` | Dedicated |

## Verification boundary

The new APIs use serializers and construction code recovered from client
3.4.159.4704, including `MatchType`, `MagicPropertyHelper`, game state elements,
sideboard handlers, and global game controls. Synthetic framed-backend tests
cover each new message family, validation failures, revisions, batches, route
identity, and game transitions. Saved two-account captures are replayed offline.

This remains an experimental library. No live advanced-mechanic or best-of-three
acceptance test has been run for this expansion, and there is no certification
for every card. Future descriptor changes, missing local definitions, unusual
copy/name/type combinations, and hidden information remain discovery boundaries.
Raw attributes/messages are retained; unknown choice encodings are not guessed.

Tournament/league entry, drafting, queue management, replays, spectators, trading,
and a local Magic rules engine are outside this gameplay expansion. The catalog
also contains older wire messages that the examined client does not use for
ordinary decisions; their presence does not imply a missing card handler.
