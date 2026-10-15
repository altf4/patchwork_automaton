# Patchwork Automaton

An experimental Python 3.11+ async gameplay library for MTGO and local XMage.
The MTGO backend talks directly to its server over TLS. It runs on Linux without Wine or the running MTGO client.
The server runs the rules; callers choose actions. The package contains no bot
policy, deck strategy, or local Magic rules engine.

Version 0.4.0 adds an opt-in common game API, XMage backend and local self-play
episode tools. See [XMage setup and usage](docs/xmage.md). The MTGO implementation builds on the authenticated connection and Legacy Burn
solitaire discovery in [docs/discovery](docs/discovery). The
[interaction checklist](docs/interaction-coverage.md) maps 135 gameplay patterns to the general API and dedicated controls. Its wire definitions
are pinned to client **3.4.159.4704**. It is not yet a complete implementation
of every interaction type.

## Install

An opt-in [numerical semantic schema](docs/semantics.md) supplies keyword flags
and parameterized card abilities for structured neural-network features.
The [Keras policy example](docs/neural.md) runs an untrained numerical model in
local XMage games, including targets, payments and combat. Use uv with the
`neural` and `xmage` extras for this optional component.
The [CPU self-play trainer](docs/training.md) learns from terminal win/loss/draw
rewards and evaluates saved checkpoints against a fixed scripted opponent.
An explicit [weak Modern Burn baseline](docs/burn-baseline.md) uses a pinned
60-card tournament list and deterministic, deliberately exploitable decisions.

```sh
python3 -m pip install -e .
```

The MTGO runtime uses only the Python standard library. XMage adds optional
WebSocket and Java dependencies described in [the setup guide](docs/xmage.md). The message catalog and
property names ship inside the package; installing a wheel does not require
the discovery directory. An optional local string table provides labels for
server-supplied string references. Without it, unresolved strings appear as
`#str:<index>` and IDs remain available.

## Read decks and observe a game

```python
import asyncio
import os

from patchwork_automaton import MTGOClient, Observation

async def main():
    async with MTGOClient() as client:
        await client.login(os.environ['MTGO_USERNAME'], os.environ['MTGO_PASSWORD'])
        saved = await client.list_decks()
        deck = next(d for d in saved.decks if d.name == 'Legacy Burn')
        game = await client.create_solitaire(deck)

        async for event in game.events():
            if isinstance(event, Observation):
                print(event.revision, event.prompt)
                print(event.actions)
                # Your application decides whether and what to submit.
            else:
                print(event.name)

asyncio.run(main())
```

Creating solitaire starts a real game. `list_decks()` also returns binders and
wishlists; `Deck.format` rejects entries whose metadata does not identify a
playable deck. For a deck assembled by the caller, pass `tuple[DeckItem, ...]`
and an explicit server format identifier:

```python
from patchwork_automaton import DeckItem

cards = (DeckItem(catalog_id=123, annotation=0, permission=0,
                  quantity=4, sideboard=False),)  # Supply your complete deck.
game = await client.create_solitaire(cards, format='CFREEFORM')
```

Newly gifted decks can have opaque `CatalogID=...` metadata instead of a saved
format; pass `format='CLEGACY'` (or the appropriate format) explicitly for
those entries. No card or format whitelist exists. The server decides deck validity and
collection eligibility. `DeckList.response_code` retains the original server
code (the observed complete-list response was 5).

## Submit a caller-selected action

For structured targets, modes, selectors, mana, combat, and counters, see
[the interaction API](docs/interactions.md). The APIs below remain available
for callers that already know the wire values.

```python
from patchwork_automaton import CardChoice, PromptChoice

# Use an enabled button code from observation.prompt['buttons'].
receipt = await game.submit(observation.decision_id, PromptChoice(button_code))

# Alternatively, use a card/action ID pair from observation.actions.
receipt = await game.submit(observation.decision_id, CardChoice(
    card_id=card_id,
    action_id=action_id,
    targets=target_ids,
    modes=mode_bytes,
    extra_mana=0,
))
```

The examples above are alternatives for a single decision. `submit()` checks
the session, game, revision, advertised ID, and duplicate submissions. A new
revision invalidates old choices even if the server timestamp did not change.
Snapshots and nested attributes are immutable.

`Submission.status == 'sent'` means the transport write completed. It does not
prove server acceptance. Consume subsequent states and `GameMessage` events
to observe outcomes. There is no automatic action retry. Caller-selected
targets, modes, flags, and extra mana use the recovered wire encoding; the
library does not calculate target legality or select costs. A target separator
is the unsigned value `0xfffffffe`; its grouping must match the advertised
interaction. Do not infer target grouping from card text alone.

The prompt includes its player index and the status includes waiting/priority
indices. `game.player_index` maps this account to the game server's player
order; it is `None` until that order arrives. Callers must use these fields to
decide whether to respond. Hand-zone IDs include a player index in the low byte.

## Direct matches and gift products

With both accounts logged in using separate client instances:

```python
challenge = await first.challenge(second.user_id, first_deck, format='CFREEFORM')
incoming = await second.wait_for_match(challenge.token)
first_game, second_game = await asyncio.gather(
    first.wait_for_game(challenge),
    second.accept_challenge(incoming, second_deck),
)
```

The challenger sends the native player-order and host-start messages after
acceptance. The reader also handles game-ready acknowledgements. This is a
D1V1 direct challenge with spectators disabled. Pass `best_of=3` for a first-two-wins
match; sideboarding and next-game APIs are described in
[interactions.md](docs/interactions.md). To finish a test explicitly, `await second.concede_match(incoming)` waits for the server's
concession acknowledgement. Final game results are retained on `game.result` and attached to `GameMessage` events;
the conceding player may receive results without a separate GameOver message.

`collection()` returns the account's owned `CollectionItem` records, including
unopened gifts. `open_product(item)` explicitly consumes an owned sealed
product and returns collection deltas. Identify the product using catalog
data before opening it. These inventory mutations are never retried
automatically. The New Account Starter Kit and its five gifted decks were
opened successfully through this API during the two-account test.

For unmodeled interactions, `game.send_raw(message_name, payload)` sends a
catalog-framed game message over the established route. It deliberately
bypasses decision validation. It is an extension point, not a claim that those
interactions are implemented or tested.

## Coverage

| Area | This release |
| --- | --- |
| TLS login, heartbeat, time sync, ACK envelopes, graceful logoff | Implemented; login/keepalive exercised live |
| Saved deck list and ordinary current-match list | Implemented |
| Own collection and explicit owned-product opening | Implemented; starter kit and five gift decks exercised live |
| Solitaire creation and game-server attachment | Implemented; original wire path exercised live |
| Direct challenge, acceptance, host start, ready ACK, concession | Implemented; two-account Legacy Burn game exercised live |
| Full/fragmented state, per-priority diffs, checksum validation | Implemented |
| Card attributes, counter maps, stack objects, split-card facets | Decoded and exposed as immutable objects |
| All eight current UI descriptors | Decoded; typed selector responses with bounds/options checks |
| Target groups and relationship constraints | Structured selections, conditional candidate filtering, deferred targets, distributions |
| Modal actions | Available modes, repetitions, weights, fixed combinations, mode-dependent target groups |
| Mana and costs | Source-specific mana payment; generic numeric/card selections for costs and alternatives |
| Combat | Advertised attacker/blocker actions, per-creature ordering and damage assignments, final confirmation |
| Unknown state elements, unknown UI selectors, other game messages | Preserved as raw data |
| Advanced interaction validation | Client-derived synthetic fixtures and saved-state replay; new shapes have not been exercised live |
| Game-server account/index mapping | Implemented; two-player order exercised live |
| Sideboarding, subsequent games, typed game results | Implemented; offline validation of acceptance/rejection, expiration, and game transitions |
| Undo, yield toggles, phase stops, game concession | Explicit native controls; no automatic retries |
| Commander, dungeon, auto-yield state | Typed immutable descriptors |
| Tournament structures and joining queues/events | Not implemented |
| Reconnect/resume | Explicit new login and attach; no transparent session resume |

Arbitrary card IDs and decks can travel through the library. **Playing every
Modern card's complete interaction sequence is not yet guaranteed.** The
shared choice types from the examined client are implemented, but exhaustive
card coverage requires live acceptance tests and edge-case fixtures. Local
card-name/type definitions are optional and needed for some selector menus
and relationship checks. The library implements no local rules or cost solver.
Unknown attribute encodings or incompatible schemas raise `ProtocolError`
instead of guessing.

## Connection and event lifecycle

Use one client per account connection and the async context manager to close
it. The background reader answers heartbeats while the caller is deciding.
`AuthenticationError` exposes the response code and challenge type; `login`
accepts `challenge=` for an explicitly supplied code. No passwords are stored
on the client and the core reads no credential files or writes runtime files.

Each game has one event consumer and a bounded queue (default 256). Consume
`game.events()` continuously. Falling behind raises `EventOverflow` through
the connection failure rather than silently dropping game state. Protocol
errors, disconnects, and request timeouts invalidate the connection and wake
pending operations. Request timeouts are terminal because a late uncorrelated
response could otherwise satisfy a subsequent request.

After disconnecting, create a new `MTGOClient`, explicitly log in, call
`current_matches()`, and `connect_game(match)`. This resynchronizes from server
state; old decisions cannot be reused. Authentication refusals are never
retried automatically. A timeout after sending a game action has an unknown
outcome; resynchronize before deciding another action.

Observations retain numeric IDs, card properties, clocks, unknown elements,
and reconstructed state bytes. `GameMessage` preserves other routed messages,
including game-over and result payloads. State decoding is intentionally
strict about boundaries and checksums. No authentication payload logging is
provided.

## Examples and discovery tools

From the repository root:

```sh
PYTHONPATH=src:. python3 -m examples.idle_session --credentials ../.creds
PYTHONPATH=src:. python3 -m examples.inspect_solitaire \
  --credentials ../.creds --deck-id YOUR_DECK_ID --strings .local/strings.json
```

The idle example makes read-only deck/match queries and keeps the connection
alive. `scripts/start-session.sh` runs it as a user service. Credential files
and `.local/` caches are ignored by Git. Do not run a second account connection
concurrently with the service.

To extract labels once from a locally installed client assembly, install
`dnfile` in a discovery environment, then:

```sh
python3 scripts/extract-strings.py /path/to/assembly.dll .local/strings.json
```

Pass the resulting JSON list as `MTGOClient(strings=strings)`. This optional
discovery step does not require running the client, and the library does not
redistribute its string-table resource.

`research/` preserves the earlier experimental service and solitaire policy
as discovery reference. It also contains `direct_match.py`, a live two-account
harness that keeps hands, plays lands, passes priority, discards if necessary,
and finishes by concession. It runs only when explicitly invoked and is
excluded from the installed package. Stop the idle service before running it:

```sh
PYTHONPATH=src:. python3 -m research.direct_match \
  --first ../.creds --second ../.creds_2 --strings .local/strings.json \
  --output .local/direct-match
```

## Development

```sh
PYTHONPATH=src:. python3 -m unittest discover -s tests -v
python3 -m pip wheel --no-deps --no-build-isolation .
```

Tests use synthetic framed backend responses rather than credentials or live
accounts. They cover public workflows, envelope negotiation, stale choices,
disconnects, cancellation/timeouts, queue overflow, codec boundaries, and
advanced interaction shapes. The 0.3.0 implementation was also checked by
replaying 82 previously captured observations without reconnecting accounts.
No CI has been configured.

Source provenance and protocol evidence are in `docs/discovery/`. The
repository's GPL-3.0 license applies; deployment-specific protocol metadata
may need updating when the official client changes.
