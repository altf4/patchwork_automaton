# Local XMage and the common gameplay API

The local integration uses the official XMage 1.4.62-V1 server, a headless
Nexus v0.4.7 Java proxy, and an async Python WebSocket adapter. No graphical
client runs. The Java proxy speaks XMage's native JBoss/Java protocol; Python
speaks its JSON protocol. This is a separate backend from MTGO.

## Install and operate the local test server

From the repository root, on Linux x86_64 with Python 3.12 or newer:

```sh
python3 scripts/xmage-local.py install
python3 scripts/xmage-local.py start
python3 scripts/xmage-local.py status
python3 scripts/xmage-local.py stop
```

`install` downloads pinned archives, verifies SHA-256 hashes and extracts them
under ignored `.local/xmage/`. Java 17 is included there; no system Java change
is required. Server code and the bundled card database remain unmodified. The
server configuration binds loopback, fixes the secondary port and retains the
stock authentication-disabled setting. Local test usernames need no password.
Use a dedicated process/port set if running more than this one instance.

`start` launches detached processes, waits for the live server socket and proxy
readiness, and leaves them running. Repeating it reuses the managed processes.
`stop` verifies recorded process identities before terminating their process
groups. Installation is explicit: starting never downloads software.

| Service | Address |
| --- | --- |
| Native XMage | `127.0.0.1:17171` (secondary `17172`) |
| Python gateway | `ws://127.0.0.1:8787` |
| Readiness | `http://127.0.0.1:8788/ready` |
| Logs | `.local/xmage/server.log`, `.local/xmage/proxy.log` |

These endpoints are local only. The setup does not create a service that starts
at boot. Stop the processes explicitly when finished.

## Exercise two headless seats

Install optional Python dependencies in your own virtual environment:

```sh
python3 -m pip install -e '.[xmage]'
PYTHONPATH=src python3 examples/xmage_selfplay.py --episodes 2 --limit 100
```

The example supplies a deliberately rudimentary controller outside the library.
Its 24 Mountain/36 Lightning Bolt deck uses Freeform Unlimited to test transport,
not Modern deck legality. It exercises mulligans, priority, land plays, casting,
targeting and mana payment, then concedes at the configured limit and waits for
both server results. It removes its table and closes both sessions on exit.
The server stays running. An ordinary `XMageSelfPlay` factory defaults to Modern;
supply a complete legal deck for that format.

## Common API

Existing imports such as `patchwork_automaton.MTGOClient` retain their behavior.
The new shared models are opt-in under `patchwork_automaton.core`:

```python
from patchwork_automaton.core import CardSpec, DeckSpec, Response
from patchwork_automaton.rl import XMageSelfPlay, EpisodeEnd

# Supply your own complete deck, then create an episode factory.
factory = XMageSelfPlay(deck)
try:
    runner = await factory.reset()
    while True:
        turn = await runner.next()
        if isinstance(turn, EpisodeEnd):
            break
        observation = turn.observation
        # Caller selects an offered option and any required numeric/structured value.
        response = your_controller(observation)
        await runner.submit(response)
finally:
    await factory.close()
```

`XMageClient` in `backends.xmage` also exposes login, deck resolution,
create/join/start/remove table, game handles, sideboard submission and native
requests. `MTGOBackend` in `backends.mtgo` wraps the existing client, solitaire
and direct-match APIs. MTGO deck resolution requires explicit collection catalog
IDs and permissions; XMage resolves card names or explicit printings.

A game handle provides `snapshot()`, `wait_for_decision()`,
`submit(decision.token, Response(option_id, value))`, `concede()` and
`wait_for_result()`. Decisions carry backend/session/game/actor/generation
identity. Responses from another game, session, actor or old decision are
rejected. No gameplay response is automatically retried. Disconnects invalidate
the session even if the external proxy later reconnects.

Observations include player life/counts/counters, visible objects, turn/phase,
active/priority player when known, pending choices and terminal outcomes. Each
seat sees its own server-provided view; the library never combines both hands.
Immutable `native` metadata preserves backend details. Opaque handles are not
interchangeable between engines.

The common API does not make the engines' decision sequences identical. MTGO
card options can require bundled targets/modes/distributions in their response;
XMage often asks these in separate callbacks. A controller must inspect the
option kind, constraints and backend. This release does not promise a single
trained policy can transfer between engines without further encoding work.

## Callback and validation coverage

| XMage callback | Response |
| --- | --- |
| Ask / choose pile | Boolean, native button labels / pile identity |
| Select / target | Advertised UUIDs, optional completion and special button |
| Ability picker | Ability UUID |
| Choice list | Actual server choice key/string |
| Amount | Integer with advertised bounds |
| Multiple amounts | Per-entry and total bounds, native string encoding |
| Mana / X mana | Mana-source clicks, special action, integer X, cancellation |
| User request dialog | Separately tracked dialog and offered button action |
| Game end | Per-seat server result; unknown result never becomes an assumed draw |

Mana source fallback clicks are marked as requiring server validation; they are
not a certified legal-action mask. Pooled mana-type selection and typed XMage
undo/yield convenience methods are not yet exposed. Advanced card scenarios,
combat, sideboarding and match transitions require further live tests. Native
metadata and `client.request(command, **args)` remain available for extensions;
raw requests bypass common decision validation. Unknown gameplay callbacks fail
explicitly rather than guessing a response.

The existing MTGO interaction checklist remains MTGO evidence. A broad semantic
mapping to XMage's generic callbacks is not proof that all 135 patterns or every
Modern card have completed live. No card-specific rules engine runs in Python.

## Optional RL integration

`DecisionRunner` follows actual pending actors, including repeated choices by
the same seat and concurrent mulligan prompts. A step answers one decision,
not one turn. Limits and timeouts truncate episodes instead of assigning losses.
Bounded trace capture is opt-in with `trace_limit`.

Install `.[rl,xmage]` to use `rl.envs.DecisionEnv` (Gymnasium, explicit opponent)
or `rl.envs.SelfPlayEnv` (PettingZoo AEC). Supply an encoder with
`observation_space`, `action_space`, `encode(Observation)` and
`decode(action, Observation) -> Response`. The library imposes no fixed object
or action cutoff. The encoder must handle dynamic choices and hidden information.
A background event loop keeps transport readers alive while a synchronous policy
thinks. Close the environment to release that thread and its sessions.

Wrappers use terminal win/loss rewards of +1/-1 and draws of zero; infrastructure
truncations and missing outcomes receive zero. Custom reward shaping belongs in
the caller. The stock server has no verified per-episode deterministic seed or
scenario injection API. `XMageSelfPlay.reset(seed=...)` rejects a seed request
explicitly; do not treat unseeded repeated games as exact replay.

## Provenance and validation

The runtime downloads (not redistributed in the Python wheel) are pinned with
SHA-256 hashes in `scripts/xmage-local.py`:

- [Official XMage release](https://github.com/magefree/mage/releases/tag/xmage_1.4.62V1),
  server reports `1.4.62-V1 (build: 2026-10-03 20:33)`; its bundled card database
  is pinned by the complete archive hash.
- [XMage Nexus v0.4.7](https://github.com/Adamychen/xmage-nexus/releases/tag/v0.4.7),
  Linux x86_64 Java runtime and proxy artifacts. Nexus source inspected at commit
  `850411ec4db71ea7c10f4c3869a63084760057c2` (MIT license). No proxy source is
  vendored or modified. The prebuilt proxy was tested against the official server.
- Upstream callback/response contracts inspected at XMage tag `xmage_1.4.62V1`,
  particularly `HumanPlayer`, `ChoiceImpl`, `MultiAmountMessage`, `PlayerAction`
  and `GameClientMessage`. The adapter checks server version and gateway protocol
  version `1` before allowing use.

On 2026-10-14 four fresh two-seat episodes each submitted 100 decisions, selected
five or six lands, exercised ask/select/target/mana callbacks and ended by explicit
concession with complementary per-seat win/loss results. This included two more
episodes after verifying shutdown and restart. Tables and sessions were cleaned
up between episodes. This validates headless game transport and lifecycle;
it is not evidence of natural wins, exhaustive mechanics or training throughput.
All 146 tests pass with the optional RL dependencies installed, including the
117 previous MTGO tests. The 0.4.0 wheel builds successfully. Additional
fixtures cover cross-backend freshness, callback races, malformed frames,
visibility, error handling, actor scheduling and RL wrapper lifecycle. No live
MTGO account was used during XMage development. No CI was added.

## Initial single-core benchmark (2026-10-14)

Six sequential games, each with 100 decision submissions total across both seats,
completed in **50.98 seconds: 7.06 games/minute**. One warm-up was excluded. Python,
proxy and server were pinned together to logical CPU 7 of an Intel Core Ultra 5
335; original CPU affinities were verified restored afterward. This is one
sequential worker with JVM internal threads sharing one CPU, not a single JVM
thread. Server startup and model inference/training are excluded.

| Per-game phase | Mean seconds |
| --- | ---: |
| Fresh sessions, table, decks and first decision | 3.329 |
| 100 gameplay decisions | 5.087 |
| Finish and receive both results | 0.062 |
| Table/session cleanup | 0.019 |
| Total | 8.496 |

Total per-game range was 8.268–8.613 seconds. Gameplay throughput was **19.66
decisions/second**, equivalent to **11.79 100-decision games/minute excluding
setup/finish/cleanup**. That latter figure is a phase-only extrapolation, not
measured sustained throughput with reused sessions. Five games ended by
concession; one ended naturally on action 100. Decisions include priority passes,
land/spell selections, targets, mulligan answers and mana payments. They are not
100 complete spell casts, or 100 decisions per seat. The Freeform Burn smoke
controller supplies actions without model inference.

Two preliminary harness attempts were excluded: a result-wait timeout after
conceding from a non-prompted seat, and an unhandled natural end on decision 100.
The final harness concedes from the prompted seat and handles natural end. All
six measured games completed with both results and no remaining proxy sessions;
this small sample is not a reliability soak test.

Reproduce with the local server already running and `[xmage]` installed:

```sh
PYTHONPATH=src:. python3 -m examples.xmage_benchmark \
  --games 6 --warmup 1 --actions 100 --one-cpu
```

The raw measured timings and configuration are saved in
[benchmarks/xmage-2026-10-14.json](benchmarks/xmage-2026-10-14.json). Runtime results
and CPU restoration metadata go under ignored `.local/xmage/`. CPU pinning
requires permission to adjust the managed local JVM processes.
