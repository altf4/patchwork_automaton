"""Live two-account protocol test. Minimal choices belong only in this harness."""
import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import struct
import time

from examples.idle_session import credentials
from patchwork_automaton import (MTGOClient, Observation, PromptChoice,
                                CardChoice, GameMessage, StaleDecision)


class TestClient(MTGOClient):
    def __init__(self, label, output, **kwargs):
        super().__init__(**kwargs)
        self.label, self.output = label, output
        self.counts = Counter()
        self.state_count = 0

    async def _dispatch(self, opcode, payload):
        self.counts[opcode] += 1
        if opcode in (960, 959, 123, 124, 125, 944, 945, 980, 2463, 1155, 108, 967):
            print(self.label, self.catalog.by_opcode[opcode]['name'], flush=True)
        if opcode == 1153:
            # Gameplay payloads only; authentication is never captured.
            self.state_count += 1
            (self.output / f'{self.label}-{self.state_count:05}.bin').write_bytes(payload)
        await super()._dispatch(opcode, payload)


def choose(game, obs):
    prompt = obs.prompt
    player = game.player_index
    if prompt is None or player is None or prompt['player'] not in (player, 255):
        return None
    buttons = [b for b in prompt['buttons'] if b['enabled']]
    if 'discard' in prompt['text'].lower():
        for action in obs.actions:
            if action.label.lower() == 'discard':
                return CardChoice(action.card_id, action.action_id), 'Discard'
    for label in ('play', 'keep'):
        for button in buttons:
            if button['label'].lower() == label:
                return PromptChoice(button['code']), button['label']
    if prompt['phase'] in (4, 10):
        for action in obs.actions:
            card = obs.cards[action.card_id]['properties']
            # Hand zone IDs include the player index in the low byte.
            if (card.get('CONTROLLER') == player and card.get('ZONE') == 1000448 + player
                    and action.label.lower() == 'play'):
                return CardChoice(action.card_id, action.action_id), 'Play land'
    for label in ('ok', 'done', 'next', 'combat', 'end turn', 'yes', 'no'):
        for button in buttons:
            if button['label'].lower() == label:
                return PromptChoice(button['code']), button['label']
    for button in buttons:
        if button['code'] == prompt.get('default'):
            return PromptChoice(button['code']), button['label']
    return None


async def play(label, game, result, finished, enough):
    async for event in game.events():
        if isinstance(event, GameMessage):
            if event.opcode in (4632, 4485):
                print(label, event.name, flush=True)
                if event.opcode == 4632:
                    result['game_over_received'][label] = True
                if event.opcode == 4485:
                    result['results_received'][label] = True
                if all(result['results_received'].values()):
                    finished.set()
            continue
        obs = event
        result['observations'][label] += 1
        result['last_turn'][label] = obs.prompt['turn'] if obs.prompt else None
        decision = choose(game, obs)
        if decision is None:
            if (obs.prompt and obs.prompt['player'] in (game.player_index, 255)
                    and not ('waiting for' in obs.prompt['text'].lower())):
                print(label, 'Waiting on unmodeled prompt:', obs.prompt['text'],
                      'buttons:', [(b['code'], b['label']) for b in obs.prompt['buttons']],
                      'selectors:', obs.prompt['selectors'], flush=True)
            continue
        choice, description = decision
        try:
            await game.submit(obs.decision_id, choice)
        except StaleDecision:
            continue
        result['actions'][label] += 1
        if description == 'Play land':
            result['lands_played'][label] += 1
        print(label, 'turn', obs.prompt['turn'], description, flush=True)
        if (min(result['lands_played'].values()) >= 1
                and min(result['actions'].values()) >= 8
                and max(result['last_turn'].values()) >= 3):
            enough.set()
        if sum(result['actions'].values()) >= 100:
            enough.set()


async def run(args):
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    strings = json.loads(args.strings.read_text()) if args.strings else ()
    result = {'started_at': time.time(), 'authenticated_accounts': 0,
              'actions': {'account1': 0, 'account2': 0},
              'lands_played': {'account1': 0, 'account2': 0},
              'observations': {'account1': 0, 'account2': 0},
              'last_turn': {'account1': 0, 'account2': 0},
              'game_over_received': {'account1': False, 'account2': False},
              'results_received': {'account1': False, 'account2': False}}
    match, tasks, first, second = None, [], None, None
    try:
        async with TestClient('account1', args.output, strings=strings) as first, TestClient(
                'account2', args.output, strings=strings) as second:
            await first.login(*credentials(args.first))
            await second.login(*credentials(args.second))
            result['authenticated_accounts'] = 2
            decks1, decks2 = await asyncio.gather(first.list_decks(), second.list_decks())
            def pick(decks, preferred=None):
                preferred_deck = next((d for d in decks.decks if d.name == preferred and d.cards), None)
                if preferred_deck is not None:
                    return preferred_deck
                valid = []
                for deck in decks.decks:
                    try:
                        deck.format
                    except ValueError:
                        continue
                    if deck.cards:
                        valid.append(deck)
                if not valid:
                    raise ValueError('Account has no playable saved deck')
                return valid[0]
            deck1, deck2 = pick(decks1, 'Legacy Burn'), pick(decks2, 'Legacy Burn')
            result['deck_card_counts'] = [sum(c.quantity for c in d.cards if not c.sideboard) for d in (deck1, deck2)]
            print('Selected deck sizes:', result['deck_card_counts'], flush=True)
            existing1, existing2 = await asyncio.gather(first.current_matches(), second.current_matches())
            shared = next((m for m in existing1 if m.state not in (10, 11, 12)
                           and any(n.token == m.token for n in existing2)), None)
            if shared is not None:
                match = shared
                incoming = next(m for m in existing2 if m.token == match.token)
                print('Resuming the existing test match:', match.id, 'state:', match.state, flush=True)
                if match.games:
                    games = await asyncio.gather(first.connect_game(match), second.connect_game(incoming))
                else:
                    games = await asyncio.gather(first.start_match(match, player_ids=(first.user_id, second.user_id)),
                                                 second.wait_for_game(incoming))
            else:
                match = await first.challenge(second.user_id, deck1, format='CFREEFORM')
                print('Direct challenge created; match', match.id, flush=True)
                incoming = await second.wait_for_match(match.token)
                games = await asyncio.gather(first.wait_for_game(match), second.accept_challenge(incoming, deck2))
            result.update(match_id=match.id, game_id=games[0].id,
                          player_indices=[g.player_index for g in games], connected_game=True)
            print('Both accounts attached; player indices:', result['player_indices'], flush=True)
            finished, enough = asyncio.Event(), asyncio.Event()
            tasks = [asyncio.create_task(play(label, game, result, finished, enough))
                     for label, game in zip(('account1', 'account2'), games)]
            sufficient = asyncio.create_task(enough.wait())
            completion = asyncio.create_task(finished.wait())
            try:
                done, _ = await asyncio.wait([*tasks, sufficient, completion], timeout=90,
                                             return_when=asyncio.FIRST_COMPLETED)
                for task in tasks:
                    if task in done:
                        await task
                if not finished.is_set():
                    print('Finishing the controlled test by account2 concession.', flush=True)
                    await second.concede_match(incoming)
                    result['concession_confirmed'] = True
                    try:
                        await asyncio.wait_for(finished.wait(), 15)
                    except TimeoutError:
                        result['game_over_wait_timed_out'] = True
            finally:
                for task in [*tasks, sufficient, completion]:
                    task.cancel()
                await asyncio.gather(*tasks, sufficient, completion, return_exceptions=True)
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error))
        # Public library errors contain codes/metadata, never credential fields.
        print('Test failed:', type(error).__name__, str(error), flush=True)
    finally:
        result['finished_at'] = time.time()
        result['success'] = bool(result.get('connected_game') and min(result['actions'].values()) > 0
                                 and all(result['results_received'].values()))
        path = args.output / 'result.json'
        path.write_text(json.dumps(result, indent=2))
        path.chmod(0o600)
        print(json.dumps(result), flush=True)
    return 0 if result['success'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', type=Path, required=True)
    parser.add_argument('--second', type=Path, required=True)
    parser.add_argument('--strings', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(asyncio.run(run(parser.parse_args())))
