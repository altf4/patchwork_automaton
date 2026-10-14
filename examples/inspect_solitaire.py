"""Explicitly create a solo game and print observations; make no decisions."""
import argparse
import asyncio
import json
from pathlib import Path

from patchwork_automaton import MTGOClient, Observation
from examples.idle_session import credentials


async def run(args):
    strings = json.loads(args.strings.read_text()) if args.strings else ()
    async with MTGOClient(strings=strings) as client:
        await client.login(*credentials(args.credentials))
        decks = await client.list_decks()
        deck = next(d for d in decks.decks if d.id == args.deck_id)
        game = await client.create_solitaire(deck)
        async for event in game.events():
            if isinstance(event, Observation):
                print(f'Revision {event.revision}: {event.prompt}', flush=True)
                for action in event.actions:
                    print(action.card_id, action.action_id, action.label, flush=True)
            else:
                print(event.name, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--credentials', type=Path, required=True)
    parser.add_argument('--deck-id', type=int, required=True)
    parser.add_argument('--strings', type=Path)
    asyncio.run(run(parser.parse_args()))
