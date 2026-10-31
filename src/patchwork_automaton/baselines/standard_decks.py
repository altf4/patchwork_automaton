"""Exact Hartford Standard lists, with mainboards ready for RustMage matches."""
from importlib.resources import files
import json

from ..core import CardSpec, DeckSpec
from ..properties import freeze


_DATA = json.loads(files('patchwork_automaton').joinpath('data/standard-baselines.json').read_text())
STANDARD_DECKS = freeze(_DATA['decks'])


def standard_deck(deck_id, include_sideboard=False):
    """Return the registered 60 cards; optionally include the registered 15-card sideboard.

    Sideboards are useful for inspection/export but cannot be passed to the
    current RustMage match adapter. Card names retain their full printed faces.
    """
    if deck_id not in STANDARD_DECKS:
        raise ValueError('Unknown Standard baseline deck: ' + str(deck_id))
    row = STANDARD_DECKS[deck_id]
    if sum(c['quantity'] for c in row['mainboard']) != 60:
        raise ValueError('Standard baseline requires exactly 60 mainboard cards')
    if sum(c['quantity'] for c in row['sideboard']) != 15:
        raise ValueError('Standard baseline requires exactly 15 sideboard cards')
    sections = ((False, 'mainboard'), (True, 'sideboard')) if include_sideboard else ((False, 'mainboard'),)
    return DeckSpec(tuple(CardSpec(c['name'], c['quantity'], sideboard=side)
                          for side, section in sections for c in row[section]), row['archetype'])
