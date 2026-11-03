"""Exact Standard registrations, selected explicitly for each experiment."""
from importlib.resources import files
import json

from ..core import CardSpec, DeckSpec
from ..properties import freeze


_DATA = json.loads(files('patchwork_automaton').joinpath('data/standard-baselines.json').read_text())
STANDARD_DECKS = freeze(_DATA['decks'])
_HARTFORD = json.loads(files('patchwork_automaton').joinpath('data/hartford-standard-decklists.json').read_text())
HARTFORD_DECKS = freeze(_HARTFORD['decks'])
DECK_CATALOGS = freeze({'standard': STANDARD_DECKS, 'hartford': HARTFORD_DECKS})


def standard_deck(deck_id, include_sideboard=False, *, catalog='standard'):
    """Return the registered 60 cards; optionally include the registered 15-card sideboard.

    Sideboards are useful for inspection/export but cannot be passed to the
    current RustMage match adapter. Card names retain their full printed faces.
    """
    if catalog not in DECK_CATALOGS:
        raise ValueError('Unknown Standard deck catalog: ' + str(catalog))
    decks = DECK_CATALOGS[catalog]
    if deck_id not in decks:
        raise ValueError('Unknown Standard baseline deck: ' + str(deck_id))
    row = decks[deck_id]
    if sum(c['quantity'] for c in row['mainboard']) != 60:
        raise ValueError('Standard baseline requires exactly 60 mainboard cards')
    if sum(c['quantity'] for c in row['sideboard']) != 15:
        raise ValueError('Standard baseline requires exactly 15 sideboard cards')
    sections = ((False, 'mainboard'), (True, 'sideboard')) if include_sideboard else ((False, 'mainboard'),)
    return DeckSpec(tuple(CardSpec(c['name'], c['quantity'], sideboard=side)
                          for side, section in sections for c in row[section]), row['archetype'])
