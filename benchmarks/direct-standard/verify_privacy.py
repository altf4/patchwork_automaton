"""Check opening-hand isolation and reject foreign/stale direct-bridge views."""
import argparse
import ctypes as C
import json
from pathlib import Path

import numpy as np
from patchwork_automaton.baselines.standard_decks import STANDARD_DECKS, standard_deck
from probe_direct import load
from verify_tensors import arrays, cjson, configure


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', type=Path, required=True)
    a = p.parse_args()
    lib = load(a.library.resolve(strict=True))
    configure(lib)
    decks = [standard_deck(name) for name in STANDARD_DECKS]
    expanded = [[card.name for card in deck.cards for _ in range(card.quantity)] for deck in decks]
    sessions, views = [], []
    try:
        for opponent in (1, 2):
            config = dict(decks=[expanded[0], expanded[opponent]], seed=22,
                          starting_player='P0', deck_rules='Constructed')
            session = lib.ds_new(json.dumps(config).encode())
            if not session:
                raise RuntimeError(lib.ds_error().decode())
            sessions.append(session)
            pair = [lib.ds_observe(session, viewer) for viewer in (0, 1)]
            if not all(pair):
                raise RuntimeError(lib.ds_error().decode())
            views.append(pair)
        obs = [[cjson(lib, lib.ds_debug_json(v)) for v in pair] for pair in views]
        for pair in obs:
            for viewer in (0, 1):
                hand = [o for o in pair[viewer]['objects'] if o['zone'] == 'Hand']
                assert len(hand) == 7 and all(o['owner'] == f'P{viewer}' for o in hand), hand
                assert pair[viewer]['players'][1-viewer]['hand_count'] == 7
        assert obs[0][0] == obs[1][0], 'Opposing deck affected player 0 private opening view'
        count = C.c_size_t()
        assert not lib.ds_prepare(views[0][1], False, C.byref(count)), 'Non-actor view exposed actions'
        encoded = []
        for pair in views:
            assert lib.ds_prepare(pair[0], False, C.byref(count)), lib.ds_error()
            encoded.append(arrays(lib, pair[0]))
        for key in encoded[0]:
            assert np.array_equal(encoded[0][key], encoded[1][key]), f'Opponent hidden cards changed {key}'
        assert not lib.ds_apply_choice(sessions[1], views[0][0], 0), 'Foreign session view accepted'
        assert lib.ds_apply_choice(sessions[0], views[0][0], 0), lib.ds_error()
        assert not lib.ds_apply_choice(sessions[0], views[0][0], 0), 'Stale view accepted'
        print(json.dumps(dict(status='complete', tests=['both private hands', 'hidden opponent deck',
            'tensor isolation', 'non-actor actions', 'foreign session', 'stale revision'])))
    finally:
        for pair in views:
            for view in pair:
                if view:
                    lib.ds_view_free(view)
        for session in sessions:
            lib.ds_free(session)


if __name__ == '__main__':
    main()
