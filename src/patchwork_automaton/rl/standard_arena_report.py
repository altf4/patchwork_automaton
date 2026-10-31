"""Pure, perspective-aware accounting for Standard arena game results.

Input rows retain the RustMage runner's raw status: ``win`` means P0 won and
``loss`` means P1 won.  Entrant wins are therefore counted from ``learner_won``;
this module validates that field against the terminal outcome.
Cutoffs and errors remain attempts, never fabricated losses or draws.
"""

from collections import Counter, defaultdict
import math


STATUSES = frozenset(('win', 'loss', 'draw', 'cutoff', 'error'))
REQUIRED = ('cell_id', 'index', 'seed', 'learner_label', 'learner_deck',
            'opponent_label', 'opponent_deck', 'learner_seat',
            'starting_player', 'status')
_Z95 = 1.959963984540054


def _wilson(wins, attempts):
    """Wilson 95% interval for wins out of completed games; draws are non-wins."""
    if not attempts:
        return None
    p = wins / attempts
    z2 = _Z95 * _Z95
    divisor = 1.0 + z2 / attempts
    center = (p + z2 / (2.0 * attempts)) / divisor
    radius = (_Z95 / divisor) * math.sqrt(
        p * (1.0 - p) / attempts + z2 / (4.0 * attempts * attempts))
    return [0.0 if wins == 0 else max(0.0, center - radius),
            1.0 if wins == attempts else min(1.0, center + radius)]


def _reason_counts(rows, status):
    return dict(sorted(Counter(str(row.get('reason') or 'unspecified') for row in rows
                               if row['status'] == status).items()))


def _counts(rows):
    """Count a possibly empty subset without assuming a particular seat."""
    statuses = Counter(row['status'] for row in rows)
    completed = statuses['win'] + statuses['loss'] + statuses['draw']
    wins = sum(row['learner_won'] is True for row in rows
               if row['status'] in ('win', 'loss'))
    losses = statuses['win'] + statuses['loss'] - wins
    draws = statuses['draw']
    return dict(
        attempted=len(rows), completed=completed,
        wins=wins, losses=losses, draws=draws,
        cutoffs=statuses['cutoff'], errors=statuses['error'],
        adjudicated_draws=sum(bool(row.get('adjudicated')) for row in rows
                              if row['status'] == 'draw'),
        win_rate=wins / completed if completed else None,
        game_score=(wins + 0.5 * draws) / completed if completed else None,
        win_rate_wilson_95_interval=_wilson(wins, completed),
        draw_reasons=_reason_counts(rows, 'draw'),
        cutoff_reasons=_reason_counts(rows, 'cutoff'),
        error_reasons=_reason_counts(rows, 'error'))


def _stats(rows):
    result = _counts(rows)
    result['on_play'] = _counts([row for row in rows
                                 if row['starting_player'] == f"P{row['learner_seat']}"])
    result['on_draw'] = _counts([row for row in rows
                                 if row['starting_player'] != f"P{row['learner_seat']}"])
    return result


def _validate_rows(rows):
    cell_identities = {}
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Arena result must be a dictionary')
        missing = [field for field in REQUIRED if field not in row]
        if missing:
            raise ValueError(f'Arena result lacks required fields: {missing}')
        for field in ('cell_id', 'learner_label', 'learner_deck',
                      'opponent_label', 'opponent_deck'):
            if not isinstance(row[field], str) or not row[field]:
                raise ValueError(f'Invalid arena {field}')
        for field in ('index', 'seed'):
            if type(row[field]) is not int or row[field] < 0:
                raise ValueError(f'Invalid arena {field}')
        if type(row['learner_seat']) is not int or row['learner_seat'] not in (0, 1):
            raise ValueError('Invalid learner seat')
        if row['starting_player'] not in ('P0', 'P1'):
            raise ValueError('Invalid starting player')
        if row['status'] not in STATUSES:
            raise ValueError(f"Unknown arena status: {row['status']!r}")
        if row['status'] in ('win', 'loss'):
            winner = 0 if row['status'] == 'win' else 1
            if (type(row.get('winner_seat')) is not int
                    or row['winner_seat'] != winner
                    or row.get('draw') is not False
                    or type(row.get('learner_won')) is not bool
                    or row['learner_won'] is not (winner == row['learner_seat'])):
                raise ValueError('Inconsistent decisive arena outcome')
        elif row['status'] == 'draw':
            if (row.get('draw') is not True or row.get('winner_seat') is not None
                    or (row.get('learner_won') is not None
                        and row.get('learner_won') is not False)):
                raise ValueError('Inconsistent arena draw')
        identity = (row['learner_label'], row['learner_deck'],
                    row['opponent_label'], row['opponent_deck'])
        previous = cell_identities.setdefault(row['cell_id'], identity)
        if previous != identity:
            raise ValueError('Arena cell ID names multiple matchup identities')
        key = (row['cell_id'], row['index'])
        if key in seen:
            raise ValueError('Duplicate arena game index within a cell')
        seen.add(key)


def summarize_arena(rows):
    """Summarize completed, cutoff, and failed arena attempts.

    Each input has a stable ``cell_id``, ``index`` and ``seed``; learner and
    opponent labels/decks; ``learner_seat`` and ``starting_player``; and the
    runner's raw ``status``. Decisive rows supply ``winner_seat``,
    ``learner_won``, and ``draw=False``; draws supply ``draw=True`` and no
    winner. Optional ``reason`` and ``adjudicated`` fields are retained
    in aggregate counts. Rates exclude cutoffs/errors and count draws as
    non-wins. The game score grants half a point for a draw.

    Returns aggregate statistics, per-entrant statistics, and per-cell matchup
    statistics. No file or model state is read or changed.
    """
    rows = list(rows)
    _validate_rows(rows)
    entrants = defaultdict(list)
    cells = defaultdict(list)
    for row in rows:
        entrants[(row['learner_label'], row['learner_deck'])].append(row)
        cells[row['cell_id']].append(row)
    return dict(
        schema=1,
        aggregate=_stats(rows),
        entrants=[dict(learner_label=label, learner_deck=deck, **_stats(group))
                  for (label, deck), group in sorted(entrants.items())],
        cells=[dict(cell_id=cell_id,
                    learner_label=group[0]['learner_label'],
                    learner_deck=group[0]['learner_deck'],
                    opponent_label=group[0]['opponent_label'],
                    opponent_deck=group[0]['opponent_deck'],
                    **_stats(group))
               for cell_id, group in sorted(cells.items())],
    )
