"""Fixed-cohort playing-strength metrics against the active RustMage baseline.

These are game outcomes, separate from both inert-opponent goldfishing and
recorded-trajectory validation. Every scheduled attempt remains in the primary
win-rate denominator. A cutoff/error is unresolved, never a fabricated loss,
draw, terminal score, or training return.
"""
from collections import Counter
import math
import random
from statistics import NormalDist


STATUSES = ('win', 'loss', 'draw', 'cutoff', 'error')
TERMINAL_STATUSES = frozenset(('win', 'loss', 'draw'))
BASELINE_MODE = 'versus_baseline'


def baseline_evaluation_config(*, seed_base, games=200, index_base=0,
                               max_decisions=6000, max_own_turns=60,
                               repeat_limit=20, greedy=True):
    """Return options for existing ``run_games`` plus reportable cohort metadata.

    ``run_games`` assigns seed_base+i, seat=index%2, and starter=(index//2)%2.
    A multiple of four attempts from an aligned index therefore balances the
    four seat/start strata. Seeds are unique per attempt, not repeated across
    seats. Reuse the complete configuration at every checkpoint and evaluate
    the frozen initial checkpoint against the same opponent/cohort.

    The caller must choose seed_base outside the training seed range. This
    function neither collects games nor accesses training trajectories.
    """
    for name, value, minimum in (
        ('games', games, 1), ('seed_base', seed_base, 0),
        ('index_base', index_base, 0), ('max_decisions', max_decisions, 1),
        ('max_own_turns', max_own_turns, 1), ('repeat_limit', repeat_limit, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if games % 4 or index_base % 4:
        raise ValueError('Seat/start balance requires games and index_base divisible by four')
    if not isinstance(greedy, bool):
        raise ValueError('greedy must be a bool')
    return dict(
        collection=dict(games=games, seed_base=seed_base, index_base=index_base,
            mode=BASELINE_MODE, output=None, max_decisions=max_decisions,
            max_own_turns=max_own_turns, repeat_limit=repeat_limit, greedy=greedy),
        opponent=dict(policy='FoundationsAggro', behavior='active',
            learner_deck="foundations_deck('red')",
            opponent_deck="foundations_deck('red')"),
        cohort=dict(method='fixed_seed_seat_start_v1', attempted_games=games,
            seed_base=seed_base, seed_last=seed_base+games-1,
            index_base=index_base, attempts_per_seat_start_stratum=games//4,
            same_seed_repeated_across_seats=False, replace_failed_attempts=False,
            training_trajectories=False),
    )


def _checked_rows(rows, expected_games=None):
    rows = list(rows)
    if expected_games is not None:
        if (isinstance(expected_games, bool) or not isinstance(expected_games, int)
                or expected_games < 0):
            raise ValueError('expected_games must be a nonnegative integer')
        if len(rows) != expected_games:
            raise ValueError('Evaluation must retain every scheduled attempt without replacement')
    indices = set()
    for row in rows:
        if row.get('mode') != BASELINE_MODE:
            raise ValueError('Expected actual versus_baseline games, not goldfish or selfplay')
        if row.get('status') not in STATUSES:
            raise ValueError('Unknown evaluation status')
        if row.get('seat') not in (0, 1) or row.get('starting_player') not in ('P0', 'P1'):
            raise ValueError('Expected a valid evaluation seat and starting player')
        index, seed = row.get('index'), row.get('seed')
        if (isinstance(index, bool) or not isinstance(index, int) or index < 0
                or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
            raise ValueError('Expected nonnegative integer evaluation index and seed')
        if index in indices:
            raise ValueError('Evaluation indices must be unique')
        indices.add(index)
        if row.get('trajectories'):
            raise ValueError('Playing-strength evaluation must not contain training trajectories')
    return rows


def _wilson_interval(wins, attempts):
    if not attempts:
        return None
    z = NormalDist().inv_cdf(.975)
    proportion = wins / attempts
    denominator = 1 + z*z/attempts
    center = (proportion + z*z/(2*attempts)) / denominator
    radius = z*math.sqrt(proportion*(1-proportion)/attempts
                         + z*z/(4*attempts*attempts)) / denominator
    return [max(0., center-radius), min(1., center+radius)]


def _counts(rows):
    statuses = Counter(row['status'] for row in rows)
    attempts = len(rows)
    terminal = sum(statuses[status] for status in TERMINAL_STATUSES)
    return dict(
        attempted_games=attempts, terminal_games=terminal,
        wins=statuses['win'], losses=statuses['loss'], draws=statuses['draw'],
        cutoffs=statuses['cutoff'], errors=statuses['error'],
        unresolved_attempts=statuses['cutoff']+statuses['error'],
        terminal_fraction=terminal/attempts if attempts else None,
        win_rate_all_attempts=statuses['win']/attempts if attempts else None,
        win_rate_all_attempts_wilson_95_interval=_wilson_interval(statuses['win'], attempts),
        terminal_win_rate=statuses['win']/terminal if terminal else None,
        terminal_score_rate=(statuses['win']+.5*statuses['draw'])/terminal if terminal else None,
    )


def match_summary(rows, *, expected_games=None):
    """Count recorded results without discarding or replacing unsuccessful runs.

    The primary rate/interval describe attempts ending in a recorded win under
    the configured limits, not eventual wins in unresolved games. Terminal-only
    rates are secondary diagnostics and can improve just by censoring games.
    No goldfish speed, synthetic terminal score, or RL return is computed.
    """
    rows = _checked_rows(rows, expected_games)
    result = _counts(rows)
    strata = {}
    for seat in (0, 1):
        for starter in ('P0', 'P1'):
            selected = [row for row in rows
                        if row['seat'] == seat and row['starting_player'] == starter]
            strata[f'seat_{seat}_start_{starter}'] = _counts(selected)
    result.update(
        metric='recorded_wins_per_scheduled_attempt', mode=BASELINE_MODE,
        win_rate_denominator='all_attempts_including_cutoffs_and_errors',
        unresolved_outcomes='unknown; no terminal score or training return assigned',
        seat_start_balanced=bool(rows) and len({s['attempted_games'] for s in strata.values()}) == 1,
        seat_start_strata=strata,
    )
    return result


def _attempt_key(row):
    return row['index'], row['seed'], row['seat'], row['starting_player']


def _paired_interval(differences, *, rng, resamples):
    if not differences:
        return None
    count = len(differences)
    boot = sorted(sum(differences[rng.randrange(count)] for _ in range(count))/count
                  for _ in range(resamples))
    def quantile(fraction):
        index = (len(boot)-1)*fraction
        lower = math.floor(index)
        upper = math.ceil(index)
        return boot[lower] + (boot[upper]-boot[lower])*(index-lower)
    return [quantile(.025), quantile(.975)]


def paired_match_improvement(initial, current, *, seed=9901, resamples=4000):
    """Pair frozen-initial/current evaluation on identical scheduled attempts.

    Each bootstrap draw resamples complete matched attempts. All attempts enter
    recorded-win differences. Terminal scores are compared only when BOTH runs
    actually terminated, with unresolved pairs explicitly counted. An engine
    failure becoming a win is visible in status_transitions, not hidden as a
    fabricated game loss. This is comparison against the same active baseline,
    not a head-to-head match against the frozen network.
    """
    initial = sorted(_checked_rows(initial), key=_attempt_key)
    current = sorted(_checked_rows(current), key=_attempt_key)
    if not initial or [_attempt_key(row) for row in initial] != [
            _attempt_key(row) for row in current]:
        raise ValueError('Expected nonempty matching indices, seeds, seats and starting players')
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError('resamples must be a positive integer')
    win_differences = [float(b['status'] == 'win')-float(a['status'] == 'win')
                       for a, b in zip(initial, current)]
    terminal_scores = {'win': 1., 'draw': .5, 'loss': 0.}
    terminal_differences = [terminal_scores[b['status']]-terminal_scores[a['status']]
        for a, b in zip(initial, current)
        if a['status'] in TERMINAL_STATUSES and b['status'] in TERMINAL_STATUSES]
    transitions = Counter(f"{a['status']}->{b['status']}" for a, b in zip(initial, current))
    return dict(
        attempted_pairs=len(initial), terminal_pairs=len(terminal_differences),
        unresolved_pairs=len(initial)-len(terminal_differences),
        recorded_attempt_win_rate_change=sum(win_differences)/len(win_differences),
        recorded_attempt_win_rate_change_bootstrap_95_interval=_paired_interval(
            win_differences, rng=random.Random(seed), resamples=resamples),
        terminal_score_change_on_terminal_pairs=(sum(terminal_differences)/len(terminal_differences)
                                               if terminal_differences else None),
        terminal_score_change_bootstrap_95_interval=_paired_interval(
            terminal_differences, rng=random.Random(seed), resamples=resamples),
        status_transitions=dict(sorted(transitions.items())),
        improved_recorded_win_attempts=sum(d > 0 for d in win_differences),
        worsened_recorded_win_attempts=sum(d < 0 for d in win_differences),
        tied_recorded_win_attempts=sum(d == 0 for d in win_differences),
        bootstrap=dict(unit='matched_attempt', seed=seed, resamples=resamples),
    )
