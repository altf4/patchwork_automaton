"""Independent balanced policy evaluation; no training or automatic replay."""
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify():
    provenance = json.loads((ROOT / 'provenance.json').read_text())
    for name, digest in provenance['files'].items():
        assert sha(name) == digest, name
    return provenance


def schedule(panel, config, decks):
    rows = []
    repeats = 40 if panel['kind'] == 'neural' else 400
    for cell, (deck, other) in enumerate(product(decks, repeat=2)):
        for repeat in range(repeats):
            seat = repeat % 2
            seating = [other, other]
            seating[seat] = deck
            index = panel['offset'] + cell * repeats + repeat
            row = dict(index=index,
                seed=config['neural_seed'] + cell * repeats + repeat if panel['kind'] == 'neural' else config['baseline_seed'] + repeat,
                pairing=sorted((deck, other), key=decks.index), decks=seating,
                learner_seat=seat, learner_deck=deck, opponent_deck=other,
                starting_player=f'P{(repeat // 2) % 2}', partition='evaluation',
                deck_catalog='hartford' if panel['kind'] == 'neural' else 'standard',
                behavior_version=panel['version'],
                opponent_kind='frozen_neural' if panel['kind'] == 'neural' else 'scripted')
            if panel['kind'] == 'neural':
                row.update(opponent_version=200,
                    opponent_checkpoint_sha256=config['models']['parent']['sha256'])
            rows.append(row)
    random.Random('identity-final-eval:' + panel['name']).shuffle(rows)
    assert len(rows) == panel['games'] == len({v['index'] for v in rows})
    counts = Counter((v['learner_deck'], v['opponent_deck'], v['learner_seat'], v['starting_player']) for v in rows)
    assert len(counts) == len(decks) ** 2 * 4 and set(counts.values()) == {repeats // 4}
    return rows


def cluster_intervals(rows):
    import numpy as np
    groups = {}
    for row in rows:
        groups.setdefault(row['seed'], []).append(row)
    keys = sorted(groups)
    assert len(keys) == 400 and len({len(v) for v in groups.values()}) == 1
    score = np.array([sum(r['learner_won'] + .5 * r['draw'] for r in groups[k]) / len(groups[k]) for k in keys])
    win = np.array([sum(r['learner_won'] for r in groups[k]) / len(groups[k]) for k in keys])
    rng = np.random.default_rng(2026110304)
    wins, scores = [], []
    for _ in range(10):
        choices = rng.integers(0, len(keys), size=(1000, len(keys)))
        wins.extend(win[choices].mean(axis=1))
        scores.extend(score[choices].mean(axis=1))
    return dict(win_rate_ci95=np.quantile(wins, [.025, .975]).tolist(),
        score_ci95=np.quantile(scores, [.025, .975]).tolist(),
        score_ci95_method='10000 bootstrap resamples of 400 seed clusters',
        win_rate_ci95_method='10000 bootstrap resamples of 400 seed clusters')


async def main(name):
    from patchwork_automaton.baselines.standard_decks import HARTFORD_DECKS, STANDARD_DECKS
    from patchwork_automaton.neural.devices import configure_runtime, device_scope
    from patchwork_automaton.neural.standard_policy import StandardPolicy
    from patchwork_automaton.neural.inference_processes import weights_digest
    from patchwork_automaton.rl.hartford_evaluation import atomic_json, validate_row, summarize, outcome_summary
    from patchwork_automaton.rl.standard_selfplay_processes import StandardSelfplayCollector
    from patchwork_automaton.rl.rustmage_processes import ProcessGameCollector
    loop = asyncio.get_running_loop()
    def tick():
        if not loop.is_closed():
            loop.call_later(.002, tick)
    loop.call_later(.002, tick)
    configure_runtime('cpu', hide_gpu=True)
    config = json.loads((ROOT / 'config.json').read_text())
    panel = next(v for v in config['panels'] if v['name'] == name)
    provenance = verify()
    folder = ROOT / name
    assert not folder.exists(), 'Refuse automatic replay or duplicate evaluation'
    (folder / 'games').mkdir(parents=True)
    decks = tuple(HARTFORD_DECKS if panel['kind'] == 'neural' else STANDARD_DECKS)
    planned = schedule(panel, config, decks)
    expected = {v['index']: v for v in planned}
    atomic_json(folder / 'schedule.json', planned)
    with device_scope('cpu'):
        focal = StandardPolicy.load(config['models'][panel['model']]['path']).model
        opponent = StandardPolicy.load(config['models']['parent']['path']).model if panel['kind'] == 'neural' else None
    models = [focal] + ([opponent] if opponent is not None else [])
    original = [weights_digest(m.get_weights()) for m in models]
    assert focal.input_format == ('sparse-identity-standard-v2' if panel['model'] == 'identity' else 'sparse-factored-standard-v1')
    if opponent is not None:
        assert opponent.input_format == 'sparse-factored-standard-v1'
    atomic_json(folder / 'manifest.json', dict(panel=panel, config=config,
        source_provenance_sha256=sha(ROOT / 'provenance.json'),
        schedule_sha256=sha(folder / 'schedule.json'),
        model_input_formats=[m.input_format for m in models],
        model_weight_digests=original, training=False, policy='sampled temperature 1'))
    rows = {}
    started = time.monotonic()
    def persist(row):
        row = dict(row, actions=None)
        i = row['index']
        assert i not in rows
        try:
            validate_row(row, expected[i], checkpoint_round=panel['version'])
        except BaseException:
            atomic_json(folder / 'failures' / f'game-{i}.json', row)
            raise
        atomic_json(folder / 'games' / f'game-{i}.json', row)
        rows[i] = row
        if len(rows) % 100 == 0:
            event = dict(panel=name, completed=len(rows), total=panel['games'], elapsed_seconds=time.monotonic() - started)
            atomic_json(folder / 'progress.json', event)
            print(json.dumps(event), flush=True)
    options = dict(game_runner='standard_ppo', version=panel['version'],
        output=None, greedy=False, diagnostics=folder / 'diagnostics',
        game_timeout=panel['game_timeout'], max_turns=50, max_decisions=6000)
    if opponent is not None:
        collector = StandardSelfplayCollector(focal, opponent, workers=48,
            inference_processes=2, opponent_inference_processes=2)
        await collector.start(version=panel['version'], opponent_version=200)
        assert [collector.policy_digest, collector.opponent_digest] == original
        options.update(opponent_version=200, opponent_sha256=config['models']['parent']['sha256'])
    else:
        collector = ProcessGameCollector(focal, workers=48, inference_processes=2, inference_device='cpu')
        await collector.start(version=panel['version'])
        assert collector.policy_digest == original[0]
    try:
        await collector.run_schedule([dict(index=g['index'], seed=g['seed'], game=g) for g in planned], progress=persist, **options)
        stats = collector.stats()
    finally:
        await collector.close()
    assert set(rows) == set(expected)
    assert [weights_digest(m.get_weights()) for m in models] == original
    verify()
    values = list(rows.values())
    summary = summarize(values, decks)
    by_opponent = {deck: outcome_summary([v for v in values if v['opponent_deck'] == deck]) for deck in decks}
    if panel['kind'] == 'scripted':
        summary.update(cluster_intervals(values))
        for deck in decks:
            by_opponent[deck].update(cluster_intervals([v for v in values if v['opponent_deck'] == deck]))
    result = dict(status='complete', training=False, completed_at=datetime.now(timezone.utc).isoformat(),
        seconds=time.monotonic() - started, summary=summary, by_opponent=by_opponent,
        stats=stats, all_source_models_and_training_artifacts_unchanged=True)
    atomic_json(folder / 'summary.json', result)
    print(json.dumps({k: summary[k] for k in ('games', 'wins', 'losses', 'draws', 'score', 'score_ci95')}), flush=True)


if __name__ == '__main__':
    asyncio.run(main(sys.argv[1]))
