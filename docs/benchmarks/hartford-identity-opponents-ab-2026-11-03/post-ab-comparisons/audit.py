"""Read-only completion audit and export for all four comparison panels."""
import base64
import gzip
import json
from pathlib import Path

import evaluate


def audit(launch):
    from patchwork_automaton.baselines.standard_decks import HARTFORD_DECKS, STANDARD_DECKS
    from patchwork_automaton.rl.hartford_evaluation import summarize
    config = evaluate.read(launch['config'])
    if evaluate.sha(launch['config']) != launch['config_sha256']:
        raise ValueError('Pinned configuration changed')
    evaluate.verify(config)
    root = Path(config['output'])
    guard = evaluate.read(Path(launch['guard']) / 'guard-status.json')
    if (guard['status'] != 'exited' or guard['guard_exit_code'] != 0 or guard['child_exit_code'] != 0
            or guard['limits']['min_available_mib'] != 1536 or guard['limits']['memory_grace_seconds'] != 5
            or not guard['windows_awake']['release']['graceful_release']):
        raise ValueError('Final resource guard has not exited normally')
    shared = evaluate.helpers(config)
    results, outcomes = {}, {}
    ids = set()
    recovered_total = 0
    for panel in config['panels']:
        folder = root / panel['name']
        decks = tuple(HARTFORD_DECKS if panel['kind'] == 'neural' else STANDARD_DECKS)
        planned = evaluate.schedule(panel, config, decks)
        expected = {r['index']: r for r in planned}
        manifest = evaluate.read(folder / 'manifest.json')
        if (manifest['config_sha256'] != launch['config_sha256']
                or manifest['schedule_sha256'] != evaluate.value_sha(planned)
                or evaluate.read(folder / 'schedule.json') != planned):
            raise ValueError('Panel schedule or provenance changed')
        rows = shared.load_results(folder, planned)
        handoffs = evaluate.load_handoffs(folder, expected, shared)
        if set(rows) != set(expected) or set(handoffs) != set(expected) or len(rows) != 10000:
            raise ValueError('Missing original outcomes or completed handoffs')
        if ids.intersection(rows):
            raise ValueError('Duplicate identity across panels')
        ids.update(rows)
        for index, handoff in handoffs.items():
            if any(rows[index].get(key) != value for key, value in handoff.items()):
                raise ValueError('Published outcome differs from original handoff')
        ordered = [rows[index] for index in sorted(rows)]
        result = evaluate.read(folder / 'summary.json')
        if (result['summary'] != summarize(ordered, decks) or result['games'] != 10000
                or result['rows_sha256'] != evaluate.value_sha(ordered)):
            raise ValueError('Summary differs from complete outcomes')
        recovered_total += len(result['recovered_completed_handoffs'])
        for path in (folder / 'transport-errors').glob('*.json'):
            if path.name.endswith('-resolution.json'):
                continue
            resolution = evaluate.read(path.with_name(path.stem + '-resolution.json'))
            if (not resolution['resolved'] or resolution['games_replayed'] != 0
                    or not resolution['all_games_validated'] or not resolution['collectors_closed']):
                raise ValueError('Unresolved transport cleanup error')
        outcomes[panel['name']] = ordered
        results[panel['name']] = result
    # Shared seeds intentionally pair the two models within each opponent panel.
    for kind in ('scripted', 'neural'):
        panels = [p for p in config['panels'] if p['kind'] == kind]
        topology = []
        for panel in panels:
            topology.append(sorted((r['seed'], r['learner_deck'], r['opponent_deck'],
                r['learner_seat'], r['starting_player']) for r in outcomes[panel['name']]))
        if topology[0] != topology[1]:
            raise ValueError('Model arms used different evaluation seeds or matchup assignments')
    evaluate.verify(config)
    payload = gzip.compress(json.dumps(outcomes, sort_keys=True, separators=(',', ':')).encode(), mtime=0)
    import hashlib
    samples = [json.loads(line) for line in
        (Path(launch['guard']) / 'host-memory-samples.jsonl').read_text().splitlines()]
    if not samples or max(s['low_memory_seconds'] for s in samples) >= 5:
        raise ValueError('Missing resource history or memory guard breach')
    evidence = dict(passed=True, games=40000, panels=4, games_per_panel=10000,
        models_source_engine_training_hashes_verified=True, all_outcome_handoffs_verified=True,
        matched_seeds_verified=True, deck_seat_start_schedules_verified=True,
        published_handoffs_recovered=recovered_total, games_replayed=0,
        guard_exit_code=0, minimum_available_gib=min(s['available_memory_bytes'] for s in samples)/1024**3,
        outcomes_gzip_sha256=hashlib.sha256(payload).hexdigest())
    return dict(audit=evidence, results=results, models=config['models'],
        outcomes_gzip_b64=base64.b64encode(payload).decode())
