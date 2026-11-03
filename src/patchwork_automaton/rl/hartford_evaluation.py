"""Restartable sampled Hartford checkpoint duels, without training records.

This module imports TensorFlow only when a real policy must be loaded. Existing
collectors can be borrowed at a round boundary; their owner retains lifecycle
responsibility and must repin the training opponent before collecting again.
"""
import asyncio
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path

from .hartford_schedule import evaluation_schedule, previous_checkpoints
from ..neural.inference_processes import weights_digest


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_model(path):
    from ..neural.devices import device_scope
    from ..neural.standard_policy import StandardPolicy
    with device_scope('cpu'):
        return StandardPolicy.load(path).model


def _runtime_pins():
    root = Path(__file__).resolve().parents[1]
    relative = ('rl/hartford_evaluation.py', 'rl/hartford_schedule.py',
                'rl/standard_selfplay_processes.py', 'rl/standard_ppo.py',
                'rl/standard_imitation.py', 'rl/standard_direct.py',
                'baselines/standard_decks.py', 'data/hartford-standard-decklists.json')
    pins = {name: sha256(root / name) for name in relative}
    mode = os.environ.get('PATCHWORK_STANDARD_PLAYBACK', 'direct')
    pins['playback_mode'] = mode
    if mode == 'direct':
        from ..neural.direct_standard import library_path
        pins['direct_library'] = sha256(library_path())
    elif mode != 'python':
        raise ValueError('Unknown playback mode')
    return pins


async def _settle_all(awaitables):
    """Drain sibling runners before surfacing an error or returning ownership."""
    results = await asyncio.gather(*awaitables, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return results


def _checkpoint_files(paths, rounds):
    result = {}
    for number in rounds:
        path = Path(paths[number]).resolve()
        if path.is_dir():
            path /= 'policy.keras'
        metadata = path.with_suffix('.manifest.json')
        result[str(number)] = dict(path=str(path), sha256=sha256(path),
                                  manifest_sha256=sha256(metadata))
    return result


def validate_row(row, planned, *, checkpoint_round):
    """Check exact schedule identity and interpret outcomes from the focal seat."""
    if not isinstance(row, dict) or any(row.get(key) != value for key, value in planned.items()):
        raise ValueError('Evaluation row differs from its pinned schedule')
    if row.get('behavior_version') != checkpoint_round:
        raise ValueError('Evaluation used the wrong focal policy version')
    if row.get('trajectories') != []:
        raise ValueError('Evaluation produced training trajectories')
    if row.get('status') not in ('win', 'loss', 'draw'):
        raise RuntimeError(f"Evaluation game {planned['index']} failed: {row.get('error', row.get('status'))}")
    if row['status'] == 'draw':
        if row.get('winner_seat') is not None or row.get('draw') is not True or row.get('learner_won') is not False:
            raise ValueError('Inconsistent evaluation draw')
    else:
        winner = row.get('winner_seat')
        if type(winner) is not int or winner not in (0, 1) or row.get('draw') is not False:
            raise ValueError('Inconsistent evaluation winner')
        if row['status'] != ('win' if winner == 0 else 'loss'):
            raise ValueError('Engine outcome status disagrees with winner seat')
        if row.get('learner_won') is not (winner == planned['learner_seat']):
            raise ValueError('Evaluation outcome is not from the focal policy perspective')
    return row


def _wilson(wins, games):
    if not games:
        return None
    z = 1.959963984540054
    rate = wins / games
    scale = 1 + z * z / games
    center = (rate + z * z / (2 * games)) / scale
    radius = z * math.sqrt(rate * (1 - rate) / games + z * z / (4 * games * games)) / scale
    return [max(0., center - radius), min(1., center + radius)]


def outcome_summary(rows):
    """Counts plus conservative intervals for independent bounded game scores."""
    rows = list(rows)
    wins = sum(row['learner_won'] for row in rows)
    draws = sum(row['draw'] for row in rows)
    games = len(rows)
    score = (wins + .5 * draws) / games if games else None
    radius = math.sqrt(math.log(40) / (2 * games)) if games else None
    adjudications = Counter(row.get('reason', 'unspecified') for row in rows if row.get('adjudicated'))
    return dict(games=games, wins=wins, losses=games - wins - draws, draws=draws,
                natural_draws=sum(row['draw'] and not row.get('adjudicated', False) for row in rows),
                adjudicated_games=sum(adjudications.values()), adjudication_reasons=dict(adjudications),
                win_rate=wins / games if games else None, win_rate_ci95=_wilson(wins, games),
                score=score, score_ci95=None if not games else [max(0., score-radius), min(1., score+radius)],
                score_ci95_method='Hoeffding bound for independent scores in [0,1]',
                win_rate_ci95_method='Wilson interval; draws count as non-wins')


def summarize(rows, deck_ids):
    rows = list(rows)
    decks = tuple(deck_ids)
    result = outcome_summary(rows)
    result['on_play'] = outcome_summary(row for row in rows if row['starting_player'] == f"P{row['learner_seat']}")
    result['on_draw'] = outcome_summary(row for row in rows if row['starting_player'] != f"P{row['learner_seat']}")
    result['deck_ids'] = list(decks)
    result['matchups'] = {
        deck: {opponent: outcome_summary(row for row in rows
               if row['learner_deck'] == deck and row['opponent_deck'] == opponent)
               for opponent in decks} for deck in decks}
    result['score_matrix'] = [[result['matchups'][deck][opponent]['score'] for opponent in decks] for deck in decks]
    result['win_rate_matrix'] = [[result['matchups'][deck][opponent]['win_rate'] for opponent in decks] for deck in decks]
    return result


def _read_rows(directory, planned, checkpoint_round):
    expected = {row['index']: row for row in planned}
    if len(expected) != len(planned):
        raise ValueError('Evaluation schedule contains duplicate IDs')
    rows = {}
    for path in sorted(directory.glob('game-*.json')):
        row = json.loads(path.read_text())
        index = row.get('index')
        if type(index) is not int or index not in expected or index in rows:
            raise ValueError('Foreign or duplicate durable evaluation result')
        if path.name != f'game-{index}.json':
            raise ValueError('Durable evaluation filename differs from game identity')
        rows[index] = validate_row(row, expected[index], checkpoint_round=checkpoint_round)
    return rows


def _rows_digest(rows):
    ordered = sorted(rows, key=lambda row: row['index'])
    return hashlib.sha256(json.dumps(ordered, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


async def evaluate_checkpoint(*, model, checkpoint_round, checkpoint_paths, output,
                              deck_ids, seed, source_manifest, games_per_cell=40,
                              workers=24, runners=2, inference_processes=2,
                              opponent_inference_processes=2, game_timeout=60,
                              max_turns=50, max_decisions=6000, opponent_model=None, collectors=None,
                              progress=None, mixed_opponents=False):
    """Evaluate current versus its latest three predecessors and return a report.

    ``checkpoint_paths`` maps integer rounds to saved policies or checkpoint
    directories. ``source_manifest`` must pin the experiment's engine, source,
    and deck catalog. Existing output is resumable only with the exact manifest.
    Borrowed ``collectors`` must already be started and are never closed here.
    Their opponent state is deliberately left at the last evaluated predecessor.
    """
    opponents = previous_checkpoints(checkpoint_round)
    if not opponents:
        raise ValueError('Round zero has no previous checkpoint to evaluate')
    if not isinstance(source_manifest, dict) or not source_manifest:
        raise ValueError('Pinned source manifest is required for evaluation')
    if not isinstance(game_timeout, (int, float)) or not math.isfinite(game_timeout) or game_timeout <= 0:
        raise ValueError('A finite positive game timeout is required')
    if type(max_turns) is not int or max_turns < 1:
        raise ValueError('A positive turn limit is required')
    if type(max_decisions) is not int or max_decisions < 1:
        raise ValueError('A positive decision limit is required')
    for name, value in (('workers', workers), ('runners', runners),
                        ('inference_processes', inference_processes),
                        ('opponent_inference_processes', opponent_inference_processes)):
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if any(value % runners for value in (workers, inference_processes, opponent_inference_processes)):
        raise ValueError('Worker and inference totals must divide evenly among runners')
    if max(inference_processes, opponent_inference_processes) > workers:
        raise ValueError('Inference replica counts cannot exceed worker count')
    if collectors is not None and (not collectors or len(collectors) != runners or opponent_model is None):
        raise ValueError('Borrowed collectors require the matching runner count and opponent model')
    if mixed_opponents and collectors is not None:
        raise ValueError('Mixed evaluation owns its version-aware collectors')
    decks = tuple(deck_ids)
    paths = {int(key): value for key, value in checkpoint_paths.items()}
    pinned = _checkpoint_files(paths, (checkpoint_round, *opponents))
    focal_digest = weights_digest(model.get_weights())
    from ..neural.standard_features import STANDARD_CANDIDATE_LIMIT
    candidate_policy = dict(limit=STANDARD_CANDIDATE_LIMIT,
                            retained='deterministic legal prefix', selection='sampled temperature 1')
    # Validate the live focal against the actual saved file, not merely its label.
    saved_focal = _load_model(pinned[str(checkpoint_round)]['path'])
    if weights_digest(saved_focal.get_weights()) != focal_digest:
        raise ValueError('Live focal weights differ from the saved checkpoint')
    if opponent_model is model:
        raise ValueError('Focal and opponent must be separate model instances')
    if opponent_model is None:
        opponent_model = saved_focal
    else:
        del saved_focal
    schedules = {}
    for opponent in opponents:
        planned = evaluation_schedule(decks, checkpoint_round=checkpoint_round,
                                      opponent_round=opponent, games_per_cell=games_per_cell, seed=seed)
        schedules[opponent] = [dict(row, opponent_checkpoint_sha256=pinned[str(opponent)]['sha256']) for row in planned]
    configuration = dict(schema=1, checkpoint_round=checkpoint_round, opponent_rounds=list(opponents),
        checkpoints=pinned, focal_weights_sha256=focal_digest, deck_ids=list(decks),
        seed=seed, games_per_cell=games_per_cell, policy='sampled temperature 1',
        game_timeout=game_timeout, max_turns=max_turns, max_decisions=max_decisions,
        workers=workers, runners=runners,
        inference_processes=inference_processes, opponent_inference_processes=opponent_inference_processes,
        source_manifest=source_manifest, runtime_pins=_runtime_pins(),
        schedule_sha256=hashlib.sha256(json.dumps(schedules, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        metric='equally weighted mean of matchup scores against each available predecessor',
        training=False, candidate_policy=candidate_policy)
    if mixed_opponents:configuration['collection_mode']='mixed-frozen-opponents-v1'
    # Normalize JSON keys/tuples before comparing with durable JSON on resume.
    configuration = json.loads(json.dumps(configuration, sort_keys=True, allow_nan=False))
    root = Path(output)
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != configuration:
            raise ValueError('Evaluation configuration, source, or checkpoint changed on resume')
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError('Nonempty evaluation output has no manifest')
        atomic_json(manifest_path, configuration)
    failures = sorted(root.glob('opponent-*/failures/game-*.json'))
    if failures:
        raise RuntimeError(f'Preserved failed evaluation attempt prevents automatic replay: {failures[0]}')
    published = None
    summary_path = root / 'summary.json'
    if summary_path.exists():
        candidate = json.loads(summary_path.read_text())
        if candidate.get('complete') is True:
            published = candidate
    all_rows = {}
    for opponent, planned in schedules.items():
        all_rows[opponent] = _read_rows(root / f'opponent-{opponent:04d}' / 'games', planned, checkpoint_round)
        if published is not None and len(all_rows[opponent]) != len(planned):
            raise ValueError('Completed evaluation is missing durable game results')
        prior_summary_path = root / f'opponent-{opponent:04d}' / 'summary.json'
        if prior_summary_path.exists():
            prior_summary = json.loads(prior_summary_path.read_text())
            if prior_summary.get('games_sha256') != _rows_digest(all_rows[opponent].values()):
                raise ValueError('Published opponent results changed or are incomplete')
    owned = collectors is None
    active_collectors = [] if owned else list(collectors)
    completed = {}
    try:
        if mixed_opponents:
            pending=[row for opponent,planned in schedules.items() for row in planned if row['index'] not in all_rows[opponent]]
            if pending:
                weights={}
                for opponent in opponents:
                    loaded=_load_model(pinned[str(opponent)]['path']);weights[opponent]=loaded.get_weights();del loaded
                digests={n:weights_digest(w) for n,w in weights.items()}
                from .standard_selfplay_processes import StandardSelfplayCollector
                active_collectors=[StandardSelfplayCollector(model,opponent_model,workers=workers//runners,
                    inference_processes=inference_processes//runners,
                    opponent_inference_processes=opponent_inference_processes//runners) for _ in range(runners)]
                await _settle_all(c.start(version=checkpoint_round,opponent_weights=weights) for c in active_collectors)
                if any(c.policy_digest!=focal_digest or c.replicas[1].version_digests!=digests for c in active_collectors):
                    raise RuntimeError('Collector has stale frozen checkpoint bank')
                by_id={row['index']:row for row in pending}
                def persist_mixed(row):
                    index=row.get('index')
                    if type(index) is not int or index not in by_id:raise ValueError('Unexpected evaluation result')
                    planned=by_id[index];opponent=planned['opponent_version'];rows=all_rows[opponent]
                    if index in rows:raise ValueError('Duplicate evaluation result')
                    directory=root/f'opponent-{opponent:04d}';row=dict(row,actions=None)
                    try:validate_row(row,planned,checkpoint_round=checkpoint_round)
                    except (ValueError,RuntimeError):
                        atomic_json(directory/'failures'/f'game-{index}.json',row);raise
                    atomic_json(directory/'games'/f'game-{index}.json',row);rows[index]=row
                    count=sum(map(len,all_rows.values()));total=sum(map(len,schedules.values()))
                    if count%25==0 or count==total:
                        status=dict(checkpoint_round=checkpoint_round,opponent_round='mixed',completed=count,total=total,phase='evaluation')
                        atomic_json(root/'progress.json',status)
                        if progress:progress(status)
                # Round-robin opponents before dispatch; each runner sees all
                # versions and no predecessor becomes a separate work barrier.
                from itertools import zip_longest
                pending=[row for group in zip_longest(*([row for row in pending if row['opponent_version']==n]
                    for n in opponents)) for row in group if row is not None]
                jobs=[dict(index=row['index'],seed=row['seed'],game=row,
                    opponent_weights_sha256=digests[row['opponent_version']],
                    diagnostics=root/f"opponent-{row['opponent_version']:04d}"/'diagnostics') for row in pending]
                await _settle_all(c.run_schedule(jobs[i::runners],progress=persist_mixed,game_runner='standard_ppo',
                    output=None,greedy=False,version=checkpoint_round,game_timeout=game_timeout,
                    max_turns=max_turns,max_decisions=max_decisions) for i,c in enumerate(active_collectors))
        for opponent, planned in schedules.items():
            rows = all_rows[opponent]
            directory = root / f'opponent-{opponent:04d}'
            by_id = {row['index']: row for row in planned}
            pending = [row for row in planned if row['index'] not in rows]
            if pending:
                loaded = _load_model(pinned[str(opponent)]['path'])
                opponent_model.set_weights(loaded.get_weights())
                del loaded
                opponent_digest = weights_digest(opponent_model.get_weights())
                if not active_collectors:
                    from .standard_selfplay_processes import StandardSelfplayCollector
                    active_collectors = [StandardSelfplayCollector(model, opponent_model,
                        workers=workers // runners, inference_processes=inference_processes // runners,
                        opponent_inference_processes=opponent_inference_processes // runners)
                        for _ in range(runners)]
                    await _settle_all(collector.start(version=checkpoint_round, opponent_version=opponent)
                                      for collector in active_collectors)
                else:
                    await _settle_all(collector.sync_weights(model, checkpoint_round) for collector in active_collectors)
                    await _settle_all(collector.sync_opponent(opponent_model, opponent) for collector in active_collectors)
                for collector in active_collectors:
                    if (collector.version != checkpoint_round or collector.opponent_version != opponent
                            or collector.policy_digest != focal_digest or collector.opponent_digest != opponent_digest):
                        raise RuntimeError('Collector has stale policy weights or checkpoint versions')
                def persist(row):
                    index = row.get('index')
                    if type(index) is not int or index not in by_id or index in rows:
                        raise ValueError('Unexpected or duplicate evaluation result')
                    row = dict(row, actions=None)
                    try:
                        validate_row(row, by_id[index], checkpoint_round=checkpoint_round)
                    except (ValueError, RuntimeError):
                        atomic_json(directory / 'failures' / f'game-{index}.json', row)
                        raise
                    atomic_json(directory / 'games' / f'game-{index}.json', row)
                    rows[index] = row
                    if len(rows) % 100 == 0 or len(rows) == len(planned):
                        status = dict(checkpoint_round=checkpoint_round, opponent_round=opponent,
                                      completed=len(rows), total=len(planned), phase='evaluation')
                        atomic_json(root / 'progress.json', status)
                        if progress:
                            progress(status)
                jobs = [dict(index=row['index'], seed=row['seed'], game=row) for row in pending]
                await _settle_all(collector.run_schedule(jobs[i::len(active_collectors)],
                    progress=persist, game_runner='standard_ppo', output=None, greedy=False,
                    version=checkpoint_round, opponent_version=opponent,
                    opponent_sha256=pinned[str(opponent)]['sha256'], diagnostics=directory / 'diagnostics',
                    game_timeout=game_timeout, max_turns=max_turns, max_decisions=max_decisions)
                    for i, collector in enumerate(active_collectors))
                if weights_digest(opponent_model.get_weights()) != opponent_digest:
                    raise RuntimeError('Opponent weights changed during read-only evaluation')
            if len(rows) != len(planned):
                raise RuntimeError('Evaluation did not complete every scheduled game')
            summary = summarize(rows.values(), decks)
            summary.update(opponent_round=opponent, checkpoint_sha256=pinned[str(opponent)]['sha256'],
                           games_sha256=_rows_digest(rows.values()))
            completed[str(opponent)] = summary
            if published is None:
                atomic_json(directory / 'summary.json', summary)
                atomic_json(root / 'summary.json', dict(complete=False, checkpoint_round=checkpoint_round,
                                                      opponents=completed))
        if weights_digest(model.get_weights()) != focal_digest:
            raise RuntimeError('Focal weights changed during read-only evaluation')
        if _checkpoint_files(paths, (checkpoint_round, *opponents)) != pinned:
            raise RuntimeError('Checkpoint files changed during evaluation')
        if _runtime_pins() != configuration['runtime_pins']:
            raise RuntimeError('Evaluation source or engine changed during evaluation')
        aggregate = summarize((row for rows in all_rows.values() for row in rows.values()), decks)
        equal_score = sum(summary['score'] for summary in completed.values()) / len(completed)
        if not math.isclose(aggregate['score'], equal_score, abs_tol=1e-12):
            raise AssertionError('Pooled result differs from equal-opponent weighting')
        result = dict(complete=True, checkpoint_round=checkpoint_round, opponent_rounds=list(opponents),
                      checkpoint_sha256=pinned[str(checkpoint_round)]['sha256'],
                      policy='sampled temperature 1', candidate_policy=candidate_policy,
                      training=False, opponents=completed,
                      aggregate=aggregate, score=equal_score,
                      metric=configuration['metric'], manifest_sha256=sha256(manifest_path))
        if published is not None and published != result:
            raise ValueError('Completed evaluation summary differs from durable game results')
        atomic_json(root / 'summary.json', result)
        atomic_json(root / 'progress.json', dict(phase='complete', checkpoint_round=checkpoint_round,
                                               completed=aggregate['games'], total=aggregate['games']))
        return result
    finally:
        if owned:
            await asyncio.gather(*(collector.close() for collector in active_collectors))
