"""Recompute the profile and matched-experiment summary from saved manifests."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'results'


def read(path):
    return json.loads(path.read_text())


def guard(name):
    root = ROOT / name
    status = read(root / 'guard-status.json')
    if status['status'] != 'exited' or status['exit_code'] != 0 or status['abort'] is not None:
        raise ValueError(f'{name} guard did not complete cleanly')
    return min(json.loads(line)['available_bytes'] for line in
               (root / 'host-memory-samples.jsonl').read_text().splitlines()) / 2**30


def cpu_breakdown(profile):
    stages = profile['stages']
    value = lambda key: stages[key]['exclusive_cpu']
    groups = {
        'inference replicas': profile['learner_replica_cpu_seconds'] + profile['opponent_replica_cpu_seconds'],
        'other game work and instrumentation': value('game_other'),
        'game-side inference pipe': value('inference_pipe'),
        'Rust observation': value('native_observe'),
        'rollout writing and close': value('rollout_write') + value('rollout_close'),
        'Rust history, choices and encoding': sum(value(k) for k in ('native_history_observe', 'native_history_take', 'native_prepare', 'native_encode')),
        'Rust action application': value('native_apply_choice'),
        'NumPy array-view setup': value('numpy_arrays'),
    }
    consumed = {'game_other', 'inference_pipe', 'native_observe', 'rollout_write',
                'rollout_close', 'native_apply_choice', 'numpy_arrays', 'native_history_observe',
                'native_history_take', 'native_prepare', 'native_encode'}
    groups['other native and action-label overhead'] = sum(v['exclusive_cpu'] for k,v in stages.items() if k not in consumed)
    groups['worker, coordinator and unaccounted child CPU'] = (profile['parent_worker_cpu_seconds'] +
        profile['coordinator_cpu_seconds'] + profile['child_cpu_seconds'] - sum(v['exclusive_cpu'] for v in stages.values()))
    total = sum(groups.values())
    return dict(total_cpu_seconds=total, categories=[dict(category=k, cpu_seconds=v, share_percent=100*v/total)
        for k,v in sorted(groups.items(),key=lambda x:-x[1])])


def validate_comparisons(manifest):
    if manifest['status'] != 'complete': raise ValueError('Experiment did not complete')
    for comparison in manifest['comparisons']:
        trajectory = comparison['trajectory']
        if comparison['matching'] != comparison['games'] or trajectory['feature_mismatch_count'] or trajectory['skipped_adjudicated']:
            raise ValueError('Game or recorded feature comparison did not match')


def main():
    profile = read(ROOT/'profile/profile/summary.json')
    pm = read(ROOT/'profile/manifest.json')
    if pm['status'] != 'complete' or pm['pilot_comparison']['identical'] != pm['pilot_comparison']['games']:
        raise ValueError('Profile or pilot failed')
    profile_guard = read(ROOT/'profile/guard/guard-status.json')
    if profile_guard['status'] != 'exited' or profile_guard['exit_code'] != 0 or profile_guard['abort'] is not None:
        raise ValueError('Profile guard did not complete cleanly')
    result = dict(profile=dict(games=profile['games'], wall_seconds=profile['wall_seconds'],
        games_per_minute=profile['games_per_minute'], decisions=profile['decisions'],
        pilot_profile_wall_ratio=pm['pilot_profile_wall_ratio'],
        service_overlapping_wall_seconds=profile['service_stage_wall_seconds'],
        mean_batch=(profile['learner_requests']+profile['opponent_requests'])/(profile['learner_calls']+profile['opponent_calls']),
        **cpu_breakdown(profile)))
    result['profile']['minimum_available_gib'] = min(json.loads(line)['available_bytes'] for line in
        (ROOT/'profile/guard/host-memory-samples.jsonl').read_text().splitlines()) / 2**30
    for name in ('preflight-shared4', 'preflight-shared2', 'preflight-threads1', 'full-threads1', 'default-check'):
        manifest=read(ROOT/name/'manifest.json'); validate_comparisons(manifest)
        result[name]=dict(comparisons=manifest['comparisons'], minimum_available_gib=guard(name), phases=[])
        for p in manifest['phases']:
            shared = p['configuration']=='shared'
            roles = ('learner',) if shared else ('learner','opponent')
            requests=sum(p[role+'_requests'] for role in roles)
            calls=sum(p[role+'_calls'] for role in roles)
            result[name]['phases'].append(dict(pair=p['pair'],configuration=p['configuration'],games=p['games'],
                wall_seconds=p['wall_seconds'],games_per_minute=p['games_per_minute'],
                inference_replica_cpu_seconds=sum(p[role+'_replica_cpu_seconds'] for role in roles),
                mean_batch=requests/max(calls,1),outcomes=p['outcomes']))
            if name in ('preflight-threads1','full-threads1'):
                expected = manifest['configurations'][p['configuration']]
                for snap in ('replica_before','replica_after'):
                    for collector in p[snap]:
                        for role in ('learner','opponent'):
                            for replica in collector[role]['replicas']:
                                if replica['runtime_intra_threads'] != expected or replica['runtime_inter_threads'] != 2:
                                    raise ValueError('Incorrect effective thread configuration')
            if name == 'default-check':
                expected = manifest['configurations'][p['configuration']]
                for snap in ('replica_before','replica_after'):
                    for collector in p[snap]:
                        for role in ('learner','opponent'):
                            for replica in collector[role]['replicas']:
                                if (replica['intra_op_threads'] != expected
                                        or replica['tf_num_intraop_threads_env'] != '2'
                                        or replica['tf_num_interop_threads_env'] != '2'):
                                    raise ValueError('Production replica thread configuration differs')
    phases=result['full-threads1']['phases']
    baseline=[p for p in phases if p['configuration']=='baseline']
    candidate=[p for p in phases if p['configuration']=='threads1']
    if len(baseline)!=2 or len(candidate)!=2: raise ValueError('Expected two pairs')
    btime=sum(p['wall_seconds'] for p in baseline); ctime=sum(p['wall_seconds'] for p in candidate)
    bcpu=sum(p['inference_replica_cpu_seconds'] for p in baseline); ccpu=sum(p['inference_replica_cpu_seconds'] for p in candidate)
    result['full-threads1']['aggregate']=dict(baseline_games_per_minute=120000/btime,
        candidate_games_per_minute=120000/ctime, throughput_increase_percent=100*(btime/ctime-1),
        wall_time_reduction_percent=100*(1-ctime/btime), baseline_replica_cpu_seconds=bcpu,
        candidate_replica_cpu_seconds=ccpu,replica_cpu_reduction_percent=100*(1-ccpu/bcpu))
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result['full-threads1']['aggregate'],indent=2))

if __name__=='__main__': main()
