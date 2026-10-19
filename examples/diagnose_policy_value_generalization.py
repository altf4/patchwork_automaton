"""80/20 saved-update audits plus fixed-opponent seed generalization matches.

Source checkpoints stay frozen. Disposable updates use terminal returns only.
Policy surrogate validation is reported separately from actual game outcomes.
"""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from . import train_rustmage  # CPU environment before TensorFlow imports.
import numpy as np
from patchwork_automaton.neural.experiment import atomic_json, make_trainer
from patchwork_automaton.neural.rustmage_policy import RustMagePolicy
from patchwork_automaton.neural.validation import GameValidator, split_games, training_normalization
from patchwork_automaton.rl.rustmage_processes import ProcessGameCollector


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def partition_gap(training, validation, field, count, *, seed=5123):
    rng = np.random.default_rng(seed)
    parts = []
    for rows in (training['games'],validation['games']):
        data = np.asarray([[r[field],r[count]] for r in rows],np.float64)
        draw = data[rng.integers(len(data),size=(4000,len(data)))].sum(axis=1)
        parts.append((data[:,0].sum()/data[:,1].sum(),draw[:,0]/draw[:,1]))
    return dict(training_minus_validation=float(parts[0][0]-parts[1][0]),
        game_bootstrap_95_interval=np.quantile(parts[0][1]-parts[1][1],[.025,.975]).tolist())


def match_summary(rows):
    scores = np.asarray([1. if g['status']=='win' else .5 if g['status']=='draw' else 0. for g in rows])
    rng = np.random.default_rng(998)
    boot = rng.choice(scores,size=(4000,len(scores))).mean(axis=1)
    return dict(attempts=len(rows),outcomes=dict(Counter(r['status'] for r in rows)),
                score=float(scores.mean()),score_game_bootstrap_95_interval=np.quantile(boot,[.025,.975]).tolist(),
                failures=sum(r['status'] not in ('win','loss','draw') for r in rows),
                scoring='win=1, draw=0.5, loss=0; cutoffs/errors retained as failed attempts, never training rewards')


async def run(args):
    args.output.mkdir(parents=True,exist_ok=args.resume)
    report = json.loads(args.report.read_text())
    checkpoints = args.report.parent/'checkpoints'
    selected = args.rounds
    files = {args.report}
    for round_id in set([r-1 for r in selected]+[10,30]):
        files.update((checkpoints/f'round-{round_id:04d}').glob('*'))
    hashes = {str(p):digest(p) for p in files if p.is_file()}
    result = dict(scope=__doc__,config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  source_sha256=hashes,updates=[],matches=[])
    if args.resume:
        previous = json.loads((args.output/'report.json').read_text())
        old_config, new_config = dict(previous['config']), dict(result['config'])
        old_config.pop('resume',None); new_config.pop('resume',None)
        if old_config != new_config or previous['source_sha256'] != hashes:
            raise ValueError('Resume requires identical inputs and untouched source checkpoints')
        result = previous
        # Preserve the interrupted initial baseline batch from the first audit.
        if 'failed_games' in result:
            result.setdefault('failed_groups',[]).append(dict(checkpoint=10,opponent='baseline',
                seeds='familiar',games=result.pop('failed_games'),
                reason='Baseline lacked combat damage assignment; entire group rerun after coverage fix'))
    result.setdefault('invocations',[]).append(dict(resume=args.resume,
        code_sha256={str(p):digest(p) for p in [Path(__file__),
            Path('src/patchwork_automaton/baselines/foundations.py'),
            Path('src/patchwork_automaton/neural/validation.py'),
            Path('src/patchwork_automaton/rl/rustmage_experiment.py')]}))
    result['training_config']=report['config']
    result['library_sha256']=report.get('library_sha256')
    result['encoder']=report.get('encoder')
    def save(event):
        atomic_json(args.output/'report.json',result)
        print(json.dumps(event),flush=True)
    for round_id in selected:
        if any(r['round']==round_id for r in result['updates']): continue
        directory = checkpoints/f'round-{round_id-1:04d}'
        model = RustMagePolicy.load(directory/'policy.keras').model
        trainer = make_trainer(model,report['config'])
        trainer.compile()
        state = json.loads((directory/'state.json').read_text())
        for optimizer,name in [(trainer.optimizer,'optimizer'),(trainer.critic_optimizer,'critic-optimizer')]:
            with np.load(directory/(name+'.npz')) as archive:
                assert len(archive.files)==len(optimizer.variables)
                for i,v in enumerate(optimizer.variables):v.assign(archive[f'v{i}'])
        trainer.rng.bit_generator.state=state['trainer_rng_state']
        games = next(r['games'] for r in report['rounds'] if r['round']==round_id)
        train,heldout,split = split_games(games,fraction=.2,seed=report['config']['seed'],round_id=round_id)
        normalization = training_normalization(train,trainer.advantage_scale_floor)
        save(dict(stage='update-start',round=round_id,split=split))
        started=time.monotonic()
        metrics=await asyncio.to_thread(trainer.update_files,[t for g in train for t in g['trajectories']])
        validator=GameValidator(model)
        fitting=await asyncio.to_thread(validator.evaluate,train,normalization)
        validation=await asyncio.to_thread(validator.evaluate,heldout,normalization)
        row=dict(round=round_id,seconds=time.monotonic()-started,split=split,update=metrics,
                 training=fitting,validation=validation,
                 policy_gap=partition_gap(fitting,validation,'policy_surrogate_gain','choices'),
                 value_gap=partition_gap(fitting,validation,'value_squared_error','value_samples'))
        result['updates'].append(row)
        save(dict(stage='update-result',round=round_id,train_value=fitting['value_after'],
                  validation_value=validation['value_after'],train_policy=fitting['policy_surrogate_gain'],
                  validation_policy=validation['policy_surrogate_gain'],policy_gap=row['policy_gap']))
    if args.games:
        familiar = report['rounds'][0]['games'][:args.games]
        if len(familiar)!=args.games:raise ValueError('Need enough common historical training seeds')
        if any(g['seed']!=familiar[0]['seed']+i or g['index']!=familiar[0]['index']+i for i,g in enumerate(familiar)):
            raise ValueError('Expected contiguous historical seeds/indices')
        used={g['seed'] for r in report['rounds'] for g in r['games']}
        used.update(g['seed'] for e in report['evaluations'] for g in e['games'])
        assert not used.intersection(range(args.seed,args.seed+args.games))
        model=RustMagePolicy.load(checkpoints/'round-0010/policy.keras').model
        collector=ProcessGameCollector(model,workers=24,inference_processes=2,wait_ms=0)
        try:
            await collector.start(version=10)
            for checkpoint in [10,30]:
                if checkpoint!=10:
                    loaded=RustMagePolicy.load(checkpoints/f'round-{checkpoint:04d}/policy.keras').model
                    model.set_weights(loaded.get_weights())
                    await collector.sync_weights(model,checkpoint)
                for opponent in ['baseline','uniform']:
                    for name,seed in [('familiar',familiar[0]['seed']),('unused',args.seed)]:
                        if any(r['checkpoint']==checkpoint and r['opponent']==opponent and r['seeds']==name
                               for r in result['matches']): continue
                        save(dict(stage='match-start',checkpoint=checkpoint,opponent=opponent,seeds=name))
                        rows=await collector.run_games(games=args.games,seed_base=seed,
                            index_base=familiar[0]['index'],mode='versus_'+opponent,max_own_turns=60)
                        if any(r['status']=='error' for r in rows):
                            result.setdefault('failed_groups',[]).append(dict(checkpoint=checkpoint,
                                opponent=opponent,seeds=name,games=rows,reason='Game error; full group needs rerun'))
                            save(dict(stage='match-error'))
                            raise RuntimeError('Game errors; inspect saved rows')
                        entry=dict(checkpoint=checkpoint,opponent=opponent,seeds=name,
                                   summary=match_summary(rows),games=rows)
                        result['matches'].append(entry)
                        save(dict(stage='match-result',**{k:v for k,v in entry.items() if k!='games'}))
        finally:
            await collector.close()
    assert hashes=={str(p):digest(Path(p)) for p in hashes}
    result['source_files_unchanged']=True
    result['status']='complete'
    save(dict(stage='complete'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rounds',nargs='*',type=int,default=[10,20,30])
    parser.add_argument('--games',type=int,default=100)
    parser.add_argument('--seed',type=int,default=30000721)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.games<0 or any(r<1 for r in args.rounds):parser.error('Invalid counts')
    asyncio.run(run(args))
