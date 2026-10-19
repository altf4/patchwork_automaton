"""Summarize saved whole-game audits and matches without running inference."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def interval(samples):
    return np.quantile(samples,[.025,.975]).tolist()


def summarize(report):
    rng=np.random.default_rng(18721)
    resamples=10000
    summary=dict(bootstrap=dict(resamples=resamples,seed=18721,
        method='Whole-game blocks. Rollout audits stratified by update; checkpoint comparisons paired by seed/seat/start.'),
        local_objectives={},checkpoint_changes=[],unfamiliar_minus_familiar=[])
    for metric,count in [('value_squared_error','value_samples'),('policy_surrogate_gain','choices')]:
        parts=[]
        for partition in ('training','validation'):
            total=np.zeros(2); draws=np.zeros((resamples,2))
            for update in report['updates']:
                pairs=np.asarray([[g[metric],g[count]] for g in update[partition]['games']],np.float64)
                total+=pairs.sum(axis=0)
                draws+=pairs[rng.integers(len(pairs),size=(resamples,len(pairs)))].sum(axis=1)
            parts.append((float(total[0]/total[1]),draws[:,0]/draws[:,1]))
        summary['local_objectives'][metric]=dict(training_mean=parts[0][0],validation_mean=parts[1][0],
            training_minus_validation=parts[0][0]-parts[1][0],
            gap_game_bootstrap_95_interval=interval(parts[0][1]-parts[1][1]))
    matches={(m['checkpoint'],m['opponent'],m['seeds']):m for m in report['matches']}
    def scores(checkpoint,opponent,seeds):
        rows=sorted(matches[checkpoint,opponent,seeds]['games'],key=lambda g:g['index'])
        if any(g['status'] not in ('win','loss','draw') for g in rows):
            raise ValueError('Failed attempts need explicit analysis before plotting win rates')
        return rows,np.asarray([1. if g['status']=='win' else .5 if g['status']=='draw' else 0. for g in rows])
    for opponent in ('baseline','uniform'):
        differences=[]
        for seeds in ('familiar','unused'):
            old,old_score=scores(10,opponent,seeds)
            new,new_score=scores(30,opponent,seeds)
            assert [[g[k] for k in ('index','seed','seat','starting_player')] for g in old] == [
                [g[k] for k in ('index','seed','seat','starting_player')] for g in new]
            delta=new_score-old_score; differences.append(delta)
            summary['checkpoint_changes'].append(dict(opponent=opponent,seeds=seeds,games=len(delta),
                score_change=float(delta.mean()),paired_game_bootstrap_95_interval=
                interval(rng.choice(delta,size=(resamples,len(delta))).mean(axis=1))))
        delta=np.concatenate(differences)
        # Retain both seed strata in the combined comparison.
        boot=sum(rng.choice(d,size=(resamples,len(d))).sum(axis=1) for d in differences)/len(delta)
        summary['checkpoint_changes'].append(dict(opponent=opponent,seeds='combined',games=len(delta),
            score_change=float(delta.mean()),paired_game_bootstrap_95_interval=interval(boot)))
        for checkpoint in (10,30):
            _,familiar=scores(checkpoint,opponent,'familiar')
            _,unused=scores(checkpoint,opponent,'unused')
            boot=(rng.choice(unused,size=(resamples,len(unused))).mean(axis=1)-
                  rng.choice(familiar,size=(resamples,len(familiar))).mean(axis=1))
            summary['unfamiliar_minus_familiar'].append(dict(checkpoint=checkpoint,opponent=opponent,
                score_gap=float(unused.mean()-familiar.mean()),game_bootstrap_95_interval=interval(boot)))
    return summary


def render(report,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    result=dict(report,analysis=summarize(report))
    # Keep per-game scores and local objective totals; raw action traces remain
    # in the local diagnostic report, including the failed pre-fix baseline batch.
    result['matches']=[dict(m,games=[{k:g[k] for k in ('index','seed','seat','starting_player','status')}
                                   for g in m['games']]) for m in report['matches']]
    result['failed_groups']=[{k:v for k,v in g.items() if k!='games'} | dict(
        attempts=len(g['games']),errors=[r for r in g['games'] if r['status']=='error'])
        for g in report.get('failed_groups',[])]
    output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    rows=[]
    for u in report['updates']:
        for p in ('training','validation'):
            for metric in ('value_mse','policy_surrogate_gain'):
                value=u[p][metric]
                rows.append(dict(experiment='disposable_update',original_games=u['round']*100,partition=p,
                    opponent='',metric=metric,mean=value['mean'],
                    low=value['game_bootstrap_95_interval'][0],high=value['game_bootstrap_95_interval'][1]))
    for m in report['matches']:
        v=m['summary']
        rows.append(dict(experiment='frozen_checkpoint_matches',original_games=m['checkpoint']*100,
            partition=m['seeds'],opponent=m['opponent'],metric='game_score',mean=v['score'],
            low=v['score_game_bootstrap_95_interval'][0],high=v['score_game_bootstrap_95_interval'][1]))
    with output.with_suffix('.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    orange,blue,gray='#bd6225','#2463c5','#526175'
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':11,
                         'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(2,2,figsize=(12,9.5))
        fig.subplots_adjust(left=.09,right=.97,top=.84,bottom=.14,hspace=.60,wspace=.29)
        fig.suptitle('Policy and value networks · generalization audit',x=.09,y=.97,
                     ha='left',fontsize=20,fontweight='bold')
        fig.text(.09,.905,'Three disposable 80/20 updates · 800 games with frozen original checkpoints',color=gray)
        for ax,metric,title,scale in [(axes[0,0],'value_mse','Value: severe training / validation gap',1),
                (axes[0,1],'policy_surrogate_gain','Policy: smaller local objective gap',1000)]:
            for partition,color,offset in [('training',orange,-35),('validation',blue,35)]:
                selected=[r for r in rows if r['metric']==metric and r['partition']==partition]
                x=np.array([r['original_games'] for r in selected])+offset
                y=np.array([r['mean'] for r in selected])*scale
                errors=np.array([[r['mean']-r['low'] for r in selected],
                                 [r['high']-r['mean'] for r in selected]])*scale
                ax.errorbar(x,y,yerr=errors,fmt='o',capsize=4,color=color,label=partition.capitalize())
            ax.axhline(1 if metric=='value_mse' else 0,color=gray,linestyle='--',linewidth=1,
                       label='Predict zero: MSE 1' if metric=='value_mse' else 'No change')
            ax.set_title(title,loc='left',fontsize=12,fontweight='bold')
            ax.set_ylabel('Mean squared error · lower is better' if scale==1 else 'Surrogate gain × 1,000 · higher is better')
            ax.set_xlabel('Saved rollout at original game count')
            ax.set_xticks([1000,2000,3000]);ax.set_xlim(650,3350)
            ax.legend(frameon=False,fontsize=9)
        for ax,opponent,title in [(axes[1,0],'baseline','Actual wins vs scripted baseline'),
                                  (axes[1,1],'uniform','Actual wins vs uniform-random opponent')]:
            for partition,color,offset in [('familiar',orange,-.08),('unused',blue,.08)]:
                selected=[r for r in rows if r['opponent']==opponent and r['partition']==partition]
                x=np.arange(len(selected))+offset
                y=np.array([r['mean'] for r in selected])
                ax.errorbar(x,y,yerr=[[r['mean']-r['low'] for r in selected],
                    [r['high']-r['mean'] for r in selected]],fmt='o',capsize=4,color=color,
                    label='Familiar seeds' if partition=='familiar' else 'Unused seeds')
                for px,py in zip(x,y):
                    ax.annotate(f'{py:.0%}',(px,py),xytext=(-24 if partition=='familiar' else 24,0),
                        textcoords='offset points',ha='center',va='center',color=color,fontweight='bold')
            ax.set_title(title,loc='left',fontsize=12,fontweight='bold')
            ax.set_xticks([0,1],['1,000','3,000']);ax.set_xlim(-.45,1.45)
            ax.set_ylim((0,.28) if opponent=='baseline' else (.65,1.01))
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_ylabel('Wins / 100 games per condition')
            ax.set_xlabel('Original frozen checkpoint: training games')
            ax.legend(frameon=False,fontsize=9,loc='upper left' if opponent=='baseline' else 'lower left')
        for ax in axes.flat:
            ax.grid(axis='y',color='#e4e9ef');ax.set_axisbelow(True)
        fig.text(.09,.035,'Top: 80 training / 20 held-out games per update; replay current recurrence. Policy proxy is not a win rate.\n'
            'Bottom: identical seed/seat/start settings across checkpoints; familiar seeds generate new trajectories vs these opponents.\n'
            'Error bars: 95% whole-game bootstrap. One training run; all 800 scored games ended naturally. Terminal rewards only.',
            color=gray,fontsize=9)
        for suffix in ('.png','.svg'): fig.savefig(output.with_suffix(suffix),dpi=160,facecolor='white')
        plt.close(fig)
    return result['analysis']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(render(json.loads(args.report.read_text()),args.output),indent=2))
