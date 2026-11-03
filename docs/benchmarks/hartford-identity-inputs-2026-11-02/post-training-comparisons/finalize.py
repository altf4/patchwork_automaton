"""Audit all new games, fetch durable evidence, and graph both comparisons."""
from datetime import datetime, timezone
import base64
import gzip
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

BASE = Path(__file__).resolve().parent
REMOTE = r'''
import base64,collections,gzip,hashlib,json,os,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
s=json.loads(sys.argv[1]);root=Path(s['output']);sys.path.insert(0,str(root))
from evaluate import verify,schedule,cluster_intervals,sha
from patchwork_automaton.baselines.standard_decks import HARTFORD_DECKS,STANDARD_DECKS
from patchwork_automaton.rl.hartford_evaluation import validate_row,summarize,outcome_summary
assert sha(root/'provenance.json')==s['provenance_sha256']
provenance=verify()
g=json.loads((Path(s['guard'])/'guard-status.json').read_text())
assert g['status']=='exited' and g['guard_exit_code']==g['child_exit_code']==g['exit_code']==0
assert g['windows_awake']['release']['graceful_release']
try:os.kill(g['child_pid'],0)
except ProcessLookupError:pass
else:raise RuntimeError('Evaluation child still alive')
result=json.loads((root/'summary.json').read_text());config=json.loads((root/'config.json').read_text())
assert config==s['config'] and result['status']=='complete' and result['games']==11200 and not result['training']
rows_by_panel={};manifests={};identities=set();audits={}
for panel in config['panels']:
 name=panel['name'];folder=root/name;decks=tuple(HARTFORD_DECKS if panel['kind']=='neural' else STANDARD_DECKS)
 planned=schedule(panel,config,decks);expected={r['index']:r for r in planned}
 assert json.loads((folder/'schedule.json').read_text())==planned
 manifest=json.loads((folder/'manifest.json').read_text());manifests[name]=manifest
 assert manifest['config']==config and manifest['panel']==panel and not manifest['training']
 assert manifest['source_provenance_sha256']==s['provenance_sha256']
 assert sha(folder/'schedule.json')==manifest['schedule_sha256']
 assert manifest['model_input_formats']==(['sparse-identity-standard-v2','sparse-factored-standard-v1'] if panel['kind']=='neural' else ['sparse-identity-standard-v2' if panel['model']=='identity' else 'sparse-factored-standard-v1'])
 rows=[];seen=set()
 for path in (folder/'games').glob('*.json'):
  row=json.loads(path.read_text());i=row['index'];assert i in expected and i not in seen and i not in identities
  assert path.name==f'game-{i}.json'
  seen.add(i);identities.add(i);validate_row(row,expected[i],checkpoint_round=panel['version']);rows.append(row)
 assert set(expected)==seen and len(rows)==panel['games']
 assert not list((folder/'failures').glob('*.json'))
 summary=summarize(rows,decks)
 by_opponent={d:outcome_summary([r for r in rows if r['opponent_deck']==d]) for d in decks}
 if panel['kind']=='scripted':
  summary.update(cluster_intervals(rows))
  for d in decks:by_opponent[d].update(cluster_intervals([r for r in rows if r['opponent_deck']==d]))
 saved=json.loads((folder/'summary.json').read_text())
 assert saved==result['panels'][name] and saved['status']=='complete' and not saved['training']
 assert saved['summary']==summary and saved['by_opponent']==by_opponent
 assert saved['all_source_models_and_training_artifacts_unchanged']
 rows_by_panel[name]=rows
 audits[name]=dict(games=len(rows),all_scheduled_identities_verified=True,balanced_decks_seats_starts=True,
  no_training_trajectories=True,summaries_recomputed=True,failed_games=0)
assert len(identities)==11200
def paired_difference(identity,parent):
 def groups(rows):
  result={}
  for r in rows:result.setdefault(r['seed'],[]).append(r)
  return result
 a,b=groups(identity),groups(parent);assert set(a)==set(b)
 changes=[]
 for seed in sorted(a):
  key=lambda r:(r['learner_deck'],r['opponent_deck'],r['learner_seat'],r['starting_player'])
  assert {key(r) for r in a[seed]}=={key(r) for r in b[seed]}
  score=lambda rows:sum(r['learner_won']+.5*r['draw'] for r in rows)/len(rows)
  changes.append(score(a[seed])-score(b[seed]))
 changes=np.array(changes);rng=np.random.default_rng(2026110305);draws=[]
 for _ in range(10):draws.extend(changes[rng.integers(0,len(changes),size=(1000,len(changes)))].mean(axis=1))
 return dict(score_difference=float(changes.mean()),ci95=np.quantile(draws,[.025,.975]).tolist(),
  method='10000 paired bootstrap resamples of 400 matched seed clusters')
a=rows_by_panel['identity-vs-scripted'];b=rows_by_panel['parent-vs-scripted']
differences={'overall':paired_difference(a,b)}
for deck in STANDARD_DECKS:differences[deck]=paired_difference([r for r in a if r['opponent_deck']==deck],[r for r in b if r['opponent_deck']==deck])
samples=[json.loads(z) for z in (Path(s['guard'])/'host-memory-samples.jsonl').read_text().splitlines()]
assert max(z['low_memory_seconds'] for z in samples)==0
for directory,count in [(Path(provenance['identity_training_output']),100000),(Path(provenance['parent_training_output']),200000)]:
 report=json.loads((directory/'report.json').read_text());assert report['completed_games']==count and report['status']=='complete'
payload=gzip.compress(json.dumps(rows_by_panel,sort_keys=True,separators=(',',':')).encode(),mtime=0)
audit=dict(status='passed',audited_at=datetime.now(timezone.utc).isoformat(),games=11200,panels=audits,
 guard_exit_code=0,minimum_available_gib=min(z['available_memory_bytes'] for z in samples)/1024**3,
 all_original_training_checkpoints_and_sources_unchanged=True,provenance_files_verified=len(provenance['files']),
 games_gzip_sha256=hashlib.sha256(payload).hexdigest(),engine=s['engine'],
 limitation='Identity arm has 100000 additional training games; this is not an isolated architecture ablation.')
print(json.dumps(dict(audit=audit,result=result,manifests=manifests,paired_baseline_differences=differences,
 games_gzip_b64=base64.b64encode(payload).decode())))
'''


def plot(result, differences):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    labels = {'mono-green-landfall': 'Mono-green landfall', 'izzet-prowess': 'Izzet prowess',
        'bw-lifegain': 'BW lifegain', 'dimir-midrange': 'Dimir midrange',
        'four-color-control': 'Four-color control', 'jund-sacrifice': 'Jund sacrifice',
        'izzet-spellementals': 'Izzet spellementals', 'jeskai-artifacts': 'Jeskai artifacts',
        'mardu-discard': 'Mardu discard', 'bant-airbending': 'Bant airbending'}
    pair = result['panels']['identity-vs-parent']['summary']
    fig = plt.figure(figsize=(14, 8.5), facecolor='white')
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.65], left=.13, right=.95, top=.8, bottom=.29, wspace=.48)
    ax, heat = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    fig.suptitle('Identity agent vs original 200K agent', x=.04, y=.97, ha='left', fontsize=21, fontweight='bold')
    fig.text(.04, .905, '4,000 games · 100 ordered Hartford matchups · 40 games per matchup', color='#52606d', fontsize=12)
    groups = [pair, pair['on_play'], pair['on_draw']]
    values = np.array([v['score'] for v in groups]) * 100
    ci = np.array([v['score_ci95'] for v in groups]) * 100
    ax.barh(range(3), values, height=.55, color=['#1768ac', '#529dc5', '#529dc5'])
    ax.errorbar(values, range(3), xerr=[values-ci[:, 0], ci[:, 1]-values], fmt='none', ecolor='#243b53', capsize=4)
    ax.set_yticks(range(3), ['Overall', 'On the play', 'On the draw']); ax.invert_yaxis()
    ax.set_xlim(0, 100); ax.xaxis.set_major_formatter(PercentFormatter(100))
    ax.axvline(50, ls='--', color='#7b8794'); ax.set_xlabel('Identity agent score'); ax.set_title('Wins + half draws')
    for i, value in enumerate(values): ax.text(min(value+4, 90), i, f'{value:.1f}%', va='center', fontweight='bold')
    decks = pair['deck_ids']; matrix = np.array(pair['score_matrix']) * 100
    image = heat.imshow(matrix, vmin=0, vmax=100, cmap='RdBu')
    heat.set_xticks(range(10), [labels[d] for d in decks], rotation=50, ha='right', fontsize=8)
    heat.set_yticks(range(10), [labels[d] for d in decks], fontsize=8)
    heat.set_xlabel('Original 200K agent deck'); heat.set_ylabel('Identity agent deck')
    heat.set_title('Score by deck matchup')
    for i in range(10):
        for j in range(10): heat.text(j, i, f'{matrix[i,j]:.0f}', ha='center', va='center', fontsize=7, color='white' if matrix[i,j] < 25 or matrix[i,j] > 75 else '#172b4d')
    bar = fig.colorbar(image, ax=heat, fraction=.045, pad=.03); bar.ax.yaxis.set_major_formatter(PercentFormatter(100))
    fig.text(.04, .075, '95% error bars: conservative Hoeffding bounds. Individual cells have 40 games. Dashed line: 50% parity.', color='#52606d')
    fig.text(.04, .035, 'Identity model: parent 200K + 100K new training games. Both seats sample at temperature 1 on the same engine.', color='#52606d')
    for ext in ('png', 'svg'): fig.savefig(BASE / ('identity-vs-parent.' + ext), dpi=170, bbox_inches='tight', pad_inches=.2)
    plt.close(fig)

    identity = result['panels']['identity-vs-scripted']
    parent = result['panels']['parent-vs-scripted']
    decks = identity['summary']['deck_ids']
    keys = ['overall'] + decks
    title_labels = ['Overall'] + [labels[d] for d in decks]
    a = [identity['summary']] + [identity['by_opponent'][d] for d in decks]
    b = [parent['summary']] + [parent['by_opponent'][d] for d in decks]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), gridspec_kw={'width_ratios': [1.65, 1]}, facecolor='white')
    fig.subplots_adjust(left=.16, right=.96, top=.76, bottom=.24, wspace=.4)
    fig.suptitle('Play against the scripted baselines', x=.04, y=.965, ha='left', fontsize=21, fontweight='bold')
    fig.text(.04, .875, 'Original three-deck StandardAggro benchmark · 3,600 games per agent · matched seeds and decklists', color='#52606d', fontsize=11)
    ax, delta = axes
    y = np.arange(4)
    for offset, group, color, name in [(-.18, b, '#9ca9b6', 'Original 200K'), (.18, a, '#1768ac', 'Identity +100K')]:
        values = np.array([s['score'] for s in group]) * 100
        ci = np.array([s['score_ci95'] for s in group]) * 100
        ax.barh(y+offset, values, height=.31, color=color, label=name)
        ax.errorbar(values, y+offset, xerr=[values-ci[:,0], ci[:,1]-values], fmt='none', ecolor='#243b53', capsize=3, lw=1)
        for pos, value, upper in zip(y+offset, values, ci[:, 1]): ax.text(upper+2, pos, f'{value:.1f}%', va='center', fontsize=9)
    ax.set_yticks(y, title_labels); ax.invert_yaxis(); ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(PercentFormatter(100)); ax.axvline(50, ls='--', color='#7b8794')
    ax.set_xlabel('Score against script (wins + half draws)'); ax.legend(loc='upper left', bbox_to_anchor=(0, 1.24), frameon=False)
    values = np.array([differences[k]['score_difference'] for k in keys]) * 100
    ci = np.array([differences[k]['ci95'] for k in keys]) * 100
    delta.errorbar(values, y, xerr=[values-ci[:,0], ci[:,1]-values], fmt='o', color='#1768ac', capsize=4)
    delta.set_yticks(y, ['']*4); delta.invert_yaxis(); delta.axvline(0, ls='--', color='#7b8794')
    delta.set_xlabel('Identity − parent (percentage points)'); delta.set_title('Paired score change', pad=18)
    for pos, value in zip(y, values): delta.annotate(f'{value:+.1f} pp', (value, pos), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=10)
    delta.margins(x=.45, y=.3)
    delta.set_ylim(ax.get_ylim())
    fig.text(.04, .11, '95% bootstrap intervals use 400 seed clusters; score changes use paired resampling across both agents.', color='#52606d', fontsize=10)
    fig.text(.04, .055, 'Same repaired engine for every arm. Extra training and identity inputs both changed; this does not isolate architecture effects.', color='#52606d', fontsize=10)
    for ext in ('png', 'svg'): fig.savefig(BASE / ('identity-vs-scripted.' + ext), dpi=180, bbox_inches='tight', pad_inches=.2)
    plt.close(fig)


def main():
    launch = json.loads((BASE / 'launch.json').read_text())
    config = {k: launch[k] for k in ('output', 'guard', 'config', 'provenance_sha256', 'engine')}
    command = ['env', *[k + '=' + v for k, v in launch['environment'].items()],
        launch['remote']['venv_python'], '-c', REMOTE, json.dumps(config)]
    run = subprocess.run(launch['remote']['ssh_argv_prefix'] + [shlex.join(command)], capture_output=True, timeout=120)
    (BASE / 'finalize-stderr.log').write_bytes(run.stderr)
    if run.returncode: raise RuntimeError(run.stderr.decode())
    value = json.loads(run.stdout)
    raw = base64.b64decode(value.pop('games_gzip_b64'))
    assert hashlib.sha256(raw).hexdigest() == value['audit']['games_gzip_sha256']
    (BASE / 'games.json.gz').write_bytes(raw)
    assert len(json.loads(gzip.decompress(raw))) == 3
    for name, data in value.items(): (BASE / (name + '.json')).write_text(json.dumps(data, indent=2) + '\n')
    plot(value['result'], value['paired_baseline_differences'])
    launch.update(status='complete_audited', completed_at=value['audit']['audited_at'], audit='audit.json', figures=['identity-vs-parent.png', 'identity-vs-scripted.png'])
    (BASE / 'launch.json').write_text(json.dumps(launch, indent=2) + '\n')
    print(json.dumps(dict(audit=value['audit'], scores={k:{z:v['summary'][z] for z in ('games','wins','losses','draws','score','score_ci95')} for k,v in value['result']['panels'].items()},paired_baseline_differences=value['paired_baseline_differences']), indent=2))


if __name__ == '__main__':
    main()
