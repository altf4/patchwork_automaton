"""Audit and graph the two agents against scripts and their shared identity parent."""
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path

from manage import HERE, ssh

REMOTE = r'''
import json,sys
from pathlib import Path
launch=json.loads(sys.stdin.read());sys.path.insert(0,launch['output'])
from audit import audit
print(json.dumps(audit(launch)))
'''


def plot(results, outcomes, output):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/compact-ppo-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter
    from patchwork_automaton.rl.hartford_evaluation import outcome_summary
    scripts = ('mono-green-landfall', 'izzet-prowess', 'bw-lifegain')
    labels = ('Overall', 'Mono-green', 'Izzet prowess', 'BW lifegain')
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'width_ratios': [2, 1]})
    fig.subplots_adjust(top=.78, bottom=.24, left=.07, right=.97, wspace=.30)
    colors = {'control': '#1768ac', 'diverse': '#c56a1b'}
    for number, model in enumerate(('control', 'diverse')):
        scripted = results[model + '-vs-baselines']['summary']
        groups = [scripted] + [outcome_summary([r for r in outcomes[model + '-vs-baselines']
            if r['opponent_deck'] == deck]) for deck in scripts]
        x = np.arange(4) + (number - .5) * .34
        values = np.array([r['score'] for r in groups]) * 100
        intervals = np.array([r['score_ci95'] for r in groups]) * 100
        axes[0].bar(x, values, width=.32, color=colors[model], label=model.capitalize(),
            yerr=[values - intervals[:, 0], intervals[:, 1] - values], capsize=3)
        for position, score in zip(x, values):
            axes[0].text(position, score+4, f'{score:.1f}%', ha='center', fontsize=9)
        neural = results[model + '-vs-identity-100k']['summary']
        axes[1].bar(number, neural['score'] * 100, color=colors[model], width=.6,
            yerr=[[100 * (neural['score'] - neural['score_ci95'][0])],
                  [100 * (neural['score_ci95'][1] - neural['score'])]], capsize=4)
        axes[1].text(number, neural['score'] * 100 + 4, f"{neural['score']:.1%}", ha='center')
    axes[0].set_xticks(range(4), labels)
    axes[0].set_title('Against scripted baselines')
    axes[0].legend(frameon=False)
    axes[1].set_xticks(range(2), ('Control', 'Diverse'))
    axes[1].set_title('Against shared 100K identity agent')
    for axis in axes:
        axis.set_ylim(0, 100)
        axis.yaxis.set_major_formatter(PercentFormatter(100))
        axis.axhline(50, ls='--', color='#8995a0', lw=1)
        axis.spines[['top', 'right']].set_visible(False)
        axis.set_ylabel('Score · wins + half draws')
    fig.suptitle('A/B agents against fixed opponents', x=.06, ha='left', fontsize=20, fontweight='bold')
    fig.text(.06, .86, '40,000 games · 10,000 per agent per panel · matched seeds and balanced seats', color='#52606d')
    fig.text(.06, .12, 'Both agents add 50K collection games to the same identity checkpoint and optimizer/RNG state.', color='#52606d')
    fig.text(.06, .075, '95% error bars use conservative Hoeffding bounds. One training run per recipe.', color='#52606d')
    fig.text(.06, .03, 'Scripts use the original 3×3 deck panel and 60s deadline. Identity duels use 100 Hartford matchups and 20s.', color='#52606d')
    for extension in ('png', 'svg'):
        fig.savefig(output / ('comparison.' + extension), dpi=170, bbox_inches='tight')
    plt.close(fig)


def main():
    launch = json.loads((HERE / 'launch.json').read_text())
    value = ssh(launch, REMOTE, launch)
    if not value['audit']['passed'] or value['audit']['games'] != 40000:
        raise ValueError('Completion audit failed')
    payload = base64.b64decode(value.pop('outcomes_gzip_b64'), validate=True)
    if hashlib.sha256(payload).hexdigest() != value['audit']['outcomes_gzip_sha256']:
        raise ValueError('Fetched outcomes differ from audit')
    outcomes = json.loads(gzip.decompress(payload))
    output = HERE / 'results'
    output.mkdir(exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Completed results already exist')
    (output / 'outcomes.json.gz').write_bytes(payload)
    for key, content in value.items():
        (output / (key + '.json')).write_text(json.dumps(content, indent=2, allow_nan=False) + '\n')
    plot(value['results'], outcomes, output)
    lines = ['# Fixed-opponent comparison results', '',
        '| Agent | Opponent | Games | Score | 95% interval | Wins / losses / draws |',
        '| --- | --- | ---: | ---: | --- | --- |']
    for name, result in value['results'].items():
        summary = result['summary']
        model = result['panel']['model']
        opponent = 'Scripted baselines' if result['panel']['kind'] == 'scripted' else 'Shared identity checkpoint 100'
        lines.append(f"| {model} | {opponent} | 10,000 | {summary['score']:.2%} | "
            f"{summary['score_ci95'][0]:.2%}–{summary['score_ci95'][1]:.2%} | "
            f"{summary['wins']} / {summary['losses']} / {summary['draws']} |")
    lines += ['', '![Comparison results](comparison.png)', '',
        'Score counts draws as half a win. Each agent used matching fresh seeds and seat/start assignments.',
        'The scripted panel uses the original three decklists. The shared reference is the identity checkpoint',
        'that started both arms, with 100K identity training games and 300K total inherited exposure.',
        'Both current agents have 350K total exposure. One trained agent represents each recipe.', '',
        f"The audit recovered {value['audit']['published_handoffs_recovered']} completed handoffs after transport cleanup errors.",
        'Original handoffs, adjudication diagnostics and any transport-error resolutions remain preserved.', '']
    (output / 'README.md').write_text('\n'.join(lines))
    launch.update(status='complete', completed_at=datetime.now(timezone.utc).isoformat(),
        completion=dict(audit=str(output / 'audit.json'), figure=str(output / 'comparison.png'), games=40000))
    (HERE / 'launch.json').write_text(json.dumps(launch, indent=2) + '\n')
    print(json.dumps(dict(audit=value['audit'], scores={k:v['summary']['score'] for k,v in value['results'].items()},
        figure=str(output / 'comparison.png')), indent=2))


if __name__ == '__main__':
    main()
