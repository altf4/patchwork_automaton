"""Audit and graph the authorized 10,000-game endpoint; never launch games."""
import argparse
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

BASE = Path(__file__).resolve().parent


def plot(result, output):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/compact-ppo-mpl')
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
    summary = result['summary']
    fig = plt.figure(figsize=(14, 8.6), facecolor='white')
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.65], left=.13, right=.95,
                           top=.79, bottom=.30, wspace=.48)
    ax, heat = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    fig.suptitle('Broader opponents vs continued self-play', x=.04, y=.97,
                 ha='left', fontsize=21, fontweight='bold')
    games = result['games']
    fig.text(.04, .905, f'{games:,} games · 100 ordered Hartford matchups · {games // 100} games per matchup',
             color='#52606d', fontsize=12)
    groups = [summary, summary['on_play'], summary['on_draw']]
    values = np.array([v['score'] for v in groups]) * 100
    ci = np.array([v['score_ci95'] for v in groups]) * 100
    ax.barh(range(3), values, height=.55, color=['#1768ac', '#529dc5', '#529dc5'])
    ax.errorbar(values, range(3), xerr=[values-ci[:, 0], ci[:, 1]-values],
                fmt='none', ecolor='#243b53', capsize=4)
    ax.set_yticks(range(3), ['Overall', 'On the play', 'On the draw'])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(PercentFormatter(100))
    ax.axvline(50, ls='--', color='#7b8794')
    ax.set_xlabel('Diverse-opponent agent score')
    ax.set_title('Wins + half draws')
    for i, value in enumerate(values):
        ax.text(min(value+4, 89), i, f'{value:.1f}%', va='center', fontweight='bold')
    decks = summary['deck_ids']
    matrix = np.array(summary['score_matrix']) * 100
    image = heat.imshow(matrix, vmin=0, vmax=100, cmap='RdBu')
    heat.set_xticks(range(10), [labels.get(d, d) for d in decks], rotation=50, ha='right', fontsize=8)
    heat.set_yticks(range(10), [labels.get(d, d) for d in decks], fontsize=8)
    heat.set_xlabel('Control self-play agent deck')
    heat.set_ylabel('Diverse-opponent agent deck')
    heat.set_title('Score by deck matchup')
    for i in range(10):
        for j in range(10):
            heat.text(j, i, f'{matrix[i,j]:.0f}', ha='center', va='center', fontsize=7,
                      color='white' if matrix[i,j] < 25 or matrix[i,j] > 75 else '#172b4d')
    bar = fig.colorbar(image, ax=heat, fraction=.045, pad=.03)
    bar.ax.yaxis.set_major_formatter(PercentFormatter(100))
    fig.text(.04, .12, 'Both branches: same identity checkpoint and optimizer/RNG state + 50K additional training games.', color='#52606d')
    fig.text(.04, .075, '95% error bars: conservative Hoeffding bounds. Draws score one half; dashed line marks 50% parity.', color='#52606d')
    fig.text(.04, .03, 'One training run per recipe. This duel measures relative strength between these two agents, not generalization.', color='#52606d')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(output / ('head-to-head.' + ext), dpi=170, bbox_inches='tight', pad_inches=.2)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', type=Path, default=BASE / 'launch.json')
    parser.add_argument('--output', type=Path, default=BASE / 'head-to-head')
    args = parser.parse_args()
    launch = json.loads(args.launch.read_text())
    remote_base = Path(launch['arms']['control']['output']).parent
    remote_launch = launch.get('remote_launch', str(remote_base / 'launch.json'))
    evaluator = launch['evaluation'].get('script', str(remote_base / 'evaluate.py'))
    command = ['env', *[k + '=' + v for k, v in launch['environment'].items()],
        launch['remote']['venv_python'], evaluator, '--launch', remote_launch, '--audit-only']
    completed = subprocess.run(launch['remote']['ssh_argv_prefix'] + [shlex.join(command)],
        capture_output=True, timeout=180)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'finalize-stderr.log').write_bytes(completed.stderr)
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode())
    value = json.loads(completed.stdout)
    if value['audit']['passed'] is not True or value['audit']['games'] != 10000:
        raise ValueError('Incomplete final comparison audit')
    payload = base64.b64decode(value.pop('games_gzip_b64'), validate=True)
    if hashlib.sha256(payload).hexdigest() != value['audit']['games_gzip_sha256']:
        raise ValueError('Fetched game evidence checksum mismatch')
    rows = json.loads(gzip.decompress(payload))
    if len(rows) != 10000 or len({r['index'] for r in rows}) != 10000:
        raise ValueError('Fetched game evidence has missing or duplicate identities')
    if value['manifest']['engine'] != launch['engine']:
        raise ValueError('Fetched evaluation uses a different pinned engine')
    existing = [path for path in args.output.iterdir() if path.is_file() and path.name != 'finalize-stderr.log']
    if existing:
        history = args.output / 'finalization-history' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        history.mkdir(parents=True)
        for path in existing:
            (history / path.name).write_bytes(path.read_bytes())
    (args.output / 'games.json.gz').write_bytes(payload)
    for name, content in value.items():
        (args.output / (name + '.json')).write_text(json.dumps(content, indent=2, allow_nan=False) + '\n')
    plot(value['result'], args.output)
    s = value['result']['summary']
    text = f'''# Identity opponent-diversity endpoint comparison

The completed 10,000-game head-to-head passed its complete schedule, outcome,
model/source/parent integrity and resource-guard audit. Both training arms
completed 50,000 additional games from the same identity checkpoint with the
same initial actor/critic optimizer and trainer RNG state.

The diverse-opponent agent scored **{s['score']:.2%}** against the control
self-play agent (95% conservative interval {s['score_ci95'][0]:.2%}–{s['score_ci95'][1]:.2%}):
{s['wins']:,} wins, {s['losses']:,} losses and {s['draws']:,} draws.
Score counts draws as half a win; raw win rate was {s['win_rate']:.2%}.

![Head-to-head results](head-to-head.png)

The panel contains 100 games in each of 100 ordered Hartford deck matchups,
with 25 games for each learner-seat/starting-player combination. Both seats
sample at temperature 1, with a 20-second per-game deadline, 50-turn limit and
6,000-decision limit. Timeout/turn-limit draws retain their diagnostics.
The audit reports {s['natural_draws']:,} natural draws and
{s['adjudicated_games']:,} adjudications. No baseline panel was run.

This is one training run per recipe and one relative-strength comparison.
The interval measures evaluation-game uncertainty, not variation across
independent training runs or performance against unseen opponents.

Evidence: [audit](audit.json), [result](result.json), [manifest](manifest.json),
[all 10,000 outcomes](games.json.gz), [SVG figure](head-to-head.svg).
'''
    (args.output / 'README.md').write_text(text)
    print(json.dumps(dict(audit=value['audit'], score=s['score'], score_ci95=s['score_ci95'],
                          figure=str(args.output / 'head-to-head.png')), indent=2))


if __name__ == '__main__':
    main()
