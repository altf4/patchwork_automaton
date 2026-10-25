"""Render performance, whole-game validation, and PPO loss from a saved run.

This reads reports only; it never imports the model or contacts the learner.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, PercentFormatter


BLUE = '#2563b4'
ORANGE = '#bd632b'
TEAL = '#127d76'
PURPLE = '#8050a2'
GRAY = '#586779'
INK = '#172d43'
GRID = '#e4eaf0'


def games_at(report, round_number):
    return int(round_number) * report['config']['games_per_round']


def date_label(report):
    captured = report.get('plot_snapshot', {}).get('captured_utc')
    if not captured:
        return 'Saved report snapshot'
    dt = datetime.fromisoformat(captured).astimezone(ZoneInfo('America/Phoenix'))
    return dt.strftime('%b %d, %Y · %-I:%M %p Phoenix')


def decoration(fig, report, title, subtitle, footer):
    fig.suptitle(title, x=.08, y=.975, ha='left', fontsize=21,
                 fontweight='bold', color=INK)
    fig.text(.08, .88, subtitle, fontsize=11, color=GRAY)
    fig.text(.08, .038, footer, fontsize=9, color=GRAY, linespacing=1.5)
    fig.text(.975, .978, date_label(report), ha='right', va='top',
             fontsize=8.5, color=GRAY)


def style_axis(ax, games=True):
    ax.grid(axis='y', color=GRID)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#a6b2bf')
    ax.spines['bottom'].set_color('#a6b2bf')
    ax.tick_params(colors=GRAY, labelsize=9)
    if games:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
        ax.set_xlabel('New self-play games collected', color=GRAY, labelpad=9)
    ax.margins(x=.04)


def save(fig, directory, name):
    for extension in ('png', 'svg'):
        fig.savefig(directory / f'{name}.{extension}', dpi=180, facecolor='white')
    plt.close(fig)


def band(ax, x, measurements, color, label, scale=1):
    y = np.asarray([m['mean'] if m else np.nan for m in measurements]) * scale
    low = np.asarray([m['game_bootstrap_95_interval'][0] if m else np.nan
                      for m in measurements]) * scale
    high = np.asarray([m['game_bootstrap_95_interval'][1] if m else np.nan
                       for m in measurements]) * scale
    ax.fill_between(x, low, high, color=color, alpha=.12, linewidth=0)
    ax.plot(x, y, '-o', color=color, label=label, markersize=4, linewidth=1.8)


def performance(report, directory):
    active = sorted(report['opponent_evaluations'], key=lambda e: e['round'])
    passive = sorted(report['evaluations'], key=lambda e: e['round'])
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.0))
    fig.subplots_adjust(left=.08, right=.97, top=.79, bottom=.23, wspace=.27)
    completed = report['training_games_completed']
    target = report['training_games_target']
    decoration(fig, report, 'Playing strength during training',
               f'{completed:,} / {target:,} new games processed · '
               f'{len(report["rounds"])} completed rounds · warm start from the earlier 1,000-game model',
               'Left: 200 fixed, seat/start-balanced attempts per checkpoint; shaded band is a pointwise 95% Wilson interval.\n'
               'Right: all 40 passive-opponent attempts count; unfinished attempts receive 26 turns (25-turn horizon + 1).\n'
               'Evaluation cohorts repeat across checkpoints. These are results from one run, not an independent final test.')
    ax = axes[0]
    x = np.array([games_at(report, e['round']) for e in active])
    y = np.array([e['summary']['win_rate_all_attempts'] for e in active])
    ci = np.array([e['summary']['win_rate_all_attempts_wilson_95_interval'] for e in active])
    ax.fill_between(x, ci[:, 0], ci[:, 1], color=TEAL, alpha=.15, linewidth=0)
    ax.plot(x, y, '-o', color=TEAL, linewidth=2.2, markersize=5)
    ax.axhline(y[0], color=GRAY, linestyle='--', linewidth=1, label=f'Initial: {y[0]:.0%}')
    ax.annotate(f'{y[-1]:.1%}  ({active[-1]["summary"]["wins"]}/'
                f'{active[-1]["summary"]["attempted_games"]})',
                (x[-1], y[-1]), xytext=(-5, -30 if len(active) > 12 else 14),
                textcoords='offset points', va='top' if len(active) > 12 else 'bottom',
                ha='right', color=TEAL, fontweight='bold', fontsize=11)
    ax.set_title('Wins against the active scripted opponent', loc='left', fontsize=12, fontweight='bold')
    ax.set_ylabel('Win rate · higher is better')
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_ylim(0, max(.22, float(ci[:, 1].max()) * 1.17))
    ax.set_yticks(np.arange(0, ax.get_ylim()[1] + .001, .05))
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    style_axis(ax)

    ax = axes[1]
    x = np.array([games_at(report, e['round']) for e in passive])
    y = np.array([e['summary']['capped_turns_mean'] for e in passive])
    ax.plot(x, y, '-o', color=PURPLE, linewidth=2.2, markersize=5)
    ax.axhline(y[0], color=GRAY, linestyle='--', linewidth=1, label=f'Initial: {y[0]:.2f} turns')
    annotation_step = max(1, int(np.ceil(len(passive) / 10)))
    for index, (px, py, e) in enumerate(zip(x, y, passive)):
        s = e['summary']
        if index % annotation_step == 0 or index == len(passive) - 1:
            ax.annotate(f'{s["win"]}/{s["games"]}', (px, py), xytext=(0, -17),
                        textcoords='offset points', ha='center', fontsize=8, color=GRAY)
    ax.annotate(f'{y[-1]:.2f} turns', (x[-1], y[-1]), xytext=(-5, 13),
                textcoords='offset points', ha='right', color=PURPLE, fontweight='bold', fontsize=11)
    ax.set_title('Time to win against the passive opponent', loc='left', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean own turns, including failures · lower is better')
    ax.set_ylim(max(0, min(y) - 2.5), max(y) + 1.6)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    ax.text(.02, .04, 'Small labels: wins / attempts', transform=ax.transAxes, fontsize=8, color=GRAY)
    style_axis(ax)
    save(fig, directory, 'performance')


def generalization(report, directory):
    rounds = [r for r in report['rounds'] if 'generalization' in r]
    x = np.array([games_at(report, r['round']) for r in rounds])
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.3))
    fig.subplots_adjust(left=.065, right=.98, top=.78, bottom=.25, wspace=.30)
    decoration(fig, report, 'Loss and generalization on unseen games',
               '400 training games + 100 held-out games each round · held-out games never enter either optimizer',
               'Bands: pointwise 95% whole-game bootstrap intervals. Both partitions replay current recurrent states from the start.\n'
               'Each round has fresh games and a changing self-play opponent; compare before/after within a round when judging overfitting.\n'
               'X includes the 20% validation partition. Policy surrogate gain is a local outcome-based proxy, not a win rate.')
    for ax, metric in zip(axes, ('value_mse', 'value_mse_change', 'policy_surrogate_gain')):
        for partition, color, label in [('training', ORANGE, 'Training'), ('validation', BLUE, 'Held-out')]:
            measurements = [r['generalization'][partition][metric] for r in rounds]
            band(ax, x, measurements, color, label, 1000 if metric == 'policy_surrogate_gain' else 1)
        style_axis(ax)
        ax.legend(frameon=False, fontsize=9, loc='upper right' if metric == 'policy_surrogate_gain' else 'best')
    axes[0].set_title('Critic error after each update', loc='left', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Mean squared error · lower is better')
    axes[0].legend(frameon=True, facecolor='white', edgecolor='white', framealpha=.95,
                   fontsize=9, loc='upper right')
    axes[0].axhline(1, color=GRAY, linestyle=':', linewidth=1)
    axes[0].text(.02, .96, 'Predict zero: MSE = 1', transform=axes[0].transAxes,
                 fontsize=8, color=GRAY, va='top')
    axes[1].set_title('Did the critic update help?', loc='left', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Error after − error before · negative is better')
    axes[1].axhline(0, color=GRAY, linestyle='--', linewidth=1)
    axes[1].text(.02, .97, 'Above zero = worse predictions', transform=axes[1].transAxes,
                 fontsize=8, color=GRAY, va='top')
    axes[2].set_title('Policy improvement on recorded choices', loc='left', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Surrogate gain × 1,000 · higher is better')
    axes[2].axhline(0, color=GRAY, linestyle='--', linewidth=1)
    save(fig, directory, 'loss-and-generalization')


def optimizer_losses(report, directory):
    rounds = report['rounds']
    x = [games_at(report, r['round']) for r in rounds]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), sharex=True)
    fig.subplots_adjust(left=.09, right=.97, top=.81, bottom=.17, hspace=.40, wspace=.26)
    decoration(fig, report, 'PPO optimization diagnostics',
               'Accepted update metrics · one point per completed round · separate actor and critic optimizers',
               'Total = policy loss + 0.5 × value loss − 0.01 × entropy; the combined total is a diagnostic only.\n'
               'These fitting losses use changing rollouts. Held-out prediction error and actual matches are the generalization checks.')
    specs = [('loss', 'Combined diagnostic loss', 'Loss', BLUE),
             ('policy_loss', 'Actor PPO loss', 'Loss', PURPLE),
             ('value_loss', 'Critic fitting loss', 'Mean squared error', ORANGE),
             ('entropy', 'Policy entropy', 'Nats', TEAL)]
    for ax, (metric, title, ylabel, color) in zip(axes.flat, specs):
        ax.plot(x, [r['update'][metric] for r in rounds], '-o', color=color,
                markersize=4, linewidth=1.8)
        ax.set_title(title, loc='left', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        style_axis(ax)
    for ax in axes[0]:
        ax.set_xlabel('')
    save(fig, directory, 'ppo-losses')


def export_data(report, directory):
    rows = []
    for r in report['rounds']:
        games = games_at(report, r['round'])
        for partition in ('training', 'validation'):
            g = r.get('generalization', {}).get(partition, {})
            for metric in ('value_mse', 'value_mse_change', 'policy_surrogate_gain'):
                m = g.get(metric)
                if m:
                    rows.append(dict(round=r['round'], collected_games=games,
                                     partition=partition, metric=metric, mean=m['mean'],
                                     ci95_low=m['game_bootstrap_95_interval'][0],
                                     ci95_high=m['game_bootstrap_95_interval'][1],
                                     attempts=g.get('attempted_games'), interval='whole_game_bootstrap'))
        for metric in ('loss', 'policy_loss', 'value_loss', 'entropy'):
            rows.append(dict(round=r['round'], collected_games=games, partition='optimizer',
                             metric=metric, mean=r['update'][metric], ci95_low='', ci95_high='',
                             attempts='', interval=''))
    for e in report.get('opponent_evaluations', []):
        s = e['summary']
        rows.append(dict(round=e['round'], collected_games=games_at(report, e['round']),
                         partition='active_opponent', metric='win_rate', mean=s['win_rate_all_attempts'],
                         ci95_low=s['win_rate_all_attempts_wilson_95_interval'][0],
                         ci95_high=s['win_rate_all_attempts_wilson_95_interval'][1],
                         attempts=s['attempted_games'], interval='wilson'))
    for e in report['evaluations']:
        rows.append(dict(round=e['round'], collected_games=games_at(report, e['round']),
                         partition='passive_opponent', metric='capped_mean_turns',
                         mean=e['summary']['capped_turns_mean'], ci95_low='', ci95_high='',
                         attempts=e['summary']['games'], interval=''))
    with (directory / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    output = args.output_dir or args.report.parent
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.labelcolor': INK, 'text.color': INK,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'}):
        performance(report, output)
        generalization(report, output)
        optimizer_losses(report, output)
    export_data(report, output)
    print(json.dumps({'output': str(output), 'rounds': len(report['rounds']),
                      'games': report['training_games_completed'],
                      'figures': ['performance', 'loss-and-generalization', 'ppo-losses']}))


if __name__ == '__main__':
    main()
