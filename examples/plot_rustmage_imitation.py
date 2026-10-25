"""Plot expert imitation diagnostics separately from RL/value losses."""
import csv
import json
from pathlib import Path


def plot_report(path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    path = Path(path)
    report = json.loads(path.read_text())
    rounds = report['rounds']
    evaluations = report['evaluations']
    x = [r['games_seen'] for r in rounds]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    fig.suptitle('Foundations mono-red · scripted baseline imitation', fontsize=18, weight='bold')
    for partition, color in (('training', '#c96520'), ('validation', '#2366cf')):
        axes[0, 0].plot(x, [r[partition]['cross_entropy'] for r in rounds], 'o-', color=color, label=partition.title())
        axes[0, 1].plot(x, [r[partition]['accuracy'] for r in rounds], 'o-', color=color, label=f'{partition.title()}: all choices')
        axes[0, 1].plot(x, [r[partition]['non_pass_accuracy'] for r in rounds], '--', color=color, label=f'{partition.title()}: non-Pass choices')
    axes[0, 0].set(title='Expert action cross entropy · lower is better', ylabel='Cross entropy (nats)')
    axes[0, 1].set(title='Exact expert action accuracy · higher is better', ylabel='Fraction correct', ylim=(0, 1.02))
    for name, color in (('sampled', '#2366cf'), ('greedy', '#29917d')):
        ex = [e['games_seen'] for e in evaluations]
        axes[1, 0].plot(ex, [e[name]['summary']['capped_turns_mean'] for e in evaluations], 'o-', color=color, label=name.title())
        axes[1, 1].plot(ex, [e[name]['summary']['win_rate'] for e in evaluations], 'o-', color=color, label=name.title())
    baseline = report['baseline']['summary']['mean_own_turns_to_win']
    axes[1, 0].axhline(baseline, color='#53647a', linestyle='--', label=f'Scripted baseline: {baseline:.2f}')
    axes[1, 0].set(title='Do-nothing opponent · failure-adjusted kill turn', ylabel='Own turn (failure = horizon + 1)')
    axes[1, 1].set(title='Completed kills against do-nothing opponent', ylabel='Fraction of attempts', ylim=(0, 1.02))
    audit = report['dataset_audit']
    fraction = audit['validation_games'] / (audit['training_games'] + audit['validation_games'])
    for ax in axes.flat:
        ax.set_xlabel(f'Expert games processed ({fraction:.0%} held out)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(path.parent / 'imitation-learning.png', dpi=140)
    fig.savefig(path.parent / 'imitation-learning.svg')
    plt.close(fig)
    with (path.parent / 'imitation-learning.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('games_seen', 'training_cross_entropy', 'validation_cross_entropy',
            'training_accuracy', 'validation_accuracy', 'training_non_pass_accuracy', 'validation_non_pass_accuracy',
            'sampled_kill_turn', 'greedy_kill_turn'))
        for r in rounds:
            e = next(e for e in evaluations if e['round'] == r['round'])
            writer.writerow((r['games_seen'], r['training']['cross_entropy'], r['validation']['cross_entropy'],
                r['training']['accuracy'], r['validation']['accuracy'], r['training']['non_pass_accuracy'],
                r['validation']['non_pass_accuracy'], e['sampled']['summary']['capped_turns_mean'],
                e['greedy']['summary']['capped_turns_mean']))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('report', type=Path)
    plot_report(p.parse_args().report)
