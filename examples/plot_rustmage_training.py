"""Plot saved held-out evaluations; never run games or change model weights.

uv run --extra plots python -m examples.plot_rustmage_training path/to/report.json
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def curve_points(report):
    """Wins-only means and uncertainty, with every failed attempt kept visible."""
    points = []
    for evaluation in sorted(report['evaluations'], key=lambda e: e['training_games']):
        games = evaluation['games']
        wins = np.asarray([g['own_turns_to_win'] for g in games if g['status']=='win'], np.float64)
        low = high = mean = None
        if len(wins):
            mean = float(wins.mean())
            rng = np.random.default_rng(9901)
            boot = rng.choice(wins, size=(5000, len(wins))).mean(axis=1)
            low, high = map(float, np.quantile(boot, [.025, .975]))
        points.append(dict(training_games=evaluation['training_games'], mean_kill_turn=mean,
            ci95_low=low, ci95_high=high, wins=len(wins), attempts=len(games),
            failures=len(games)-len(wins), completion_rate=len(wins)/len(games) if games else 0.,
            capped_turns_mean=evaluation['summary']['capped_turns_mean'],
            checkpoint=evaluation['checkpoint']))
    return points


def plot_losses(report, output):
    """Plot accepted post-update metrics on each round's own frozen rollout."""
    import matplotlib.pyplot as plt
    if not report['rounds']: return []
    rows = []
    for round_ in report['rounds']:
        metrics = round_['update']
        row = dict(training_games=round_['round']*report['config']['games_per_round'],
                   round=round_['round'])
        row.update({k: metrics.get(k) for k in ('loss','policy_loss','value_loss','entropy',
            'exact_kl','max_state_kl','gradient_norm','rejected_epochs','accepted_steps',
            'value_loss_before','critic_steps','critic_gradient_norm')})
        rows.append(row)
    output = Path(output)
    csv_path = output.with_suffix('.csv')
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with plt.rc_context({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.spines.top':False, 'axes.spines.right':False}):
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), sharex=True)
        fig.subplots_adjust(left=.095, right=.97, top=.84, bottom=.16, hspace=.34, wspace=.28)
        fig.suptitle('Foundations mono-red · PPO training diagnostics', x=.095, y=.97,
                     ha='left', fontsize=18, fontweight='bold')
        fig.text(.095, .915, 'Accepted post-update metrics · one point per self-play collection round',
                 fontsize=11, color='#526175')
        x = [r['training_games'] for r in rows]
        panels = [('loss','Total PPO loss','#2463c5'), ('policy_loss','Policy loss','#8b51a3'),
                  ('value_loss','Critic loss (mean squared error)','#bd6225'),
                  ('entropy','Policy entropy (nats)','#258374')]
        for ax, (key, label, color) in zip(axes.flat, panels):
            ax.plot(x, [r[key] for r in rows], color=color, marker='o', markersize=3.5, linewidth=1.7)
            if key=='value_loss' and all(r['value_loss_before'] is not None for r in rows):
                ax.plot(x,[r['value_loss_before'] for r in rows],color='#7f8b99',linestyle=':',
                        linewidth=1.2,label='Before update (same rollout)')
                ax.lines[0].set_label('After update')
                ax.legend(fontsize=8,frameon=False)
            ax.set_title(label, loc='left', fontsize=11, fontweight='bold')
            ax.grid(axis='y', color='#e4e9ef'); ax.set_axisbelow(True)
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        for ax in axes[-1]: ax.set_xlabel('Self-play games attempted')
        separate=report['config'].get('architecture')=='separate'
        fig.text(.095, .035,
            ('Actor and critic use independent optimizers; the combined total is a diagnostic only.\n' if separate else '')+
            'Total = policy loss + 0.5 × value loss − 0.01 × entropy. Rewards remain terminal win/loss/draw only.\n'
            'Rounds contain different self-play data; losses need not fall monotonically. Use held-out kill turn to assess progress.',
            fontsize=8.5, color='#526175')
        paths = []
        for suffix in ('.png', '.svg'):
            path = output.with_suffix(suffix)
            fig.savefig(path, dpi=160, facecolor='white'); paths.append(str(path))
        plt.close(fig)
    print(json.dumps(dict(stage='loss_plot', files=paths, csv=str(csv_path))), flush=True)
    return paths


def plot_critic(report, output):
    """Distinguish fitting the rollout from predicting its outcomes beforehand."""
    import matplotlib.pyplot as plt
    rounds = report['rounds']
    if not rounds or not all(r['update'].get('behavior_critic') for r in rounds): return []
    rows = [dict(training_games=r['round']*report['config']['games_per_round'],
                 post_update_mse=r['update']['value_loss'], **r['update']['behavior_critic']) for r in rounds]
    output = Path(output)
    with output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    x = [r['training_games'] for r in rows]
    def column(name): return [np.nan if r[name] is None else r[name] for r in rows]
    with plt.rc_context({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.spines.top':False, 'axes.spines.right':False}):
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), sharex=True)
        fig.subplots_adjust(left=.095, right=.97, top=.84, bottom=.17, hspace=.37, wspace=.28)
        fig.suptitle('Foundations mono-red · critic learning', x=.095, y=.97,
                     ha='left', fontsize=19, fontweight='bold')
        fig.text(.095, .915, 'Fresh-game predictions were recorded before training on those outcomes',
                 fontsize=11, color='#526175')
        loss, explained, spread, mean = axes.flat
        loss.plot(x, column('mse'), color='#2463c5', marker='o', label='Fresh games, before update')
        loss.plot(x, column('post_update_mse'), color='#bd6225', marker='o', label='Same games, after update')
        loss.plot(x, column('zero_baseline_mse'), color='#7f8b99', linestyle='--', label='Always predict zero')
        loss.set_title('Critic mean squared error · lower is better', loc='left', fontweight='bold', fontsize=10)
        loss.legend(frameon=False, fontsize=8)
        explained.plot(x, column('explained_variance'), color='#258374', marker='o')
        explained.axhline(0, color='#7f8b99', linestyle='--')
        explained.set_title('Fresh-game explained variance · higher is better', loc='left', fontweight='bold', fontsize=10)
        spread.plot(x, column('prediction_std'), color='#8b51a3', marker='o')
        spread.axhline(0, color='#7f8b99', linestyle='--')
        spread.set_title('Fresh-game prediction standard deviation', loc='left', fontweight='bold', fontsize=10)
        mean.plot(x, column('prediction_mean'), color='#2463c5', marker='o', label='Mean prediction')
        mean.plot(x, column('target_mean'), color='#7f8b99', linestyle=':', marker='o', label='Mean terminal target')
        mean.set_title('Fresh-game prediction bias', loc='left', fontweight='bold', fontsize=10)
        mean.legend(frameon=False, fontsize=8)
        for ax in axes.flat:
            ax.grid(axis='y', color='#e4e9ef'); ax.set_axisbelow(True)
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        for ax in axes[-1]: ax.set_xlabel('Self-play games attempted')
        fig.text(.095, .035,
            'X marks the end of each collection round. Fresh-game predictions use the checkpoint from the preceding update.\n'
            'These are successive on-policy rollouts, not one fixed validation set. Prediction spread alone does not measure accuracy.\n'
            'Targets remain terminal win/loss/draw (+1 / −1 / 0); no intermediate rewards.',
            fontsize=8.5, color='#526175')
        paths = []
        for suffix in ('.png', '.svg'):
            path = output.with_suffix(suffix)
            fig.savefig(path, dpi=160, facecolor='white'); paths.append(str(path))
        plt.close(fig)
    print(json.dumps(dict(stage='critic_plot', files=paths, csv=str(output.with_suffix('.csv')))), flush=True)
    return paths


def plot_report(path, output=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    path = Path(path)
    report = json.loads(path.read_text())
    points = curve_points(report)
    if not points: raise ValueError('No checkpoint evaluations in report')
    output = Path(output) if output else path.parent/'learning-curve'
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix('.csv')
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]))
        writer.writeheader(); writer.writerows(points)
    x = np.asarray([p['training_games'] for p in points])
    def column(name): return np.asarray([np.nan if p[name] is None else p[name] for p in points])
    blue, orange, gray = '#2463c5', '#bd6225', '#526175'
    with plt.rc_context({'font.family':'DejaVu Sans', 'font.size':11,
                         'axes.spines.top':False, 'axes.spines.right':False}):
        fig, (ax, success) = plt.subplots(2, 1, figsize=(11.5, 7.8), sharex=True,
            gridspec_kw={'height_ratios':[3.4,1]})
        fig.subplots_adjust(left=.095, right=.98, top=.86, bottom=.15, hspace=.14)
        fig.suptitle('Foundations mono-red · PPO learning curve', x=.095, y=.97,
                     ha='left', fontsize=19, fontweight='bold')
        fig.text(.095, .918,
                 f"{report['config']['eval_games']} fixed goldfish seeds per checkpoint · lower kill turn is better"
                 + (' · independent actor / critic' if report['config'].get('architecture')=='separate' else ''),
                 fontsize=11, color=gray)
        ax.plot(x, column('mean_kill_turn'), color=blue, marker='o', linewidth=2.4,
                label='Neural policy: mean own turns to win')
        ax.fill_between(x, column('ci95_low'), column('ci95_high'), color=blue, alpha=.13,
                        label='95% bootstrap interval over evaluation games')
        baseline = report.get('baseline',{}).get('summary',{}).get('mean_own_turns_to_win')
        if baseline is not None:
            ax.axhline(baseline, color=gray, linestyle='--', linewidth=1.5,
                       label=f'Scripted baseline: {baseline:.2f} turns')
        if any(p['failures'] for p in points):
            ax.plot(x, column('capped_turns_mean'), color=orange, linestyle=':', linewidth=2,
                    label=f"Failure-adjusted score (failure = {report['config']['eval_horizon']+1} turns)")
        ax.set_ylabel('Own turns to win (successful attempts)')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis='y', color='#e4e9ef'); ax.set_axisbelow(True)
        ax.legend(loc='upper right', fontsize=9, frameon=False)
        for point in (points[0], points[-1]) if len(points)>1 else points:
            if point['mean_kill_turn'] is not None:
                ax.annotate(f"{point['mean_kill_turn']:.2f}",
                    (point['training_games'], point['mean_kill_turn']),
                    xytext=(0,11), textcoords='offset points', ha='center',
                    color=blue, fontweight='bold')
        success.plot(x, 100*column('completion_rate'), color=blue, marker='o', linewidth=1.8)
        for point in points:
            success.annotate(f"{point['wins']}/{point['attempts']}",
                (point['training_games'],100*point['completion_rate']),
                xytext=(0,-18), textcoords='offset points', ha='center', fontsize=8, color=gray)
        success.set_ylim(-5,110); success.set_yticks([0,50,100])
        success.yaxis.set_major_formatter(PercentFormatter(100))
        success.set_ylabel('Completed kills')
        failures = any(r['status'] not in ('win','loss','draw') for round_ in report['rounds'] for r in round_['games'])
        success.set_xlabel('Self-play games attempted' if failures else 'Self-play games completed')
        success.set_xticks(x)
        success.grid(axis='y', color='#e4e9ef'); success.set_axisbelow(True)
        fig.text(.095, .025,
            'Training rewards: terminal win/loss/draw only. Evaluation: sampled policy vs a do-nothing opponent.\n'
            'Intervals describe evaluation-game variability for one training run; failures are shown separately.',
            fontsize=8.5, color=gray)
        paths = []
        for suffix in ('.png', '.svg'):
            target = output.with_suffix(suffix)
            fig.savefig(target, dpi=160, facecolor='white')
            paths.append(str(target))
        plt.close(fig)
    print(json.dumps(dict(stage='plot', files=paths, csv=str(csv_path))), flush=True)
    return (paths + plot_losses(report, output.parent/(output.name+'-losses'))
            + plot_critic(report, output.parent/(output.name+'-critic')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, help='Output stem (PNG, SVG and CSV are written)')
    args = parser.parse_args()
    plot_report(args.report, args.output)
