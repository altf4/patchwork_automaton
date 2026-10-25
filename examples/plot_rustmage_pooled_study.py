"""Plot backed-up pooled-study JSON only; no games, models or TensorFlow.

Repeated monitoring uses one common game cohort. Independent confirmation is
reported separately, only when the finite study is complete. These intervals
are conditional on one training seed and this fixed opponent/deck matchup.
"""
import argparse
from contextlib import contextmanager
import csv
import json
import math
import os
from pathlib import Path
import re
import tempfile


@contextmanager
def _plot_configuration():
    previous = os.environ.get('MPLCONFIGDIR')
    with tempfile.TemporaryDirectory(prefix='rustmage-mpl-') as temporary:
        os.environ['MPLCONFIGDIR'] = temporary
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop('MPLCONFIGDIR', None)
            else:
                os.environ['MPLCONFIGDIR'] = previous


def read_json(path):
    return json.loads(Path(path).read_bytes())


def _number(value):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def collect_points(study_root):
    root = Path(study_root)
    study = read_json(root/'study.json') if (root/'study.json').exists() else {}
    report = read_json(root/'training/report.json')
    monitors = []
    for path in sorted(root.glob('monitor*.json')):
        match = re.fullmatch(r'monitor-(imitation|\d{4,})\.json', path.name)
        if match is None:
            continue
        evaluation = read_json(path)
        if evaluation.get('status') != 'complete':
            continue
        summary = evaluation['summary']
        attempts = summary['attempted_games']
        if (type(attempts) is not int or attempts < 1
                or ('games' in evaluation and len(evaluation['games']) != attempts)):
            raise ValueError('Incomplete monitoring denominator: '+str(path))
        rate = _number(summary.get('win_rate_all_attempts'))
        interval = summary.get('win_rate_all_attempts_wilson_95_interval')
        if rate is None or not 0 <= rate <= 1 or not isinstance(interval, list) or len(interval) != 2:
            raise ValueError('Missing monitoring win rate/interval')
        round_id = 0 if match[1] == 'imitation' else int(match[1])
        monitors.append(dict(round=round_id, collected_games=round_id*1000,
            win_rate=rate, ci95_low=float(interval[0]), ci95_high=float(interval[1]),
            attempted_games=attempts, errors=summary.get('errors'), cutoffs=summary.get('cutoffs'),
            checkpoint=evaluation.get('checkpoint'), checkpoint_sha256=evaluation.get('checkpoint_sha256'),
            cohort_seed=evaluation.get('config', {}).get('cohort', {}).get('seed_base'), source=path.name))
    monitors.sort(key=lambda row: row['round'])
    if len({row['round'] for row in monitors}) != len(monitors):
        raise ValueError('Duplicate monitoring checkpoint')
    seeds = {row['cohort_seed'] for row in monitors if row['cohort_seed'] is not None}
    if len(seeds) > 1:
        raise ValueError('Monitoring checkpoints do not share their declared cohort')
    critics = []
    games_per_round = report.get('config', {}).get('games_per_round', 1000)
    for entry in sorted(report.get('rounds', []), key=lambda row:row['round']):
        point = dict(round=entry['round'], collected_games=entry['round']*games_per_round)
        for partition in ('training', 'validation'):
            metric = entry.get('generalization', {}).get(partition, {})
            after = metric.get('value_after', {})
            before = metric.get('value_before', {})
            interval = (metric.get('value_mse') or {}).get('game_bootstrap_95_interval')
            point.update({f'{partition}_mse':_number(after.get('mse')),
                f'{partition}_mse_ci95_low':_number(interval[0]) if interval else None,
                f'{partition}_mse_ci95_high':_number(interval[1]) if interval else None,
                f'{partition}_explained_variance':_number(after.get('explained_variance')),
                f'{partition}_before_mse':_number(before.get('mse')),
                f'{partition}_terminal_games':metric.get('terminal_games')})
        critics.append(point)
    primary = None
    if study.get('status') == 'complete':
        metric = study.get('paired_comparisons', {}).get('primary-40k-minus-4k')
        if metric is not None:
            primary = dict(comparison='40k minus 4k',
                win_rate_change=metric['recorded_attempt_win_rate_change'],
                ci95=metric['recorded_attempt_win_rate_change_bootstrap_95_interval'])
    return dict(schema_version=1, study_status=study.get('status', 'unknown'),
        monitoring=monitors, critic=critics, primary_independent_confirmation=primary,
        limitations=[
            'One training seed; game intervals do not measure training-seed variability.',
            'Win rates measure this mono-red matchup against the fixed FoundationsAggro opponent.',
            'Monitoring checkpoints share one cohort; repeated looks are descriptive.',
            'Each round holds out whole games from both optimizers; validation games change each round.',
            'Critic curves replay current recurrent states after the update; they are not the frozen-memory fitting loss.',
            'The horizontal axis counts all collected games, including the whole-game holdout.'])


def _csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_study(study_root, output):
    data = collect_points(study_root)
    if not data['monitoring'] and not data['critic']:
        raise ValueError('No completed monitoring or critic measurements to plot')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep plotting configuration out of the project and user configuration.
    with _plot_configuration():
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter, FuncFormatter
        import numpy as np
        colors = dict(training='#b56b24', validation='#2167a8')
        with plt.rc_context({'font.family':'DejaVu Sans', 'font.size':10,
                             'axes.spines.top':False, 'axes.spines.right':False}):
            fig = plt.figure(figsize=(11.5, 8.3))
            grid = fig.add_gridspec(2, 2, height_ratios=(1.05, 1), hspace=.43, wspace=.30)
            strength = fig.add_subplot(grid[0, :])
            mse, explained = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
            fig.subplots_adjust(left=.085, right=.97, top=.83, bottom=.19)
            fig.suptitle('40k pooled PPO study · fixed-opponent learning', x=.085, y=.97,
                         ha='left', fontsize=18, fontweight='bold')
            fig.text(.085, .915, 'Laptop + Raytracer collection · GPU learner · one declared training seed',
                     fontsize=11, color='#526175')
            fig.text(.085, .875, 'Saved snapshot status: '+data['study_status'], color='#526175', fontsize=10)
            points = data['monitoring']
            if points:
                x = np.asarray([p['collected_games'] for p in points])
                y = np.asarray([p['win_rate'] for p in points])
                strength.errorbar(x, y, yerr=[y-np.asarray([p['ci95_low'] for p in points]),
                    np.asarray([p['ci95_high'] for p in points])-y], color='#2167a8',
                    marker='o', markersize=5, linewidth=2, capsize=4)
            strength.set_title('Sampled-policy win rate · shared monitoring cohort', loc='left',
                               fontsize=11, fontweight='bold')
            strength.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
            strength.set_xticks([0, 4000, 10000, 20000, 40000])
            strength.set_xlim(-800, 40800)
            strength.set_ylabel('Wins / all attempts')
            strength.set_xlabel('Collected games (includes the 20% holdout)')
            primary = data['primary_independent_confirmation']
            if primary:
                effect, (low, high) = primary['win_rate_change'], primary['ci95']
                text = f'Independent confirmation · 40k − 4k: {100*effect:+.2f} percentage points  [95% CI {100*low:+.2f}, {100*high:+.2f}]'
            else:
                text = 'Independent final confirmation: pending completion of the preregistered finite study'
            strength.text(0, -.33, text, transform=strength.transAxes, fontsize=9, color='#526175')
            rows = data['critic']
            x = [row['collected_games'] for row in rows]
            for partition, color in colors.items():
                def series(key):
                    return np.asarray([np.nan if row[key] is None else row[key] for row in rows])
                mse.plot(x, series(partition+'_mse'), '-o', color=color, markersize=3,
                         label='Train games' if partition == 'training' else 'Held-out games')
                if rows:
                    mse.fill_between(x, series(partition+'_mse_ci95_low'),
                                     series(partition+'_mse_ci95_high'), color=color, alpha=.12)
                explained.plot(x, series(partition+'_explained_variance'), '-o', color=color,
                               markersize=3, label='Train games' if partition == 'training' else 'Held-out games')
            mse.set_title('Current-memory value MSE · lower is better', loc='left', fontsize=10.5, fontweight='bold')
            explained.set_title('Value explained variance · higher is better', loc='left', fontsize=10.5, fontweight='bold')
            explained.axhline(0, color='#7f8b99', linestyle='--', linewidth=1)
            for axis in (strength, mse, explained):
                axis.grid(axis='y', color='#e4e9ef')
                axis.set_axisbelow(True)
                axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _:f'{value/1000:g}k' if value else '0'))
            for axis in (mse, explained):
                axis.set_xlabel('Collected games')
                axis.legend(frameon=False, fontsize=8)
            fig.text(.085, .045,
                'Bars: marginal 95% Wilson intervals; MSE bands: 95% whole-game bootstrap. Checkpoints share monitoring seeds.\n'
                'Fresh whole-game holdout each round; current recurrent states replayed after fitting. These are changing validation sets.\n'
                'Intervals are conditional on one training seed and this fixed opponent/deck matchup; they do not establish general playing strength.',
                fontsize=8.5, color='#526175', linespacing=1.5)
            files = []
            for extension in ('.png', '.svg'):
                path = output.with_suffix(extension)
                fig.savefig(path, dpi=160, facecolor='white')
                files.append(str(path))
            plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    _csv(output.with_name(output.name+'-monitoring').with_suffix('.csv'), data['monitoring'])
    _csv(output.with_name(output.name+'-critic').with_suffix('.csv'), data['critic'])
    return dict(files=files, data=str(output.with_suffix('.json')))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Figure path prefix')
    args = parser.parse_args(argv)
    print(json.dumps(plot_study(args.study_root, args.output)))


if __name__ == '__main__':
    main()
