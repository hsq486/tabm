"""E9: locate the reproduced effect size inside the official result distribution.

Reads the `report.json` files shipped with the official repository (one per
dataset, method and seed) and answers: "how large is TabM's gain over MLP on the
datasets the authors ran, and where do our three reproductions sit?"

Effect size is defined before looking at our numbers:

* regression: relative reduction of the test error, `(mlp - tabm) / mlp`;
* classification: absolute difference of test accuracy, `tabm - mlp`.

Both are "higher is better for TabM". The official effect per dataset is the
mean over the seeds present in the repository.

Example:
    python paper/scripts/locate_effect_size.py
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

OUR_METHODS = {'mlp': 'mlp', 'tabm': 'tabm'}
OUR_LABELS = {
    'california': 'california (ours)',
    'churn': 'churn (ours)',
    'house': 'house (ours)',
}
OFFICIAL_COLOR = '#95a5a6'
OUR_COLOR = '#e74c3c'


def read_report(path: Path) -> dict:
    with path.open(encoding='utf-8') as file:
        return json.load(file)


def is_regression(report: dict) -> bool:
    return 'rmse' in report['metrics']['test']


def official_effects(paper_dir: Path) -> list[dict]:
    """Collect one effect size per dataset from the official reports."""
    root = paper_dir / 'exp'
    datasets = sorted(
        path.name
        for path in (root / 'tabm').iterdir()
        if (path / '0-evaluation' / '0' / 'report.json').exists()
    )
    rows = []
    for dataset in datasets:
        mlp_path = root / 'mlp' / dataset / '0-evaluation' / '0' / 'report.json'
        tabm_path = root / 'tabm' / dataset / '0-evaluation' / '0' / 'report.json'
        if not mlp_path.exists():
            continue
        mlp_report = read_report(mlp_path)
        tabm_report = read_report(tabm_path)
        regression = is_regression(tabm_report)
        mlp_score = mlp_report['metrics']['test']['score']
        tabm_score = tabm_report['metrics']['test']['score']
        if regression:
            effect = (tabm_score - mlp_score) / abs(mlp_score)
            unit = 'relative_error_reduction'
        else:
            effect = tabm_score - mlp_score
            unit = 'accuracy_difference'
        rows.append(
            {
                'dataset': dataset,
                'task': 'regression' if regression else 'classification',
                'unit': unit,
                'mlp_score': mlp_score,
                'tabm_score': tabm_score,
                'effect': effect,
                'tabm_n_parameters': tabm_report['n_parameters'],
                'mlp_n_parameters': mlp_report['n_parameters'],
                'tabm_k': tabm_report['config']['model']['k'],
            }
        )
    return rows


def our_effects(results_dir: Path) -> list[dict]:
    """Average the three seeds of our own runs and apply the same formula."""
    path = results_dir / 'comparison.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing comparison table: {path}')
    with path.open(encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if row['method'] not in OUR_METHODS:
            continue
        key = (row['dataset'], row['method'])
        grouped.setdefault(key, []).append(float(row['score']))
    result = []
    for dataset in sorted({dataset for dataset, _ in grouped}):
        if (dataset, 'mlp') not in grouped or (dataset, 'tabm') not in grouped:
            continue
        mlp_score = statistics.mean(grouped[(dataset, 'mlp')])
        tabm_score = statistics.mean(grouped[(dataset, 'tabm')])
        regression = dataset != 'churn'
        effect = (
            (tabm_score - mlp_score) / abs(mlp_score)
            if regression
            else tabm_score - mlp_score
        )
        result.append(
            {
                'dataset': dataset,
                'task': 'regression' if regression else 'classification',
                'mlp_score': mlp_score,
                'tabm_score': tabm_score,
                'effect': effect,
            }
        )
    return result


def percentile(value: float, distribution: list[float]) -> float:
    """Share of official datasets whose effect is not larger than `value`."""
    if not distribution:
        raise ValueError('Empty distribution')
    return 100.0 * sum(1 for item in distribution if item <= value) / len(distribution)


def plot(rows: list[dict], ours: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for axis, task in zip(axes, ('regression', 'classification')):
        official = sorted(
            (row for row in rows if row['task'] == task),
            key=lambda row: row['effect'],
        )
        values = [row['effect'] for row in official]
        axis.scatter(
            values,
            range(len(values)),
            color=OFFICIAL_COLOR,
            s=45,
            label='official (repository)',
            zorder=2,
        )
        for index, row in enumerate(official):
            axis.annotate(
                row['dataset'],
                (row['effect'], index),
                xytext=(4, -3),
                textcoords='offset points',
                fontsize=7,
                color='#555555',
            )
        ours_here = [row for row in ours if row['task'] == task]
        for row in ours_here:
            axis.scatter(
                [row['effect']],
                [-1.0],
                marker='v',
                s=70,
                color=OUR_COLOR,
                zorder=3,
                label='our reproduction' if row is ours_here[0] else None,
            )
            axis.annotate(
                OUR_LABELS.get(row['dataset'], row['dataset']),
                (row['effect'], -1.0),
                xytext=(5, 0),
                textcoords='offset points',
                fontsize=7,
                color=OUR_COLOR,
                va='center',
            )
        axis.axvline(0, color='black', linewidth=0.8)
        axis.set_title(f'{task}: TabM gain over MLP')
        axis.set_xlabel(
            'relative error reduction'
            if task == 'regression'
            else 'accuracy difference'
        )
        axis.set_yticks([])
        axis.set_ylim(-1.9, len(values) - 0.5)
        axis.grid(axis='x', alpha=0.3)
        axis.legend(fontsize=7, loc='lower right')
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--paper-dir',
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument('--dpi', type=int, default=200)
    args = parser.parse_args()
    plt.rcParams['savefig.dpi'] = args.dpi

    results_dir = args.paper_dir / 'results'
    output_dir = results_dir / 'E9'
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = official_effects(args.paper_dir)
    ours = our_effects(results_dir)
    by_task: dict[str, list[float]] = {}
    for row in rows:
        by_task.setdefault(row['task'], []).append(row['effect'])

    with (output_dir / 'official_effects.csv').open(
        'w', newline='', encoding='utf-8'
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    located = []
    for row in ours:
        distribution = by_task.get(row['task'], [])
        located.append(
            {
                **{
                    key: row[key]
                    for key in ('dataset', 'task', 'mlp_score', 'tabm_score', 'effect')
                },
                'official_n': len(distribution),
                'official_median': statistics.median(distribution),
                'percentile': percentile(row['effect'], distribution),
            }
        )
    with (output_dir / 'our_position.csv').open(
        'w', newline='', encoding='utf-8'
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(located[0].keys()))
        writer.writeheader()
        writer.writerows(located)

    plot(rows, ours, output_dir / 'effect_size_distribution.png')

    for task, values in by_task.items():
        print(
            f'{task:16} n={len(values)}  median={statistics.median(values):+.4f}'
            f'  min={min(values):+.4f}  max={max(values):+.4f}'
            f'  negative={sum(1 for value in values if value < 0)}'
        )
    for row in located:
        print(
            f'{row["dataset"]:12} effect={row["effect"]:+.4f}'
            f'  percentile={row["percentile"]:.0f}%'
            f'  (official median {row["official_median"]:+.4f})'
        )
    print(f'\nWrote {output_dir}')


if __name__ == '__main__':
    main()
