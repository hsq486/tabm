"""Generate the E6 figures from the tables written by `analyze_errors.py`.

The figures are a pure presentation layer: every number comes from the CSV
tables in `results/E6` (and `results/comparison.csv` for parameter counts), so
the plots can be regenerated without rerunning any experiment.

Example:
    python paper/scripts/plot_e6.py
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

DISPLAY_NAMES = {
    'mlp': 'MLP',
    'mlp-piecewiselinear': 'MLP†',
    'tabm': 'TabM',
    'tabm-packed': 'TabM-packed',
    'tabm-piecewiselinear': 'TabM + emb.',
}
COLORS = {
    'mlp': '#7f8c8d',
    'mlp-piecewiselinear': '#e67e22',
    'tabm': '#2980b9',
    'tabm-packed': '#8e44ad',
    'tabm-piecewiselinear': '#27ae60',
}
ORDER = list(DISPLAY_NAMES)


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding='utf-8') as file:
        return list(csv.DictReader(file))


def method_order(methods: set[str]) -> list[str]:
    known = [method for method in ORDER if method in methods]
    return known + sorted(methods - set(known))


def plot_error_by_decile(rows: list[dict], output: Path) -> None:
    datasets = ['california', 'house']
    figure, axes = plt.subplots(1, len(datasets), figsize=(10.0, 4.0))
    for axis, dataset in zip(axes, datasets):
        subset = [row for row in rows if row['dataset'] == dataset]
        for method in method_order({row['method'] for row in subset}):
            points = sorted(
                (row for row in subset if row['method'] == method),
                key=lambda row: int(float(row['decile'])),
            )
            axis.plot(
                [int(float(row['decile'])) for row in points],
                [float(row['mae']) for row in points],
                marker='o',
                markersize=4,
                linewidth=1.6,
                label=DISPLAY_NAMES.get(method, method),
                color=COLORS.get(method),
            )
        axis.set_title(f'{dataset}: MAE by target decile')
        axis.set_xlabel('Target decile (1 = lowest)')
        axis.set_ylabel('Mean absolute error')
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
        if dataset == 'house':
            axis.set_yscale('log')
            axis.set_ylabel('Mean absolute error (log scale)')
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def plot_calibration(rows: list[dict], summary: list[dict], output: Path) -> None:
    subset = [row for row in rows if row['dataset'] == 'churn']
    errors = {
        row['method']: float(row['ece'])
        for row in summary
        if row['dataset'] == 'churn' and row['ece']
    }
    figure, axis = plt.subplots(figsize=(5.2, 4.6))
    axis.plot(
        [0, 1], [0, 1], linestyle='--', color='black', linewidth=1, label='Perfect'
    )
    for method in method_order({row['method'] for row in subset}):
        points = sorted(
            (row for row in subset if row['method'] == method),
            key=lambda row: int(float(row['bin'])),
        )
        label = DISPLAY_NAMES.get(method, method)
        if method in errors:
            label = f'{label} (ECE {errors[method]:.3f})'
        axis.plot(
            [float(row['mean_probability']) for row in points],
            [float(row['observed_rate']) for row in points],
            marker='o',
            markersize=4,
            linewidth=1.6,
            label=label,
            color=COLORS.get(method),
        )
    axis.set_title('churn: reliability diagram')
    axis.set_xlabel('Mean predicted probability')
    axis.set_ylabel('Observed positive rate')
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, loc='upper left')
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def plot_parameters_vs_error(rows: list[dict], output: Path) -> None:
    errors = defaultdict(list)
    parameters = {}
    for row in rows:
        if row['dataset'] != 'california':
            continue
        errors[row['method']].append(-float(row['score']))
        parameters[row['method']] = int(row['n_parameters'])
    figure, axis = plt.subplots(figsize=(5.2, 4.6))
    largest = max(parameters.values())
    for method, values in errors.items():
        mean_error = sum(values) / len(values)
        axis.scatter(
            parameters[method],
            mean_error,
            s=70,
            color=COLORS.get(method),
            zorder=3,
            label=DISPLAY_NAMES.get(method, method),
        )
        on_the_right = parameters[method] > 0.6 * largest
        axis.annotate(
            DISPLAY_NAMES.get(method, method),
            (parameters[method], mean_error),
            textcoords='offset points',
            xytext=(-9, 8) if on_the_right else (7, 8),
            ha='right' if on_the_right else 'left',
            fontsize=8,
        )
    axis.set_xscale('log')
    axis.set_title('california: error vs parameter count')
    axis.set_xlabel('Trainable parameters (log scale)')
    axis.set_ylabel('Test RMSE (mean of 3 seeds)')
    axis.margins(x=0.22, y=0.14)
    axis.grid(alpha=0.3, which='both')
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=Path(__file__).resolve().parents[1] / 'results',
    )
    parser.add_argument('--dpi', type=int, default=200)
    args = parser.parse_args()
    plt.rcParams['savefig.dpi'] = args.dpi

    e6 = args.results_dir / 'E6'
    output = e6 / 'figures'
    output.mkdir(parents=True, exist_ok=True)

    plot_error_by_decile(
        read_rows(e6 / 'error_by_decile.csv'), output / 'error_by_decile.png'
    )
    plot_calibration(
        read_rows(e6 / 'calibration.csv'),
        read_rows(e6 / 'summary.csv'),
        output / 'calibration_churn.png',
    )
    plot_parameters_vs_error(
        read_rows(args.results_dir / 'comparison.csv'), output / 'params_vs_error.png'
    )
    print(f'Wrote 3 figures to {output}')


if __name__ == '__main__':
    main()
