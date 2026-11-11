"""Error analysis (E6) on the predictions already produced by E2/E3/E3b.

No model is retrained here: the script reads `predictions.npz` from the run
directories, aligns it with the official test targets and writes tables that
answer three questions.

1. Where does the error live (bias, tail, dependence on the target value)?
2. Are the mistakes the same across methods, or does TabM fix different rows?
3. For the binary task, is the probability output calibrated?

Example:
    python paper/scripts/analyze_errors.py --seeds 0 1 2
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METHOD_DIRECTORIES = {
    'mlp': 'E2-baseline/{method}/{dataset}/{seed}',
    'mlp-piecewiselinear': 'E2-baseline/{method}/{dataset}/{seed}',
    'tabm': 'E3/{method}/{dataset}/{seed}',
    'tabm-packed': 'E3/{method}/{dataset}/{seed}',
    'tabm-piecewiselinear': 'E3b/{dataset}/{seed}',
}
REFERENCE_METHOD = 'tabm'
DECILES = 10
CALIBRATION_BINS = 10


def run_directory(paper_dir: Path, method: str, dataset: str, seed: int) -> Path:
    """Return the output directory of one run."""
    template = METHOD_DIRECTORIES[method]
    return (
        paper_dir
        / 'exp'
        / 'reproduce'
        / template.format(method=method, dataset=dataset, seed=seed)
    )


def load_predictions(directory: Path) -> np.ndarray:
    """Return the test-split predictions of one run."""
    path = directory / 'predictions.npz'
    if not path.exists():
        raise FileNotFoundError(f'Missing predictions: {path}')
    return np.load(path)['test']


def load_task_type(paper_dir: Path, dataset: str) -> str:
    info = json.loads((paper_dir / 'data' / dataset / 'info.json').read_text('utf-8'))
    return info['task_type']


def load_targets(paper_dir: Path, dataset: str) -> np.ndarray:
    return np.load(paper_dir / 'data' / dataset / 'Y_test.npy')


def regression_summary(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Return bias/tail statistics of the residuals (prediction - target)."""
    residual = prediction - target
    absolute = np.abs(residual)
    return {
        'rmse': float(np.sqrt(np.mean(residual**2))),
        'mae': float(np.mean(absolute)),
        'bias': float(np.mean(residual)),
        'residual_std': float(np.std(residual)),
        'q05': float(np.quantile(residual, 0.05)),
        'q95': float(np.quantile(residual, 0.95)),
        'worst_1pct_mae': float(
            np.mean(np.sort(absolute)[-max(1, len(absolute) // 100) :])
        ),
        'tail_share': float(np.mean(absolute > 2 * np.std(residual))),
    }


def error_by_target_decile(
    prediction: np.ndarray, target: np.ndarray
) -> list[dict[str, float]]:
    """Group test rows by target rank and report the mean absolute error.

    The split is rank-based (equal counts) rather than quantile-edge based,
    because `house` contains many tied targets: duplicated edges would collapse
    the low bins into a handful of rows and make the first decile meaningless.
    """
    chunks = np.array_split(np.argsort(target, kind='stable'), DECILES)
    rows = []
    for bucket, index in enumerate(chunks, start=1):
        rows.append(
            {
                'decile': bucket,
                'n': int(len(index)),
                'target_mean': float(np.mean(target[index])),
                'mae': float(np.mean(np.abs(prediction[index] - target[index]))),
            }
        )
    return rows


def binary_summary(
    prediction: np.ndarray, target: np.ndarray
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Return confusion counts, derived scores and a calibration table."""
    probabilities = prediction.astype(np.float64)
    if probabilities.min() < 0.0 or probabilities.max() > 1.0:
        probabilities = 1.0 / (1.0 + np.exp(-probabilities))
    predicted = (probabilities >= 0.5).astype(int)
    positive = target.astype(int) == 1

    true_positive = int(np.sum(predicted[positive] == 1))
    false_negative = int(np.sum(predicted[positive] == 0))
    false_positive = int(np.sum(predicted[~positive] == 1))
    true_negative = int(np.sum(predicted[~positive] == 0))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    summary = {
        'tn': true_negative,
        'fp': false_positive,
        'fn': false_negative,
        'tp': true_positive,
        'accuracy': (true_positive + true_negative) / len(target),
        'precision': precision,
        'recall': recall,
        'f1': 2 * precision * recall / max(1e-12, precision + recall),
        'brier': float(np.mean((probabilities - target.astype(np.float64)) ** 2)),
    }

    edges = np.linspace(0.0, 1.0, CALIBRATION_BINS + 1)
    bucket = np.clip(np.digitize(probabilities, edges) - 1, 0, CALIBRATION_BINS - 1)
    rows = []
    for index in range(CALIBRATION_BINS):
        mask = bucket == index
        if not mask.any():
            continue
        rows.append(
            {
                'bin': index + 1,
                'n': int(mask.sum()),
                'mean_probability': float(np.mean(probabilities[mask])),
                'observed_rate': float(np.mean(target[mask])),
            }
        )
    expected_calibration_error = sum(
        row['n'] / len(target) * abs(row['mean_probability'] - row['observed_rate'])
        for row in rows
    )
    summary['ece'] = expected_calibration_error
    return summary, rows


def pairwise(reference: np.ndarray, other: np.ndarray, target: np.ndarray) -> dict:
    """Compare squared errors of two methods on the same test rows."""
    reference_error = (reference - target) ** 2
    other_error = (other - target) ** 2
    return {
        'wins': int(np.sum(reference_error < other_error)),
        'losses': int(np.sum(reference_error > other_error)),
        'ties': int(np.sum(reference_error == other_error)),
        'mean_squared_error_ratio': float(
            np.mean(other_error) / max(1e-12, np.mean(reference_error))
        ),
        'error_correlation': float(np.corrcoef(reference_error, other_error)[0, 1]),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows whose columns may differ (regression vs classification)."""
    fieldnames: list[str] = []
    for row in rows:
        fieldnames.extend(key for key in row if key not in fieldnames)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, restval='')
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    """Average numeric fields over seeds and keep the standard deviation."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for group_key, group in groups.items():
        merged = dict(zip(keys, group_key))
        numeric = [
            key
            for key, value in group[0].items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        for key in numeric:
            values = [row[key] for row in group]
            merged[key] = float(np.mean(values))
            merged[f'{key}_std'] = float(np.std(values))
        result.append(merged)
    return sorted(result, key=lambda row: tuple(str(row[key]) for key in keys))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--paper-dir',
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help='Path to paper/ (contains exp/ and data/)',
    )
    parser.add_argument(
        '--datasets', nargs='+', default=['california', 'churn', 'house']
    )
    parser.add_argument('--methods', nargs='+', default=list(METHOD_DIRECTORIES))
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.paper_dir / 'results' / 'E6'
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    decile_rows = []
    calibration_rows = []
    pair_rows = []

    for dataset in args.datasets:
        task_type = load_task_type(args.paper_dir, dataset)
        target = load_targets(args.paper_dir, dataset)
        is_regression = task_type == 'regression'

        for seed in args.seeds:
            predictions = {}
            for method in args.methods:
                directory = run_directory(args.paper_dir, method, dataset, seed)
                if not (directory / 'predictions.npz').exists():
                    continue
                predictions[method] = load_predictions(directory)

            for method, prediction in predictions.items():
                if len(prediction) != len(target):
                    raise ValueError(
                        f'{dataset}/{method}/{seed}: {len(prediction)} predictions '
                        f'but {len(target)} targets'
                    )
                if is_regression:
                    summary = regression_summary(prediction, target)
                else:
                    summary, calibration = binary_summary(prediction, target)
                    for row in calibration:
                        row.update({'dataset': dataset, 'method': method, 'seed': seed})
                        calibration_rows.append(row)
                row = {'dataset': dataset, 'method': method, 'seed': seed}
                row.update(summary)
                summary_rows.append(row)

            if is_regression:
                for method, prediction in predictions.items():
                    for row in error_by_target_decile(prediction, target):
                        row.update({'dataset': dataset, 'method': method, 'seed': seed})
                        decile_rows.append(row)

            if REFERENCE_METHOD in predictions:
                for method, prediction in predictions.items():
                    if method == REFERENCE_METHOD:
                        continue
                    row = pairwise(predictions[REFERENCE_METHOD], prediction, target)
                    row.update({'dataset': dataset, 'method': method, 'seed': seed})
                    pair_rows.append(row)

    write_csv(output_dir / 'summary_by_seed.csv', summary_rows)
    write_csv(
        output_dir / 'summary.csv', aggregate(summary_rows, ('dataset', 'method'))
    )
    if decile_rows:
        write_csv(
            output_dir / 'error_by_decile.csv',
            aggregate(decile_rows, ('dataset', 'method', 'decile')),
        )
    if calibration_rows:
        write_csv(
            output_dir / 'calibration.csv',
            aggregate(calibration_rows, ('dataset', 'method', 'bin')),
        )
    if pair_rows:
        write_csv(
            output_dir / 'pairwise_vs_tabm.csv',
            aggregate(pair_rows, ('dataset', 'method')),
        )

    print(f'{"dataset":12}{"method":24}{"metric":>10}{"value":>14}{"std":>12}')
    for row in aggregate(summary_rows, ('dataset', 'method')):
        for metric in ('rmse', 'accuracy', 'ece', 'worst_1pct_mae', 'bias'):
            if metric in row:
                print(
                    f'{row["dataset"]:12}{row["method"]:24}{metric:>10}'
                    f'{row[metric]:>14.5f}{row[f"{metric}_std"]:>12.5f}'
                )
    print(f'\nWrote {output_dir}')


if __name__ == '__main__':
    main()
