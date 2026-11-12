"""Summarize the five-seed main comparison and the E8 label-noise runs.

The script only reads report.json files that training already wrote. It does
not train. Regression scores in those reports are negative RMSE; this script
converts them back to RMSE before averaging.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parents[1]
DATASETS = ('california', 'churn', 'house')
SEEDS = (0, 1, 2, 3, 4)
NOISE_LEVELS = (0.0, 0.1, 0.2)
BOOTSTRAP_DRAWS = 10_000


def read_report(path: Path) -> dict[str, float | int]:
    """Return the test metric and the bookkeeping fields used in the tables."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    test = payload['metrics']['test']
    row: dict[str, float | int] = {
        'score': float(test['score']),
        'n_parameters': int(payload['n_parameters']),
        'best_step': int(payload['best_step']),
    }
    if 'rmse' in test:
        row['value'] = float(test['rmse'])
        row['direction'] = 1  # lower is better; stored only as a marker below
    else:
        row['value'] = float(test['accuracy'])
    return row


def clean_report(method: str, dataset: str, seed: int) -> Path:
    """Path of an already finished clean run."""
    if method == 'mlp':
        return (
            PAPER
            / 'exp'
            / 'reproduce'
            / 'E2-baseline'
            / 'mlp'
            / dataset
            / str(seed)
            / 'report.json'
        )
    if method == 'tabm':
        return (
            PAPER
            / 'exp'
            / 'reproduce'
            / 'E3'
            / 'tabm'
            / dataset
            / str(seed)
            / 'report.json'
        )
    raise ValueError(method)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a CSV with a stable header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(values: list[float]) -> tuple[float, float]:
    """Percentile interval for the mean of a paired difference."""
    draw = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(0)
    index = rng.integers(0, draw.size, size=(BOOTSTRAP_DRAWS, draw.size))
    means = draw[index].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def better(dataset: str, mlp_value: float, tabm_value: float) -> float:
    """Positive when TabM beats MLP on that dataset's primary metric."""
    if dataset == 'churn':
        return tabm_value - mlp_value
    return mlp_value - tabm_value


def summarize_main() -> None:
    """Mean, sample standard deviation, paired difference and bootstrap CI."""
    by_seed: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    paired: list[dict[str, object]] = []

    for dataset in DATASETS:
        mlp_values: list[float] = []
        tabm_values: list[float] = []
        gaps: list[float] = []
        for seed in SEEDS:
            mlp = read_report(clean_report('mlp', dataset, seed))
            tabm = read_report(clean_report('tabm', dataset, seed))
            gap = better(dataset, float(mlp['value']), float(tabm['value']))
            mlp_values.append(float(mlp['value']))
            tabm_values.append(float(tabm['value']))
            gaps.append(gap)
            by_seed.append(
                {
                    'dataset': dataset,
                    'seed': seed,
                    'mlp': mlp['value'],
                    'tabm': tabm['value'],
                    'tabm_minus_mlp_advantage': gap,
                    'tabm_parameters': tabm['n_parameters'],
                    'mlp_parameters': mlp['n_parameters'],
                }
            )
        low, high = bootstrap_ci(gaps)
        metric = 'accuracy' if dataset == 'churn' else 'rmse'
        summary.append(
            {
                'dataset': dataset,
                'metric': metric,
                'n_seeds': len(SEEDS),
                'mlp_mean': statistics.fmean(mlp_values),
                'mlp_std': statistics.stdev(mlp_values),
                'tabm_mean': statistics.fmean(tabm_values),
                'tabm_std': statistics.stdev(tabm_values),
                'advantage_mean': statistics.fmean(gaps),
                'advantage_std': statistics.stdev(gaps),
                'advantage_ci95_low': low,
                'advantage_ci95_high': high,
                'tabm_wins': sum(gap > 0 for gap in gaps),
                'ties': sum(gap == 0 for gap in gaps),
            }
        )
        paired.append(
            {
                'dataset': dataset,
                'metric': metric,
                **{f'seed_{seed}': gap for seed, gap in zip(SEEDS, gaps, strict=True)},
            }
        )

    out = PAPER / 'results' / 'main5'
    write_rows(out / 'by_seed.csv', by_seed)
    write_rows(out / 'summary.csv', summary)
    write_rows(out / 'paired.csv', paired)


def summarize_e8() -> None:
    """Attach the clean runs as the 0% row and summarize 0/10/20%."""
    by_seed: list[dict[str, object]] = []
    for level in NOISE_LEVELS:
        for method in ('mlp', 'tabm'):
            for seed in (0, 1, 2):
                if level == 0.0:
                    path = clean_report(method, 'california', seed)
                else:
                    path = (
                        PAPER
                        / 'exp'
                        / 'reproduce'
                        / 'E8'
                        / method
                        / 'california'
                        / f'noise={level}'
                        / str(seed)
                        / 'report.json'
                    )
                row = read_report(path)
                by_seed.append(
                    {
                        'dataset': 'california',
                        'method': method,
                        'noise': level,
                        'seed': seed,
                        'rmse': row['value'],
                        'n_parameters': row['n_parameters'],
                        'best_step': row['best_step'],
                    }
                )

    summary: list[dict[str, object]] = []
    for level in NOISE_LEVELS:
        grouped = {
            method: [
                float(row['rmse'])
                for row in by_seed
                if row['noise'] == level and row['method'] == method
            ]
            for method in ('mlp', 'tabm')
        }
        gaps = [
            better('california', mlp, tabm)
            for mlp, tabm in zip(grouped['mlp'], grouped['tabm'], strict=True)
        ]
        low, high = bootstrap_ci(gaps)
        summary.append(
            {
                'dataset': 'california',
                'noise': level,
                'mlp_mean': statistics.fmean(grouped['mlp']),
                'mlp_std': statistics.stdev(grouped['mlp']),
                'tabm_mean': statistics.fmean(grouped['tabm']),
                'tabm_std': statistics.stdev(grouped['tabm']),
                'advantage_mean': statistics.fmean(gaps),
                'advantage_ci95_low': low,
                'advantage_ci95_high': high,
                'tabm_wins': sum(gap > 0 for gap in gaps),
            }
        )

    out = PAPER / 'results' / 'E8'
    write_rows(out / 'by_seed.csv', by_seed)
    write_rows(out / 'summary.csv', summary)


def main() -> None:
    """Write both summaries. Missing reports raise FileNotFoundError."""
    summarize_main()
    summarize_e8()


if __name__ == '__main__':
    main()
