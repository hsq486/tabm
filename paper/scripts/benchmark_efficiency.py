"""Measure parameter count, training step cost and inference throughput.

The wall-clock numbers recorded in `runs.csv` mix two effects: the cost of one
optimizer step and the number of steps that early stopping happened to allow.
This script removes the second effect by timing a *fixed* number of steps for
every architecture on identical synthetic data, so the comparison isolates the
architectural cost.

The measurement uses `tabm_reference.Model`, i.e. the same ensemble layers as the
training code, and a california-shaped workload (8 features, 16,512 rows).

Example:
    python paper/scripts/benchmark_efficiency.py
"""

import argparse
import csv
import importlib.util
import statistics
import time
import types
import warnings
from pathlib import Path

import torch

ARCHITECTURES = (
    ('plain', None),
    ('tabm-mini', 8),
    ('tabm', 8),
    ('tabm-packed', 8),
)


def load_reference(repo_root: Path) -> types.ModuleType:
    """Load `tabm_reference.py` from the repository root by explicit path."""
    path = repo_root / 'tabm_reference.py'
    if not path.exists():
        raise FileNotFoundError(f'Reference implementation not found: {path}')
    spec = importlib.util.spec_from_file_location('tabm_reference', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        spec.loader.exec_module(module)
    return module


def build_model(
    reference: types.ModuleType, arch_type: str, k: int | None, n_features: int
) -> torch.nn.Module:
    """Create one model with the frozen 2x256 backbone."""
    return reference.Model(
        n_num_features=n_features,
        cat_cardinalities=[],
        n_classes=None,
        backbone={'type': 'MLP', 'n_blocks': 2, 'd_block': 256, 'dropout': 0.0},
        bins=None,
        arch_type=arch_type,
        k=k,
    )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@torch.no_grad()
def measure_inference(
    model: torch.nn.Module, batch_size: int, iterations: int, n_features: int
) -> float:
    """Return the number of samples processed per second in eval mode."""
    model.eval()
    x = torch.randn(batch_size, n_features)
    for _ in range(3):
        model(x_num=x)
    start = time.perf_counter()
    for _ in range(iterations):
        model(x_num=x)
    elapsed = time.perf_counter() - start
    return batch_size * iterations / elapsed


def measure_training_step(
    model: torch.nn.Module,
    batch_size: int,
    warmup: int,
    steps: int,
    n_features: int,
) -> float:
    """Return the median duration of one forward+backward+update step in ms."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_function = torch.nn.functional.mse_loss
    x = torch.randn(batch_size, n_features)
    y = torch.randn(batch_size, 1)

    durations = []
    model.train()
    for index in range(warmup + steps):
        start = time.perf_counter()
        optimizer.zero_grad()
        prediction = model(x_num=x).mean(dim=1)
        loss = loss_function(prediction, y)
        loss.backward()
        optimizer.step()
        if index >= warmup:
            durations.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(durations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--repo-root',
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help='Repository root containing tabm_reference.py',
    )
    parser.add_argument('--n-features', type=int, default=8)
    parser.add_argument('--n-train', type=int, default=16512)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--infer-batch', type=int, default=4096)
    parser.add_argument('--infer-iters', type=int, default=20)
    parser.add_argument('--threads', type=int, default=14)
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='CSV path (default: results/E5/benchmark.csv)',
    )
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    reference = load_reference(args.repo_root)

    output = (
        args.output or args.repo_root / 'paper' / 'results' / 'E5' / 'benchmark.csv'
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = args.n_train // args.batch_size
    rows = []
    for arch_type, k in ARCHITECTURES:
        model = build_model(reference, arch_type, k, args.n_features)
        step_ms = measure_training_step(
            model, args.batch_size, args.warmup, args.steps, args.n_features
        )
        row = {
            'arch_type': arch_type,
            'k': k if k is not None else 1,
            'n_parameters': count_parameters(model),
            'step_ms': round(step_ms, 3),
            'epoch_seconds': round(step_ms * steps_per_epoch / 1000.0, 2),
            'inference_samples_per_sec': round(
                measure_inference(
                    model, args.infer_batch, args.infer_iters, args.n_features
                ),
                1,
            ),
            'threads': args.threads,
            'batch_size': args.batch_size,
        }
        rows.append(row)
        print(
            f'{arch_type:12} params={row["n_parameters"]:>9}'
            f'  step={row["step_ms"]:>7.2f} ms'
            f'  epoch={row["epoch_seconds"]:>7.2f} s'
            f'  inference={row["inference_samples_per_sec"]:>9.1f} samples/s'
        )

    with output.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f'\nWrote {output}')


if __name__ == '__main__':
    main()
