"""Run the extension matrix: seeds 3-4 for the main comparison, then E8.

Part A, five-seed main comparison
    TabM and MLP only, datasets california / churn / house, seeds 3 and 4.
    Same configs as E2 (official MLP) and E3 (reduced TabM). Threads stay at 14.

Part B, E8 label noise, pre-registered before looking at the new numbers
    Dataset california. Methods MLP and TabM. Seeds 0, 1, 2.
    Training labels only. Regression noise is Gaussian with a standard
    deviation equal to 10% or 20% of the clean training-label standard
    deviation, added after standardization on the clean training labels.
    Validation and test labels stay clean. The 0% row is the existing E2/E3
    run of the same config and seed; it is not trained again.
    Support requires TabM to keep a lower mean RMSE than MLP at 10%, with at
    least two of the three seeds pointing the same way. The 20% row is
    reported either way and does not have to preserve the gap.

Each finished run is recorded in a CSV ledger next to its result tables, and a
run is skipped when its DONE marker exists, so the script can be restarted
after an interruption without retraining anything.

Usage:
    python scripts/run_extensions.py            # run everything
    python scripts/run_extensions.py --dry-run   # build and check configs only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'bin' / 'model.py'
CONFIGS = ROOT / 'configs'

LEDGER_HEADER = (
    'timestamp,commit,method,dataset,seed,exit_code,duration_sec,status,report'
)
LEDGER_HEADER_NOISE = (
    'timestamp,commit,method,dataset,noise,seed,exit_code,duration_sec,status,report'
)

# Insert label_noise as a TOP-LEVEL key, immediately after `amp = true`.
# Appending it to the end of the file would put it inside the last `[...]`
# table (both sources end with [model.backbone]), where the top-level lookup in
# bin/model.py never sees it: the run would then either fail on an unexpected
# backbone kwarg or silently train on clean labels and report it as noise.
ANCHOR = 'amp = true'


def read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def set_seed(text: str, seed: int) -> str:
    out = []
    replaced = False
    for line in text.splitlines():
        if not replaced and line.strip().startswith('seed'):
            head, _, _ = line.partition('=')
            if head.strip() == 'seed':
                out.append(f'seed = {seed}')
                replaced = True
                continue
        out.append(line)
    if not replaced:
        raise RuntimeError('no top-level seed line to replace')
    return '\r\n'.join(out) + '\r\n'


def set_label_noise(text: str, level: str, source: Path) -> str:
    lines = text.splitlines()
    if any(line.strip().startswith('label_noise') for line in lines):
        raise RuntimeError(f'label_noise is already present in {source}')
    try:
        anchor = next(i for i, line in enumerate(lines) if line.strip() == ANCHOR)
    except StopIteration:
        raise RuntimeError(
            f'top-level {ANCHOR!r} anchor not found in {source}; refusing to '
            'guess where label_noise belongs'
        ) from None
    lines.insert(anchor + 1, f'label_noise = {level}')

    text = '\r\n'.join(lines) + '\r\n'
    key_pos = text.index(f'\nlabel_noise = {level}')
    table_pos = text.index('\n[')
    if key_pos > table_pos:
        raise RuntimeError(
            f'label_noise landed inside a table in {source}; it must stay a '
            'top-level key'
        )
    return text


def short_commit() -> str:
    try:
        out = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'
    return out.stdout.strip()


def append_ledger(
    ledger_dir: Path,
    header: str,
    row: list[str],
) -> None:
    ledger_dir.mkdir(parents=True, exist_ok=True)
    path = ledger_dir / 'runs.csv'
    if not path.exists():
        with path.open('w', encoding='utf-8', newline='') as f:
            f.write(header + '\r\n')
    with path.open('a', encoding='utf-8', newline='') as f:
        f.write(','.join(row) + '\r\n')


def run_one(
    *,
    python: str,
    ledger_dir: Path,
    header: str,
    run_dir: Path,
    config_text: str,
    method: str,
    dataset: str,
    seed: int,
    noise: str | None,
    commit: str,
    env: dict[str, str],
) -> bool:
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / f'{seed}.toml'
    log_path = run_dir / f'{seed}.log'
    done_path = run_dir / str(seed) / 'DONE'
    report_path = run_dir / str(seed) / 'report.json'

    config_path.write_text(config_text, encoding='utf-8', newline='')

    if done_path.exists():
        print(f'skip {method} / {dataset} / seed={seed} (already done)', flush=True)
        return True

    label = f'{method} / {dataset} / seed={seed}'
    if noise is not None:
        label += f' / noise={noise}'
    print(f'run {label}', flush=True)

    started = time.time()
    with log_path.open('wb') as log:
        proc = subprocess.run(
            [python, str(MODEL), str(config_path), '--force'],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            check=False,
        )
    duration = round(time.time() - started, 1)
    ok = proc.returncode == 0 and done_path.exists()
    status = 'ok' if ok else 'failed'

    relative = report_path.relative_to(ROOT.parent).as_posix()
    row = [
        time.strftime('%Y-%m-%d %H:%M:%S'),
        commit,
        method,
        dataset,
    ]
    if noise is not None:
        row.append(noise)
    row += [str(seed), str(proc.returncode), str(duration), status, relative]
    append_ledger(ledger_dir, header, row)

    print(f'{status} in {duration} s -> {relative}', flush=True)
    if not ok and log_path.stat().st_size:
        tail = log_path.read_text(encoding='utf-8', errors='replace').splitlines()
        print('  last lines of the log:', flush=True)
        for line in tail[-6:]:
            print(f'  | {line}', flush=True)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--python',
        default=r'D:\Anaconda\envs\tabm\python.exe',
        help='interpreter of the environment the experiments run in',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='only build and validate the configs, do not train',
    )
    parser.add_argument(
        '--part',
        choices=('all', 'a', 'b'),
        default='all',
        help='a: seeds 3-4 only, b: E8 only, all: both',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=0,
        help='train at most N pending runs (0 means no limit)',
    )
    args = parser.parse_args()

    if not Path(args.python).exists():
        raise SystemExit(f'python executable not found: {args.python}')

    env = dict(os.environ)
    env['OMP_NUM_THREADS'] = '14'
    env['MKL_NUM_THREADS'] = '14'
    commit = short_commit()

    main_cases = []
    for dataset in ('california', 'churn', 'house'):
        mlp_source = ROOT / 'exp' / 'mlp' / dataset / '0-evaluation' / '0.toml'
        tabm_source = CONFIGS / f'E3-tabm-{dataset}.toml'
        for seed in (3, 4):
            main_cases.append(
                (
                    'mlp',
                    dataset,
                    seed,
                    ROOT / 'results' / 'E2',
                    LEDGER_HEADER,
                    ROOT / 'exp' / 'reproduce' / 'E2-baseline' / 'mlp' / dataset,
                    set_seed(read_text(mlp_source), seed),
                )
            )
            main_cases.append(
                (
                    'tabm',
                    dataset,
                    seed,
                    ROOT / 'results' / 'E3',
                    LEDGER_HEADER,
                    ROOT / 'exp' / 'reproduce' / 'E3' / 'tabm' / dataset,
                    set_seed(read_text(tabm_source), seed),
                )
            )

    noise_cases = []
    mlp_noise_source = ROOT / 'exp' / 'mlp' / 'california' / '0-evaluation' / '0.toml'
    tabm_noise_source = CONFIGS / 'E3-tabm-california.toml'
    for level in ('0.1', '0.2'):
        for seed in (0, 1, 2):
            noise_cases.append(
                (
                    'mlp',
                    'california',
                    seed,
                    level,
                    ROOT / 'results' / 'E8',
                    LEDGER_HEADER_NOISE,
                    ROOT
                    / 'exp'
                    / 'reproduce'
                    / 'E8'
                    / 'mlp'
                    / 'california'
                    / f'noise={level}',
                    set_label_noise(
                        set_seed(read_text(mlp_noise_source), seed),
                        level,
                        mlp_noise_source,
                    ),
                )
            )
            noise_cases.append(
                (
                    'tabm',
                    'california',
                    seed,
                    level,
                    ROOT / 'results' / 'E8',
                    LEDGER_HEADER_NOISE,
                    ROOT
                    / 'exp'
                    / 'reproduce'
                    / 'E8'
                    / 'tabm'
                    / 'california'
                    / f'noise={level}',
                    set_label_noise(
                        set_seed(read_text(tabm_noise_source), seed),
                        level,
                        tabm_noise_source,
                    ),
                )
            )

    total = len(main_cases) + len(noise_cases)
    print(f'{total} runs queued, commit {commit}', flush=True)

    if args.dry_run:
        for case in main_cases:
            print(f'  ok {case[0]} / {case[1]} / seed={case[2]}', flush=True)
        print(f'  {len(noise_cases)} E8 configs validated', flush=True)
        return 0

    if args.part == 'a':
        noise_cases = []
    elif args.part == 'b':
        main_cases = []

    failures = 0
    trained = 0

    def budget_left() -> bool:
        return args.limit <= 0 or trained < args.limit

    print('=== seeds 3 and 4: MLP and TabM ===', flush=True)
    for method, dataset, seed, ledger_dir, header, run_dir, text in main_cases:
        if not budget_left():
            break
        if not (run_dir / str(seed) / 'DONE').exists():
            trained += 1
        ok = run_one(
            python=args.python,
            ledger_dir=ledger_dir,
            header=header,
            run_dir=run_dir,
            config_text=text,
            method=method,
            dataset=dataset,
            seed=seed,
            noise=None,
            commit=commit,
            env=env,
        )
        failures += 0 if ok else 1

    print('=== E8 label noise on california ===', flush=True)
    for method, dataset, seed, level, ledger_dir, header, run_dir, text in noise_cases:
        if not budget_left():
            break
        if not (run_dir / str(seed) / 'DONE').exists():
            trained += 1
        ok = run_one(
            python=args.python,
            ledger_dir=ledger_dir,
            header=header,
            run_dir=run_dir,
            config_text=text,
            method=method,
            dataset=dataset,
            seed=seed,
            noise=level,
            commit=commit,
            env=env,
        )
        failures += 0 if ok else 1

    if failures:
        print(f'EXTENSIONS FAILED: {failures} run(s)', flush=True)
        return 1
    print('EXTENSIONS DONE', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
