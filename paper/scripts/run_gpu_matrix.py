"""Generate and run the GPU experiment matrices (E10-E14) on the second machine.

Every run derives from a frozen TOML template; only the listed overrides are
applied, so each matrix stays reproducible and auditable. Results are written
next to their configs and summarised into ``paper/results/gpu/<matrix>/``.

Usage (from ``paper/``):
    python scripts/run_gpu_matrix.py e11 --seeds 0,1,2
    python scripts/run_gpu_matrix.py all --dry-run
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent
MODEL = PAPER / 'bin' / 'model.py'
OUT_ROOT = PAPER / 'exp' / 'reproduce' / 'gpu'
RES_ROOT = PAPER / 'results' / 'gpu'

DATASETS = [
    'adult',
    'black-friday',
    'california',
    'churn',
    'covtype2',
    'diamond',
    'higgs-small',
    'house',
    'microsoft',
    'otto',
]
CORE = ['california', 'churn', 'house']

# d_block multipliers used by E13 to bracket TabM's parameter count.
# churn needs multipliers below 1: the official MLP (6x752, 2.85M params) is
# far wider than the official TabM (4x192, 0.19M params), so equal-parameter
# controls must narrow the MLP, not widen it.
EQUAL_PARAM_WIDTHS = {
    'california': [2.0, 2.5, 3.0],
    'churn': [0.22, 0.26, 0.30],
    'house': [1.5, 2.0, 2.5],
}
E14_LR_FACTORS = [0.5, 1.0, 2.0]
E14_DROPOUTS = [0.1, None, 0.3]  # None = official value


@dataclass
class Run:
    matrix: str
    method: str
    dataset: str
    seed: int
    source: Path
    tag: str
    overrides: dict[tuple[str, str], object] = field(default_factory=dict)


def official(method: str, dataset: str) -> Path:
    return PAPER / 'exp' / method / dataset / '0-evaluation' / '0.toml'


def frozen(name: str) -> Path:
    return PAPER / 'configs' / name


def get_key(text: str, section: str, key: str) -> str:
    cur = ''
    for line in text.splitlines():
        m = re.match(r'\s*\[(.+?)\]\s*$', line)
        if m:
            cur = m.group(1).strip()
            continue
        if cur == section:
            m = re.match(rf'\s*{re.escape(key)}\s*=\s*(.+?)\s*$', line)
            if m:
                return m.group(1).strip().strip('"\'')
    raise KeyError(f'{key!r} not found in section {section!r}')


def set_key(text: str, section: str, key: str, value: object) -> str:
    lines = text.splitlines()
    cur = ''
    for i, line in enumerate(lines):
        m = re.match(r'\s*\[(.+?)\]\s*$', line)
        if m:
            cur = m.group(1).strip()
            continue
        if cur == section and re.match(rf'\s*{re.escape(key)}\s*=', line):
            lines[i] = f'{key} = {value}'
            return '\n'.join(lines) + '\n'

    # Key absent: append it at the end of the requested section.
    if section:
        start = None
        for i, line in enumerate(lines):
            if re.match(rf'\s*\[{re.escape(section)}\]\s*$', line):
                start = i
        if start is None:
            lines += ['', f'[{section}]']
            start = len(lines) - 1
        end = len(lines)
        for j in range(start + 1, len(lines)):
            if lines[j].lstrip().startswith('['):
                end = j
                break
        while end > start + 1 and not lines[end - 1].strip():
            end -= 1
        lines.insert(end, f'{key} = {value}')
    else:
        lines.insert(0, f'{key} = {value}')
    return '\n'.join(lines) + '\n'


def build(matrix: str, seeds: dict[str, list[int]]) -> list[Run]:
    runs: list[Run] = []

    if matrix == 'e10':
        # Same frozen configs as the CPU phase; FP32 is the invariance check.
        for dataset in CORE:
            for seed in seeds['e10']:
                runs.append(
                    Run(
                        'e10',
                        'mlp',
                        dataset,
                        seed,
                        official('mlp', dataset),
                        'fp32',
                        {('', 'amp'): 'false'},
                    )
                )
                runs.append(
                    Run(
                        'e10',
                        'tabm',
                        dataset,
                        seed,
                        frozen(f'E3-tabm-{dataset}.toml'),
                        'fp32',
                        {('', 'amp'): 'false'},
                    )
                )
        for seed in seeds['e10']:
            runs.append(
                Run(
                    'e10',
                    'mlp',
                    'california',
                    seed,
                    official('mlp', 'california'),
                    'bf16',
                    {('', 'amp'): 'true'},
                )
            )
            runs.append(
                Run(
                    'e10',
                    'tabm',
                    'california',
                    seed,
                    frozen('E3-tabm-california.toml'),
                    'bf16',
                    {('', 'amp'): 'true'},
                )
            )

    elif matrix == 'e11':
        for dataset in DATASETS:
            for method in ('mlp', 'tabm'):
                for seed in seeds['e11']:
                    runs.append(
                        Run(
                            'e11',
                            method,
                            dataset,
                            seed,
                            official(method, dataset),
                            'full',
                            {('', 'amp'): 'true'},
                        )
                    )

    elif matrix == 'e12':
        for dataset in ('california', 'house'):
            base = frozen(f'E3-tabm-{dataset}.toml')
            ref_text = official('tabm', dataset).read_text(encoding='utf-8')
            n_blocks = get_key(ref_text, 'model.backbone', 'n_blocks')
            d_block = get_key(ref_text, 'model.backbone', 'd_block')
            for seed in seeds['e12']:
                runs.append(Run('e12', 'tabm', dataset, seed, base, 'r0', {}))
                runs.append(
                    Run('e12', 'tabm', dataset, seed, base, 'r1', {('model', 'k'): 32})
                )
                runs.append(
                    Run(
                        'e12',
                        'tabm',
                        dataset,
                        seed,
                        base,
                        'r2',
                        {
                            ('model', 'k'): 32,
                            ('model.backbone', 'n_blocks'): n_blocks,
                            ('model.backbone', 'd_block'): d_block,
                        },
                    )
                )
                runs.append(
                    Run(
                        'e12',
                        'tabm',
                        dataset,
                        seed,
                        official('tabm', dataset),
                        'r3',
                        {},
                    )
                )

    elif matrix == 'e13':
        for dataset, factors in EQUAL_PARAM_WIDTHS.items():
            ref_text = official('mlp', dataset).read_text(encoding='utf-8')
            base_d = int(get_key(ref_text, 'model.backbone', 'd_block'))
            for factor in factors:
                tag = f'w{factor:g}'
                for seed in seeds['e13']:
                    runs.append(
                        Run(
                            'e13',
                            'mlp',
                            dataset,
                            seed,
                            official('mlp', dataset),
                            tag,
                            {('model.backbone', 'd_block'): round(base_d * factor)},
                        )
                    )

    elif matrix == 'e14':
        for dataset in ('california', 'churn'):
            ref_text = official('tabm', dataset).read_text(encoding='utf-8')
            base_lr = float(get_key(ref_text, 'optimizer', 'lr'))
            base_drop = float(get_key(ref_text, 'model.backbone', 'dropout'))
            for lr_factor in E14_LR_FACTORS:
                for drop in E14_DROPOUTS:
                    drop_value = base_drop if drop is None else drop
                    tag = f'lr{lr_factor:g}_dr{drop_value:.2f}'
                    for seed in seeds['e14']:
                        runs.append(
                            Run(
                                'e14',
                                'tabm',
                                dataset,
                                seed,
                                official('tabm', dataset),
                                tag,
                                {
                                    ('optimizer', 'lr'): f'{base_lr * lr_factor:.10g}',
                                    ('model.backbone', 'dropout'): f'{drop_value:.10g}',
                                },
                            )
                        )

    else:
        raise ValueError(f'unknown matrix: {matrix}')

    return runs


def run_dir(run: Run) -> tuple[Path, Path]:
    """Return the output directory and config path without touching the disk."""
    tag_dir = OUT_ROOT / run.matrix / run.method / run.dataset / run.tag
    return tag_dir, tag_dir / f'{run.seed}.toml'


def prepare(run: Run) -> tuple[Path, Path]:
    tag_dir, config_path = run_dir(run)
    tag_dir.mkdir(parents=True, exist_ok=True)

    text = run.source.read_text(encoding='utf-8')
    text = set_key(text, '', 'seed', run.seed)
    for (section, key), value in run.overrides.items():
        text = set_key(text, section, key, value)
    # ``Path.write_text(newline=...)`` needs Python 3.10; open() works on 3.9 too.
    with config_path.open('w', encoding='utf-8', newline='\n') as f:
        f.write(text)
    return tag_dir, config_path


LEDGER = [
    'timestamp',
    'commit',
    'matrix',
    'method',
    'dataset',
    'tag',
    'seed',
    'device',
    'amp',
    'torch',
    'exit_code',
    'duration_sec',
    'status',
    'report',
    'command',
]


def append_row(path: Path, row: dict) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER)
        if new:
            writer.writeheader()
        writer.writerow(row)


def summarise(matrix: str) -> None:
    rows = []
    for report in sorted((OUT_ROOT / matrix).rglob('report.json')):
        data = json.loads(report.read_text(encoding='utf-8'))
        rel = report.relative_to(OUT_ROOT / matrix).parts
        rows.append(
            {
                'method': rel[0],
                'dataset': rel[1],
                'tag': rel[2],
                'seed': report.parent.name,
                'score': data['metrics']['test']['score'],
                'n_parameters': data.get('n_parameters'),
                'best_step': data.get('best_step'),
            }
        )
    if not rows:
        return
    out = RES_ROOT / matrix / 'summary.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'summary: {out} ({len(rows)} runs)')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('matrix', choices=['e10', 'e11', 'e12', 'e13', 'e14', 'all'])
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument(
        '--datasets', default='', help='comma-separated subset, e11/e10 only'
    )
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    matrices = (
        ['e10', 'e11', 'e12', 'e13', 'e14'] if args.matrix == 'all' else [args.matrix]
    )
    seed_list = [int(s) for s in args.seeds.split(',') if s]
    seeds = {m: seed_list for m in matrices}
    if matrices == ['e14'] or args.matrix == 'all':
        seeds['e14'] = [s for s in seed_list if s in (0, 1)]

    runs = []
    for matrix in matrices:
        runs += build(matrix, seeds)
    if args.datasets:
        keep = {s.strip() for s in args.datasets.split(',')}
        runs = [r for r in runs if r.dataset in keep]

    print(f'{len(runs)} runs; python={args.python}')
    failures = 0
    for index, run in enumerate(runs, 1):
        tag_dir, config_path = run_dir(run)
        report = tag_dir / str(run.seed) / 'report.json'
        shown = config_path.relative_to(PAPER.parent).as_posix()
        if args.dry_run:
            print(f'[{index}/{len(runs)}] dry-run: {shown}')
            continue
        if report.exists() and not args.force:
            print(f'[{index}/{len(runs)}] skip (done): {shown}')
            continue
        prepare(run)

        print(
            f'[{index}/{len(runs)}] {run.matrix}/{run.method}/{run.dataset}/'
            f'{run.tag}/seed={run.seed}',
            flush=True,
        )
        started = time.time()
        log = tag_dir / f'{run.seed}.log'
        with log.open('w', encoding='utf-8') as f:
            code = subprocess.call(
                [args.python, str(MODEL), str(config_path), '--force'],
                stdout=f,
                stderr=subprocess.STDOUT,
            )
        elapsed = round(time.time() - started, 1)
        ok = code == 0 and report.exists()
        if not ok:
            failures += 1

        info = {
            'device': 'cuda',
            'amp': run.overrides.get(('', 'amp'), 'true'),
            'torch': '',
            'commit': '',
        }
        with contextlib.suppress(Exception):
            import torch

            info['torch'] = torch.__version__
        with contextlib.suppress(Exception):
            info['commit'] = subprocess.check_output(
                ['git', '-C', str(PAPER.parent), 'rev-parse', '--short', 'HEAD'],
                text=True,
            ).strip()

        append_row(
            RES_ROOT / run.matrix / 'runs.csv',
            {
                'timestamp': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                **info,
                'matrix': run.matrix,
                'method': run.method,
                'dataset': run.dataset,
                'tag': run.tag,
                'seed': run.seed,
                'exit_code': code,
                'duration_sec': elapsed,
                'status': 'ok' if ok else 'failed',
                'report': report.relative_to(PAPER.parent).as_posix(),
                'command': (
                    'python bin/model.py '
                    f'{config_path.relative_to(PAPER.parent).as_posix()} --force'
                ),
            },
        )

    for matrix in matrices:
        summarise(matrix)

    print(f'finished: {len(runs) - failures}/{len(runs)} ok')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
