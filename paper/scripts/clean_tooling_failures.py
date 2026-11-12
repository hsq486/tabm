"""Split tooling-fault rows out of the run ledgers.

A row is treated as a tooling fault, not a result, when it failed within two
seconds: importing torch and loading the dataset alone takes longer than that,
so such a row means the trainer never started. They came from two aborted
launcher attempts on 2026-11-12 (a host-shell pipeline fault, then a
process-tree kill); they are moved to results/_tooling_failures.csv instead of
being deleted, so the main ledgers only contain runs that actually ran.
"""

from __future__ import annotations

import csv
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
LEDGERS = (
    PAPER / 'results' / 'E2' / 'runs.csv',
    PAPER / 'results' / 'E3' / 'runs.csv',
    PAPER / 'results' / 'E8' / 'runs.csv',
)
ARCHIVE = PAPER / 'results' / '_tooling_failures.csv'


def is_tooling_fault(row: dict[str, str]) -> bool:
    """A failed row under two seconds means the trainer never started."""
    if row['status'] != 'failed':
        return False
    try:
        duration = float(row['duration_sec'])
    except (TypeError, ValueError):
        return False
    return duration < 2.0


def main() -> None:
    archived: list[dict[str, str]] = []
    kept_total = 0

    for ledger in LEDGERS:
        with ledger.open(encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
            header = list(rows[0].keys()) if rows else []

        kept = []
        for row in rows:
            if is_tooling_fault(row):
                row = dict(row)
                row['source_ledger'] = ledger.relative_to(PAPER).as_posix()
                archived.append(row)
            else:
                kept.append(row)

        with ledger.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=header, lineterminator='\r\n')
            writer.writeheader()
            writer.writerows(kept)
        kept_total += len(kept)
        print(f'{ledger.name}: kept {len(kept)}, archived {len(rows) - len(kept)}')

    if archived:
        fields = [
            'timestamp',
            'commit',
            'method',
            'dataset',
            'noise',
            'seed',
            'exit_code',
            'duration_sec',
            'status',
            'report',
            'source_ledger',
        ]
        exists = ARCHIVE.exists()
        with ARCHIVE.open('a' if exists else 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=fields, lineterminator='\r\n', restval=''
            )
            if not exists:
                writer.writeheader()
            writer.writerows(archived)
        print(f'archived {len(archived)} rows -> {ARCHIVE.name}')
    print('rows left in ledgers:', kept_total)


if __name__ == '__main__':
    main()
