"""Check upload files against the bundled official template; Python stdlib only."""
import argparse
import csv
import math
from pathlib import Path


def check(path):
    template_path = Path(__file__).resolve().parent / 'inputs/template.csv'
    with template_path.open(newline='', encoding='utf-8-sig') as fh:
        reader = csv.DictReader(fh)
        columns, expected = reader.fieldnames, list(reader)
    with Path(path).open(newline='', encoding='utf-8-sig') as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != columns:
            raise ValueError('Columns do not match upload template; do not submit a _sources.csv')
        rows = list(reader)
    if len(rows) != len(expected):
        raise ValueError(f'Expected {len(expected)} rows, found {len(rows)}')
    ids = set()
    for position, (row, ref) in enumerate(zip(rows, expected), 1):
        if row['id'] != ref['id'] or row['id'] in ids:
            raise ValueError(f'ID/order mismatch at row {position}')
        ids.add(row['id'])
        if row['video_id'].strip() != ref['video_id'].strip() or row['question'] != ref['question']:
            raise ValueError(f'Video/question mismatch at row {position}')
        try:
            value = float(row['parsed_value'])
        except (TypeError, ValueError):
            raise ValueError(f'Non-numeric answer at row {position}') from None
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'Non-positive/non-finite answer at row {position}')
    print(f'PASS: {path} — {len(rows)} rows; IDs, videos, questions and positive finite answers verified.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    for file in parser.parse_args().files:
        check(file)
