#!/usr/bin/env python3
"""Check the release manifest and final figure/table assets without running experiments."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows = list(csv.DictReader((ROOT / 'RELEASE_FILES.csv').open(encoding='utf-8')))
    for row in rows:
        path = ROOT / row['path']
        if not path.is_file() or path.stat().st_size != int(row['bytes']) or digest(path) != row['sha256']:
            raise SystemExit('Release mismatch: ' + row['path'])
    figures = list(csv.DictReader((ROOT / 'paper/FIGURE_INDEX.csv').open(encoding='utf-8')))
    for row in figures:
        if digest(ROOT / row['path']) != row['sha256']:
            raise SystemExit('Figure mismatch: ' + row['path'])
    table_root = ROOT / 'paper/tables/revision'
    tables = json.loads((table_root / 'table_index.json').read_text(encoding='utf-8'))
    for row in tables:
        if digest(table_root / row['snippet_path']) != row['snippet_sha256']:
            raise SystemExit('Table mismatch: ' + row['snippet_path'])
    if len(figures) != 18 or len(tables) != 23:
        raise SystemExit('Unexpected final figure/table count')
    print(json.dumps({'status': 'PASS', 'manifest_files': len(rows),
                      'figure_assets': len(figures), 'table_snippets': len(tables),
                      'scope': 'file integrity only; no simulation rerun'}, indent=2))


if __name__ == '__main__':
    main()
