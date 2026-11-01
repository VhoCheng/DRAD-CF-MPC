#!/usr/bin/env python3
"""Expand archived CSV inputs without changing their values or overwriting different files."""
from pathlib import Path
import csv,gzip,hashlib
root=Path(__file__).resolve().parent
for row in csv.DictReader((root/'SOURCE_MANIFEST.csv').open()):
    if row['storage_format']!='gzip':continue
    src=root/row['path']; dst=src.with_suffix('')
    data=gzip.decompress(src.read_bytes())
    if hashlib.sha256(data).hexdigest()!=row['sha256_original']:raise RuntimeError('Hash mismatch: '+str(src))
    if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest()!=row['sha256_original']:raise RuntimeError('Refusing to overwrite different data: '+str(dst))
    dst.write_bytes(data)
    print(dst.relative_to(root))
