#!/usr/bin/env python3
"""Verify supplied file hashes and syntax; this does not rerun experimental outcomes."""
from pathlib import Path
import ast,csv,hashlib,json,gzip
root=Path(__file__).resolve().parent
count=0
for row in csv.DictReader((root/'SOURCE_MANIFEST.csv').open()):
    p=root/row['path']; data=p.read_bytes()
    if hashlib.sha256(data).hexdigest()!=row['sha256_packaged']:raise RuntimeError('Packaged hash mismatch: '+row['path'])
    original=gzip.decompress(data) if row['storage_format']=='gzip' else data
    if hashlib.sha256(original).hexdigest()!=row['sha256_original']:raise RuntimeError('Source hash mismatch: '+row['path'])
    if p.suffix=='.py':ast.parse(original.decode('utf-8-sig'),filename=row['path'])
    count+=1
print(json.dumps({'source_files_verified':count,'status':'PASS','scope':'hashes and syntax, not full experimental reproduction'},indent=2))
