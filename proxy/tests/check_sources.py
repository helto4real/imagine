#!/usr/bin/env python3
"""Verify the reference snapshot and byte-identical workflow/profile copies."""
import argparse,hashlib,json,pathlib,subprocess
p=argparse.ArgumentParser();p.add_argument('--reference',required=True,type=pathlib.Path);p.add_argument('--sdk',type=pathlib.Path);p.add_argument('--aio-reference',type=pathlib.Path);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[1];lock=json.loads((root/'reference/source-lock.json').read_text())
assert subprocess.check_output(['git','-C',str(a.reference),'rev-parse','HEAD'],text=True).strip()==lock['commit']
def check(path,expected):assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,f'hash mismatch: {path}'
for name,digest in lock['sha256'].items():check(a.reference/name,digest)
for name in lock['copied_files']:check(root/name,lock['sha256'][name])
if a.sdk:
 for name,digest in lock['sdk_sha256'].items():check(a.sdk/name,digest)
if a.aio_reference:
 for name,digest in lock['aio_sha256'].items():check(a.aio_reference/name,digest)
print(f"PASS: {len(lock['sha256'])} reference files, {len(lock['copied_files'])} byte-identical copies" + ('; SDK verified' if a.sdk else '') + ('; AIO verified' if a.aio_reference else ''))
