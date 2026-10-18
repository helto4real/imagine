#!/usr/bin/env python3
"""Restore named compiler configuration inputs from the locked source header."""
import argparse, hashlib, json, re, subprocess
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--check', action='store_true')
a = p.parse_args()
project = Path(__file__).resolve().parents[2]
lock = json.loads((project / 'design/stage-1-triton-toolchain-lock.json').read_text())['reference_environment_header']
header = (a.source / lock['relative_path']).read_bytes()
assert hashlib.sha256(header).hexdigest() == lock['sha256'], 'Compiler header changed'
section = header.decode().split('CACHE_INVALIDATING_ENV_VARS', 1)[1].split('};', 1)[0]
names = sorted(re.findall(r'"([A-Za-z0-9_]+)"', section))
assert names == lock['invalidating_names'], 'Frozen compiler variable list changed'
source = '// Generated from the locked CACHE_INVALIDATING_ENV_VARS header.\npub(crate) const NAMES: &[&str] = &[' + ','.join(json.dumps(v) for v in names) + '];\n'
source = subprocess.run(['rustfmt', '--edition', '2024', '--emit', 'stdout'], input=source, text=True, capture_output=True, check=True).stdout
if a.check:
    assert a.output.read_text() == source, 'Generated compiler names changed'
else:
    a.output.write_text(source)
print('Verified 37 frozen compiler configuration names')
