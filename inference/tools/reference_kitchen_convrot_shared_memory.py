#!/usr/bin/env python3
"""Kitchen's ConvRot INT8 fused-quantizer shared-memory estimate over a row/width grid.

The original function is taken from the installed Kitchen source by AST after its
hash is checked against the selected reference; nothing is imported or launched.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from reference_paths import reference_dir

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--site', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
project = Path(__file__).resolve().parents[2]
rel = 'comfy_kitchen/backends/cuda/__init__.py'
manifest = json.loads((reference_dir(project) / 'source-manifest.json').read_text())
expected = next(e['sha256'] for e in manifest if e['root'] == 'site' and e['path'] == rel)
source = (a.site / rel).read_bytes()
assert hashlib.sha256(source).hexdigest() == expected, rel
tree = ast.parse(source)
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_convrot_int8_fused_shared_memory_bytes')
space = {}
exec(compile(ast.Module(body=[node], type_ignores=[]), rel, 'exec'), space)
estimate = space['_convrot_int8_fused_shared_memory_bytes']
rows = [1, 2, 7, 512, 4096]
widths = [256, 512, 1024, 2560, 3072, 3584, 4096, 6144, 6400, 8192, 12287, 12288, 13824, 16384, 24576]
cases = [dict(rows=m, width=k, bytes=estimate(m, k)) for m in rows for k in widths]
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(dict(schema=1, source=rel, source_sha256=expected, cases=cases), indent=1) + '\n')
print('Captured', len(cases), 'ConvRot INT8 shared-memory estimates')
