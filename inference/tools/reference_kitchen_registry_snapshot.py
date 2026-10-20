#!/usr/bin/env python3
"""Locked original backend status/capability tables; no model/tensor payloads."""
import argparse,hashlib,json,sys
from pathlib import Path
from reference_paths import reference_dir
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--require-cuda',action='store_true');a=p.parse_args()
project=Path(__file__).resolve().parents[2];manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text())
sources={}
for e in manifest:
 if e['root']=='site'and e['path'].startswith('comfy_kitchen/')and e['path'].endswith('.py'):
  digest=hashlib.sha256((a.site/e['path']).read_bytes()).hexdigest();assert digest==e['sha256'];sources[e['path']]=digest
sys.path.insert(0,str(a.site))
import comfy_kitchen as ck
backends=ck.registry.list_backends()
if a.require_cuda:assert backends['cuda']['available']and backends['triton']['available'],'Expected the approved CUDA reference environment; refusing a CPU-only golden'
a.output.write_text(json.dumps(dict(schema=1,sources=sources,backends=backends),indent=2)+'\n')
print('Captured locked original backend tables; model weights/tensor payloads: 0')
