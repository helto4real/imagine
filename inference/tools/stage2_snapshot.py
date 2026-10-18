#!/usr/bin/env python3
"""Freeze/verify Stage 2 source, fixtures, resources and referenced source bytes."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path);p.add_argument('--verify',type=Path)
a=p.parse_args();project=Path(__file__).resolve().parents[2]
def digest(path):return hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
if a.verify:
 snapshot=json.loads(a.verify.read_text())
 for name,expected in snapshot['implementation'].items():assert digest(project/name)==expected,name
 for name,expected in snapshot['reference_inputs'].items():assert digest(Path(name))==expected,name
 print(f'Verified {len(snapshot["implementation"])} implementation/fixture/resource inputs and {len(snapshot["reference_inputs"])} original source inputs')
elif a.output:
 files=[project/'inference'/n for n in ['Cargo.toml','Cargo.lock','README.md']]
 for folder in ['src','tests','native','tools','resources','profiles','fixtures']:
  files.extend(f for f in (project/'inference'/folder).rglob('*')if f.is_file()and'__pycache__'not in f.parts and f.suffix!='.gz')
 files.extend((project/'design/reference-local'/n)for n in ['reference-lock.json','source-manifest.json','source-acquisition-lock.json','native-linkage-lock.json'])
 originals={}
 for f in (project/'inference/fixtures').glob('*.json'):
  data=json.loads(f.read_text())
  for value in data.get('sources',{}).values():
   if isinstance(value,dict)and'path'in value and'sha256'in value:originals[value['path']]=value['sha256']
 schema=json.loads((project/'inference/resources/qwen-image21-schema.json').read_text())
 for name,h in schema['sources'].items():originals[str(Path('/home/thhel/git/ComfyUI')/name)]=h
 for name,h in originals.items():assert digest(Path(name))==h,name
 snapshot=dict(schema=1,implementation={str(f.relative_to(project)):digest(f)for f in sorted(set(files))},reference_inputs=originals,excluded='Existing Stage 1 gzip archive remains covered by its preserved archive manifest/checkpoint')
 a.output.write_text(json.dumps(snapshot,indent=2)+'\n');print(f'Captured {len(files)} inputs, {len(originals)} original sources')
else:p.error('Choose --output or --verify')
