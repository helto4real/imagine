#!/usr/bin/env python3
"""Lock Git-tracked tokenizer/config assets and boundary helper source only."""
import hashlib
import importlib.util
import json
import subprocess
import tarfile
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
if (BASE/'boundary-source.tar.gz').exists():
    raise SystemExit('Supplement already frozen; refusing to overwrite it.')
spec=importlib.util.spec_from_file_location('capture_tool',BASE/'tools/capture.py')
capture=importlib.util.module_from_spec(spec);spec.loader.exec_module(capture)
root=capture.COMFY
lock_path=BASE/'reference-lock.json'
lock=json.loads(lock_path.read_text())
assert capture.cmd('git','-C',str(root),'rev-parse','HEAD')==lock['comfy_revision']
tracked=subprocess.check_output(['git','-C',str(root),'ls-files','-z'],text=True).split('\0')
manifest=json.loads((BASE/'source-manifest.json').read_text())
inventory=json.loads((BASE/'inference-inventory.json').read_text())
known={(x['root'],x['path']) for x in manifest}
selected=[]
for rel in tracked:
    if not rel: continue
    p=Path(rel)
    asset=rel.startswith(('comfy/','comfy_extras/')) and p.suffix in {'.json','.txt','.model'}
    boundary=(rel.startswith('app/') or rel in {'execution.py','server.py','protocol.py','comfyui_version.py'}) and p.suffix=='.py'
    if (asset or boundary) and ('comfy',rel) not in known:selected.append(rel)
with tarfile.open(BASE/'boundary-source.tar.gz','w:gz') as tar:
    for rel in sorted(selected):
        p=root/rel
        assert not p.is_symlink()
        manifest.append(dict(root='comfy',path=rel,size=p.stat().st_size,sha256=capture.digest(p)))
        tar.add(p,arcname='comfy/'+rel,recursive=False)
        if p.suffix=='.py':
            item=capture.scan(rel,p.read_text(encoding='utf-8-sig'))
            item['category']='shared-boundary-helper; graph-server-implementation-excluded'
            inventory.append(item)
capture.save('source-manifest.json',sorted(manifest,key=lambda x:(x['root'],x['path'])))
capture.save('inference-inventory.json',inventory)
lock['source_count']=len(manifest)
for name in ['source-manifest.json','inference-inventory.json','boundary-source.tar.gz','tools/supplement.py']:
    lock['artifacts'][name]=capture.digest(BASE/name)
lock_path.write_text(json.dumps(lock,indent=2)+'\n')
print(f'Locked {len(selected)} additional Git-tracked source assets/helpers; no UI example images read.')
