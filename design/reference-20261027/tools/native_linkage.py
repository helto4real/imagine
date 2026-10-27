#!/usr/bin/env python3
"""Fingerprint installed ELF dependencies statically. Never loads a library."""
import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
SITE=Path('/home/thhel/git/ComfyUI/.venv/lib/python3.12/site-packages')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4194304),b''):h.update(block)
    return h.hexdigest()


def readelf(flag,p):
    result=subprocess.run(['readelf',flag,str(p)],capture_output=True,text=True,check=False)
    return result.stdout if result.returncode==0 else ''


def capture():
    target=BASE/'native-linkage-lock.json'
    if target.exists():raise SystemExit('Refusing to overwrite native linkage lock.')
    deps=json.loads((BASE/'dependencies.json').read_text())
    selected=set()
    for name in deps['code_fingerprinted_distributions']:
        directory=SITE/deps['distributions'][name]['metadata_dir']
        for row in csv.reader((directory/'RECORD').read_text().splitlines()):
            if not row or row[0].startswith(('../','/')):continue
            p=SITE/row[0]
            if re.search(r'\.so(?:\..+)?$',p.name) and p.is_file() and p.resolve().is_relative_to(SITE):selected.add(p)
    records=[];needed=set()
    for p in sorted(selected):
        dynamic=readelf('-d',p)
        names=re.findall(r'\(NEEDED\).*?\[(.*?)\]',dynamic)
        needed.update(names)
        records.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,
                            needed=names,rpath=re.findall(r'\((?:RUNPATH|RPATH)\).*?\[(.*?)\]',dynamic),
                            build_id=re.findall(r'Build ID: (\S+)',readelf('-n',p)),
                            compiler_comments=readelf('-p.comment',p)))
    bundled={p.name for p in selected}
    system=[];unresolved=[]
    for name in sorted(needed-bundled):
        candidates=[Path('/usr/lib')/name,Path('/lib')/name,Path('/lib64')/name]
        p=next((p for p in candidates if p.is_file()),None)
        if p is None:unresolved.append(name);continue
        system.append(dict(soname=name,path=str(p),resolved_path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size,
                           status='installed candidate; active runtime loader resolution not observed'))
    obj=dict(policy='RECORD-listed native ELF files including versioned SONAMEs; no dlopen, ldd, imports, GPU initialization or runtime reads',
             distribution_objects=records,system_candidates=system,unresolved_sonames=unresolved)
    target.write_text(json.dumps(obj,indent=2)+'\n')
    print(f'Locked {len(records)} package ELF files and {len(system)} system candidates; {len(unresolved)} unresolved SONAMEs.')
    print('Unresolved:',unresolved)


def verify():
    data=json.loads((BASE/'native-linkage-lock.json').read_text())
    for row in data['distribution_objects']+data['system_candidates']:
        assert sha(Path(row['path']))==row['sha256'],row['path']
    print('PASS: package ELF and system candidate fingerprints match.')


def audit():
    """Conservative static closure used to prepare isolated runtime mounts."""
    from collections import defaultdict, deque
    target=BASE/'native-linkage-lock.json'
    data=json.loads(target.read_text())
    objects={row['path']:row for row in data['distribution_objects']}
    names=defaultdict(list)
    for path in objects:names[Path(path).name].append(path)
    roots=[str(SITE/rel)for rel in ['comfy_aimdo/aimdo.so',
        'comfy_kitchen/backends/cuda/_C.abi3.so','torch/lib/libc10.so',
        'torch/lib/libc10_cuda.so','torch/lib/libtorch.so',
        'torch/lib/libtorch_cpu.so','torch/lib/libtorch_cuda.so']]
    queue=deque(roots);seen=set();unresolved=set()
    system={row['soname']for row in data['system_candidates']}
    while queue:
        path=queue.popleft()
        if path in seen:continue
        row=objects[path];assert sha(Path(path))==row['sha256'],path
        seen.add(path)
        for name in row['needed']:
            if name in names:queue.extend(names[name])
            elif name not in system:unresolved.add(name)
    assert not unresolved,sorted(unresolved)
    data['initial_target_linkage_audit']=dict(root_objects=roots,
        package_object_closure=sorted(seen),unresolved_sonames=[],
        method='static conservative package-name closure; hashes verified; not runtime load proof',
        dynamic_driver='AIMDO resolves libcuda dynamically; isolated Rust linkage validation follows')
    target.write_text(json.dumps(data,indent=2)+'\n')
    lock=BASE/'reference-lock.json';reference=json.loads(lock.read_text())
    reference['artifacts']['native-linkage-lock.json']=sha(target)
    reference['artifacts']['tools/native_linkage.py']=sha(Path(__file__))
    lock.write_text(json.dumps(reference,indent=2)+'\n')
    print(f'Verified static closure of {len(seen)} package objects; no unresolved SONAMEs.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['capture','verify','audit'])
    args=parser.parse_args()
    {'capture':capture,'verify':verify,'audit':audit}[args.mode]()
