#!/usr/bin/env python3
"""Source/package-only reference capture. Never imports reference code.

Only Git-listed source paths and installed distribution metadata/RECORD-listed
code are read. No model roots, custom_nodes, logs, outputs or user folders.
"""
import argparse
import ast
import csv
import email.parser
import hashlib
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
COMFY = Path('/home/thhel/git/ComfyUI')
SITE = COMFY / '.venv/lib/python3.12/site-packages'
PREFIXES = ('comfy/', 'comfy_extras/', 'comfy_api/', 'comfy_api_nodes/',
            'comfy_execution/', 'comfy_config/', 'comfy_types/')
ROOT_FILES = {'nodes.py', 'node_helpers.py', 'folder_paths.py',
              'requirements.txt', 'pyproject.toml', 'LICENSE', 'cuda_malloc.py',
              'latent_preview.py', 'app/latent_preview.py'}
CODE = {'.py', '.pyi', '.h', '.hpp', '.c', '.cpp', '.cu', '.cuh', '.so',
        '.a', '.cmake', '.toml'}


def cmd(*args):
    return subprocess.check_output(args, text=True).strip()


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def expr(n):
    return ast.unparse(n)


def category(path):
    if path.startswith('comfy_api_nodes/'):
        return 'bundled-provider-boundary-review'
    if path.startswith('comfy_execution/'):
        return 'graph-framework-excluded; shared-inference-helpers-review'
    if path.startswith('comfy_api/'):
        return 'schema-framework; inference-contract-review'
    if 'ldm/' in path: return 'model-architectures'
    if 'text_encoders/' in path: return 'conditioning-encoders'
    if 'weight_adapter/' in path: return 'weight-patching'
    if path.startswith('comfy_extras/') or path == 'nodes.py':
        return 'builtin-operations; inference-vs-io-ui-review'
    return 'inference-core-and-shared-dependencies'


def scan(path, data):
    tree = ast.parse(data, filename=path)
    symbols, imports, registrations, dynamic = [], [], [], []
    class Visitor(ast.NodeVisitor):
        def __init__(self): self.scope = []
        def visit_ClassDef(self, n):
            symbols.append(dict(id=f'{path}:{n.lineno}', name='.'.join(self.scope+[n.name]),
                                kind='class', line=n.lineno, end=n.end_lineno,
                                bases=[expr(x) for x in n.bases], status='inventoried'))
            self.scope.append(n.name); self.generic_visit(n); self.scope.pop()
        def visit_FunctionDef(self, n):
            calls = sorted({expr(x.func) for x in ast.walk(n) if isinstance(x, ast.Call)})
            symbols.append(dict(id=f'{path}:{n.lineno}', name='.'.join(self.scope+[n.name]),
                                kind='function', line=n.lineno, end=n.end_lineno,
                                signature=expr(n.args), decorators=[expr(x) for x in n.decorator_list],
                                calls=calls, status='inventoried'))
            self.scope.append(n.name); self.generic_visit(n); self.scope.pop()
        visit_AsyncFunctionDef = visit_FunctionDef
        def visit_Import(self, n):
            imports.extend(dict(module=x.name, line=n.lineno, conditional='static; not evaluated') for x in n.names)
        def visit_ImportFrom(self, n):
            imports.append(dict(module='.'*n.level+(n.module or ''), names=[x.name for x in n.names], line=n.lineno,
                                conditional='static; not evaluated'))
        def visit_Call(self, n):
            name=expr(n.func)
            if any(x in name.lower() for x in ('register', 'schema', 'get_node_list')):
                registrations.append(dict(line=n.lineno, call=expr(n)))
            if any(x in name for x in ('import_module', '__import__', 'load_library', 'cpp_extension', 'compile', 'load_custom_node')):
                dynamic.append(dict(line=n.lineno, call=expr(n), status='requires-source-path-resolution'))
            self.generic_visit(n)
    Visitor().visit(tree)
    # Keep assignments and their enclosing predicates via source line anchors.
    # Expressions are NOT evaluated: conditional and generated registries remain visible.
    tables=[]
    for n in ast.walk(tree):
        if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets=n.targets if isinstance(n, ast.Assign) else [n.target]
            if n.value is None: continue
            for target in targets:
                name=expr(target)
                if (name in {'models','adapters','adapter_maps','KSAMPLER_NAMES','SAMPLER_NAMES','SCHEDULER_HANDLERS','QUANT_ALGOS','NODE_CLASS_MAPPINGS','__all__','extra_nodes'}
                    or any(x in name for x in ('QUANT_ALGOS[','NODE_CLASS_MAPPINGS[','_priority'))):
                    tables.append(dict(target=name, line=n.lineno, expression=expr(n.value)))
    return dict(path=path, category=category(path), symbols=symbols, imports=imports,
                registrations=registrations, tables=tables, dynamic=dynamic)


def normalize(s): return re.sub(r'[-_.]+', '-', s).lower()


def capture():
    if (OUT/'reference-lock.json').exists():
        raise SystemExit('Refusing to replace frozen reference; use --verify. Reference updates require approval.')
    tracked=cmd('git','-C',str(COMFY),'ls-files','-z').split('\0')
    selected=sorted(x for x in tracked if x and (x in ROOT_FILES or x.startswith(PREFIXES))
                    and (Path(x).suffix in CODE or x in ROOT_FILES))
    manifest=[]; inventory=[]
    for rel in selected:
        p=COMFY/rel
        if p.is_symlink(): raise SystemExit(f'Unexpected source symlink: {rel}')
        manifest.append(dict(root='comfy', path=rel, size=p.stat().st_size, sha256=digest(p)))
        if p.suffix=='.py': inventory.append(scan(rel,p.read_text(encoding='utf-8-sig')))
    print(f'Scanned {len(selected)} Comfy source files',flush=True)
    # Distribution metadata only: never execute pip, site hooks or reference Python.
    packages={}; owners={}
    for d in sorted(SITE.glob('*.dist-info')):
        meta=d/'METADATA'
        if not meta.is_file(): continue
        m=email.parser.Parser().parsestr(meta.read_text())
        name=normalize(m['Name'])
        records=[]
        if (d/'RECORD').is_file():
            records=[row[0] for row in csv.reader((d/'RECORD').read_text().splitlines()) if row]
        packages[name]=dict(name=m['Name'],version=m['Version'],metadata_dir=d.name,
                            requires=m.get_all('Requires-Dist',[]),
                            metadata_sha256=digest(meta),records=records)
        for rel in records:
            if not rel.startswith(('../','/')):
                top=rel.split('/')[0].split('.')[0]
                if not top.endswith('-info'): owners.setdefault(top,set()).add(name)
    kitchen=[]
    for rel in packages['comfy-kitchen']['records']:
        p=SITE/rel
        if rel.startswith('comfy_kitchen/') and p.suffix in CODE and p.is_file():
            kitchen.append(dict(root='site',path=rel,size=p.stat().st_size,sha256=digest(p)))
            if p.suffix=='.py': inventory.append(scan(rel,p.read_text(encoding='utf-8-sig')))
    manifest+=kitchen
    imports=sorted({x['module'].split('.')[0] for f in inventory for x in f['imports'] if not x['module'].startswith('.')})
    required=set(); unresolved=[]
    for name in imports:
        if name in owners: required.update(owners[name])
        elif name not in sys.stdlib_module_names and name not in {'comfy','comfy_extras','comfy_api','comfy_api_nodes','comfy_execution','comfy_config','comfy_types','nodes','node_helpers','folder_paths','latent_preview','cuda_malloc'}:
            unresolved.append(name)
    # Requirements are captured conservatively, including marker/extra alternatives.
    declared=[]
    for line in (COMFY/'requirements.txt').read_text().splitlines():
        m=re.match(r'^([A-Za-z0-9][A-Za-z0-9_.-]*)',line)
        if m:
            name=normalize(m[1]);declared.append(dict(requirement=line,name=name,installed=name in packages))
            if name in packages: required.add(name)
    required.add('comfy-kitchen')
    pending=list(required); missing=set()
    while pending:
        name=pending.pop()
        for spec in packages[name]['requires']:
            m=re.match(r'^([A-Za-z0-9_.-]+)',spec)
            if not m: continue
            dep=normalize(m[1])
            if dep in packages and dep not in required: required.add(dep);pending.append(dep)
            elif dep not in packages: missing.add(dep)
    print(f'Fingerprinting code for conservative closure: {len(required)} distributions',flush=True)
    dep_files={}; missing_files=[]
    for name in sorted(required):
        pkg=packages[name]
        for rel in pkg['records']:
            # Strict package-code allowlist: no data/model/image payloads, no escaping SITE.
            if rel.startswith(('../','/')): continue
            p=SITE/rel
            if p.suffix not in CODE and p.name not in {'WHEEL','RECORD','METADATA','CMakeLists.txt'}: continue
            if not p.is_file(): missing_files.append(rel);continue
            if not p.resolve().is_relative_to(SITE.resolve()): continue
            if rel not in dep_files:
                dep_files[rel]=dict(root='site',path=rel,size=p.stat().st_size,sha256=digest(p))
    for v in packages.values(): v.pop('records')
    save('dependencies.json',dict(policy='conservative static union; markers/extras NOT evaluated; presence is not activation',
                                declared=declared, distributions=packages,
                                code_fingerprinted_distributions=sorted(required),
                                unmatched_import_roots=unresolved,missing_requirement_alternatives=sorted(missing),
                                missing_record_files=sorted(set(missing_files))))
    all_files={(x['root'],x['path']):x for x in manifest+list(dep_files.values())}
    manifest=sorted(all_files.values(),key=lambda x:(x['root'],x['path']))
    save('source-manifest.json',manifest)
    save('inference-inventory.json',inventory)
    # Freeze exact permitted Comfy and Kitchen source bytes, not model/runtime files.
    with tarfile.open(OUT/'reference-source.tar.gz','w:gz') as tar:
        for x in manifest:
            if x['root']=='comfy' or (x['path'].startswith('comfy_kitchen/') and Path(x['path']).suffix!='.so'):
                p=(COMFY if x['root']=='comfy' else SITE)/x['path']
                tar.add(p,arcname=x['root']+'/'+x['path'],recursive=False)
    torch_vars={}
    for n in ast.parse((SITE/'torch/version.py').read_text()).body:
        if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant):
            torch_vars.update({expr(t):n.value.value for t in n.targets})
    env=dict(os_release=Path('/etc/os-release').read_text(),
             installed_packages=cmd('pacman','-Q','linux','nvidia-utils','cuda','gcc','glibc','python','rustup'),
             gpu='NVIDIA GeForce RTX 4070 Ti SUPER (user specified; not probed)',
             torch_source_version=torch_vars,
             note='Installed package identities only; no driver/device initialization, runtime settings or GPU workload inspected.')
    save('environment.json',env)
    artifacts=['source-manifest.json','inference-inventory.json','dependencies.json','environment.json','reference-source.tar.gz']
    save('reference-lock.json',dict(schema=1,scope='base inference; no graphs/UI; initial Arch Linux + RTX 4070 Ti SUPER',
                                  comfy_revision=cmd('git','-C',str(COMFY),'rev-parse','HEAD'),
                                  roots=dict(comfy=str(COMFY),site=str(SITE)),
                                  kitchen_version=packages['comfy-kitchen']['version'],
                                  torch_version=packages['torch']['version'],
                                  source_count=len(manifest),
                                  artifacts={x:digest(OUT/x) for x in artifacts},
                                  capture_script_sha256=digest(Path(__file__)),
                                  activation='not observed; static source predicates only',
                                  production_rust_libtorch_kitchen_build='not implemented; no build tuple accepted'))
    print(f'Captured {len(manifest)} fingerprints, {len(inventory)} parsed modules',flush=True)


def verify():
    lock=json.loads((OUT/'reference-lock.json').read_text())
    for name,expected in lock['artifacts'].items():
        assert digest(OUT/name)==expected,name
    assert digest(Path(__file__))==lock['capture_script_sha256'],'capture script changed'
    assert cmd('git','-C',str(COMFY),'rev-parse','HEAD')==lock['comfy_revision'],'reference HEAD changed'
    manifest=json.loads((OUT/'source-manifest.json').read_text())
    for x in manifest:
        p=Path(lock['roots'][x['root']])/x['path']
        assert digest(p)==x['sha256'],str(p)
    # Verify preserved source bytes without extracting or importing anything.
    with tarfile.open(OUT/'reference-source.tar.gz','r:gz') as tar:
        expected={x['root']+'/'+x['path']:x['sha256'] for x in manifest}
        for entry in tar:
            assert entry.isfile() and entry.name in expected,entry.name
            assert hashlib.sha256(tar.extractfile(entry).read()).hexdigest()==expected[entry.name],entry.name
    print(f'PASS: artifact lock, source archive and {len(manifest)} current file fingerprints match.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['capture','verify'])
    args=parser.parse_args()
    capture() if args.mode=='capture' else verify()
