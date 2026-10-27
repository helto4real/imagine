#!/usr/bin/env python3
"""Derive semantic indexes from the frozen source archive, without execution."""
import ast
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def build():
    nodes, models, tables, edges, branches, builtins = [], [], [], [], [], []
    counts=Counter()
    with tarfile.open(BASE/'reference-source.tar.gz') as tar:
        for member in tar:
            if not member.name.endswith('.py'): continue
            text=tar.extractfile(member).read().decode('utf-8-sig')
            tree=ast.parse(text); path=member.name
            counts[path.split('/')[1]]+=1
            for n in ast.walk(tree):
                if isinstance(n,ast.ClassDef):
                    methods=[x for x in n.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))]
                    schema=[x for x in methods if x.name in {'define_schema','INPUT_TYPES'}]
                    if schema:
                        nodes.append(dict(id=f'{path}:{n.name}',path=path,line=n.lineno,
                                          class_name=n.name,bases=[ast.unparse(x) for x in n.bases],
                                          contracts=[dict(method=x.name,line=x.lineno,source=ast.unparse(x)) for x in schema],
                                          entrypoints=[dict(name=x.name,line=x.lineno,signature=ast.unparse(x.args)) for x in methods if x.name not in {'define_schema','INPUT_TYPES'}],
                                          class_attributes=[ast.unparse(x) for x in n.body if isinstance(x,(ast.Assign,ast.AnnAssign))],
                                          scope='provider-boundary-review' if '/comfy_api_nodes/' in path else 'retain-inference-semantics; ui-io-fields-reviewed-separately',
                                          implementation='not-started',validation='source-only'))
                    if path=='comfy/comfy/supported_models.py':
                        models.append(dict(name=n.name,line=n.lineno,bases=[ast.unparse(x) for x in n.bases],
                                           attributes=[ast.unparse(x) for x in n.body if isinstance(x,(ast.Assign,ast.AnnAssign))],
                                           methods=[dict(name=x.name,line=x.lineno,signature=ast.unparse(x.args)) for x in methods],
                                           status='inventoried; no weights inspected'))
                if isinstance(n,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
                    targets=n.targets if isinstance(n,ast.Assign) else [n.target]
                    value=n.value
                    if value is None:continue
                    for t in targets:
                        name=ast.unparse(t)
                        if (name in {'models','adapters','adapter_maps','KSAMPLER_NAMES','SAMPLER_NAMES','SCHEDULER_HANDLERS','QUANT_ALGOS','NODE_CLASS_MAPPINGS','extras_files'}
                            or name.startswith(('QUANT_ALGOS[','NODE_CLASS_MAPPINGS['))):
                            tables.append(dict(path=path,line=n.lineno,target=name,expression=ast.unparse(value)))
                            if name=='extras_files' and isinstance(value,ast.List):
                                builtins.extend(ast.literal_eval(value))
            # Retain branch ancestry rather than evaluating hardware/optional imports.
            def walk(n,guards):
                if isinstance(n,(ast.Import,ast.ImportFrom)):
                    edges.append(dict(path=path,line=n.lineno,source=ast.unparse(n),guards=guards))
                if isinstance(n,ast.If):
                    branches.append(dict(path=path,line=n.lineno,test=ast.unparse(n.test)))
                    for child in n.body:walk(child,guards+[f'if {ast.unparse(n.test)}'])
                    for child in n.orelse:walk(child,guards+[f'else of line {n.lineno}'])
                    return
                if isinstance(n,ast.Try):
                    for child in n.body:walk(child,guards+[f'try at line {n.lineno}'])
                    for handler in n.handlers:
                        for child in handler.body:walk(child,guards+[f'except {ast.unparse(handler.type) if handler.type else "any"} at line {handler.lineno}'])
                    for child in n.orelse+n.finalbody:walk(child,guards+[f'try else/finally at line {n.lineno}'])
                    return
                for child in ast.iter_child_nodes(n):walk(child,guards)
            walk(tree,[])
    obj=dict(schema=1,policy='static source union; predicates retained; no runtime activation inferred',
             nodes=nodes,model_classes=models,registry_assignments=tables,import_edges=edges,
             branch_predicates=branches,builtin_extra_node_files=builtins,
             module_counts=dict(counts))
    target=BASE/'capability-index.json'
    target.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
    lock_path=BASE/'reference-lock.json'
    lock=json.loads(lock_path.read_text())
    lock['artifacts'][target.name]=hashlib.sha256(target.read_bytes()).hexdigest()
    lock['artifacts']['tools/index.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    lock_path.write_text(json.dumps(lock,indent=2)+'\n')
    print(f'Indexed {len(nodes)} node contracts, {len(models)} model classes, {len(tables)} registry expressions, {len(edges)} guarded imports, {len(branches)} predicates, {len(builtins)} built-in extra files.')


if __name__=='__main__':build()
