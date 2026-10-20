#!/usr/bin/env python3
"""Unmodified copy_nested_dicts: bounded native-container alias graphs/errors."""
import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from reference_paths import reference_dir

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--comfy',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert sys.version_info[:3]==(3,12,13)
path=a.comfy/'comfy/patcher_extension.py'
digest=hashlib.sha256(path.read_bytes()).hexdigest()
manifest=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text())
assert digest==next(v['sha256'] for v in manifest if v['root']=='comfy' and v['path']=='comfy/patcher_extension.py')
tree=ast.parse(path.read_text())
method=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='copy_nested_dicts')
namespace={}
exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)

def graph(roots):
    seen,nodes={},[]
    def visit(value):
        if isinstance(value,(dict,list,tuple)):
            identity=id(value)
            if identity in seen: return {'ref':seen[identity]}
            index=len(nodes);seen[identity]=index;nodes.append(None)
            if isinstance(value,dict): node={'dict':[[visit(k),visit(v)] for k,v in value.items()]}
            else: node={('list' if isinstance(value,list) else 'tuple'):list(map(visit,value))}
            nodes[index]=node
            return {'ref':index}
        return value
    result=list(map(visit,roots))
    return dict(roots=result,nodes=nodes)

cases=[]
for depth in range(6):
    for repeated in [1,2,4]:
        leaf={None:7}
        shared=[leaf,leaf]
        opaque=(leaf,shared)
        root=leaf
        for i in range(depth+1):
            root={(j,):root for j in range(repeated)} | {None:shared,(-1,):opaque}
        initial=graph([root])
        result=namespace['copy_nested_dicts'](root)
        cases.append(dict(initial=initial,result=graph([root,result]),error=None))
for value in [[],[None],(),(None,),None,3,'own text']:
    initial=graph([value]);error=None;result=None
    try: result=namespace['copy_nested_dicts'](value)
    except Exception as exception: error=[type(exception).__name__,str(exception)]
    cases.append(dict(initial=initial,result=graph([value,result]),error=error))
assert 'torch' not in sys.modules
a.output.write_text(json.dumps(dict(schema=1,python_version=list(sys.version_info[:3]),source_sha256=digest,
    trained_model_weights=0,fixture_tensors=0,cases=cases),separators=(',',':'))+'\n')
print(f'Captured {len(cases)} bounded original nested-copy alias/error histories; no Torch or weights')
