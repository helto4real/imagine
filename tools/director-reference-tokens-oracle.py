#!/usr/bin/env python3
"""CPU-only reference selection comparison against the actual Director parser."""
import ast
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
OWN=Path('/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e')
SOURCE=Path('/home/thhel/git/comfyui-helto-director/shared/timeline/references.py')

def main():
    destination=Path(sys.argv[1]);assert destination.parent==OWN and destination.resolve()==destination
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(OWN)]))['filesystems'][0]
    assert mount['source']=='/dev/mapper/veracrypt1' and not os.statvfs(OWN).f_flag&os.ST_RDONLY
    destination.mkdir(exist_ok=False)
    nodes=[]
    for node in ast.parse(SOURCE.read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REFERENCE_TAG_RE' for t in node.targets):nodes.append(node)
        if isinstance(node,ast.FunctionDef) and node.name in {'parse_reference_tags','normalize_reference_alias','_finite_float_or_none'}:nodes.append(node)
    assert len(nodes)==4
    scope={'re':re,'math':math,'Any':Any}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),scope)
    cases=['','@keeper','@KEEPER','@keeper_long @keeper','@keeper-long','@keeper.long','@9keeper','@@keeper','mail@keeper','(@keeper)','é@keeper','中@keeper','_@keeper','@keeper[0.5]','@keeper[ @visitor ] @third','@keeper[ @visitor','@keeper[] @keeper','@keeper[NaN]','@a[1_000] @b','@a[hi]@b','@a @B @a','@a\n@b','@a\u0301@b','@a[0.5][@b]']
    for prefix in ['','x','1','_','@',' ','.','-','é','中']:
        for suffix in ['','_long','-long','.long']:
            cases.append(prefix+'@Keeper'+suffix)
    expected=[sorted({tag['alias'] for tag in scope['parse_reference_tags'](case)}) for case in cases]
    actual=json.loads(subprocess.check_output([str(ROOT/'comfyui/h3-video-night-20261021/target/debug/examples/reference_tokens_probe')],input=json.dumps(cases).encode()))
    records=[{'text':text,'director':a,'rust':b,'passed':a==b} for text,a,b in zip(cases,expected,actual,strict=True)]
    result={'passed':all(r['passed'] for r in records),'cases':records,'source':str(SOURCE),'cpu_only':True,'scope':'Alias selection only; strength values and reference replacement are separate requirements'}
    (destination/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps({'passed':result['passed'],'cases':len(records)}))
    raise SystemExit(0 if result['passed'] else 1)

if __name__=='__main__':main()
