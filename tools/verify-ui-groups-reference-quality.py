#!/usr/bin/env python3
"""Compare actual CPU worker geometry with unchanged Director source, no media."""
import argparse,ast,hashlib,json,os,subprocess
from pathlib import Path
OWN=Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')
SOURCE=Path('/home/thhel/git/comfyui-helto-director/shared/h3')
def main():
    p=argparse.ArgumentParser();p.add_argument('probe',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    assert a.probe.is_relative_to(OWN) and a.output.is_relative_to(OWN)
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(OWN)]))['filesystems'][0]
    assert mount['source']=='/dev/mapper/cryptroot[/@home]' and not os.statvfs(OWN).f_flag&os.ST_RDONLY
    # Explicit, already approved Standard/High/Extra High test budgets.
    # Do not read Director's configuration; only its pure geometry function.
    scope={'H3_REFERENCE_QUALITY_PIXELS':{'Standard':1_000_000,'High':2_000_000,'Extra High':4_000_000}}
    path=SOURCE/'reference_quality.py';code=path.read_text();nodes=[n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='reference_image_geometry'];assert len(nodes)==1
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    results=[]
    for case in json.loads(a.probe.read_text())['cases']:
        expected=scope['reference_image_geometry'](case['input_width'],case['input_height'],case['quality'])
        for key in ['content_width','content_height','width','height']:assert case[key]==expected[key]
        results.append({**case,'exact':True})
    report={'passed':True,'scope':'Geometry at explicit approved budgets; no configuration read, tensor resize/padding pixels or GPU claim','budgets':scope['H3_REFERENCE_QUALITY_PIXELS'],'cases':results,'sources':[{'path':str(path),'sha256':hashlib.sha256(code.encode()).hexdigest()}]}
    with a.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({'passed':True,'cases':len(results),'output':str(a.output)}))
if __name__=='__main__':main()
