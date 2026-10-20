#!/usr/bin/env python3
"""Copy exact built-in scheduler constant tables; no tensor/model imports."""
import argparse,ast,hashlib,json
from pathlib import Path
from reference_paths import reference_dir
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--comfy',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
manifest=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text())
out=dict(schema=1,sources={})
for label,path in [('ays','comfy_extras/nodes_align_your_steps.py'),('gits','comfy_extras/nodes_gits.py')]:
 source=a.comfy/path;expected=next(e['sha256']for e in manifest if e['root']=='comfy'and e['path']==path)
 assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
 out['sources'][path]=expected
 table=next(ast.literal_eval(n.value)for n in ast.parse(source.read_text()).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='NOISE_LEVELS'for t in n.targets))
 out[label]=[[key,value]for key,value in table.items()]
a.output.write_text(json.dumps(out,separators=(',',':'))+'\n')
