#!/usr/bin/env python3
"""Locked original kernel excerpts/AST positions for cold scalar type errors."""
import argparse,ast,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--check',action='store_true');a=p.parse_args();project=Path(__file__).resolve().parents[2];manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text());out=['// Generated from the locked original kernel source, never runtime data.']
for tag,module,name,epsilon in [('ADALN','adaln','_adaln_fwd_kernel','eps'),('RMS_ROPE','rms_rope','rms_rope_kernel','epsilon')]:
 relative=f'comfy_kitchen/backends/triton/{module}.py';data=(a.site/relative).read_bytes();digest=hashlib.sha256(data).hexdigest();assert digest==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==relative);text=data.decode();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name==name);positions=sorted((n.lineno,n.col_offset)for n in ast.walk(node)if isinstance(n,ast.BinOp)and isinstance(n.op,ast.Add)and isinstance(n.right,ast.Name)and n.right.id==epsilon)
 out += [f'// {relative}: {digest}',f'pub(crate) const {tag}_SOURCE:&str={json.dumps(ast.get_source_segment(text,node),ensure_ascii=False)};',f'pub(crate) const {tag}_DEFINITION:u32={node.lineno};',f'pub(crate) const {tag}_EPSILON_POSITIONS:&[(u32,u32)]=&{positions!r};'.replace('[(','[(').replace(')]',')]')]
source='\n'.join(out)+'\n';source=subprocess.run(['rustfmt','--edition','2024','--emit','stdout'],input=source,text=True,capture_output=True,check=True).stdout
if a.check:assert a.output.read_text()==source,'Locked scalar error source/positions changed'
else:a.output.write_text(source)
