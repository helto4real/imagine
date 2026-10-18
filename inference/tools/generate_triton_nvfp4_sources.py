#!/usr/bin/env python3
"""Locked NVFP4 conditional excerpts and AST positions; source data only."""
import argparse,ast,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--check',action='store_true');a=p.parse_args();project=Path(__file__).resolve().parents[2];manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text());relative='comfy_kitchen/backends/triton/quantization.py';data=(a.site/relative).read_bytes();digest=hashlib.sha256(data).hexdigest();assert digest==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==relative);text=data.decode();out=['// Generated from frozen NVFP4 source, never runtime data.',f'// {relative}: {digest}']
for tag,name in [('QUANTIZE','quantize_nvfp4_kernel_tl'),('DEQUANTIZE','dequantize_nvfp4_kernel_tl')]:
 node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name==name);positions=[(n.lineno,n.col_offset)for n in ast.walk(node)if isinstance(n,ast.If)and isinstance(n.test,ast.Name)and n.test.id=='hi_first'];assert len(positions)==1
 out += [f'pub const {tag}_SOURCE:&str={json.dumps(ast.get_source_segment(text,node),ensure_ascii=False)};',f'pub const {tag}_DEFINITION:u32={node.lineno};',f'pub const {tag}_POSITION:(u32,u32)={positions[0]!r};']
source=subprocess.run(['rustfmt','--edition','2024','--emit','stdout'],input='\n'.join(out)+'\n',text=True,capture_output=True,check=True).stdout
if a.check:assert a.output.read_text()==source,'Locked NVFP4 conditional source/positions changed'
else:a.output.write_text(source)
