#!/usr/bin/env python3
"""Freeze original UNet literal elements and compile its name-map body to Rust."""
import argparse,ast,hashlib,json,marshal,subprocess
from pathlib import Path
from generate_model_detection import Compiler,quoted
from reference_paths import reference_dir
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--comfy',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
source=a.comfy/'comfy/utils.py';digest=hashlib.sha256(source.read_bytes()).hexdigest();manifest=json.loads((reference_dir(Path(__file__).resolve().parents[2])/'source-manifest.json').read_text());assert digest==next(v['sha256']for v in manifest if v['root']=='comfy'and v['path']=='comfy/utils.py')
tree=ast.parse(source.read_text());names=['UNET_MAP_ATTENTIONS','TRANSFORMER_BLOCKS','UNET_MAP_RESNET','UNET_MAP_BASIC'];literal={}
for node in tree.body:
 if isinstance(node,ast.Assign):
  for target in node.targets:
   if isinstance(target,ast.Name)and target.id in names:literal[target.id]=node.value
out=f'// Generated from locked comfy/utils.py; source SHA256: {digest}\n// Literal source order is retained. Hash state and CPython construction run natively.\nuse super::detector_values::*;\nuse crate::{{native::Error,patching::options::Options}};\n'
for name in names:
 value=literal[name]
 if isinstance(value,ast.Set):
  values=[ast.literal_eval(v)for v in value.elts]
  # All-name string interning must not introduce another constant rebuild.
  strings=[s for v in values for s in(v if isinstance(v,tuple)else(v,))];assert all('.'in s for s in strings)
  if isinstance(values[0],tuple):ty='&[(&str,&str)]';items=[f'({quoted(x)},{quoted(y)})'for x,y in values]
  else:ty='&[&str]';items=list(map(quoted,values))
 elif isinstance(value,ast.Dict):
  values=[(ast.literal_eval(k),ast.literal_eval(v))for k,v in zip(value.keys,value.values)];ty='&[(&str,&str)]';items=[f'({quoted(x)},{quoted(y)})'for x,y in values]
 else:raise ValueError(ast.dump(value))
 out+=f'pub(super) const {name}:{ty}=&['+','.join(items)+'];\n'
 if isinstance(value,ast.Set):
  cached=sorted(values,key=marshal.dumps)
  items=[f'({quoted(x)},{quoted(y)})'for x,y in cached]if isinstance(cached[0],tuple)else list(map(quoted,cached))
  out+=f'pub(super) const {name}_MARSHALLED:{ty}=&['+','.join(items)+'];\n'
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='unet_to_diffusers');assert isinstance(fn.body[-1],ast.Return)
c=Compiler('comfy/utils.py');out+='pub(super) fn run(mut cx:Context)->Result<Options,Error>{\n'+c.statements(fn.body[:-1]);tail=c.expr(fn.body[-1].value);out+=(tail[:-1]if tail.endswith('?')else'Ok('+tail+')')+'}\n'
a.output.write_text(out);subprocess.run(['rustfmt','--edition','2024',str(a.output)],check=True);print('Generated original UNet literals and whole name-map body',digest,flush=True)
