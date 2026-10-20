#!/usr/bin/env python3
"""Retain frozen class inheritance, literal match predicates and registry order.

Build/development only. Constructors and forward methods are deliberately not
substituted by descriptors; the shared dispatcher requires a native constructor.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from reference_paths import reference_dir

def quoted(value):return json.dumps(value,ensure_ascii=False)
def literal(node):
 match node:
  case ast.Constant(value=None):return 'V::None'
  case ast.Constant(value=v) if isinstance(v,bool):return 'V::Bool('+str(v).lower()+')'
  case ast.Constant(value=v) if isinstance(v,int):return 'V::Int('+str(v)+')'
  case ast.Constant(value=v) if isinstance(v,float):return 'V::Float('+repr(v)+'_f64)'
  case ast.Constant(value=v) if isinstance(v,str):return 'V::Text('+quoted(v)+'.into())'
  case ast.UnaryOp(op=ast.USub(),operand=ast.Constant(value=v)) if isinstance(v,(int,float)):return literal(ast.Constant(value=-v))
  case ast.List(elts=v):return 'V::List(vec!['+','.join(literal(n)for n in v)+'])'
  case ast.Tuple(elts=v):return 'V::Tuple(vec!['+','.join(literal(n)for n in v)+'])'
  case ast.Dict(keys=k,values=v):
   assert all(isinstance(n,ast.Constant)and isinstance(n.value,str)for n in k)
   return 'V::Dict(vec!['+','.join('('+quoted(key.value)+'.into(),'+literal(value)+')'for key,value in zip(k,v))+'])'
 raise ValueError('Unknown registry literal: '+ast.dump(node))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--comfy',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 project=Path(__file__).resolve().parents[2];manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text());trees={};hashes={}
 for relative in ['comfy/supported_models_base.py','comfy/supported_models.py']:
  path=a.comfy/relative;digest=hashlib.sha256(path.read_bytes()).hexdigest();expected=next(n['sha256']for n in manifest if n['root']=='comfy'and n['path']==relative);assert digest==expected,relative
  trees[relative]=ast.parse(path.read_text());hashes[relative]=digest
 base=next(n for n in trees['comfy/supported_models_base.py'].body if isinstance(n,ast.ClassDef)and n.name=='BASE')
 nodes=[base]+[n for n in trees['comfy/supported_models.py'].body if isinstance(n,ast.ClassDef)]
 index={n.name:i for i,n in enumerate(nodes)};rows=[]
 for node in nodes:
  if node.name=='BASE':parent='None'
  else:
   assert len(node.bases)==1;parent_name=node.bases[0].id if isinstance(node.bases[0],ast.Name)else node.bases[0].attr;parent=f'Some({index[parent_name]})';assert index[parent_name]<index[node.name]
   assert not any(isinstance(n,ast.FunctionDef)and n.name=='matches'for n in node.body),'New matches override needs native port'
  assignments={name.id:n.value for n in node.body if isinstance(n,ast.Assign)for name in n.targets if isinstance(name,ast.Name)}
  config='None'if 'unet_config'not in assignments else 'Some(Options::from_value('+literal(assignments['unet_config'])+'))'
  required='None'
  if 'required_keys'in assignments:
   value=assignments['required_keys'];assert isinstance(value,(ast.Dict,ast.Set))
   keys=value.keys if isinstance(value,ast.Dict)else value.elts
   assert all(isinstance(k,ast.Constant)and isinstance(k.value,str)for k in keys)
   assert all('{' not in k.value.replace('{}','')and '}'not in k.value.replace('{}','')and k.value.count('{}')<=1 for k in keys),'New key format requires native binding'
   required='Some(vec!['+','.join(quoted(k.value)+'.into()'for k in keys)+'])'
  rows.append(f'ClassDef{{name:{quoted(node.name)},parent:{parent},own_config:{config},own_required:{required}}}')
 registry=next(n.value for n in trees['comfy/supported_models.py'].body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='models'for t in n.targets));assert isinstance(registry,ast.List)
 order=[index[n.id]for n in registry.elts];assert len(order)==len(set(order))
 text='//! Generated frozen class match data; not model forward implementations.\n'
 text+=''.join(f'//! {name} SHA256 {digest}\n'for name,digest in hashes.items())
 text+='use super::model_registry::ClassDef;\nuse crate::{execution_value::Value as V,patching::options::Options};\n'
 text+='pub(super) fn classes()->Vec<ClassDef>{vec!['+','.join(rows)+']}\n'
 text+='pub(super) const ORDER:&[usize]=&['+','.join(map(str,order))+'];\n'
 a.output.write_text(text);subprocess.run(['rustfmt','--edition','2024',str(a.output)],check=True)
 print('Generated',len(nodes),'class definitions and',len(order),'ordered model registrations',flush=True)
if __name__=='__main__':main()
