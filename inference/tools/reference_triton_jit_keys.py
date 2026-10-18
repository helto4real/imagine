#!/usr/bin/env python3
"""Both original ASTSource and actual JIT _pack_args keys, without IR or GPU work.

Existing frontend matrices supply typed signatures. Their AST is cut immediately
at construction of ASTSource; no lowering, compiler stages or tensor calls run.
The original JIT binder then derives its own signature/constants/attributes.
"""
from reference_paths import reference_dir
import argparse,ast,hashlib,json,sys,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();project=Path(__file__).resolve().parents[2]
manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text())
for name in ['triton/runtime/jit.py','triton/compiler/compiler.py']:
 assert hashlib.sha256((a.site/name).read_bytes()).hexdigest()==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name)
sys.path.insert(0,str(a.site))
from triton.compiler import ASTSource
from triton.runtime.cache import get_cache_key
records=[]
def collect(source,backend,options):
 fn=source.fn;names=fn.arg_names;specialization=[];bound={}
 for i,name in enumerate(names):
  kind=source.signature[name];constant=source.constants.get((i,));bound[name]=constant
  attr='D'if source.attrs.get((i,))else''
  specialization.append((kind,constant if kind=='constexpr'else attr if kind.startswith('*')or kind in ['i32','i64','u32','u64']else None))
 _,signature,constants,attrs=fn._pack_args(backend,{},bound,specialization,options)
 jit=ASTSource(fn,signature,constants,attrs)
 sorted_sig=[v for k,v in sorted(source.signature.items())]
 get_key=lambda x:x.cache_key if hasattr(x,'cache_key')else str(x)
 constants_key='-'.join(get_key(v)for _,v in sorted(source.constants.items()))
 key=f'{fn.cache_key}-{source.attrs}-{sorted_sig}-{constants_key}'
 assert hashlib.sha256(key.encode()).hexdigest()==source.hash()
 records.append(dict(function=fn.__name__,argument_names=names,explicit_source_key=key,explicit_ast_digest=source.hash(),jit_ast_digest=jit.hash(),jit_compiler_digest=hashlib.sha256(get_cache_key(jit,backend,options,{}).encode()).hexdigest(),options_hash=options.hash()))
class Cut(ast.NodeTransformer):
 found=False
 def visit_For(self,node):
  for i,stmt in enumerate(node.body):
   if isinstance(stmt,ast.Assign)and isinstance(stmt.value,ast.Call)and isinstance(stmt.value.func,ast.Name)and stmt.value.func.id=='ASTSource':
    name=stmt.targets[0].id;node.body=node.body[:i+1]+ast.parse(f'collect({name},backend,options)').body;self.found=True;return node
  return self.generic_visit(node)
with tempfile.TemporaryDirectory(prefix='imagine-jit-key-source-')as own:
 for family,filename,extra in [('fp8','frontend',[]),('rowwise','rowwise_frontend',[]),('adaln','adaln_frontend',[]),('rope','rope_frontend',[]),('rms_rope','rms_rope_frontend',[]),('mxfp8','block_quant_frontend',['--family','mxfp8']),('nvfp4','block_quant_frontend',['--family','nvfp4']),('dequant_nvfp4','block_quant_frontend',['--family','dequant_nvfp4']),('int8','int8_frontend',[]),('w4a8','w4a8_frontend',[]),('na','na_frontend',[])]:
  path=Path(__file__).with_name('reference_triton_'+filename+'.py');tree=ast.parse(path.read_text());cut=Cut();body=[]
  for stmt in tree.body:
   body.append(cut.visit(stmt))
   if cut.found:break
  assert cut.found;tree.body=body;ast.fix_missing_locations(tree);sys.argv=[str(path),'--site',str(a.site),'--output',str(Path(own)/family),*extra];begin=len(records);exec(compile(tree,str(path),'exec'),dict(__file__=str(path),collect=collect))
  for c in records[begin:]:c['family']=family
  print(family,len(records)-begin,flush=True)
a.output.write_text(json.dumps(dict(triton_jit_keys=records),indent=2)+'\n');print(len(records),'original explicit/JIT key pairs; no lowering or GPU work',flush=True)
