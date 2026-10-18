#!/usr/bin/env python3
"""Development-only transcription of the pinned MXFP8 and integer helper bodies."""
from reference_paths import reference_dir
import argparse,ast,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--quantization',type=Path,required=True);p.add_argument('--standard',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
project=Path(__file__).resolve().parents[2]
manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text());hashes={e['path']:e['sha256']for e in manifest if e['root']=='site'}
paths={'quantization':a.quantization,'standard':a.standard}
for name,path in paths.items():
 relative='comfy_kitchen/backends/triton/quantization.py'if name=='quantization'else'triton/language/standard.py'
 assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[relative]
modules={k:ast.parse(p.read_text())for k,p in paths.items()}
functions={k:next(n for n in modules[group].body if isinstance(n,ast.FunctionDef)and n.name==name)for k,group,name in [('main','quantization','quantize_mxfp8_kernel_tl'),('swizzle','quantization','_compute_swizzled_scale_offset'),('cdiv','standard','cdiv')]}
out=['// Generated from the locked MXFP8 and integer helper source; no saved IR.'];next_id=0
scalar_names=[n.arg for n in functions['main'].args.args[3:9]]
out.append('pub const SCALAR_NAMES:[&str;6]='+json.dumps(scalar_names)+';')
def emit(s):out.append(s)
def pos(n):return f'({n.lineno},{n.col_offset})'
def lit(s):return json.dumps(s)
def tmp(code):
 global next_id
 name=f't{next_id}';next_id+=1;emit(f'let {name}={code};');return name

def literal(n):
 if isinstance(n,ast.Constant):return n.value
 if isinstance(n,ast.UnaryOp)and isinstance(n.op,ast.USub):return-literal(n.operand)
 if isinstance(n,ast.BinOp)and isinstance(n.op,ast.Pow):return literal(n.left)**literal(n.right)
 raise ValueError('Not a fixed literal')

def expr(n,name):
 if isinstance(n,ast.Name):return n.id+'.clone()'
 if isinstance(n,ast.Constant):return f'Value::number(f64::from_bits({__import__("struct").unpack(">Q",__import__("struct").pack(">d",n.value))[0]}))'if isinstance(n.value,float)else f'Value::integer({int(n.value)})'
 if isinstance(n,ast.UnaryOp):
  if isinstance(n.operand,ast.Name)and n.operand.id=='fp8_max':return 'Value::number(-448.0)'
  return expr(ast.copy_location(ast.Constant(literal(n)),n),name)
 if isinstance(n,ast.BinOp)and isinstance(n.op,ast.Pow):return expr(ast.copy_location(ast.Constant(literal(n)),n),name)
 if isinstance(n,(ast.BinOp,ast.Compare)):
  left=n.left;right=n.right if isinstance(n,ast.BinOp)else n.comparators[0];op=n.op if isinstance(n,ast.BinOp)else n.ops[0]
  op={ast.Add:'add',ast.Sub:'sub',ast.Mult:'mul',ast.Div:'true_div',ast.FloorDiv:'div',ast.Mod:'rem',ast.Eq:'eq',ast.Lt:'lt'}[type(op)]
  a=expr(left,name);b=expr(right,name);return tmp(f'b.bin({lit(op)},{a},{b},{lit(name)},{pos(right)})')
 if isinstance(n,ast.BoolOp):
  assert isinstance(n.op,ast.And)and len(n.values)==2
  a=expr(n.values[0],name);c=expr(n.values[1],name);return tmp(f'b.bin("and",{a},{c},{lit(name)},{pos(n.values[1])})')
 if isinstance(n,ast.Call):
  if isinstance(n.func,ast.Attribute)and n.func.attr=='to':
   a=expr(n.func.value,name);target=n.args[-1].attr;ty={'float32':'Ty::Float(Element::F32)','int32':'Ty::I32','uint8':'Ty::U8','float8e4nv':'Ty::Float(Element::E4M3)'}[target]
   return tmp(f'b.cast({a},{ty},{lit(name)},{pos(n.args[-1])})')
  if isinstance(n.func,ast.Name):
   assert n.func.id=='_compute_swizzled_scale_offset';xs=[expr(x,name)for x in n.args]
   return tmp(f'b.swizzle(vec![{",".join(xs)}],s.source_path,{lit(name)},{pos(n.args[-1])})')
  assert isinstance(n.func,ast.Attribute)
  op=n.func.attr
  if op=='program_id':
   axis=next(k.value.value for k in n.keywords if k.arg=='axis');return tmp(f'b.emit("tt.get_program_id {"xyz"[axis]} : i32".into(),Ty::I32,false,{lit(name)},{pos(n.keywords[-1])})')
  if op=='arange':return tmp(f'b.emit(format!("tt.make_range {{{{end = {{}} : i32, start = 0 : i32}}}} : tensor<{{}}xi32>",s.block_size,s.block_size),Ty::I32,true,{lit(name)},{pos(n.args[-1])})')
  args=[expr(x,name)for x in n.args]
  if op=='load':
   mask=expr(next(k.value for k in n.keywords if k.arg=='mask'),name)
   return tmp(f'b.load({args[0]},{mask},{lit(name)},{pos(n.args[-1])})')
  if op=='store':
   mask=next((k.value for k in n.keywords if k.arg=='mask'),None)
   mask='None'if mask is None else f'Some({expr(mask,name)})'
   emit(f'b.store_optional({args[0]},{args[1]},{mask},{pos(n.args[-1])});');return''
  if op=='where':return tmp(f'b.select({",".join(args)},{lit(name)},{pos(n.args[-1])})')
  if op in ['minimum','maximum']:return tmp(f'b.extreme({lit(op)},{",".join(args)},{lit(name)},{pos(n.args[-1])})')
  if op in ['abs','log2','ceil','exp2']:
   return tmp(f'b.math({lit("absf"if op=="abs"else op)},{args[0]},{lit(name)},{pos(n.args[-1])})')
  if op=='max':return tmp(f'b.reduce_max({args[0]},s.standard_source_path,{lit(name)},{pos(n.args[-1])})')
  if op=='cdiv':return tmp(f'b.cdiv({args[0]},{args[1]},s.standard_source_path,{lit(name)},{pos(n.args[-1])})')
  raise ValueError(ast.dump(n))
 raise ValueError(ast.dump(n))

def statement(n):
 if isinstance(n,(ast.Assign,ast.AnnAssign)):
  target=n.target if isinstance(n,ast.AnnAssign)else n.targets[0];name=target.id;a=expr(n.value,name)
  if isinstance(n,ast.AnnAssign):emit(f'let {name}={a};')
  else:emit(f'let {name}=b.tensor({a},{lit(name)},{pos(n.value)});')
 elif isinstance(n,ast.Expr):
  if not isinstance(n.value,ast.Constant):expr(n.value,'')
 elif isinstance(n,ast.If):
  cond=expr(n.test,'');marker=tmp(f'b.begin_if({cond},{pos(n.test)})')
  emit('{')
  for v in n.body:statement(v)
  emit('}');emit(f'b.end_if({marker});')
 elif isinstance(n,ast.For):
  assert len(n.iter.args)==1
  marker=tmp(f'b.begin_for(i64::from(s.blocks_per_program),"{n.target.id}",{pos(n.iter.args[-1])})');emit('{')
  emit(f'let {n.target.id}={marker}.1.clone();')
  for v in n.body:statement(v)
  emit('}');emit(f'b.end_for({marker}.0);')
 elif isinstance(n,ast.Return):
  v=expr(n.value,'');emit(f'let result={v};');emit(f'if result.constexpr{{b.line("tt.return".into(),"",{pos(n.value)});}}else{{b.line(format!("tt.return {{}} : {{}}",result.id,result.typed(b.block)),"",{pos(n.value)});}}');emit('result')
 else:raise ValueError(ast.dump(n))

emit('fn lower_main(b:&mut Builder,s:&Specialization<\'_>,pointers:&[Value],scalars:&[Value]){')
for i,n in enumerate(functions['main'].args.args[:3]):emit(f'let {n.arg}=pointers[{i}].clone();')
for i,n in enumerate(scalar_names):emit(f'let {"_m"if n=="m"else n}=scalars[{i}].clone();')
emit('let block_size=Value::integer(i64::from(s.block_size));let blocks_per_program=Value::integer(i64::from(s.blocks_per_program));')
for n in functions['main'].body:statement(n)
emit(f'b.line("tt.return".into(),"",{pos(functions["main"].body[-1])});');emit('}')
for name,args in [('cdiv',['x','div']),('swizzle',['in_row','in_col','n_col_blocks','padded_scale_cols'])]:
 emit(f'fn lower_{name}(b:&mut Builder,values:&[Value])->Value{{')
 for i,arg in enumerate(args):emit(f'let {arg}=values[{i}].clone();')
 for n in functions[name].body:statement(n)
 emit('}')
a.output.write_text('\n'.join(out)+'\n');print('Generated locked MXFP8 typed expressions')
