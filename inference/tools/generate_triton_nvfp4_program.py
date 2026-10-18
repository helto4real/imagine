#!/usr/bin/env python3
"""Development-only typed transcription of locked NVFP4 kernel expressions."""
from reference_paths import reference_dir
import argparse,ast,hashlib,json,struct
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--family',choices=['quantize','dequantize'],required=True);a=p.parse_args()
project=Path(__file__).resolve().parents[2];manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text());expected=next(e['sha256']for e in manifest if e['root']=='site'and e['path']=='comfy_kitchen/backends/triton/quantization.py');assert hashlib.sha256(a.source.read_bytes()).hexdigest()==expected
module=ast.parse(a.source.read_text());fn=next(n for n in module.body if isinstance(n,ast.FunctionDef)and n.name==a.family+'_nvfp4_kernel_tl');out=['// Generated from the locked NVFP4 source; no saved IR.',f'// Source SHA256: {expected}'];next_id=0
scalars=[n.arg for n in fn.args.args[4:fn.args.args.index(next(n for n in fn.args.args if n.arg=='block_size'))]]
out.append('pub const SCALAR_NAMES: [&str;'+str(len(scalars))+'] = '+json.dumps(scalars)+';')
def emit(s):out.append(s)
def pos(n):return f'({n.lineno},{n.col_offset})'
def lit(s):return json.dumps(s)
def tmp(code):
 global next_id
 name=f't{next_id}';next_id+=1;emit(f'let {name}={code};');return name

def expr(n,name):
 if isinstance(n,ast.Name):return n.id+'.clone()'
 if isinstance(n,ast.Constant):
  if isinstance(n.value,float):return f'Value::number(f64::from_bits({struct.unpack(">Q",struct.pack(">d",n.value))[0]}))'
  return f'Value::integer({int(n.value)})'
 if isinstance(n,(ast.BinOp,ast.Compare)):
  left=n.left;right=n.right if isinstance(n,ast.BinOp)else n.comparators[0];op=n.op if isinstance(n,ast.BinOp)else n.ops[0]
  op={ast.Add:'add',ast.Sub:'sub',ast.Mult:'mul',ast.Div:'true_div',ast.FloorDiv:'div',ast.Mod:'rem',ast.Eq:'eq',ast.Lt:'lt',ast.BitAnd:'and',ast.RShift:'shr'}[type(op)]
  x=expr(left,name);y=expr(right,name);return tmp(f'b.bin({lit(op)},{x},{y},{lit(name)},{pos(right)})')
 if isinstance(n,ast.BoolOp):
  assert isinstance(n.op,ast.And)and len(n.values)==2
  x=expr(n.values[0],name);y=expr(n.values[1],name);return tmp(f'b.bin("and",{x},{y},{lit(name)},{pos(n.values[1])})')
 if isinstance(n,ast.Subscript):
  assert isinstance(n.slice,ast.Tuple)and len(n.slice.elts)==2 and isinstance(n.slice.elts[0],ast.Slice)and n.slice.elts[1].value is None
  x=expr(n.value,name);return tmp(f'b.expand({x},1,{lit(name)},{pos(n.slice.elts[0])})')
 if isinstance(n,ast.Call):
  # Triton visits keyword values before positional arguments, and visits bound
  # receivers before either. The location follows the final visited argument.
  if isinstance(n.func,ast.Attribute)and n.func.attr in ['to','cast']:
   x=expr(n.func.value,name);target=n.args[0].attr;ty={'float32':'Ty::Float(Element::F32)','float16':'Ty::Float(Element::F16)','uint8':'Ty::U8','uint16':'Ty::U16','uint32':'Ty::U32','float8e4nv':'Ty::Float(Element::E4M3)'}[target]
   return tmp(f'b.{"bitcast"if n.keywords else"cast"}({x},{ty},{lit(name)},{pos(n.args[-1])})')
  if isinstance(n.func,ast.Name):
   assert n.func.id=='_compute_swizzled_scale_offset';xs=[expr(x,name)for x in n.args];return tmp(f'b.swizzle(vec![{",".join(xs)}],s.source_path,{lit(name)},{pos(n.args[-1])})')
  assert isinstance(n.func,ast.Attribute);op=n.func.attr
  if op=='inline_asm_elementwise':
   kws={k.arg:k.value for k in n.keywords};xs=[expr(x,name)for x in kws['args'].elts];dtype={'uint16':'Ty::U16','uint32':'Ty::U32'}[kws['dtype'].attr]
   return tmp(f'b.inline_asm({lit(kws["asm"].value)},{lit(kws["constraints"].value)},vec![{",".join(xs)}],{dtype},{kws["pack"].value},{lit(name)},{pos(n.keywords[-1])})')
  if op=='program_id':
   axis=n.args[0].value if n.args else next(k.value.value for k in n.keywords if k.arg=='axis');point=n.args[-1]if n.args else n.keywords[-1]
   return tmp(f'b.emit("tt.get_program_id {"xyz"[axis]} : i32".into(),Ty::I32,false,{lit(name)},{pos(point)})')
  if op=='arange':
   end=expr(n.args[-1],name);return tmp(f'b.range({end},{lit(name)},{pos(n.args[-1])})')
  kw={k.arg:expr(k.value,name) for k in n.keywords if k.arg in ['mask','other']};xs=[expr(x,name)for x in n.args];point=n.args[-1]
  if op=='load':return tmp(f'b.load_legacy({xs[0]},{"Some("+kw["mask"]+")"if"mask"in kw else"None"},{"Some("+kw["other"]+")"if"other"in kw else"None"},{lit(name)},{pos(point)})')
  if op=='store':emit(f'b.store_optional({xs[0]},{xs[1]},{"Some("+kw["mask"]+")"if"mask"in kw else"None"},{pos(point)});');return''
  if op=='where':return tmp(f'b.select({",".join(xs)},{lit(name)},{pos(point)})')
  if op in ['minimum','maximum']:return tmp(f'b.extreme({lit(op)},{",".join(xs)},{lit(name)},{pos(point)})')
  if op=='abs':return tmp(f'b.math("absf",{xs[0]},{lit(name)},{pos(point)})')
  if op=='max':return tmp(f'b.reduce_max({xs[0]},s.standard_source_path,{lit(name)},{pos(point)})')
  if op=='sum':return tmp(f'b.reduce_nv_sum({xs[0]},s.standard_source_path,{lit(name)},{pos(point)})')
  if op=='cdiv':return tmp(f'b.cdiv({xs[0]},{xs[1]},s.standard_source_path,{lit(name)},{pos(point)})')
 raise ValueError(ast.dump(n))

def statement(n):
 if isinstance(n,(ast.Assign,ast.AnnAssign)):
  target=n.target if isinstance(n,ast.AnnAssign)else n.targets[0]
  if isinstance(target,ast.Tuple):
   assert isinstance(n.value,ast.Tuple)
   for t,v in zip(target.elts,n.value.elts):emit(f'let {t.id}={expr(v,t.id)};')
   return
  name=target.id;x=expr(n.value,name);emit(f'let {name}='+ (x if isinstance(n,ast.AnnAssign)else f'b.tensor({x},{lit(name)},{pos(n.value)})')+';')
 elif isinstance(n,ast.Expr):
  if not isinstance(n.value,ast.Constant):expr(n.value,'')
 elif isinstance(n,ast.If):
  if isinstance(n.test,ast.Name)and n.test.id=='hi_first':
   names=['asm_arg_hi','asm_arg_lo']if a.family=='quantize'else['result_even','result_odd'];pair=','.join(names);emit(f'let ({pair})=if s.hi_first{{')
   for v in n.body:statement(v)
   emit(f'({pair})'+'}else{')
   for v in n.orelse:statement(v)
   emit(f'({pair})'+'};')
  else:
   condition=expr(n.test,'');marker=tmp(f'b.begin_if({condition},{pos(n.test)})');emit('{')
   for v in n.body:statement(v)
   emit('}');emit(f'b.end_if({marker});')
 elif isinstance(n,ast.For):
  marker=tmp(f'b.begin_for(i64::from(s.blocks_per_program),"{n.target.id}",{pos(n.iter.args[-1])})');emit('{');emit(f'let {n.target.id}={marker}.1.clone();')
  for v in n.body:statement(v)
  emit('}');emit(f'b.end_for({marker}.0);')
 else:raise ValueError(ast.dump(n))
emit('fn lower_main(b:&mut Builder,s:&Specialization<\'_>,pointers:&[Value],scalars:&[Value]){')
for i,n in enumerate(fn.args.args[:4]):emit(f'let {n.arg}=pointers[{i}].clone();')
for i,name in enumerate(scalars):emit(f'let {"_"if name in ["m","padded_scale_rows"]else""}{name}=scalars[{i}].clone();')
emit('let block_size=Value::integer(i64::from(s.block_size));')
emit('let blocks_per_program=Value::integer(i64::from(s.blocks_per_program));'if a.family=='quantize'else'let tile_size=Value::integer(i64::from(s.tile_size));')
for n in fn.body:statement(n)
emit(f'b.line("tt.return".into(),"",{pos(fn.body[-1])});');emit('}')
a.output.write_text('\n'.join(out)+'\n');print(a.family,'NVFP4 typed source generated')
