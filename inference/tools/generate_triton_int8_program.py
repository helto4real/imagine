#!/usr/bin/env python3
"""Development-only source transcription of locked integer matmul and grouped-unpack kernels."""
from reference_paths import reference_dir
import argparse,ast,hashlib,json,struct,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--family',choices=['int8','w4a8'],default='int8');p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
project=Path(__file__).resolve().parents[2];manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text());expected=next(e['sha256']for e in manifest if e['root']=='site'and e['path']==('comfy_kitchen/backends/triton/quantization.py'if a.family=='int8'else'comfy_kitchen/backends/triton/w4a8_int8.py'));assert hashlib.sha256(a.source.read_bytes()).hexdigest()==expected
module=ast.parse(a.source.read_text());out=['// Generated from the locked INT8 matmul source; no saved IR.'if a.family=='int8'else'// Generated from locked integer-kernel source; no saved IR.',f'// Source SHA256: {expected}'];next_id=0;last_position=None;origins={}
functions=[next(n for n in module.body if isinstance(n,ast.FunctionDef)and n.name==name)for name in (['_int8_matmul_dequant_kernel','_int8_matmul_dequant_per_row_kernel']if a.family=='int8'else['_dequant_int4_grouped_to_int8_kernel'])]
pointer_count=6 if a.family=='int8'else 4
scalars=[n.arg for n in functions[0].args.args[pointer_count:pointer_count+9]]
assert all(scalars==[n.arg for n in fn.args.args[pointer_count:pointer_count+9]]for fn in functions)
out.append('pub const SCALAR_NAMES:[&str;9]='+json.dumps(scalars)+';')
def emit(s):out.append(s)
def pos(n):return f'({n.lineno},{n.col_offset})'
def lit(s):return json.dumps(s)
def tmp(code):
 global next_id,last_position
 name=f't{next_id}';next_id+=1;found=re.search(r',\((\d+),(\d+)\)\)$',code);last_position=tuple(map(int,found.groups()))if found else None;emit(f'let {name}={code};');return name

def expr(n,name):
 if isinstance(n,ast.UnaryOp):
  assert isinstance(n.op,ast.USub)and isinstance(n.operand,ast.Constant);return expr(ast.copy_location(ast.Constant(value=-n.operand.value),n),name)
 if isinstance(n,ast.Name):return n.id+'.clone()'
 if isinstance(n,ast.Constant):
  if isinstance(n.value,float):return f'Value::number(f64::from_bits({struct.unpack(">Q",struct.pack(">d",n.value))[0]}))'
  return f'Value::integer({int(n.value)})'
 if isinstance(n,(ast.BinOp,ast.Compare)):
  left=n.left;right=n.right if isinstance(n,ast.BinOp)else n.comparators[0];op=n.op if isinstance(n,ast.BinOp)else n.ops[0]
  op={ast.Add:'add',ast.Sub:'sub',ast.Mult:'mul',ast.Div:'true_div',ast.FloorDiv:'div',ast.Mod:'rem',ast.Eq:'eq',ast.Lt:'lt',ast.BitAnd:'and',ast.RShift:'shr',ast.LShift:'shl',ast.BitOr:'or'}[type(op)]
  x=expr(left,name);y=expr(right,name);return tmp(f'b.bin({lit(op)},{x},{y},{lit(name)},{pos(right)})')
 if isinstance(n,ast.BoolOp):
  assert isinstance(n.op,ast.And)and len(n.values)==2
  x=expr(n.values[0],name);y=expr(n.values[1],name);return tmp(f'b.bin("and",{x},{y},{lit(name)},{pos(n.values[1])})')
 if isinstance(n,ast.Subscript):
  assert isinstance(n.slice,ast.Tuple)and len(n.slice.elts)==2
  axis=0 if isinstance(n.slice.elts[0],ast.Constant)else 1;point=n.slice
  x=expr(n.value,name);return tmp(f'b.expand({x},{axis},{lit(name)},{pos(point)})')
 if isinstance(n,ast.Call):
  # Triton visits keyword values before positional arguments, and visits bound
  # receivers before either. The location follows the final visited argument.
  if isinstance(n.func,ast.Attribute)and n.func.attr in ['to','cast']:
   x=expr(n.func.value,name);target=n.args[0].attr;ty={'int8':'Ty::I8','int32':'Ty::I32','float32':'Ty::Float(Element::F32)','float16':'Ty::Float(Element::F16)','uint8':'Ty::U8','uint16':'Ty::U16','uint32':'Ty::U32','float8e4nv':'Ty::Float(Element::E4M3)'}[target]
   return tmp(f'b.{"bitcast"if n.keywords else"cast"}({x},{ty},{lit(name)},{pos(n.args[-1])})')
  if isinstance(n.func,ast.Name):
   assert n.func.id=='min';xs=[expr(x,name)for x in n.args];return tmp(f'b.extreme("minimum",{xs[0]},{xs[1]},{lit(name)},{pos(n.args[-1])})')
  assert isinstance(n.func,ast.Attribute);op=n.func.attr
  if op=='rint':
   x=expr(n.args[0],name);return tmp(f'b.w4a8_rint({x},{lit(name)},{pos(n.args[-1])})')
  if op=='clamp':
   xs=[expr(x,name)for x in n.args];return tmp(f'b.w4a8_clamp({",".join(xs)},{lit(name)},{pos(n.args[-1])})')
  if op=='zeros':
   assert isinstance(n.args[0],ast.Tuple);shape=[expr(x,name)for x in n.args[0].elts]
   return tmp(f'b.int8_zeros({shape[0]},{shape[1]},s.standard_source_path,{lit(name)},{pos(n.args[0])})')
  if op=='dot':
   xs=[expr(x,name)for x in n.args];return tmp(f'b.int8_dot({xs[0]},{xs[1]},{lit(name)},{pos(n.args[-1])})')
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
 if isinstance(n,ast.Assign):
  assert isinstance(n.targets[0],ast.Name);name=n.targets[0].id;x=expr(n.value,name);emit(f'let {name}=b.tensor({x},{lit(name)},{pos(n.value)});');origins[name]=last_position
 elif isinstance(n,ast.AugAssign):
  assert isinstance(n.op,ast.Add);name=n.target.id;x=expr(n.value,name);emit(f'let {name}=b.bin("add",{name}.clone(),{x},{lit(name)},{pos(n.value)});')
 elif isinstance(n,ast.Expr):
  if not isinstance(n.value,ast.Constant):expr(n.value,'')
 elif isinstance(n,ast.If):
  if a.family=='int8':
   assert n.test.id=='has_bias'and not n.orelse;emit('let c=if s.has_bias {')
   for v in n.body:statement(v)
   emit('c}else{c};')
  else:
   cb=isinstance(n.test,ast.Name);names=['v_low','v_high']if cb else['low','high'];condition='s.has_cb'if cb else's.bits==6';emit(f'let ({",".join(names)})=if {condition}'+'{')
   for v in n.body:statement(v)
   emit('('+','.join(names)+')}else{')
   for v in n.orelse:statement(v)
   emit('('+','.join(names)+')};')
 elif isinstance(n,ast.For):
  lo=expr(n.iter.args[0],'');hi=expr(n.iter.args[1],'');frame=tmp(f'b.begin_int8_loop({lo},{hi},&[a_ptrs.clone(),b_ptrs.clone(),accumulator.clone()],[{','.join(str(origins[key]).replace(' ','')for key in ['a_ptrs','b_ptrs','accumulator'])}],{pos(n.iter.args[-1])})');emit('{')
  emit(f'let {n.target.id}={frame}.iv.clone();')
  for i,name in enumerate(['a_ptrs','b_ptrs','accumulator']):emit(f'let {name}={frame}.arguments[{i}].clone();')
  for v in n.body:statement(v)
  emit(f'b.end_int8_loop(&{frame},&[a_ptrs,b_ptrs,accumulator],{pos(n.body[-1])});');emit('}');emit(f'let accumulator={frame}.results[2].clone();')
 else:raise ValueError(ast.dump(n))
for fn,label in zip(functions,['tensorwise','per_row']if a.family=='int8'else['w4a8']):
 emit(f'fn lower_{label}(b:&mut Builder,s:&Specialization<\'_>,pointers:&[Value],scalars:&[Value])'+'{')
 for i,n in enumerate(fn.args.args[:pointer_count]):emit(f'let {n.arg}=pointers[{i}].clone();')
 for i,name in enumerate(scalars):emit(f'let {name}=scalars[{i}].clone();')
 for name in (['block_m','block_n','block_k','group_size_m']if a.family=='int8'else['block_n','block_kh','bits']):emit(f'let {name}=Value::integer(i64::from(s.{name}));')
 for n in fn.body:statement(n)
 emit(f'b.line("tt.return".into(),"",{pos(fn.body[-1])});');emit('}')
program='\n'.join(out)+'\n';program=re.sub(r'\bb\.(?!clone\b)','builder.',program).replace('b:&mut Builder','builder:&mut Builder');program=re.sub(r'let c=(builder\.tensor\([^;\n]+\));\nc\}else\{c\};',r'\1}else{c};',program);
if a.family=='w4a8':
 program=program.replace('s:&Specialization','spec:&Specialization');program=re.sub(r'\bs\.(has_cb|bits|block_n|block_kh)',r'spec.\1',program)
a.output.write_text(program);print('Generated locked '+a.family+' program')
