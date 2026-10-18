#!/usr/bin/env python3
from reference_paths import reference_dir
import argparse,ast,hashlib,json,re,struct
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();project=Path(__file__).resolve().parents[2];source=a.source;expected=next(e['sha256']for e in json.loads((reference_dir(project)/'source-manifest.json').read_text())if e['root']=='site'and e['path']=='comfy_kitchen/backends/triton/na.py');assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
fn=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='_na3d_kernel');out=['// Generated from locked NA3D source; no saved IR.',f'// Source SHA256: {expected}'];scalars=[n.arg for n in fn.args.args[4:13]];out.append('pub const SCALAR_NAMES:[&str;9]='+json.dumps(scalars)+';');counter=0;last_position=None;origins={}
def emit(s):out.append(s)
def pos(n):return f'({n.lineno},{n.col_offset})'
def lit(s):return json.dumps(s)
def tmp(code):
 global counter,last_position
 name=f't{counter}';counter+=1;match=re.search(r',\((\d+),(\d+)\)\)$',code);last_position=tuple(map(int,match.groups()))if match else None;emit(f'let {name}={code};');return name

def shape(n):
 assert isinstance(n,ast.Tuple);return 'vec!['+','.join(f'{expr(x,"")}.literal.unwrap() as u32'for x in n.elts)+']'
def expr(n,name):
 if isinstance(n,ast.Name):return 'Value::number(-3.0e38)'if n.id=='_NEG_INF'else n.id+'.clone()'
 if isinstance(n,ast.UnaryOp):
  assert isinstance(n.op,ast.USub)and isinstance(n.operand,ast.Constant);return expr(ast.copy_location(ast.Constant(value=-n.operand.value),n),name)
 if isinstance(n,ast.Constant):
  return f'Value::number(f64::from_bits({struct.unpack(">Q",struct.pack(">d",n.value))[0]}))'if isinstance(n.value,float)else f'Value::integer({int(n.value)})'
 if isinstance(n,(ast.BinOp,ast.Compare)):
  left=n.left;right=n.right if isinstance(n,ast.BinOp)else n.comparators[0];op=n.op if isinstance(n,ast.BinOp)else n.ops[0];op={ast.Add:'add',ast.Sub:'sub',ast.Mult:'mul',ast.Div:'true_div',ast.FloorDiv:'div',ast.Mod:'rem',ast.Lt:'lt',ast.GtE:'ge',ast.BitAnd:'and'}[type(op)];x=expr(left,name);y=expr(right,name);return tmp(f'builder.bin({lit(op)},{x},{y},{lit(name)},{pos(right)})')
 if isinstance(n,ast.Subscript):
  assert isinstance(n.slice,ast.Tuple);axis=0 if isinstance(n.slice.elts[0],ast.Constant)else 1;x=expr(n.value,name);return tmp(f'builder.expand({x},{axis},{lit(name)},{pos(n.slice)})')
 if isinstance(n,ast.Call):
  op=n.func.attr
  if op=='to':
   x=expr(n.func.value,name);target=n.args[0];ty='Ty::Float(Element::F32)'if isinstance(target,ast.Attribute)and isinstance(target.value,ast.Name)and target.value.id=='tl'else'v_blk.ty'if isinstance(target,ast.Attribute)and target.attr=='dtype'else'out_ptr.ty.pointee()'
   return tmp(f'builder.cast({x},{ty},{lit(name)},{pos(target)})')
  if op in ['full','zeros']:
   dims=shape(n.args[0]);value=expr(n.args[1],name)if op=='full'else'';point=n.args[-1]
   return tmp(f'builder.na_{op}({dims},{value+","if op=="full"else""}{"spec.standard_source_path,"if op=="zeros"else""}{lit(name)},{pos(point)})')
  if op=='program_id':return tmp(f'builder.emit("tt.get_program_id {"xyz"[n.args[0].value]} : i32".into(),Ty::I32,false,{lit(name)},{pos(n.args[0])})')
  if op=='arange':end=expr(n.args[-1],name);return tmp(f'builder.range({end},{lit(name)},{pos(n.args[-1])})')
  kw={k.arg:expr(k.value,name)for k in n.keywords if k.arg in ['mask','other']};xs=[expr(x,name)for x in n.args];point=n.args[-1]
  if op=='load':return tmp(f'builder.load_legacy({xs[0]},{"Some("+kw["mask"]+")"if"mask"in kw else"None"},{"Some("+kw["other"]+")"if"other"in kw else"None"},{lit(name)},{pos(point)})')
  if op=='store':emit(f'builder.store_optional({xs[0]},{xs[1]},{"Some("+kw["mask"]+")"if"mask"in kw else"None"},{pos(point)});');return''
  if op=='where':return tmp(f'builder.select({",".join(xs)},{lit(name)},{pos(point)})')
  if op in ['minimum','maximum']:return tmp(f'builder.extreme({lit(op)},{",".join(xs)},{lit(name)},{pos(point)})')
  if op=='trans':return tmp(f'builder.na_trans({xs[0]},{lit(name)},{pos(point)})')
  if op=='dot':return tmp(f'builder.na_dot({xs[0]},{xs[1]},{str(bool(n.keywords)).lower()},{lit(name)},{pos(point)})')
  if op=='exp':return tmp(f'builder.math("exp",{xs[0]},{lit(name)},{pos(point)})')
  if op in ['max','sum']:return tmp(f'builder.na_reduce({xs[0]},{str(op=="sum").lower()},spec.standard_source_path,{lit(name)},{pos(point)})')
 raise ValueError(ast.dump(n))
def statement(n,depth=0):
 if isinstance(n,ast.Assign):
  name=n.targets[0].id;x=expr(n.value,name);emit(f'let {name}=builder.tensor({x},{lit(name)},{pos(n.value)});');origins[name]=last_position
 elif isinstance(n,ast.Expr):
  if not isinstance(n.value,ast.Constant):expr(n.value,'')
 elif isinstance(n,ast.If):
  condition=n.test.id;names={'causal_t':['t_lo','t_hi'],'causal_h':['h_lo','h_hi'],'causal_w':['w_start','w_end','w_lo','w_hi'],'is_fp32':[n.body[0].targets[0].id]}[condition];test='spec.input==Element::F32'if condition=='is_fp32'else'spec.'+condition;emit('let ('+','.join(names)+')=if '+test+'{')
  for v in n.body:statement(v,depth)
  emit('('+','.join(names)+')}else{')
  for v in n.orelse:statement(v,depth)
  emit('('+','.join(names)+')};')
 elif isinstance(n,ast.For):
  if depth==0:emit('let origins=['+','.join(f'builder.loc({lit(name)},{str(origins[name]).replace(" ","")})'for name in ['m_i','l_i','acc'])+'];')
  xs=[expr(x,'')for x in n.iter.args];step=xs[2]if len(xs)>2 else'Value::integer(1)';point=n.iter.args[-1];frame=tmp(f'builder.na_begin_loop([{xs[0]},{xs[1]},{step}],{lit(n.target.id)},[m_i.clone(),l_i.clone(),acc.clone()],&origins,{pos(point)})');emit('{');emit(f'let {n.target.id}={frame}.iv.clone();');
  if depth<2:emit(f'let origins={frame}.origins.clone();')
  for i,name in enumerate(['m_i','l_i','acc']):emit(f'let {name}={frame}.arguments[{i}].clone();')
  for v in n.body:statement(v,depth+1)
  last=n.body[-1]
  if isinstance(last,ast.Assign):yield_at=pos(last)
  else:yield_at=pos(last)
  emit(f'builder.na_end_loop(&{frame},[m_i,l_i,acc],{yield_at});');emit('}')
  for i,name in enumerate(['m_i','l_i','acc']):emit(f'let {"_m_i"if depth==0 and name=="m_i"else name}={frame}.results[{i}].clone();')
 else:raise ValueError(ast.dump(n))
out.append('pub const FUNCTION_SOURCE:&str='+json.dumps(ast.get_source_segment(source.read_text(),fn),ensure_ascii=False)+';')
out.append(f'pub const FUNCTION_DEFINITION:u32={fn.lineno};')
inner=next(n for n in ast.walk(fn)if isinstance(n,ast.For)and n.target.id=='wk0')
conditions={n.body[0].targets[0].id:n for n in ast.walk(fn)if isinstance(n,ast.If)and isinstance(n.test,ast.Name)and n.test.id=='is_fp32'}
def dot_in(node):return next(n for n in ast.walk(node)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='dot')
for label,branch in [('IEEE',conditions['s'].body[0]),('DEFAULT',conditions['s'].orelse[0])]:out.append(f'pub const QUERY_DOT_{label}_POSITION:(u32,u32)='+pos(dot_in(branch))+';')
out.append('pub const INNER_LOOP_POSITION:(u32,u32)='+pos(inner)+';');out.append('pub const ACCUMULATOR_DOT_POSITION:(u32,u32)='+pos(dot_in(conditions['acc'].body[0]))+';')
emit("fn lower_na(builder:&mut Builder,spec:&Specialization<'_>,kernels:[Value;3],pointers:&[Value],scalars:&[Value],scale:Value){")
for i,n in enumerate(fn.args.args[:4]):emit(f'let {n.arg}=pointers[{i}].clone();')
for i,name in enumerate(scalars):emit(f'let {name}=scalars[{i}].clone();')
for i,name in enumerate(['kt','kh','kw']):emit(f'let {name}=kernels[{i}].clone();')
for name in ['hd','hd_pad','block_q','block_k']:emit(f'let {name}=Value::integer(i64::from(spec.{name}));')
for n in fn.body:statement(n)
emit(f'builder.line("tt.return".into(),"",{pos(fn.body[-1])});');emit('}')
program='\n'.join(out)+'\n';program=re.sub(r'\((s|acc)\)',r'\1',program)
program=re.sub(r'let (s|acc)=(builder\.tensor\([^;\n]+\));\n\1\}',r'\2}',program)
a.output.write_text(program)
