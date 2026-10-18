#!/usr/bin/env python3
"""Development-only source-to-Rust transcription of the locked RMS-RoPE expressions.

The engine consumes compiled Rust, never this generator, Python, source AST,
or captured IR. Type promotion and lowering live in triton_frontend_rope.rs.
"""
import argparse, ast, hashlib, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
project=Path(__file__).resolve().parents[2]
manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text())
expected=next(e['sha256']for e in manifest if e['root']=='site'and e['path']=='comfy_kitchen/backends/triton/rms_rope.py')
assert hashlib.sha256(a.source.read_bytes()).hexdigest()==expected
module=ast.parse(a.source.read_text());main=next(n for n in module.body if isinstance(n,ast.FunctionDef)and n.name=='rms_rope_kernel')
out=['// Generated from locked rms_rope.py by tools/generate_triton_rms_rope_program.py.',f'// Source SHA256: {expected}']
scalar_names=[n.arg for n in main.args.args[4:20]]
out.append('pub const SCALAR_NAMES: [&str;16] = '+json.dumps(scalar_names)+';')
next_id=0

def emit(s):out.append(s)
def pos(n):return f'({n.lineno},{n.col_offset})'
def lit(s):return json.dumps(s)
def tmp(code):
    global next_id
    n=f't{next_id}';next_id+=1;emit(f'let {n} = {code};');return n

def expr(n,name):
    if isinstance(n,ast.Name):return n.id+'.clone()'
    if isinstance(n,ast.Constant):
        assert isinstance(n.value,(int,float))and(not isinstance(n.value,float)or n.value==0)
        return f'Value::integer({int(n.value)})'
    if isinstance(n,(ast.BinOp,ast.Compare)):
        left=n.left;right=n.right if isinstance(n,ast.BinOp)else n.comparators[0]
        op=n.op if isinstance(n,ast.BinOp)else n.ops[0]
        op={ast.Add:'add',ast.Mult:'mul',ast.FloorDiv:'div',ast.Mod:'rem',ast.Div:'true_div',ast.Lt:'lt',ast.Eq:'eq'}[type(op)]
        x=expr(left,name);y=expr(right,name)
        return tmp(f'b.bin({lit(op)},{x},{y},{lit(name)},{pos(right)})')
    if isinstance(n,ast.Call):
        if isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='tl':
            op=n.func.attr
            if op=='program_id':return tmp(f'b.emit("tt.get_program_id {"xyz"[n.args[0].value]} : i32".into(),Ty::I32,false,{lit(name)},{pos(n.args[-1])})')
            if op=='arange':return tmp(f'b.emit(format!("tt.make_range {{{{end = {{}} : i32, start = 0 : i32}}}} : tensor<{{}}xi32>",b.block,b.block),Ty::I32,true,{lit(name)},{pos(n.args[-1])})')
            if op=='sum':
                x=expr(n.args[0],name)
                return tmp(f'b.reduce_sum({x},s.standard_source_path,{lit(name)},{pos(n.args[-1])})')
            if op=='where':
                xs=[expr(x,name)for x in n.args]
                return tmp(f'b.select({",".join(xs)},{lit(name)},{pos(n.args[-1])})')
            if op=='load':
                x=expr(n.args[0],name);mask=expr(next(k.value for k in n.keywords if k.arg=='mask'),name)
                return tmp(f'b.load({x},{mask},{lit(name)},{pos(n.args[-1])})')
            if op=='store':
                xs=[expr(x,name)for x in n.args];mask=expr(next(k.value for k in n.keywords if k.arg=='mask'),name)
                emit(f'b.store({",".join(xs)},{mask},{pos(n.args[-1])});');return ''
        if isinstance(n.func,ast.Attribute)and n.func.attr=='rsqrt':
            x=expr(n.args[0],name)
            return tmp(f'b.emit(format!("math.rsqrt {{}} : {{}}",{x}.id,{x}.typed(b.block)),Ty::Float(Element::F32),false,{lit(name)},{pos(n.args[-1])})')
        if isinstance(n.func,ast.Attribute)and n.func.attr=='to':
            x=expr(n.func.value,name)
            target=n.args[-1]
            ty='s.input'if isinstance(target,ast.Name)and target.id=='norm_dtype'else'b.compute'if isinstance(target,ast.Name)and target.id=='compute_dtype'else'Element::F32'
            return tmp(f'b.cast({x},Ty::Float({ty}),{lit(name)},{pos(target)})')
        raise ValueError(ast.dump(n))
    raise ValueError(ast.dump(n))

def statement(n):
    if isinstance(n,ast.Assign):
        name=n.targets[0].id
        if name=='full_offsets':emit('b.block=s.block_size*2;')
        x=expr(n.value,name)
        emit(f'let {name}=b.tensor({x},{lit(name)},{pos(n.value)});')
        if name=='inv_rms':emit('b.block=s.block_size;')
    elif isinstance(n,ast.Expr)and isinstance(n.value,ast.Constant):pass
    elif isinstance(n,ast.Expr):expr(n.value,'')
    else:raise ValueError(ast.dump(n))

emit('fn lower_rms(b:&mut Builder,s:&Specialization<\'_>,pointers:&[Value],scalars:&[Value],epsilon:Value) {')
for i,n in enumerate(main.args.args[:4]):emit(f'let {n.arg}=pointers[{i}].clone();')
for i,n in enumerate(scalar_names):emit(f'let {n}=scalars[{i}].clone();')
for n in main.body:
    if isinstance(n,ast.If):
        assert isinstance(n.test,ast.Name)and n.test.id=='split_half'
        emit('let(dim_idx_0,dim_idx_1)=if s.split_half {')
        for v in n.body:statement(v)
        emit('(dim_idx_0,dim_idx_1)}else{')
        for v in n.orelse:statement(v)
        emit('(dim_idx_0,dim_idx_1)};')
    else:statement(n)
emit('b.line("tt.return".into(),"",(92,4));')
emit('}')
a.output.write_text('\n'.join(out)+'\n')
