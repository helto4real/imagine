#!/usr/bin/env python3
"""Generate static, fallible Rust tensor expressions from the locked formulas.

This is a build/development source generator, not a Python interpreter or a
runtime AST. Control flow, model execution and history ownership remain Rust.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from reference_paths import reference_dir

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--comfy',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
project=Path(__file__).resolve().parents[2]
relative='comfy/k_diffusion/sampling.py'
source=(a.comfy/relative).read_text()
manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text())
expected=next(e['sha256']for e in manifest if e['root']=='comfy'and e['path']==relative)
assert hashlib.sha256(source.encode()).hexdigest()==expected
tree=ast.parse(source)
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='sample_ipndm_v')

def expression(n):
    if isinstance(n,ast.Constant):return repr(n.value),False
    if isinstance(n,ast.Name):return n.id,True
    if isinstance(n,ast.Subscript)and isinstance(n.value,ast.Name)and n.value.id=='t_steps':
        return f'sigmas.f_get({ast.unparse(n.slice)})?',True
    if isinstance(n,ast.UnaryOp)and isinstance(n.op,ast.USub):
        value,tensor=expression(n.operand)
        return (f'({value}).f_neg()?'if tensor else '-'+value),tensor
    if isinstance(n,ast.BinOp):
        left,lt=expression(n.left);right,rt=expression(n.right)
        if not lt and not rt:
            # Only literal arithmetic in the selected immutable source formulas.
            value=eval(compile(ast.Expression(n),'<literal>','eval'),{'__builtins__':{}},{})
            return repr(value),False
        op={ast.Add:'add',ast.Sub:'sub',ast.Mult:'mul',ast.Div:'div'}[type(n.op)]
        if lt and rt:return f'({left}).f_{op}(&({right}))?',True
        if lt:return f'({left}).f_{op}_scalar({right})?',True
        if op=='sub':return f'({right}).f_rsub_scalar({left})?',True
        if op=='div':return f'({right}).f_reciprocal()?.f_mul_scalar({left})?',True
        return f'({right}).f_{op}_scalar({left})?',True
    raise AssertionError(ast.dump(n))

out=['// Generated from locked comfy/k_diffusion/sampling.py::sample_ipndm_v.',
     '// IPNDM source attribution: zju-pi/diff-sampler, Apache-2.0 (as in reference).',
     f'// Source SHA256: {expected}',
     'use crate::native::Error;','use tch::Tensor;','use super::locals::SourceLocals;',
     'pub(super) fn coefficients(frame:&mut SourceLocals,sigmas:&Tensor,i:i64,order:i64)->Result<Vec<Tensor>,Error>{',
     'let t_cur=sigmas.f_get(i)?;let t_next=sigmas.f_get(i+1)?;', 'match order {']
for order in [2,3,4]:
    branch=next(n for n in ast.walk(function)if isinstance(n,ast.If)and isinstance(n.test,ast.Compare)and isinstance(n.test.left,ast.Name)and n.test.left.id=='order'and isinstance(n.test.comparators[0],ast.Constant)and n.test.comparators[0].value==order)
    out.append(f'{order}=>{{')
    for statement in branch.body:
        assert isinstance(statement,ast.Assign)and len(statement.targets)==1
        name=statement.targets[0].id
        if name=='x_next':continue
        value,tensor=expression(statement.value)
        assert tensor
        out.append(f'let {name}=frame.keep("{name}",{value});')
    out.append('Ok(vec!['+','.join(f'coeff{i}'for i in range(1,order+1))+'])},')
out.extend(['_=>Ok(Vec::new()),','}','}'])
a.output.write_text('\n'.join(out)+'\n')
import subprocess
subprocess.run(['rustfmt','--edition','2024',str(a.output)],check=True)
print('Generated static Rust expressions for all variable-step IPNDM orders')
