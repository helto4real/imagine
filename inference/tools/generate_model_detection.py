#!/usr/bin/env python3
"""Compile the locked detector's control flow into native Rust source.

Development/build tool only. The emitted function has no Python parser, module
loader, bytecode or interpreter dependency at runtime. Unknown syntax fails
generation rather than substituting a model-family rule.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from reference_paths import reference_dir

def quoted(value):
    return json.dumps(value,ensure_ascii=False)

class Compiler:
    def __init__(self,source_label="comfy/model_detection.py"):self.serial=0;self.source_label=source_label
    def unique(self):self.serial+=1;return f'detector_tmp_{self.serial}'
    def expr(self,node):
        match node:
            case ast.Constant(value=None):return 'none()'
            case ast.Constant(value=value)if isinstance(value,bool):return f'boolean({str(value).lower()})'
            case ast.Constant(value=value)if isinstance(value,int):return f'integer({value}_i128)?'
            case ast.Constant(value=value)if isinstance(value,float):return f'float({repr(value)}_f64)'
            case ast.Constant(value=value)if isinstance(value,str):return f'text({quoted(value)})'
            case ast.Name(id=name):return f'cx.get({quoted(name)})?'
            case ast.List(elts=values):return 'list(vec!['+','.join(self.expr(n)for n in values)+'])'
            case ast.Tuple(elts=values):return 'tuple(vec!['+','.join(self.expr(n)for n in values)+'])'
            case ast.Dict(keys=keys,values=values):return 'cx.mapping(vec!['+','.join(f'({self.expr(k)},{self.expr(v)})'for k,v in zip(keys,values))+'])?'
            case ast.Attribute(value=value,attr=attr):return f'cx.attribute(&{self.expr(value)},{quoted(attr)})?'
            case ast.Subscript(value=value,slice=ast.Slice(lower=lo,upper=hi,step=step)):
                opt=lambda n:'None'if n is None else 'Some('+self.expr(n)+')'
                return f'cx.slice(&{self.expr(value)},{opt(lo)},{opt(hi)},{opt(step)})?'
            case ast.Subscript(value=value,slice=index):return f'cx.index(&{self.expr(value)},&{self.expr(index)})?'
            case ast.BinOp(left=left,op=op,right=right):return f'cx.binary({quoted(type(op).__name__)},&{self.expr(left)},&{self.expr(right)})?'
            case ast.UnaryOp(op=op,operand=value):return f'cx.unary({quoted(type(op).__name__)},&{self.expr(value)})?'
            case ast.BoolOp(op=op,values=values):
                current=self.expr(values[-1])
                for value in reversed(values[:-1]):
                    name=self.unique();test=f'cx.truth(&{name})?'
                    current=('{let '+name+'='+self.expr(value)+';if '+test+'{'+current+'}else{'+name+'}}')if isinstance(op,ast.And)else('{let '+name+'='+self.expr(value)+';if '+test+'{'+name+'}else{'+current+'}}')
                return current
            case ast.Compare(left=left,ops=ops,comparators=right):
                name=self.unique();lines=[f'let {"mut "if len(ops)>1 else ""}{name}={self.expr(left)};']
                for i,(op,value)in enumerate(zip(ops,right)):
                    other=self.unique();lines.append(f'let {other}={self.expr(value)};')
                    call=f'cx.compare({quoted(type(op).__name__)},&{name},&{other})?'
                    if i==len(ops)-1:lines.append(call)
                    else:lines.append(f'if !cx.truth(&{call})?{{return Ok(boolean(false));}}{name}={other};')
                if len(ops)==1:return '{'+''.join(lines)+'}'
                return '(|| -> Result<Options,Error>{'+''.join(lines[:-1])+f'Ok({lines[-1]})'+'})()?'
            case ast.IfExp(test=test,body=body,orelse=other):return f'if cx.truth(&{self.expr(test)})?{{{self.expr(body)}}}else{{{self.expr(other)}}}'
            case ast.JoinedStr(values=values):
                parts=[self.expr(n)if isinstance(n,ast.Constant)else f'cx.string(&{self.expr(n.value)})?'for n in values]
                return 'cx.concatenate_text(&['+','.join(parts)+'])?'
            case ast.Call(func=ast.Name(id=name),args=[ast.GeneratorExp()as generator],keywords=[])if name in ['any','all','sum']:
                return self.comprehension(generator,name)
            case ast.Call(func=ast.Name(id='len'),args=[ast.SetComp()as comp],keywords=[]):return self.comprehension(comp,'unique_len')
            case ast.ListComp():return self.comprehension(node,'list')
            case ast.Call(func=ast.Name(id='filter'),args=[ast.Lambda()as predicate,items],keywords=[]):
                assert len(predicate.args.args)==1
                var=predicate.args.args[0].arg
                return self.comprehension(ast.ListComp(elt=ast.Name(id=var),generators=[ast.comprehension(target=ast.Name(id=var),iter=items,ifs=[predicate.body],is_async=0)]),'list')
            case ast.Call(func=ast.Attribute(value=ast.Name(id=module),attr=name),args=args,keywords=keywords)if module in ['math','torch','json']:
                return f'cx.function({quoted(module+"."+name)},vec!['+','.join(self.expr(n)for n in args)+'],vec!['+','.join(f'({quoted(n.arg)},{self.expr(n.value)})'for n in keywords)+'])?'
            case ast.Call(func=ast.Attribute(value=value,attr=name),args=args,keywords=keywords):
                return f'cx.method(&{self.expr(value)},{quoted(name)},vec!['+','.join(self.expr(n)for n in args)+'],vec!['+','.join(f'({quoted(n.arg)},{self.expr(n.value)})'for n in keywords)+'])?'
            case ast.Call(func=ast.Name(id=name),args=args,keywords=[]):return f'cx.function({quoted(name)},vec!['+','.join(self.expr(n)for n in args)+'],vec![])?'
            case _:raise ValueError(f'Unsupported detector expression {ast.dump(node,include_attributes=False)}')
    def comprehension(self,node,kind):
        assert len(node.generators)==1 and not node.generators[0].is_async
        gen=node.generators[0];assert isinstance(gen.target,ast.Name)
        body='cx.set('+quoted(gen.target.id)+',item);'
        for condition in gen.ifs:body+='if !cx.truth(&'+self.expr(condition)+')?{return Ok(None);}'
        body+='Ok(Some('+self.expr(node.elt)+'))'
        return 'cx.comprehension(cx.iter(&'+self.expr(gen.iter)+')?,'+quoted(kind)+',|cx,item|{'+body+'})?'
    def assign(self,target,value):
        if isinstance(target,ast.Name):return f'cx.set({quoted(target.id)},{value});'
        if isinstance(target,ast.Subscript):return f'cx.set_item(&{self.expr(target.value)},&{self.expr(target.slice)},{value})?;'
        if isinstance(target,(ast.Tuple,ast.List)):
            name=self.unique();return f'let {name}={value};'+''.join(self.assign(item,f'cx.index(&{name},&integer({i})?)?')for i,item in enumerate(target.elts))
        raise ValueError(f'Unsupported assignment {ast.dump(target)}')
    def statements(self,nodes):
        output=[]
        for node in nodes:
            line=f'// {self.source_label}:{node.lineno}\n'
            match node:
                case ast.Assign(targets=targets,value=value):
                    if len(targets)==1:line+=self.assign(targets[0],self.expr(value))
                    else:
                        name=self.unique();line+=f'let {name}={self.expr(value)};'+''.join(self.assign(target,name+'.clone()')for target in targets)
                case ast.AugAssign(target=target,op=op,value=value):line+=self.assign(target,f'cx.binary({quoted(type(op).__name__)},&{self.expr(target)},&{self.expr(value)})?')
                case ast.If(test=test,body=body,orelse=other):line+=f'if cx.truth(&{self.expr(test)})?{{{self.statements(body)}}}'+('else{'+self.statements(other)+'}'if other else'')
                case ast.For(target=target,iter=items,body=body,orelse=[]):line+='for detector_item in cx.iter(&'+self.expr(items)+')?{'+self.assign(target,'detector_item')+self.statements(body)+'}'
                case ast.Return(value=value):
                    expr=self.expr(value)
                    line+='return '+(expr[:-1]if expr.endswith('?')else'Ok('+expr+')')+';'
                case ast.Expr(value=ast.Constant(value=str())):continue
                case ast.Expr(value=value):line+=self.expr(value)+';'
                case ast.Break():line+='break;'
                case ast.Pass():line+=';'
                case ast.Raise(exc=ast.Call(func=ast.Name(id=name),args=[message],keywords=[]),cause=None):line+=f'return Err(cx.exception({quoted(name)},&{self.expr(message)})?);'
                case ast.Try(body=body,handlers=[ast.ExceptHandler(type=ast.Name(id='Exception'),name=None,body=[ast.Pass()])],orelse=[],finalbody=[]):line+='cx.attempt(|cx|{'+self.statements(body)+'Ok(())})?;'
                case _:raise ValueError(f'Unsupported detector statement {ast.dump(node,include_attributes=False)}')
            output.append(line+'\n')
        return ''.join(output)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--comfy',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--function',choices=['detect_unet_config','unet_config_from_diffusers_unet'],default='detect_unet_config');args=parser.parse_args()
    project=Path(__file__).resolve().parents[2];source=args.comfy/'comfy/model_detection.py';digest=hashlib.sha256(source.read_bytes()).hexdigest()
    manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text());assert digest==next(e['sha256']for e in manifest if e['root']=='comfy'and e['path']=='comfy/model_detection.py')
    fn=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef)and n.name==args.function)
    assert isinstance(fn.body[-1],ast.Return)
    compiler=Compiler();compiled=compiler.statements(fn.body[:-1]);tail=compiler.expr(fn.body[-1].value)
    compiled+=tail[:-1]if tail.endswith('?')else'Ok('+tail+')'
    out='// Generated from locked comfy/model_detection.py; edit the generator.\n'+f'// Source SHA256: {digest}\n'+'use super::detector_values::*;\nuse crate::{native::Error,patching::options::Options};\npub(super) fn run(mut cx:Context)->Result<Options,Error>{\n'+compiled+'}\n'
    args.output.write_text(out)
    import subprocess
    subprocess.run(['rustfmt','--edition','2024',str(args.output)],check=True)
    print('Compiled all detector statements into native Rust; source SHA256',digest)
if __name__=='__main__':main()
