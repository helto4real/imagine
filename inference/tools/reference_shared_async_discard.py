#!/usr/bin/env python3
"""Original async-error discard control flow and own byte-object lifetimes."""
from pathlib import Path
loader=Path(__file__).with_name('reference_model_management.py')
exec(compile(loader.read_text().split('\n# Offloaded/retention/',1)[0],str(loader),'exec'))
import sys
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='discard_cuda_async_error')
space={}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),space)
classes={name:getattr(__import__('builtins'),name)for name in ['RuntimeError','NotImplementedError','RecursionError','ValueError','TypeError','AttributeError','IndexError','MemoryError','OverflowError','KeyboardInterrupt','SystemExit','BaseException']}
classes.update({name:type(name,(RuntimeError,),{})for name in ['OutOfMemoryError','AcceleratorError','_LinAlgError','OwnRuntimeChild']})
def run(name,fail):
    trace=[];alive=set();count=[0];event_index=[0]
    def emit(event):
        trace.append(event);index=event_index[0];event_index[0]+=1
        if fail==index:raise classes[name]('own async-discard failure')
    class Token:
        def __init__(self,name):self.name=name;alive.add(name)
        def __add__(self,other):emit(['add']);return Token('sum')
        def __del__(self):alive.remove(self.name);trace.append(['drop',self.name,sorted(alive)])
    def device():emit(['device']);return 'cuda:0'
    def tensor(values,dtype,device):
        count[0]+=1;name=['a','b'][count[0]-1]
        assert values==[1]and dtype=='uint8';emit(['tensor',name,device]);return Token(name)
    def synchronize():emit(['sync',sorted(alive)])
    space.update(torch=NS(tensor=tensor,uint8='uint8'),get_torch_device=device,synchronize=synchronize)
    error=None
    try:space['discard_cuda_async_error']()
    except BaseException as ex:error=[type(ex).__name__,str(ex)]
    return dict(exception=name,failure=fail,runtime=issubclass(classes[name],RuntimeError),events=trace,error=error,alive=sorted(alive))
cases=[run(name,fail)for name in classes for fail in [None,*range(6)]]
assert 'torch' not in sys.modules
a.output.write_text(json.dumps(dict(schema=1,source_sha256=digest,trained_model_weights=0,fixture_tensors=0,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original async-discard histories; own tokens only, no Torch or tensors')
