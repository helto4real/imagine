#!/usr/bin/env python3
"""Original OOM dispatch, class/properties/string short circuits and re-raise."""
from pathlib import Path
loader=Path(__file__).with_name('reference_model_management.py')
exec(compile(loader.read_text().split('\n# Offloaded/retention/',1)[0],str(loader),'exec'))
import itertools,sys
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ['is_oom','raise_non_oom']]
space={};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),space)
def run(kind,code,message,mode,fail=None):
    trace=[]
    def emit(event):
        trace.append(event)
        if fail==len(trace)-1:raise ValueError('own oom-dispatch failure')
    class Text:
        def __str__(self):
            emit(['str'])
            return None if message=='badstr'else message
    class Oom(Text,RuntimeError):pass
    class OomChild(Oom):pass
    class Accel(Text,RuntimeError):
        @property
        def error_code(self):
            emit(['code'])
            if code=='attr':raise AttributeError('own absent code')
            if code=='error':raise ValueError('own code failure')
            return code
    class AccelChild(Accel):pass
    class Both(Oom,Accel):pass
    class PlainRuntime(Text,RuntimeError):pass
    class PlainValue(Text,ValueError):pass
    classes=dict(oom=Oom,oom_child=OomChild,accel=Accel,accel_child=AccelChild,both=Both,runtime=PlainRuntime,value=PlainValue)
    original=classes[kind]('own-source-exception-payload')
    def instance(value,cls):emit(['instance','oom'if cls is Oom else'accelerator']);return isinstance(value,cls)
    def discard():emit(['discard'])
    space.update(OOM_EXCEPTION=Oom,ACCELERATOR_ERROR=Accel,isinstance=instance,discard_cuda_async_error=discard)
    output,error=None,None
    try:output=space['is_oom'if mode=='check'else'raise_non_oom'](original)
    except BaseException as ex:error=['same-error']if ex is original else['new-error',type(ex).__name__,str(ex)]
    return dict(kind=kind,code=code,message=message,mode=mode,failure=fail,events=trace,output=output,error=error)
cases=[]
for args in itertools.product(['oom','oom_child','accel','accel_child','both','runtime','value'],[2,7,None,'attr','error'],['out of memory','🧠 OUT OF MEMORY','ordinary','badstr'],['check','raise']):
    case=run(*args);cases.append(case);cases.extend(run(*args,fail=i)for i in range(len(case['events'])))
assert 'torch'not in sys.modules
a.output.write_text(json.dumps(dict(schema=1,source_sha256=digest,trained_model_weights=0,fixture_tensors=0,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original OOM dispatch/re-raise histories; no Torch or tensors')
