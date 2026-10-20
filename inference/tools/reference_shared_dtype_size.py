#!/usr/bin/env python3
"""Unmodified dtype_size with comparison truth proxies and attribute exceptions."""
from pathlib import Path
import sys
registered_types='--registered-types' in sys.argv
if registered_types:sys.argv.remove('--registered-types')
native_kinds='--native-kinds' in sys.argv
if native_kinds:sys.argv.remove('--native-kinds')
loader=Path(__file__).with_name('reference_model_management.py')
exec(compile(loader.read_text().split('\n# Offloaded/retention/',1)[0],str(loader),'exec'))
import itertools,sys
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='dtype_size')
space=dict(torch=NS(float16='half',bfloat16='bf16',float32='float'))
exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),space)
if native_kinds or registered_types:
    lock=json.loads((reference_dir(project)/'reference-lock.json').read_text())
    sys.path.insert(0,lock['roots']['site'])
    import torch
    assert torch.__version__.split('+')[0]==lock['torch_version']
    assert not torch.cuda.is_initialized()
    space['torch']=torch
    kinds=[('Uint8','uint8'),('Int8','int8'),('Int16','int16'),('Int','int32'),('Int64','int64'),('Half','float16'),('Float','float32'),('Double','float64'),('ComplexHalf','complex32'),('ComplexFloat','complex64'),('ComplexDouble','complex128'),('Bool','bool'),('QInt8','qint8'),('QUInt8','quint8'),('QInt32','qint32'),('BFloat16','bfloat16'),('Float8e5m2','float8_e5m2'),('Float8e4m3fn','float8_e4m3fn'),('Float8e5m2fnuz','float8_e5m2fnuz'),('Float8e4m3fnuz','float8_e4m3fnuz')]
    cases=[dict(kind=kind,dtype=str(getattr(torch,name)),size=space['dtype_size'](getattr(torch,name)))for kind,name in kinds]
    if registered_types:
        for kind,value in [('ShellE8M0',torch.float8_e8m0fnu),('ShellPackedE2M1',torch.float4_e2m1fn_x2),('ScalarInt',int),('ScalarFloat',float),('ScalarBool',bool),('ScalarStr',str),('ScalarNoneType',type(None))]:
            cases.append(dict(kind=kind,dtype=str(value),size=space['dtype_size'](value)))
    assert not torch.cuda.is_initialized()
    a.output.write_text(json.dumps(dict(schema=1,source_sha256=digest,torch_version=torch.__version__,trained_model_weights=0,fixture_tensors=0,cuda_initialized=False,cases=cases),separators=(',',':'))+'\n')
    print('Captured',len(cases),'original native dtype sizes; no tensors, weights or CUDA context')
    raise SystemExit
def run(mask,attribute,fail=None):
    trace=[];sentinel=object()
    def emit(event):
        trace.append(event)
        if fail==len(trace)-1:raise ValueError('own dtype-size failure')
    class Truth:
        def __init__(self,index):self.index=index
        def __bool__(self):emit(['bool',self.index]);return bool(mask&(1<<self.index))
    class Dtype:
        def __eq__(self,other):
            index=['half','bf16','float'].index(other);emit(['eq',index]);return Truth(index)
        @property
        def itemsize(self):
            emit(['itemsize'])
            if attribute in ['AttributeError','RuntimeError','KeyboardInterrupt','SystemExit','BaseException']:
                raise getattr(__import__('builtins'),attribute)('own itemsize failure')
            return dict(eight=8,negative=-7,none=None,false=False,object=sentinel)[attribute]
    result,error=None,None
    try:
        result=space['dtype_size'](Dtype())
        if result is sentinel:result='same-object'
    except BaseException as ex:error=[type(ex).__name__,str(ex)]
    return dict(mask=mask,attribute=attribute,failure=fail,events=trace,output=result,error=error)
cases=[]
for mask,attribute in itertools.product(range(8),['eight','negative','none','false','object','AttributeError','RuntimeError','KeyboardInterrupt','SystemExit','BaseException']):
    case=run(mask,attribute);cases.append(case);cases.extend(run(mask,attribute,i)for i in range(len(case['events'])))
assert 'torch' not in sys.modules
a.output.write_text(json.dumps(dict(schema=1,source_sha256=digest,trained_model_weights=0,fixture_tensors=0,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original dtype-size histories; no Torch or weights')
