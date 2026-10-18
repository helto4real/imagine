#!/usr/bin/env python3
"""Frozen is_oom and raise_non_oom, without real allocation failures."""
from pathlib import Path
loader=Path(__file__).with_name('reference_qwen_image21_core.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
namespace.update(OOM_EXCEPTION=torch.OutOfMemoryError,ACCELERATOR_ERROR=torch.AcceleratorError)
discarded=[]
namespace['discard_cuda_async_error']=lambda:discarded.append(True)
definitions('comfy/model_management.py',['is_oom','raise_non_oom'])
cases=[]
for kind,code,message in [(0,0,'bounded allocation failure'),(1,2,'bounded accelerator failure'),(1,7,'CUDA OUT OF MEMORY'),(1,7,'bounded unrelated accelerator failure'),(2,2,'out of memory'),(2,0,'bounded unrelated runtime failure')]:
 error=[torch.OutOfMemoryError,torch.AcceleratorError,RuntimeError][kind](message)
 if kind==1:error.error_code=code
 discarded.clear();oom=namespace['is_oom'](error);calls=len(discarded)
 discarded.clear();escaped=None
 try:namespace['raise_non_oom'](error)
 except Exception as caught:
  assert caught is error
  escaped=dict(class_name=type(caught).__name__,message=str(caught))
 cases.append(dict(kind=kind,code=code,message=message,oom=oom,discard_calls=calls,escaped=escaped))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,safety='No tensor allocations or actual OOM; source classification and exception identity only',cases=cases),indent=2)+'\n')
print(f'Captured {len(cases)} original exception classification cases')
