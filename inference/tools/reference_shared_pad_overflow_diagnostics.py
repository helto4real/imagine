#!/usr/bin/env python3
"""Retain an original size-overflow diagnostic; no normalization or model load."""
from pathlib import Path
loader=Path(__file__).with_name('reference_lora.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
weight=torch.zeros((0,0),dtype=torch.float32,device='cpu')
try:
 namespace['pad_tensor_to_shape'](weight,[2**64-1,8])
except Exception as e:
 reject_resource_failure(e)
 result=dict(error=type(e).__name__,message=str(e))
else:raise AssertionError('Expected the original invalid shape to fail before allocation')
assert result['error']=='TypeError'
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,device='cpu',allocated_input_elements=0,shape=[2**64-1,8],result=result),indent=2)+'\n')
print('Captured original CPU shape-overflow diagnostic without tensor payload allocation')
