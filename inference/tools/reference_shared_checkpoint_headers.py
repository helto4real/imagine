#!/usr/bin/env python3
"""Actual AIMDO loader Unicode/JSON header errors and explicit causes."""
from pathlib import Path
loader=Path(__file__).with_name('reference_checkpoint.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
raws=[b'',b' ',b'{',b'[',b'{"x":}',b'{"x":1,}',b'{"x" 1}',b'{"x":01}',b'{"x":1} {}',b'{"x":1}\xff',b'\xff',b'\xe2\x82',b'\xed\xa0\x80',b'\xef\xbb\xbf{}',b'null',b'false',b'42',b'1e500',b'NaN',b'Infinity',b'[]',b'[1,2]',b'"own"',b'-'+b'1'*4301,b'1'*4301,b'{"__metadata__":{"x":'+b'1'*4301+b'}}',b'{}',b' { } \r\n',b'{"__metadata__":{"note":"own"}}',b'{"__metadata__":42}',b'{"__metadata__":[]}',b'{"__metadata__":null}',b'{"__metadata__":{"a":1,"b":2},"__metadata__":{"a":3}}']
cases=[]
for index,raw in enumerate(raws):
 path=a.work/'owned-header.safetensors';path.write_bytes(struct.pack('<Q',len(raw))+raw)
 try:
  ns['load_safetensors'](str(path));parsed=json.loads(raw.decode('utf-8'));result=dict(items=list(parsed.items()))
 except Exception as error:
  from reference_checks import reject_resource_failure
  reject_resource_failure(error)
  result=dict(error=type(error).__name__,message=str(error).replace(str(path),'{file}'),suppress_context=error.__suppress_context__,cause=None if error.__cause__ is None else dict(error=type(error.__cause__).__name__,message=str(error.__cause__)),context_is_cause=error.__context__ is not None and error.__context__ is error.__cause__)
 cases.append(dict(label=str(index),bytes=raw.hex(),**result))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,max_file_bytes=max(8+len(v)for v in raws),cases=cases),indent=2)+'\n');print('Captured',len(cases),'original header/cause histories; no trained weights')
