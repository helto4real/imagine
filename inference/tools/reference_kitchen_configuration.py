#!/usr/bin/env python3
"""Frozen Kitchen import-time configuration expressions; no GPU execution."""
from reference_paths import reference_dir
import argparse,hashlib,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();project=Path(__file__).resolve().parents[2];relative='comfy_kitchen/backends/cuda/__init__.py';data=(a.site/relative).read_bytes();manifest=json.loads((reference_dir(project)/'source-manifest.json').read_text());assert hashlib.sha256(data).hexdigest()==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==relative);assert b'_INT4_INT8_WEIGHT_CHUNK_N = max(1, int(os.environ.get("COMFY_KITCHEN_INT4_INT8_WEIGHT_CHUNK_N", "4096")))'in data;assert sys.version_info[:3]==(3,12,13);assert sys.get_int_max_str_digits()==4300
values=[None,b'1',b'0',b'-1',b'4096',b' 17 ',b'+32',b'1_024',b'0016',str(2**63-1).encode(),str(2**63).encode(),b'9'*128,b'-'+b'9'*128,b'',b'own',b'1.5',b'0x10',b'1__2',b'_1',b'1_',b'+',b'\x1c1',b'1\x1f',b'\xff',b'1\xc3\x28',b'9'*4300,b'9'*4301,b'0'*4301,'١٢٣'.encode(),'１２_３'.encode(),'\u200317\u2003'.encode(),'²'.encode(),b'x'*210,b'\xff'*210];cases=[]
for value in values:
 text=value.decode('utf8','surrogateescape')if value is not None else'4096';c=dict(raw_hex=value.hex()if value is not None else None)
 try:
  n=max(1,int(text));c['bound']=min(2**63-1,n);c['consumed']=[min(rows,n)for rows in [0,1,16,4096,2**31,2**63-1]]
 except Exception as e:c.update(error=type(e).__name__,message=str(e))
 if value is None:c.update(disable_cutlass=False,w4_chunked=True,force_fallback=False)
 else:c.update(disable_cutlass=text=='1',w4_chunked=text!='0',force_fallback=text=='1')
 cases.append(c)
a.output.write_text(json.dumps(dict(kitchen_configuration=cases,scope='Source-verified import expressions on own supplied environment values only; no GPU execution'),indent=2)+'\n');print('Captured',len(cases),'original configuration cases')
