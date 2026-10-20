#!/usr/bin/env python3
"""Bounded original MMA packed-record inverse, own initialized bytes only."""
from pathlib import Path
loader=Path(__file__).with_name('reference_shared_schedules.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
from reference_layout_completion import encoder
path='comfy_kitchen/tensor/w4a8_packed.py';source=a.site/path
sha=hashlib.sha256(source.read_bytes()).hexdigest()
assert sha==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==path)
ns=dict(torch=torch)
names=['_pack_codes','w4a8_mma_record_bytes','_mma_bits','_fragment_major_inverse','unpack_w4a8_mma_weight']
tree=ast.parse(source.read_text());body=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names]
assert len(body)==len(names)
exec(compile(ast.Module(body=body,type_ignores=[]),str(source),'exec'),ns)
encode=encoder(torch,lambda t:[]if t.numel()==0 else t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist())
cases=[]
with torch.no_grad():
 for device in ['cpu','cuda:0']:
  for n,k,bits in [(16,256,4),(32,512,4),(16,256,6),(32,512,6),(0,256,4),(16,0,4)]:
   count=n*k*ns['w4a8_mma_record_bytes'](bits)//512
   packed=((torch.arange(count,device=device,dtype=torch.int32)*37+11)%256).to(torch.int8)
   for dtype in [torch.float8_e4m3fn,torch.float32,torch.float16]:
    for variant in ['ordinary','strided','bad_type','bad_shape','bad_n','bad_k','bad_size']:
     p=packed;nn=n;kk=k
     if variant=='strided':
      storage=torch.zeros(count*2,device=device,dtype=torch.int8);storage[::2]=p;p=storage[::2]
     elif variant=='bad_type':p=p.to(torch.uint8)
     elif variant=='bad_shape':p=p.reshape(1,-1)
     elif variant=='bad_n':nn=n+1
     elif variant=='bad_k':kk=k+1
     elif variant=='bad_size':p=torch.zeros(count+1,device=device,dtype=torch.int8)
     c=dict(device=device,n=nn,k=kk,kind=str(dtype),label=f'{n}/{k}/{bits}/{dtype}/{variant}',input=encode(p))
     try:c['output']=encode(ns['unpack_w4a8_mma_weight'](p,nn,kk,dtype))
     except Exception as e:
      reject_resource_failure(e);c.update(error=type(e).__name__,message=str(e))
     cases.append(c)
a.output.write_text(json.dumps(dict(schema=1,sources={path:sha},cases=cases),separators=(',',':'))+'\n')
print(f'Captured {len(cases)} original MMA codec cases; model weights allocated: 0')
