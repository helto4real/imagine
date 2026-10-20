#!/usr/bin/env python3
"""Locked COND/area behavior using only bounded initialized synthetic tensors."""
from pathlib import Path
loader = Path(__file__).with_name('reference_shared_schedules.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n', 1)[0], str(loader), 'exec'))
import collections
import logging
definitions('comfy/utils.py', ['repeat_to_batch_size'])
ns.update(logging=logging, collections=collections, comfy=NS(utils=NS(repeat_to_batch_size=ns['repeat_to_batch_size'])))
definitions('comfy/conds.py', ['is_equal', 'CONDRegular', 'CONDNoiseShape', 'CONDCrossAttn', 'CONDConstant', 'CONDList'])
definitions('comfy/samplers.py', ['add_area_dims', 'get_area_and_mult'])
from reference_layout_completion import encoder
encode = encoder(torch, lambda t: t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().tolist())

def data(shape, device, dtype, offset=0):
    return (((torch.arange(math.prod(shape), device=device, dtype=torch.float32)+offset)%29)*.03125-.40625).reshape(shape).to(dtype)

def result(fn):
    try: return dict(output=encode(fn()))
    except Exception as e:
        reject_resource_failure(e)
        return dict(error=type(e).__name__, message=str(e))

conds, comparisons, areas, equals = [], [], [], []
with torch.no_grad():
    for device in ['cpu', 'cuda:0']:
        for dtype in [torch.float32, torch.bfloat16]:
            for name in ['CONDRegular', 'CONDNoiseShape', 'CONDCrossAttn', 'CONDConstant', 'CONDList']:
                x = data((2, 3, 12, 16), device, dtype)
                if name == 'CONDCrossAttn': x = data((2, 6, 3), device, dtype)
                elif name == 'CONDConstant': x = dict(first=[1, True, 'own'], second=(data((2,3), device, dtype), None))
                elif name == 'CONDList': x = [data((2, 3, 4), device, dtype), data((1,2,3), device, dtype, 7)]
                cond = ns[name](x)
                for batch in [0, 1, 2, 5]:
                    for area in [None, [8, 10, 2, 3]] if name == 'CONDNoiseShape' else [None]:
                        r = result(lambda: cond.process_cond(batch, area=area).cond)
                        conds.append(dict(name=name, device=device, input=encode(x), batch=batch, area=area, size=result(cond.size), **r))
            for n in [3, 4, 6, 8, 10, 0]:
                for name in ['CONDRegular', 'CONDCrossAttn']:
                    a1, b1 = data((2, 6, 3), device, dtype), data((2,n,3), device, dtype, 3)
                    c1, c2 = ns[name](a1), ns[name](b1)
                    comparisons.append(dict(name=name, device=device, a=encode(a1), b=encode(b1), can=result(lambda: c1.can_concat(c2)), concat=result(lambda: c1.concat([c2]))))
            for name, a1, b1 in [
                ('CONDConstant', dict(x=[1,2],y=True), dict(y=True,x=[1,2])),
                ('CONDConstant', [1,2], (1,2)),
                ('CONDList', [data((2,3),device,dtype)], [data((2,3),device,dtype,3)]),
                ('CONDList', [data((2,3),device,dtype)], [data((1,3),device,dtype,3)]),
            ]:
                c1,c2=ns[name](a1),ns[name](b1)
                comparisons.append(dict(name=name,device=device,a=encode(a1),b=encode(b1),can=result(lambda:c1.can_concat(c2)),concat=result(lambda:c1.concat([c2]))))
            for x in [[], [data((2,3),device,dtype)], [data((2,3),device,dtype), torch.tensor([1,2,3],device=device)], [torch.tensor([0],device=device)]]:
                conds.append(dict(name='CONDList',device=device,input=encode(x),batch=2,area=None,size=result(lambda:ns['CONDList'](x).size()),**result(lambda:ns['CONDList'](x).process_cond(2).cond)))
            for dims, area in [([32,40],None),([32,40],[24,30,3,5]),([32,40],[64,64,5,7]),([12,16,20],[8,10,2,3]),([12,16,20],[8,12,14,2,3,4]),([32,40],[4,20,22,0,0,1])]:
                x=data((2,2,*dims),device,dtype,3)
                timestep=torch.tensor([.75,.5],device=device)
                for mask_type in ['none','positive','signed']:
                    opts=dict(model_conds={},uuid='own-synthetic',strength=.75)
                    if area is not None: opts['area']=area
                    if mask_type!='none':
                        mask=data((1,*dims),device,torch.float32,5)
                        if mask_type=='positive':mask=mask.abs()
                        opts.update(mask=mask,mask_strength=.875)
                    out=ns['get_area_and_mult'](opts,x,timestep)
                    areas.append(dict(device=device,input=encode(x),time=encode(timestep),area=area,mask=encode(opts.get('mask')),mask_strength=.875,strength=.75,output=dict(input=encode(out.input_x),mult=encode(out.mult),area=out.area)))
            for a1,b1 in [(torch.tensor([1,2],device=device),torch.tensor([1.,2.],device=device)),(torch.tensor(float('nan'),device=device),torch.tensor(float('nan'),device=device)),([1,2],(1,2)),(2**53+1,float(2**53)),(2**63-1,float(2**63)),(torch.tensor([1,2],device=device),1),(torch.tensor(1,device=device),1),([torch.tensor([1,2],device=device)],[1]),(dict(a=[1,2],b=True),dict(b=1,a=[1,2]))]:
                equals.append(dict(device=device,a=encode(a1),b=encode(b1),**result(lambda:ns['is_equal'](a1,b1))))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,conditions=conds,comparisons=comparisons,areas=areas,equals=equals),separators=(',',':'))+'\n')
print(f'Captured {len(conds)} COND process/size, {len(comparisons)} concat, {len(areas)} areas, {len(equals)} equality cases; no model weights')
