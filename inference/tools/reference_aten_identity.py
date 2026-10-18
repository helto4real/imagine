#!/usr/bin/env python3
"""Original ordinary ATen object identity, separate from storage aliasing."""
from reference_layout_completion import encoder

def capture(torch, devices, bits):
    encode=encoder(torch,bits);cases=[]
    def leaves(v):
        if isinstance(v,torch.Tensor):return[v]
        if isinstance(v,(tuple,list)):return[t for x in v for t in leaves(x)]
        return[]
    for dev in devices:
        x=torch.arange(4,device=dev,dtype=torch.float32).reshape(2,2)
        for op,args in [('aten::contiguous.default',[x]),('aten::to.dtype',[x,torch.float32]),('aten::detach.default',[x]),('aten::view.default',[x,[2,2]]),('aten::broadcast_tensors.default',[[x,x]]),('aten::split.Tensor',[x,1,0]),('aten::alias.default',[x]),('aten::to.dtype',[x,torch.float16])]:
            name,overload=op.split('::')[1].split('.');out=getattr(getattr(torch.ops.aten,name),overload)(*args);ins=leaves(args);outs=leaves(out)
            cases.append(dict(input_object_indices=[next(j for j,p in enumerate(ins)if p is i)for i in ins],device=dev,operation=op,args=[encode(v)for v in args],output=encode(out),input_identities=[[o is i for i in ins]for o in outs],input_aliases=[[o.data_ptr()==i.data_ptr()for i in ins]for o in outs],output_identities=[[o is p for p in outs]for o in outs]))
    return dict(aten_object_identity=cases)
