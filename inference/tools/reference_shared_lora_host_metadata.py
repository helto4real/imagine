#!/usr/bin/env python3
"""Original LoRA loader item/tolist types, consumed keys and aliases."""
from pathlib import Path
loader=Path(__file__).with_name('reference_lora.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n',1)[0],str(loader),'exec'))
host_path=Path(__file__).with_name('reference_shared_tensor_host.py')
host_tree=ast.parse(host_path.read_text());host_scope=dict(struct=__import__('struct'))
host_fn=next(node for node in host_tree.body if isinstance(node,ast.FunctionDef)and node.name=='encode_host')
exec(compile(ast.Module(body=[host_fn],type_ignores=[]),str(host_path),'exec'),host_scope)
def encoded(value):return dict(type='none')if value is None else host_scope['encode_host'](value)
fixture=json.loads((project/'inference/fixtures/shared-tensor-host.json').read_text())
def build(desc):
 if desc['kind']=='meta':return torch.empty(desc['shape'],dtype=getattr(torch,desc['dtype']),device='meta')
 if desc['kind']=='quantized':
  value=torch.arange(math.prod(desc['shape']),dtype=torch.float32).reshape(desc['shape'])*.25+.25
  return torch.quantize_per_tensor(value,.125,0,getattr(torch,desc['dtype']))
 raw=bytes(desc['bytes'])
 value=torch.empty(desc['shape'],dtype=getattr(torch,desc['dtype']))if not raw else torch.frombuffer(bytearray(raw),dtype=getattr(torch,desc['dtype'])).reshape(desc['shape'])
 value=value.to(desc['device'])
 if desc['layout']=='transpose':value=value.transpose(0,1)
 if desc['layout']=='neg':value=torch._neg_view(value)
 return value
descriptors=[case['input']for case in fixture['cases']if not case['input']['dtype'].startswith('complex')]
alpha=[None]+[d for d in descriptors if
 (d['kind']=='dense'and d['layout']=='plain'and(d['shape']==[]or(d['shape']==[2]and d['dtype']in['float32','int64','bool'])))or
 (d['kind']=='meta'and d['shape']in[[],[2]])or(d['kind']=='quantized'and d['shape']in[[],[2]])]
reshape=[None]+descriptors
def find(kind,shape,dtype,device='cpu'):
 return next(d for d in descriptors if d['kind']==kind and d['shape']==shape and d['dtype']==dtype and d['device']==device and d.get('layout','plain')=='plain')
small=[None,find('dense',[2],'float32'),find('dense',[2,1],'bool'),find('meta',[],'float32','meta'),find('quantized',[2],'qint8'),find('dense',[2],'int64')]
pairs=[];seen=set()
for left,right in [(left,right)for left in alpha for right in small]+[(find('dense',[],'float32'),right)for right in reshape]:
 key=json.dumps([left,right],sort_keys=True)
 if key not in seen:seen.add(key);pairs.append((left,right))
cases=[]
for left,right in pairs:
 state={'unit.lora_up.weight':data((8,3),'cpu',torch.float32,1),'unit.lora_down.weight':data((3,8),'cpu',torch.float32,2)}
 if left is not None:state['unit.alpha']=build(left)
 if right is not None:state['unit.reshape_weight']=build(right)
 state=dict(sorted(state.items()));capture.records=[];result={}
 try:
  loaded=namespace['load_lora'](state,{'unit':'unit.weight'})
  item=loaded['unit.weight'];result=dict(alpha=encoded(item.weights[2]),reshape=encoded(item.weights[5]),keys=sorted(item.loaded_keys),missing=[key for key in state if key not in item.loaded_keys],aliases=[item.weights[0]is state['unit.lora_up.weight'],item.weights[1]is state['unit.lora_down.weight']])
 except Exception as error:
  reject_resource_failure(error);result=dict(error=type(error).__name__,message=str(error))
 cases.append(dict(alpha_input=left,reshape_input=right,diagnostics=capture.records,**result))
a.output.write_text(json.dumps(dict(schema=1,sources=sources,trained_model_weights=0,max_weight_elements=24,cases=cases),separators=(',',':'))+'\n')
print('Captured',len(cases),'original typed LoRA metadata loader histories; no trained weights')
