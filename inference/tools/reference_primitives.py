#!/usr/bin/env python3
"""Generate only synthetic primitive fixtures, outside ordinary Dreamchat.

Run using -S: explicit frozen package source, no .pth/site startup hooks.
No model loaders, user prompts, history, runtime reference images or caches.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

parser=argparse.ArgumentParser()
parser.add_argument('--site',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--triton-frontend-reference',type=Path,required=True)
parser.add_argument('--triton-specialization-reference',type=Path,required=True)
args=parser.parse_args()
sys.path.insert(0,str(args.site))
import torch
from comfy_kitchen.backends.eager import quantization as eager
from comfy_kitchen.backends import cuda as native
from comfy_kitchen.registry import BackendRegistry
from comfy_kitchen.constraints import FunctionConstraints, ParamConstraint, ExactDims, DivisibleBy
from comfy_kitchen.tensor import base as tensor_base
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
from comfy_kitchen.tensor import int8 as int8_layout
from comfy_kitchen.tensor import fp8 as fp8_layout
from comfy_kitchen.tensor import nvfp4 as nvfp4_layout
from comfy_kitchen.tensor import mxfp8 as mxfp8_layout
from comfy_kitchen import float_utils
from reference_fp8_ops import capture as capture_fp8_math
from reference_checks import reject_resource_failure
from reference_convrot import capture as capture_convrot
from reference_packing import capture as capture_packing
from reference_int8_utils import capture as capture_int8_utils
from reference_convrot_weight import capture as capture_convrot_weight
from reference_convrot_w4a4 import capture as capture_convrot_w4a4
from reference_int8_eager_ops import capture as capture_int8_eager_ops
from reference_int8_linear import capture as capture_int8_linear
from reference_cuda_linear import capture as capture_cuda_linear
from reference_cuda_int8_m1 import capture as capture_cuda_int8_m1
from reference_cuda_int8_gemv import capture as capture_cuda_int8_gemv
from reference_cuda_int8_gemm import capture as capture_cuda_int8_gemm
from reference_cuda_int8_activation import capture as capture_cuda_int8_activation
from reference_cuda_int8_zero_k import capture as capture_cuda_int8_zero_k
from reference_svdquant import capture as capture_svdquant, controlled as controlled_svdquant
from reference_svdquant_quantize import capture as capture_svdquant_quantize
from reference_int4_forced import capture as capture_int4_forced, controlled as controlled_int4_forced
from reference_int4_family import capture as capture_int4_family, controlled as controlled_int4_family
from reference_awq import capture as capture_awq
from reference_awq_layout import capture as capture_awq_layout
from reference_binding import capture as capture_binding
from reference_backend_bindings import capture as capture_backend_bindings
from reference_conv_bindings import capture as capture_conv_bindings
from reference_quant_bindings import capture as capture_quant_bindings
from reference_sol_bindings import capture_frozen as capture_sol_bindings, capture_ordinary as capture_sol_ordinary_bindings
from reference_w4a4_linear_bindings import capture as capture_w4a4_linear_bindings
from reference_w4a4_cache_bindings import capture as capture_w4a4_cache_bindings
from reference_na3d_bindings import capture as capture_na3d_bindings
from reference_rope_bindings import capture as capture_rope_bindings
from reference_w4a8_bindings import capture as capture_w4a8_bindings
from reference_sol_cuda_bindings import capture as capture_sol_cuda_bindings
from reference_layout_completion import capture as capture_layout_completion
from reference_svd_layout import capture as capture_svd_layout
from reference_kitchen_direct import capture as capture_kitchen_direct
from reference_svd_attributes import capture as capture_svd_attributes
from reference_sage import capture as capture_sage
from reference_triton_fp8_bindings import capture as capture_triton_fp8_bindings
from reference_triton_rowwise_bindings import capture as capture_triton_rowwise_bindings
from reference_triton_adaln_bindings import capture as capture_triton_adaln_bindings
from reference_triton_rope_bindings import capture as capture_triton_rope_bindings, capture_known_storage as capture_triton_rope_known_storage
from reference_triton_rms_rope_bindings import capture as capture_triton_rms_rope_bindings
from reference_triton_mxfp8_bindings import capture as capture_triton_mxfp8_bindings
from reference_triton_nvfp4_bindings import capture as capture_triton_nvfp4_bindings
from reference_aten_identity import capture as capture_aten_identity
from reference_triton_runtime_failures import capture as capture_triton_runtime_failures
from reference_triton_file_cache import capture as capture_triton_file_cache
from reference_triton_driver import capture as capture_triton_driver
from reference_sol_chunked import capture_fab as capture_sol_fab, capture_chunked as capture_sol_chunked


def bits(tensor):
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().flatten().tolist()


devices=['cpu']
if torch.cuda.is_available():devices.append('cuda:0')
if devices!=['cpu','cuda:0']:raise RuntimeError('Initial-platform reference validation requires CUDA; CPU-only fixtures are incomplete')
result={'schema':1,'inputs':'synthetic only','torch':torch.__version__,'devices':devices,'rng':[],'fp8':[],'registry':[],'storage':[],'shell_dtypes':[],'int8':[],'native_quantizers':[],'layout_shapes':[],'layout_dequantization':[],'requantization_metadata':[],'packed_layouts':[],'block_scales':[],'e8m0_decode':[],'mxfp8_eager':[],'nvfp4_eager_decode':[]}
result['fp8_math']=capture_fp8_math(torch,devices,bits)
result.update(capture_convrot(torch,native,bits))
result.update(capture_packing(torch,devices,bits))
result.update(capture_int8_utils(torch,devices,bits))
result.update(capture_convrot_weight(torch,native,bits))
result.update(capture_convrot_w4a4(torch,native,bits))
result.update(capture_int8_eager_ops(torch,native,devices,bits))
result.update(capture_int8_linear(torch,native,devices,bits))
result.update(capture_cuda_linear(torch,native,devices,bits))
result.update(capture_cuda_int8_m1(torch,native,bits))
result.update(capture_cuda_int8_gemv(torch,native,bits))
result.update(capture_cuda_int8_gemm(torch,native,bits))
result.update(capture_cuda_int8_activation(torch,native,bits))
result["controlled_cuda_int8_linear_zero_k"]=capture_cuda_int8_zero_k(torch,native,bits)
result["svdquant_scaled_mm"]=capture_svdquant(torch,native,bits)
result["controlled_svdquant_nongroup"]=controlled_svdquant(torch,native,bits)
result.update(capture_svdquant_quantize(torch,native,bits))
result["int4_forced_small_m"]=capture_int4_forced(torch,native,bits)
result["controlled_int4_forced_zero_k"]=controlled_int4_forced(torch,native,bits)
result.update(capture_int4_family(torch,native,bits))
result["controlled_int4_remaining_zero_k"]=controlled_int4_family(torch,native,bits)
result.update(capture_awq(torch,native,devices,bits))
result.update(capture_awq_layout(torch,devices,bits))
result.update(capture_binding(torch,devices,bits))
result.update(capture_backend_bindings(torch,devices,bits))
result.update(capture_rope_bindings(torch,devices,bits))
result.update(capture_quant_bindings(torch,devices,bits))
result.update(capture_w4a4_linear_bindings(torch,devices,bits))
result.update(capture_w4a4_cache_bindings(torch,devices,bits))
result.update(capture_na3d_bindings(torch,devices,bits))
result.update(capture_w4a8_bindings(torch,devices,bits))
result.update(capture_sol_cuda_bindings(torch,devices,bits))
result.update(capture_layout_completion(torch,devices,bits))
result.update(capture_svd_layout(torch,devices,bits))
result.update(capture_kitchen_direct(torch,devices,bits))
result.update(capture_svd_attributes(torch,devices,bits))
result.update(capture_sage(torch,devices,bits))
result.update(capture_sol_fab(torch,devices,bits))
result.update(capture_sol_chunked(torch,devices,bits))
result.update(capture_triton_driver(torch,devices,bits,args.triton_frontend_reference))
result.update(capture_triton_fp8_bindings(torch,devices,bits))
result.update(capture_triton_rowwise_bindings(torch,devices,bits))
rowwise_frontend=json.loads((args.triton_frontend_reference.parent.parent/'triton-rowwise-frontend/reference.json').read_text())
result['triton_rowwise_specialization']=rowwise_frontend['cases']
result['triton_rowwise_function_cache_key']=rowwise_frontend['function_cache_key']
result.update(capture_triton_adaln_bindings(torch,devices,bits))
adaln_frontend=json.loads((args.triton_frontend_reference.parent.parent/'triton-adaln-frontend/reference.json').read_text())
result['triton_adaln_specialization']=adaln_frontend['cases']
result['triton_adaln_function_cache_key']=adaln_frontend['function_cache_key']
result.update(capture_triton_rope_bindings(torch,devices,bits))
result.update(capture_triton_rope_known_storage(torch,bits))
rope_frontend=json.loads((args.triton_frontend_reference.parent.parent/'triton-rope-frontend/reference.json').read_text())
result['triton_rope_specialization']=rope_frontend['cases']
result['triton_rope_function_cache_key']=rope_frontend['function_cache_key']
result.update(capture_triton_rms_rope_bindings(torch,devices,bits))
rms_rope_frontend=json.loads((args.triton_frontend_reference.parent.parent/'triton-rms-rope-frontend/reference.json').read_text())
result['triton_rms_rope_specialization']=rms_rope_frontend['cases']
result['triton_rms_rope_function_cache_key']=rms_rope_frontend['function_cache_key']
for family,folder in [('adaln','adaln'),('rms_rope','rms-rope')]:
    result[f'triton_{family}_specialization']+=json.loads((args.triton_frontend_reference.parent.parent/f'triton-{folder}-scalar-frontend/reference.json').read_text())['cases']
    result[f'triton_{family}_specialization']+=json.loads((args.triton_frontend_reference.parent.parent/f'triton-{folder}-uint-frontend/reference.json').read_text())['cases']
result.update(capture_triton_mxfp8_bindings(torch,devices,bits))
mxfp8_frontend=json.loads((args.triton_frontend_reference.parent.parent/'triton-mxfp8-frontend/reference.json').read_text())
result['triton_mxfp8_specialization']=mxfp8_frontend['cases']
result['triton_mxfp8_function_cache_key']=mxfp8_frontend['function_cache_key']
result.update(capture_triton_nvfp4_bindings(torch,devices,bits))
for key,folder in [('nvfp4','nvfp4'),('dequant_nvfp4','dequant-nvfp4')]:
    nv_frontend=json.loads((args.triton_frontend_reference.parent.parent/f'triton-{folder}-frontend/reference.json').read_text())
    result[f'triton_{key}_specialization']=nv_frontend['cases']
    result[f'triton_{key}_function_cache_key']=nv_frontend['function_cache_key']

# Each owns a fresh original module/autotuner context, matching the separate
# candidate test context. This is host-only fixture capture, never engine IPC.
import subprocess
for family in ['int8','w4a8','na']:
    capture_path=args.output.parent/f'triton-{family}-bindings.json'
    subprocess.run([sys.executable,'-S','-B',str(Path(__file__).resolve().parent/f'reference_triton_{family}_bindings.py'),'--site',str(args.site),'--output',str(capture_path)],check=True)
    result.update(json.loads(capture_path.read_text()))
    integer_frontend=json.loads((args.triton_frontend_reference.parent.parent/f'triton-{family}-frontend/reference.json').read_text())
    result[f'triton_{family}_specialization']=integer_frontend['cases']
    result[f'triton_{family}_function_cache_keys']=integer_frontend['function_cache_keys']
    if family=='int8':
        result['triton_int8_specialization']+=json.loads((args.triton_frontend_reference.parent.parent/'triton-int8-double-frontend/reference.json').read_text())['cases']
    if family=='na':
        metadata={c['index']:c['metadata']for c in integer_frontend['compiled']}
        for c in result['triton_na_specialization']:
            if c['index']in metadata:c['compiled_metadata']=metadata[c['index']]
        result['triton_na_source_errors']=integer_frontend['source_errors']
        unsigned=json.loads((args.triton_frontend_reference.parent.parent/'triton-na-uint-frontend/reference.json').read_text())
        metadata={c['index']:c['metadata']for c in unsigned['compiled']}
        for c in unsigned['cases']:
            if c['index']in metadata:c['compiled_metadata']=metadata[c['index']]
        result['triton_na_specialization']+=unsigned['cases']

signature_capture=args.output.parent/'backend-python-signatures.json'
subprocess.run([sys.executable,'-S','-B',str(Path(__file__).resolve().parent/'reference_backend_signatures.py'),'--site',str(args.site),'--output',str(signature_capture),'--check-rust',str(Path(__file__).resolve().parents[1]/'src/backend_signatures.rs')],check=True)
signature_data=json.loads(signature_capture.read_text())
assert not signature_data['unresolved']
result['backend_signature_errors']=signature_data['cases']
for name in ['jit_keys','hooks','stream_order','default_dtype','environment','scalar_bindings','object_arguments']:
    capture_path=args.output.parent/f'triton-{name}-contracts.json'
    subprocess.run([sys.executable,'-S','-B',str(Path(__file__).resolve().parent/f'reference_triton_{name}.py'),'--site',str(args.site),'--output',str(capture_path)],check=True)
    result.update(json.loads(capture_path.read_text()))
dynamic_capture=args.output.parent/'dynamic-arguments-contracts.json'
subprocess.run([sys.executable,'-S','-B',str(Path(__file__).resolve().parent/'reference_dynamic_arguments.py'),'--site',str(args.site),'--output',str(dynamic_capture)],check=True)
result.update(json.loads(dynamic_capture.read_text()))
seed_capture=args.output.parent/'dynamic-seed-state-contracts.json'
subprocess.run([sys.executable,'-S','-B',str(Path(__file__).resolve().parent/'reference_dynamic_seed_state.py'),'--site',str(args.site),'--output',str(seed_capture)],check=True)
result.update(json.loads(seed_capture.read_text()))
na_host_capture=args.output.parent/'dynamic-na-host-contracts.json'
subprocess.run([sys.executable,'-S','-B',str(Path(__file__).resolve().parent/'reference_dynamic_na_host.py'),'--site',str(args.site),'--output',str(na_host_capture)],check=True)
result.update(json.loads(na_host_capture.read_text()))
for script,key,extra in [
    ('reference_registry_structured.py','structured_metadata',[]),
    ('reference_registry_sol_values.py','sol_structured',['--project',str(Path(__file__).resolve().parents[2])]),
    ('reference_input_activation_values.py','input_activation_values',['--fused-matrix']),
    ('reference_seed_types.py','seed_types',[]),
    ('reference_constexpr_state.py','constexpr_state',[]),
    ('reference_cuda_na_values.py','cuda_na_values',[]),
    ('reference_dynamic_nvfp4_literals.py','dynamic_nvfp4_literals',[]),
    ('reference_uint64_seeds.py','uint64_seeds',[]),
    ('reference_int8_rms_epsilon.py','int8_rms_epsilon',['--fused-matrix']),
    ('reference_kitchen_configuration.py','kitchen_configuration',[]),
]:
    capture_path=args.output.parent/f'{key}-contracts.json'
    subprocess.run([sys.executable,'-S','-B',str(Path(__file__).resolve().parent/script),'--site',str(args.site),'--output',str(capture_path),*extra],check=True)
    result[key]=json.loads(capture_path.read_text())[key]
result['na_values']=[]
for device in ['cpu','cuda']:
    capture_path=args.output.parent/f'na-values-{device}-contracts.json'
    subprocess.run([sys.executable,'-S','-B',str(Path(__file__).resolve().parent/'reference_na_values.py'),'--site',str(args.site),'--output',str(capture_path),'--device',device],check=True)
    result['na_values'].extend(json.loads(capture_path.read_text())['na_values'])
result['tensor_repr']=[]
for device in ['cpu','cuda']:
    capture_path=args.output.parent/f'tensor-repr-{device}-contracts.json'
    subprocess.run([sys.executable,'-S','-B',str(Path(__file__).resolve().parent/'reference_tensor_repr.py'),'--site',str(args.site),'--output',str(capture_path),'--device',device],check=True)
    result['tensor_repr'].extend(json.loads(capture_path.read_text())['tensor_repr'])
unsigned_scalars=args.output.parent/'triton-scalar-uint-contracts.json'
subprocess.run([sys.executable,'-S','-B',str(Path(__file__).resolve().parent/'reference_triton_scalar_bindings.py'),'--site',str(args.site),'--output',str(unsigned_scalars),'--unsigned-only'],check=True)
result['triton_scalar_bindings']+=json.loads(unsigned_scalars.read_text())['triton_scalar_bindings']
result['triton_nvfp4_literal_ast']=json.loads((args.triton_frontend_reference.parent.parent/'triton-nvfp4-literals/reference.json').read_text())['literal_cases']
import tempfile
with tempfile.TemporaryDirectory(prefix='compiler-mode-source-',dir=args.output.parent) as own:
    capture_path=args.output.parent/'triton-compiler-modes.json'
    subprocess.run([sys.executable,'-S','-B',str(Path(__file__).resolve().parent/'reference_triton_compiler_modes.py'),'--site',str(args.site),'--root',str(Path(own)/'modes'),'--output',str(capture_path),'--frontend-reference',str(args.triton_frontend_reference.parent/'case-0.expected.ttir')],check=True)
    result['triton_compiler_modes']=json.loads(capture_path.read_text())['records']
result.update(capture_aten_identity(torch,['cpu'],bits))
result.update(capture_triton_file_cache(args.output.parent/"file-cache-source"))
result.update(capture_triton_runtime_failures(torch,args.output.parent/"triton-compiler-profile.json",args.triton_frontend_reference.parent/"standalone/case-0.cubin",args.output.parent/"triton-runtime-source"))
specialization_reference=json.loads(args.triton_specialization_reference.read_text())
import hashlib
for name,h in specialization_reference['source_hashes'].items():
    assert hashlib.sha256((args.site/name).read_bytes()).hexdigest()==h, name
result['triton_specialization']=specialization_reference['cases']
result['triton_cache_environments']=specialization_reference['environments']
result['triton_specialization_hash_inputs']={k:v for k,v in specialization_reference.items()if k not in ['cases','environments']}

if "cuda:0" in devices:
    result["sol_eager_bindings"]=capture_sol_bindings(torch)["sol_eager_bindings"]
    result.update(capture_sol_ordinary_bindings(torch))
result.update(capture_conv_bindings(torch,devices,bits))
result['int4_rows']=[]
for dtype in [torch.float32,torch.float16,torch.bfloat16]:
    for name,group in [('plain',None),('convrot',16),('convrot',64),('convrot',256),('to_int8',256),('convrot',32),('to_int8',64)]:
        for shape in [(4,16),(2,2,64),(4,256),(4,30)]:
            for seed in [None,0,123]:
                x=(torch.arange(torch.tensor(shape).prod().item(),device='cuda:0',dtype=torch.float32)*0.03125-4).reshape(shape).to(dtype)
                x.reshape(-1,shape[-1])[0].zero_()
                case={'dtype':str(dtype),'operation':name,'group':group,'shape':shape,'seed':seed}
                try:
                    if name=='plain':q,s=native.quantize_int4_rowwise(x,stochastic_rounding=seed)
                    elif name=='convrot':q,s=native.quantize_int4_rowwise_convrot64(x,group,stochastic_rounding=seed)
                    else:q,s=native.quantize_int4_rowwise_convrot64_to_int8(x,group,stochastic_rounding=seed)
                    case.update(payload=bits(q),scale=bits(s),output_shape=list(q.shape),scale_shape=list(s.shape))
                except Exception as error:
                    reject_resource_failure(error)
                    case.update(error=type(error).__name__,message=str(error))
                result['int4_rows'].append(case)

def packed_layout_cases(device):
    definitions=[
        ('nvfp4',nvfp4_layout.TensorCoreNVFP4Layout,nvfp4_layout._handle_nvfp4_transpose,
         torch.arange(128,dtype=torch.uint8,device=device).reshape(16,8),
         dict(scale=torch.tensor(0.5,dtype=torch.float64,device=device),block_scale=torch.ones((128,4),device=device).to(torch.float8_e4m3fn))),
        ('mxfp8',mxfp8_layout.TensorCoreMXFP8Layout,mxfp8_layout._handle_mxfp8_transpose,
         torch.zeros((32,32),device=device).to(torch.float8_e4m3fn),
         dict(scale=torch.full((128,4),127,dtype=torch.uint8,device=device).view(torch.float8_e8m0fnu))),
    ]
    for name,layout,handler,payload,kwargs in definitions:
        root=tensor_base.QuantizedTensor(payload,layout.__name__,layout.Params(orig_shape=(3,5),orig_dtype=torch.float16,**kwargs))
        value=root
        for step in range(3):
            if step:value=handler(value,(value,),{})
            state=value.state_dict(prefix='weight')
            owners={'weight':value._qdata,'weight_scale':value._params.block_scale,'weight_scale_2':value._params.scale} if name=='nvfp4' else {'weight':value._qdata,'weight_scale':value._params.scale}
            result['packed_layouts'].append({'device':device,'format':name,'transposes':step,'shape':list(value._params.orig_shape),'storage_shape':list(value._qdata.shape),'strides':list(value._qdata.stride()),'transposed':value._params.transposed,'payload':bits(value._qdata),'payload_alias':value._qdata.data_ptr()==root._qdata.data_ptr(),'fields':{field:{'bytes':bits(getattr(value._params,field)),'dtype':str(getattr(value._params,field).dtype),'aliased':getattr(value._params,field).data_ptr()==getattr(root._params,field).data_ptr()} for field in value._params._tensor_fields()},'state':{key:{'bytes':bits(tensor),'aliased':tensor.data_ptr()==owners[key].data_ptr()} for key,tensor in state.items()},'state_keys':list(state)})

def layout_cases(device):
    payload=torch.arange(32,device=device,dtype=torch.float32).reshape(4,8).to(torch.float8_e4m3fn)
    scale=torch.tensor(0.5,device=device,dtype=torch.float64)
    params=fp8_layout.TensorCoreFP8Layout.Params(scale=scale,orig_dtype=torch.float16,orig_shape=(4,8))
    original=tensor_base.QuantizedTensor(payload,'TensorCoreFP8Layout',params)
    def describe(operation,value,root):
        state=value.state_dict(prefix='weight')
        return {'device':device,'operation':operation,'layout':value._layout_cls,'shape':list(value._params.orig_shape),'storage_shape':list(value._qdata.shape),'strides':list(value._qdata.stride()),'kind':str(value._params.orig_dtype),'payload':bits(value._qdata),'scale':bits(value._params.scale),'payload_alias':value._qdata.data_ptr()==root._qdata.data_ptr(),'scale_alias':value._params.scale.data_ptr()==root._params.scale.data_ptr(),'transposed':getattr(value._params,'transposed',False),'state':{k:{'bytes':bits(v),'aliased':v.data_ptr()==(value._qdata if k=='weight' else value._params.scale).data_ptr()} for k,v in state.items()}}
    shape_ops=[('view',torch.ops.aten.view.default,(2,16)),('reshape',torch.ops.aten.reshape.default,(8,4)),('transpose',torch.ops.aten.t.default,None)]
    for name,op,shape in shape_ops:
        value=fp8_layout._make_fp8_shape_handler(op)(original,(original,) if shape is None else (original,shape),{})
        result['layout_shapes'].append(describe('fp8_'+name,value,original))
    transposed=fp8_layout._make_fp8_shape_handler(torch.ops.aten.t.default)(original,(original,),{})
    for name,op in [('view_noncontiguous',torch.ops.aten.view.default),('reshape_noncontiguous',torch.ops.aten.reshape.default)]:
        try:
            value=fp8_layout._make_fp8_shape_handler(op)(transposed,(transposed,(32,)),{})
            result['layout_shapes'].append(describe('fp8_'+name,value,original))
        except RuntimeError as error:
            reject_resource_failure(error)
            result['layout_shapes'].append({'device':device,'operation':'fp8_'+name,'error':type(error).__name__})
    data=torch.arange(32,device=device,dtype=torch.int8).reshape(4,8)
    qt=tensor_base.QuantizedTensor(data,'TensorWiseINT8Layout',TensorWiseINT8Layout.Params(scale=torch.tensor(0.5,device=device),orig_dtype=torch.float16,orig_shape=(4,8)))
    transposed=int8_layout._handle_int8_transpose(qt,(qt,),{})
    result['layout_shapes'].append(describe('int8_transpose',transposed,qt))
    restored=int8_layout._handle_int8_transpose(transposed,(transposed,),{})
    result['layout_shapes'].append(describe('int8_transpose_twice',restored,qt))
    for shape,transposed in [((4,8),False),((3,5),False),((5,3),True)]:
        value=tensor_base.QuantizedTensor(data,'TensorWiseINT8Layout',TensorWiseINT8Layout.Params(scale=torch.tensor(0.5,device=device),orig_dtype=torch.float16,orig_shape=shape,transposed=transposed))
        decoded=value.dequantize()
        result['layout_dequantization'].append({'device':device,'shape':shape,'transposed':transposed,'output_shape':list(decoded.shape),'strides':list(decoded.stride()),'bytes':bits(decoded)})
    for scale_shape in [(),(1,),(4,1)]:
        for is_weight in [False,True]:
            for convrot in [False,True]:
                value=tensor_base.QuantizedTensor(data,'TensorWiseINT8Layout',TensorWiseINT8Layout.Params(scale=torch.full(scale_shape,0.5,device=device),orig_dtype=torch.float16,orig_shape=(4,8),is_weight=is_weight,convrot=convrot,convrot_groupsize=64))
                result['requantization_metadata'].append({'device':device,'scale_shape':scale_shape,'is_weight':is_weight,'convrot':convrot,'kwargs':TensorWiseINT8Layout.requantize_kwargs(value)})
for device in devices:
    layout_cases(device)
    packed_layout_cases(device)
    for dtype in [torch.float32,torch.float16,torch.bfloat16]:
        for shape in [(3,16),(128,32),(129,48)]:
            for hi_first in [False,True]:
                packed=torch.arange(shape[0]*shape[1],dtype=torch.int32,device=device).to(torch.uint8).reshape(shape)
                scales=(torch.arange(shape[0]*(shape[1]//8),dtype=torch.float32,device=device).remainder(11)*0.125).reshape(shape[0],shape[1]//8).to(torch.float8_e4m3fn)
                tiled=float_utils.to_blocked(scales,flatten=False)
                decoded=eager.dequantize_nvfp4(packed,torch.tensor(0.7,device=device),tiled,output_type=dtype,hi_first=hi_first)
                result['nvfp4_eager_decode'].append({'device':device,'dtype':str(dtype),'shape':shape,'hi_first':hi_first,'bytes':bits(decoded),'output_shape':list(decoded.shape),'strides':list(decoded.stride())})
    for shape in [(3,5),(128,4),(129,7)]:
        for transposed in [False,True]:
            raw=torch.arange(shape[0]*shape[1],dtype=torch.int32,device=device).to(torch.uint8).reshape(shape)
            if transposed:raw=raw.t()
            for flatten in [False,True]:
                blocked=float_utils.to_blocked(raw,flatten=flatten)
                restored=float_utils.from_blocked(blocked,*raw.shape)
                result['block_scales'].append({'device':device,'shape':shape,'transposed':transposed,'flatten':flatten,'output_shape':list(blocked.shape),'strides':list(blocked.stride()),'bytes':bits(blocked),'restored':bits(restored),'restored_strides':list(restored.stride())})
    result['e8m0_decode'].append({'device':device,'bytes':bits(float_utils.e8m0_to_f32(torch.arange(256,dtype=torch.int32,device=device).to(torch.uint8)))})
    for dtype in [torch.float32,torch.float16,torch.bfloat16]:
        for shape,pad in [((3,64),False),((17,65),True),((128,256),False)]:
            x=(torch.arange(shape[0]*shape[1],dtype=torch.float32,device=device)*0.03125-16).reshape(shape).to(dtype)
            x[0,:32]=0
            q,s=eager.quantize_mxfp8(x,pad_32x=pad)
            decoded=eager.dequantize_mxfp8(q,s,output_type=dtype)
            result['mxfp8_eager'].append({'device':device,'dtype':str(dtype),'shape':shape,'pad':pad,'storage_shape':list(q.shape),'scale_shape':list(s.shape),'payload':bits(q),'scale':bits(s),'decoded':bits(decoded)})
    generator=torch.Generator(device=device).manual_seed(42)
    initial=bits(generator.get_state())
    draws=[]
    for kind in [torch.float32,torch.float16,torch.bfloat16]:
        draws.append({'kind':str(kind),'shape':[4,16],'bytes':bits(torch.randn((4,16),generator=generator,dtype=kind,device=device))})
    draws.append({'kind':'torch.uint8','shape':[4,16],'bytes':bits(torch.randint(0,256,(4,16),generator=generator,dtype=torch.uint8,device=device))})
    result['rng'].append({'device':device,'seed':42,'initial_state':initial,'draws':draws,'final_state':bits(generator.get_state())})
    for dtype in [torch.float32,torch.float16,torch.bfloat16]:
        x=(torch.arange(64,device=device,dtype=torch.float32)*0.125-4).reshape(4,16).to(dtype)
        x[0].zero_()
        for rowwise in [False,True]:
            for seed in [None,0,123]:
                for scale_mode in (['recalculate'] if rowwise else ['recalculate','zero','explicit']):
                    scale=None if scale_mode=='recalculate' else torch.tensor(0. if scale_mode=='zero' else 0.03125,device=device,dtype=torch.float32)
                    q,s=eager.quantize_int8_rowwise(x,stochastic_rounding=seed) if rowwise else eager.quantize_int8_tensorwise(x,scale=scale,stochastic_rounding=seed)
                    result['int8'].append({'device':device,'dtype':str(dtype),'rowwise':rowwise,'seed':seed,'scale_mode':scale_mode,'payload':bits(q),'scale':bits(s),'decoded':bits(eager.dequantize_int8_simple(q,s))})
    if device!='cpu':
        for dtype in [torch.float32,torch.float16,torch.bfloat16]:
            for shape,pad in [((32,64),False),((17,65),True)]:
                x=(torch.arange(shape[0]*shape[1],device=device,dtype=torch.float32)*0.03125-16).reshape(shape).to(dtype)
                for hi_first in [False,True]:
                    scale=torch.tensor(0.7,device=device,dtype=torch.float32)
                    case={'operation':'nvfp4','dtype':str(dtype),'shape':shape,'pad':pad,'hi_first':hi_first}
                    try:
                        q,s=native.quantize_nvfp4(x,scale,pad_16x=pad,hi_first=hi_first)
                        decoded=native.dequantize_nvfp4(q,scale,s,output_type=dtype,hi_first=hi_first)
                        case.update(payload=bits(q),scale=bits(s),decoded=bits(decoded))
                    except RuntimeError as error:
                        reject_resource_failure(error)
                        case['error']=str(error)
                    result['native_quantizers'].append(case)
                case={'operation':'mxfp8','dtype':str(dtype),'shape':shape,'pad':pad}
                try:
                    q,s=native.quantize_mxfp8(x,pad_32x=pad)
                    case.update(payload=bits(q),scale=bits(s))
                except RuntimeError as error:
                    reject_resource_failure(error)
                    case['error']=str(error)
                result['native_quantizers'].append(case)
            for seed in [None,0,123]:
                x=(torch.arange(256,device=device,dtype=torch.float32)*0.03125-4).reshape(4,64).to(dtype)
                q,s=native.quantize_int8_rowwise(x,stochastic_rounding=seed)
                result['native_quantizers'].append({'operation':'int8_rowwise','dtype':str(dtype),'seed':seed,'payload':bits(q),'scale':bits(s)})
    payload=torch.arange(32,device=device,dtype=torch.int8).reshape(4,8).t()
    params=TensorWiseINT8Layout.Params(scale=torch.tensor([0.5],device=device),orig_dtype=torch.float16,orig_shape=(8,4))
    original=tensor_base.QuantizedTensor(payload,'TensorWiseINT8Layout',params)
    def storage_case(name,value,initialized=True):
        return {'device':device,'operation':name,'shape':list(value._params.orig_shape),'kind':str(value._params.orig_dtype),'storage_kind':str(value._qdata.dtype),'storage_shape':list(value._qdata.shape),'strides':list(value._qdata.stride()),'payload':bits(value._qdata) if initialized else None,'scale':bits(value._params.scale),'payload_alias':value._qdata.data_ptr()==payload.data_ptr(),'scale_alias':value._params.scale.data_ptr()==params.scale.data_ptr()}
    operations={
        'noop':tensor_base._handle_to(original,(original,),{}),
        'dtype':tensor_base._handle_to(original,(original,),{'dtype':torch.bfloat16}),
        'clone':tensor_base._handle_clone(original,(original,),{}),
        'detach':tensor_base._handle_detach(original,(original,),{}),
        'contiguous':tensor_base._handle_contiguous(original,(original,),{}),
        'force_copy':tensor_base._handle_to_copy(original,(original,),{}),
        'cpu':tensor_base._handle_to(original,(original,),{'device':torch.device('cpu')}),
        'empty_like':tensor_base._handle_empty_like(original,(original,),{'dtype':torch.bfloat16}),
    }
    result['storage'].extend(storage_case(name,value,name!='empty_like') for name,value in operations.items())
    source=tensor_base.QuantizedTensor((torch.arange(32,device=device,dtype=torch.int16)+100).to(torch.int8).reshape(8,4),'TensorWiseINT8Layout',TensorWiseINT8Layout.Params(scale=torch.tensor([2.],device=device),orig_dtype=torch.bfloat16,orig_shape=(8,4)))
    tensor_base._handle_copy_(original,(original,source),{'non_blocking':False})
    result['storage'].append(storage_case('copy_from',original))
    for dtype,scalar,code,bits_count,lanes in [(torch.float8_e8m0fnu,44,14,8,1),(torch.float4_e2m1fn_x2,45,17,4,2)]:
        raw=torch.tensor([0,127,128,255],dtype=torch.uint8,device=device)
        value=raw.view(dtype)
        result['shell_dtypes'].append({'device':device,'dtype':str(dtype),'scalar_type':scalar,'dlpack_code':code,'bits':bits_count,'lanes':lanes,'element_bytes':value.element_size(),'raw':bits(value),'aliased':raw.data_ptr()==value.data_ptr()})
    for input_kind in [torch.float32,torch.float16,torch.bfloat16]:
        x=(torch.arange(64,device=device,dtype=torch.float32)*0.125-4).reshape(4,16).to(input_kind)
        scale=torch.tensor(0.7,device=device,dtype=torch.float32)
        for output_kind in [torch.float8_e4m3fn,torch.float8_e5m2]:
            backend=eager if device=='cpu' else native
            quantized=backend.quantize_per_tensor_fp8(x,scale,output_kind)
            decoded=backend.dequantize_per_tensor_fp8(quantized,scale,input_kind)
            case={'device':device,'input_kind':str(input_kind),'output_kind':str(output_kind),'quantized':bits(quantized),'dequantized':bits(decoded)}
            if device!='cpu':
                rng=torch.randint(0,256,x.shape,device=device,dtype=torch.uint8,generator=torch.Generator(device=device).manual_seed(123))
                case['rounding_rng']=bits(rng)
                rounded=native.stochastic_rounding_fp8(x,rng,output_kind)
                case['rounded']=bits(rounded)
            result['fp8'].append(case)
for device in devices:
    for width in [16,17]:
        for explicit in [None,'cuda','eager','missing']:
            for override in [None,'cuda','eager']:
                registry=BackendRegistry()
                registry.__dict__['_compute_capability']=(8,9)
                constraints=FunctionConstraints(params={'x':ParamConstraint(dtypes=frozenset({torch.float32}),shape_rules=(ExactDims(2),DivisibleBy(1,16)))},default_devices=frozenset({'cuda'}),min_compute_capability=(8,0))
                registry.register('cuda',SimpleNamespace(quantize=lambda:'cuda'),{'quantize':constraints})
                registry.register('eager',SimpleNamespace(quantize=lambda:'eager'),{'quantize':FunctionConstraints()})
                kwargs={'x':torch.empty((4,width),device=device,dtype=torch.float32)}
                case={'device':device,'width':width,'explicit':explicit,'override':override}
                try:
                    if override:
                        with registry.use_backend(override):selected=registry.get_implementation('quantize',backend=explicit,kwargs=kwargs)()
                    else:selected=registry.get_implementation('quantize',backend=explicit,kwargs=kwargs)()
                    case['selected']=selected
                except Exception as error:
                    reject_resource_failure(error)
                    case['error']=type(error).__name__
                result['registry'].append(case)
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,indent=2)+'\n')
print('Generated synthetic fixtures:',{key:len(value) for key,value in result.items() if isinstance(value,list)})
