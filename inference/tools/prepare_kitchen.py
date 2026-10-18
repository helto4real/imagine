#!/usr/bin/env python3
"""Generate a metadata-only adapter to pinned upstream launch wrappers."""
import hashlib
import json
import re
from pathlib import Path
import sys

project=Path(__file__).resolve().parents[2]
sources=Path(sys.argv[1]);output=Path(sys.argv[2]);output.mkdir(parents=True,exist_ok=True)
lock=json.loads((project/'design/reference-local/source-acquisition-lock.json').read_text())
# Validate transitive native headers as well as the direct compilation units.
# Never advance the reference or accept an edited fetched dependency silently.
for name in ['comfy-kitchen','cutlass','flash-attention']:
    for rel,sha in lock['repos'][name]['source_file_sha256'].items():
        path=sources/name/rel
        assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==sha,(name,rel)
source=sources/'comfy-kitchen/comfy_kitchen/backends/cuda/dlpack_bindings.cpp'
expected=lock['repos']['comfy-kitchen']['source_file_sha256']['comfy_kitchen/backends/cuda/dlpack_bindings.cpp']
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
text=source.read_text()
wrappers=text.split('NB_MODULE(_C, m) {',1)[0]
wrappers=re.sub(r'^#include <nanobind/[^>]+>\n','',wrappers,flags=re.M)
wrappers=wrappers.replace('namespace nb = nanobind;','')
for before,after in [('nb::ndarray','TensorView'),('nb::device::cuda','CudaDevice'),('nb::ndim','Dims'),('nb::dlpack::dtype_code','DTypeCode'),('nb::dlpack::dtype','DType'),('nb::object','OptionalTensorView'),('nb::cast','view_cast'),('nb::dtype','view_dtype'),('nb::dict','PlanMap')]:
    wrappers=wrappers.replace(before,after)
assert 'nb::' not in wrappers
(output/'kitchen_launch_wrappers.inc').write_text(wrappers)
registrations=re.findall(r'm\.def\(\s*"([^"]+)"\s*,\s*&([a-zA-Z0-9_]+)',text)
assert len(registrations)==len(set(x[0] for x in registrations))
(output/'kitchen_dispatch.inc').write_text('\n'.join(f'if(std::strcmp(name,"{name}")==0) {{ *result=invoke(&{fn},args,n); return; }}' for name,fn in registrations)+'\n')
cmake=(source.parent/'CMakeLists.txt').read_text()
match=re.search(r'set\(CUDA_SOURCES\s+(.*?)\n\)',cmake,re.S)
cuda_sources=match[1].split()
for rel in cuda_sources:
    path=source.parent/rel
    key='comfy_kitchen/backends/cuda/'+rel
    assert hashlib.sha256(path.read_bytes()).hexdigest()==lock['repos']['comfy-kitchen']['source_file_sha256'][key]
(output/'kernel_sources.cmake').write_text('set(KITCHEN_CUDA_SOURCES\n'+'\n'.join('  "'+str(source.parent/rel)+'"' for rel in cuda_sources)+'\n)\n')
(output/'binding-manifest.json').write_text(json.dumps({'pinned_wrapper_sha256':expected,'binding_transform':'metadata adapter only; original bodies and kernels retained','native_operations':registrations,'cuda_sources':cuda_sources,'generated_wrapper_sha256':hashlib.sha256(wrappers.encode()).hexdigest()},indent=2)+'\n')
print(f'Prepared {len(registrations)} native entrypoints and {len(cuda_sources)} unchanged CUDA sources.')

# Reuse the original Python-facing ATen error-name translation without Python.
error_source=sources/'pytorch/torch/csrc/Exceptions.cpp'
error_header=sources/'pytorch/torch/csrc/Exceptions.h'
for path in [error_source,error_header]:
    rel=str(path.relative_to(sources/'pytorch'))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==lock['repos']['pytorch']['source_file_sha256'][rel]
error_text=error_source.read_text()
start=error_text.index('static void processErrorMsgInplace(std::string& str) {')
end=error_text.index('void translate_exception_to_python(',start)
error_functions=error_text[start:end]
assert 'PyErr_' not in error_functions and 'PyObject' not in error_functions
(output/'pytorch_error_message.inc').write_text(error_functions)
(output/'pytorch-error-manifest.json').write_text(json.dumps(dict(
    source_sha256=hashlib.sha256(error_source.read_bytes()).hexdigest(),
    exception_header_sha256=hashlib.sha256(error_header.read_bytes()).hexdigest(),
    generated_sha256=hashlib.sha256(error_functions.encode()).hexdigest(),
    transformation='exact two pure C++ function bodies, enclosed in standalone private namespace',
    functions=['processErrorMsgInplace','processErrorMsg'],python_runtime_required=False),indent=2)+'\n')
