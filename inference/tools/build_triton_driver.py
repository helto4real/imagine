#!/usr/bin/env python3
"""Preserve original driver bodies; replace only Python ownership/error binding."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--site',type=Path,required=True);a=p.parse_args()
project=Path(__file__).resolve().parents[2];manifest=json.loads((project/'design/reference-local/source-manifest.json').read_text());name='triton/backends/nvidia/driver.c';path=a.site/name
actual=hashlib.sha256(path.read_bytes()).hexdigest();assert actual==next(e['sha256']for e in manifest if e['root']=='site'and e['path']==name)
src=path.read_text();start=src.index('static void _launch(');end=src.index('\nstatic PyObject *data_ptr_str',start);launch=src[start:end]
start=src.index('  CUfunction fun;',src.index('static PyObject *loadBinary('));end=src.index('  Py_END_ALLOW_THREADS;',start);load=src[start:end].replace('  Py_BEGIN_ALLOW_THREADS;','')
load='static void original_load(const char* name,const void* data,int shared,int device,void** module,void** function,int* info) {\n'+load+'\n*module=mod;*function=fun;info[0]=n_regs;info[1]=n_spills;info[2]=n_max_threads;\n}\n'
out=a.root/'triton-driver';out.mkdir(exist_ok=True);generated=out/'original_triton_driver.inc';generated.write_text(launch+'\n'+load)
binary=out/'libimagine_triton_driver.so';cmd=['clang++','-std=c++20','-shared','-fPIC','-O3','-I',str(a.site/'triton/backends/nvidia/include'),'-I',str(out),str(project/'inference/native/triton_driver.cpp'),'-L/usr/lib','-l:libcuda.so.1','-ldl','-o',str(binary)]
subprocess.run(cmd,check=True)
report=dict(schema=1,source_driver_sha256=actual,original_launch_body_sha256=hashlib.sha256(launch.encode()).hexdigest(),generated_sha256=hashlib.sha256(generated.read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),command=cmd,scope='original load/launch plus typed pointer marshalling; Python object/extractor layer not present')
(out/'build.json').write_text(json.dumps(report,indent=2)+'\n');print('Original NVIDIA driver load/launch binding built',flush=True)
