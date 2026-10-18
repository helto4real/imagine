#!/usr/bin/env python3
"""Build the matching original Pillow imaging core with ownership-only C hooks."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1]/'resources/pillow-12.3.0');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
project=Path(__file__).resolve().parents[2]
lock=json.loads((project/'inference/resources/pillow-source-lock.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,h in lock['source_sha256'].items():assert digest(a.source/name)==h,name
a.output.mkdir(parents=True,exist_ok=True)
subprocess.run(['cmake','-S',str(project/'inference/native/pillow'),'-B',str(a.output),'-DPILLOW_SOURCE='+str(a.source.resolve()),'-DCMAKE_C_COMPILER=clang'],check=True)
subprocess.run(['cmake','--build',str(a.output),'-j2'],check=True)
library=a.output/'libimagine_pillow.so'
undefined=subprocess.check_output(['nm','-D','--undefined-only',str(library)],text=True)
assert not any(line.split()[-1].startswith(('Py','_Py'))for line in undefined.splitlines()if line.split())
linked=subprocess.check_output(['ldd',str(library)],text=True);assert'not found'not in linked and'python'not in linked.lower()
provenance=dict(schema=1,upstream=lock,source_root=str(a.source.resolve()),library_sha256=digest(library),adapter_sha256={str(f.relative_to(project)):digest(f)for f in sorted((project/'inference/native/pillow').rglob('*'))if f.is_file()},undefined_symbols=undefined,linked_libraries=linked,python_runtime_required=False,transformation='Original core C/header files unchanged; buffer ownership, allocator gate/GIL-section and error hooks supplied by standalone C ABI')
(a.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('Prepared original Pillow core without Python:',provenance['library_sha256'])
