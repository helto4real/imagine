#!/usr/bin/env python3
"""Compare original linking/optimization in an empty Python-free CPU namespace."""
import argparse,hashlib,json,re,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--site',type=Path,required=True);a=p.parse_args()
reference=a.root/'triton-extern-full';output=reference/'standalone';output.mkdir(exist_ok=True)
oracle=json.loads((reference/'reference.json').read_text());binary=a.root/'triton-standalone/make/bin/imagine-triton-llvm'
def sha(path):return hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
undefined=subprocess.check_output(['nm','-D','--undefined-only',str(binary)],text=True);assert not re.search(r'\b(?:Py[A-Z]|_Py)',undefined)
linkage=subprocess.check_output(['ldd',str(binary)],text=True);assert 'not found'not in linkage
libraries=sorted(set(Path(x)for x in re.findall(r'(?:=>\s*)?(/[^\s]+)',linkage)));mounts=[]
for path in libraries:
    assert 'python'not in str(path).lower();mounts+=['--ro-bind',str(path),str(path)]
for name,h in oracle['libraries'].items():assert sha(Path(name))==h;mounts+=['--ro-bind',name,name]
base=['bwrap','--unshare-all','--die-with-parent','--new-session','--clearenv','--proc','/proc','--dev','/dev','--tmpfs','/tmp','--dir','/bin','--dir','/usr/bin','--dir','/usr/lib','--dir','/usr/lib64','--symlink','usr/lib','/lib','--symlink','usr/lib64','/lib64',*mounts,'--ro-bind',str(binary),'/bin/compiler','--ro-bind',str(reference),'/reference','--bind',str(output),'/output','--chdir','/tmp']
report=dict(schema=1,result='in progress',interpreter_available=False,python_library_available=False,gpu_available=False,network_namespace='isolated',compiler_sha256=sha(binary),reference_sha256=sha(reference/'reference.json'),system_elf_sha256={str(path):sha(path)for path in libraries},cases=[])
try:
    for c in oracle['cases']:
        assert sha(reference/c['input'])==c['input_sha256'];target=output/f"case-{c['index']}.ll"
        result=subprocess.run([*base,'/bin/compiler','optimize','/reference/'+c['input'],'/output/'+target.name,'sm_89','+ptx87','87',*c['libraries']],capture_output=True,text=True)
        if 'error'in c:
            assert c['error']=='ValueError'and result.returncode==3,(c,result)
            assert result.stderr.removesuffix('\n')==c['message'],(c,result.stderr)
            assert not target.exists()
            detail=dict(index=c['index'],error='exact ValueError/message')
        else:
            assert result.returncode==0,(c,result.stderr)
            assert sha(reference/c['expected'])==c['expected_sha256']
            assert target.read_bytes()==(reference/c['expected']).read_bytes(),(c,sha(target))
            detail=dict(index=c['index'],optimized_llir='byte-exact',sha256=sha(target))
        report['cases'].append(detail);print(f"External LLVM case {c['index']}: exact",flush=True)
    report['result']='passed'
except Exception as error:report.update(result='failed',error=str(error));raise
finally:(a.root/'runtime/triton-extern-isolation.json').write_text(json.dumps(report,indent=2)+'\n')
