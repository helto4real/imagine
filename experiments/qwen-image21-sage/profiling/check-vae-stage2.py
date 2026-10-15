"""Run the stage-2 component and whole-VAE memory gates; no image inference."""
from pathlib import Path
import argparse, hashlib, json, os, re, subprocess
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'experiments/qwen-image21-sage/baselines/1600x896-v1'
OUT=ROOT/'experiments/qwen-image21-sage/runs/vae-stage2-strips-build'
p=argparse.ArgumentParser();p.add_argument('--model-root',type=Path,required=True);a=p.parse_args()
flags=json.loads((BASE/'baseline-environment.json').read_text())
env={k:v for k,v in os.environ.items() if not k.startswith(('SD_','GGML_'))}
assert not env.get('LD_PRELOAD')
env.update(flags)
vae=a.model_root/'vae/qwen-image2.1/qwen_image_2.1_vae_bf16.safetensors'
status={};memory={}
for name,args in [('probe-control',['measure-vae-stage2','0',str(vae)]),('probe-candidate',['measure-vae-stage2','1',str(vae)]),('synthetic-production',['test-vae-stage2-strips','full']),('synthetic-fallbacks',['test-vae-stage2-strips'])]:
    with (OUT/(name+'.log')).open('w') as log:
        r=subprocess.run([str(OUT/args[0]),*args[1:]],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    status[name]=r.returncode
    text=(OUT/(name+'.log')).read_text()
    print(name,r.returncode,text,flush=True)
    (OUT/'check-status.json').write_text(json.dumps(status,indent=2)+'\n')
    assert r.returncode==0, name
    if name.startswith('probe'):
        assert 'UNSUPPORTED' not in text
        memory[name]=float(re.search(r'direct=([\d.]+) MiB scheduler=0',text)[1])
        if name=='probe-control':
            assert memory[name]==8073.242432 and 'nodes=2265' in text, 'Control does not match frozen full graph'
prod=(OUT/'synthetic-production.log').read_text()
assert prod.count('bytes=825753600')==2 and 'PASS all comparisons and retained-input integrity' in prod
summary={'workspace_mib':memory,'component_bitidentical':True,'component_bytes':825753600,'fallbacks_passed':True,'image_gate_passed':memory['probe-candidate']<memory['probe-control'],'flags':flags,'candidate_flag':{'SD_VAE_STRIP_STAGE2':'1'},'probe_uses_checkpoint_header_only':True}
(OUT/'gate-results.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
