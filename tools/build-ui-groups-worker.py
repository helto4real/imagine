#!/usr/bin/env python3
"""Isolated H3 worker source/target for the approved UI component transport."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / 'comfyui/ui-groups-20261021'
SOURCE = CANDIDATE / 'source'
BASELINE = CANDIDATE / 'worker-baseline'
OWN = Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')
BASE = 'cd0ef53be69dd539ef0ffdf0cb03f4c1c5835fc7'

def prepare():
    if not BASELINE.exists():
        BASELINE.mkdir()
        archive = subprocess.check_output(['git', 'archive', BASE, 'inference', 'design/h3-stage4-evidence/selected-model-sha256.json'], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive)) as files:
            files.extractall(BASELINE, filter='data')
        overlay = ['inference/src/models/minimax_h3/sampling.rs'] + [f'inference/src/h3_product/{name}.rs' for name in ['mod','native','deployment','wire','image','sampling','motion','references','reference_video','reference_window']]
        for name in overlay:
            dest = BASELINE / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, dest)
        (BASELINE/'prepared.json').write_text(json.dumps({'frozen_base':BASE,'overlay':overlay},indent=2))
    if not (SOURCE/'inference').exists():
        shutil.copytree(BASELINE/'inference', SOURCE/'inference')
        for path in [Path('inference/tools/h3-product-cpu/Cargo.toml'),Path('inference/tools/h3-product-cpu/Cargo.lock'),*(ROOT/'inference/tools/h3-product-cpu/src').rglob('*.rs'),*(ROOT/'inference/tools/h3-product-cpu/tests').rglob('*.rs')]:
            relative=path.relative_to(ROOT) if path.is_absolute() else path
            dest=SOURCE/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,dest)
        for name in ['design/h3-stage4-evidence/selected-model-sha256.json','design/h3-i2v-model-lock.json']:
            dest=SOURCE/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(BASELINE/name if (BASELINE/name).exists() else ROOT/name,dest)

def environment():
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(OWN)]))['filesystems'][0]
    assert (mount['target'],mount['source'],mount['fstype'])==('/home','/dev/mapper/cryptroot[/@home]','btrfs') and not os.statvfs(OWN).f_flag & os.ST_RDONLY
    state=OWN/'worker-build';state.mkdir(exist_ok=True)
    env=dict(os.environ,CARGO_HOME=str(ROOT/'comfyui/h3-video-night-20261021/cargo-native'),CARGO_TARGET_DIR=str(CANDIDATE/'target-worker'),CARGO_BUILD_JOBS='2',CARGO_INCREMENTAL='0',CARGO_PROFILE_DEV_DEBUG='0',CARGO_PROFILE_TEST_DEBUG='0',RUSTFLAGS='-C debuginfo=0',LIBTORCH='/home/thhel/git/ComfyUI/.venv/lib/python3.12/site-packages/torch',LIBTORCH_CXX11_ABI='1',CXX='clang++',IMAGINE_CUDA_INCLUDE=str(ROOT/'comfyui/moss-20261018/cuda13/nvidia/cu13/include'),LD_LIBRARY_PATH='/home/thhel/git/ComfyUI/.venv/lib/python3.12/site-packages/torch/lib',TMPDIR=str(OWN/'tmp'),CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    for key,name in [('XDG_CACHE_HOME','cache'),('TORCH_HOME','torch'),('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda'),('HF_HOME','hf')]:
        path=state/name;path.mkdir(exist_ok=True);env[key]=str(path)
    env.pop('LIBTORCH_USE_PYTORCH',None);env.pop('LIBTORCH_BYPASS_VERSION_CHECK',None)
    return env

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','build','clippy','fmt','wire','cache_cpu','dynamic_cpu','preview_cpu','boundary_cpu','shifts','seed','reference_quality','upscale','loras','text_probe','text_probe_clippy']);args=parser.parse_args()
    env=environment();prepare()
    if args.mode=='prepare':print(SOURCE/'inference');return 0
    run='worker-'+uuid.uuid4().hex[:8];diagnostics=OWN/'evidence'/f'{run}.txt'
    if args.mode=='wire':
        env['CARGO_TARGET_DIR']=str(CANDIDATE/'target-wire')
        command=['cargo','test','--manifest-path',str(SOURCE/'inference/tools/h3-product-cpu/Cargo.toml'),'--locked','--offline']
    elif args.mode=='cache_cpu':command=['cargo','test','--manifest-path',str(SOURCE/'inference/Cargo.toml'),'--locked','--offline','--lib','h3_product::reference_cache::tests','--','--test-threads=1']
    elif args.mode=='dynamic_cpu':command=['cargo','test','--manifest-path',str(SOURCE/'inference/Cargo.toml'),'--locked','--offline','--lib','checkpoint::dynamic','--','--test-threads=1']
    elif args.mode=='boundary_cpu':command=['cargo','test','--manifest-path',str(SOURCE/'inference/Cargo.toml'),'--locked','--offline','--lib','h3_product::reference_window::tests','--','--test-threads=1']
    elif args.mode=='preview_cpu':
        env['IMAGINE_H3_PREVIEW_ORACLE']=str(OWN/'g11-preview-components-cp29-20261022-01/director-preview-oracle.json')
        command=['cargo','test','--manifest-path',str(SOURCE/'inference/Cargo.toml'),'--locked','--offline','--lib','h3_product::preview','--','--include-ignored','--test-threads=1']
    elif args.mode=='fmt':command=['cargo','fmt','--manifest-path',str(SOURCE/'inference/Cargo.toml'),'--all']
    elif args.mode in ['text_probe','text_probe_clippy']:
        command=['cargo','clippy'if args.mode=='text_probe_clippy'else'build','--manifest-path',str(SOURCE/'inference/Cargo.toml'),'--locked','--offline','--example','h3_ui_text_component']
        if args.mode=='text_probe_clippy':command+=['--','-D','warnings']
    elif args.mode in ['upscale','loras']:command=['cargo','build','--manifest-path',str(SOURCE/'inference/Cargo.toml'),'--locked','--offline','--example','h3_ui_upscale_component'if args.mode=='upscale'else'h3_ui_loras_cpu']
    elif args.mode in ['shifts','seed','reference_quality']:command=['cargo','build','--manifest-path',str(SOURCE/'inference/Cargo.toml'),'--locked','--offline','--example','h3_ui_'+args.mode+'_cpu']
    else:
        command=['cargo',args.mode,'--manifest-path',str(SOURCE/'inference/Cargo.toml'),'--locked','--offline','--bin','imagine-h3-worker']
        if args.mode=='clippy':command+=['--','-D','warnings']
    with diagnostics.open('x')as output:result=subprocess.run(command,cwd=SOURCE,env=env,stdout=output,stderr=subprocess.STDOUT)
    report={'exit':result.returncode,'diagnostics':str(diagnostics),'mode':args.mode,'target':env['CARGO_TARGET_DIR'],'activated':False}
    if not result.returncode and args.mode=='upscale':report['probe_binary']=str(CANDIDATE/'target-worker/debug/examples/h3_ui_upscale_component')
    if not result.returncode and args.mode in ['shifts','seed','reference_quality']:
        probe=OWN/'evidence'/f'{run}-{args.mode}.json'
        values=subprocess.check_output([str(CANDIDATE/('target-worker/debug/examples/h3_ui_'+args.mode+'_cpu'))],env=env,cwd=SOURCE)
        with probe.open('xb')as output:output.write(values)
        report['probe_output']=str(probe)
    if not result.returncode and args.mode=='build':
        package=CANDIDATE/'worker-packages'/run;package.mkdir(parents=True);binary=package/'imagine-h3-worker';shutil.copy2(CANDIDATE/'target-worker/debug/imagine-h3-worker',binary)
        report.update(package=str(package),sha256=hashlib.sha256(binary.read_bytes()).hexdigest());(package/'candidate.json').write_text(json.dumps(report,indent=2))
    (OWN/'evidence'/f'{run}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
    if result.returncode:print('\n'.join(diagnostics.read_text().splitlines()[-25:]))
    return result.returncode

if __name__=='__main__':raise SystemExit(main())
