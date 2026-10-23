#!/usr/bin/env python3
"""Run one approved synthetic H3 phase under the canonical Ticket/namespace.

All writable outputs and all reused tensor inputs must belong to this experiment.
No product deployment, service management, private input or fallback coordinator.
"""
import argparse
import array
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OWN=ROOT/'comfyui/h3-memory-20261023'
SOURCE=OWN/'source'
sys.path.insert(0,str(SOURCE/'inference/tools'))
from h3_gpu_commands import PYTHON_ROOT, native_command, original_command, sha


def run(args):
    assert re.fullmatch('[a-zA-Z0-9_-]+',args.name)
    work=OWN/'runs'/args.name/args.engine
    work.mkdir(parents=True,exist_ok=True)
    output=work/args.phase
    assert not output.exists(), 'Never overwrite retained evidence'
    if args.inputs:
        inputs=args.inputs.resolve()
        assert inputs.is_relative_to(OWN/'runs')
        for phase in ('text','reference','dit'):
            if phase==args.phase or not(inputs/phase).is_dir():continue
            destination=work/phase
            if destination.exists():continue
            destination.mkdir()
            names={'text':['context','tags'],'reference':['latent'],'dit':['video-latent','audio-latent']}[phase]
            for name in names:
                for suffix in ('bin','json'):
                    source=inputs/phase/(name+'.'+suffix)
                    subprocess.run(['cp','--reflink=auto','--',str(source),str(destination/source.name)],check=True)
                    assert sha(source)==sha(destination/source.name)
    if args.reference:
        data=array.array('f')
        for y in range(256):
            for x in range(384):data.extend((.1,.25,.8)if 112<=x<272 and 48<=y<208 else(.75,.7,.6))
        raw=data.tobytes(); p=work/'synthetic-reference.bin'
        if p.exists():assert p.read_bytes()==raw
        else:p.write_bytes(raw)
        (work/'synthetic-reference.json').write_text(json.dumps(dict(shape=[1,256,384,3],dtype='float32',sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))))
    profile=SOURCE/'inference/profiles/h3-short-av.json'
    lock=SOURCE/'design/h3-stage4-evidence/selected-model-sha256.json'
    spec=json.loads(profile.read_text());locks=json.loads(lock.read_text());assert locks['result']=='passed'
    role={'text':'text','reference':'video_vae','dit':'dit','video':'video_vae','audio':'audio_vae'}[args.phase]
    entry=next(v for v in locks['components']if v['role']==role)
    modelroot=Path('/mnt/media/gemensam/AI/ComfyUI/models')
    checkpoint=modelroot/spec['checkpoints'][role]
    assert checkpoint==Path(entry['path']) and checkpoint.stat().st_size==entry['file_bytes']
    site=Path('/home/thhel/git/ComfyUI/.venv/lib/python3.12/site-packages')
    dependencies=ROOT/'comfyui/moss-20261018'
    retained=Path('/home/thhel/.codex/worktrees/be07/imagine/comfyui/h3-stage4-20261020')
    bridge=retained/'native-build/kitchen-manylinux'
    os.environ['IMAGINE_H3_KITCHEN_LIBRARY']=str(bridge/'libimagine_kitchen.so')
    os.environ['IMAGINE_H3_KITCHEN_PROVENANCE']=str(bridge/'provenance.json')
    readonly=[(profile,profile),(lock,lock),(checkpoint,checkpoint)]
    environment=dict(H3_MEMORY_GEOMETRY=args.geometry,H3_MEMORY_VARIANT=args.variant,H3_MEMORY_CAPTURE=str(int(args.capture)))
    if args.oom:environment['H3_MEMORY_OOM']='1'
    record=dict(schema=1,engine=args.engine,variant=args.variant,geometry=args.geometry,reference=args.reference,capture=args.capture,trace=args.trace,oom=args.oom,phase=args.phase,result='preflight',output_root=str(work),selected_component=entry,profile_sha256=sha(profile),model_lock_sha256=sha(lock))
    evidence=work/(args.phase+'-selection.json')
    assert not evidence.exists()
    if args.engine=='native':
        built=OWN/'target/debug/h3-memory-probe'
        digest=sha(built)
        binary=work/('worker-'+digest)
        if not binary.exists():
            subprocess.run(['cp','--reflink=auto','--',str(built),str(binary)],check=True)
            binary.chmod(0o500)
        assert sha(binary)==digest
        command=native_command(SOURCE,dependencies,site,binary,work)
        record['binary_sha256']=sha(binary)
        for src,dst in readonly:command+=['--ro-bind',str(src),str(dst)]
        command+=['--bind',str(work),str(work)]
        worker=['/runtime/primitives','--profile',str(profile),'--model-lock',str(lock),'--model-root',str(modelroot),'--own-root',str(work),'--phase',args.phase]
        if args.reference:worker+=['--reference']
    else:
        script=SOURCE/'inference/tools/reference_h3_memory.py'
        files=[script,SOURCE/'inference/tools/reference_h3_loading.py',SOURCE/'inference/tools/reference_paths.py']
        for name in ('h3-schema.json','h3-component-modules.json'):
            p=SOURCE/'inference/resources'/name;readonly.append((p,p))
        command=original_command(SOURCE,site,work,files,readonly)
        environment.update(IMAGINE_H3_PROFILE=str(profile),IMAGINE_H3_MODEL_LOCK=str(lock),IMAGINE_H3_MODEL_ROOT=str(modelroot),IMAGINE_H3_PHASE=args.phase,IMAGINE_H3_REFERENCE=str(int(args.reference)),IMAGINE_H3_LOAD_ONLY='0',IMAGINE_H3_SCHEMA=str(SOURCE/'inference/resources/h3-schema.json'),IMAGINE_H3_MODULES=str(SOURCE/'inference/resources/h3-component-modules.json'))
        worker=[str(PYTHON_ROOT/'bin/python'),'-S','-B',str(script),'--site',str(site),'--comfy','/home/thhel/git/ComfyUI','--own-root',str(work),'--output',str(output/'phase.json')]
        record['oracle_sha256']=sha(script)
    if args.trace:
        library=OWN/'activity.so';assert library.is_file()
        command+=['--ro-bind',str(library),'/runtime/h3-activity.so','--ro-bind','/opt/cuda/lib64/libcupti.so.13','/runtime/libcupti.so.13']
        environment.update(LD_PRELOAD='/runtime/h3-activity.so',QWEN_ACTIVITY_TRACE=str(work/(args.phase+'-activity.tsv')))
    for k,v in environment.items():command+=['--setenv',k,v]
    command+=['--',*worker]
    manifest=work/(args.phase+'-reservation.json')
    # Explicitly selected by Tomas: existing private-storage product coordinator.
    # Never probe other addresses or fall back to uncoordinated execution.
    runtime=Path(os.environ['XDG_RUNTIME_DIR']);assert runtime.is_absolute()
    socket=runtime/'imagine/gpu.sock'
    controller=retained/'target-reservation/debug/imagine-native-gpu-reservation'
    namespace_entry=retained/'target-reservation/debug/namespace_entry'
    command=[str(controller),'--process-form','bwrap-pid-namespace','--namespace-entry',str(namespace_entry),'--socket',str(socket),'--job','h3-memory-'+args.name+'-'+args.engine+'-'+args.phase,'--bytes','1','--timeout','1800','--manifest',str(manifest),'--','/usr/bin/bwrap',*command]
    record['command']=command;record['result']='running'
    evidence.write_text(json.dumps(record,indent=2)+'\n')
    started=time.monotonic();cancelled=False
    with(work/(args.phase+'-probe.log')).open('w')as log:
        process=subprocess.Popen(command,cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT)
        try:
            while process.poll()is None:
                metrics=output/'metrics.jsonl'
                if args.cancel and not cancelled and metrics.exists() and 'h3.sampling.0.x' in metrics.read_text():
                    process.send_signal(signal.SIGTERM);cancelled=True
                time.sleep(.1)
        except BaseException:
            process.send_signal(signal.SIGTERM);process.wait();raise
        record['returncode']=process.wait()
    record['seconds']=time.monotonic()-started
    if manifest.exists():record['reservation']=json.loads(manifest.read_text())
    lifecycle=record.get('reservation',{}).get('report',{})
    passed=(record['returncode']==0 and not args.cancel and not args.oom)or(cancelled and record['returncode']!=0 and lifecycle.get('worker',{}).get('state')=='cancelled')
    if args.oom and (output/'phase.json').exists():
        phase=json.loads((output/'phase.json').read_text())
        passed=record['returncode']!=0 and 'out of memory' in phase.get('error','').lower() and 'cleanup_error'not in phase and phase['unloaded']['placement']['loaded_bytes']==0 and phase['unloaded']['pinned_bytes']==0
    if args.trace and not args.cancel:
        trace=work/(args.phase+'-activity.tsv')
        passed=passed and trace.exists() and trace.read_text().rstrip().endswith('# dropped_records=0')
    passed=passed and lifecycle.get('namespace_exited')is True and lifecycle.get('worker',{}).get('finish_sent')is True
    record['result']='passed'if passed else'failed';record['cancelled_after_step']=cancelled
    evidence.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k]for k in ('engine','variant','geometry','phase','result','returncode','seconds')}),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True)
    p.add_argument('--engine',choices=['native','original'],required=True)
    p.add_argument('--variant',choices=['control','candidate'],required=True)
    p.add_argument('--geometry',choices=['short','large'],required=True)
    p.add_argument('--phase',choices=['text','reference','dit','video','audio'],required=True)
    p.add_argument('--inputs',type=Path)
    for flag in ('capture','reference','trace','cancel','oom'):p.add_argument('--'+flag,action='store_true')
    a=p.parse_args();assert not a.cancel or a.phase=='dit'
    assert not a.oom or (a.engine=='native'and a.phase=='dit'and not a.cancel)
    run(a)
