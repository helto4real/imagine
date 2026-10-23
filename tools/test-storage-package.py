#!/usr/bin/env python3
"""Bounded immutable-storage-package smoke; no private normal startup or jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import struct
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
E=Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package',type=Path)
    parser.add_argument('case')
    args=parser.parse_args()
    package=args.package.resolve()
    assert package.parent==ROOT/'comfyui/products'
    assert args.case.isascii() and args.case.isalnum() and args.case.startswith('storage')
    case=E/args.case;case.mkdir()
    for name in ['home','runtime','tmp','models','cache']:(case/name).mkdir()
    active=ROOT/'comfyui/active-product.sh';before=active.read_bytes()
    metadata=json.loads((package/'package.json').read_text())
    manifest=json.loads((package/'runtime.json').read_text())
    assert metadata['private_storage'] and not metadata['selected_for_next_start']
    assert manifest['h3']['identity']['protocol']==16
    assert 'IMAGINE_REFERENCE_PROBE_OUTPUT' not in manifest['h3']['environment']
    for name,expected in metadata['proxy_sha256'].items():
        assert hashlib.sha256((package/name).read_bytes()).hexdigest()==expected
    for name,expected in metadata['comfy_scripts_sha256'].items():
        assert hashlib.sha256((package/'comfy-worker'/name).read_bytes()).hexdigest()==expected
    env=dict(os.environ,HOME=str(case/'home'),XDG_RUNTIME_DIR=str(case/'runtime'),
             TMPDIR=str(case/'tmp'),CUDA_VISIBLE_DEVICES='',QWEN_MODEL_ROOT=str(case/'models'),
             PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
             XDG_CACHE_HOME=str(case/'cache'))
    report=dict(passed=False,package=str(package),normal_private_server_started=False,
                reason='No existing private workspace/settings may be read by acceptance',
                real_model_jobs=0,gpu=False,runtime_logs_read=False)
    worker=None
    def cli(name,*args,environment=env):
        return subprocess.run([str(package/name),*args],env=environment,cwd=case,
                              capture_output=True,timeout=20)
    def exact(pipe,count):
        output=bytearray();deadline=time.monotonic()+20
        while len(output)<count:
            remaining=deadline-time.monotonic()
            assert remaining>0 and select.select([pipe],[],[],remaining)[0]
            chunk=os.read(pipe.fileno(),count-len(output));assert chunk
            output.extend(chunk)
        return bytes(output)
    def frame():
        size=struct.unpack('>I',exact(worker.stdout,4))[0];assert 0<size<=512*1024
        value=json.loads(exact(worker.stdout,size))
        assert exact(worker.stdout,4)==b'\0'*4
        return value
    try:
        for binary in ['imagine-server','imagine-ui','st-proxy-launch']:
            result=cli(binary,'--help')
            assert result.returncode==0 and result.stdout and b'--qa-private-home' not in result.stdout
        rejected=cli('imagine-ui','--qa-private-home')
        assert rejected.returncode!=0
        missing=env.copy();missing.pop('XDG_RUNTIME_DIR')
        rejected=cli('imagine-server',environment=missing)
        assert rejected.returncode==1 and rejected.stderr==b'Imagine server startup or shutdown failed.\n'
        assert not (case/'home/.config').exists() and not (case/'home/.imagine').exists()
        report['normal_cli_and_missing_runtime_fail_closed']=True
        h3=manifest['h3'];worker_env=dict(env,**h3['environment'])
        worker_env.update(IMAGINE_H3_MODEL_ROOT=str(case/'models'),
                          IMAGINE_H3_DEPENDENCY_MAP=h3['dependency_map'],
                          IMAGINE_H3_DEPLOYMENT=json.dumps(h3['identity']))
        with (case/'worker.log').open('x') as errors:
            worker=subprocess.Popen([str(package/'imagine-h3-worker')],env=worker_env,cwd=case,
                                    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=errors)
        assert frame()==dict(type='ready',deployment=h3['identity'])
        worker.stdin.write(struct.pack('>I',0));worker.stdin.flush()
        assert frame()==dict(type='error',id=None,code='invalid_request')
        assert worker.wait(timeout=10)==1
        report['h3_ready16_and_invalid_frame']=True
        report['passed']=True
    finally:
        if worker and worker.poll() is None:
            worker.terminate()
            try:worker.wait(timeout=10)
            except subprocess.TimeoutExpired:worker.kill();worker.wait(timeout=5)
        report['own_processes_reaped']=worker is None or worker.poll() is not None
        report['active_selection_unchanged']=active.read_bytes()==before
        assert report['active_selection_unchanged']
        (case/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))
    return 0

if __name__=='__main__':raise SystemExit(main())
