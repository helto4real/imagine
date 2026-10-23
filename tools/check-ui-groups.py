#!/usr/bin/env python3
"""Sequential isolated checks and immutable code-only UI candidate package."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
CANDIDATE=ROOT/'comfyui/ui-groups-20261021'
EVIDENCE=Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01/evidence')

def main():
    mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(EVIDENCE)]))['filesystems'][0]
    if (mount['target'],mount['source'],mount['fstype'])!=('/home','/dev/mapper/cryptroot[/@home]','btrfs') or os.statvfs(EVIDENCE).f_flag & os.ST_RDONLY:
        raise SystemExit('Own encrypted test destination is unavailable or read-only.')
    run='checks-'+uuid.uuid4().hex[:8]
    reports=[]
    package=CANDIDATE/'packages'/run
    package.mkdir(parents=True)
    tasks=[
        ['test','--locked','--offline','-p','imagine-ui','-p','imagine-video-contract','--lib'],
        ['test','--locked','--offline','-p','imagine-server','--test','video_api','--test','video_prompt_profiles','--test','video_persistence','--test','video_storage','--test','video_assembly','--test','video_jobs','--test','video_media_health','--test','lora_catalog'],
        ['test','--locked','--offline','-p','imagine-server','--lib','video::reference'],
        ['test','--locked','--offline','-p','imagine-server','--lib','sampling_preview'],
        ['test','--locked','--offline','-p','imagine-server','--lib','video::streaming_import'],
        ['test','--locked','--offline','-p','imagine-ui','--test','streaming_import'],
        ['test','--locked','--offline','-p','imagine-server','--lib','video::review'],
        ['test','--locked','--offline','-p','imagine-server','--lib','video::media_health'],
        ['test','--locked','--offline','-p','imagine-server','--lib','video::waveform'],
        ['test','--locked','--offline','-p','imagine-server','--lib','video::authoring::visuals'],
        ['test','--locked','--offline','-p','imagine-server','--lib','video::persistence::tests'],
        ['test','--locked','--offline','-p','imagine-server','--lib','video::bundle::lock_tests'],
        ['test','--locked','--offline','-p','imagine-server','--lib','video::compatibility::tests'],
        ['test','--locked','--offline','-p','imagine-server','--lib','video::adapters::h3::tests'],
        ['test','--locked','--offline','-p','imagine-server','--lib','video::adapters::h3::bindings'],
        ['test','--locked','--offline','-p','imagine-server','--test','h3_options','--test','h3_sampling','--test','h3_protocol','--test','h3_geometry','--test','k1_k2','--test','h3_pipe'],
        ['test','--locked','--offline','-p','imagine-server','--lib','enhancement::tests'],
        ['clippy','--locked','--offline','-p','imagine-server','--lib','--','-D','warnings'],
        ['clippy','--locked','--offline','-p','imagine-ui','--all-targets','--','-D','warnings'],
        ['clippy','--locked','--offline','-p','imagine-ui','--all-targets','--features','ui-preview','--','-D','warnings'],
        ['build','--locked','--offline','-p','imagine-server','-p','imagine-ui','--bins'],
        ['build','--locked','--offline','-p','imagine-ui','--features','ui-preview','--bin','imagine-ui','-p','imagine-server','--example','timeline_fixture_server'],
    ]
    for i,args in enumerate(tasks):
        started=time.monotonic();log=EVIDENCE/f'{run}-{i}.txt'
        with log.open('x') as output:
            result=subprocess.run([sys.executable,str(ROOT/'tools/ui-groups-cargo.py'),*args],stdout=output,stderr=subprocess.STDOUT)
        text=log.read_text()
        reports.append({'args':args,'exit':result.returncode,'seconds':time.monotonic()-started,'diagnostics':str(log),'test_results':[line for line in text.splitlines() if line.startswith('test result:')], 'failed_test_cases':[line for line in text.splitlines() if line.startswith('test ') and line.endswith(' ... FAILED')]})
        (EVIDENCE/f'{run}.json').write_text(json.dumps({'package':str(package),'checks':reports},indent=2))
        print(json.dumps(reports[-1]),flush=True)
        if result.returncode:
            print('Check failed; structured result recorded. Runtime console tails are not displayed.');return result.returncode
        binary=CANDIDATE/'target/debug'
        if args[0]=='build' and '--features' not in args:
            for name in ['imagine-server','imagine-ui']:shutil.copy2(binary/name,package/name)
        elif args[0]=='build':
            shutil.copy2(binary/'imagine-ui',package/'imagine-ui-qa')
            shutil.copy2(binary/'examples/timeline_fixture_server',package/'timeline_fixture_server')
    records={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in package.iterdir() if p.is_file()}
    (package/'candidate.json').write_text(json.dumps({'activated':False,'source':str(CANDIDATE/'source'),'binaries':records,'checks':str(EVIDENCE/f'{run}.json')},indent=2))
    print(str(package),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
