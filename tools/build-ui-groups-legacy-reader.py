#!/usr/bin/env python3
"""Build the unchanged frozen server as a library; no live services or models."""
import json,os,subprocess,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OWN=Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')
def main():
    m=json.loads(subprocess.check_output(['findmnt','-J','-T',str(OWN)]))['filesystems'][0]
    assert (m['target'],m['source'],m['fstype'])==('/home','/dev/mapper/cryptroot[/@home]','btrfs') and not os.statvfs(OWN).f_flag&os.ST_RDONLY
    crate=ROOT/'comfyui/ui-groups-20261021/legacy-reader'
    env=dict(os.environ,CARGO_TARGET_DIR=str(crate/'target'),CARGO_BUILD_JOBS='2',CARGO_INCREMENTAL='0',CARGO_PROFILE_DEV_DEBUG='0',TMPDIR=str(OWN/'tmp'),CUDA_VISIBLE_DEVICES='')
    report=OWN/'evidence'/('legacy-reader-build-'+uuid.uuid4().hex[:8]+'.txt')
    with report.open('x')as output:result=subprocess.run(['cargo','build','--offline',*(['--locked']if(crate/'Cargo.lock').exists()else[])],cwd=crate,env=env,stdout=output,stderr=subprocess.STDOUT)
    print(json.dumps({'exit':result.returncode,'diagnostics':str(report),'binary':str(crate/'target/debug/imagine-ui-legacy-reader-probe')}))
    if result.returncode:print('\n'.join(report.read_text().splitlines()[-25:]))
    return result.returncode
if __name__=='__main__':raise SystemExit(main())
