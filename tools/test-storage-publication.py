#!/usr/bin/env python3
"""Synthetic package/launcher contracts; never starts real server/proxy/services."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
E = Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')

def main():
    # These are synthetic executables/deployment metadata, never private media.
    # Use the permitted code/build disk, just as for real immutable packages.
    case = ROOT / 'comfyui/ui-groups-20261021/packages' / ('storage-publication-test-' + uuid.uuid4().hex[:8])
    case.mkdir()
    spec = importlib.util.spec_from_file_location('publication', ROOT/'tools/publish-video-product.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = case
    module.ACTIVE = case/'comfyui/active-product.sh'
    module.ACTIVE.parent.mkdir()
    source = case/'source'
    (source/'shared').mkdir(parents=True)
    (source/'shared/h3_protocol.rs').write_text('pub const VERSION: u32 = 16;')
    (source/'proxy').mkdir()
    stack = ROOT/'comfyui/ui-groups-20261021/source/proxy/st-stack.zsh'
    shutil.copy2(stack, source/'proxy/st-stack.zsh')
    (source/'workers/comfy').mkdir(parents=True)
    for name in module.COMFY_SCRIPTS:(source/'workers/comfy'/name).write_text('# own synthetic package fixture\n')
    binaries = case/'binaries'
    binaries.mkdir()
    for name in ['server','ui','h3','image', *module.PROXY_BINARIES]:
        p=binaries/name
        p.write_text('#!/bin/sh\nprintf "%s\\n" "own-stub:'+name+'" "${ST_PROXY_COORDINATOR_SOCKET:-}" "${IMAGINE_COORDINATOR_SOCKET:-}" "$@"\n')
        p.chmod(0o755)
    deps=case/'deps.json';deps.write_text('{}')
    runtime=case/'runtime.json'
    runtime.write_text(json.dumps({'binary':str(binaries/'image'),'files':{str(binaries/'image'):'own'},
      'h3':{'binary':str(binaries/'h3'),'files':{str(binaries/'h3'):'own'},'dependency_map':str(deps),
            'identity':{'worker_sha256':'own','protocol':16},'environment':{},'profile_checkpoints':{}}}))
    module.ACTIVE.write_text(module.selector({'IMAGINE_PRODUCT_SERVER':binaries/'server','IMAGINE_PRODUCT_UI':binaries/'ui','IMAGINE_PRODUCT_RUNTIME':runtime}))
    def call(*args):
        before=sys.argv
        sys.argv=['publish',*map(str,args)]
        try:
            with contextlib.redirect_stdout(io.StringIO()) as output:module.main()
            return Path(output.getvalue().strip())
        finally:sys.argv=before
    def stage(private):
        args=['--server',binaries/'server','--ui',binaries/'ui','--worker',binaries/'h3',
              '--source-root',source,'--proxy-dir',binaries]
        if private:args.append('--private-storage')
        return call(*args)
    first=stage(True)
    selected=module.selected((first/'active-product.sh').read_bytes())
    assert selected['IMAGINE_PRIVATE_STORAGE']=='1'
    assert Path(selected['IMAGINE_PRODUCT_PROXY_LAUNCH']).parent==first
    assert 'IMAGINE_PRIVATE_STORAGE' not in module.selected((first/'rollback-active-product.sh').read_bytes())
    call('--activate',first)
    second=stage(True)
    rollback=module.selected((second/'rollback-active-product.sh').read_bytes())
    assert rollback['IMAGINE_PRIVATE_STORAGE']=='1'
    assert Path(rollback['IMAGINE_PRODUCT_PROXY_LAUNCH']).parent==second/'rollback'
    for name in module.PROXY_BINARIES:
        assert (first/name).read_bytes()==(second/'rollback'/name).read_bytes()
    assert (first/'st-stack.zsh').read_bytes()==(second/'rollback/st-stack.zsh').read_bytes()
    for name in module.COMFY_SCRIPTS:
        assert (first/'comfy-worker'/name).read_bytes()==(second/'rollback/comfy-worker'/name).read_bytes()
    call('--activate',second)
    call('--rollback',second)
    assert module.selected(module.ACTIVE.read_bytes())==rollback
    for launcher in ['start-server.sh','start-proxy.sh']:
        shutil.copy2(ROOT/launcher,case/launcher)
    env=os.environ.copy();env['XDG_RUNTIME_DIR']=str(case/'runtime');env['HOME']=str(case/'home')
    env.pop('ST_PROXY_COORDINATOR_SOCKET',None);env.pop('IMAGINE_COORDINATOR_SOCKET',None)
    for launcher,stub in [('start-proxy.sh','own-stub:st-proxy-launch'),('start-server.sh','own-stub:server')]:
        result=subprocess.run(['bash',str(case/launcher),'--help'],env=env,capture_output=True,text=True)
        assert result.returncode==0,(launcher,result.stderr)
        assert stub in result.stdout
        assert str(case/'runtime/imagine/gpu.sock') in result.stdout
        assert not (case/'.imagine').exists()
        missing=env.copy();missing.pop('XDG_RUNTIME_DIR')
        result=subprocess.run(['bash',str(case/launcher),'--help'],env=missing,capture_output=True,text=True)
        assert result.returncode==2 and 'own-stub:' not in result.stdout
    subprocess.run(['zsh','-n',str(stack)],check=True)
    print(json.dumps({'passed':True,'case':str(case),'synthetic_only':True,'real_services_started':False,
                      'contracts':['bundle','activate','rollback-proxy-and-storage-flag','shared-runtime-socket','no-runtime-fallback','shell-syntax']}))

if __name__=='__main__':main()
