#!/usr/bin/env python3
"""Real H3 DiT on numeric fixtures only. No submitted text/jobs/media or dumps.

Holds the ordinary exclusive coordinator socket; no standalone fallback.
Console output is preserved and never read. Only numeric component diagnostics
are returned. This is not the denied Exp/masked-step diagnostic.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import select
import socket
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
C = ROOT/'comfyui/ui-groups-20261021'
OWN = Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')
CONFIG = Path('/home/thhel/tmp/.codex-h3-res20-compare-20261020-6f0b84c2/full-fp16-20261020-5e4a923b/candidate-runtime.json')
TEST = 'h3_product::native::sampler_gpu_test::named_sampling_with_manual_sigmas_on_real_dit'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--prepared', action='store_true')
    args = parser.parse_args()
    assert args.case.isascii() and args.case.isalnum()
    mount = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(OWN)]))['filesystems'][0]
    assert (mount['target'],mount['source'],mount['fstype']) == ('/home','/dev/mapper/cryptroot[/@home]','btrfs')
    assert not os.statvfs(OWN).f_flag & os.ST_RDONLY
    case = OWN/args.case
    spec = importlib.util.spec_from_file_location('own_build',ROOT/'tools/build-ui-groups-worker.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = module.environment()
    if not args.prepared:
        case.mkdir()
        for directory in ['evidence','tmp','cache','config','share']:
            (case/directory).mkdir()
        command = ['cargo','test','--manifest-path',str(C/'source/inference/Cargo.toml'),
                   '--locked','--offline','--lib',TEST,'--no-run','--message-format=json']
        build = subprocess.run(command,cwd=C/'source',env=env,text=True,capture_output=True)
        (case/'evidence/compiler.txt').write_text(build.stderr)
        binary = None
        for line in build.stdout.splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get('reason') == 'compiler-artifact' and item.get('executable') and item['profile']['test']:
                binary = item['executable']
            if item.get('reason') == 'compiler-message':
                with (case/'evidence/compiler.txt').open('a') as output:
                    output.write(item['message'].get('rendered',''))
        result = {'exit':build.returncode,'binary':binary,'test':TEST}
        (case/'evidence/build.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({'build':result,'case':str(case)}),flush=True)
        if build.returncode or not binary:
            return 1
    else:
        result = json.loads((case/'evidence/build.json').read_text())
        assert result['exit'] == 0 and result['test'] == TEST
        binary = result['binary']
    if args.prepare_only:
        return 0
    assert Path(binary).resolve().is_relative_to(C/'target-worker')
    runtime = json.loads(CONFIG.read_text())['h3']  # Own static build configuration only.
    env.update(runtime['environment'])
    env.update(CUDA_VISIBLE_DEVICES='0',TMPDIR=str(case/'tmp'),PYTHONDONTWRITEBYTECODE='1',
               XDG_CACHE_HOME=str(case/'cache'),XDG_CONFIG_HOME=str(case/'config'),
               XDG_DATA_HOME=str(case/'share'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
               IMAGINE_CP40_SAMPLER_RESULT=str(case/'evidence/component.json'),
               IMAGINE_CP40_MODEL_ROOT=runtime['model_root'],IMAGINE_CP40_EXCLUSIVE_LEASE='held')
    for key,name in [('TORCH_HOME','torch'),('HF_HOME','hf'),('TRITON_CACHE_DIR','triton'),
                     ('CUDA_CACHE_PATH','cuda'),('IMAGINE_WORKER_CACHE','worker')]:
        directory = case/'cache'/name
        directory.mkdir(exist_ok=True)
        env[key] = str(directory)
    # Briefly allow transient aggregate load to settle; never stop another job.
    # The ordinary coordinator still owns the exclusive inference admission.
    for _ in range(20):
        preflight = subprocess.check_output(['/usr/bin/nvidia-smi',
            '--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],env=env,text=True).strip()
        if int(preflight.split(',')[0]) == 0:
            break
        time.sleep(1)
    else:
        print(json.dumps({'blocked':'GPU active; no model process/lease started','aggregate':preflight}))
        return 2
    ticket = None
    process = None
    report = {'case':str(case),'passed':False,'aggregate_before':preflight,
              'coordinator':'ordinary exclusive socket; no alternate execution',
              'logs_read':False,'submitted_text_jobs_media':False,'latent_dumps':False}

    def receive():
        line = b''
        while not line.endswith(b'\n'):
            if not select.select([ticket],[],[],30)[0]:
                print(json.dumps({'waiting_for_own_exclusive_lease':True}),flush=True)
                continue
            part = ticket.recv(1)
            if not part:
                raise RuntimeError('Own coordinator socket closed')
            line += part
        return json.loads(line)
    try:
        ticket = socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        ticket.connect(str(ROOT/'.imagine/gpu.sock'))
        ticket.sendall((json.dumps({'job':'imagine-cp40-numeric-sampler-'+uuid.uuid4().hex,'bytes':64*1024*1024})+'\n').encode())
        assert receive()['queued']
        assert receive()['granted']
        report['exclusive_lease_granted'] = True
        with (case/'evidence/component-console.log').open('x') as console:
            process = subprocess.Popen([binary,TEST,'--ignored','--exact','--test-threads=1'],
                cwd=C/'source',env=env,stdout=console,stderr=subprocess.STDOUT)
        report['own_pid'] = process.pid
        print(json.dumps({'phase':'own_numeric_dit_component','pid':process.pid,'case':str(case)}),flush=True)
        start = time.monotonic()
        while process.poll() is None:
            if select.select([ticket],[],[],.5)[0]:
                raise RuntimeError('Own coordinator lease lost')
            assert time.monotonic()-start < 600,'Bounded own component timed out'
        report['exit'] = process.returncode
        if (case/'evidence/component.json').is_file():
            # Numeric test-fixture diagnostics only: no job/session/media data.
            report['component'] = json.loads((case/'evidence/component.json').read_text())
        report['passed'] = process.returncode == 0 and report.get('component',{}).get('result',{}).get('passed',False)
    except Exception as error:
        report['error'] = repr(error)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=60)
        report['own_process_reaped'] = process is None or process.poll() is not None
        if ticket is not None:
            if report.get('exclusive_lease_granted'):
                try:
                    ticket.sendall(b'{"finished":true}\n')
                except OSError:
                    pass
            ticket.close()
        report['own_ticket_closed'] = True
        (case/'evidence/result.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
    return 0 if report['passed'] and report['own_process_reaped'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
