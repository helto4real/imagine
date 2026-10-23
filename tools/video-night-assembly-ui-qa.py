#!/usr/bin/env python3
"""Opt-in native UI acceptance, confined to new synthetic encrypted night cases."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

N = Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e')
ROOT = Path(__file__).resolve().parents[1]


def click(name):
    return {'action': 'click_control', 'name': name}


def scroll(dy):
    return [{'action': 'scroll_pixels', 'x': 450, 'y': 350, 'delta_y': dy}, {'action': 'wait_frames', 'frames': 5}]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('case')
    p.add_argument('mode', choices=['ready', 'cancel', 'publication', 'admission', 'transition'])
    p.add_argument('--ui-groups', action='store_true')
    p.add_argument('--direct', action='store_true')
    p.add_argument('--window-size',type=float,nargs=2)
    p.add_argument('--unfinished',action='store_true')
    p.add_argument('--fail-generation',action='store_true')
    p.add_argument('--cancel-generation',action='store_true')
    p.add_argument('--filename')
    p.add_argument('--freeze-retry',action='store_true')
    p.add_argument('--package', type=Path, required=True)
    args = p.parse_args()
    assert not(args.fail_generation and args.cancel_generation)
    assert not args.freeze_retry or (args.ui_groups and args.direct and args.mode=='ready')
    assert not args.unfinished or (args.ui_groups and args.direct and args.mode=='ready')
    if args.unfinished and args.filename is None:args.filename='Own Final β'
    assert args.case.isalnum() and args.case.startswith('q')
    package = args.package.resolve()
    assert package.parent == ROOT/('comfyui/ui-groups-20261021/packages' if args.ui_groups else 'comfyui/h3-video-night-20261021/packages')
    mount = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(N)]))['filesystems'][0]
    assert (mount['target'], mount['source'], mount['fstype']) == ('/home', '/dev/mapper/cryptroot[/@home]', 'btrfs') and not os.statvfs(N).f_flag & os.ST_RDONLY
    case = (N/'ui-groups-01' if args.ui_groups else N)/args.case
    case.mkdir()
    (case/'evidence').mkdir()
    env = os.environ.copy()
    env.update(TMPDIR=str(N/'tmp'), XDG_CACHE_HOME=str(case/'cache'), XDG_CONFIG_HOME=str(case/'config'), XDG_DATA_HOME=str(case/'share'), QWEN_MODEL_ROOT=str(case/'models'), CUDA_VISIBLE_DEVICES='', LIBGL_ALWAYS_SOFTWARE='1', PYTHONDONTWRITEBYTECODE='1')
    for name in ['cache', 'config', 'share', 'models']:
        (case/name).mkdir()
    result = {'case': str(case), 'mode': args.mode, 'package': str(package), 'backend': 'synthetic CPU only', 'canonical_gpu_coordinator_used': False, 'software_rendering_requested': True}
    server = ui = None
    try:
        flags = ['--assembly', '--with-audio']
        if args.mode == 'cancel': flags += ['--assembly-long']
        if args.mode == 'publication': flags += ['--publication-failure']
        if args.mode == 'admission': flags += ['--drop-first-admission']
        if args.fail_generation:flags+=['--fail-first']
        if args.cancel_generation:flags+=['--slow']
        if args.freeze_retry:flags+=['--drop-first-freeze']
        with (case/'evidence/server.log').open('w') as f:
            server = subprocess.Popen([str(package/'timeline_fixture_server'), str(case), *flags], env=env, stdout=f, stderr=subprocess.STDOUT)
        result['server_pid'] = server.pid
        deadline = time.monotonic()+30
        while not (case/'evidence/endpoint.json').exists():
            assert server.poll() is None and time.monotonic() < deadline
            time.sleep(.05)
        endpoint = json.loads((case/'evidence/endpoint.json').read_text())
        assert endpoint['api'].startswith('http://127.0.0.1:') and endpoint['backend'].startswith('synthetic CPU')
        def rpc(path,body=None):
            request=urllib.request.Request(endpoint['api']+path,data=json.dumps(body).encode()if body is not None else None,headers={'X-Imagine-Session':session,'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=20)as response:return json.load(response)
        initial=None
        if args.unfinished:
            with urllib.request.urlopen(endpoint['api']+'/v1/session')as response:session=json.load(response)['id']
            location=rpc('/v1/video/locations',{'schema_version':1,'path':endpoint['project']})['location_id']
            initial=rpc('/v1/video/projects/open',{'schema_version':1,'location_id':location})
            scene=initial['project']['scenes'][-1]
            initial=rpc('/v1/video/projects/'+initial['project']['id']+'/review',{'schema_version':1,'expected_project_revision':initial['project']['project_revision'],'scene_id':scene['id'],'take_id':scene['accepted_take']['id'],'action':'candidate'})
        result['api'] = endpoint['api']
        prefs = case/'ui/.imagine'
        prefs.mkdir(exist_ok=True)
        (prefs/'ui-settings.json').write_text(json.dumps({'output_folder': str(case/'output'), 'privacy_mode': False}))
        actions = [{'action': 'wait_idle'}, {'action': 'open', 'folder': endpoint['project']}, {'action': 'wait_idle'}, {'action': 'expect', 'takes': 3, 'accepted_takes': 2 if args.unfinished else 3, 'deliveries': 0}, click('queue-toggle')]
        if args.window_size:actions[1:1]=[{'action':'resize','width':args.window_size[0],'height':args.window_size[1]},{'action':'wait_frames','frames':12}]
        if args.mode == 'transition':
            actions += [click('boundary-1'), {'action': 'wait_idle'},
                        {'action': 'screenshot', 'file': 'boundary-before-transition.png'},
                        {'action': 'scroll_to_control', 'scroll': 'inspector-scroll', 'name': 'assembly-boundary-transition'},
                        click('assembly-boundary-transition'), {'action': 'wait_idle'}]
        actions += [click('assembly-open')]
        if args.filename is not None:actions += [click('assembly-output-filename'),{'action':'key','key':'A','ctrl':True},{'action':'text','text':args.filename}]
        if args.direct:actions += [{'action':'wait_time','milliseconds':350},{'action':'screenshot','file':'final-direct-options.png'}]
        if not args.direct: actions += [*scroll(-1200), click('assembly-freeze'), {'action':'wait_idle'}]
        if args.mode == 'transition':
            actions += [{'action': 'expect', 'assembly_error': False},
                        *([] if args.direct else scroll(-1200)),
                        {'action': 'screenshot', 'file': 'transition-options.png'}]
        actions += [click('assembly-render-current')] if args.direct else [*scroll(-1200), click('assembly-render')]
        if args.unfinished:
            actions += [click('assembly-close')]
            if not args.fail_generation and not args.cancel_generation:
                actions += [click('assembly-open'),click('assembly-output-filename'),{'action':'key','key':'A','ctrl':True},{'action':'text','text':'Own later filename choice'},{'action':'screenshot','file':'final-pending-next-draft.png'},click('assembly-close')]
            if args.fail_generation or args.cancel_generation:
                if args.cancel_generation:actions += [{'action':'wait_status','status':'running','count':1},click('run-cancel-current')]
                actions += [{'action':'wait_run_status','status':'cancelled' if args.cancel_generation else 'failed','count':1},{'action':'wait_time','milliseconds':200},{'action':'wait_idle'},{'action':'expect','deliveries':0},{'action':'screenshot','file':'final-generation-stopped.png'},click('assembly-open'),click('assembly-render-current'),click('assembly-close')]
        if args.mode == 'cancel':
            actions += [{'action': 'wait_assembly_phase', 'phase': 'encoding_video'}, *scroll(1200), click('assembly-close'), click('job-cancel-0'), {'action': 'wait_status', 'status': 'cancelled', 'count': 1}, {'action': 'wait_idle'}, {'action': 'expect', 'deliveries': 0, 'takes': 3}, {'action': 'screenshot', 'file': 'montage-cancelled.png'}, click('job-retry-0')]
        elif args.mode == 'publication':
            actions += [{'action': 'wait_status', 'status': 'failed', 'count': 1}, *scroll(1200), click('assembly-close'), {'action': 'screenshot', 'file': 'publication-failed.png'}, {'action': 'wait_time', 'milliseconds': 2500}, click('job-retry-0')]
        elif args.mode == 'admission':
            actions += [{'action': 'wait_idle'}, *scroll(1200), click('assembly-close'), {'action': 'screenshot', 'file': 'admission-unconfirmed.png'}, click('submission-reconnect')]
        if args.freeze_retry:
            actions += [{'action':'wait_final_failure','code':'synthetic_freeze_lost'},{'action':'expect','deliveries':0},click('assembly-open'),click('assembly-output-filename'),{'action':'key','key':'A','ctrl':True},{'action':'text','text':'Own later choice after preparation failed'},{'action':'wait_time','milliseconds':350},{'action':'screenshot','file':'final-preparation-retry.png'},click('assembly-retry-preparation'),click('assembly-close')]
        actions += [{'action': 'wait_status', 'status': 'succeeded', 'count': 4 if args.unfinished else 1}, {'action': 'wait_idle'}]
        if args.mode == 'transition': actions += [{'action':'wait_assembly_phase','phase':'saved'}, *scroll(-1200), {'action':'screenshot','file':'transition-final-warning.png'}, *scroll(1200), click('assembly-close')]
        actions += [{'action': 'expect', 'deliveries': 1, 'takes': 6 if args.unfinished else 3, 'accepted_takes': 3}, {'action': 'screenshot', 'file': 'montage-recovered.png'}]
        if args.unfinished:actions += [{'action':'open','folder':endpoint['project']},{'action':'wait_idle'},{'action':'expect','deliveries':1,'takes':6,'accepted_takes':3}]
        actions += [{'action':'quit'}]
        plan = case/'qa-plan.json'
        plan.write_text(json.dumps({'destination': endpoint['destination'], 'actions': actions}, indent=2))
        command=[str(package/('imagine-ui-qa' if args.ui_groups else 'imagine-ui')), '--project', str(case/'ui'), '--api', endpoint['api'], '--title', f'Imagine Video Night QA {args.case} - {args.mode}', '--qa-script', str(plan)]
        if args.window_size:command+=['--qa-window-size',*[str(n) for n in args.window_size]]
        with (case/'evidence/ui.log').open('w') as f:
            ui = subprocess.Popen(command, env=env, stdout=f, stderr=subprocess.STDOUT)
        result['ui_pid'] = ui.pid
        (case/'evidence/running.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        restored = False
        deadline = time.monotonic()+145
        while ui.poll() is None:
            assert time.monotonic() < deadline, 'own UI launcher timeout'
            if args.mode == 'publication' and not restored and (case/'publication-failed.png').exists():
                with urllib.request.urlopen(endpoint['api']+'/v1/session') as response:
                    session = json.load(response)['id']
                document = json.loads((Path(endpoint['project'])/'project.json').read_text())
                project_id = document['project']['id']
                req = urllib.request.Request(endpoint['api']+'/v1/video/projects/'+project_id, headers={'X-Imagine-Session': session})
                with urllib.request.urlopen(req) as response: result['project_before_retry'] = json.load(response)
                req = urllib.request.Request(endpoint['api']+'/__fixture/allow-publication', data=b'', method='POST')
                with urllib.request.urlopen(req) as response: result['restored'] = json.load(response)
                restored = True
            time.sleep(.05)
        result['ui_exit'] = ui.wait()
        result['qa'] = json.loads((case/'native-qa-result.json').read_text())
        if args.freeze_retry and result['qa']['passed']:
            failure=result['qa']['final_failures'][0]
            assert failure['failure']['code']=='synthetic_freeze_lost' and failure['failure']['operation']=='freeze' and failure['failure']['retryable'] and failure['retry_preparation']
            result['explicit_preparation_retry']={'original_choices_preserved':True,'observed_failure':failure['failure'],'no_automatic_retry':True}
        if result['qa']['passed']:
            saved = json.loads((Path(endpoint['project'])/'project.json').read_text())
            assert len(saved['deliveries']) == 1
            exports = [p for p in (case/'output').glob('*/*')if p.suffix.lower()=='.mp4']
            assert len(exports) == (4 if args.unfinished else 1)
            if args.filename is not None:
                filename=args.filename if args.filename.lower().endswith('.mp4')else args.filename+'.mp4'
                exports=[p for p in exports if p.name==filename];assert len(exports)==1
                assert saved['deliveries'][0]['plan']['options']['output_filename']==filename
            if args.unfinished:
                runs=rpc('/v1/video/runs')['runs'];complete=[r for r in runs if r['status']=='completed'];assert len(complete)==1 and complete[0]['mode']=='all_fresh_seeds' and len(complete[0]['targets'])==3
                chosen={s['accepted_take']['id'] for s in saved['project']['scenes']}
                seeds={t['seed'] for t in complete[0]['targets']};assert len(seeds)==3 and not seeds&{t['provenance']['seed']for t in initial['takes']}
                assert {t['completed_take_id']for t in complete[0]['targets']}==chosen
                assert {c['take_id']for c in saved['deliveries'][0]['plan']['clips']}==chosen
                assert {t['id']for t in initial['takes']}.issubset({t['id']for t in saved['takes']})
                result['generation_to_final']={'mode':'all_fresh_seeds','regenerated_scenes':3,'seeds':sorted(seeds),'previous_takes_retained':True,'final_uses_run_results':True,'reopened':True,'failed_then_retried':args.fail_generation,'cancelled_then_retried':args.cancel_generation}
            result['export_path'] = str(exports[0])
            result['export_sha256'] = hashlib.sha256(exports[0].read_bytes()).hexdigest()
            if args.mode == 'publication':
                retained = result['restored']['retained_encoded_results']
                assert len(retained) == 1 and retained[0]['sha256'] == result['export_sha256']
            result['probe'] = json.loads(subprocess.check_output(['/usr/bin/ffprobe', '-v', 'error', '-count_frames', '-show_entries', 'format=duration:stream=codec_name,nb_read_frames,sample_rate,channels,pix_fmt,color_space,duration', '-of', 'json', str(exports[0])]))
            frozen=saved['deliveries'][0]['plan'];fps=frozen['delivery']['fps']['num']/frozen['delivery']['fps']['den'];seconds=frozen['frames']/fps
            if args.mode == 'transition':
                assert frozen['warnings'] == ['Transition was not generated; joining the existing Takes.']
                assert len(frozen['clips']) == 3 and frozen['clips'][1]['boundary'] == 'transition'
                assert frozen['clips'][1]['overlap'] == 0 and len(saved['takes']) == 3
                result['transition_semantics'] = {'generated_conditioning': False, 'explicit_warning': frozen['warnings'][0], 'no_extra_bridge_take': True, 'concat_without_extra_frames': True}
            video,audio=result['probe']['streams'];assert video['codec_name']=='h264' and video['pix_fmt']=='yuv420p' and video['color_space']=='bt709' and int(video['nb_read_frames'])==frozen['frames']
            assert audio['codec_name']=='aac' and audio['channels']==2 and int(audio['sample_rate'])==44100
            assert abs(float(result['probe']['format']['duration'])-seconds)<.001 and abs(float(audio['duration'])-seconds)<.001
            import array,math
            pcm=array.array('f',subprocess.check_output(['ffmpeg','-v','error','-i',str(exports[0]),'-map','0:a:0','-f','f32le','-acodec','pcm_f32le','-']))
            rms=math.sqrt(sum(x*x for x in pcm)/len(pcm));peak=max(abs(x)for x in pcm);assert math.isfinite(rms) and rms>0 and peak>0
            result['audio_pcm']={'rms':rms,'peak':peak,'samples_per_channel':len(pcm)//2,'decoded_in_memory':True}
            result['frozen_duration_verified']={'frames':frozen['frames'],'fps':fps,'seconds':seconds}
            result['persisted_delivery_plan'] = True
    except Exception as error:
        result['launcher_error'] = repr(error)
    finally:
        if ui and ui.poll() is None: ui.terminate(); ui.wait(timeout=15)
        if server:
            if server.poll() is None: server.terminate()
            result['server_exit'] = server.wait(timeout=30)
        result['own_processes_reaped'] = all(p is None or p.poll() is not None for p in [ui, server])
        (case/'evidence/launch-result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({k:v for k,v in result.items() if k != 'project_before_retry'}, indent=2), flush=True)
    raise SystemExit(0 if result.get('qa', {}).get('passed') and result.get('persisted_delivery_plan') and result['own_processes_reaped'] else 1)


if __name__ == '__main__':
    main()
