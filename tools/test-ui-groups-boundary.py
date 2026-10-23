#!/usr/bin/env python3
"""Own native UI -> frozen real media -> synthetic sampler -> Take receipt."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OWN = Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('case')
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--width', type=int, default=1100)
    p.add_argument('--method', choices=['auto','combo','influence'], default='auto')
    p.add_argument('--nonadjacent-fall', action='store_true',
                   help='Target Scene 3 across Scene 2→3, using explicit Scene 1 media (Combo by default).')
    args = p.parse_args()
    assert args.case.isalnum()
    package = args.package.resolve()
    method = 'combo' if args.nonadjacent_fall and args.method == 'auto' else args.method
    target = 3 if args.nonadjacent_fall else 2
    target_scene = f'scene_qa_{target}'
    target_section = f'section-section_qa_{target}'
    boundary_left = f'scene_qa_{target-1}'
    assert package.parent == ROOT/'comfyui/ui-groups-20261021/packages'
    mount = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(OWN)]))['filesystems'][0]
    assert (mount['target'], mount['source'], mount['fstype']) == ('/home', '/dev/mapper/cryptroot[/@home]', 'btrfs')
    assert not os.statvfs(OWN).f_flag & os.ST_RDONLY
    case = OWN/args.case
    case.mkdir()
    for name in ['evidence', 'tmp', 'cache', 'config', 'share', 'models', 'ui']:
        (case/name).mkdir()
    env = dict(os.environ, TMPDIR=str(case/'tmp'), CUDA_VISIBLE_DEVICES='', LIBGL_ALWAYS_SOFTWARE='1',
               SDL_AUDIODRIVER='dummy', PYTHONDONTWRITEBYTECODE='1', XDG_CACHE_HOME=str(case/'cache'),
               XDG_CONFIG_HOME=str(case/'config'), XDG_DATA_HOME=str(case/'share'), QWEN_MODEL_ROOT=str(case/'models'))
    result = {'case': str(case), 'package': str(package), 'width': args.width, 'gpu': False,
              'backend': 'synthetic CPU sampler; real freeze/decode/queue/export', 'method': method,
              'nonadjacent_fall': args.nonadjacent_fall, 'target_scene': target_scene,
              'boundary_left': boundary_left, 'media_source_scene': 'scene_qa_1', 'production_changed': False,
              'runtime_logs_read': False}
    server = ui = None
    click = lambda name: {'action': 'click_control', 'name': name}
    idle = {'action': 'wait_idle'}
    scroll = lambda name: {'action': 'scroll_to_control', 'scroll': 'inspector-scroll', 'name': name}
    edit = lambda name, text: [scroll(name), click(name), {'action': 'key', 'key': 'A', 'ctrl': True},
                               {'action': 'text', 'text': text}]
    try:
        with (case/'evidence/server.log').open('x') as output:
            server = subprocess.Popen([str(package/'timeline_fixture_server'), str(case), '--assembly', '--with-audio'],
                                      env=env, stdout=output, stderr=subprocess.STDOUT)
        deadline = time.monotonic()+30
        while not (case/'evidence/endpoint.json').exists():
            assert server.poll() is None and time.monotonic() < deadline
            time.sleep(.05)
        endpoint = json.loads((case/'evidence/endpoint.json').read_text())
        api = endpoint['api']
        assert api.startswith('http://127.0.0.1:') and endpoint['backend'].startswith('synthetic CPU')
        prefs = case/'ui/.imagine'
        prefs.mkdir()
        (prefs/'ui-settings.json').write_text(json.dumps({'privacy_mode': False, 'output_folder': str(case/'output')}))
        actions = [idle, {'action': 'open', 'folder': endpoint['project']}, idle,
                   click(f'boundary-{target-1}'), idle, scroll('assembly-boundary-transition'),
                   click('assembly-boundary-transition'), idle,
                   scroll('boundary-tail-custom'), click('boundary-tail-custom'), idle,
                   *edit('boundary-tail-frames', '23'), {'action': 'key', 'key': 'Enter'}, idle,
                   *edit('boundary-transition-prompt', 'The camera turns through the doorway.'), idle,
                   scroll('boundary-audio-reference'), click('boundary-audio-reference'), idle,
                   {'action': 'screenshot', 'file': 'boundary-settings.png'},
                   click(target_section), idle, click('timeline-generate'), idle,
                   {'action': 'wait_status', 'status': 'succeeded', 'count': 1}, idle,
                   {'action': 'expect', 'takes': 4, 'accepted_takes': 3 if method == 'auto' else 2},
                   {'action': 'scroll_to_control', 'scroll': 'take-strip-scroll', 'name': 'take-compare'},
                   {'action': 'hover_control', 'name': 'take-compare'}, {'action': 'wait_time', 'milliseconds': 700},
                   {'action': 'screenshot', 'file': 'boundary-generated.png'}, {'action': 'quit'}]
        if method != 'auto':
            index = actions.index(click(target_section))
            actions[index:index] = [scroll('continuity-source'), click('continuity-source'),
                click('continuity-source-0'), idle, scroll('continuity-method'), click('continuity-method'),
                click('continuity-'+method), idle,
                scroll('continuity-source-status'),
                {'action':'screenshot','file':'continuity-settings.png'}]
        plan = case/'qa-plan.json'
        plan.write_text(json.dumps({'destination': endpoint['destination'], 'actions': actions}, indent=2))
        with (case/'evidence/ui.log').open('x') as output:
            ui = subprocess.Popen([str(package/'imagine-ui-qa'), '--project', str(case/'ui'), '--api', api,
                                   '--title', 'Imagine own boundary QA', '--qa-window-size', str(args.width), '760',
                                   '--qa-script', str(plan)], env=env, stdout=output, stderr=subprocess.STDOUT)
        result.update(server_pid=server.pid, ui_pid=ui.pid, api=api)
        print(json.dumps(result), flush=True)
        result['ui_exit'] = ui.wait(timeout=145)
        qa = json.loads((case/'native-qa-result.json').read_text())
        result['qa'] = qa
        assert qa['passed'], qa['error']
        saved = json.loads((Path(endpoint['project'])/'project.json').read_text())
        assert len(saved['takes']) == 4
        boundary = next(b for b in saved['project']['assembly']['boundaries'] if b['right'] == target_scene)
        assert boundary['left'] == boundary_left
        assert boundary['mode'] == 'transition' and boundary['tail_frames'] == 23
        assert boundary['transition_prompt'] == 'The camera turns through the doorway.'
        assert boundary['audio_continuity'] == 'reference_previous_audio'
        receipt = saved['takes'][-1]['provenance']['boundary']
        assert receipt['applied'] and receipt['selection']['effective_tail_frames'] == 22
        assert receipt['dependency']['requested_tail_frames'] == 23
        assert receipt['dependency']['source_shot_id'] == 'scene_qa_1'
        assert receipt['dependency']['target_shot_id'] == target_scene
        assert receipt['dependency']['boundary_id'] == {'left': boundary_left, 'right': target_scene}
        assert saved['takes'][-1]['scene_id'] == target_scene
        captures = list((case/'evidence').glob('generated-input-*.json'))
        assert len(captures) == 1
        captured = json.loads(captures[0].read_text())
        assert captured['boundary_receipt']['applied']
        assert captured['boundary_receipt']['dependency'] == receipt['dependency']
        assert captured['boundary_receipt']['selection'] == receipt['selection']
        assert (captured['start_image'] is not None) == (method != 'influence')
        if method == 'influence':
            assert len(captured['reference_shapes']['images']) == 1
        assert captured['reference_shapes']['videos'][0]['frames'] == 22
        # The fixture's [51/24,73/24] second interval rounds its two half-sample
        # endpoints independently, as Director does: 134138 - 93712.
        assert captured['reference_shapes']['videos'][0]['audio_samples'] == 40426
        result.update(boundary=boundary, receipt=receipt, input_capture=str(captures[0]),
                      input_shapes=captured['reference_shapes'], no_extra_bridge_take=True, passed=True)
    except Exception as error:
        result['error'] = repr(error)
    finally:
        for name, process in [('ui', ui), ('server', server)]:
            if process:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT if name == 'server' else signal.SIGTERM)
                result[name+'_exit'] = process.wait(timeout=30)
        result['own_processes_reaped'] = all(p is None or p.poll() is not None for p in [ui, server])
        (case/'evidence/result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({k: result.get(k) for k in ['passed', 'error', 'own_processes_reaped']}), flush=True)
    return 0 if result.get('passed') and result['own_processes_reaped'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
