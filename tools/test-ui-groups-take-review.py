#!/usr/bin/env python3
"""Own synthetic native A/B review: source-equivalent pairing and live review API."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OWN = Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case')
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--width', type=int, default=1100)
    parser.add_argument('--failure', action='store_true')
    args = parser.parse_args()
    assert args.case.isalnum()
    package = args.package.resolve()
    assert package.is_relative_to(ROOT/'comfyui/ui-groups-20261021/packages')
    mount = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(OWN)]))['filesystems'][0]
    assert (mount['target'], mount['source'], mount['fstype']) == ('/home', '/dev/mapper/cryptroot[/@home]', 'btrfs')
    assert not os.statvfs(OWN).f_flag & os.ST_RDONLY
    case = OWN/args.case
    case.mkdir()
    for name in ['evidence', 'tmp', 'cache', 'config', 'share', 'models', 'ui']:
        (case/name).mkdir()
    env = dict(os.environ, TMPDIR=str(case/'tmp'), CUDA_VISIBLE_DEVICES='',
               LIBGL_ALWAYS_SOFTWARE='1', SDL_AUDIODRIVER='dummy',
               PYTHONDONTWRITEBYTECODE='1', XDG_CACHE_HOME=str(case/'cache'),
               XDG_CONFIG_HOME=str(case/'config'), XDG_DATA_HOME=str(case/'share'),
               QWEN_MODEL_ROOT=str(case/'models'))
    server = ui = None
    result = {'case': str(case), 'width': args.width, 'height': 760, 'gpu': False,
              'production_changed': False, 'runtime_logs_read': False, 'audio_device': 'SDL dummy', 'injected_missing_media': args.failure}
    try:
        command = [str(package/'timeline_fixture_server'), str(case), '--review-takes']
        if args.failure:
            command.append('--fail-first-comparison')
        with (case/'evidence/server.log').open('x') as output:
            server = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT)
        deadline = time.monotonic()+30
        while not (case/'evidence/endpoint.json').exists():
            assert server.poll() is None and time.monotonic() < deadline
            time.sleep(.05)
        endpoint = json.loads((case/'evidence/endpoint.json').read_text())
        assert endpoint['api'].startswith('http://127.0.0.1:') and endpoint['backend'].startswith('synthetic CPU')
        prefs = case/'ui/.imagine'
        prefs.mkdir()
        (prefs/'ui-settings.json').write_text(json.dumps({'privacy_mode': False, 'output_folder': str(case/'output')}))
        click = lambda name: {'action': 'click_control', 'name': name}
        idle = {'action': 'wait_idle'}
        ready = {'action': 'wait_comparison', 'ready': True}
        first, middle, latest = 'take_qa_first', 'take_qa_review_1', 'take_qa_review_2'
        expect = lambda ids, audio='muted', playing=False, **extra: {'action': 'expect_comparison', 'ids': ids, 'playing': playing, 'audio': audio, 'audio_child': playing and audio != 'muted', **extra}
        status = lambda take, value, accepted: {'action': 'wait_take_status', 'take': take, 'status': value, 'accepted': accepted}
        shot = lambda name: {'action': 'screenshot', 'file': name+'.png'}
        actions = [idle, {'action': 'open', 'folder': endpoint['project']}, idle,
                   click('timeline-toggle-shelf'), click('timeline-toggle-inspector'),
                   click('take-compare'), idle, ready, expect([first, latest]), {'action':'key','key':'Delete'}, {'action':'expect','scenes':3}, shot('default-accepted-latest')]
        if args.failure:
            actions += [click('comparison-audio-a'), click('comparison-play'), {'action': 'wait_time', 'milliseconds': 350}, expect([first, latest], 'A', True, position_min=.2)]
        actions += [click('comparison-1-take'), click('comparison-1-take-'+first), idle]
        if args.failure:
            actions += [{'action': 'wait_comparison', 'ready': False},
                        {'action': 'expect_error_contains', 'text': 'preview unavailable'}, shot('pair-load-error'),
                        click('take-compare'), idle, ready]
        actions += [ready, expect([latest, first], 'A' if args.failure else 'muted', position_min=.2 if args.failure else 0), click('comparison-audio-a'), click('comparison-play'),
                    {'action': 'wait_time', 'milliseconds': 350}, expect([latest, first], 'A', True, position_min=.2),
                    click('comparison-swap'), expect([first, latest], 'A', True, position_min=.2), shot('swap-single-audio'),
                    click('comparison-loop'), {'action': 'wait_time', 'milliseconds': 3200},
                    expect([first, latest], 'A', True, loops_min=1), click('comparison-play'), click('comparison-audio-b'),
                    click('comparison-0-take'), click('comparison-0-take-'+middle), idle, ready,
                    expect([middle, latest], 'B'), click('comparison-0-accept'), idle,
                    status(middle, 'accepted', True), status(first, 'candidate', False),
                    click('comparison-1-reject'), idle, status(latest, 'rejected', False),
                    expect([middle, latest], 'B'), shot('per-side-decisions'),
                    click('comparison-close'), click('take-0'), idle, {'action':'expect','preview':True},
                    {'action':'scroll_control','name':'take-strip-scroll','delta_y':-10000}, shot('take-strip-selected'), click('take-set-a'),
                    {'action':'scroll_control','name':'take-strip-scroll','delta_y':10000}, click('take-2'), idle,
                    {'action':'scroll_control','name':'take-strip-scroll','delta_y':-10000}, click('take-set-b'),
                    {'action':'scroll_control','name':'take-strip-scroll','delta_y':10000}, click('take-compare'), idle, ready, expect([first, latest]),
                    click('comparison-0-take'), click('comparison-0-take-'+middle), idle, ready, expect([middle, latest]), click('comparison-audio-b'),
                    {'action': 'hover_control', 'name': 'comparison-swap'}, {'action': 'wait_time', 'milliseconds': 700}, shot('swap-tooltip'),
                    {'action': 'privacy', 'enabled': True}, {'action': 'focus', 'focused': False},
                    {'action': 'wait_frames', 'frames': 8}, expect([middle, latest], 'B'), shot('hidden-review'),
                    {'action': 'focus', 'focused': True}, {'action': 'privacy', 'enabled': False},
                    click('comparison-close'), {'action': 'save'}, idle,
                    {'action': 'open', 'folder': endpoint['project']}, idle,
                    status(middle, 'accepted', True), status(latest, 'rejected', False),
                    click('take-compare'), idle, ready, expect([middle, latest]), shot('review-reopened'),
                    click('comparison-0-reject'), idle, status(middle, 'rejected', False),
                    click('comparison-1-accept'), idle, status(latest, 'accepted', True),
                    expect([middle, latest]), click('comparison-close'), {'action': 'save'}, idle, {'action': 'quit'}]
        # Native Area fade-in is time-based; inspect settled frames rather than
        # captures taken only a few automation frames after reopening a modal.
        actions = [step for action in actions for step in
                   ([{'action':'wait_time','milliseconds':250}, action] if action['action']=='screenshot' else
                    [action, {'action':'wait_time','milliseconds':250}] if action['action']=='scroll_control' or action.get('name')=='comparison-close' else [action])]
        plan = case/'qa-plan.json'
        plan.write_text(json.dumps({'destination': endpoint['destination'], 'actions': actions}, indent=2))
        with (case/'evidence/ui.log').open('x') as output:
            ui = subprocess.Popen([str(package/'imagine-ui-qa'), '--project', str(case/'ui'), '--api', endpoint['api'],
                                   '--title', 'Imagine own G12 Take review QA', '--qa-window-size', str(args.width), '760',
                                   '--qa-script', str(plan)], env=env, stdout=output, stderr=subprocess.STDOUT)
        result.update(api=endpoint['api'], server_pid=server.pid, ui_pid=ui.pid, actions=len(actions))
        print(json.dumps(result), flush=True)
        result['ui_exit'] = ui.wait(timeout=180)
        result['qa'] = json.loads((case/'native-qa-result.json').read_text())
        assert result['qa']['passed'], result['qa'].get('error')
        assert any(c['ids']==[middle,latest] and c['loop_enabled'] and c['audio']=='B' for c in result['qa']['comparisons'])
        result['passed'] = True
    except Exception as error:
        result['error'] = repr(error)
    finally:
        for name, process in [('ui', ui), ('server', server)]:
            if process:
                if process.poll() is None:
                    process.terminate()
                result[name+'_exit'] = process.wait(timeout=30)
        result['own_processes_reaped'] = all(p is None or p.poll() is not None for p in [ui, server])
        (case/'evidence/result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({key: result.get(key) for key in ['passed', 'error', 'own_processes_reaped']}), flush=True)
    return not result.get('passed', False)


if __name__ == '__main__':
    raise SystemExit(main())
