#!/usr/bin/env python3
"""Native Animate acceptance using only a new procedural image and CPU fixture API."""
import argparse
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import time
import zlib

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / 'comfyui/ui-groups-20261021'
OWN = Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('case')
    p.add_argument('--width', type=int, choices=[600, 1100], required=True)
    p.add_argument('--ui', type=Path, required=True)
    args = p.parse_args()
    assert args.case.isalnum() and args.case.startswith('animate45')
    assert args.ui.resolve().is_relative_to(C)
    mount = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(OWN)]))['filesystems'][0]
    assert (mount['target'], mount['source'], mount['fstype']) == ('/home', '/dev/mapper/cryptroot[/@home]', 'btrfs')
    case = OWN / args.case
    case.mkdir()
    for name in ['tmp', 'cache', 'config', 'share', 'models', 'evidence']:
        (case/name).mkdir()
    env = dict(os.environ, TMPDIR=str(case/'tmp'), CUDA_VISIBLE_DEVICES='',
               LIBGL_ALWAYS_SOFTWARE='1', SDL_AUDIODRIVER='dummy', PYTHONDONTWRITEBYTECODE='1',
               XDG_CACHE_HOME=str(case/'cache'), XDG_CONFIG_HOME=str(case/'config'),
               XDG_DATA_HOME=str(case/'share'), QWEN_MODEL_ROOT=str(case/'models'))
    original = case/'own-portrait.png'
    def chunk(kind, data):
        return struct.pack('!I', len(data))+kind+data+struct.pack('!I', zlib.crc32(kind+data))
    original.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR', struct.pack('!2I5B', 320, 480, 8, 2, 0, 0, 0))+
                         chunk(b'IDAT', zlib.compress((b'\0'+bytes([70, 100, 160])*320)*480))+chunk(b'IEND', b''))
    server = ui = None
    result = dict(case=str(case), width=args.width, gpu=False, runtime_logs_read=False)
    try:
        server = subprocess.Popen([str(C/'target/debug/examples/timeline_fixture_server'), str(case), '--private-storage'],
                                  env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic()+30
        while not (case/'evidence/endpoint.json').exists():
            assert server.poll() is None and time.monotonic() < deadline
            time.sleep(.05)
        endpoint = json.loads((case/'evidence/endpoint.json').read_text())
        click = lambda name: {'action': 'click_control', 'name': name}
        idle = {'action': 'wait_idle'}
        actions = [idle, {'action': 'prepare_animate_image', 'path': str(original)},
                   {'action': 'wait_time', 'milliseconds': 300}, click('image-animate'), idle,
                   {'action': 'expect_project', 'pointer': '/default_model/variant', 'value': 'i2v'},
                   {'action': 'expect_project', 'pointer': '/scenes/0/bindings/0/role', 'value': 'first_frame'},
                   {'action': 'expect_project', 'pointer': '/scenes/0/bindings/0/enabled', 'value': True},
                   {'action': 'expect', 'scenes': 1, 'takes': 0},
                   {'action': 'wait_time', 'milliseconds': 700},
                   {'action': 'screenshot', 'file': 'animate-timeline.png'},
                   {'action': 'save'}, idle, {'action': 'quit'}]
        plan = case/'plan.json'
        plan.write_text(json.dumps(dict(destination=endpoint['destination'], actions=actions)))
        ui = subprocess.Popen([str(args.ui), '--project', str(case/'ui'), '--api', endpoint['api'],
                               '--title', 'Imagine own Animate acceptance', '--qa-script', str(plan),
                               '--qa-window-size', str(args.width), '760'],
                              env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert ui.wait(timeout=100) == 0
        qa = json.loads((case/'native-qa-result.json').read_text())
        result.update({k: qa[k] for k in ['passed', 'error', 'actions_completed', 'screenshots', 'layout']})
        assert qa['passed'], qa['error']
        documents = list((case/'private/projects').glob('*/project.json'))
        # Read only this process's newly created procedural project, never production data.
        generated = [json.loads(path.read_text()) for path in documents]
        animated = [d for d in generated if d['project']['default_model']['variant'] == 'i2v']
        assert len(animated) == 1
        document = animated[0]
        assert document['resources'][0]['source'] == str(original)
        assert len(document['project']['scenes']) == 1
        assert not document.get('takes')
        result['saved_first_frame_original'] = str(original)
    finally:
        for name, process in [('ui', ui), ('server', server)]:
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGTERM if name == 'ui' else signal.SIGINT)
                process.wait(timeout=30)
        result['own_processes_reaped'] = all(p is None or p.poll() is not None for p in [ui, server])
        (case/'evidence/result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
