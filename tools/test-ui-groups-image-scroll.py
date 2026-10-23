#!/usr/bin/env python3
"""Bounded image scroll QA: empty own UI, no server, jobs, models or screenshots.

--self-test checks only the synthetic action plan. Native execution requires a
separately authorized UI build/run and the QA-only workspace scroll hooks.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
OWN = Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')
OUTER = 'image-workspace-scroll'
INNER = 'image-inspector-scroll'
SEED = 'image-seed-mode'


def actions():
    """Real wheel/mouse input; observation never sets scroll or editor state."""
    wait = lambda: {'action': 'wait_time', 'milliseconds': 700}
    click = lambda name: {'action': 'click_control', 'name': name}
    observe = lambda name, label, baseline=None: {
        'action': 'observe_scroll', 'name': name, 'label': label,
        'compare_to': baseline, 'minimum_y': 250 if name == OUTER else 0,
        'visible_control': SEED,
    }
    sequence = [
        # WaitIdle requires a connected Timeline session. This deliberately
        # disconnected image-only case has neither a server nor a session.
        wait(), click('workspace-image'), wait(), click('queue-toggle'), wait(),
        {'action': 'scroll_control', 'name': OUTER, 'delta_y': -1000}, wait(),
        {'action': 'scroll_to_control', 'scroll': INNER, 'name': SEED}, wait(),
        observe(OUTER, 'outer-bottom'), observe(INNER, 'inspector-bottom'),
    ]
    for label, target in [('header', 'workspace-image'), ('queue', 'queue-toggle'), ('field', SEED)]:
        sequence.extend([
            {'action': 'hover_control', 'name': target}, wait(),
            observe(OUTER, 'outer-after-'+label, 'outer-bottom'),
            observe(INNER, 'inspector-after-'+label, 'inspector-bottom'),
        ])
    sequence.extend([
        # A full unclipped rectangle alone does not prove input reaches the
        # lower field. Click the actual ComboBox and observe its popup state.
        click(SEED), {'action': 'expect_combo', 'name': SEED, 'open': True},
        {'action': 'key', 'key': 'Escape'}, wait(),
        {'action': 'expect_combo', 'name': SEED, 'open': False},
        observe(OUTER, 'outer-after-popup', 'outer-bottom'),
        observe(INNER, 'inspector-after-popup', 'inspector-bottom'),
        {'action': 'quit'},
    ])
    return sequence


class PlanTests(unittest.TestCase):
    def test_no_generation_server_media_or_screenshot_actions(self):
        sequence = actions()
        allowed = {'wait_time', 'click_control', 'scroll_control', 'scroll_to_control',
                   'observe_scroll', 'hover_control', 'expect_combo', 'key', 'quit'}
        self.assertLessEqual({action['action'] for action in sequence}, allowed)
        self.assertEqual([a['name'] for a in sequence if a['action'] == 'click_control'],
                         ['workspace-image', 'queue-toggle', SEED])

    def test_mouse_motion_is_checked_against_both_original_offsets(self):
        checks = [a for a in actions() if a['action'] == 'observe_scroll']
        self.assertEqual(len(checks), 10)
        self.assertEqual({a['compare_to'] for a in checks},
                         {None, 'outer-bottom', 'inspector-bottom'})
        self.assertTrue(all(a['visible_control'] == SEED for a in checks))
        self.assertTrue(all(a['minimum_y'] == 250 for a in checks if a['name'] == OUTER))

    def test_popup_open_and_escape_are_both_observed(self):
        checks = [a['open'] for a in actions() if a['action'] == 'expect_combo']
        self.assertEqual(checks, [True, False])


def run(case_name, package):
    assert case_name.isascii() and case_name.isalnum()
    package = package.resolve(strict=True)
    assert package.is_relative_to(ROOT/'comfyui/ui-groups-20261021/packages')
    mount = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(OWN)]))['filesystems'][0]
    assert (mount['target'], mount['source'], mount['fstype']) == (
        '/home', '/dev/mapper/cryptroot[/@home]', 'btrfs')
    assert not os.statvfs(OWN).f_flag & os.ST_RDONLY
    case = OWN/case_name
    case.mkdir()
    for name in ['evidence', 'tmp', 'cache', 'config', 'share', 'models', 'ui', 'output']:
        (case/name).mkdir()
    prefs = case/'ui/.imagine'
    prefs.mkdir()
    (prefs/'ui-settings.json').write_text(json.dumps({
        'privacy_mode': False, 'output_folder': str(case/'output'),
    }))
    destination = {
        'root': str(case), 'mount': '/home', 'source': '/dev/mapper/cryptroot',
        # Linux statvfs exposes the two native i32 statfs fsid words as one u64.
        'filesystem': 'btrfs',
        'fsid': list(struct.unpack('=ii', struct.pack('=Q', os.statvfs(case).f_fsid))),
    }
    plan = case/'qa-plan.json'
    plan.write_text(json.dumps({'destination': destination, 'actions': actions()}, indent=2))
    env = dict(os.environ, TMPDIR=str(case/'tmp'), CUDA_VISIBLE_DEVICES='',
               LIBGL_ALWAYS_SOFTWARE='1', SDL_AUDIODRIVER='dummy', PYTHONDONTWRITEBYTECODE='1',
               XDG_CACHE_HOME=str(case/'cache'), XDG_CONFIG_HOME=str(case/'config'),
               XDG_DATA_HOME=str(case/'share'), QWEN_MODEL_ROOT=str(case/'models'))
    # Hold a newly assigned loopback port without listen(): the UI cannot reach
    # a user server, and this case never starts a server or submits a job.
    ui = None
    with socket.socket() as unavailable:
        unavailable.bind(('127.0.0.1', 0))
        api = f'http://127.0.0.1:{unavailable.getsockname()[1]}'
        result = {'case': str(case), 'package': str(package), 'gpu': False,
                  'server_started': False, 'jobs_submitted': False, 'screenshots': False,
                  'runtime_logs_read': False, 'width': 600, 'height': 760}
        try:
            with (case/'evidence/ui.log').open('x') as output:
                ui = subprocess.Popen([
                    str(package/'imagine-ui-qa'), '--project', str(case/'ui'), '--api', api,
                    '--title', 'Imagine own empty image scroll QA', '--qa-window-size', '600', '760',
                    '--qa-script', str(plan),
                ], env=env, stdout=output, stderr=subprocess.STDOUT)
            result['ui_pid'] = ui.pid
            result['ui_exit'] = ui.wait(timeout=140)
            # Only the separate geometry report. Never parse the general QA
            # report's prompts/jobs/session/final diagnostics or any logs.
            qa = json.loads((case/'native-scroll-result.json').read_text())
            result['actions_completed'] = qa['actions_completed']
            result['scroll_checks'] = qa['scroll_checks']
            result['layout'] = qa['layout']
            result['qa_passed'] = qa['passed']
            assert qa['passed'], 'Native scroll assertions failed; own log remains unread'
            assert qa['layout']['logical_size'] == [600, 760]
            assert len(qa['scroll_checks']) == 10
            assert all(value['queue_expanded'] for value in qa['scroll_checks'].values())
            result['passed'] = result['ui_exit'] == 0
        except Exception as error:
            result['error'] = repr(error)
        finally:
            if ui is not None:
                if ui.poll() is None:
                    ui.send_signal(signal.SIGTERM)
                result['ui_exit'] = ui.wait(timeout=30)
            result['own_processes_reaped'] = ui is None or ui.poll() is not None
            (case/'evidence/result.json').write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
    return 0 if result.get('passed') and result['own_processes_reaped'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', nargs='?')
    parser.add_argument('--package', type=Path)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(PlanTests)
        return 0 if unittest.TextTestRunner().run(suite).wasSuccessful() else 1
    if args.plan_only:
        print(json.dumps(actions(), indent=2))
        return 0
    if not args.case or not args.package:
        parser.error('native run requires a new own case and --package')
    return run(args.case, args.package)


if __name__ == '__main__':
    raise SystemExit(main())
