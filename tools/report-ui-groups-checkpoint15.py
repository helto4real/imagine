#!/usr/bin/env python3
"""Freeze own structured acceptance results; never read runtime console logs."""
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OWN = Path('/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e/ui-groups-01')


def main():
    mount = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(OWN)]))['filesystems'][0]
    assert (mount['target'], mount['source'], mount['fstype']) == ('/run/media/veracrypt1', '/dev/mapper/veracrypt1', 'ext3')
    assert not os.statvfs(OWN).f_flag & os.ST_RDONLY
    evidence = OWN/'evidence'
    checks = evidence/'checks-f4a97856.json'
    build = json.loads(checks.read_text())
    assert len(build['checks']) == 14 and all(c['exit'] == 0 for c in build['checks'])
    count = sum(int(re.search(r'(\d+) passed', line).group(1)) for c in build['checks'] for line in c['test_results'])
    assert count == 255
    cases = []
    failures = {'qchain02', 'qchain04', 'qchain06'}
    for name in ['qchain01', 'qchain02', 'qchain03', 'qchain04', 'qchain05', 'qchain06', 'qchain07', 'qchain08', 'qfilename01', 'qfilename02']:
        path = OWN/name/'evidence/launch-result.json'
        d = json.loads(path.read_text())
        assert d['qa']['passed'] == (name not in failures)
        assert d['own_processes_reaped'] and d['server_exit'] == 0 and d['ui_exit'] == 0
        cases.append({
            'case': name, 'result': str(path), 'package': d['package'],
            'passed': d['qa']['passed'], 'error': d['qa']['error'],
            'actions': d['qa']['actions_completed'], 'seconds': d['qa']['seconds'],
            'layout': d['qa']['layout'], 'screenshots': d['qa']['screenshots'],
            'final_diagnostics': d['qa'].get('final_diagnostics'),
            'server_pid': d['server_pid'], 'ui_pid': d['ui_pid'],
            'server_exit': 0, 'ui_exit': 0, 'own_processes_reaped': True,
            'generation_to_final': d.get('generation_to_final'),
            'duration': d.get('frozen_duration_verified'), 'pcm': d.get('audio_pcm'),
            'export_path': d.get('export_path'), 'export_sha256': d.get('export_sha256'),
            'retained_result': d.get('restored'),
        })
    report = {
        'checkpoint': 15, 'complete_plan': False, 'published': False,
        'storage_stage_started': False, 'package': build['package'], 'protocol': 11,
        'tests_passed': count, 'check_stages': 14, 'checks': str(checks), 'native': cases,
        'core_functionally_complete': [4, 5, 10], 'shared_final_acceptance': False,
        'partially_implemented_this_checkpoint': [13], 'group8_changed_this_checkpoint': False,
        'inference_worker_changed': False, 'gpu_jobs': 0, 'gpu_reservations_taken': 0,
        'own_processes_reaped': True, 'saved_runtime_console_logs_read': False,
        'console_note': 'A failing integration test earlier automatically displayed own synthetic scheduler status lines. Console tail display was removed; no saved runtime console log was intentionally queried.',
        'user_data_read': False, 'production_changed': False,
        'new_rejections': [], 'earlier_denied_operations_retried': False,
        'visual_final_approved': False,
        'visually_inspected': [{'case': 'qchain07', 'file': 'final-direct-options.png', 'logical_size': [1100, 760], 'scope': 'Final top actions, filename and audio controls only; remaining controls scroll. Not whole-UI acceptance.'}],
        'unresolved_failures': [
            'qchain02/qchain06: replacement Run completed, Final preparation produced generic video-API network error; root cause unproved. Later passes do not prove a fix.',
            'qchain04: QA-only requested wait exceeded its10-second maximum; corrected, not a product failure.',
        ],
        'control_inventory': [
            'Filename accepts Unicode/spaces; only a basename is required to preserve the selected encrypted destination; no length quota or model allowlist.',
            'Existing atomic no-replace publication preserves originals on name collisions.',
            'Pending Final compares actual captured inputs and uses existing Run/admission semantics. Session/project changes end only continuation, preserving server work.',
            'Connection/timeout/interrupted-body categories improve actual error feedback without URLs, payloads or raw errors; no automatic retry loop or new gate.',
        ],
        'open_requirements': [
            'Group13 intermittent network failure and explicit recovery, generated transitions, larger/mixed-FPS, named-delivery older readers and final runtime/visual acceptance',
            'Group8 duplicate/move/context commands/right split selection/conditional Fit/range/display/waveform/600px/keyboard/popup/Hide',
            'Groups4/5 editor save-error presentation, old rich formats/rollback and common acceptance remain explicit',
            'All other exact15-group criteria and K1/K2/BUG001/002/003 in updated source plan',
            'Full Python functional/algorithmic, audited memory and Exp/P08 gaps; denied operations remain prohibited',
        ],
        'next': 'Isolate intermittent Final API failure with structured own fixture failure/recovery evidence, then remaining Timeline commands/range/views/waveform and whole closure ledger before publication; storage only afterward.',
    }
    with (evidence/'UI-GROUPS-CHECKPOINT-15.json').open('x') as out:
        json.dump(report, out, indent=2)
    plan = (ROOT/'design/ui-group-implementation.md').read_text()
    body = plan[plan.index('## Checkpoint 15:'):]
    header = f'# UI groups checkpoint15 — verified partial delivery\n\nCandidate: `{build["package"]}`; protocol11; not published.\n\n255 tests /14 stages pass. All own workers reaped; no GPU reservation or production change. Two intermittent network failures remain unproved, not accepted as repaired.\n\n'
    header += 'Detailed results: `UI-GROUPS-CHECKPOINT-15.json`. Source/diff: `source-checkpoint-15.json`, `ui-groups-checkpoint-14-to-15.patch`.\n\n'
    with (evidence/'UI-GROUPS-CHECKPOINT-15.md').open('x') as out:
        out.write(header+body)
    print(json.dumps({'report': str(evidence/'UI-GROUPS-CHECKPOINT-15.md'), 'package': build['package'], 'tests': count, 'passed_cases': len(cases)-len(failures), 'failed_cases_preserved': sorted(failures), 'own_processes_reaped': True, 'gpu_jobs': 0, 'published': False}))


if __name__ == '__main__':
    main()
