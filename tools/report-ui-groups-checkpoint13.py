#!/usr/bin/env python3
"""Freeze own fixture results, never runtime console logs or user data."""
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
    evidence = OWN / 'evidence'
    build = json.loads((evidence / 'checks-97c2787f.json').read_text())
    assert len(build['checks']) == 14 and all(row['exit'] == 0 for row in build['checks'])
    count = sum(int(re.search(r'(\d+) passed', line).group(1))
                for row in build['checks'] for line in row['test_results'])
    assert count == 217
    cases = []
    for name in ['characterui04', 'characterui05', 'storyui14', 'storyui15', 'storyui16', 'libraryui15']:
        path = OWN / name / 'evidence/result.json'
        result = json.loads(path.read_text())
        assert result['passed'] and result['own_processes_reaped'] and result['qa']['passed']
        assert result['server_exit'] == 0 and result['ui_exit'] == 0
        cases.append({
            'case': name, 'result': str(path),
            'actions': result['qa']['actions_completed'],
            'seconds': result['qa']['seconds'], 'layout': result['qa']['layout'],
            'screenshots': result['qa']['screenshots'],
            'server_pid': result['server_pid'], 'ui_pid': result['ui_pid'],
            'server_exit': result['server_exit'], 'ui_exit': result['ui_exit'],
            'own_processes_reaped': True,
        })
    report = {
        'checkpoint': 13, 'complete_plan': False, 'published': False,
        'storage_stage_started': False, 'package': build['package'], 'protocol': 11,
        'checks': str(evidence / 'checks-97c2787f.json'), 'tests_passed': count,
        'check_stages': 14, 'native': cases,
        'functional_core_groups_completed_this_checkpoint': [4, 5],
        'shared_final_group_acceptance': False,
        'worker_unchanged_from_checkpoint12': True,
        'gpu_jobs': 0, 'gpu_reservations_taken': 0, 'own_processes_reaped': True,
        'runtime_console_logs_read': False, 'user_data_read': False,
        'production_changed': False, 'production_selector': 'comfyui/products/20261021T165201Z-6dd93d2e',
        'new_rejections': [], 'earlier_denied_operations_retried': False,
        'new_implementation': [
            'Standalone full character editor, no project required, contextual image/voice import',
            'Portable rich image/voice/reference/relationship metadata and resource-ID remapping',
            'Profile-owned Story transport, explicit advanced override, preferred Story profile persistence',
            'Replace preserves historical Takes/resources while replacing scene, Section and audio settings',
            'Retained progress on failure and actual observed-work ETA during expansion',
        ],
        'visual_inspection': {
            'inspected': ['characterui05/library-character-error-retained.png',
                          'storyui15/story-failed-retained.png',
                          'storyui15/story-estimated-progress.png'],
            'final_visual_approved': False,
            'observed_remaining': 'Character Save error is in parent library and partly obscured by editor; move feedback inside editor during layout closure.',
        },
        'open_requirements': [
            'Editor-local Save error; broad missing-media/storage-error UI feedback',
            'Rich metadata, provider/pair and removed-scene Take older-reader/rollback compatibility',
            'Timeline, optional reviewed Generate, Final and exact all15 closure ledger',
            'LoRA catalog/filter/info/short names/formats/warnings/errors',
            'K1/K2 and own-fixture bugs001/002/003, full image/video regression',
            'Shared normal/smaller long-value, empty/full, privacy, keyboard, tooltip and icon visual acceptance',
            'Protocol11 complete final runtime/start/compatibility acceptance before publication',
            'Full Python functional/algorithmic and audited memory parity; Exp/P08 remaining gaps',
            'Earlier specifically denied Exp/log operations remain prohibited',
        ],
        'next': 'Complete coherent Timeline, optional reviewed Generate and Final flows, with finite LoRA UI closure; then remaining ledger and common final acceptance before publication.',
    }
    with (evidence / 'UI-GROUPS-CHECKPOINT-13.json').open('x') as output:
        json.dump(report, output, indent=2)
    plan = (ROOT / 'design/ui-group-implementation.md').read_text()
    body = plan[plan.index('## Checkpoint 13:'):]
    prefix = '# UI groups checkpoint13 — verified partial delivery\n\n'
    prefix += f'Candidate `{build["package"]}`, protocol11; not published.\n\n'
    prefix += '217 tests and14 check stages pass. Six native own-fixture flows pass; all own processes reaped. No GPU or reservation, no user data or runtime console logs read.\n\n'
    prefix += 'Evidence: `UI-GROUPS-CHECKPOINT-13.json`, `checks-97c2787f.json`, source-checkpoint-13.json and ui-groups-checkpoint-12-to-13.patch. Native cases are listed below.\n\n'
    with (evidence / 'UI-GROUPS-CHECKPOINT-13.md').open('x') as output:
        output.write(prefix + body)
    print(json.dumps({'report': str(evidence / 'UI-GROUPS-CHECKPOINT-13.md'),
                      'package': build['package'], 'tests': count,
                      'native': [{'case': c['case'], 'actions': c['actions']} for c in cases],
                      'own_processes_reaped': True, 'gpu_jobs': 0, 'published': False}))


if __name__ == '__main__':
    main()
