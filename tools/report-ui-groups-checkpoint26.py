#!/usr/bin/env python3
"""Record own A/B vertical-flow acceptance without promoting all of G12 to done."""
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
    prior = json.loads((evidence/'UI-GROUPS-CHECKPOINT-25.json').read_text())
    checks = json.loads((evidence/'checks-57dd0b77.json').read_text())
    assert len(checks['checks']) == 17 and all(c['exit'] == 0 for c in checks['checks'])
    tests = sum(int(re.search(r'(\d+) passed', s).group(1)) for c in checks['checks'] for s in c['test_results'])
    native = {}
    for name in ['takereview10', 'takereview11', 'selectionui12']:
        v = json.loads((OWN/name/'evidence/result.json').read_text())
        assert v['passed'] and v['own_processes_reaped']
        q = v['qa']
        native[name] = {'report': str(OWN/name/'evidence/result.json'), 'passed': True, 'size': q['layout']['logical_size'], 'actions': q['actions_completed'], 'seconds': q['seconds'], 'screenshots': q['screenshots'], 'comparisons': q['comparisons'], 'own_processes_reaped': True}
    oracle = json.loads((evidence/'director-take-selection-cp26-20261022-01.json').read_text())
    groups = prior['groups']
    g12 = next(g for g in groups if g['group'] == 12)
    g12.update(remaining_implementation=['Manual Attach as Take and Captures', 'Cleanup/removal with provenance and external-original protection'], remaining_acceptance=['Actual native Attach/capture/reopen/removal/cancel/error flows with own fixtures', 'Broader A/B missing/changed-media, obsolete-loading/cancel and full keyboard/whole-UI final acceptance; bounded default/explicit pair, swap, per-side review, single audio, loop, failure/retry and reopen now pass at normal/small sizes'], blocker='Remaining Attach/Captures/removal implementation and explicit acceptance')
    report = {'checkpoint': 26, 'complete_plan': False, 'published': False, 'storage_stage_started': False, 'ui_server_package': checks['package'], 'protocol': prior['protocol'], 'worker_package': prior['worker_package'], 'worker_sha256': prior['worker_sha256'], 'test_cases': tests, 'check_stages': 17, 'native_cases': native, 'director_pair_oracle_cases': len(oracle['cases']), 'g8_functionally_complete': True, 'g12_ab_vertical_flow_complete': True, 'g12_functionally_complete': False, 'groups': groups, 'gpu_used': False, 'audio_device': 'SDL dummy; physical speakers not tested', 'own_processes_active': False, 'runtime_console_logs_read': False, 'user_data_read': False, 'no_denied_operations_retried': True, 'full_python_pipeline_parity': False, 'remaining_memory_parity': prior['remaining_memory_parity'], 'next': 'Main: finish G12 Attach as Take/Captures and provenance-safe removal; side A G15 and side B G7 may use CP25 manifest with isolated files/targets. Then remaining continuity/preview/Final and shared final acceptance. Publication and storage stage remain later.'}
    with (evidence/'UI-GROUPS-CHECKPOINT-26.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    lines = ['# Remaining UI requirements after CP26', '', 'Functional slices and final acceptance remain separate. No complete-plan/full Python-pipeline acceptance is asserted.', '']
    for group in groups:
        lines += [f'## Group {group["group"]}', '', 'Remaining implementation:']
        lines += ['- '+s for s in group['remaining_implementation']] or ['- No additional known implementation gap; acceptance below remains.']
        lines += ['', 'Remaining acceptance:'] + ['- '+s for s in group['remaining_acceptance']] + ['']
    lines += ['## Cross-cutting', '- Full Python functional/algorithmic memory parity remains open.', '- Whole UI visual/keyboard/Hide acceptance, then reversible publication, then separate storage stage.', '- Specific denied Exp/log operations remain prohibited.']
    with (evidence/'UI-GROUPS-OPEN-26.md').open('x') as stream:
        stream.write('\n'.join(lines)+'\n')
    print(json.dumps({'checkpoint': 26, 'tests': tests, 'oracle_cases': len(oracle['cases']), 'native': {name: v['actions'] for name, v in native.items()}, 'package': checks['package'], 'published': False}))


if __name__ == '__main__':
    main()
