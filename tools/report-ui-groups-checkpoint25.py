#!/usr/bin/env python3
"""Record finite G8 implementation and actual own native acceptance evidence."""
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OWN = Path('/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e/ui-groups-01')
PACKAGE = ROOT/'comfyui/ui-groups-20261021/packages/checks-7fb46fce'


def main():
    mount = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(OWN)]))['filesystems'][0]
    assert (mount['target'], mount['source'], mount['fstype']) == ('/run/media/veracrypt1', '/dev/mapper/veracrypt1', 'ext3')
    assert not os.statvfs(OWN).f_flag & os.ST_RDONLY
    evidence = OWN/'evidence'
    checks = json.loads((evidence/'checks-7fb46fce.json').read_text())
    assert len(checks['checks']) == 17 and all(c['exit'] == 0 for c in checks['checks'])
    tests = sum(int(re.search(r'(\d+) passed', s).group(1)) for c in checks['checks'] for s in c['test_results'])
    native = {}
    for name in ['selectionui09', 'selectionui10', 'cp25group01', 'cp25group02']:
        result = json.loads((OWN/name/'evidence/result.json').read_text())
        assert result['passed'] and result['own_processes_reaped']
        qa = result['qa']
        native[name] = {'report': str(OWN/name/'evidence/result.json'), 'actions': qa['actions_completed'], 'seconds': qa['seconds'], 'passed': True, 'own_processes_reaped': True, 'screenshots': qa['screenshots'], 'navigation_checks': qa['navigation_checks']}
    prior = json.loads((evidence/'UI-GROUPS-CHECKPOINT-24.json').read_text())
    groups = prior['groups']
    g8 = next(g for g in groups if g['group'] == 8)
    g8.update(remaining_implementation=[], remaining_acceptance=['Joint whole-UI final regression/visual/keyboard acceptance alongside the remaining groups. The bounded G8 command, normal/small layout and waveform flows now pass.'], blocker='No known G8 implementation blocker; common final acceptance remains')
    report = {'checkpoint': 25, 'complete_plan': False, 'published': False, 'storage_stage_started': False, 'ui_server_package': str(PACKAGE), 'protocol': prior['protocol'], 'worker_package': prior['worker_package'], 'worker_sha256': prior['worker_sha256'], 'check_report': str(evidence/'checks-7fb46fce.json'), 'check_stages': 17, 'test_cases': tests, 'native_cases': native, 'groups': groups, 'g8_functionally_complete': True, 'g8_finally_accepted': False, 'gpu_used': False, 'own_processes_active': False, 'runtime_console_logs_read': False, 'user_data_read': False, 'no_denied_operations_retried': True, 'remaining_memory_parity': prior['remaining_memory_parity'], 'full_python_pipeline_parity': False, 'historical_user_generate_root_cause_proven': False, 'next': 'Complete remaining G9 continuity/runtime paths, G11 sampling preview and G12 Takes/AB/Captures, with bounded parallel G7 catalog and G15 ingress/chooser work; final whole-UI acceptance then publication, storage later.'}
    with (evidence/'UI-GROUPS-CHECKPOINT-25.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    lines = ['# Remaining UI requirements after CP25', '', 'Implementation and final acceptance remain separate. No full Python-pipeline acceptance is asserted.', '']
    for group in groups:
        lines += [f'## Group {group["group"]}', '', 'Remaining implementation:']
        lines += ['- '+s for s in group['remaining_implementation']] or ['- No additional known implementation gap; acceptance below remains.']
        lines += ['', 'Remaining acceptance:'] + ['- '+s for s in group['remaining_acceptance']] + ['']
    lines += ['## Cross-cutting', '- Full Python functional/algorithmic memory parity remains open.', '- The reproduced large-reference path works; historical user Generate root cause remains unknown.', '- Whole UI visual/keyboard/Hide acceptance, then reversible publication, then separate storage stage.']
    with (evidence/'UI-GROUPS-OPEN-25.md').open('x') as stream:
        stream.write('\n'.join(lines)+'\n')
    print(json.dumps({'checkpoint': 25, 'tests': tests, 'native': {k: v['actions'] for k, v in native.items()}, 'package': str(PACKAGE), 'published': False}))


if __name__ == '__main__':
    main()
