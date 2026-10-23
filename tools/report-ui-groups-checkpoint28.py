#!/usr/bin/env python3
"""Own incremental CP28 evidence; no production or runtime-console reads."""
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
    e = OWN/'evidence'
    report = json.loads((e/'UI-GROUPS-CHECKPOINT-27.json').read_text())
    checks = json.loads((e/'checks-18dc5e31.json').read_text())
    assert len(checks['checks']) == 20 and all(c['exit'] == 0 for c in checks['checks'])
    tests = sum(int(re.search(r'(\d+) passed', line).group(1)) for c in checks['checks'] for line in c['test_results']) + 1
    native = {}
    for name in ['importtake03', 'importtake04', 'capturescp28', 'maintenancecp28b']:
        result = json.loads((OWN/name/'evidence/result.json').read_text())
        qa = result['qa']
        assert result['passed'] and qa['passed'] and result['own_processes_reaped'], name
        native[name] = {'report': str(OWN/name/'evidence/result.json'), 'passed': True,
                        'size': qa['layout']['logical_size'], 'actions': qa['actions_completed'],
                        'seconds': qa['seconds'], 'screenshots': qa['screenshots'], 'own_processes_reaped': True}
    groups = {g['group']: g for g in report['groups']}
    groups[9]['remaining_implementation'] = [s for s in groups[9]['remaining_implementation'] if 'Blend Seam default' not in s]
    groups[12]['remaining_implementation'] = [s for s in groups[12]['remaining_implementation'] if 'Arbitrary project/external video' not in s]
    groups[12]['remaining_acceptance'] = [
        'External video project-picker Attach/Accept/Save/reopen/idempotent reimport/Preview passes actual normal/small UI. Final CPU processor also exports unknown-recipe Takes to 48 frames, 2.000 s, MP4/H264/AAC. Native OS chooser and whole-UI Final acceptance remain.',
        'Broader changed-media/obsolete preview loads, many-Take keyboard/Hide and whole-UI final visual acceptance. Generated Captures and cross-Scene source/cleanup regressions pass again in the integrated package.'
    ]
    groups[12]['blocker'] = 'No import/provenance implementation blocker; retained receipt presentation and common/native-chooser/export acceptance remain.'
    groups[15]['remaining_implementation'] = []
    groups[15]['remaining_acceptance'] = [
        'Native OS replacement picker and Cancel: implemented with cancellation/late-response CPU proofs, but no actual OS chooser automation is available in the current tool surface.',
        'Dense >2 GiB first Attach+Save requires more encrypted free space. Current free bytes are recorded; original plus upload alone exceeds 4 GiB, before project staging and saved copies. Inherited sparse-source tests are not full acceptance.',
        'Broader valid formats and large-source CheckMedia/Relink. Integrated small real HTTP and actual native project-asset Relink/Save/reopen pass with stable ID and preserved bindings.'
    ]
    groups[15]['blocker'] = 'Dense large-source capacity and actual OS picker acceptance; streaming ingress/descriptor probes and selector cancellation are integrated.'
    stat = os.statvfs(OWN)
    report.update(checkpoint=28, ui_server_package=checks['package'], test_cases=tests, check_stages=21,
                  native_cases=native, side_a_integrated=True, g12_external_take_implemented=True,
                  complete_plan=False, published=False, storage_stage_started=False,
                  own_processes_active=False, gpu_used=False, encrypted_free_bytes=stat.f_bavail*stat.f_frsize,
                  native_os_picker_tested=False, dense_large_first_attach_save_tested=False,
                  supplemental_assembly_tests={'passed': 6, 'new_distinct_case': 'accepted_external_takes_export_without_generation_recipes', 'exit': 0, 'frames': 48, 'seconds': 2.0, 'video': 'h264', 'audio': 'aac'},
                  prior_report=str(e/'UI-GROUPS-CHECKPOINT-27.json'))
    report['introduced_controls'] += [
        'Imported video uses its real content identity and source asset/revision; unknown model/seed/recipe stays absent. Import probes the selected video stream rather than trusting a file extension.',
        'Old-format export reports that an unknown imported recipe is not representable; normal Save remains available.',
        'Streaming IO uses bounded 64 KiB buffers and actual length/free-space/write/cancel outcomes; no fixed >2 GiB AV admission limit.'
    ]
    report['known_limits'] = [s for s in report['known_limits'] if 'G15 dense first' not in s]
    report['known_limits'] += groups[15]['remaining_acceptance'][:2]
    report['next'] = 'Finish retained receipt presentation and imported Take Final export, then actual continuity/Exp transport, sampling preview and generated-transition Final. Continue all OPEN28 acceptance; no publication or storage migration before full UI acceptance.'
    text = ['# UI groups checkpoint 28', '', f'Candidate: {checks["package"]}. {tests} distinct registered CPU cases pass across 20 combined stages plus a supplemental six-test assembly regression (one new case). Two intentionally ignored >2 GiB tests remain unrun.', '',
            'External video can be attached as a Take with provenance:null, no invented seed/model, actual content fingerprint and explicit source asset/revision. Existing generated recipes retain their object-shaped JSON. Cross-Scene reuse preserves import identity and repeated import reuses the existing Take.', '',
            'Import and ordinary Accept/Reject retain memory/disk agreement on pre-rename save failure; completed-rename durability uncertainty keeps the committed document. Fault injection and Save/reopen regression pass. External originals remain unchanged.', '',
            'Supplemental actual CPU Final export accepts unknown-recipe Takes: 48 frames, 2.000 s, MP4/H264/AAC, unchanged input bytes. The complete six-case assembly target passes; this is not a new GPU/Python or whole-UI Final acceptance.', '',
            'A G15 streaming modules and hooks are integrated. AV upload, reopen and library/assembly probes use descriptors without the old 2 GiB buffering gate. UI replacement selection handles Cancel and late callbacks; long file IO uses its separate client transport. Actual project-asset Relink and cache regression pass in the native UI.', '',
            'Visual QA detected the Captures modal expanding beyond a small window. Metadata and tool actions now occupy separate wrapping rows with bounded width. The fixed candidate was rerun and visually inspected at normal and small sizes.', '',
            'G9 new Blend Seam overlap is exactly Director\'s 3 video frames. Existing values 6 and user-edited 11 survive JSON reopening and mode edits; no migration changes old projects.', '', 'Actual UI evidence:']
    text += [f'- {name}: {v["actions"]} actions, {v["size"]}, {v["seconds"]:.3f}s; {v["report"]}' for name, v in native.items()]
    text += ['', 'Explicit acceptance limits:'] + ['- '+s for s in groups[15]['remaining_acceptance']]
    text += ['', f'Encrypted free bytes at checkpoint: {report["encrypted_free_bytes"]}. No deletion, full snapshot, GPU work, user process changes, denied reads or production publication. All own test processes reaped.', '',
             'CPU MP4/native fixtures are not full H3/Python pipeline or memory-parity evidence. Those gaps and all still missing implementations remain in OPEN28.', '', 'Next: '+report['next']]
    assert not (e/'UI-GROUPS-CHECKPOINT-28.json').exists()
    assert not (e/'UI-GROUPS-CHECKPOINT-28.md').exists()
    assert not (e/'UI-GROUPS-OPEN-28.md').exists()
    (e/'UI-GROUPS-CHECKPOINT-28.json').write_text(json.dumps(report, indent=2))
    (e/'UI-GROUPS-CHECKPOINT-28.md').write_text('\n'.join(text)+'\n')
    lines = ['# Remaining UI requirements after CP28', '', 'Functional implementation and whole-UI acceptance are tracked separately. No full-plan or full Python parity assertion.', '']
    for g in report['groups']:
        lines += [f'## Group {g["group"]}', '', 'Remaining implementation:'] + (['- '+s for s in g['remaining_implementation']] or ['- No additional known implementation gap; acceptance remains below.'])
        lines += ['', 'Remaining acceptance:'] + ['- '+s for s in g['remaining_acceptance']] + ['']
    lines += ['## Cross-cutting', '- Full Python functional/algorithmic memory parity remains open.',
              '- Whole UI visual/keyboard/Hide acceptance, then reversible publication, then separate storage stage.',
              '- Specific denied Exp/log operations remain prohibited.']
    (e/'UI-GROUPS-OPEN-28.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'checkpoint': 28, 'tests': tests, 'checks': 21, 'package': checks['package'], 'published': False,
                      'free_bytes': report['encrypted_free_bytes'], 'native': native}, indent=2))


if __name__ == '__main__':
    main()
