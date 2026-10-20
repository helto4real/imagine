#!/usr/bin/env python3
"""Verify retained Stage 4 gates and assemble a source-addressed handoff index."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--source-snapshot', type=Path, required=True)
p.add_argument('--native-provenance', type=Path, required=True)
p.add_argument('--own-root', type=Path, required=True)
a = p.parse_args()
project = Path(__file__).resolve().parents[2]
base = project / 'design/h3-stage4-evidence'
assert not a.output.exists()
assert a.own_root.resolve().is_relative_to(project / 'comfyui/h3-stage4-20261020')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


snapshot = read(a.source_snapshot)
for name, digest in snapshot['implementation'].items():
    assert sha(project / name) == digest, name
for name, digest in snapshot['reference_inputs'].items():
    assert sha(Path(name)) == digest, name
provenance = read(a.native_provenance)
library = a.native_provenance.parent / 'libimagine_kitchen.so'
assert sha(library) == provenance['library_sha256']
for name, digest in provenance['native_sources'].items():
    assert sha(project / name) == digest, name
checks = [base / 'complete-cpu-v4/selection.json',
          base / 'pin-discard-cuda-06/selection.json',
          base / 'sol-chunked-cuda-01/selection.json',
          base / 'full-primitives-torch-registration-01/selection.json',
          base / 'trained-lifecycle-cancel-after-loaded-02/selection.json',
          base / 'trained-lifecycle-missing-input-02/selection.json']
checks += [base / (name + '-cuda-source-build-02/selection.json')
           for name in ('video', 'audio', 'dit', 'sampling', 'conditioning', 'quantized')]
comparisons = []
for run, phases in [('noref-final', ('text', 'dit', 'video', 'audio')),
                    ('reference-final', ('text', 'reference', 'dit', 'video', 'audio')),
                    ('noref-final-repeat', ('text', 'dit', 'video', 'audio'))]:
    comparisons += [base / f'trained-{run}-{phase}-comparison-01.json' for phase in phases]
checks += comparisons
for path in checks:
    assert read(path)['result'] == 'passed', path
for path in checks:
    value = read(path)
    if 'bridge_sha256' in value:
        assert value['bridge_sha256'] == provenance['library_sha256'], path
    for run in value.get('runs', [value]):
        if 'reservation' in run:
            report = run['reservation']['report']
            assert report['namespace_exited'] and report['worker']['child_reaped']
            assert report['worker']['finish_sent']
models = read(base / 'selected-model-sha256.json')
assert models['result'] == 'passed' and len(models['components']) == 4
memory = []
for run, engine in [('noref-final-repeat-01', 'native'), ('noref-cold-01', 'original'),
                    ('reference-final-01', 'native'), ('reference-cold-01', 'original')]:
    phases = ('text', 'reference', 'dit', 'video', 'audio') if run.startswith('reference') else ('text', 'dit', 'video', 'audio')
    for phase in phases:
        path = base / f'trained-{run}-{engine}-{phase}/selection.json'
        selection = read(path)
        assert selection['result'] == 'passed' and not selection.get('exercise')
        report = selection['phase_report']
        if engine == 'native':
            assert selection['bridge_sha256'] == provenance['library_sha256']
            peak = report['allocator_after_release']['peak_allocated']
            after = report['allocator_after_release']['allocated']
            assert report['unloaded']['placement']['loaded_bytes'] == 0
            assert report['unloaded']['pinned_bytes'] == 0
        else:
            peak = report['peak_allocated_bytes']
            after = report['cuda_after_release']['allocated']
        memory.append(dict(run=run, engine=engine, phase=phase,
                           peak_allocated_bytes=peak, allocated_after_release=after,
                           phase_selection_sha256=sha(path)))
clips = []
for run, name in [('noref-final-repeat-01', 'no-reference'), ('reference-final-01', 'synthetic-reference')]:
    path = a.own_root / 'trained' / run / 'native' / (name + '.mp4')
    metadata = read(path.with_suffix('.json'))
    assert metadata['result'] == 'passed' and sha(path) == metadata['export_sha256']
    clips.append(dict(path=str(path), sha256=sha(path), metadata_sha256=sha(path.with_suffix('.json'))))
record = dict(schema=1, result='passed',
              scope='Selected standalone H3 Stage 4 core, technical 384x256 / 39-frame CFG1 Euler profile',
              stage3_registry_complete=False, product_integration=False, visual_quality_acceptance=False,
              source_snapshot_sha256=sha(a.source_snapshot),
              native_library_sha256=provenance['library_sha256'],
              native_provenance_sha256=sha(a.native_provenance),
              model_lock_sha256=sha(base / 'selected-model-sha256.json'),
              checks={str(path.relative_to(project)): sha(path) for path in checks},
              cpu_h3_tests=12, cuda_h3_synthetic_histories=101,
              full_primitive_tests=read(base / 'full-primitives-torch-registration-01/selection.json')['tests_passed'],
              exact_trained_phase_comparisons=len(comparisons),
              exact_trained_artifact_comparisons=sum(len(read(path)['cases']) for path in comparisons),
              lifecycle='Real trained cancellation after load, failure cleanup, subsequent reference-image encode and complete repeated generation; namespace reaped before finish send',
              memory=memory, clips=clips,
              limits='No broker finish acknowledgement, controller SIGKILL resilience, production integration, broad Stage 3 completeness or quality/performance benchmark is claimed')
a.output.write_text(json.dumps(record, indent=2) + '\n')
print('PASS selected H3 Stage 4 acceptance:', len(checks), 'retained checks')
