#!/usr/bin/env python3
"""Reserved native-only regression probes with synthetic data and no weights."""
import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from h3_gpu_commands import CANONICAL_SOCKET, native_command, reservation_command, sha

p = argparse.ArgumentParser(description=__doc__)
for name in ('root', 'site', 'binary', 'own-root', 'reservation', 'entry', 'output'):
    p.add_argument('--' + name, type=Path, required=True)
p.add_argument('--test', choices=[
    'cuda_convrot_dequant_tests::cuda_pin_refusal_discard_does_not_poison_next_convrot_launch',
    'sol_chunked_tests::sol_chunked_producer_preserves_bootstrap_streaming_statistics_routing_and_errors',
    'full-primitives',
], required=True)
a = p.parse_args()
project = Path(__file__).resolve().parents[2]
assert a.own_root.resolve().is_relative_to(project / 'comfyui/h3-stage4-20261020')
assert not a.output.exists()
a.output.mkdir(parents=True)
work = a.own_root / 'native-probes' / a.output.name
work.mkdir(parents=True)
frozen = work / 'primitives'
shutil.copyfile(a.binary, frozen)
frozen.chmod(0o500)
args = native_command(project, a.root, a.site, frozen, work)
args += ['--', '/runtime/primitives', '--test-threads=1', '--nocapture']
if a.test != 'full-primitives':
    args += [a.test, '--exact']
    if a.test.startswith('cuda_convrot_dequant_tests::'):
        args += ['--ignored']
manifest = work / 'reservation.json'
command = reservation_command(a.reservation, a.entry, CANONICAL_SOCKET,
                              'h3-stage4-' + a.output.name, manifest, args, timeout=1800)
(a.output / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
record = dict(schema=1, result='running', trained_weights=0, test=a.test,
              binary_sha256=sha(frozen), controller_sha256=sha(a.reservation),
              entry_sha256=sha(a.entry))
emitted = json.loads((work / 'native-command.json').read_text())
record['bridge_sha256'] = emitted['bridge_sha256']
record['bridge_provenance_sha256'] = emitted['bridge_provenance_sha256']
selection = a.output / 'selection.json'
selection.write_text(json.dumps(record, indent=2) + '\n')
with (a.output / 'worker.log').open('w') as log:
    completed = subprocess.run(command, cwd=project, stdout=log, stderr=subprocess.STDOUT)
record['returncode'] = completed.returncode
record['reservation'] = json.loads(manifest.read_text())
assert record['reservation']['report']['namespace_exited'] is True
assert record['reservation']['report']['worker']['finish_sent'] is True
log = (a.output / 'worker.log').read_text()
summaries = re.findall(r'test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;', log)
record['tests_passed'] = int(summaries[0][0]) if len(summaries) == 1 else 0
record['tests_failed'] = int(summaries[0][1]) if len(summaries) == 1 else 0
record['tests_ignored'] = int(summaries[0][2]) if len(summaries) == 1 else 0
selection_matches = record['tests_passed'] > 1 if a.test == 'full-primitives' else record['tests_passed'] == 1
if a.test == 'full-primitives':
    announced = re.findall(r'^running (\d+) tests$', log, re.MULTILINE)
    record['announced_tests'] = int(announced[0]) if len(announced) == 1 else 0
    selection_matches &= record['announced_tests'] == record['tests_passed'] + record['tests_failed'] + record['tests_ignored']
record['result'] = 'passed' if completed.returncode == 0 and selection_matches else 'failed'
selection.write_text(json.dumps(record, indent=2) + '\n')
raise SystemExit(0 if record['result'] == 'passed' else 1)
