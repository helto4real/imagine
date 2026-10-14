"""Summarize opt-in allocation and capacity diagnostics, including aborted runs."""
import argparse
import hashlib
import json
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
args = parser.parse_args()
run = args.run.resolve()
lines = (run/'generation.log').read_text(errors='replace').splitlines()
phase = 'conditioning_or_setup'
events = []
evictions = []
owners = {}
for index, line in enumerate(lines, 1):
    if 'generating image:' in line: phase = 'diffusion'
    if 'decoding 1 latents' in line: phase = 'vae'
    if phase == 'vae' and 'released params backend buffers' in line:
        evictions.append({'line': index, 'text': line})
    if '[memtrace] ' not in line: continue
    payload = line.split('[memtrace] ', 1)[1]
    kind = payload.split()[0]
    item = {'kind': kind, 'line': index, 'model_phase': phase}
    for key, value in re.findall(r'(\w+)=("[^"]*"|[^\s]+)', payload):
        if value.startswith('"'): value = value[1:-1]
        elif value.isdecimal(): value = int(value)
        item[key] = value
    events.append(item)
    if kind == 'execute': owners[item['owner']] = item['name']
for item in events:
    if 'owner' in item: item['runner_name'] = owners.get(item['owner'], 'unknown')
vae = [e for e in events if e['model_phase'] == 'vae']
pools = [e for e in vae if e['kind'] == 'pool']
failed = [e for e in pools if e.get('phase') == 'after' and e.get('status') != 0]
vae_owner = next((e['owner'] for e in vae if e['kind'] == 'execute' and e.get('name') == 'wan_vae'), None)
execute_line = next((e['line'] for e in vae if e['kind'] == 'execute' and e.get('name') == 'wan_vae'), None)
checks = [e for e in vae if e['kind'] == 'capacity' and e.get('owner') == vae_owner and e['line'] < execute_line]
last_ready = next((e for e in reversed(checks) if e['fits'] == 1), None)
last_device = next((e for e in reversed(vae) if e['kind'] == 'device' and e.get('owner') == vae_owner and e['line'] < execute_line), None)
failures = []
for failure in failed:
    before = next(e for e in reversed(pools) if e.get('phase') == 'before' and e['pool'] == failure['pool'] and e['line'] < failure['line'])
    tensors = [e for e in vae if e['kind'] == 'tensor' and before['line'] < e['line'] < failure['line']]
    failures.append({'before': before, 'after': failure, 'tensors': tensors,
                     'growth_shortfall_bytes': max(0, before['grow']-before['free'])})
summary = {'run': str(run), 'binary_sha256': (run/'binary-sha256.txt').read_text().strip(),
           'process': json.loads((run/'result.json').read_text()), 'trace_events': len(events),
           'vae_owner': vae_owner, 'last_ready_capacity_before_vae': last_ready,
           'last_device_before_vae': last_device,
           'vae_evictions_before_execution': [e for e in evictions if execute_line and e['line'] < execute_line],
           'vae_pool_growth_events': pools, 'failed_allocations': failures,
           'limits': 'Metadata logging and memory queries may affect timing. Pool logs cover growth, not every reuse or destruction; not a whole-device allocation census.'}
if (run/'image.png').exists():
    from PIL import Image
    image = Image.open(run/'image.png').convert('RGB')
    pixels = hashlib.sha256(image.tobytes()).hexdigest()
    summary['image'] = {'size': image.size, 'rgb_sha256': pixels,
                        'bit_identical_reference': pixels == '253d91ada4f98feb50bc9139619d00526594123006d37a9b4589e15cfe43a7f2'}
(run/'memory-events.json').write_text(json.dumps(events, indent=2)+'\n')
(run/'memory-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
