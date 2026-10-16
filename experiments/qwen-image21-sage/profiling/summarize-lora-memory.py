"""Summarize authorized allocation sizes; not an estimate of physical peak VRAM."""
import argparse
import collections
import csv
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('run', type=Path)
a = p.parse_args()
runs = json.loads((a.run / 'timings.json').read_text())['runs']
with (a.run / 'memory.tsv').open() as f:
    events = [{k: int(v) for k, v in r.items()} for r in csv.DictReader(f, delimiter='\t')]
# CUPTI 13.4 DEVICE=3. Do not sum allocation/release events into physical
# usage: this backend also uses VMM and the event balance is not a VRAM gauge.
stats = {r['case']: dict(allocated_device_bytes=0, allocation_count=0,
                        allocation_sizes=collections.Counter()) for r in runs}
for event in events:
    if event['memory_kind'] != 3 or event['operation'] != 1:
        continue
    for run in runs:
        if run['start_ns'] <= event['timestamp_ns'] <= run['end_ns']:
            s = stats[run['case']]
            s['allocated_device_bytes'] += event['bytes']
            s['allocation_count'] += 1
            s['allocation_sizes'][event['bytes']] += 1
            break
result = dict(scope='CUPTI DEVICE allocation event sizes; not physical peak or live VRAM', runs=stats)
(a.run / 'memory-summary.json').write_text(json.dumps(result, indent=2) + '\n')
for name, s in stats.items():
    print(name, 'allocation_count=', s['allocation_count'],
          'largest_allocation_sizes=', sorted(s['allocation_sizes'].items(), reverse=True)[:6])
