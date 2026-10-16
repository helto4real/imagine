"""Read only the controlled fixed-name interval and request timing records."""
import argparse
import collections
import csv
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('run',type=Path)
p.add_argument('phases',type=Path)
a=p.parse_args()
t=json.loads((a.run/'timings.json').read_text())
runs=t['runs'] if isinstance(t,dict) else t
allowed=set('merge.total weights.load lora.catalog merge.adapter_init merge.adapter_mapping merge.operand_up merge.operand_down merge.capacity merge.backend_sync merge.original_scales merge.cuda_total merge.scale_store engine.apply_loras engine.conditioning loader.total loader.validate loader.process_files loader.file_reader loader.poll_sleep loader.join loader.worker lora.attach model.load_gpu patcher.load patcher.weight lora.calculate lora.product_add lora.requantize'.split())
events=[]
for row in a.phases.read_text().splitlines():
 label,start,end=row.split('\t');assert label in allowed
 start,end=int(start),int(end);assert 0<start<=end
 events.append((label,start,end))
def union(intervals):
 total=0;last=0
 for start,end in sorted(intervals):
  total+=max(0,end-max(last,start));last=max(last,end)
 return total/1e9
# Optional GPU intervals let us distinguish sleep with/without concurrent GPU work.
gpu_intervals=[]
if (a.run/'activity.tsv').exists():
 with (a.run/'activity.tsv').open() as f:
  for row in csv.DictReader((line for line in f if not line.startswith('#')),delimiter='\t'):
   start,end=int(row['start_ns']),int(row['end_ns'])
   if start>0 and end>=start:gpu_intervals.append((start,end))
merged_gpu=[]
for start,end in sorted(gpu_intervals):
 if merged_gpu and start<=merged_gpu[-1][1]:merged_gpu[-1]=(merged_gpu[-1][0],max(end,merged_gpu[-1][1]))
 else:merged_gpu.append((start,end))
def gpu_overlap(intervals):
 result=0;index=0
 for start,end in sorted(intervals):
  while index<len(merged_gpu) and merged_gpu[index][1]<=start:index+=1
  j=index
  while j<len(merged_gpu) and merged_gpu[j][0]<end:
   c,d=merged_gpu[j];result+=max(0,min(end,d)-max(start,c));j+=1
 return result/1e9
report=[]
for run in runs:
 groups=collections.defaultdict(list)
 for label,start,end in events:
  if run['start_ns']<=start<=end<=run['end_ns']:groups[label].append((start,end))
 item=dict(case=run.get('case',run.get('name')),seconds=run.get('seconds',run.get('total_seconds')),
  phases={name:dict(count=len(intervals),sum_seconds=sum(y-x for x,y in intervals)/1e9,union_seconds=union(intervals)) for name,intervals in sorted(groups.items())})
 if 'engine.conditioning' in groups:
  merge_outside=[(x,y) for x,y in groups['merge.total'] if not any(c<=x<=y<=d for c,d in groups['engine.conditioning'])]
  item['merge_outside_conditioning_seconds']=union(merge_outside)
 if 'loader.worker' in groups and 'loader.join' in groups:
  wait_groups={}
  for parent in ['merge.operand_up','merge.operand_down','merge.original_scales']:
   waits=[];idle_tails=[];workers=[];joined=[];reader=[]
   for start,end in groups[parent]:
    part_waits=[(x,y) for x,y in groups['loader.poll_sleep'] if start<=x<=y<=end]
    part_workers=[(x,y) for x,y in groups['loader.worker'] if start<=x<=y<=end]
    assert part_workers, 'Missing worker measurement'
    done=max(y for x,y in part_workers)
    idle_tails.extend((max(x,done),y) for x,y in part_waits if y>done)
    waits.extend(part_waits);workers.extend(part_workers)
    joined.extend((x,y) for x,y in groups['loader.join'] if start<=x<=y<=end)
    reader.extend((x,y) for x,y in groups['loader.file_reader'] if start<=x<=y<=end)
   wait_groups[parent]=dict(parent_seconds=union(groups[parent]),sleep_count=len(waits),
    sleep_seconds=union(waits),worker_seconds=union(workers),join_seconds=union(joined),
    catalog_reader_seconds=union(reader),sleep_after_workers_done_seconds=union(idle_tails),
    sleep_after_workers_done_without_gpu_seconds=union(idle_tails)-gpu_overlap(idle_tails))
  item['loader_wait_breakdown']=wait_groups
 report.append(item)
(a.run/'activation-summary.json').write_text(json.dumps(dict(runs=report),indent=2)+'\n')
for item in report:
 print(item['case'],round(item['seconds'],4))
 for name,values in item['phases'].items():
  print(' ',name,values['count'],round(values['sum_seconds'],6))
 if 'loader_wait_breakdown' in item: print(' wait breakdown',item['loader_wait_breakdown'])
 if 'merge_outside_conditioning_seconds' in item: print(' merge outside conditioning',round(item['merge_outside_conditioning_seconds'],6))
