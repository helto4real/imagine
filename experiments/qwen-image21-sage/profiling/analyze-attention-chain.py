"""Attribute isolated-chain GPU work; exclude warmups and event-only runs."""
import collections
import csv
import json
from pathlib import Path
import subprocess
import sys

root=Path(sys.argv[1])
summary=json.loads((root/'summary.json').read_text())
output=[]
def group(name):
    if 'cutlass::Kernel2' in name:return 'int8_gemm_dequant'
    if 'convrot' in name:return 'convrot_quantize'
    if 'rms_norm_rope' in name or 'rope_kernel' in name:return 'rms_rope'
    if 'concat_cont' in name or 'CatArray' in name:return 'prefix_concat'
    if 'qk_int_sv_' in name:return 'attention_core'
    if any(s in name for s in ['sage_key_mean','sage_quantize','sage_value_scale','detect_k_anchor','quant_qk_fused','quant_v_int8']):return 'attention_quantize'
    if 'cpy_' in name or 'convert_unary' in name or 'direct_copy' in name:return 'layout_cast'
    if name.startswith('copy_kind_'):return 'memcpy'
    return 'unclassified'
for item in summary:
    if not item['traced']:continue
    path=root/item['run'];metadata=json.loads((path/'result.json').read_text())
    windows=[b for b in metadata['batches'] if b['part']=='whole']
    text=(path/'activity.tsv').read_text()
    assert text.splitlines()[-1]=='# dropped_records=0'
    rows=list(csv.DictReader((l for l in text.splitlines() if not l.startswith('#')),delimiter='\t'))
    chosen=[];crossing=[]
    for row in rows:
        start,end=int(row['start_ns']),int(row['end_ns'])
        for window in windows:
            if end>window['start_ns'] and start<window['end_ns']:
                if start>=window['start_ns'] and end<=window['end_ns']:chosen.append(row)
                else:crossing.append(row)
    assert not crossing
    names=list(dict.fromkeys(r['name'] for r in chosen))
    demangled=dict(zip(names,subprocess.check_output(['c++filt',*names],text=True).splitlines()))
    iterations=sum(b['iterations'] for b in windows)
    groups=collections.defaultdict(lambda:dict(calls=0,ms=0.,bytes=0))
    kernels=collections.defaultdict(lambda:dict(calls=0,ms=0.,bytes=0))
    for row in chosen:
        name=demangled[row['name']];label=group(name)
        for target in [groups[label],kernels[name]]:
            target['calls']+=1;target['ms']+=(int(row['end_ns'])-int(row['start_ns']))/1e6;target['bytes']+=int(row['bytes'])
    assert 'unclassified' not in groups,groups.get('unclassified')
    native=item['variant']!='kitchen'
    assert groups['int8_gemm_dequant']['calls']==iterations*4
    assert groups['convrot_quantize']['calls']==iterations*(2 if native else 4)
    assert groups['rms_rope']['calls']==iterations*(2 if native else 1)
    assert groups['attention_core']['calls']==iterations
    assert groups['prefix_concat']['calls']==iterations*2
    assert groups['layout_cast']['calls']==iterations*(3 if item['variant']=='native' else 2 if native else 1)
    for table in [groups,kernels]:
        for value in table.values():value['ms_per_chain']=value['ms']/iterations
    record=dict(run=item['run'],prefix=item['prefix'],variant=item['variant'],iterations=iterations,groups=groups,kernels=kernels,dropped=0,boundary_crossing=0)
    output.append(record)
    (path/'attribution.json').write_text(json.dumps(record,indent=2)+'\n')
(root/'attribution.json').write_text(json.dumps(output,indent=2)+'\n')
for record in output:
    print(record['prefix'],record['variant'],{k:round(v['ms_per_chain'],4) for k,v in record['groups'].items()})
