"""Validate the synthetic and metadata evidence; fail rather than invent an image pass."""
from pathlib import Path
import csv,hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/vae-shortcut-strips-build'
status=json.loads((OUT/'check-status.json').read_text())
assert all(v==0 for v in status.values()),status
comparisons=[]
for name in ('synthetic-small','synthetic-production'):
 log=(OUT/(name+'.log')).read_text()
 hits=re.findall(r'COMPARE (\d+)x(\d+) target=(\d) first=(\d) differing_floats=(\d+) max_abs_error=(\S+) nonfinite=(\d+) bytes=(\d+)',log)
 assert len(hits)==(5 if name=='synthetic-small' else 1)
 for w,h,target,first,diff,error,nonfinite,nbytes in hits:
  assert int(diff)==0 and float(error)==0 and int(nonfinite)==0
  comparisons.append({'width':int(w),'height':int(h),'target':bool(int(target)),'first_chunk':bool(int(first)),'bytes':int(nbytes),'bitidentical':True})
 assert log.count('live_inputs_unchanged=1')==2*len(hits)
assert comparisons[-1]['bytes']==825753600
before=json.loads((OUT/'before-hashes.json').read_text());after=json.loads((OUT/'after-hashes.json').read_text())
changed=[f for f in before if before[f]!=after[f]]
assert len(changed)==1 and changed[0].endswith('/src/model/vae/wan_vae.hpp')
assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in after.items())
assert hashlib.sha256((OUT/'sd-server').read_bytes()).hexdigest()==json.loads((OUT/'binary-hashes.json').read_text())['sd-server']
assert (OUT/'probe-control/graph-1.tsv').read_bytes()==(ROOT/'runs/vae-strip-build/probe-strips-contiguous/graph-1.tsv').read_bytes()
assert (OUT/'test-source.cpp').read_bytes()==(ROOT/'profiling/test-vae-shortcut-strips.cpp').read_bytes()
for mode in ('control','shortcut'):
 assert (OUT/('probe-'+mode)/'graph-1.tsv').read_bytes()==(OUT/('allocation-'+mode)/'graph-1.tsv').read_bytes()
analysis=json.loads((OUT/'allocation-analysis.json').read_text())
assert analysis['shortcut']['graph_mib']>analysis['control']['graph_mib']
row_probes={}
for rows,directory in ((16,'rows-16'),(32,'probe-shortcut'),(64,'rows-64')):
 log=(OUT/directory/'run.log').read_text()
 memory=float(re.search(r'direct=([\d.]+) MiB scheduler=0',log)[1])
 assert memory==8669.360596
 graph=list(csv.DictReader((OUT/directory/'graph-1.tsv').open(),delimiter='\t'))
 assert sum(r['op']=='ACC' for r in graph)==(224+rows-1)//rows
 assert not any(r['op']=='CONT' and r['shape']=='2,400,224,2304' for r in graph)
 row_probes[rows]={'graph_mib':memory,'acc_nodes':(224+rows-1)//rows}
result={'comparisons':comparisons,'live_inputs_unchanged':True,'small_and_production_suites_passed':True,'row_size_metadata_probes':row_probes,'control_graph_matches_previous':True,'source_binary_hashes_verified':True,'graph_mib_control':analysis['control']['graph_mib'],'graph_mib_shortcut':analysis['shortcut']['graph_mib'],'graph_delta_mib':analysis['shortcut']['graph_mib']-analysis['control']['graph_mib'],'image_gate_passed':False,'image_tests_run':False,'retention':'off','block_strip':'off','new_shortcut_default':'off'}
(OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
