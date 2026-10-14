"""Validate the combined shortcut/stage-3 experiment without claiming model-image acceptance."""
from pathlib import Path
import csv,hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/vae-shortcut-stage3-build'
status=json.loads((OUT/'check-status.json').read_text())
assert status=={'probe-control':0,'probe-shortcut':0,'probe-stage3':0,'probe-combined':0,'synthetic-small':1,'synthetic-production':0}
prod=(OUT/'synthetic-production.log').read_text();small=(OUT/'synthetic-small.log').read_text()
assert prod.count('bytes=412876800')==2 and 'PASS complete convolution byte identity' in prod
assert 'PASS all comparisons and retained-input integrity' in prod
assert 'differing_floats=2288498 max_abs_error=6.375' in small
before=json.loads((OUT/'before-hashes.json').read_text());after=json.loads((OUT/'after-hashes.json').read_text())
changed=[f for f in before if before[f]!=after[f]]
assert len(changed)==1 and changed[0].endswith('/src/model/vae/wan_vae.hpp')
assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in after.items())
assert hashlib.sha256((OUT/'sd-server').read_bytes()).hexdigest()==json.loads((OUT/'binary-hashes.json').read_text())['sd-server']
assert (OUT/'test-source.cpp').read_bytes()==(ROOT/'profiling/test-vae-stage3-strips.cpp').read_bytes()
assert (OUT/'probe-control/graph-1.tsv').read_bytes()==(ROOT/'runs/vae-strip-build/probe-strips-contiguous/graph-1.tsv').read_bytes()
assert (OUT/'probe-shortcut/graph-1.tsv').read_bytes()==(ROOT/'runs/vae-shortcut-strips-build/probe-shortcut/graph-1.tsv').read_bytes()
for mode in ('control','combined'):
 assert (OUT/('allocation-'+mode)/'graph-1.tsv').read_bytes()==(OUT/('probe-'+mode)/'graph-1.tsv').read_bytes()
measurements={}
for mode,expected,acc,sets in [('control',8663.867432,0,28),('shortcut',8669.360596,7,28),('stage3',9057.617432,0,35),('combined',8669.356201,7,35)]:
 log=(OUT/('probe-'+mode)/'run.log').read_text()
 memory=float(re.search(r'direct=([\d.]+) MiB scheduler=0',log)[1]);assert memory==expected
 assert 'growth=394.000000 MiB' in log
 graph=list(csv.DictReader((OUT/('probe-'+mode)/'graph-1.tsv').open(),delimiter='\t'))
 assert sum(r['op']=='ACC' for r in graph)==acc
 assert sum(r['op']=='SET' for r in graph)==sets
 measurements[mode]=memory
result={'production_convolution_bitidentical':True,'production_bytes_compared':412876800,'input_unchanged':True,'small_shape_bitidentity':False,'small_failure_shape':[65,129,576,288],'small_differing_floats':2288498,'small_max_abs_error':6.375,'graph_mib':measurements,'image_gate_passed':False,'image_tests_run':False,'combined_full_graph_bitidentity_tested':False,'control_and_shortcut_metadata_match_previous':True,'source_binary_hashes_verified':True,'retention':'off','block_strip':'off','new_stage3_flag_default':'off'}
(OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
