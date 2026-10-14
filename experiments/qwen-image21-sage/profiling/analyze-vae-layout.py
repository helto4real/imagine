"""Validate loaded-graph/probe correspondence and account for VAE admission bytes."""
from pathlib import Path
import csv,hashlib,json,re
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/20261014T143229Z-vae-layout-ggy122ez'
BUILD=ROOT/'runs/vae-layout-diagnostic-build'
REF='c672ca9049d2826d3c713c3ead1f41edaca4498e201b0249663341f6cbd7db1c'
FALLBACK='a80c12191b007ab9f1141f4dd5d20d387cba9282aa52c0364817b7d7218ba9bf'
def allocations(text):
    out={}
    for m in re.finditer(r'VAETRACE ALLOC node=(\d+) name=.* buffer=(\d+) chunk=(\d+) offset=(\d+) size=(\d+)',text):
        value=tuple(map(int,m.groups()[1:]));node=int(m[1])
        assert node not in out or out[node]==value
        out[node]=value
    return out
paths=json.loads((OUT/'run-paths.json').read_text());results={}
for mode,info in paths.items():
    run=Path(info['run']);graph=(OUT/mode/'graph-1.tsv').read_bytes()
    corrected=(OUT/('loaded-shapes-'+mode)/'graph-1.tsv').read_bytes()
    assert graph==corrected
    live=(run/'native-request-0.log').read_text().split('VAETRACE BEGIN graph=2')[0]
    probe=(OUT/('loaded-shapes-'+mode)/'probe.log').read_text()
    a=allocations(live);b=allocations(probe);assert a==b
    graph_bytes=int(re.search(r'VAETRACE PLAN direct=(\d+) scheduler=0 pending_cache=0',live)[1])
    assert max(offset+size for buffer,chunk,offset,size in a.values())==graph_bytes
    before=json.loads((run/'source-before.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items())
    plan=json.loads((run/'plan.json').read_text())
    assert hashlib.sha256(Path(plan['native_binary']).read_bytes()).hexdigest()==plan['native_sha256']
    images=json.loads((run/'native-results.json').read_text());assert len(images)==(2 if mode=='control' else 1)
    for r in images:
        with Image.open(run/f"native-image-{r['index']}.png") as im:
            assert im.size==(1600,896)
            h=hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
            assert h==r['rgb_sha256']==(REF if mode=='control' else FALLBACK)
        assert r['single_segment_forwards']==80
    cap=re.search(r'VAETRACE CAPACITY pending=(\d+) missing_weights=(\d+) pool_resident=(\d+) pool_growth=(\d+) safety=(\d+) required_device=(\d+) required_budget=(\d+)',live)
    pending,missing,resident,growth,safety,device,budget=map(int,cap.groups())
    assert pending==graph_bytes
    assert pending+missing+growth+safety==device
    assert pending+missing+resident+growth==budget
    results[mode]={'graph_sha256':hashlib.sha256(graph).hexdigest(),'metadata_byte_identical':True,
        'allocation_entries_identical':len(a),'graph_bytes':graph_bytes,'graph_mib':graph_bytes/1048576,
        'missing_vae_weight_bytes':missing,'cuda_pool_resident_bytes':resident,'cuda_pool_growth_bytes':growth,
        'safety_bytes':safety,'required_device_bytes':device,'required_device_mib':device/1048576,
        'required_budget_bytes':budget,'required_budget_mib':budget/1048576,
        'image_hash':images[0]['rgb_sha256'],'reference_images_passed':mode=='control',
        'source_binary_hashes_unchanged':True}
results['diagnosis']={'old_probe_fusion_graph_bytes':11561984256,
    'missing_from_old_probe_bytes':results['fusion']['graph_bytes']-11561984256,
    'fusion_graph_regression_bytes':results['fusion']['graph_bytes']-results['control']['graph_bytes'],
    'root_cause':'old probe forced is_2D=true via dummy decoder.conv1.weight ne[2]=64; loaded checkpoint selects CausalConv3d with singleton temporal kernels and explicit spatial PAD before 2D lowering',
    'remaining_unexplained_admission_bytes':0,'runtime_optimization_changes':False}
(OUT/'analysis.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
