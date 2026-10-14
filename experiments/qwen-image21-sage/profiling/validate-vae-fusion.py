"""Validate the stopped VAE fusion experiment without accepting tiled fallback."""
import csv, hashlib, json, statistics
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/20261014T140436Z-vae-fusion-gtgudjxf'
BUILD = ROOT / 'runs/vae-upscale-im2col-build'
REFERENCE = 'c672ca9049d2826d3c713c3ead1f41edaca4498e201b0249663341f6cbd7db1c'
paths = {label: Path((OUT / (label + '.log')).read_text().splitlines()[0])
         for label in ('control_A', 'fusion_A')}
rows = []
validation = {}
for label, path in paths.items():
    results = json.loads((path / 'native-results.json').read_text())
    assert len(results) == (3 if label == 'control_A' else 1)
    plan = json.loads((path / 'plan.json').read_text())
    assert hashlib.sha256(Path(plan['native_binary']).read_bytes()).hexdigest() == plan['native_sha256']
    before = json.loads((path / 'source-before.json').read_text())
    current = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in before}
    assert before == current, 'source drift'
    samples = []
    with (path / 'gpu.csv').open() as f:
        for row in csv.DictReader(f):
            samples.append(float(row[' memory.used [MiB]'].strip().split()[0]))
    for row in results:
        i = row['index']
        with Image.open(path / f'native-image-{i}.png') as im:
            assert im.size == (1600, 896)
            digest = hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
            assert digest == row['rgb_sha256']
            assert 'parameters' not in im.info and 'prompt' not in im.info
        log = (path / f'native-request-{i}.log').read_text()
        tiled = 'retrying with spatial tiling' in log
        assert (digest == REFERENCE) == (label == 'control_A')
        assert tiled == (label == 'fusion_A')
        assert row['single_segment_forwards'] == 80
        assert row['cache_hits'] == (0 if i == 0 else 2)
        if i > 0:
            assert row['qwen3vl_graph_calls'] == 0
        assert (row['vae_fusion_events'] > 0) == (label == 'fusion_A')
        assert row['retained_idle_events'] == 0
        rows.append(dict(row, series=label, tiled_fallback=tiled, matches_reference=digest == REFERENCE))
    validation[label] = {'run': str(path), 'images': len(results),
        'whole_gpu_peak_mib_sampled': max(samples), 'source_hashes_unchanged': True,
        'binary_hash_valid': True, 'png_metadata_private_prompts_absent': True}
assert json.loads((paths['control_A'] / 'plan.json').read_text())['prompt_hashes'] == json.loads((paths['fusion_A'] / 'plan.json').read_text())['prompt_hashes']
control = [r for r in rows if r['series'] == 'control_A' and not r['first']]
summary = {'status': 'rejected', 'reason': 'first candidate image failed reference identity after full VAE memory admission failed and spatial tiling was retried',
    'retention_run': False, 'planned_counterbalanced_comparison_completed': False,
    'control_warm_samples': len(control),
    'control_warm_mean': {key: statistics.mean(r[key] for r in control) for key in ('saved_seconds','sampling_seconds','vae_seconds','vae_workspace_mib')},
    'validation': validation,
    'limitation': 'No successful full-resolution untiled candidate VAE execution or warm candidate timing. Metadata-only runner uses architecture shapes, not loaded checkpoint metadata.'}
(OUT / 'validated-results.json').write_text(json.dumps(rows, indent=2) + '\n')
(OUT / 'outcome.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
