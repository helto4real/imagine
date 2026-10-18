#!/usr/bin/env python3
"""Compare only explicitly supplied isolated Qwen smoke-test artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
p = argparse.ArgumentParser(description=__doc__)
for n in ('site', 'rust', 'python', 'output'):
    p.add_argument('--' + n, type=Path, required=True)
p.add_argument('--allow-mismatch', action='store_true', help='Diagnostic reporting; default is a strict regression gate')
a = p.parse_args()
sys.path.insert(0, str(a.site))
import numpy as np
from PIL import Image
rust = json.loads((a.rust / 'result.json').read_text())
reference = json.loads((a.python / 'result.json').read_text())
if 'source_profile' in rust:
    assert rust['source_profile'] == reference['profile'], 'Profiles differ'
metrics = {}
boundaries = [('context', rust['context_shape']), ('latent', rust['latent_shape']), ('image-rgba', rust['pixels_shape'])]
if reference['profile'].get('custom_sigmas'):
    boundaries.append(('sigmas', [reference['profile']['steps']+1]))
for name, shape in boundaries:
    assert shape == reference['shapes'][name], name
    paths = [root / (name + '-f32.bin') for root in (a.rust, a.python)]
    x, y = [np.fromfile(f, dtype='<f4').reshape(shape) for f in paths]
    delta = np.abs(x.astype(np.float64) - y.astype(np.float64))
    finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
    metrics[name] = dict(shape=shape, exact_bytes=paths[0].read_bytes() == paths[1].read_bytes(),
        hashes=[hashlib.sha256(f.read_bytes()).hexdigest() for f in paths], all_finite=finite,
        unequal_values=int(np.count_nonzero(x != y)), values=int(x.size),
        max_abs=float(delta.max()), mean_abs=float(delta.mean()), rmse=float(np.sqrt(np.mean(delta ** 2))))
images = [np.asarray(Image.open(root / 'image.png').convert('RGBA')) for root in (a.rust, a.python)]
delta = np.abs(images[0].astype(np.int16) - images[1].astype(np.int16))
metrics['png_rgba8'] = dict(exact=bool(np.array_equal(*images)), unequal_channels=int(np.count_nonzero(delta)),
    unequal_pixels=int(np.count_nonzero(np.any(delta != 0, axis=-1))), max_abs=int(delta.max()), mean_abs=float(delta.mean()))
report = dict(schema=1, metrics=metrics, timings_seconds=dict(rust=rust['timings_seconds'], python=reference['timings_seconds']),
    scope=f'One first-process {reference["profile"]["width"]}x{reference["profile"]["height"]} {reference["profile"]["steps"]}-step synthetic {reference["profile"]["variant"]} DiT sample; OS model cache already warm. Not a production quality or steady-state benchmark.',
    production_runtime_read=False)
a.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
if not a.allow_mismatch and not (all(metrics[n]['exact_bytes'] and metrics[n]['all_finite'] for n, _ in boundaries) and metrics['png_rgba8']['exact']):
    raise SystemExit('Exact numerical parity failed; see the saved comparison')
