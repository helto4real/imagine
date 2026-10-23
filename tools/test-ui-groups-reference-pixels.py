#!/usr/bin/env python3
"""CPU Director geometry/Pillow Lanczos/edge padding versus actual Rust entry."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument('case')
    args = parser.parse_args()
    assert args.case.isalnum()
    root = Path('/home/thhel/ChatGPT/imagine')
    own = Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')
    director = Path('/home/thhel/git/comfyui-helto-director/shared')
    case = own / args.case
    case.mkdir()
    assert not os.statvfs(case).f_flag & os.ST_RDONLY
    runtime = json.loads((Path('/home/thhel/tmp/.codex-h3-res20-compare-20261020-6f0b84c2') /
        'full-fp16-20261020-5e4a923b/candidate-runtime.json').read_text())
    pillow = runtime['h3']['environment']['IMAGINE_PILLOW_LIBRARY']
    probe = root / 'comfyui/ui-groups-20261021/target-worker/debug/examples/cp39_reference_pixels'
    scope = {'torch': torch, 'np': np, 'Image': Image, 'F': F,
             'CROP_MODE_STRETCH_TO_FIT': 'stretch', 'CROP_MODE_CROP': 'crop',
             'H3_REFERENCE_QUALITY_PIXELS': {'Standard': 1000000, 'High': 2000000, 'Extra High': 4000000}}
    sources = []

    def load(path, names):
        code = path.read_text()
        nodes = [n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name in names]
        assert {n.name for n in nodes} == set(names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
        sources.append({'path': str(path), 'sha256': hashlib.sha256(code.encode()).hexdigest(), 'functions': names})

    load(director / 'media.py', ['_snap', '_resize_image'])
    load(director / 'h3/reference_quality.py', ['reference_image_geometry', 'prepare_reference_image'])

    def resize(image, width, height):
        # Deliberately use allowed Director media's real Pillow Lanczos provider.
        # Do not import/read comfy.utils or pretend this executes its wrapper.
        return scope['_resize_image'](image, width, height, 'stretch', 1)

    scope['_resize'] = resize
    cases = [(33, 17, 'Standard', 'reference'), (1, 1, 'Standard', 'reference'),
             (640, 480, 'High', 'reference'), (1301, 773, 'Standard', 'reference'),
             (1301, 1601, 'High', 'reference'), (2001, 2033, 'Extra High', 'reference'),
             (257, 513, 'Extra High', 'boundary')]
    report = {'scope': 'Actual Rust reference-image entry versus source-executed Director geometry/preparation with its allowed media Pillow Lanczos provider; not original comfy.utils wrapper or full Python model pipeline',
              'runtime_initialized': False, 'gpu': False, 'comfy_imported': False, 'raw_tensors_persisted': False,
              'cases': [], 'sources': sources, 'passed': False}
    for width, height, quality, role in cases:
        y, x, channel = np.ogrid[:height, :width, :3]
        rgb = ((x * 13 + y * 17 + channel * 53 + ((x ^ y) % 19) * 11) % 256).astype(np.uint8)
        image = torch.from_numpy(rgb.astype(np.float32) / np.float32(255.)).unsqueeze(0)
        if role == 'boundary':
            # Source-execute boundary geometry too; it delegates to _resize above.
            if 'prepare_boundary_picture' not in scope:
                import math
                scope['math'] = math
                load(director / 'h3/reference_quality.py', ['prepare_boundary_picture'])
            expected = scope['prepare_boundary_picture'](image, 416, 224)
        else:
            expected, _ = scope['prepare_reference_image'](image, quality)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(own / 'tmp'))
        process = subprocess.run([str(probe), pillow, str(width), str(height), quality, role],
            input=expected.contiguous().numpy().astype('<f4').tobytes(), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env)
        if process.stderr:
            (case / f'compiler-or-probe-{width}-{height}.txt').write_bytes(process.stderr)
        result = json.loads(process.stdout)
        report['cases'].append({'input': [width, height], 'quality': quality, 'role': role, **result})
        (case / 'result.json').write_text(json.dumps(report, indent=2))
        assert process.returncode == 0 and result['byte_identical'], report['cases'][-1]
        print(json.dumps(report['cases'][-1]), flush=True)
    report['passed'] = True
    report['sources'] = sources
    (case / 'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'passed': True, 'cases': len(report['cases']), 'report': str(case / 'result.json')}))


if __name__ == '__main__':
    main()
