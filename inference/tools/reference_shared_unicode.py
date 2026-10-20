#!/usr/bin/env python3
"""Own Unicode/hash metadata, including source lone surrogates; no Torch."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    assert sys.version_info[:3] == (3, 12, 13)
    points = [[], [0], [1], [255], [256], [0x7ff], [0xd7ff],
              [0xd800], [0xdbff], [0xdc00], [0xdfff], [0xe000],
              [0xffff], [0x10000], [0x10ffff], [0xd800, 0xdc00],
              [0, 0xd800, 0, 0xdfff, 0xffff], [0x10ffff, 0xd800],
              list(map(ord, 'transformer_options')), list(map(ord, 'räksmörgås'))]
    for width in range(1, 34):
        points.append([([0, 127, 255, 256, 0xd800, 0xdfff, 0xffff, 0x10ffff][i % 8]) for i in range(width)])
    if args.child:
        values = [''.join(map(chr, p)) for p in points]
        assert 'torch' not in sys.modules
        print(json.dumps([dict(points=p, hash=hash(s), utf8=(s if not any(0xd800 <= n <= 0xdfff for n in p) else None), truth=bool(s)) for p, s in zip(points, values)]))
        return
    assert args.output is not None
    runs = []
    for seed in [0, 1, 42, 123456, 2147483647, 4294967295]:
        encoded = subprocess.check_output([sys.executable, '-S', '-B', __file__, '--child'], env={'PYTHONHASHSEED': str(seed)})
        runs.append(dict(seed=seed, cases=json.loads(encoded)))
    args.output.write_text(json.dumps(dict(schema=1, python_version=list(sys.version_info[:3]), hash_algorithm=sys.hash_info.algorithm, runs=runs, trained_model_weights=0, fixture_tensors=0), separators=(',', ':')) + '\n')
    print(f'Captured {sum(len(r["cases"]) for r in runs)} original Unicode hashes; no Torch or weights')


if __name__ == '__main__':
    main()
