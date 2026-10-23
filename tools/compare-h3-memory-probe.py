#!/usr/bin/env python3
"""Compare only this experiment's successful captured synthetic phase outputs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OWN = ROOT / 'comfyui/h3-memory-20261023'
sys.path.insert(0, str(OWN / 'source/inference/tools'))
from compare_h3_trained_phase import artifacts, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Never overwrite retained evidence'
    assert args.output.resolve().is_relative_to(OWN)
    selections = []
    for path in (args.left, args.right):
        assert path.resolve().is_relative_to(OWN / 'runs')
        selection = json.loads(path.read_text())
        assert selection['result'] == 'passed' and selection['capture']
        assert not selection['oom'] and not selection['cancelled_after_step']
        report = selection['reservation']['report']
        assert report['namespace_exited'] is True
        assert report['worker']['finish_sent'] is True
        selections.append(selection)
    left, right = selections
    for key in ('phase', 'reference', 'geometry', 'profile_sha256', 'model_lock_sha256'):
        assert left[key] == right[key], key
    assert left['selected_component']['sha256'] == right['selected_component']['sha256']
    a, b = [artifacts(s, OWN / 'runs') for s in selections]
    cases = [dict(name=name, equal=a.get(name) == b.get(name),
                  left=a.get(name), right=b.get(name)) for name in sorted(a.keys() | b.keys())]
    passed = a.keys() == b.keys() and all(case['equal'] for case in cases)
    result = dict(schema=1, result='passed' if passed else 'failed',
                  criterion='Exact payload SHA-256, shape, dtype, byte count and token entries',
                  phase=left['phase'], geometry=left['geometry'], reference=left['reference'],
                  cases=cases, selections={str(p): digest(p) for p in (args.left, args.right)})
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['result'].upper(), left['phase'], len(cases), 'exact artifact comparisons')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
