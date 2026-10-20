#!/usr/bin/env python3
"""Compare retained own trained H3 phase artifacts with exact byte criteria."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def artifacts(selection, own_root):
    root = Path(selection['output_root']).resolve()
    assert root.is_relative_to(own_root), 'Only explicit own probe artifacts'
    root /= selection['phase']
    result = {}
    for metadata_path in sorted(root.glob('*.json')):
        if metadata_path.name in ('phase.json', 'gpu-ready.json'):
            continue
        metadata = json.loads(metadata_path.read_text())
        if metadata_path.name == 'tokens.json':
            result['tokens'] = metadata
            continue
        assert set(metadata) == {'shape', 'dtype', 'sha256', 'bytes'}, metadata_path
        payload = metadata_path.with_suffix('.bin')
        assert payload.stat().st_size == metadata['bytes'], payload
        assert digest(payload) == metadata['sha256'], payload
        result[metadata_path.stem] = metadata
    assert result, 'A phase without artifacts is not an inference comparison'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native-selection', 'original-selection', 'own-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    own_root = args.own_root.resolve()
    assert own_root.is_relative_to(project / 'comfyui/h3-stage4-20261020')
    assert not args.output.exists(), 'Retain previous comparisons'
    native = json.loads(args.native_selection.read_text())
    original = json.loads(args.original_selection.read_text())
    assert native['engine'] == 'native' and original['engine'] == 'original'
    for selection in (native, original):
        assert selection['result'] == 'passed' and not selection['load_only'] and not selection.get('exercise')
        assert selection['reservation']['report']['namespace_exited'] is True
        assert selection['reservation']['report']['worker']['finish_sent'] is True
    for key in ('phase', 'reference', 'profile_sha256', 'model_lock_sha256'):
        assert native[key] == original[key], key
    assert native['selected_component']['sha256'] == original['selected_component']['sha256']
    left = artifacts(native, own_root)
    right = artifacts(original, own_root)
    assert left.keys() == right.keys(), 'Both engines must expose every same phase artifact'
    cases = [dict(name=name, equal=left[name] == right[name],
                  native=left[name], original=right[name]) for name in left]
    passed = all(case['equal'] for case in cases)
    result = dict(schema=1, result='passed' if passed else 'failed',
                  criterion='Exact payload SHA-256, shape, dtype, byte count and token entries',
                  phase=native['phase'], reference=native['reference'],
                  profile_sha256=native['profile_sha256'],
                  model_sha256=native['selected_component']['sha256'], cases=cases,
                  selections={str(path): digest(path) for path in
                              (args.native_selection, args.original_selection)})
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['result'].upper(), native['phase'], len(cases), 'exact artifact comparisons')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
