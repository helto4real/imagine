#!/usr/bin/env python3
"""Original list/dict copying on tiny own object graphs; no imports of Torch."""
import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from reference_paths import reference_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--general-keys', action='store_true')
    args = parser.parse_args()
    assert sys.version_info[:3] == (3, 12, 13)
    path = args.comfy / 'comfy/utils.py'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = json.loads((reference_dir(Path(__file__).resolve().parents[2]) / 'source-manifest.json').read_text())
    assert digest == next(v['sha256'] for v in manifest if v['root'] == 'comfy' and v['path'] == 'comfy/utils.py')
    fn = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'deepcopy_list_dict')
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
    copy = namespace['deepcopy_list_dict']
    definitions = [
        ['text', 'first'], ['text', 'second'], ['text', 'renamed'],
        ['object', 'a'], ['object', 'b'], ['none'], ['int', 1024],
        ['list', []], ['list', [3, 4, 3]], ['dict', [[0, 8], [1, 8]]],
        ['tuple', [8, 9]], ['list', [9, 10, 8, 9]], ['dict', [[0, 11], [1, 10]]],
        ['list', [13]], ['dict', [[0, 14]]], ['list', [16]], ['dict', [[0, 15]]],
        ['list', [8, 13]], ['dict', [[0, 8], [1, 14]]],
        ['list', [20]], ['tuple', [19]],
    ]
    if args.general_keys:
        definitions += [
            ['tuple', [0, 1]],
            ['dict', [[5, 8], [21, 12], [3, 10]]],
            ['list', [22, 22, 5, 21, 3]],
            ['dict', [[21, 23], [5, 22]]],
            ['dict', [[0, 23], [5, 23], [21, 24]]],
        ]
    class Token:
        def __init__(self, name):
            self.name = name

    def restore():
        values = []
        for definition in definitions:
            kind = definition[0]
            values.append([] if kind == 'list' else {} if kind == 'dict' else
                          sys.intern(definition[1]) if kind == 'text' else
                          Token(definition[1]) if kind == 'object' else
                          None if kind in ('none', 'tuple') else definition[1])
        for i, definition in enumerate(definitions):
            if definition[0] == 'tuple':
                values[i] = tuple(values[j] for j in definition[1])
        for i, definition in enumerate(definitions):
            if definition[0] == 'list':
                values[i].extend(values[j] for j in definition[1])
            elif definition[0] == 'dict':
                values[i].update((values[k], values[v]) for k, v in definition[1])
        return values

    def snapshot(values, result, memo):
        objects = list(values)
        def index(value):
            for i, known in enumerate(objects):
                if value is known:
                    return i
            objects.append(value)
            return len(objects) - 1
        roots = dict(output=None if result is missing else index(result),
                     memo=[[next(i for i, v in enumerate(values) if id(v) == identity), index(value)]
                           for identity, value in memo.items()])
        encoded = []
        position = 0
        while position < len(objects):
            value = objects[position]
            if isinstance(value, (list, tuple)):
                record = [type(value).__name__, [index(v) for v in value]]
            elif isinstance(value, dict):
                record = ['dict', [[index(k), index(v)] for k, v in value.items()]]
            elif isinstance(value, Token):
                record = ['object', value.name]
            elif value is None:
                record = ['none']
            else:
                record = [type(value).__name__ if not isinstance(value, str) else 'text', value]
            encoded.append(record)
            position += 1
        return dict(**roots, objects=encoded)

    missing = object()
    cases, recursive = [], []
    # The source owns a fresh memo for None. The other cases expose the caller's
    # actual memo, including key substitution and partially completed children.
    seeds = [None, [], [[8, 4]], [[8, 7]], [[0, 2]], [[0, 7]], [[13, 3]], [[14, 4]], [[15, 3]], [[19, 4]]]
    if args.general_keys:
        seeds += [[[5, 7]], [[21, 7]], [[3, 4]], [[5, 0]], [[21, 3]]]
    for root in range(len(definitions)):
        for seed in seeds + [[[root, 3]]]:
            values = restore()
            memo = {} if seed is None else {id(values[k]): values[v] for k, v in seed}
            result = missing
            try:
                result = copy(values[root], None if seed is None else memo)
                outcome = dict(ok=True)
            except Exception as error:
                assert isinstance(error, (TypeError, RecursionError)), repr(error)
                outcome = dict(error=type(error).__name__, message=str(error))
            case = dict(root=root, seed=seed, **outcome, state=snapshot(values, result, memo))
            (recursive if outcome.get('error') == 'RecursionError' else cases).append(case)
    assert 'torch' not in sys.modules
    args.output.write_text(json.dumps(dict(schema=1, python_version=list(sys.version_info[:3]),
        source_sha256=digest, trained_model_weights=0, fixture_tensors=0,
        definitions=definitions, cases=cases, recursive=recursive), separators=(',', ':')) + '\n')
    print(f'Captured {len(cases)} exact copying cases and {len(recursive)} source recursion failures; no Torch or weights', flush=True)


if __name__ == '__main__':
    main()
