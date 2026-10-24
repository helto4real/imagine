"""Structural parity guard for the component extraction; reads source only."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOKEN = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\w+|[^\s]', re.MULTILINE)


def functions(source):
    tokens = [m.group() for m in TOKEN.finditer(source)
              if not m.group().startswith(('//', '/*'))]
    out = defaultdict(Counter)
    for i in range(len(tokens) - 2):
        if tokens[i] != 'fn' or not tokens[i + 1].isidentifier():
            continue
        start = i + 2
        while start < len(tokens) and tokens[start] not in ('{', ';'):
            start += 1
        if start == len(tokens) or tokens[start] == ';':
            continue
        depth, end = 1, start + 1
        while depth and end < len(tokens):
            depth += (tokens[end] == '{') - (tokens[end] == '}')
            end += 1
        out[tokens[i + 1]][tuple(tokens[start:end])] += 1
    return out


class ComponentExtraction(unittest.TestCase):
    def test_original_function_bodies_are_preserved_exactly(self):
        lock = json.loads((ROOT / 'native/product/build-lock.json').read_text())
        groups = {
            'offload.rs': ['components/offload.rs', 'models/qwen_image21/offload.rs'],
            'ownership.rs': ['components/ownership.rs'],
            'checkpoint.rs': ['components/checkpoint.rs', 'models/qwen_image21/checkpoint.rs'],
            'components.rs': ['components/inventory.rs', 'models/qwen_image21/components.rs'],
        }
        count = 0
        for old, destinations in groups.items():
            path = 'inference/src/models/qwen_image21/' + old
            revision = lock['core_revision'] if path in lock['core_files'] else lock['source_base']
            before = subprocess.check_output(['git', 'show', f'{revision}:{path}'], cwd=ROOT, text=True)
            after = '\n'.join((ROOT / 'inference/src' / p).read_text() for p in destinations)
            actual = functions(after)
            for name, bodies in functions(before).items():
                with self.subTest(module=old, function=name):
                    self.assertFalse(bodies - actual[name], 'Original operation body changed')
                count += sum(bodies.values())
        self.assertGreater(count, 60)

    def test_h3_terminal_release_operation_body_is_unchanged(self):
        path = 'inference/src/h3_product/lifecycle.rs'
        before = subprocess.check_output(['git', 'show', f'HEAD:{path}'], cwd=ROOT, text=True)
        self.assertEqual(functions(before), functions((ROOT / path).read_text()))


if __name__ == '__main__':
    unittest.main()
