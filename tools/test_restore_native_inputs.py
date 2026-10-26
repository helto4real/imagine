"""Native input restoration with synthetic wheels, files and profiles; no downloads."""
import json
import os
from pathlib import Path
import tempfile
import unittest
import zipfile

import product_build as build
import restore_native_inputs as restore


class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='imagine-restore-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'comfyui/wheels').mkdir(parents=True)
        # As after a reboot: the project links into a cleared temporary folder.
        (self.root / 'comfyui/parity').symlink_to(self.root / 'cleared')
        self.verifier = build.Verifier(self.root / 'receipt.json')

    def wheel(self, name, members):
        path = self.root / 'comfyui/wheels' / name
        with zipfile.ZipFile(path, 'w') as archive:
            for member, data in members.items():
                archive.writestr(member, data)
        return {'filename': name, 'url': 'https://invalid.example/' + name, 'sha256': build.digest(path)}

    def test_headers_come_from_wheels_and_must_match_the_tree(self):
        target = self.root / 'comfyui/parity/cuda/include'
        expected = {'cuda.h': b'runtime', 'cccl/a.h': b'cccl'}
        tree = build.key({name: build.hashlib.sha256(data).hexdigest() for name, data in sorted(expected.items())})
        native = {'build': {'cuda_include': str(target), 'header_trees': {str(target): tree}}}
        prefix = 'nvidia/cu13/include/'
        packages = [self.wheel('runtime.whl', {prefix + 'cuda.h': b'runtime', 'nvidia/cu13/lib/x.so': b'lib'}),
                    self.wheel('cccl.whl', {prefix + 'cccl/a.h': b'cccl'})]
        lock = {'wheel_cache': 'comfyui/wheels',
                'cuda_headers': {'member_prefix': prefix, 'packages': packages}}
        restore.headers(self.root, lock, native, self.verifier, offline=True)
        self.assertFalse((self.root / 'comfyui/parity').is_symlink())
        self.assertEqual((target / 'cccl/a.h').read_bytes(), b'cccl')
        self.assertFalse((target / 'x.so').exists())
        # An existing tree is verified, not replaced.
        (target / 'cuda.h').write_bytes(b'edited')
        with self.assertRaisesRegex(ValueError, 'Pinned tree changed'):
            restore.headers(self.root, lock, native, self.verifier, offline=True)

    def test_wrong_wheel_content_leaves_nothing_behind(self):
        target = self.root / 'comfyui/parity/include'
        native = {'build': {'cuda_include': str(target), 'header_trees': {str(target): '0' * 64}}}
        packages = [self.wheel('runtime.whl', {'nvidia/cu13/include/cuda.h': b'other'})]
        lock = {'wheel_cache': 'comfyui/wheels',
                'cuda_headers': {'member_prefix': 'nvidia/cu13/include/', 'packages': packages}}
        with self.assertRaisesRegex(ValueError, 'Pinned tree changed'):
            restore.headers(self.root, lock, native, self.verifier, offline=True)
        self.assertEqual(list(target.parent.iterdir()), [])

    def test_compiler_inputs_copy_only_pinned_candidates(self):
        original = self.root / 'site/kitchen.py'
        original.parent.mkdir()
        original.write_text('kernel source')
        tool = self.root / 'comfyui/toolchain/triton-opt'
        tool.parent.mkdir()
        tool.write_text('rebuilt tool')
        profile = {'tools': {'triton_opt': {'path': str(self.root / 'comfyui/parity/bin/triton-opt'),
                                            'sha256': build.digest(tool)}},
                   'source_file': {'path': str(original), 'copy': str(self.root / 'comfyui/parity/src/kitchen.py'),
                                   'sha256': build.digest(original), 'mtime_ns': 1_700_000_000_000_000_000},
                   'external_libraries': []}
        profile_path = self.root / 'comfyui/profile.json'
        profile_path.write_text(json.dumps(profile))
        native = {'image': {'environment': {'IMAGINE_TRITON_COMPILER_PROFILE': str(profile_path)},
                            'files': {str(profile_path): build.digest(profile_path)}}}
        lock = {'compiler_inputs': {'candidates': {'triton_opt': [str(tool)]}}}
        restore.compiler_inputs(self.root, lock, native, self.verifier)
        copy = Path(profile['source_file']['copy'])
        self.assertEqual(copy.read_text(), 'kernel source')
        self.assertEqual(copy.stat().st_mtime_ns, profile['source_file']['mtime_ns'])
        self.assertEqual(Path(profile['tools']['triton_opt']['path']).read_text(), 'rebuilt tool')
        # A candidate without the pinned hash is never copied.
        Path(profile['tools']['triton_opt']['path']).unlink()
        tool.write_text('different build')
        with self.assertRaisesRegex(ValueError, 'No candidate with the pinned hash'):
            restore.compiler_inputs(self.root, lock, native, self.verifier)
        self.assertFalse(Path(profile['tools']['triton_opt']['path']).exists())

    def test_links_outside_comfyui_are_never_replaced(self):
        (self.root / 'elsewhere').symlink_to(self.root / 'cleared')
        with self.assertRaisesRegex(ValueError, 'outside comfyui'):
            restore.open_parents(self.root, self.root / 'elsewhere/file')
        self.assertTrue((self.root / 'elsewhere').is_symlink())

    def test_real_lock_names_every_header_wheel_with_a_hash(self):
        lock = restore.load(build.ROOT)
        names = [p['name'] for p in lock['cuda_headers']['packages']]
        self.assertEqual(len(names), len(set(names)))
        for package in lock['cuda_headers']['packages']:
            self.assertTrue(package['url'].startswith('https://files.pythonhosted.org/'))
            self.assertTrue(package['url'].endswith('/' + package['filename']))
            self.assertEqual(len(package['sha256']), 64)
        native = build.load_lock(build.ROOT)
        self.assertIn(native['build']['cuda_include'], native['build']['header_trees'])


if __name__ == '__main__':
    unittest.main()
