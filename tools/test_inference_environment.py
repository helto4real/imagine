"""Synthetic installer/integrity checks; no downloads, Torch imports or CUDA."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_inference_environment as environment


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='imagine-environment-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lock = {'schema': 1, 'environment': '.venv-inference', 'wheel_cache': 'wheels',
                     'interpreter': {'python': '3.12.13', 'system': 'Linux', 'machine': 'x86_64'},
                     'packages': [{'name': 'pip', 'version': '26.2.1', 'filename': 'fixture.whl',
                                   'url': 'https://example.invalid/fixture.whl', 'sha256': '0' * 64}]}

    def test_wrong_interpreter_never_creates_environment_or_downloads(self):
        with patch.object(environment, 'identity', return_value={'python': '3.14.0'}), \
             patch.object(environment, 'wheels') as download:
            with self.assertRaisesRegex(ValueError, 'Interpreter mismatch'):
                environment.prepare(self.root, self.lock, 'python')
        download.assert_not_called()
        self.assertFalse((self.root / '.venv-inference').exists())

    def test_missing_or_corrupt_cached_wheel_never_installs(self):
        with self.assertRaisesRegex(ValueError, 'Missing cached wheel'):
            environment.wheels(self.root, self.lock, True)
        (self.root / 'wheels/fixture.whl').write_bytes(b'corrupt')
        with patch.object(environment.urllib.request, 'urlopen') as download:
            with self.assertRaisesRegex(ValueError, 'Wheel hash mismatch'):
                environment.wheels(self.root, self.lock, False)
        download.assert_not_called()
        self.assertFalse((self.root / '.venv-inference').exists())

    def test_existing_environment_is_verified_without_overwrite(self):
        target = self.root / '.venv-inference'
        target.mkdir()
        marker = target / 'keep'
        marker.write_text('existing environment')
        with patch.object(environment, 'verify', side_effect=ValueError('mismatch')), \
             patch.object(environment, 'wheels') as download:
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                environment.prepare(self.root, self.lock, 'python')
        download.assert_not_called()
        self.assertEqual(marker.read_text(), 'existing environment')

    def native_fixture(self):
        site = self.root / '.venv-inference/lib/python3.12/site-packages'
        library = site / 'torch/lib/library.so'
        library.parent.mkdir(parents=True)
        library.write_bytes(b'frozen library')
        header = site / 'torch/include/torch.h'
        header.parent.mkdir()
        header.write_bytes(b'frozen header')
        lock = {'schema': 1, 'build': {'libtorch': str(site / 'torch'), 'header_trees': {
            str(header.parent): environment.build.key({'torch.h': environment.build.digest(header)})}},
            'image': {'files': {str(library): environment.build.digest(library)}}, 'h3': {'files': {}}}
        path = self.root / environment.build.LOCK
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(lock))
        return library, header

    def test_library_and_header_changes_fail_original_product_hashes(self):
        library, header = self.native_fixture()
        with patch.object(environment, 'check_identity'), \
             patch.object(environment.subprocess, 'check_output', return_value='{"pip":"26.2.1"}'), \
             patch.object(environment.subprocess, 'run'):
            environment.verify(self.root, self.lock)
            library.write_bytes(b'changed library')
            with self.assertRaisesRegex(ValueError, 'Pinned artifact changed'):
                environment.verify(self.root, self.lock)
            library.write_bytes(b'frozen library')
            header.write_bytes(b'changed header')
            with self.assertRaisesRegex(ValueError, 'Pinned tree changed'):
                environment.verify(self.root, self.lock)

    def test_library_symlink_to_another_environment_is_rejected(self):
        library, _ = self.native_fixture()
        external = self.root / 'another-environment.so'
        library.rename(external)
        library.symlink_to(external)
        with patch.object(environment, 'check_identity'), \
             patch.object(environment.subprocess, 'check_output', return_value='{"pip":"26.2.1"}'), \
             patch.object(environment.subprocess, 'run'):
            with self.assertRaisesRegex(ValueError, 'Library escapes'):
                environment.verify(self.root, self.lock)


if __name__ == '__main__':
    unittest.main()
