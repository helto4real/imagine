#!/usr/bin/env python3
"""Archive integrity and non-destructive restore checks with tiny owned files."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from regression_archive import digest, partition, replay


class RegressionArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.bundle = self.base / 'bundle'
        self.bundle.mkdir()
        self.root = self.base / 'restored'

    def bundle_with(self, members, manifest_files=None):
        archive = self.bundle / 'oracles.tar.gz'
        files = {}
        with tarfile.open(archive, 'w:gz') as output:
            for name, payload in members:
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                output.addfile(info, io.BytesIO(payload))
                files[name] = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        (self.bundle / 'manifest.json').write_text(json.dumps(dict(
            archive_sha256=digest(archive), files=files if manifest_files is None else manifest_files)))
        return archive

    def quiet_replay(self, root=None):
        with contextlib.redirect_stdout(io.StringIO()):
            replay(self.bundle, root)

    def test_restore_preserves_existing_matching_file_and_is_repeatable(self):
        self.bundle_with([('runtime/own.json', b'{"own":1}\n')])
        self.quiet_replay(self.root)
        self.quiet_replay(self.root)
        self.assertEqual((self.root / 'runtime/own.json').read_bytes(), b'{"own":1}\n')

    def test_split_gzip_stream_restores_across_part_boundaries(self):
        self.bundle_with([('runtime/own.json', b'own' * 4096)])
        path = self.bundle / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['archive_parts'] = partition(self.bundle, limit=19)
        path.write_text(json.dumps(manifest))
        self.assertGreater(len(manifest['archive_parts']), 1)
        self.quiet_replay(self.root)
        self.assertEqual((self.root / 'runtime/own.json').read_bytes(), b'own' * 4096)
        first = self.bundle / manifest['archive_parts'][0]['name']
        first.write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, 'Archive part integrity'):
            self.quiet_replay(self.root)

    def test_corrupt_archive_is_rejected_before_restore(self):
        archive = self.bundle_with([('runtime/own.json', b'own')])
        with archive.open('ab') as handle:
            handle.write(b'corruption')
        with self.assertRaisesRegex(ValueError, 'Archive SHA-256'):
            self.quiet_replay(self.root)
        self.assertFalse(self.root.exists())

    def test_different_existing_file_is_not_overwritten(self):
        self.bundle_with([('runtime/own.json', b'own')])
        self.root.mkdir()
        target = self.root / 'runtime/own.json'
        target.parent.mkdir()
        target.write_bytes(b'preserve')
        with self.assertRaises(FileExistsError):
            self.quiet_replay(self.root)
        self.assertEqual(target.read_bytes(), b'preserve')

    def test_path_traversal_and_symlink_escape_are_rejected(self):
        self.bundle_with([('../escape', b'own')])
        with self.assertRaisesRegex(ValueError, 'Unsafe archive'):
            self.quiet_replay(self.root)
        self.bundle_with([('link/escape', b'own')])
        self.root.mkdir(exist_ok=True)
        (self.root / 'link').symlink_to(self.base)
        with self.assertRaisesRegex(ValueError, 'escapes restore root'):
            self.quiet_replay(self.root)
        self.assertFalse((self.base / 'escape').exists())

    def test_member_hash_mismatch_never_installs_the_bad_file(self):
        self.bundle_with([('runtime/own.json', b'own')], {'runtime/own.json': dict(bytes=3, sha256='0' * 64)})
        with self.assertRaisesRegex(ValueError, 'Member SHA-256'):
            self.quiet_replay(self.root)
        self.assertFalse((self.root / 'runtime/own.json').exists())

    def test_missing_member_is_rejected(self):
        self.bundle_with([], {'runtime/missing': dict(bytes=0, sha256=hashlib.sha256(b'').hexdigest())})
        with self.assertRaisesRegex(ValueError, 'Missing archive'):
            self.quiet_replay()


if __name__ == '__main__':
    unittest.main()
